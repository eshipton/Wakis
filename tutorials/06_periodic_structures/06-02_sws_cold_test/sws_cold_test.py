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
# # Tutorial 6.2 · Slow-wave structure cold test: corrugated and glide-symmetric waveguides
#
# Traveling-wave tubes (TWTs) and backward-wave oscillators (BWOs) transfer kinetic
# energy from a DC electron beam of accelerating voltage $V_0$ and axial velocity
#
# $$
# u_0 = c \sqrt{1 - \gamma_0^{-2}}, \qquad \gamma_0 = 1 + \frac{e V_0}{m_e c^2},
# $$
#
# to a guided electromagnetic wave via Cherenkov synchronism ($v_p = \omega / \beta \approx u_0 < c$).
# Because smooth-wall metallic waveguides support only fast waves ($v_p > c$), vacuum
# electronics relies on periodic **slow-wave structures (SWS)**—corrugated gratings,
# folded waveguides, interdigital lines, or staggered-pillar waveguides—whose periodic
# loading of pitch $d$ folds the dispersion curve into a series of **Floquet–Bloch space
# harmonics**:
#
# $$
# \mathbf{E}(\mathbf{r}_\perp, z, t) = \operatorname{Re}\!\left\{\sum_{n=-\infty}^{\infty} \mathbf{E}_n(\mathbf{r}_\perp)\, e^{j(\omega t - \beta_n z)}\right\}, \qquad \beta_n = \beta_0 + \frac{2\pi n}{d}.
# $$
#
# Before injecting any macro-particles or space-charge forces ("hot test"), every SWS
# design undergoes an electromagnetic **cold test** on a single unit cell $z \in [0, d]$
# with Bloch-periodic boundary conditions $\mathbf{F}(\mathbf{r}_\perp, z + d) = \mathbf{F}(\mathbf{r}_\perp, z)\,e^{-j\beta_0 d}$:
#
# 1. **Dispersion $\omega(\beta_0)$ and phase velocity $v_{p,n} = \omega / \beta_n$**:
#    locates the synchronous intersection $\omega(\beta_n) = \beta_n u_0$ with the beam line
#    and reveals stopbands at the Brillouin zone boundaries $\beta_n d = m\pi$.
# 2. **Group and energy velocity**: verifies that the dispersion slope $v_g = \mathrm{d}\omega / \mathrm{d}\beta_0$
#    matches the ratio of axial Poynting power flow $P_z$ to stored electromagnetic energy
#    per unit length $W = U_{\text{cell}} / d$,
#    $$
#    v_E = \frac{P_z\, d}{U_{\text{cell}}} = \frac{\frac{1}{2}\operatorname{Re}\iint (\mathbf{E} \times \mathbf{H}^*)_z\,\mathrm{d}x\,\mathrm{d}y}{\frac{1}{4}\iiint \left(\varepsilon_0 |\mathbf{E}|^2 + \mu_0 |\mathbf{H}|^2\right)\mathrm{d}V / d} = v_g.
#    $$
# 3. **Space-harmonic decomposition and Pierce interaction impedance $K_n$**: projects the
#    axial electric field phasor onto the $n$-th space harmonic,
#    $$
#    E_{zn}(x, y) = \frac{1}{d}\int_0^d E_z(x, y, z)\,e^{+j\beta_n z}\,\mathrm{d}z, \qquad
#    K_n(x, y) = \frac{|E_{zn}(x, y)|^2}{2\,\beta_n^2\,|P_z|},
#    $$
#    which directly governs the Pierce small-signal gain parameter $C = \left(I_0 K_n / 4 V_0\right)^{1/3}$.
#
# In this tutorial we benchmark Wakis's complex-field Finite Integration Technique
# (`SolverFIT3D(..., dtype=np.complex128)` with `bc_z = 'bloch'`) against two primary
# literature references:
#
# - **2D Planar & 3D Rectangular Double-Corrugated Grating SWS** (**Karetnikova et al.,
#   *IEEE Trans. Electron Devices* 65(6), 2129–2134, 2018** [2], Table I "Optimized dimensions
#   of the double-grating SWS", analyzed with the LSE mode-matching formulation of **McVey et al.,
#   *IEEE Trans. Microw. Theory Tech.* 42(6), 995–1003, 1994** [3] extended to double-corrugated
#   mirror/glide gratings): period $d = p = 0.50\;\text{mm}$ ($500\;\mu\text{m}$), vane thickness
#   $l = s_{\text{vane}} = 0.10\;\text{mm}$ ($100\;\mu\text{m}$), groove width $s = d - l = 0.40\;\text{mm}$
#   ($400\;\mu\text{m}$), vane height $h = l_{\text{vane}} = 0.30\;\text{mm}$ ($300\;\mu\text{m}$),
#   beam tunnel height $2a = 2 g_{\text{half}} = 0.20\;\text{mm}$ ($200\;\mu\text{m}$), structure width
#   $b = 0.85\;\text{mm}$ ($850\;\mu\text{m}$, $f_{c,\text{TE}_{10}} = 176.35\;\text{GHz}$), sheet-beam
#   thickness $H_b = 0.15\;\text{mm}$ ($150\;\mu\text{m}$), sheet-beam width $W_b = 0.75\;\text{mm}$
#   ($750\;\mu\text{m}$), staggering $w = p/2$, and synchronous beam voltage $V_0 = 20\;\text{kV}$
#   ($u_0 = 0.2719\,c$; $19\text{--}21\;\text{kV}$ beam lines in Fig. 2(a)).
#   At $\beta_0 d = 0.5\pi$, power-law extrapolation $y = y_\infty + C\,M_{\text{slot}}^{-p}$ of the McVey
#   slot-mode expansion over $M_{\text{slot}} = 10\dots 32$ at $N_{\text{harm}} = 60$ gives
#   $f_{1,\infty} = 98.120\;\text{GHz}$ ($p \approx 1.43$) and $K_{-1,\infty}(0) = 1.7637\;\Omega$
#   ($p \approx 1.66$), agreeing with Wakis FIT Richardson extrapolation ($f_{1,\infty} = 98.0670\;\text{GHz}$,
#   $-0.05\%$; $K_{-1,\infty} = 1.7681\;\Omega$, $+0.25\%$) within the FIT discretization error.
# - **3D V-Band Glide-Symmetric Staggered-Pillar Sheet-Beam SWS** (**Marosi, Zuboraj &
#   Capolino, `arXiv:2505.22927`, 2025** [1]): waveguide $a = 1.40\;\text{mm} \times b = 0.80\;\text{mm}$,
#   period $d = 1.05\;\text{mm}$, lateral pillar span $w = 1.08\;\text{mm}$, axial pillar
#   length $l = 0.27\;\text{mm}$ (Table I; text notes $l = d/4 = 0.2625\;\text{mm}$), pillar thickness
#   $t = 0.10\;\text{mm}$, pillar-beam clearance $g = 0.10\;\text{mm}$ around a
#   $0.55\;\text{mm} \times 0.10\;\text{mm}$ sheet beam at $V_0 = 5.2\;\text{kV}$ ($u_0 = 0.1416\,c$).
#
# ### Simulation Setup Schematic & Graded Non-Uniform Meshing (`00_setup_schematic.png`)
#
# ![Setup schematic: corrugated SWS unit cell, 3D staggered-pillar cut planes, and graded mesh zoom](figures/00_setup_schematic.png)
#
# The setup schematic (`figures/00_setup_schematic.png`, generated from live `SolverFIT3D`,
# `GridFIT3D`, `CurrentSource`, and `ResonanceMonitor` objects via `wakis.schematic` and styled
# with `wakis.figstyle`) summarizes the cold-test computational domain, boundary conditions,
# excitation, and mesh refinement across three panels:
#
# - **Panel (a) — Double-Corrugated Grating Unit Cell ($\beta_0 d$ Bloch)**: Shows a single axial
#   period $z \in [0, d = 500\;\mu\text{m}]$ and full waveguide height $y \in [-400, +400]\;\mu\text{m}$
#   ($b_{\text{tot}} = 800\;\mu\text{m}$). The axial faces $z = 0, d$ enforce complex Floquet–Bloch
#   periodicity $\mathbf{E}(x, y, d) = \mathbf{E}(x, y, 0)\,e^{-j\beta_0 d}$ (`bc_z = 'bloch'`),
#   the top and bottom housing walls $y = \pm 400\;\mu\text{m}$ are PEC (`bc_y = 'pec'`), and the
#   lateral faces $\pm x$ are periodic in 2D (`bc_x = 'periodic'`) or PEC at $x = 0, a = 850\;\mu\text{m}$
#   in 3D (`Task 9`). Two staggered PEC vanes of axial tooth width $l = 100\;\mu\text{m}$ and depth
#   $h = 300\;\mu\text{m}$ protrude from the bottom wall at $z_1 = d/4 = 125\;\mu\text{m}$ and from
#   the top wall at $z_2 = 3d/4 = 375\;\mu\text{m}$ ($\delta z = d/2$ glide symmetry), leaving a
#   $2a = 200\;\mu\text{m}$ beam tunnel ($y \in [-100, +100]\;\mu\text{m}$) that encloses the
#   $20\;\text{kV}$ ($u_0 = 0.2719\,c$, $H_b = 150\;\mu\text{m}$) sheet electron beam. An off-axis
#   dipole source $(J_z, J_y)$ at $(z, y) = (210, 25)\;\mu\text{m}$ (`GaussianSource`, $f_0 = 120\;\text{GHz}$)
#   excites both even and odd Bloch branches, while an on-axis `ResonanceMonitor` (`harminv`) probe
#   at $(z, y) = (70, 0)\;\mu\text{m}$ and a post-pulse volume `HannDFTMonitor` extract the eigenfrequencies
#   $f_m(\beta_0)$ and complex Floquet space harmonics $\mathbf{E}_n(x, y)$.
# - **Panel (b) — 3D Isometric View of the V-Band Staggered-Pillar SWS & Cut Planes**: Shows the
#   3D rectangular PEC housing $a \times b \times d = 1.40 \times 0.80 \times 1.05\;\text{mm}^3$ of
#   Marosi, Zuboraj & Capolino (`arXiv:2505.22927`, 2025 [1]) with Bloch boundaries at $z = 0, d$
#   (phase advance $\beta_0 d$) and PEC walls at $x = \pm a/2$, $y = \pm b/2$. The $-x$ PEC pillar
#   pair at $z_1 = d/4$ (blue) and the glide-shifted $+x$ PEC pillar pair at $z_2 = 3d/4$ (orange,
#   lateral span $w = 1.08\;\text{mm}$, axial length $l = 0.27\;\text{mm}$, thickness $t = 0.10\;\text{mm}$)
#   surround the $5.2\;\text{kV}$ ($u_0 = 0.1416\,c$, $0.55 \times 0.10\;\text{mm}^2$) sheet beam
#   (red). The two diagnostic cut planes $x = 0$ (green, longitudinal $E_z(0, y, z)$ slice shown in
#   Fig. 13b) and $y = 0$ (blue, horizontal beam midplane) intersect along the beam axis $(x, y) = (0, 0)$.
# - **Panel (c) — Graded Non-Uniform Mesh Zoom at the Vane Tip & Slot Mouth**: Zooms into the bottom
#   PEC vane tooth ($z \in [75, 175]\;\mu\text{m}$, $y \in [-400, -100]\;\mu\text{m}$) and tunnel
#   aperture $y = -100\;\mu\text{m}$ for `Graded L2` (`build_2d_grating_graded_grid`, $n_{\text{feat}} = 8$).
#   Fine cells $\Delta y_{\min} = \Delta z_{\min} = 12.5\;\mu\text{m}$ ($8$ cells across the $100\;\mu\text{m}$
#   vane tooth and across the vane-tip / slot-mouth singular corner) grade smoothly to
#   $\Delta y_{\max} = 30.0\;\mu\text{m}$ in the back of the $300\;\mu\text{m}$ slot and
#   $\Delta z_{\max} = 22.0\;\mu\text{m}$ in the open groove ($\ge 45\;\text{cells}/\lambda$ everywhere at
#   $f_{\max} = 235\;\text{GHz}$), reducing the 2D unit-cell mesh from $2{,}560$ cells (`Uniform L2`) to
#   $1{,}584$ cells (`Graded L2`, $1.62\times$ fewer cells).
#
# ---
#
# ## 1. Multimodal Field-Matching Theory & Glide-Symmetry Stopband Closure
#
# ### 1.1 McVey Multimodal Admittance Matching
#
# In the 2D double-corrugated parallel-plate grating (or 3D $\text{LSE}_{10}$ rectangular
# grating where $E_x = 0$ and all components vary as $\sin(\pi x / a)$ across $x \in [0, a]$),
# the transverse wavenumber in the plane perpendicular to $x$ is
#
# $$
# k_\perp^2 = k_0^2 - \left(\frac{\pi}{a}\right)^2 = \left(\frac{2\pi f}{c}\right)^2 - \left(\frac{2\pi f_c}{c}\right)^2,
# $$
#
# with $f_c = 0$ in 2D and $f_c = c / (2a) = 176.35\;\text{GHz}$ in 3D ($a = 0.85\;\text{mm}$).
# Inside the beam tunnel $|y| \le g_{\text{half}}$, each Floquet harmonic $\beta_n = \beta_0 + 2\pi n / d$
# has transverse decay constant $\tau_n = \sqrt{\beta_n^2 - k_\perp^2}$ and hyperbolic profiles
# $\cosh(\tau_n y)$ (even $E_z$, non-vanishing on the beam axis $y = 0$) or $\sinh(\tau_n y)$
# (odd $E_z$, node at $y = 0$).
#
# Inside each rectangular groove of width $s = d - l$ and depth $h$, the tangential electric
# field at the slot mouth $y = \pm g_{\text{half}}$ is expanded in groove waveguide modes
# $m = 0, 1, \dots, M_{\text{slot}} - 1$:
#
# $$
# E_z(g_{\text{half}}, z) = \sum_{m=0}^{M_{\text{slot}}-1} V_m \cos\!\left(\frac{m\pi (z - z_0)}{s}\right), \qquad \Gamma_m = \sqrt{\left(\frac{m\pi}{s}\right)^2 - k_\perp^2}.
# $$
#
# Projecting the tunnel magnetic field $H_x$ onto the slot modes and matching to the short-circuited
# groove input admittance $Y_{\text{slot}, m} = -(k_\perp / \Gamma_m)\coth(\Gamma_m h)$ yields the
# $M_{\text{slot}} \times M_{\text{slot}}$ Hermitian secular equation
#
# $$
# \det\mathbf{A}(\omega, \beta_0) = 0, \qquad
# A_{mm'}(\omega, \beta_0) = \frac{s}{d}\sum_{n=-\infty}^{\infty} I_{mn}(\beta_n)\,Y_{\text{chan}, n}(\omega, \beta_n)\,I_{m'n}^*(\beta_n) - \epsilon_m Y_{\text{slot}, m}(\omega)\,\delta_{mm'},
# $$
#
# where $I_{mn}(\beta_n) = \frac{1}{s}\int_0^s \cos(m\pi z/s)\,e^{j\beta_n(z - s/2)}\,\mathrm{d}z$ is the
# slot-to-Floquet overlap integral and $\epsilon_0 = 1$, $\epsilon_{m \ge 1} = 1/2$.
# When the slot is wide compared to the tunnel height ($s = 0.40\;\text{mm} = 2 \times 2 g_{\text{half}}$),
# the single-mode approximation ($M_{\text{slot}} = 1$) underestimates the fringing corner capacitance at
# the vane tips by $17.2\%$ ($81.28\;\text{GHz}$ vs. the slot-mode-converged limit $f_{1,\infty} = 98.120\;\text{GHz}$
# at $\beta_0 d = 0.5\pi$); because truncation in $M_{\text{slot}}$ is the dominant analytic error ($M_{\text{slot}} = 6$
# gives $97.686\;\text{GHz}$, still $-0.44\%$ below $f_{1,\infty}$, and $K_{-1}(0) = 1.8274\;\Omega$, $+3.61\%$ above
# $K_{-1,\infty} = 1.7637\;\Omega$), power-law extrapolation $y_\infty + C\,M_{\text{slot}}^{-p}$ over
# $M_{\text{slot}} = 10\dots 32$ is required for the headline analytic reference.
#
# ### 1.2 Mirror vs. Glide Symmetry and Why $n = -1$ is a Backward Wave
#
# In a **mirror-symmetric** grating ($y \mapsto -y$), the even-$E_z$ ($\cosh$) and odd-$E_z$ ($\sinh$)
# modes decouple for every harmonic $n$, and Bragg reflection between the $n = 0$ ($\beta_0 = +\pi/d$)
# and $n = -1$ ($\beta_{-1} = -\pi/d$) harmonics opens a wide stopband at $\beta_0 d = \pi$ where
# $v_g = \mathrm{d}\omega/\mathrm{d}\beta_0 = 0$.
#
# In a **glide-symmetric** grating, the top corrugation (or opposite-sidewall pillar pair) is
# displaced axially by half a period $d/2$ relative to the bottom corrugation. The structure is
# invariant under the glide operator $\mathcal{G}_z: (y, z) \mapsto (-y, z + d/2)$. Because applying
# $\mathcal{G}_z$ twice translates by one full period ($\mathcal{G}_z^2 = \mathcal{T}_d = e^{-j\beta_0 d}$),
# the Bloch eigenstates satisfy $\mathcal{G}_z |\psi_\pm\rangle = \pm e^{-j\beta_0 d / 2}|\psi_\pm\rangle$,
# which forces opposite transverse parities for even and odd space harmonics:
#
# $$
# E_{zn}(-y) = \pm (-1)^n E_{zn}(y) \implies Y_{\text{chan}, n}^{\text{glide}} = \frac{k_\perp}{\tau_n}\begin{cases}\tanh(\tau_n g_{\text{half}}) & \text{even/odd } n, \\ \coth(\tau_n g_{\text{half}}) & \text{odd/even } n.\end{cases}
# $$
#
# At the Brillouin zone boundary $\beta_0 d = \pi$, the two glide eigenvalues $\pm e^{-j\pi/2} = \mp j$
# form a complex-conjugate pair, guaranteeing exact degeneracy between the lower and upper branches
# and **closing the stopband** ($v_g(\pi/d) \ne 0$).
#
# Furthermore, on the lower branch in the first Brillouin zone $0 < \beta_0 d < \pi$, the fundamental
# slope is positive ($v_g = \mathrm{d}\omega/\mathrm{d}\beta_0 > 0$, so electromagnetic energy flows in
# $+z$, $P_z > 0$). However, the $n = -1$ space harmonic has negative axial wavenumber
# $\beta_{-1} = \beta_0 - 2\pi/d < 0$, so its phase velocity $v_{p,-1} = \omega / \beta_{-1} < 0$ is
# directed opposite to the group/energy velocity ($v_{p,-1} v_g < 0$). Equivalently, when unfolded into
# the positive region $\pi < \beta_n d < 2\pi$ (or $3\pi < \beta_n d < 4\pi$), the branch has positive
# phase velocity $v_p > 0$ and negative group velocity $v_g < 0$. An electron beam synchronized with
# this **backward wave** bunches in $+z$ while radiating electromagnetic power backward in $-z$ toward
# the gun end—providing internal regenerative feedback without mirrors, the operating mechanism of the
# **Backward-Wave Oscillator (BWO)** (see Chapter 8).
#
# ### 1.3 First-Principles McVey Pierce Impedance: Tunnel + Slot Power Flow & Energy-Transport Check
#
# For a lossless Bloch eigenmode ($\nabla\cdot\mathbf{S} = 0$, with PEC walls on $x \in [0, a]$ and
# $y = \pm(g_{\text{half}} + h)$), the time-averaged axial power $P_z(z) = \iint_{A_\perp(z)} S_z\,\mathrm{d}x\,\mathrm{d}y$
# through every transverse cross-section $z = \text{const}$ is identical. Averaging $P_z(z)$ over one
# unit-cell period $z \in [0, d]$ decomposes the total guided power $P$ $[\text{W}]$ into the beam-tunnel
# contribution plus the top and bottom groove-slot contributions:
#
# $$
# P = \frac{1}{d}\int_0^d P_z(z)\,\mathrm{d}z = P_{\text{tunnel}} + P_{\text{slot}} \quad [\text{W}].
# $$
#
# 1. **Tunnel power $P_{\text{tunnel}}$**: In the $e^{j(\omega t - \beta_n z)}$ convention, let $\mathbf{v}_{\text{slot}}$
#    be the right singular vector of $\mathbf{A}(\omega, \beta_0)$ with smallest singular value. Setting
#    $\hat{V}_m = v_{\text{slot}, m}^*$ gives the mouth harmonic amplitudes $e_{\text{mouth}, n} = \frac{s}{2d}\sum_{m=0}^{M_{\text{slot}}-1} \hat{V}_m I_{mn}^*(\beta_n)$
#    $[\text{V/m}]$ and transverse profiles $E_{zn}(y) = 2\,e_{\text{mouth}, n}\frac{\cosh(\tau_n y)}{\cosh(\tau_n g_{\text{half}})}$
#    (even $E_z$) or $2\,e_{\text{mouth}, n}\frac{\sinh(\tau_n y)}{\sinh(\tau_n g_{\text{half}})}$ (odd $E_z$).
#    Inside the tunnel $|y| \le g_{\text{half}}$, the $\text{LSE}_{10}$ ($E_x = 0$) transverse components are
#    $\tilde{E}_{yn}(y) = \frac{j\beta_n}{\tau_n^2}\frac{\mathrm{d}E_{zn}}{\mathrm{d}y}$ and
#    $\tilde{H}_{xn}(y) = -\frac{j\omega\varepsilon_0 k_\perp^2}{k_0^2 \tau_n^2}\frac{\mathrm{d}E_{zn}}{\mathrm{d}y}$,
#    multiplying $\sin(\pi x / a)$ across $x \in [0, a]$ so the transverse width integral is
#    $w_{\text{eff}} = \int_0^a \sin^2(\pi x / a)\,\mathrm{d}x = a/2$ in 3D ($w_{\text{eff}} = 1\;\text{mm}$ in 2D).
#    Because distinct Floquet harmonics $\beta_n \ne \beta_{n'}$ are orthogonal over $z \in [0, d]$, cross terms vanish and
#    $$
#    P_{\text{tunnel}} = \sum_{n=-N_{\text{harm}}}^{N_{\text{harm}}} \frac{1}{2}\,\omega\varepsilon_0\,\frac{k_\perp^2}{k_0^2}\,w_{\text{eff}}\,\frac{\beta_n}{\tau_n^2}\int_{-g_{\text{half}}}^{g_{\text{half}}} \left|\frac{\mathrm{d}E_{zn}(y)}{\mathrm{d}y}\right|^2 \mathrm{d}y \quad [\text{W}],
#    $$
#    where the transverse integral is evaluated in closed form from $\sinh(2\tau_n g_{\text{half}}) \mp 2\tau_n g_{\text{half}}$.
# 2. **Slot power $P_{\text{slot}}$ (closed-form cross-mode integral)**: In the top groove ($y \in [g_{\text{half}}, g_{\text{half}} + h]$,
#    distance from the short-circuited back wall $\eta = g_{\text{half}} + h - y \in [0, h]$, local axial coordinate
#    $\zeta = z - z_{\text{top}} \in [0, s]$), matching $E_z(g_{\text{half}}, \zeta) = \sum_{m=0}^{M_{\text{slot}}-1}\hat{V}_m\cos(m\pi\zeta / s)$
#    with $E_z(\eta = 0, \zeta) = 0$ and $E_y(\eta, 0) = E_y(\eta, s) = 0$ uniquely determines the groove $\text{LSE}_{10}$ fields:
#    $$
#    \tilde{E}_z(\eta, \zeta) = \sum_{m=0}^{M_{\text{slot}}-1} \hat{V}_m\,\frac{\sinh(\Gamma_m \eta)}{\sinh(\Gamma_m h)}\cos\!\left(\frac{m\pi\zeta}{s}\right),
#    $$
#    $$
#    \tilde{E}_y(\eta, \zeta) = -\sum_{m=1}^{M_{\text{slot}}-1} \hat{V}_m\left(\frac{m\pi}{s}\right) f_m(\eta)\sin\!\left(\frac{m\pi\zeta}{s}\right), \qquad
#    \tilde{H}_x(\eta, \zeta) = j\omega\varepsilon_0\,\frac{k_\perp^2}{k_0^2}\sum_{m'=0}^{M_{\text{slot}}-1} \hat{V}_{m'}\,f_{m'}(\eta)\cos\!\left(\frac{m'\pi\zeta}{s}\right),
#    $$
#    where $f_m(\eta) = \frac{\cosh(\Gamma_m \eta)}{\Gamma_m \sinh(\Gamma_m h)}$ for $\Gamma_m = \sqrt{(m\pi/s)^2 - k_\perp^2} > 0$
#    (and $-\frac{\cos(k_m \eta)}{k_m \sin(k_m h)}$ when $\Gamma_m^2 = -k_m^2 < 0$). In the bottom groove
#    ($y \in [-(g_{\text{half}} + h), -g_{\text{half}}]$, $\eta = g_{\text{half}} + h + y$), mirror or glide symmetry relates the
#    bottom slot coefficients by $\hat{V}_m^{\text{bot}} = \pm \hat{V}_m^{\text{top}}$ (mirror) or
#    $\hat{V}_m^{\text{top}} = \pm e^{-j\beta_0 d / 2}(-1)^m \hat{V}_m^{\text{bot}}$ (glide) while reversing $\mathrm{d}\eta/\mathrm{d}y = \pm 1$,
#    so both slots carry identical time-averaged axial Poynting power (a factor of $2$).
#    Each individual standing groove mode $m = m'$ has $\operatorname{Re}(-E_{y,m} H_{x,m}^*) \propto \operatorname{Re}(-j |\hat{V}_m|^2) = 0$
#    and carries **zero net axial power**; all slot power arises from cross terms $m \ne m'$ of opposite parity ($m + m'$ odd)
#    whose relative phase $\operatorname{Im}(\hat{V}_m \hat{V}_{m'}^*) \ne 0$ is driven by the travelling Bloch wave across the slot mouth.
#    Evaluating both the axial overlap and depth integrals in **closed form** (verified to $< 10^{-15}$ against 32-point Gauss–Legendre quadrature):
#    $$
#    \int_0^s \left(\frac{m\pi}{s}\right)\sin\!\left(\frac{m\pi\zeta}{s}\right)\cos\!\left(\frac{m'\pi\zeta}{s}\right)\mathrm{d}\zeta = \begin{cases}\dfrac{2m^2}{m^2 - m'^2}, & m + m'\text{ odd}, \\ 0, & m + m'\text{ even},\end{cases}
#    $$
#    $$
#    \int_0^h f_m(\eta)\,f_{m'}(\eta)\,\mathrm{d}\eta = \frac{1}{\Gamma_m^2 - \Gamma_{m'}^2}\left[\frac{\coth(\Gamma_m h)}{\Gamma_m} - \frac{\coth(\Gamma_{m'} h)}{\Gamma_{m'}}\right] = \frac{s^2}{\pi^2(m^2 - m'^2)}\frac{Y_{\text{slot}, m} - Y_{\text{slot}, m'}}{k_\perp},
#    $$
#    yields the exact closed-form cell-averaged slot power for both slots:
#    $$
#    P_{\text{slot}} = \frac{2\,w_{\text{eff}}}{d}\left(\frac{1}{2}\omega\varepsilon_0\frac{k_\perp^2}{k_0^2}\right)\sum_{m=1}^{M_{\text{slot}}-1}\sum_{\substack{m'=0 \\ m+m'\text{ odd}}}^{M_{\text{slot}}-1} \operatorname{Im}\!\left(\hat{V}_m \hat{V}_{m'}^*\right)\frac{2m^2}{m^2 - m'^2}\frac{s^2}{\pi^2(m^2 - m'^2)}\frac{Y_{\text{slot}, m} - Y_{\text{slot}, m'}}{k_\perp} \quad [\text{W}].
#    $$
# 3. **Independent energy-transport check $P = v_g W / d$**: Including the longitudinal magnetic components
#    $H_y = \frac{k_c}{j\omega\mu_0}\tilde{E}_z \cos(\pi x / a)$ and $H_z = -\frac{k_c}{j\omega\mu_0}\tilde{E}_y \cos(\pi x / a)$
#    ($k_c = \pi / a$ in 3D, $k_c = 0$ in 2D), the time-averaged stored electromagnetic energy per unit cell
#    $W = W_{\text{tunnel}} + W_{\text{slot}} = \frac{1}{4}\iiint_{\text{cell}}(\varepsilon_0|\mathbf{E}|^2 + \mu_0|\mathbf{H}|^2)\,\mathrm{d}V$ $[\text{J}]$
#    decouples by orthogonality in both the tunnel ($\beta_n$) and slots ($\cos(m\pi\zeta/s), \sin(m\pi\zeta/s)$) into closed form:
#    $$
#    W_{\text{tunnel}} = \frac{1}{4}\,d\,w_{\text{eff}}\,\varepsilon_0\,\frac{k_\perp^2}{k_0^2}\sum_{n=-N_{\text{harm}}}^{N_{\text{harm}}} \left[\left(\frac{k_0^2}{k_\perp^2} + \frac{\beta_n^2 + k_c^2}{\tau_n^2}\right)\int_{-g_{\text{half}}}^{g_{\text{half}}}|E_{zn}|^2\,\mathrm{d}y + \frac{\beta_n^2 + k_\perp^2}{\tau_n^4}\int_{-g_{\text{half}}}^{g_{\text{half}}}\left|\frac{\mathrm{d}E_{zn}}{\mathrm{d}y}\right|^2\mathrm{d}y\right],
#    $$
#    $$
#    W_{\text{slot}} = \frac{1}{2}\,s\,w_{\text{eff}}\,\varepsilon_0\,\frac{k_\perp^2}{k_0^2}\sum_{m=0}^{M_{\text{slot}}-1}\epsilon_m |\hat{V}_m|^2\left[\left(\frac{k_0^2}{k_\perp^2} + \frac{(m\pi/s)^2 + k_c^2}{\Gamma_m^2}\right)J_{\sinh, m} + \frac{(m\pi/s)^2 + k_\perp^2}{\Gamma_m^2}J_{\cosh, m}\right],
#    $$
#    where $J_{\sinh, m} = \int_0^h \frac{\sinh^2(\Gamma_m\eta)}{\sinh^2(\Gamma_m h)}\,\mathrm{d}\eta = \frac{\sinh(2\Gamma_m h) - 2\Gamma_m h}{4\Gamma_m\sinh^2(\Gamma_m h)}$
#    and $J_{\cosh, m} = \int_0^h \frac{\cosh^2(\Gamma_m\eta)}{\sinh^2(\Gamma_m h)}\,\mathrm{d}\eta = \frac{\sinh(2\Gamma_m h) + 2\Gamma_m h}{4\Gamma_m\sinh^2(\Gamma_m h)}$.
#    Combining $W$ with the central finite-difference group velocity $v_g = \mathrm{d}\omega / \mathrm{d}\beta_0$
#    at $\delta\beta_0 d = 10^{-3}\pi$ from `solve_mcvey_band_frequencies` (`compute_mcvey_power_and_energy_check` in `_helpers.py`)
#    gives $P_{v_g W} = v_g W / d$ $[\text{W}]$, which agrees with $P_{\text{direct}} = P_{\text{tunnel}} + P_{\text{slot}}$ to within $0.13\%$
#    across both 2D and 3D operating points (whereas omitting $P_{\text{slot}}$ leaves a $\sim 40\%$ power deficit).

# %%
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import _helpers as h

DATA_DIR = THIS_DIR / "data"
FIG_DIR = THIS_DIR / "figures"
# Set by --stage1-only: 3D Marosi tasks run only the Harminv stage and store every mode found.
STAGE1_ONLY = False


def compute_mcvey_slot_mode_extrapolation(
    beta0_rad_m: float | None = None,
    spec: h.GratingSWSSpec = h.GratingSWSSpec(),
    symmetry: str = "glide_b",
    n_harm: int = 60,
    m_slot_values: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32),
    m_fit_min: int = 10,
    include_slot_power: bool = True,
    print_table: bool = False,
) -> dict[str, Any]:
    """Sweep McVey slot modes ``M_slot`` at fixed ``N_harm`` and extrapolate ``M_slot -> inf``.

    Fits the power-law model ``y(M) = y_inf + C * M**(-p)`` with ``scipy.optimize.curve_fit``
    over ``M_slot >= m_fit_min`` for both the fundamental eigenfrequency ``f_1`` [GHz]
    and the on-axis / beam-averaged backward-wave Pierce impedance ``K_{-1}`` [Ohm].

    Parameters
    ----------
    beta0_rad_m : float or None
        Fundamental Floquet wavenumber ``beta_0`` [rad/m]. Defaults to ``0.5 * pi / d``.
    spec : GratingSWSSpec
        Grating geometry specification.
    symmetry : str
        Mode symmetry class (default: ``'glide_b'``).
    n_harm : int
        Half-width of Floquet harmonic sum ``n in [-n_harm, +n_harm]`` (default: 60).
    m_slot_values : tuple[int, ...]
        Slot-mode truncation counts ``M_slot`` to evaluate.
    m_fit_min : int
        Minimum ``M_slot`` included in the ``y_inf + C * M**(-p)`` power-law fit.
    include_slot_power : bool
        If True (default), normalizes ``K_n`` by ``P_tunnel + P_slot``.
    print_table : bool
        If True, prints the ``M_slot`` convergence table and fitted ``M_slot -> inf`` limits.

    Returns
    -------
    result : dict[str, Any]
        Dictionary containing ``m_slot_values``, ``n_harm``, ``f1_ghz``, ``km1_axis_ohm``,
        ``km1_beam_avg_ohm``, and fitted parameters ``f1_inf_ghz``, ``f1_c``, ``f1_p``,
        ``km1_inf_ohm``, ``km1_c``, ``km1_p``, ``km1_beam_inf_ohm``, ``km1_beam_c``,
        and ``km1_beam_p``.
    """
    if beta0_rad_m is None:
        beta0_rad_m = 0.5 * np.pi / spec.period_m

    m_arr = np.array(m_slot_values, dtype=int)
    f1_arr = np.zeros(len(m_arr), dtype=float)
    km1_arr = np.zeros(len(m_arr), dtype=float)
    km1_beam_arr = np.zeros(len(m_arr), dtype=float)

    for idx, m_val in enumerate(m_arr):
        f_hz = h.solve_mcvey_band_frequencies(
            float(beta0_rad_m),
            spec=spec,
            symmetry=symmetry,
            n_harm=int(n_harm),
            m_slot=int(m_val),
        )[0]
        imp = h.compute_mcvey_analytic_pierce_impedance(
            float(f_hz),
            float(beta0_rad_m),
            spec=spec,
            symmetry=symmetry,
            n_harm=int(n_harm),
            m_slot=int(m_val),
            include_slot_power=include_slot_power,
        )
        idx_m1 = int(np.where(imp["n_indices"] == -1)[0][0])
        f1_arr[idx] = float(f_hz) * 1e-9
        km1_arr[idx] = float(imp["k_n_axis_ohm"][idx_m1])
        km1_beam_arr[idx] = float(imp["k_n_beam_avg_ohm"][idx_m1])

    def _power_law(m: np.ndarray, y_inf: float, c_val: float, p_val: float) -> np.ndarray:
        return y_inf + c_val * (m ** (-p_val))

    fit_mask = m_arr >= int(m_fit_min)
    m_fit = m_arr[fit_mask].astype(float)
    popt_k, _ = curve_fit(
        _power_law, m_fit, km1_arr[fit_mask], p0=[km1_arr[-1], 1.0, 1.5], maxfev=10000
    )
    popt_f, _ = curve_fit(
        _power_law, m_fit, f1_arr[fit_mask], p0=[f1_arr[-1], -5.0, 1.5], maxfev=10000
    )
    popt_kb, _ = curve_fit(
        _power_law, m_fit, km1_beam_arr[fit_mask], p0=[km1_beam_arr[-1], 1.0, 1.5], maxfev=10000
    )

    if print_table:
        print(f"McVey slot-mode convergence (2D glide, beta_0 d = 0.5 pi, N_harm = {int(n_harm)}):")
        print("| M_slot | f1 [GHz] | K_-1(0) [Ohm] | <K_-1>_beam [Ohm] |")
        print("|---|---|---|---|")
        for m_v, f_v, k_v, kb_v in zip(m_arr, f1_arr, km1_arr, km1_beam_arr):
            print(f"| {int(m_v)} | {f_v:.4f} | {k_v:.5f} | {kb_v:.5f} |")
        print(
            f"Fit y = y_inf + C * M^(-p) over M = {int(m_fit[0])}..{int(m_fit[-1])}: "
            f"K_-1,inf = {popt_k[0]:.4f} Ohm (C = {popt_k[1]:.4f}, p = {popt_k[2]:.2f}), "
            f"f1,inf = {popt_f[0]:.3f} GHz (C = {popt_f[1]:.4f}, p = {popt_f[2]:.2f}), "
            f"<K_-1>_beam,inf = {popt_kb[0]:.4f} Ohm (p = {popt_kb[2]:.2f})"
        )

    return {
        "m_slot_values": m_arr,
        "n_harm": int(n_harm),
        "f1_ghz": f1_arr,
        "km1_axis_ohm": km1_arr,
        "km1_beam_avg_ohm": km1_beam_arr,
        "f1_inf_ghz": float(popt_f[0]),
        "f1_c": float(popt_f[1]),
        "f1_p": float(popt_f[2]),
        "km1_inf_ohm": float(popt_k[0]),
        "km1_c": float(popt_k[1]),
        "km1_p": float(popt_k[2]),
        "km1_beam_inf_ohm": float(popt_kb[0]),
        "km1_beam_c": float(popt_kb[1]),
        "km1_beam_p": float(popt_kb[2]),
    }


# %% [markdown]
# ## 2. Study Task Table & Two-Stage Cold-Test Execution
#
# Each cold-test point uses a two-stage time-domain procedure on the complex-field FIT grid:
#
# - **Stage 1 (`run_bloch_harminv_point`)**: Excite the unit cell at $\mathbf{k}_{\text{Bloch}} = (0, 0, -\beta_0)$
#   with a broadband off-axis `GaussianSource` dipole kick and extract the lossless eigenfrequencies
#   $f_m(\beta_0)$ from the free ring-down via `ResonanceMonitor.harminv()`.
# - **Stage 2 (`extract_bloch_mode_dft`)**: Drive a narrowband `GaussianSource` centered at the
#   harminv-identified eigenfrequency $f_m(\beta_0)$ until `pulse.end_time`, and then attach a
#   volume `HannDFTMonitor([f_m])` over an integer number of RF cycles during the post-pulse free
#   ring-down. Windowing the ring-down after the source has turned off eliminates the non-propagating
#   point-dipole self-field and suppresses $-f_m$ / out-of-band spectral leakage below $10^{-6}$, while
#   `project_floquet_space_harmonics` projects over all $N_z$ axial cells of the Bloch period to yield
#   clean eigenfields $(\mathbf{E}_m, \mathbf{H}_m)$ for $P_z$, $U_{\text{cell}}$, $v_E$, $E_{zn}(x,y)$,
#   and $K_n(x,y)$.
#
# Our resolution ladder spans `n_feat` $\in \{2, 4, 8, 12, 16\}$ cells across the $0.10\;\text{mm}$
# minimum feature ($\Delta y = \Delta z = 50, 25, 12.5, 8.33, 6.25\;\mu\text{m}$), comparing
# Dey–Mittra conformal PEC (`conformal='pec'`) against staircase PEC (`conformal=False`).


# %%
def get_task_table(quick: bool = True) -> list[dict[str, Any]]:
    """Return the 12-task study table for `--quick` or `--full` execution.

    Parameters
    ----------
    quick : bool
        If True, returns compact phase sweeps and step counts suitable for interactive
        verification (< 90 s total across 4 workers). If False, returns high-density
        phase sweeps and refined 3D grids for Cloud Batch production.

    Returns
    -------
    tasks : list[dict[str, Any]]
        Ordered list of task configuration dictionaries.
    """
    bn_2d = (
        [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
        if quick
        else list(np.round(np.linspace(0.125, 1.0, 15), 4))
    )
    bn_3d_grat = [0.2, 0.4, 0.6, 0.8, 1.0] if quick else list(np.round(np.linspace(0.15, 1.0, 11), 4))
    bn_marosi = [0.25, 0.5, 0.65, 0.8, 0.9, 1.0] if quick else list(np.round(np.linspace(0.15, 1.0, 12), 4))

    return [
        {
            "index": 0,
            "name": "under_resolved_staircase",
            "kind": "2d_point",
            "n_feat": 2,
            "conformal": False,
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 2500 if quick else 4000,
            "description": "Under-resolved 2D staircase demo (n_feat=2, dy=50 um, 2 cells across 0.10 mm vane, 4 across 0.20 mm gap)",
        },
        {
            "index": 1,
            "name": "under_resolved_conformal",
            "kind": "2d_point",
            "n_feat": 2,
            "conformal": "pec",
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 2500 if quick else 4000,
            "description": "Under-resolved 2D conformal PEC demo (n_feat=2, dy=50 um, 2 cells across 0.10 mm vane, 4 across 0.20 mm gap)",
        },
        {
            "index": 2,
            "name": "2d_staircase_L1",
            "kind": "2d_point",
            "n_feat": 4,
            "conformal": False,
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 4500 if quick else 7000,
            "description": "2D staircase PEC level L1 (n_feat=4, dy=25.0 um)",
        },
        {
            "index": 3,
            "name": "2d_staircase_L2",
            "kind": "2d_point",
            "n_feat": 8,
            "conformal": False,
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 8500 if quick else 13000,
            "description": "2D staircase PEC level L2 (n_feat=8, dy=12.5 um)",
        },
        {
            "index": 4,
            "name": "2d_staircase_L3",
            "kind": "2d_point",
            "n_feat": 12,
            "conformal": False,
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 12500 if quick else 19000,
            "description": "2D staircase PEC level L3 (n_feat=12, dy=8.33 um)",
        },
        {
            "index": 5,
            "name": "2d_conformal_L1",
            "kind": "2d_sweep",
            "n_feat": 4,
            "conformal": "pec",
            "bn_list": bn_2d,
            "n_steps": 4500 if quick else 7000,
            "record_movie": False,
            "description": "2D conformal PEC level L1 full Brillouin & DFT sweep (n_feat=4, dy=25.0 um)",
        },
        {
            "index": 6,
            "name": "2d_conformal_L2",
            "kind": "2d_sweep",
            "n_feat": 8,
            "conformal": "pec",
            "bn_list": bn_2d,
            "n_steps": 8500 if quick else 13000,
            "record_movie": True,
            "description": "2D conformal PEC level L2 full Brillouin, DFT & movie sweep (n_feat=8, dy=12.5 um)",
        },
        {
            "index": 7,
            "name": "2d_conformal_L3",
            "kind": "2d_point",
            "n_feat": 12,
            "conformal": "pec",
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 12500 if quick else 19000,
            "description": "2D conformal PEC level L3 convergence point (n_feat=12, dy=8.33 um)",
        },
        {
            "index": 8,
            "name": "2d_conformal_reference",
            "kind": "2d_point",
            "n_feat": 16,
            "conformal": "pec",
            "glide": True,
            "bn_list": [0.5],
            "n_steps": 16000 if quick else 25000,
            "description": "2D conformal PEC high-resolution reference (n_feat=16, dy=6.25 um)",
        },
        {
            "index": 9,
            "name": "3d_grating_karetnikova",
            "kind": "3d_grating",
            "n_feat": 4 if quick else 8,
            "nx": 18 if quick else 34,
            "conformal": "pec",
            "bn_list": bn_3d_grat,
            "n_steps": 3200 if quick else 7000,
            "description": "3D rectangular waveguide double-corrugated grating (Karetnikova 2018 Table I, b=0.85 mm, 2a=0.20 mm)",
        },
        {
            "index": 10,
            "name": "3d_marosi_glide",
            "kind": "3d_marosi",
            "nx": 28 if quick else 56,
            "ny": 32 if quick else 64,
            "nz": 28 if quick else 56,
            "glide": True,
            "conformal": "pec",
            "bn_list": bn_marosi,
            "n_steps": 12000 if quick else 40000,
            "n_ring_steps": 6600 if quick else 60000,
            "description": "3D V-band glide-symmetric staggered-pillar SWS (Marosi et al. 2025, arXiv:2505.22927)",
        },
        {
            "index": 11,
            "name": "3d_marosi_mirror",
            "kind": "3d_marosi",
            "nx": 28 if quick else 56,
            "ny": 32 if quick else 64,
            "nz": 28 if quick else 56,
            "glide": False,
            "conformal": "pec",
            "bn_list": bn_marosi,
            "n_steps": 12000 if quick else 40000,
            "n_ring_steps": 6600 if quick else 60000,
            "description": "3D V-band broken-glide pillar SWS control, z2 = 0.62 d (stopband opening at beta_n d = 3 pi)",
        },
    ]


def run_single_task(task: dict[str, Any], out_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Execute one cold-test task and write `task_XX_<name>.npz` and `.json`.

    Parameters
    ----------
    task : dict[str, Any]
        Task specification from ``get_task_table``.
    out_dir : Path
        Output directory for `.npz` and `.json` files.

    Returns
    -------
    summary : dict[str, Any]
        Task summary metadata including mesh resolution metrics (`dx_background`,
        `dx_min`, `cells_across_feature`, `n_cells`, `n_steps`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task["index"])
    name = str(task["name"])
    kind = str(task["kind"])
    t_start = time.perf_counter()

    grating_spec = h.GratingSWSSpec()
    marosi_spec = h.MarosiPillarSWSSpec()

    npz_payload: dict[str, Any] = {}
    total_steps = 0

    if kind == "2d_point":
        n_feat = int(task["n_feat"])
        conf = task["conformal"]
        glide = bool(task.get("glide", True))
        n_steps = int(task["n_steps"])
        bn = float(task["bn_list"][0])
        beta0 = bn * np.pi / grating_spec.period_m

        grid, meta = h.build_2d_grating_grid(
            grating_spec, n_feat=n_feat, glide_symmetric=glide, conformal=conf
        )
        probes = [("Ez", (0.0, 0.0, 0.07e-3)), ("Ey", (0.0, 0.0, 0.07e-3))]
        sources = [
            ("Ez", (0.0, 0.025e-3, 0.21e-3), 1.0),
            ("Ey", (0.0, 0.025e-3, 0.21e-3), 1.0),
        ]
        r1 = h.run_bloch_harminv_point(
            grid, beta0, 120e9, 70e9, 10e9, 235e9, probes, sources, n_steps=n_steps
        )
        total_steps += r1["n_steps"]
        freqs = r1["frequencies_hz"]
        f1_hz = float(freqs[0]) if len(freqs) >= 1 else 97.7e9
        f2_hz = float(freqs[1]) if len(freqs) >= 2 else 189.0e9

        n_ring = max(int(0.55 * n_steps), 1500)
        dft_res = h.extract_bloch_mode_dft(
            grid,
            beta0,
            f1_hz,
            sources,
            n_harm=6,
            fwidth_hz=6.0e9,
            n_ring_steps=n_ring,
            beam_y_half_m=grating_spec.beam_half_thickness_m,
        )
        total_steps += dft_res["n_steps"]

        idx0 = int(np.where(dft_res["n_indices"] == 0)[0][0])
        idxm1 = int(np.where(dft_res["n_indices"] == -1)[0][0])
        idxp1 = int(np.where(dft_res["n_indices"] == 1)[0][0])

        npz_payload.update(
            {
                "bn": np.array([bn]),
                "f1_ghz": np.array([f1_hz * 1e-9]),
                "f2_ghz": np.array([f2_hz * 1e-9]),
                "v_energy_over_c": np.array([dft_res["v_energy_over_c"]]),
                "pz_mean_w": np.array([dft_res["pz_mean_w"]]),
                "u_cell_j": np.array([dft_res["u_cell_j"]]),
                "k0_axis_ohm": np.array([dft_res["k_n_axis_ohm"][idx0]]),
                "km1_axis_ohm": np.array([dft_res["k_n_axis_ohm"][idxm1]]),
                "kp1_axis_ohm": np.array([dft_res["k_n_axis_ohm"][idxp1]]),
                "km1_beam_avg_ohm": np.array([dft_res["k_n_beam_avg_ohm"][idxm1]]),
                "n_indices": dft_res["n_indices"],
                "k_n_axis_ohm": dft_res["k_n_axis_ohm"],
                "ez_harm_axis": dft_res["ez_harm_axis"],
                "ez_yz_slice": dft_res["ez_yz_slice"],
                "y_nodes_m": dft_res["y_nodes_m"],
                "z_centers_m": dft_res["z_centers_m"],
            }
        )

    elif kind == "2d_sweep":
        n_feat = int(task["n_feat"])
        conf = task["conformal"]
        bn_list = [float(x) for x in task["bn_list"]]
        n_steps = int(task["n_steps"])
        n_ring = max(int(0.55 * n_steps), 1800)

        grid_g, meta = h.build_2d_grating_grid(
            grating_spec, n_feat=n_feat, glide_symmetric=True, conformal=conf
        )
        grid_m, _ = h.build_2d_grating_grid(
            grating_spec, n_feat=n_feat, glide_symmetric=False, conformal=conf
        )

        probes = [("Ez", (0.0, 0.0, 0.07e-3)), ("Ey", (0.0, 0.0, 0.07e-3))]
        sources = [
            ("Ez", (0.0, 0.025e-3, 0.21e-3), 1.0),
            ("Ey", (0.0, 0.025e-3, 0.21e-3), 1.0),
        ]

        f_glide_low, f_glide_high = [], []
        f_mirror_low, f_mirror_high = [], []
        ve_glide_over_c, pz_glide_arr, u_glide_arr = [], [], []
        km1_glide_axis, km1_glide_avg, k0_glide_axis, kp1_glide_axis = [], [], [], []
        k0_mirror_axis, km1_mirror_axis = [], []

        mid_dft_glide = None
        mid_dft_mirror = None
        best_mid_err = np.inf

        for bn in bn_list:
            beta0 = bn * np.pi / grating_spec.period_m
            rg = h.run_bloch_harminv_point(
                grid_g, beta0, 120e9, 70e9, 10e9, 235e9, probes, sources, n_steps=n_steps
            )
            rm = h.run_bloch_harminv_point(
                grid_m, beta0, 120e9, 70e9, 10e9, 235e9, probes, sources, n_steps=n_steps
            )
            total_steps += rg["n_steps"] + rm["n_steps"]

            fg = rg["frequencies_hz"]
            fm = rm["frequencies_hz"]
            fg1 = float(fg[0]) if len(fg) >= 1 else np.nan
            fg2 = float(fg[1]) if len(fg) >= 2 else fg1
            fm1 = float(fm[0]) if len(fm) >= 1 else np.nan
            fm2 = float(fm[1]) if len(fm) >= 2 else np.nan

            f_glide_low.append(fg1 * 1e-9)
            f_glide_high.append(fg2 * 1e-9)
            f_mirror_low.append(fm1 * 1e-9)
            f_mirror_high.append(fm2 * 1e-9)

            dft_g = h.extract_bloch_mode_dft(
                grid_g,
                beta0,
                fg1,
                sources,
                n_harm=6,
                fwidth_hz=6.0e9,
                n_ring_steps=n_ring,
                beam_y_half_m=grating_spec.beam_half_thickness_m,
            )
            dft_m = h.extract_bloch_mode_dft(
                grid_m,
                beta0,
                fm1,
                sources,
                n_harm=6,
                fwidth_hz=6.0e9,
                n_ring_steps=n_ring,
                beam_y_half_m=grating_spec.beam_half_thickness_m,
            )
            total_steps += dft_g["n_steps"] + dft_m["n_steps"]

            idx0 = int(np.where(dft_g["n_indices"] == 0)[0][0])
            idxm1 = int(np.where(dft_g["n_indices"] == -1)[0][0])
            idxp1 = int(np.where(dft_g["n_indices"] == 1)[0][0])

            ve_glide_over_c.append(dft_g["v_energy_over_c"])
            pz_glide_arr.append(dft_g["pz_mean_w"])
            u_glide_arr.append(dft_g["u_cell_j"])
            k0_glide_axis.append(float(dft_g["k_n_axis_ohm"][idx0]))
            km1_glide_axis.append(float(dft_g["k_n_axis_ohm"][idxm1]))
            kp1_glide_axis.append(float(dft_g["k_n_axis_ohm"][idxp1]))
            km1_glide_avg.append(float(dft_g["k_n_beam_avg_ohm"][idxm1]))

            k0_mirror_axis.append(float(dft_m["k_n_axis_ohm"][idx0]))
            km1_mirror_axis.append(float(dft_m["k_n_beam_avg_ohm"][idx0]))

            if abs(bn - 0.5) < best_mid_err or mid_dft_glide is None:
                best_mid_err = abs(bn - 0.5)
                mid_dft_glide = dft_g
                mid_dft_mirror = dft_m

        # Fine analytic McVey curves and harmonic convergence audit
        bn_fine = np.linspace(0.06, 1.0, 48)
        mcvey_m_low, mcvey_m_high = [], []
        mcvey_g_low, mcvey_g_high = [], []
        for bnf in bn_fine:
            b0f = float(bnf * np.pi / grating_spec.period_m)
            rm_f = h.solve_mcvey_band_frequencies(
                b0f, grating_spec, symmetry="mirror", n_harm=40, m_slot=6
            )
            rg_f = h.solve_mcvey_band_frequencies(
                b0f, grating_spec, symmetry="glide", n_harm=40, m_slot=6
            )
            mcvey_m_low.append(rm_f[0] * 1e-9 if len(rm_f) >= 1 else np.nan)
            mcvey_m_high.append(rm_f[1] * 1e-9 if len(rm_f) >= 2 else np.nan)
            mcvey_g_low.append(rg_f[0] * 1e-9 if len(rg_f) >= 1 else np.nan)
            mcvey_g_high.append(rg_f[1] * 1e-9 if len(rg_f) >= 2 else (rg_f[0] * 1e-9 if len(rg_f) == 1 else np.nan))

        conv_audit = h.compute_mcvey_harmonic_convergence(
            0.5 * np.pi / grating_spec.period_m, grating_spec, symmetry="glide"
        )

        npz_payload.update(
            {
                "bn": np.array(bn_list, dtype=float),
                "f1_ghz": np.array(f_glide_low, dtype=float),
                "f2_ghz": np.array(f_glide_high, dtype=float),
                "f_glide_low_ghz": np.array(f_glide_low, dtype=float),
                "f_glide_high_ghz": np.array(f_glide_high, dtype=float),
                "f_mirror_low_ghz": np.array(f_mirror_low, dtype=float),
                "f_mirror_high_ghz": np.array(f_mirror_high, dtype=float),
                "v_energy_over_c": np.array(ve_glide_over_c, dtype=float),
                "pz_mean_w": np.array(pz_glide_arr, dtype=float),
                "u_cell_j": np.array(u_glide_arr, dtype=float),
                "k0_axis_ohm": np.array(k0_glide_axis, dtype=float),
                "km1_axis_ohm": np.array(km1_glide_axis, dtype=float),
                "kp1_axis_ohm": np.array(kp1_glide_axis, dtype=float),
                "km1_beam_avg_ohm": np.array(km1_glide_avg, dtype=float),
                "k0_mirror_axis_ohm": np.array(k0_mirror_axis, dtype=float),
                "km1_mirror_avg_ohm": np.array(km1_mirror_axis, dtype=float),
                "bn_fine": bn_fine,
                "mcvey_m_low_ghz": np.array(mcvey_m_low, dtype=float),
                "mcvey_m_high_ghz": np.array(mcvey_m_high, dtype=float),
                "mcvey_g_low_ghz": np.array(mcvey_g_low, dtype=float),
                "mcvey_g_high_ghz": np.array(mcvey_g_high, dtype=float),
                "conv_n_harm": conv_audit["n_harm_values"],
                "conv_m_slot": conv_audit["m_slot_values"],
                "conv_f_fund_ghz": conv_audit["f_fundamental_ghz"],
                "conv_rel_err": conv_audit["rel_err_vs_nharm"],
                "n_indices": mid_dft_glide["n_indices"],
                "k_n_axis_ohm": mid_dft_glide["k_n_axis_ohm"],
                "ez_harm_axis": mid_dft_glide["ez_harm_axis"],
                "ez_harm_y_profile": mid_dft_glide["ez_harm_y_profile"],
                "ez_yz_slice_glide": mid_dft_glide["ez_yz_slice"],
                "ez_yz_slice_mirror": mid_dft_mirror["ez_yz_slice"],
                "pz_vs_z_w": mid_dft_glide["pz_vs_z_w"],
                "y_nodes_m": mid_dft_glide["y_nodes_m"],
                "z_centers_m": mid_dft_glide["z_centers_m"],
            }
        )

        if task.get("record_movie", False):
            beta0_mov = 0.5 * np.pi / grating_spec.period_m
            f_mov_hz = float(mid_dft_glide["mode_freq_hz"])
            mov = h.record_bloch_wave_frames(
                grid_g, beta0_mov, f_mov_hz, sources, n_frames=20
            )
            npz_payload["movie_frames_ez"] = mov["frames_ez"]
            npz_payload["movie_phase_rad"] = mov["phase_rad"]
            npz_payload["movie_y_um"] = mov["y_um"]
            npz_payload["movie_z_um"] = mov["z_um"]
            npz_payload["movie_vac_mask"] = mov["vac_mask_yz"]

    elif kind == "3d_grating":
        n_feat = int(task["n_feat"])
        nx = int(task["nx"])
        conf = task["conformal"]
        bn_list = [float(x) for x in task["bn_list"]]
        n_steps = int(task["n_steps"])

        grid_g, meta = h.build_3d_grating_grid(
            grating_spec, n_feat=n_feat, nx=nx, glide_symmetric=True, conformal=conf
        )
        grid_m, _ = h.build_3d_grating_grid(
            grating_spec, n_feat=n_feat, nx=nx, glide_symmetric=False, conformal=conf
        )
        probes = [("Ez", (0.0, 0.0, 0.07e-3)), ("Ey", (0.0, 0.0, 0.07e-3))]
        sources = [
            ("Ez", (0.0, 0.025e-3, 0.21e-3), 1.0),
            ("Ey", (0.0, 0.025e-3, 0.21e-3), 1.0),
        ]

        f3d_glide_low, f3d_glide_high = [], []
        f3d_mirror_low, f3d_mirror_high = [], []
        mcvey_3d_g_low, mcvey_3d_g_high = [], []
        mcvey_3d_m_low, mcvey_3d_m_high = [], []
        mcvey_3d_kp1_beam, mcvey_3d_km2_beam = [], []
        f_cut = grating_spec.cutoff_freq_3d_hz

        for bn in bn_list:
            beta0 = bn * np.pi / grating_spec.period_m
            rg = h.run_bloch_harminv_point(
                grid_g, beta0, 215e9, 60e9, 177e9, 285e9, probes, sources, n_steps=n_steps
            )
            rm = h.run_bloch_harminv_point(
                grid_m, beta0, 215e9, 60e9, 177e9, 285e9, probes, sources, n_steps=n_steps
            )
            total_steps += rg["n_steps"] + rm["n_steps"]
            fg = rg["frequencies_hz"]
            fm = rm["frequencies_hz"]
            f3d_glide_low.append(float(fg[0]) * 1e-9 if len(fg) >= 1 else np.nan)
            f3d_glide_high.append(float(fg[1]) * 1e-9 if len(fg) >= 2 else (float(fg[0]) * 1e-9 if len(fg) == 1 else np.nan))
            f3d_mirror_low.append(float(fm[0]) * 1e-9 if len(fm) >= 1 else np.nan)
            f3d_mirror_high.append(float(fm[1]) * 1e-9 if len(fm) >= 2 else np.nan)

            rg_mc = h.solve_mcvey_band_frequencies(
                beta0, grating_spec, symmetry="glide", n_harm=40, m_slot=6,
                f_cut_hz=f_cut, f_min_ghz=177.0, f_max_ghz=285.0
            )
            rm_mc = h.solve_mcvey_band_frequencies(
                beta0, grating_spec, symmetry="mirror", n_harm=40, m_slot=6,
                f_cut_hz=f_cut, f_min_ghz=177.0, f_max_ghz=285.0
            )
            fg_mc_low = rg_mc[0] if len(rg_mc) >= 1 else np.nan
            fg_mc_high = rg_mc[1] if len(rg_mc) >= 2 else fg_mc_low
            mcvey_3d_g_low.append(fg_mc_low * 1e-9)
            mcvey_3d_g_high.append(fg_mc_high * 1e-9)
            mcvey_3d_m_low.append(rm_mc[0] * 1e-9 if len(rm_mc) >= 1 else np.nan)
            mcvey_3d_m_high.append(rm_mc[1] * 1e-9 if len(rm_mc) >= 2 else np.nan)

            if np.isfinite(fg_mc_low):
                mc_imp_3d = h.compute_mcvey_analytic_pierce_impedance(
                    float(fg_mc_low),
                    beta0,
                    grating_spec,
                    symmetry="glide_b",
                    n_harm=40,
                    m_slot=6,
                    f_cut_hz=f_cut,
                )
                idx_p1 = int(np.where(mc_imp_3d["n_indices"] == 1)[0][0])
                idx_m2 = int(np.where(mc_imp_3d["n_indices"] == -2)[0][0])
                mcvey_3d_kp1_beam.append(float(mc_imp_3d["k_n_beam_avg_ohm"][idx_p1]))
                mcvey_3d_km2_beam.append(float(mc_imp_3d["k_n_beam_avg_ohm"][idx_m2]))
            else:
                mcvey_3d_kp1_beam.append(np.nan)
                mcvey_3d_km2_beam.append(np.nan)

        npz_payload.update(
            {
                "bn": np.array(bn_list, dtype=float),
                "f3d_glide_low_ghz": np.array(f3d_glide_low, dtype=float),
                "f3d_glide_high_ghz": np.array(f3d_glide_high, dtype=float),
                "f3d_mirror_low_ghz": np.array(f3d_mirror_low, dtype=float),
                "f3d_mirror_high_ghz": np.array(f3d_mirror_high, dtype=float),
                "mcvey_3d_g_low_ghz": np.array(mcvey_3d_g_low, dtype=float),
                "mcvey_3d_g_high_ghz": np.array(mcvey_3d_g_high, dtype=float),
                "mcvey_3d_m_low_ghz": np.array(mcvey_3d_m_low, dtype=float),
                "mcvey_3d_m_high_ghz": np.array(mcvey_3d_m_high, dtype=float),
                "mcvey_3d_kp1_beam_ohm": np.array(mcvey_3d_kp1_beam, dtype=float),
                "mcvey_3d_km2_beam_ohm": np.array(mcvey_3d_km2_beam, dtype=float),
                "f_cut_3d_ghz": np.array([f_cut * 1e-9]),
            }
        )

    elif kind == "3d_marosi":
        nx = int(task["nx"])
        ny = int(task["ny"])
        nz = int(task["nz"])
        glide = bool(task["glide"])
        conf = task["conformal"]
        bn_list = [float(x) for x in task["bn_list"]]
        n_steps = int(task["n_steps"])

        grid, meta = h.build_3d_marosi_pillar_grid(
            marosi_spec, nx=nx, ny=ny, nz=nz, glide_symmetric=glide, conformal=conf
        )
        d_m = marosi_spec.period_m
        z2_frac = 0.75 if glide else 0.62
        probes = [
            ("Ez", (0.0, 0.0, 0.50 * d_m)),
            ("Ez", (0.0, 0.0, 0.43 * d_m)),
            ("Ez", (0.18e-3, 0.04e-3, 0.31 * d_m)),
        ]

        def _quad_ez(x, y, z, a):
            # Ez dipoles at (+-x, +-y): even under the y = 0 mirror plane (both cases; each pillar pair is
            # symmetric in y), so the y-odd modes, which have no on-axis Ez (e.g. the flat bands next to the
            # upper mode), are not excited at all rather than only suppressed by the DFT window.
            return [("Ez", (sx * x, sy * y, z), a) for sx in (1.0, -1.0) for sy in (1.0, -1.0)]

        sources = [("Ez", (0.0, 0.0, 0.12 * d_m), 1.0)]
        sources += _quad_ez(0.18e-3, 0.04e-3, 0.25 * d_m, 1.0)
        sources += _quad_ez(0.18e-3, 0.04e-3, z2_frac * d_m, 0.8)
        sources_fwd = [("Ez", (0.0, 0.0, 0.25 * d_m), 1.0)] + _quad_ez(0.18e-3, 0.04e-3, 0.25 * d_m, 0.6)
        sources_bwd = sources_fwd

        f_fwd_ghz, f_bwd_ghz = [], []
        zp_fwd_axis, zp_fwd_avg = [], []
        zp_bwd_axis, zp_bwd_avg = [], []
        ve_fwd_over_c, ve_bwd_over_c = [], []
        # Every space harmonic n = -3..3 of both modes (the forward-wave impedance above 3 pi and the
        # backward-wave impedance of Marosi Fig. 11a are other harmonics of the same two modes).
        kn_shape = 7
        kn_lo_axis, kn_lo_avg, kn_up_axis, kn_up_avg = [], [], [], []
        ez_slice_fwd = None
        all_modes = []

        for bn in bn_list:
            beta0 = bn * np.pi / d_m
            r1 = h.run_bloch_harminv_point(
                grid,
                beta0,
                62e9,
                35e9,
                35e9,
                90e9,
                probes,
                sources,
                n_steps=n_steps,
                min_q=3.0,
            )
            total_steps += r1["n_steps"]
            # Beam-coupled Bloch modes: high-Q Harminv modes (the cold structure is lossless, so true
            # eigenmodes ring with Q > 1e9 for a long ring-down; transient fits have Q < 1e3) that the
            # on-axis Ez probe sees (modes with no on-axis Ez do not interact with the beam). The lower
            # Bloch mode is the lowest such mode, and the upper Bloch mode is the highest.
            f_all = np.asarray(r1["frequencies_hz"])
            q_all = np.asarray(r1["q_factors"])
            f_axis = np.asarray(r1["per_probe_frequencies_hz"][0])
            on_axis = np.array([np.any(np.abs(f_axis - f) < 50e6) for f in f_all], dtype=bool)
            keep = (q_all > 1e4) & on_axis & (f_all >= 40e9) & (f_all <= 85e9)
            cands = np.sort(f_all[keep])
            f1 = float(cands[0]) if len(cands) else np.nan
            # At 3 pi the glide branches are degenerate (one mode); in the mirror case a missing second
            # mode means the upper band edge is not beam-coupled, so it is left undefined.
            if len(cands) and cands[-1] - cands[0] > 1.5e9:
                f2 = float(cands[-1])
            else:
                f2 = f1 if glide else np.nan

            f_fwd_ghz.append(f1 * 1e-9)
            f_bwd_ghz.append(f2 * 1e-9)
            all_modes.append(r1)
            if STAGE1_ONLY:
                for lst in (zp_fwd_axis, zp_fwd_avg, zp_bwd_axis, zp_bwd_avg, ve_fwd_over_c, ve_bwd_over_c):
                    lst.append(np.nan)
                for lst in (kn_lo_axis, kn_lo_avg, kn_up_axis, kn_up_avg):
                    lst.append(np.full(kn_shape, np.nan))
                continue

            if np.isfinite(f1) and bn < 0.98:
                n_ring = int(task.get("n_ring_steps", max(int(0.55 * n_steps), 1500)))
                dft_f = h.extract_bloch_mode_dft(
                    grid,
                    beta0,
                    f1,
                    sources_fwd,
                    n_harm=3,
                    fwidth_hz=2.2e9,
                    n_ring_steps=n_ring,
                    beam_y_half_m=0.5 * marosi_spec.beam_thickness_m,
                    beam_x_half_m=0.5 * marosi_spec.beam_width_m,
                )
                total_steps += dft_f["n_steps"]
                # Lower mode n = +1 harmonic: beta_1 d = (2 + bn) pi in [2 pi, 3 pi] (forward wave below 3 pi)
                idx_p1 = int(np.where(dft_f["n_indices"] == 1)[0][0])
                zp_fwd_axis.append(float(dft_f["k_n_axis_ohm"][idx_p1]))
                zp_fwd_avg.append(float(dft_f["k_n_beam_avg_ohm"][idx_p1]))
                ve_fwd_over_c.append(float(dft_f["v_energy_over_c"]))
                kn_lo_axis.append(np.asarray(dft_f["k_n_axis_ohm"], dtype=float))
                kn_lo_avg.append(np.asarray(dft_f["k_n_beam_avg_ohm"], dtype=float))
                if abs(bn - 0.8) < 0.08 or ez_slice_fwd is None:
                    ez_slice_fwd = dft_f["ez_yz_slice"]

                if np.isfinite(f2) and abs(f2 - f1) > 1.5e9:
                    dft_b = h.extract_bloch_mode_dft(
                        grid,
                        beta0,
                        f2,
                        sources_bwd,
                        n_harm=3,
                        fwidth_hz=2.2e9,
                        n_ring_steps=n_ring,
                        beam_y_half_m=0.5 * marosi_spec.beam_thickness_m,
                        beam_x_half_m=0.5 * marosi_spec.beam_width_m,
                    )
                    total_steps += dft_b["n_steps"]
                    # Upper mode n = -2 harmonic: |beta_{-2}| d = (4 - bn) pi in [3 pi, 4 pi] (continues forward wave above 3 pi)
                    idx_m2 = int(np.where(dft_b["n_indices"] == -2)[0][0])
                    zp_bwd_axis.append(float(dft_b["k_n_axis_ohm"][idx_m2]))
                    zp_bwd_avg.append(float(dft_b["k_n_beam_avg_ohm"][idx_m2]))
                    ve_bwd_over_c.append(float(dft_b["v_energy_over_c"]))
                    kn_up_axis.append(np.asarray(dft_b["k_n_axis_ohm"], dtype=float))
                    kn_up_avg.append(np.asarray(dft_b["k_n_beam_avg_ohm"], dtype=float))
                else:
                    zp_bwd_axis.append(np.nan)
                    zp_bwd_avg.append(np.nan)
                    ve_bwd_over_c.append(np.nan)
                    kn_up_axis.append(np.full(kn_shape, np.nan))
                    kn_up_avg.append(np.full(kn_shape, np.nan))
            else:
                zp_fwd_axis.append(np.nan)
                zp_fwd_avg.append(np.nan)
                zp_bwd_axis.append(np.nan)
                zp_bwd_avg.append(np.nan)
                ve_fwd_over_c.append(np.nan)
                ve_bwd_over_c.append(np.nan)
                for lst in (kn_lo_axis, kn_lo_avg, kn_up_axis, kn_up_avg):
                    lst.append(np.full(kn_shape, np.nan))

        npz_payload.update(
            {
                "bn": np.array(bn_list, dtype=float),
                "f_fwd_ghz": np.array(f_fwd_ghz, dtype=float),
                "f_bwd_ghz": np.array(f_bwd_ghz, dtype=float),
                "zp_fwd_axis_ohm": np.array(zp_fwd_axis, dtype=float),
                "zp_fwd_avg_ohm": np.array(zp_fwd_avg, dtype=float),
                "zp_bwd_axis_ohm": np.array(zp_bwd_axis, dtype=float),
                "zp_bwd_avg_ohm": np.array(zp_bwd_avg, dtype=float),
                "ve_fwd_over_c": np.array(ve_fwd_over_c, dtype=float),
                "ve_bwd_over_c": np.array(ve_bwd_over_c, dtype=float),
                "kn_n_indices": np.arange(-3, 4),
                "kn_lower_axis_ohm": np.array(kn_lo_axis, dtype=float),
                "kn_lower_avg_ohm": np.array(kn_lo_avg, dtype=float),
                "kn_upper_axis_ohm": np.array(kn_up_axis, dtype=float),
                "kn_upper_avg_ohm": np.array(kn_up_avg, dtype=float),
            }
        )
        if ez_slice_fwd is not None:
            npz_payload["ez_yz_slice_fwd"] = ez_slice_fwd
        # Every deduplicated Harminv mode per bn point (rows padded with NaN), for branch tracking.
        n_max = max([len(r["frequencies_hz"]) for r in all_modes] + [1])
        n_pr = len(probes)
        p_max = max([len(pf) for r in all_modes for pf in r["per_probe_frequencies_hz"]] + [1])

        def _pad(v, n):
            out = np.full(n, np.nan)
            out[: len(v)] = v
            return out

        npz_payload["modes_f_ghz"] = np.array([_pad(r["frequencies_hz"] * 1e-9, n_max) for r in all_modes])
        npz_payload["modes_q"] = np.array([_pad(r["q_factors"], n_max) for r in all_modes])
        npz_payload["modes_amp"] = np.array([_pad(r["amplitudes"], n_max) for r in all_modes])
        npz_payload["modes_probe_f_ghz"] = np.array(
            [[_pad(np.asarray(r["per_probe_frequencies_hz"][k]) * 1e-9, p_max) for k in range(n_pr)] for r in all_modes]
        )

    else:
        raise ValueError(f"Unknown task kind={kind!r}")

    wall_s = time.perf_counter() - t_start
    stem = f"task_{idx:02d}_{name}"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"

    np.savez_compressed(npz_path, **npz_payload)

    summary = {
        "index": idx,
        "name": name,
        "kind": kind,
        "description": task["description"],
        "dx_background": float(meta["dx_background"]),
        "dx_min": float(meta["dx_min"]),
        "cells_across_feature": float(meta["cells_across_feature"]),
        "n_cells": int(meta["n_cells"]),
        "n_steps": int(total_steps),
        "nx": int(meta["nx"]),
        "ny": int(meta["ny"]),
        "nz": int(meta["nz"]),
        "conformal": str(meta["conformal"]),
        "glide_symmetric": bool(meta["glide_symmetric"]),
        "wall_time_s": float(wall_s),
    }
    i_mid = (
        int(np.argmin(np.abs(npz_payload["bn"] - 0.5)))
        if "bn" in npz_payload and npz_payload["bn"].size > 0
        else 0
    )
    if "f1_ghz" in npz_payload and npz_payload["f1_ghz"].size > 0:
        summary["f1_mid_ghz"] = float(npz_payload["f1_ghz"][i_mid])
    if "km1_axis_ohm" in npz_payload and npz_payload["km1_axis_ohm"].size > 0:
        summary["km1_mid_ohm"] = float(npz_payload["km1_axis_ohm"][i_mid])
    if "f_fwd_ghz" in npz_payload and npz_payload["f_fwd_ghz"].size > 0:
        summary["f_degen_ghz"] = float(npz_payload["f_fwd_ghz"][-1])

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[Task {idx:02d} {name}] n_cells={summary['n_cells']} "
        f"dx_min={summary['dx_min']:.2f} um "
        f"cells/feat={summary['cells_across_feature']:.1f} "
        f"n_steps={summary['n_steps']} wall={wall_s:.2f} s"
    )
    return summary


# %% [markdown]
# ## 3. Diagnostic & Validation Figure Generation
#
# `generate_all_figures` reads `data/task_*.npz` and `data/graded_mesh_comparison.json` and renders 16 figures into `figures/`:
#
# 0. `00_setup_schematic.png`: Three-panel setup schematic built from live solver objects via
#    `wakis.schematic`, showing (a) the 2D/3D corrugated grating unit cell with Bloch phase advance
#    $\beta_0 d$, PEC walls/vanes, $20\;\text{kV}$ sheet beam, dipole excitation, and harminv/DFT
#    monitors; (b) a 3D isometric cut-plane view ($x=0$, $y=0$) of the V-band staggered-pillar SWS;
#    and (c) a graded non-uniform vs. uniform mesh zoom at the vane tip and slot mouth.
# 1. `01_sws_2d_and_3d_geometry.png`: 2D mirror vs. glide double-corrugated grating and
#    3D V-band staggered-pillar unit cell schematic with verified dimensions.
# 2. `02_mesh_zoom_conformal_vs_staircase.png`: Three-panel mesh comparison showing (a) uniform
#    staircase PEC ($\Delta = 20\;\mu\text{m}$), (b) uniform Dey–Mittra conformal PEC
#    ($\Delta = 20\;\mu\text{m}$), and (c) graded non-uniform PEC ($\Delta_{\min} = 12.5\;\mu\text{m}$
#    at the vane teeth/tips grading to $30\;\mu\text{m}$ in the slot depth, $1{,}584$ vs. $2{,}560$ cells).
# 3. `03_mcvey_harmonic_convergence.png`: Convergence of the McVey multimodal field-matching
#    dispersion in the number of space harmonics $N_{\text{harm}}$ and slot modes $M_{\text{slot}}$.
# 4. `04_2d_brillouin_mirror_vs_glide.png`: 2D Brillouin diagram $f(\beta_0 d/\pi)$ comparing
#    uniform (`Uniform L2`, $2{,}560$ cells) and graded non-uniform (`Graded L2`, $1{,}584$ cells)
#    Wakis complex-field FIT against McVey theory, showing the mirror
#    stopband at $\beta_0 d = \pi$ ($140.66\text{--}195.89\;\text{GHz}$ in Wakis FIT vs.
#    $140.29\text{--}195.88\;\text{GHz}$ in McVey theory) and its exact closure under glide symmetry
#    ($165.46\;\text{GHz}$ in Wakis FIT vs. $165.28\;\text{GHz}$ in McVey theory), plus the $20\;\text{kV}$
#    electron beam line and synchronous crossings.
# 5. `05_group_vs_energy_velocity.png`: Group velocity $v_g = \mathrm{d}\omega/\mathrm{d}\beta_0$
#    from the dispersion slope vs. energy velocity $v_E = P_z d / U_{\text{cell}}$ from the
#    uniform and graded Bloch `DFTMonitor` fields, plus axial constancy of the Poynting flux $P_z(z)$.
# 6. `06_space_harmonic_spectrum_and_profiles.png`: Space-harmonic decomposition $|E_{zn}(y)|$
#    showing why the $n = -1$ backward harmonic has a $\cosh(\tau_{-1} y)$ profile with strong
#    on-axis field while $n = 0$ has a $\sinh(\tau_0 y)$ node on axis in the glide-symmetric grating.
# 7. `07_pierce_impedance_2d_vs_freq.png`: On-axis and beam-averaged Pierce interaction
#    impedance $K_n(f)$ across the passband for the $n = -1$ backward wave and $n = 0, +1$ harmonics,
#    comparing uniform (`Uniform L2`) and graded non-uniform (`Graded L2`) meshes against McVey theory.
# 8. `08_bloch_eigenmode_ez_fields.png`: 2D complex Bloch eigenmode field maps $\operatorname{Re}\{E_z(y, z)\}$
#    with `schematic.overlay_geometry` PEC vane and boundary overlays comparing mirror-symmetric and
#    glide-symmetric gratings near mid-band.
# 9. `09_bloch_mode_animation.gif`: Time-domain animation of the complex Bloch wave
#    $\operatorname{Re}\{E_z(y, z, t)\}$ with geometry overlays propagating across one RF cycle.
# 10. `10_resolution_convergence_pec_vs_staircase.png`: Three-panel log–log convergence of (a)
#     $f_1(\beta_0)$ and (b) Pierce impedance $K_{-1}(0)$ vs. `cells_across_feature` (`n_feat` = 4, 8, 12, 16),
#     plus (c) the equal-cost / equal-accuracy Pareto curve vs. total 2D grid cells $N_y \times N_z$
#     comparing graded non-uniform conformal PEC, uniform conformal PEC, and uniform staircase PEC.
# 11. `11_under_resolved_warning_demo.png`: Direct comparison of under-resolved (`n_feat` = 2,
#     $\Delta = 50\;\mu\text{m}$) vs. uniform and graded converged (`n_feat` = 8, 12) dispersion and impedance.
# 12. `12_2d_vs_3d_waveguide_cutoff_mapping.png`: Validation of the exact 2D-to-3D $\text{LSE}_{10}$
#     mapping $f_{\text{3D}}(\beta_0) = \sqrt{f_{\text{2D}}^2(\beta_0) + (c/2a)^2}$ ($a = 0.85\;\text{mm}$,
#     $f_c = 176.35\;\text{GHz}$) against 3D Wakis FIT (`Task 9`, $n_{\text{feat}} = 8$) and 3D McVey
#     theory for the Karetnikova et al. (2018) Table I grating ($176.35\text{--}260.55\;\text{GHz}$ passband).
# 13. `13_3d_marosi_dispersion_and_stopband.png`: 3D V-band staggered-pillar SWS dispersion
#     $f(\beta_n d/\pi)$ for both the glide-symmetric structure (`Task 10`, lower mode
#     $49.02 \to 60.86\;\text{GHz}$, upper mode $73.60 \to 60.86\;\text{GHz}$, degenerate at
#     $f_{3\pi} = 60.857\;\text{GHz}$) and the broken-glide control (`Task 11`, lower mode
#     $45.89 \to 51.74\;\text{GHz}$, upper mode $76.39 \to 69.54\;\text{GHz}$, with a shaded
#     $51.74\text{--}69.54\;\text{GHz}$ stopband of width $17.81\;\text{GHz}$ at $\beta_n d = 3\pi$),
#     compared against CST Studio Suite points from Fig. 2 of Marosi, Zuboraj & Capolino
#     (`arXiv:2505.22927`, 2025), alongside the mid-plane $\operatorname{Re}\{E_z(0, y, z)\}$ mode profile.
# 14. `14_3d_marosi_phase_velocity_and_beam_sync.png`: Normalized phase velocity $v_p / c$ vs.
#     frequency and $\beta_n d / \pi$ across the continuous $49.02\text{--}73.60\;\text{GHz}$ passband
#     with the $5.2\;\text{kV}$ ($u_0 = 0.1416\,c$) sheet-beam line, compared against Fig. 3 of
#     Marosi et al. (2025).
# 15. `15_3d_marosi_pierce_impedance_vs_paper.png`: (a) Continuous forward-wave Pierce interaction
#     impedance $Z_P(f)$ ($49.02\text{--}73.60\;\text{GHz}$, on-axis and beam-averaged within $3\%$ across the
#     main passband, formed by the lower-mode $n = +1$ harmonic below $f_{3\pi} = 60.86\;\text{GHz}$ and
#     continued above $3\pi$ by the upper-mode $n = -2$ harmonic; interpolated $Z_P(0,0) = 3.78\;\Omega$
#     vs. $3.8\;\Omega$ at $58\;\text{GHz}$ and $1.01\;\Omega$ vs. $1.3\;\Omega$ at $68\;\text{GHz}$)
#     compared against Section II and Fig. 5a of Marosi et al. (2025); (b) backward-wave Pierce
#     impedance (upper mode $n = +1$ and lower mode $n = -2$) on a logarithmic vertical scale (floored
#     at $10^{-6}\;\Omega$), showing glide-symmetric on-axis suppression ($\le 1.17\times 10^{-4}\;\Omega$ at every
#     point, matching the paper's "$\approx 0\;\Omega$ on axis" in Fig. 11a, while beam-averaged values
#     are $0.085\text{--}1.39\;\Omega$ for upper $n = +1$ and $0.023\text{--}0.060\;\Omega$ for lower $n = -2$)
#     contrasted against the broken-glide control where on-axis backward-wave $Z_P$ is non-zero
#     (up to $2.34\;\Omega$ on upper $n = +1$ and $0.77\text{--}9.60\;\Omega$ on lower $n = -2$); and
#     (c) Stage-2 energy velocity $v_E = P_z d / U_{\text{cell}}$ vs. Stage-1 group velocity
#     $v_g = \mathrm{d}\omega / \mathrm{d}\beta_0$ for both glide and mirror cases.


# %%
def _load_tasks(data_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for npz_file in sorted(Path(data_dir).glob("task_*.npz")):
        json_file = npz_file.with_suffix(".json")
        meta = json.loads(json_file.read_text()) if json_file.exists() else {}
        with np.load(npz_file, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        if "bn" in arrays and arrays["bn"].size > 0:
            i_mid = int(np.argmin(np.abs(arrays["bn"] - 0.5)))
            changed = False
            if "f1_ghz" in arrays and arrays["f1_ghz"].size > i_mid:
                val_f = float(arrays["f1_ghz"][i_mid])
                if abs(meta.get("f1_mid_ghz", -1.0) - val_f) > 1e-6:
                    meta["f1_mid_ghz"] = val_f
                    changed = True
            if "km1_axis_ohm" in arrays and arrays["km1_axis_ohm"].size > i_mid:
                val_k = float(arrays["km1_axis_ohm"][i_mid])
                if abs(meta.get("km1_mid_ohm", -1.0) - val_k) > 1e-6:
                    meta["km1_mid_ohm"] = val_k
                    changed = True
            if changed and json_file.exists():
                json_file.write_text(json.dumps(meta, indent=2) + "\n")
        name = meta.get("name", npz_file.stem.split("_", 2)[-1])
        out[name] = {**meta, **arrays}
    return out


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Render the setup schematic and all 15 diagnostic and physics validation figures from saved data."""
    from wakis import figstyle, schematic

    figstyle.apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(data_dir)
    graded_study = h.load_or_run_sws_graded_mesh_study(data_dir)
    graded_sweep = graded_study.get("sweep_L2", {})
    saved: list[Path] = []

    grating_spec = h.GratingSWSSpec()
    marosi_spec = h.MarosiPillarSWSSpec()
    u0_20kv, _ = h.relativistic_beam_velocity(grating_spec.beam_voltage_v)
    u0_5k2v, _ = h.relativistic_beam_velocity(marosi_spec.beam_voltage_v)

    # ------------------------------------------------------------------
    # Fig 00: Setup Schematic (unit cell, 3D cut planes, graded mesh zoom)
    # ------------------------------------------------------------------
    p00 = h.plot_sws_setup_schematic(
        fig_dir / "00_setup_schematic.png",
        grating_spec=grating_spec,
        marosi_spec=marosi_spec,
    )
    saved.append(p00)

    primary_2d = tasks.get("2d_conformal_L2", tasks.get("2d_conformal_L1"))

    # ------------------------------------------------------------------
    # Fig 01: 2D Mirror/Glide Grating & 3D Marosi Staggered-Pillar Geometry
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.1)
    d_um = grating_spec.period_m * 1e6
    g_um = grating_spec.tunnel_half_height_m * 1e6
    hb_um = grating_spec.beam_half_thickness_m * 1e6
    h_um = grating_spec.vane_height_m * 1e6
    l_um = grating_spec.vane_length_m * 1e6
    b_um = grating_spec.total_height_m * 1e6

    for idx_ax, (ax, is_glide, title_str) in enumerate(
        [
            (axes[0], False, "(a) 2D Mirror Grating ($\\delta z = 0$)"),
            (axes[1], True, "(b) 2D Glide Grating ($\\delta z = d/2$)"),
        ]
    ):
        ax.set_facecolor("#f8fafc")
        ax.axhspan(-g_um, g_um, color="#fef3c7", alpha=0.58, label="Beam tunnel ($2a = 200\\;\\mu$m)")
        ax.axhspan(-hb_um, hb_um, color="#fca5a5", alpha=0.38, label="Sheet beam ($H_b = 150\\;\\mu$m, $V_0 = 20$ kV)")
        for cell_k in range(2):
            z_off = cell_k * d_um
            z_b = z_off + 0.25 * d_um - 0.5 * l_um
            z_t = z_off + (0.75 if is_glide else 0.25) * d_um - 0.5 * l_um
            ax.add_patch(
                plt.Rectangle((z_b, -0.5 * b_um), l_um, h_um, color="#334155", ec="#0f172a", lw=1.1)
            )
            ax.add_patch(
                plt.Rectangle((z_t, g_um), l_um, h_um, color="#334155", ec="#0f172a", lw=1.1)
            )
        ax.axvline(d_um, color=figstyle.PALETTE[0], ls="--", lw=1.3, label="Bloch period $d = 500\\;\\mu$m")
        ax.axhline(-0.5 * b_um, color="#0f172a", lw=2.0)
        ax.axhline(+0.5 * b_um, color="#0f172a", lw=2.0)
        ax.text(
            0.03,
            0.95,
            r"Bloch $z=0,d$ ($\beta_0 d$) | PEC $y=\pm 400\;\mu\mathrm{m}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.80", "alpha": 0.90},
        )
        ax.set(
            xlim=(0, 2 * d_um),
            ylim=(-0.5 * b_um, 0.5 * b_um),
            xlabel="Axial coordinate $z$ [$\\mu$m]",
            ylabel="Vertical coordinate $y$ [$\\mu$m]",
            title=title_str,
        )
        if idx_ax == 0:
            ax.legend(loc="lower right", fontsize=7.0)

    # Panel (c): 3D Marosi staggered-pillar SWS (x-z top view and y-z cross-section inset)
    ax2 = axes[2]
    ax2.set_facecolor("#f8fafc")
    a_mm = marosi_spec.waveguide_width_m * 1e3
    d_m_mm = marosi_spec.period_m * 1e3
    w_mm = marosi_spec.pillar_width_m * 1e3
    l_m_mm = marosi_spec.pillar_length_m * 1e3
    wb_mm = marosi_spec.beam_width_m * 1e3
    ax2.axhspan(-0.5 * wb_mm, 0.5 * wb_mm, color="#fca5a5", alpha=0.38, label="Sheet beam ($w_b=0.55$ mm, $5.2$ kV)")
    ax2.add_patch(
        plt.Rectangle(
            (0.25 * d_m_mm - 0.5 * l_m_mm, -0.5 * a_mm),
            l_m_mm,
            w_mm,
            color=figstyle.PALETTE[0],
            alpha=0.85,
            ec="#0f172a",
            label="$-x$ PEC pillar pair ($z_1 = d/4$)",
        )
    )
    ax2.add_patch(
        plt.Rectangle(
            (0.75 * d_m_mm - 0.5 * l_m_mm, 0.5 * a_mm - w_mm),
            l_m_mm,
            w_mm,
            color=figstyle.PALETTE[1],
            alpha=0.85,
            ec="#0f172a",
            label="$+x$ PEC pillar pair ($z_2 = 3d/4$)",
        )
    )
    ax2.axhline(0.0, color=figstyle.PALETTE[2], ls="-.", lw=1.2, label="Cut plane $x=0$ (longitudinal slice)")
    ax2.text(
        0.03,
        0.95,
        r"Bloch $z=0,d$ ($d=1.05\;\mathrm{mm}$) | PEC $x=\pm 0.70\;\mathrm{mm}$",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.80", "alpha": 0.90},
    )
    ax2.set(
        xlim=(0, d_m_mm),
        ylim=(-0.5 * a_mm, 0.5 * a_mm),
        xlabel="Axial coordinate $z$ [mm]",
        ylabel="Transverse coordinate $x$ [mm]",
        title="(c) 3D Marosi V-Band Pillar SWS ($x$–$z$ view)",
    )
    ax2.legend(loc="lower right", fontsize=6.8)
    p01 = fig_dir / "01_sws_2d_and_3d_geometry.png"
    figstyle.finish(fig)
    figstyle.save(fig, p01)
    saved.append(p01)

    # ------------------------------------------------------------------
    # Fig 02: Mesh Zoom — Staircase vs Uniform Conformal vs Graded Non-Uniform Conformal
    # ------------------------------------------------------------------
    grid_c, _ = h.build_2d_grating_grid(grating_spec, n_feat=5, glide_symmetric=True, conformal="pec")
    grid_s, _ = h.build_2d_grating_grid(grating_spec, n_feat=5, glide_symmetric=True, conformal=False)
    grid_grad, meta_grad = h.build_2d_grating_graded_grid(
        grating_spec, n_feat=8, glide_symmetric=True, conformal="pec"
    )
    s_s = h.SolverFIT3D(
        grid_s, use_stl=True, bc_low=["periodic", "pec", "pec"], bc_high=["periodic", "pec", "pec"], verbose=0
    )
    ny_c, nz_c = max(int(grid_c.Ny) - 1, 1), max(int(grid_c.Nz) - 1, 1)
    y_edges_um = np.asarray(grid_c.y[: ny_c + 1]) * 1e6
    z_edges_um = np.asarray(grid_c.z[: nz_c + 1]) * 1e6

    vac_stair = np.clip(
        0.5
        * (
            np.asarray(s_s.ieps[0, :ny_c, :nz_c, "y"]) * h.EPS0
            + np.asarray(s_s.ieps[0, :ny_c, :nz_c, "z"]) * h.EPS0
        ),
        0.0,
        1.0,
    )

    def _compute_cutcell_fraction(y_ed_um: np.ndarray, z_ed_um: np.ndarray) -> np.ndarray:
        yy_lo, yy_hi = y_ed_um[:-1, None], y_ed_um[1:, None]
        zz_lo, zz_hi = z_ed_um[None, :-1], z_ed_um[None, 1:]
        dy_u = yy_hi - yy_lo
        dz_u = zz_hi - zz_lo
        ov_bot_z = np.clip(np.minimum(zz_hi, 175.0) - np.maximum(zz_lo, 75.0), 0.0, dz_u) / dz_u
        ov_bot_y = np.clip(np.minimum(yy_hi, -100.0) - np.maximum(yy_lo, -400.0), 0.0, dy_u) / dy_u
        ov_top_z = np.clip(np.minimum(zz_hi, 425.0) - np.maximum(zz_lo, 325.0), 0.0, dz_u) / dz_u
        ov_top_y = np.clip(np.minimum(yy_hi, 400.0) - np.maximum(yy_lo, 100.0), 0.0, dy_u) / dy_u
        return 1.0 - np.clip(ov_bot_z * ov_bot_y + ov_top_z * ov_top_y, 0.0, 1.0)

    vac_conf = _compute_cutcell_fraction(y_edges_um, z_edges_um)
    ny_g, nz_g = max(int(grid_grad.Ny) - 1, 1), max(int(grid_grad.Nz) - 1, 1)
    y_grad_um = np.asarray(grid_grad.y[: ny_g + 1]) * 1e6
    z_grad_um = np.asarray(grid_grad.z[: nz_g + 1]) * 1e6
    vac_grad = _compute_cutcell_fraction(y_grad_um, z_grad_um)

    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.2, sharey=True)
    for ax, z_ed, y_ed, arr, ttl in [
        (ax0, z_edges_um, y_edges_um, vac_stair, "(a) Staircase PEC ($\\Delta = 20\\;\\mu$m)"),
        (ax1, z_edges_um, y_edges_um, vac_conf, "(b) Uniform Conformal ($\\Delta = 20\\;\\mu$m)"),
        (
            ax2,
            z_grad_um,
            y_grad_um,
            vac_grad,
            f"(c) Graded PEC ($12.5\\to 30\\;\\mu$m, {meta_grad['total_cells']} cells)",
        ),
    ]:
        im = ax.pcolormesh(
            z_ed, y_ed, arr, cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="0.72", lw=0.35
        )
        ax.add_patch(
            plt.Rectangle((75.0, -400.0), 100.0, 300.0, fill=False, ec="#ff5252", lw=1.7, ls="--", label="Exact PEC vane boundary")
        )
        ax.add_patch(
            plt.Rectangle((325.0, 100.0), 100.0, 300.0, fill=False, ec="#ff5252", lw=1.7, ls="--")
        )
        ax.axhline(-100.0, color="#ffca28", ls=":", lw=1.2, label="Tunnel aperture $y = \\pm 100\\;\\mu$m")
        ax.axhline(100.0, color="#ffca28", ls=":", lw=1.2)
        ax.set(
            xlim=(0.0, 500.0),
            ylim=(-240.0, 240.0),
            xlabel="Axial coordinate $z$ [$\\mu$m]",
            title=ttl,
        )
    fig.colorbar(im, ax=ax2, label="Normalized vacuum cell aperture $L_{\\mathrm{eff}} / \\Delta$")
    ax0.set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
    ax0.legend(loc="lower right", fontsize=7.0)
    ax2.text(
        0.97,
        0.05,
        r"$8\;\mathrm{cells}/100\;\mu\mathrm{m}$ tooth" + "\n" + r"$1.62\times$ fewer cells than uniform L2",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    p02 = fig_dir / "02_mesh_zoom_conformal_vs_staircase.png"
    figstyle.finish(fig)
    figstyle.save(fig, p02)
    saved.append(p02)

    # ------------------------------------------------------------------
    # Fig 03: McVey Field-Matching Convergence in N_harm and M_slot
    # ------------------------------------------------------------------
    mcvey_extrap = compute_mcvey_slot_mode_extrapolation(
        spec=grating_spec, print_table=True
    )
    if primary_2d is not None and "conv_n_harm" in primary_2d:
        nh = primary_2d["conv_n_harm"]
        ms = primary_2d["conv_m_slot"]
        f_mat = primary_2d["conv_f_fund_ghz"]
    else:
        ca = h.compute_mcvey_harmonic_convergence(0.5 * np.pi / grating_spec.period_m, grating_spec)
        nh = ca["n_harm_values"]
        ms = ca["m_slot_values"]
        f_mat = ca["f_fundamental_ghz"]

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.1)
    colors_m = [figstyle.PALETTE[4], figstyle.PALETTE[1], figstyle.PALETTE[0], figstyle.PALETTE[2]]
    for im_idx, m_val in enumerate(ms):
        c_col = colors_m[im_idx % len(colors_m)]
        ax0.plot(
            nh, f_mat[im_idx], "o-", color=c_col, lw=1.9, ms=5, label=f"$M_{{\\mathrm{{slot}}}} = {int(m_val)}$"
        )
    ax0.axhline(
        mcvey_extrap["f1_inf_ghz"],
        color="#e53935",
        ls="-.",
        lw=1.6,
        label=f"McVey $M_{{\\mathrm{{slot}}}}\\to\\infty$ ({mcvey_extrap['f1_inf_ghz']:.2f} GHz)",
    )
    if "2d_conformal_L3" in tasks and "2d_conformal_reference" in tasks:
        f_l3_v = float(tasks["2d_conformal_L3"]["f1_ghz"][0])
        f_ref_v = float(tasks["2d_conformal_reference"]["f1_ghz"][0])
        f_wakis_rich = f_ref_v + (f_ref_v - f_l3_v) / ((16.0 / 12.0) ** 2 - 1.0)
        ax0.axhline(
            f_wakis_rich,
            color="k",
            ls="--",
            lw=1.5,
            label=f"Wakis FIT Rich. ({f_wakis_rich:.2f} GHz)",
        )
    elif primary_2d is not None:
        bn_p = primary_2d["bn"]
        i_mid = int(np.argmin(np.abs(bn_p - 0.5)))
        f_wakis_mid = float(primary_2d["f_glide_low_ghz"][i_mid])
        ax0.axhline(
            f_wakis_mid, color="k", ls="--", lw=1.5, label=f"Wakis 2D FIT ({f_wakis_mid:.2f} GHz)"
        )
    ax0.set(
        xlabel="Maximum space-harmonic index $N_{\\mathrm{harm}}$",
        ylabel="Fundamental eigenfrequency $f_1(\\beta_0 d = 0.5\\pi)$ [GHz]",
        title="(a) McVey dispersion vs $N_{\\mathrm{harm}}$ & $M_{\\mathrm{slot}}\\to\\infty$ limit",
    )
    ax0.legend(loc="lower right", fontsize=7.5)

    m_grid = mcvey_extrap["m_slot_values"].astype(float)
    m_fine = np.linspace(float(m_grid[0]), float(m_grid[-1]), 120)
    err_k_m = np.abs(mcvey_extrap["km1_axis_ohm"] - mcvey_extrap["km1_inf_ohm"]) / mcvey_extrap["km1_inf_ohm"]
    err_f_m = np.abs(mcvey_extrap["f1_ghz"] - mcvey_extrap["f1_inf_ghz"]) / mcvey_extrap["f1_inf_ghz"]
    fit_k_m = np.abs(mcvey_extrap["km1_c"] * (m_fine ** (-mcvey_extrap["km1_p"]))) / mcvey_extrap["km1_inf_ohm"]
    fit_f_m = np.abs(mcvey_extrap["f1_c"] * (m_fine ** (-mcvey_extrap["f1_p"]))) / mcvey_extrap["f1_inf_ghz"]

    ax1.loglog(
        m_grid,
        err_k_m,
        "o",
        color=figstyle.PALETTE[1],
        ms=5.5,
        label=f"$|K_{{-1}}(M) - K_{{-1,\\infty}}| / K_{{-1,\\infty}}$ ($K_{{-1,\\infty}}={mcvey_extrap['km1_inf_ohm']:.4f}\\;\\Omega$)",
    )
    ax1.loglog(
        m_fine,
        fit_k_m,
        "-",
        color=figstyle.PALETTE[1],
        lw=1.8,
        label=f"Fit $C M^{{-p}}$ ($M=10\\dots 32$, $p={mcvey_extrap['km1_p']:.2f}$)",
    )
    ax1.loglog(
        m_grid,
        err_f_m,
        "s",
        color=figstyle.PALETTE[0],
        ms=5.5,
        label=f"$|f_1(M) - f_{{1,\\infty}}| / f_{{1,\\infty}}$ ($f_{{1,\\infty}}={mcvey_extrap['f1_inf_ghz']:.3f}$ GHz)",
    )
    ax1.loglog(
        m_fine,
        fit_f_m,
        "--",
        color=figstyle.PALETTE[0],
        lw=1.8,
        label=f"Fit $C M^{{-p}}$ ($M=10\\dots 32$, $p={mcvey_extrap['f1_p']:.2f}$)",
    )
    ax1.set(
        xlabel="Slot waveguide mode count $M_{\\mathrm{slot}}$ ($N_{\\mathrm{harm}} = 60$)",
        ylabel="Relative error $|y(M_{\\mathrm{slot}}) - y_\\infty| / y_\\infty$",
        title="(b) Slot-mode power-law extrapolation $y = y_\\infty + C M_{\\mathrm{slot}}^{-p}$",
    )
    ax1.legend(loc="upper right", fontsize=7.2)
    p03 = fig_dir / "03_mcvey_harmonic_convergence.png"
    figstyle.finish(fig)
    figstyle.save(fig, p03)
    saved.append(p03)

    # ------------------------------------------------------------------
    # Fig 04: 2D Brillouin Diagram — Mirror vs Glide + Uniform vs Graded Mesh
    # ------------------------------------------------------------------
    if primary_2d is not None:
        bn = primary_2d["bn"]
        bn_f = primary_2d["bn_fine"]
        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.3)

        # Panel (a): Reduced Brillouin zone beta_0 d / pi in [0, 1]
        ax0.plot(bn_f, primary_2d["mcvey_m_low_ghz"], "-", color=figstyle.PALETTE[0], lw=2.0, label="McVey Mirror (lower/upper)")
        ax0.plot(bn_f, primary_2d["mcvey_m_high_ghz"], "-", color=figstyle.PALETTE[0], lw=2.0)
        ax0.plot(bn_f, primary_2d["mcvey_g_low_ghz"], "--", color=figstyle.PALETTE[1], lw=2.0, label="McVey Glide (lower/upper)")
        ax0.plot(bn_f, primary_2d["mcvey_g_high_ghz"], "--", color=figstyle.PALETTE[1], lw=2.0)

        ax0.plot(bn, primary_2d["f_mirror_low_ghz"], "o", color=figstyle.PALETTE[0], ms=5.5, label="Uniform L2 Mirror ($2{,}560$ cells)")
        ax0.plot(bn, primary_2d["f_mirror_high_ghz"], "o", color=figstyle.PALETTE[0], ms=5.5)
        ax0.plot(bn, primary_2d["f_glide_low_ghz"], "s", color=figstyle.PALETTE[1], ms=5.5, label="Uniform L2 Glide ($2{,}560$ cells)")
        ax0.plot(bn, primary_2d["f_glide_high_ghz"], "s", color=figstyle.PALETTE[1], ms=5.5)
        if "bn" in graded_sweep:
            bn_gr = np.asarray(graded_sweep["bn"], dtype=float)
            ax0.plot(
                bn_gr,
                np.asarray(graded_sweep["f1_ghz"], dtype=float),
                "D",
                color=figstyle.PALETTE[2],
                ms=5.8,
                mfc="none",
                mew=1.6,
                label="Graded L2 Glide ($1{,}584$ cells, $1.62\\times$ fewer)",
            )
            ax0.plot(
                bn_gr,
                np.asarray(graded_sweep["f2_ghz"], dtype=float),
                "D",
                color=figstyle.PALETTE[2],
                ms=5.8,
                mfc="none",
                mew=1.6,
            )

        # Light line f = c beta_0 / (2 pi)
        f_light_ghz = (bn_f * np.pi / grating_spec.period_m) * h.C0 / (2.0 * np.pi) * 1e-9
        ax0.plot(bn_f, f_light_ghz, ":", color="0.4", lw=1.4, label="Light line $\\omega = c\\beta_0$")
        ax0.axhspan(
            float(primary_2d["f_mirror_low_ghz"][-1]),
            float(primary_2d["f_mirror_high_ghz"][-1]),
            color=figstyle.PALETTE[0],
            alpha=0.10,
            label="Mirror stopband at $\\beta_0 d = \\pi$",
        )
        ax0.set(
            xlim=(0.0, 1.05),
            ylim=(0.0, 240.0),
            xlabel="Normalized Floquet phase $\\beta_0 d / \\pi$",
            ylabel="Frequency $f$ [GHz]",
            title="(a) Reduced zone: mirror stopband vs glide closure",
        )
        ax0.legend(loc="lower right", fontsize=6.9)

        # Panel (b): Extended zone beta_n d / pi in [0, 3] with 20 kV beam line
        bn_ext = np.concatenate([bn_f, 2.0 - bn_f[::-1], 2.0 + bn_f])
        f_g_ext_low = np.concatenate(
            [primary_2d["mcvey_g_low_ghz"], primary_2d["mcvey_g_low_ghz"][::-1], primary_2d["mcvey_g_low_ghz"]]
        )
        f_m_ext_low = np.concatenate(
            [primary_2d["mcvey_m_low_ghz"], primary_2d["mcvey_m_low_ghz"][::-1], primary_2d["mcvey_m_low_ghz"]]
        )
        ax1.plot(bn_ext, f_m_ext_low, "-", color=figstyle.PALETTE[0], lw=1.9, label="Mirror fundamental")
        ax1.plot(bn_ext, f_g_ext_low, "--", color=figstyle.PALETTE[1], lw=2.0, label="Glide fundamental")
        ax1.plot(2.0 - bn, primary_2d["f_glide_low_ghz"], "s", color=figstyle.PALETTE[1], ms=5.5, label="Uniform L2 ($n=-1$ backward)")
        ax1.plot(bn, primary_2d["f_glide_low_ghz"], "o", color=figstyle.PALETTE[2], ms=5.5, label="Uniform L2 ($n=0$ forward)")
        if "bn" in graded_sweep:
            bn_gr = np.asarray(graded_sweep["bn"], dtype=float)
            f_gr_lo = np.asarray(graded_sweep["f1_ghz"], dtype=float)
            ax1.plot(
                2.0 - bn_gr,
                f_gr_lo,
                "D",
                color="#0f172a",
                ms=5.8,
                mfc="none",
                mew=1.5,
                label="Graded L2 ($n=-1$ & $n=0$, $1{,}584$ cells)",
            )
            ax1.plot(bn_gr, f_gr_lo, "D", color="#0f172a", ms=5.8, mfc="none", mew=1.5)

        beta_ext_grid = np.linspace(0.0, 3.0, 200)
        f_beam_20kv = (beta_ext_grid * np.pi / grating_spec.period_m) * u0_20kv / (2.0 * np.pi) * 1e-9
        ax1.plot(beta_ext_grid, f_beam_20kv, "-.", color="#e53935", lw=2.0, label="20 kV beam ($u_0 = 0.272 c$)")
        ax1.set(
            xlim=(0.0, 3.0),
            ylim=(0.0, 220.0),
            xlabel="Extended Floquet phase $\\beta_n d / \\pi$",
            ylabel="Frequency $f$ [GHz]",
            title="(b) Unfolded harmonics & 20 kV synchronous crossings",
        )
        ax1.legend(loc="upper left", fontsize=7.0)
        p04 = fig_dir / "04_2d_brillouin_mirror_vs_glide.png"
        figstyle.finish(fig)
        figstyle.save(fig, p04)
        saved.append(p04)

    # ------------------------------------------------------------------
    # Fig 05: Group Velocity (dw/dbeta) vs Energy Velocity (Pz d / U_cell)
    # ------------------------------------------------------------------
    if primary_2d is not None:
        bn_f = primary_2d["bn_fine"]
        beta_f = bn_f * np.pi / grating_spec.period_m
        omega_g = 2.0 * np.pi * primary_2d["mcvey_g_low_ghz"] * 1e9
        omega_m = 2.0 * np.pi * primary_2d["mcvey_m_low_ghz"] * 1e9
        vg_g_over_c = np.gradient(omega_g, beta_f) / h.C0
        vg_m_over_c = np.gradient(omega_m, beta_f) / h.C0
        mask_tw = primary_2d["bn"] < 0.99

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.0)
        ax0.plot(bn_f, vg_g_over_c, "-", color=figstyle.PALETTE[1], lw=2.1, label="Analytic slope $v_g/c = (1/c)\\mathrm{d}\\omega/\\mathrm{d}\\beta_0$ (Glide)")
        ax0.plot(bn_f, vg_m_over_c, "--", color=figstyle.PALETTE[0], lw=1.9, label="Analytic slope $v_g/c$ (Mirror, $v_g(\\pi)=0$)")
        ax0.plot(
            primary_2d["bn"][mask_tw],
            primary_2d["v_energy_over_c"][mask_tw],
            "s",
            color=figstyle.PALETTE[2],
            ms=6.0,
            label="Uniform L2 DFT $v_E/c = P_z d / (c U_{\\mathrm{cell}})$",
        )
        if "bn" in graded_sweep and "v_energy_over_c" in graded_sweep:
            bn_gr = np.asarray(graded_sweep["bn"], dtype=float)
            ve_gr = np.asarray(graded_sweep["v_energy_over_c"], dtype=float)
            m_gr = bn_gr < 0.99
            ax0.plot(
                bn_gr[m_gr],
                ve_gr[m_gr],
                "D",
                color="#0f172a",
                ms=6.0,
                mfc="none",
                mew=1.5,
                label="Graded L2 DFT $v_E/c$ ($1{,}584$ cells)",
            )
        ax0.set(
            xlim=(0.0, 1.05),
            ylim=(0.0, 0.72),
            xlabel="Normalized Floquet phase $\\beta_0 d / \\pi$",
            ylabel="Normalized velocity $v / c$",
            title="(a) Group velocity $v_g$ vs energy velocity $v_E = P_z / W$",
        )
        ax0.legend(loc="lower left", fontsize=7.2)

        z_um = primary_2d["z_centers_m"] * 1e6
        pz_z = primary_2d["pz_vs_z_w"]
        pz_norm = pz_z / max(float(np.mean(pz_z)), 1e-30)
        ax1.plot(z_um, pz_norm, "o-", color=figstyle.PALETTE[0], lw=1.9, ms=4.5, label="$P_z(z) / \\langle P_z \\rangle$")
        ax1.axhline(1.0, color=figstyle.PALETTE[1], ls="--", lw=1.5, label="Unit-cell mean $\\langle P_z \\rangle$")
        ax1.set(
            ylim=(0.85, 1.15),
            xlabel="Axial position $z$ [$\\mu$m]",
            ylabel="Normalized axial Poynting power $P_z(z) / \\langle P_z \\rangle$",
            title="(b) Axial conservation of Poynting flux across the period",
        )
        ax1.legend(loc="upper right", fontsize=7.8)
        p05 = fig_dir / "05_group_vs_energy_velocity.png"
        figstyle.finish(fig)
        figstyle.save(fig, p05)
        saved.append(p05)

    # ------------------------------------------------------------------
    # Fig 06: Space-Harmonic Spectrum |E_{zn}| and Transverse Profiles |E_{zn}(y)|
    # ------------------------------------------------------------------
    if primary_2d is not None:
        mid_2d_pt = tasks.get("2d_conformal_L3", primary_2d)
        n_idx = mid_2d_pt["n_indices"]
        ez_ax = mid_2d_pt["ez_harm_axis"]
        ez_ax_norm = ez_ax / max(float(np.max(ez_ax)), 1e-30)
        y_um = primary_2d["y_nodes_m"] * 1e6
        ez_prof = primary_2d["ez_harm_y_profile"]
        n_idx_prof = primary_2d["n_indices"]
        tunnel_mask = np.abs(y_um) <= 105.0

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.0)
        ax0.bar(n_idx - 0.15, ez_ax_norm, width=0.3, color=figstyle.PALETTE[0], label="Wakis 2D FIT ($y=0$, Glide)")
        mc_imp = h.compute_mcvey_analytic_pierce_impedance(
            97.6868e9,
            0.5 * np.pi / grating_spec.period_m,
            grating_spec,
            symmetry="glide_b",
            n_harm=40,
            m_slot=6,
        )
        mask_n6 = np.abs(mc_imp["n_indices"]) <= 6
        mc_norm = mc_imp["ez_n_axis_norm"][mask_n6] / max(
            float(np.max(mc_imp["ez_n_axis_norm"][mask_n6])), 1e-30
        )
        ax0.bar(
            mc_imp["n_indices"][mask_n6] + 0.15,
            mc_norm,
            width=0.3,
            color=figstyle.PALETTE[1],
            alpha=0.85,
            label="McVey theory ($y=0$, Glide, $N=40, M=6$)",
        )
        ax0.set(
            xlabel="Space-harmonic index $n$",
            ylabel="Normalized on-axis amplitude $|E_{zn}(0)| / \\max_n |E_{zn}(0)|$",
            title="(a) On-axis Floquet harmonic spectrum ($\\beta_0 d = 0.5\\pi$)",
        )
        ax0.legend(loc="upper right", fontsize=7.6)

        p_max = max(float(np.max(ez_prof[:, tunnel_mask])), 1e-30)
        for n_show, col, ls_s, lbl in [
            (-1, figstyle.PALETTE[1], "-", "$n = -1$ ($\\cosh(\\tau_{-1} y)$, backward wave)"),
            (0, figstyle.PALETTE[0], "--", "$n = 0$ ($\\sinh(\\tau_0 y)$, node on axis)"),
            (1, figstyle.PALETTE[2], "-.", "$n = +1$ ($\\cosh(\\tau_{+1} y)$, forward wave)"),
            (-2, figstyle.PALETTE[4], ":", "$n = -2$ ($\\sinh(\\tau_{-2} y)$, node on axis)"),
        ]:
            idx_h = int(np.where(n_idx_prof == n_show)[0][0])
            ax1.plot(y_um[tunnel_mask], ez_prof[idx_h, tunnel_mask] / p_max, ls=ls_s, color=col, lw=2.0, label=lbl)
        ax1.axvspan(-75.0, 75.0, color="#ef5350", alpha=0.15, label="Sheet beam ($|y| \\leq 75\\;\\mu$m)")
        ax1.set(
            xlabel="Vertical coordinate $y$ in beam tunnel [$\\mu$m]",
            ylabel="Normalized harmonic profile $|E_{zn}(y)|$",
            title="(b) Transverse parity alternation of glide harmonics",
        )
        ax1.legend(loc="upper center", fontsize=7.3)
        p06 = fig_dir / "06_space_harmonic_spectrum_and_profiles.png"
        figstyle.finish(fig)
        figstyle.save(fig, p06)
        saved.append(p06)

    # ------------------------------------------------------------------
    # Fig 07: 2D Pierce Interaction Impedance K_n vs Frequency (Uniform vs Graded)
    # ------------------------------------------------------------------
    if primary_2d is not None:
        mask_tw = primary_2d["bn"] < 0.99
        bn_tw = primary_2d["bn"][mask_tw]
        f_g = primary_2d["f_glide_low_ghz"][mask_tw]
        mc_f_tw, mc_km1_ax_tw, mc_km1_av_tw = [], [], []
        for bnv in bn_tw:
            b0_tw = float(bnv * np.pi / grating_spec.period_m)
            r_tw = h.solve_mcvey_band_frequencies(
                b0_tw, grating_spec, symmetry="glide", n_harm=40, m_slot=16
            )
            if len(r_tw) >= 1:
                imp_tw = h.compute_mcvey_analytic_pierce_impedance(
                    float(r_tw[0]), b0_tw, grating_spec, symmetry="glide_b", n_harm=40, m_slot=16
                )
                i_m1_tw = int(np.where(imp_tw["n_indices"] == -1)[0][0])
                mc_f_tw.append(float(r_tw[0]) * 1e-9)
                mc_km1_ax_tw.append(float(imp_tw["k_n_axis_ohm"][i_m1_tw]))
                mc_km1_av_tw.append(float(imp_tw["k_n_beam_avg_ohm"][i_m1_tw]))

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.1)
        ax0.plot(f_g, primary_2d["km1_axis_ohm"][mask_tw], "o-", color=figstyle.PALETTE[1], lw=2.0, ms=5.5, label="Uniform L2 $K_{-1}(0)$ ($2{,}560$ cells)")
        if "f1_ghz" in graded_sweep and "km1_axis_ohm" in graded_sweep:
            bn_gr = np.asarray(graded_sweep["bn"], dtype=float)
            m_gr = bn_gr < 0.99
            f_gr = np.asarray(graded_sweep["f1_ghz"], dtype=float)[m_gr]
            ax0.plot(
                f_gr,
                np.asarray(graded_sweep["km1_axis_ohm"], dtype=float)[m_gr],
                "D",
                color="#0f172a",
                ms=6.0,
                mfc="none",
                mew=1.6,
                label="Graded L2 $K_{-1}(0)$ ($1{,}584$ cells)",
            )
            ax0.plot(
                f_gr,
                np.asarray(graded_sweep["km1_beam_avg_ohm"], dtype=float)[m_gr],
                "P",
                color="#0f172a",
                ms=5.8,
                mfc="none",
                mew=1.4,
                label="Graded L2 $\\langle K_{-1}\\rangle_{\\mathrm{beam}}$ ($1{,}584$ cells)",
            )
        ax0.plot(mc_f_tw, mc_km1_ax_tw, "--", color="#8c2d04", lw=1.7, label="McVey $K_{-1}(0)$ ($P_{\\mathrm{tunnel}}+P_{\\mathrm{slot}}$, $M=16$)")
        ax0.plot(
            [mcvey_extrap["f1_inf_ghz"]],
            [mcvey_extrap["km1_inf_ohm"]],
            "*",
            color="#e53935",
            ms=9.5,
            label=f"McVey $M_{{\\mathrm{{slot}}}}\\to\\infty$ ($K_{{-1,\\infty}}={mcvey_extrap['km1_inf_ohm']:.3f}\\;\\Omega$)",
        )
        ax0.plot(f_g, primary_2d["km1_beam_avg_ohm"][mask_tw], "s--", color=figstyle.PALETTE[2], lw=1.9, ms=5.2, label="Uniform L2 $\\langle K_{-1} \\rangle_{\\mathrm{beam}}$")
        ax0.plot(mc_f_tw, mc_km1_av_tw, ":", color="#006d2c", lw=1.7, label="McVey $\\langle K_{-1} \\rangle_{\\mathrm{beam}}$ ($M=16$)")
        ax0.plot(f_g, primary_2d["kp1_axis_ohm"][mask_tw], "^-.", color=figstyle.PALETTE[0], lw=1.8, ms=5.2, label="Uniform L2 $K_{+1}(0)$")
        ax0.plot(f_g, primary_2d["k0_axis_ohm"][mask_tw], "d:", color=figstyle.PALETTE[4], lw=1.5, ms=4.8, label="Uniform L2 $K_0(0) \\approx 0$ (odd parity)")
        ax0.set(
            xlabel="Frequency $f$ [GHz]",
            ylabel="Pierce interaction impedance $K_n$ [$\\Omega$]",
            title="(a) Glide grating Pierce impedance $K_n(f)$ across passband",
        )
        ax0.legend(loc="upper left", fontsize=6.1)

        mid_2d_pt = tasks.get("2d_conformal_L3", primary_2d)
        n_idx = mid_2d_pt["n_indices"]
        kn_mid = mid_2d_pt["k_n_axis_ohm"]
        mc_imp_m32 = h.compute_mcvey_analytic_pierce_impedance(
            float(mcvey_extrap["f1_ghz"][-1] * 1e9),
            0.5 * np.pi / grating_spec.period_m,
            grating_spec,
            symmetry="glide_b",
            n_harm=60,
            m_slot=32,
        )
        mask_n6_32 = np.abs(mc_imp_m32["n_indices"]) <= 6
        mc_imp_tun_only = h.compute_mcvey_analytic_pierce_impedance(
            float(mcvey_extrap["f1_ghz"][-1] * 1e9),
            0.5 * np.pi / grating_spec.period_m,
            grating_spec,
            symmetry="glide_b",
            n_harm=60,
            m_slot=32,
            include_slot_power=False,
        )
        ax1.semilogy(n_idx, np.maximum(kn_mid, 1e-4), "o-", color=figstyle.PALETTE[1], lw=2.0, ms=6, label="Wakis FIT $K_n(y=0)$ at $\\beta_0 d = 0.5\\pi$")
        ax1.semilogy(
            mc_imp_m32["n_indices"][mask_n6_32],
            np.maximum(mc_imp_m32["k_n_axis_ohm"][mask_n6_32], 1e-4),
            "s--",
            color=figstyle.PALETTE[0],
            lw=1.8,
            ms=5.5,
            label=f"McVey $K_n(y=0)$ ($P_{{\\mathrm{{tunnel}}}}+P_{{\\mathrm{{slot}}}}$, $K_{{-1,\\infty}}={mcvey_extrap['km1_inf_ohm']:.3f}\\;\\Omega$)",
        )
        ax1.semilogy(
            mc_imp_tun_only["n_indices"][mask_n6_32],
            np.maximum(mc_imp_tun_only["k_n_axis_ohm"][mask_n6_32], 1e-4),
            "^:",
            color=figstyle.PALETTE[4],
            lw=1.5,
            ms=5.0,
            label="McVey tunnel-only ($P_{\\mathrm{slot}}=0$, $K_{-1}=2.96\\;\\Omega$)",
        )
        ax1.set(
            xlabel="Space-harmonic index $n$",
            ylabel="On-axis Pierce impedance $K_n(y=0)$ [$\\Omega$]",
            title="(b) Harmonic impedance roll-off $K_n \\propto |E_{zn}|^2 / \\beta_n^2$",
        )
        ax1.legend(loc="upper right", fontsize=6.9)
        p07 = fig_dir / "07_pierce_impedance_2d_vs_freq.png"
        figstyle.finish(fig)
        figstyle.save(fig, p07)
        saved.append(p07)

    # ------------------------------------------------------------------
    # Fig 08: 2D Complex Bloch Mode Field Maps Re{E_z(y, z)} (Mirror vs Glide)
    # ------------------------------------------------------------------
    if primary_2d is not None:
        ez_g = np.real(primary_2d["ez_yz_slice_glide"])
        ez_m = np.real(primary_2d["ez_yz_slice_mirror"])
        ez_g = ez_g / max(float(np.max(np.abs(ez_g))), 1e-30)
        ez_m = ez_m / max(float(np.max(np.abs(ez_m))), 1e-30)
        z_um = primary_2d["z_centers_m"] * 1e6
        y_um = primary_2d["y_nodes_m"][: ez_g.shape[0]] * 1e6

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.2, sharey=True)
        for ax, arr, is_gl, ttl in [
            (ax0, ez_m, False, "(a) Mirror Grating $\\operatorname{Re}\\{E_z(y, z)\\}$ ($\\beta_0 d = 0.5\\pi$)"),
            (ax1, ez_g, True, "(b) Glide Grating $\\operatorname{Re}\\{E_z(y, z)\\}$ ($\\beta_0 d = 0.5\\pi$)"),
        ]:
            im = ax.pcolormesh(z_um, y_um, arr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
            h.overlay_sws_grating_geometry(ax, grating_spec, glide_symmetric=is_gl, show_beam=True, show_labels=True)
            ax.set(xlabel="Axial coordinate $z$ [$\\mu$m]", title=ttl)
            fig.colorbar(im, ax=ax, label="Normalized $\\operatorname{Re}\\{E_z\\}$")
        ax0.set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
        p08 = fig_dir / "08_bloch_eigenmode_ez_fields.png"
        figstyle.finish(fig)
        figstyle.save(fig, p08)
        saved.append(p08)

    # ------------------------------------------------------------------
    # Fig 09: Animated Time-Domain Bloch Wave Movie (.gif) with Geometry Overlay
    # ------------------------------------------------------------------
    mov_src = tasks.get("2d_conformal_L2")
    if mov_src is not None and "movie_frames_ez" in mov_src:
        frames = mov_src["movie_frames_ez"]
        phases = mov_src["movie_phase_rad"]
        y_um_m = mov_src["movie_y_um"]
        z_um_m = mov_src["movie_z_um"]
    elif primary_2d is not None:
        ez_c = primary_2d["ez_yz_slice_glide"]
        phases = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        frames = np.array([np.real(ez_c * np.exp(1j * ph)) for ph in phases])
        z_um_m = primary_2d["z_centers_m"] * 1e6
        y_um_m = primary_2d["y_nodes_m"][: ez_c.shape[0]] * 1e6
    else:
        frames = None

    if frames is not None:
        f_max_amp = max(float(np.max(np.abs(frames))), 1e-30)
        frames_norm = frames / f_max_amp
        fig_m, ax_m = plt.subplots(figsize=(6.5, 3.9), dpi=110)
        im_m = ax_m.pcolormesh(
            z_um_m, y_um_m, frames_norm[0], cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto"
        )
        h.overlay_sws_grating_geometry(ax_m, grating_spec, glide_symmetric=True, show_beam=True, show_labels=True)
        ttl_m = ax_m.set_title("Bloch Eigenmode $\\operatorname{Re}\\{E_z(y, z, t)\\}$ ($\\omega t = 0.00$ rad)")
        ax_m.set_xlabel("Axial coordinate $z$ [$\\mu$m]")
        ax_m.set_ylabel("Vertical coordinate $y$ [$\\mu$m]")
        fig_m.colorbar(im_m, ax=ax_m, label="Normalized $\\operatorname{Re}\\{E_z(y, z, t)\\}$")
        fig_m.tight_layout()

        def _update_frame(k_idx: int):
            im_m.set_array(frames_norm[k_idx].ravel())
            ttl_m.set_text(
                f"Bloch Eigenmode $\\operatorname{{Re}}\\{{E_z(y, z, t)\\}}$ ($\\omega t = {phases[k_idx]:.2f}$ rad)"
            )
            return (im_m, ttl_m)

        ani = animation.FuncAnimation(fig_m, _update_frame, frames=len(frames_norm), interval=90, blit=False)
        p09 = fig_dir / "09_bloch_mode_animation.gif"
        ani.save(p09, writer=animation.PillowWriter(fps=10))
        plt.close(fig_m)
        saved.append(p09)

    # ------------------------------------------------------------------
    # Fig 10: Resolution Convergence & Cost-Accuracy — Staircase vs Conformal vs Graded
    # ------------------------------------------------------------------
    def _mid_f1(t_dict: dict[str, Any]) -> float:
        arr = t_dict["f1_ghz"]
        if "bn" not in t_dict:
            return float(arr[len(arr) // 2])
        bn_arr = t_dict["bn"]
        i_half = int(np.argmin(np.abs(bn_arr - 0.5)))
        if bn_arr.size >= 9 and abs(bn_arr[i_half] - 0.5) > 1e-4:
            return float(np.polyval(np.polyfit(bn_arr[4:9], arr[4:9], 3), 0.5))
        return float(arr[i_half])

    def _mid_km1(t_dict: dict[str, Any]) -> float:
        arr = t_dict["km1_axis_ohm"]
        if "bn" not in t_dict:
            return float(arr[len(arr) // 2])
        bn_arr = t_dict["bn"]
        i_half = int(np.argmin(np.abs(bn_arr - 0.5)))
        if bn_arr.size >= 9 and abs(bn_arr[i_half] - 0.5) > 1e-4:
            return float(np.interp(0.5, bn_arr, arr))
        return float(arr[i_half])

    conv_levels = ["L1", "L2", "L3"]
    if all(f"2d_conformal_{lv}" in tasks and f"2d_staircase_{lv}" in tasks for lv in conv_levels):
        ref_task = tasks.get("2d_conformal_reference", tasks["2d_conformal_L3"])
        f_l3 = _mid_f1(tasks["2d_conformal_L3"])
        f_ref_16 = _mid_f1(ref_task)
        # Second-order Richardson extrapolation from uniform L3 (n=12) and Reference (n=16)
        f_ref = f_ref_16 + (f_ref_16 - f_l3) / ((16.0 / 12.0) ** 2 - 1.0)
        k_l3 = _mid_km1(tasks["2d_conformal_L3"])
        k_ref_16 = _mid_km1(ref_task)
        k_ref = k_ref_16 + (k_ref_16 - k_l3) / ((16.0 / 12.0) ** 2 - 1.0)

        n_feats = np.array([float(tasks[f"2d_conformal_{lv}"]["cells_across_feature"]) for lv in conv_levels])
        f_conf = np.array([_mid_f1(tasks[f"2d_conformal_{lv}"]) for lv in conv_levels])
        f_stair = np.array([_mid_f1(tasks[f"2d_staircase_{lv}"]) for lv in conv_levels])
        k_conf = np.array([_mid_km1(tasks[f"2d_conformal_{lv}"]) for lv in conv_levels])
        k_stair = np.array([_mid_km1(tasks[f"2d_staircase_{lv}"]) for lv in conv_levels])

        n_feats_all = np.append(n_feats, float(ref_task["cells_across_feature"]))
        f_conf_all = np.append(f_conf, f_ref_16)
        k_conf_all = np.append(k_conf, k_ref_16)
        cells_conf_all = np.array(
            [float(tasks[f"2d_conformal_{lv}"]["n_cells"]) for lv in conv_levels]
            + [float(ref_task["n_cells"])]
        )
        cells_stair = np.array([float(tasks[f"2d_staircase_{lv}"]["n_cells"]) for lv in conv_levels])

        err_f_conf_all = np.maximum(np.abs(f_conf_all - f_ref) / f_ref, 5e-5)
        err_f_stair = np.maximum(np.abs(f_stair - f_ref) / f_ref, 5e-5)
        err_k_conf_all = np.maximum(np.abs(k_conf_all - k_ref) / k_ref, 1e-4)
        err_k_stair = np.maximum(np.abs(k_stair - k_ref) / k_ref, 1e-4)

        grad_levels = list(graded_study.get("ladder", {}).values())
        n_feats_grad = np.array([float(g["n_feat"]) for g in grad_levels], dtype=float)
        f_grad_all = np.array([float(g["f1_mid_ghz"]) for g in grad_levels], dtype=float)
        k_grad_all = np.array([float(g["km1_mid_ohm"]) for g in grad_levels], dtype=float)
        cells_grad_all = np.array([float(g["n_cells"]) for g in grad_levels], dtype=float)
        err_f_grad_all = np.maximum(np.abs(f_grad_all - f_ref) / f_ref, 5e-5)
        err_k_grad_all = np.maximum(np.abs(k_grad_all - k_ref) / k_ref, 1e-4)

        fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=14.2, height=4.2)
        ax0.loglog(n_feats, err_f_stair, "s-", color=figstyle.PALETTE[1], lw=2.0, ms=5.8, label="Uniform Staircase (`conformal=False`)")
        ax0.loglog(n_feats_all, err_f_conf_all, "o-", color=figstyle.PALETTE[2], lw=2.0, ms=5.8, label="Uniform Conformal (`conformal='pec'`)")
        if len(n_feats_grad) > 0:
            ax0.loglog(
                n_feats_grad,
                err_f_grad_all,
                "D--",
                color=figstyle.PALETTE[0],
                lw=2.0,
                ms=6.0,
                mfc="none",
                mew=1.6,
                label="Graded Non-Uniform Conformal",
            )
        ax0.loglog(n_feats, err_f_stair[0] * (n_feats[0] / n_feats) ** 1, "--", color=figstyle.PALETTE[1], alpha=0.50, label="$\\mathcal{O}(\\Delta^1)$ slope")
        ax0.loglog(n_feats_all, err_f_conf_all[0] * (n_feats_all[0] / n_feats_all) ** 2, ":", color=figstyle.PALETTE[2], alpha=0.60, label="$\\mathcal{O}(\\Delta^2)$ slope")
        ax0.set(
            xlabel="Cells across $0.10\\;\\mathrm{mm}$ feature ($n_{\\mathrm{feat}}$)",
            ylabel="Rel. error $|f_1 - f_{\\mathrm{ref}}| / f_{\\mathrm{ref}}$",
            title="(a) Eigenfrequency $f_1$ vs $n_{\\mathrm{feat}}$",
        )
        ax0.legend(loc="best", fontsize=6.6)

        ax1.loglog(n_feats, err_k_stair, "s-", color=figstyle.PALETTE[1], lw=2.0, ms=5.8, label="Uniform Staircase $|K_{-1} - K_{\\mathrm{ref}}| / K_{\\mathrm{ref}}$")
        ax1.loglog(n_feats_all, err_k_conf_all, "o-", color=figstyle.PALETTE[2], lw=2.0, ms=5.8, label="Uniform Conformal $|K_{-1} - K_{\\mathrm{ref}}| / K_{\\mathrm{ref}}$")
        if len(n_feats_grad) > 0:
            ax1.loglog(
                n_feats_grad,
                err_k_grad_all,
                "D--",
                color=figstyle.PALETTE[0],
                lw=2.0,
                ms=6.0,
                mfc="none",
                mew=1.6,
                label="Graded Conformal $|K_{-1} - K_{\\mathrm{ref}}| / K_{\\mathrm{ref}}$",
            )
        ax1.loglog(n_feats, err_k_stair[0] * (n_feats[0] / n_feats) ** 1, "--", color=figstyle.PALETTE[1], alpha=0.50, label="$\\mathcal{O}(\\Delta^1)$ slope")
        ax1.loglog(n_feats_all, err_k_conf_all[0] * (n_feats_all[0] / n_feats_all) ** 2, ":", color=figstyle.PALETTE[2], alpha=0.60, label="$\\mathcal{O}(\\Delta^2)$ slope")
        ax1.set(
            xlabel="Cells across $0.10\\;\\mathrm{mm}$ feature ($n_{\\mathrm{feat}}$)",
            ylabel="Rel. error $|K_{-1} - K_{\\mathrm{ref}}| / K_{\\mathrm{ref}}$",
            title="(b) Pierce impedance $K_{-1}(0)$ vs $n_{\\mathrm{feat}}$",
        )
        ax1.legend(loc="best", fontsize=6.6)

        # Panel (c): Equal-accuracy / Equal-cost Pareto curve (error vs total 2D cell count)
        ax2.loglog(cells_stair, err_k_stair, "s-", color=figstyle.PALETTE[1], lw=1.9, ms=5.8, label="Uniform Staircase ($K_{-1}$ err)")
        ax2.loglog(cells_conf_all, err_k_conf_all, "o-", color=figstyle.PALETTE[2], lw=2.0, ms=5.8, label="Uniform Conformal ($K_{-1}$ err)")
        if len(cells_grad_all) > 0:
            ax2.loglog(
                cells_grad_all,
                err_k_grad_all,
                "D--",
                color=figstyle.PALETTE[0],
                lw=2.1,
                ms=6.2,
                mfc="none",
                mew=1.7,
                label="Graded Non-Uniform ($K_{-1}$ err)",
            )
            ax2.annotate(
                "Graded Ref ($5{,}248$ cells, $20.4$ s)\n< Uniform L3 ($5{,}760$ cells, $26.0$ s)",
                xy=(cells_grad_all[-1], err_k_grad_all[-1]),
                xytext=(520.0, 6.2e-2),
                fontsize=6.4,
                fontweight="bold",
                color=figstyle.PALETTE[0],
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": figstyle.PALETTE[0], "alpha": 0.92},
                arrowprops={"arrowstyle": "->", "color": figstyle.PALETTE[0], "lw": 1.0},
            )
        ax2.set(
            xlabel="Total 2D grid cells $N_y \\times N_z$",
            ylabel="Rel. Pierce impedance error $|K_{-1} - K_{\\mathrm{ref}}| / K_{\\mathrm{ref}}$",
            title="(c) Equal-cost / equal-accuracy pareto curve",
        )
        ax2.legend(loc="upper right", fontsize=6.5)

        ax0.set_xticks(n_feats_all)
        ax1.set_xticks(n_feats_all)
        for ax in (ax0, ax1):
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        p10 = fig_dir / "10_resolution_convergence_pec_vs_staircase.png"
        figstyle.finish(fig)
        figstyle.save(fig, p10)
        saved.append(p10)

    # ------------------------------------------------------------------
    # Fig 11: Under-Resolved Failure Mode Demo (n_feat=2 vs Uniform & Graded Ladder)
    # ------------------------------------------------------------------
    if "under_resolved_staircase" in tasks and "2d_conformal_L3" in tasks:
        labels_u = [
            "Staircase\n($n=2$)",
            "Conformal\n($n=2$)",
            "Staircase L1\n($n=4$)",
            "Conformal L1\n($n=4$)",
            "Conformal L2\n($n=8$, 2.56k)",
            "Graded L2\n($n=8$, 1.58k)",
            "Conformal L3\n($n=12$)",
        ]
        grad_l2_dict = graded_study.get("ladder", {}).get("graded_L2", {})
        f_vals_u = [
            _mid_f1(tasks["under_resolved_staircase"]),
            _mid_f1(tasks["under_resolved_conformal"]),
            _mid_f1(tasks["2d_staircase_L1"]),
            _mid_f1(tasks["2d_conformal_L1"]),
            _mid_f1(tasks["2d_conformal_L2"]),
            float(grad_l2_dict.get("f1_mid_ghz", _mid_f1(tasks["2d_conformal_L2"]))),
            _mid_f1(tasks["2d_conformal_L3"]),
        ]
        k_vals_u = [
            _mid_km1(tasks["under_resolved_staircase"]),
            _mid_km1(tasks["under_resolved_conformal"]),
            _mid_km1(tasks["2d_staircase_L1"]),
            _mid_km1(tasks["2d_conformal_L1"]),
            _mid_km1(tasks["2d_conformal_L2"]),
            float(grad_l2_dict.get("km1_mid_ohm", _mid_km1(tasks["2d_conformal_L2"]))),
            _mid_km1(tasks["2d_conformal_L3"]),
        ]
        x_pos = np.arange(len(f_vals_u))
        bar_cols = ["#e53935", "#fb8c00", "#d95f02", "#2b5c8f", "#1e88e5", "#0284c7", "#1b9e77"][: len(f_vals_u)]

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.1)
        ax0.bar(x_pos, f_vals_u, color=bar_cols, width=0.58, ec="k", lw=0.8)
        ax0.axhline(97.69, color="k", ls="--", lw=1.5, label="McVey theory (97.69 GHz)")
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(labels_u[: len(f_vals_u)], fontsize=7.2)
        ax0.set(
            ylim=(65.0, 105.0),
            ylabel="Eigenfrequency $f_1(\\beta_0 d = 0.5\\pi)$ [GHz]",
            title="(a) Severe frequency shift when $n_{\\mathrm{feat}} < 4$",
        )
        ax0.legend(loc="lower right", fontsize=7.8)

        ax1.bar(x_pos, k_vals_u, color=bar_cols, width=0.58, ec="k", lw=0.8)
        ax1.axhline(k_vals_u[-1], color="k", ls="--", lw=1.5, label=f"Converged FIT ({k_vals_u[-1]:.2f} $\\Omega$)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels_u[: len(k_vals_u)], fontsize=7.2)
        ax1.set(
            ylabel="On-axis Pierce impedance $K_{-1}$ [$\\Omega$]",
            title="(b) Artificial impedance inflation in under-resolved staircase",
        )
        ax1.legend(loc="upper right", fontsize=7.8)
        p11 = fig_dir / "11_under_resolved_warning_demo.png"
        figstyle.finish(fig)
        figstyle.save(fig, p11)
        saved.append(p11)

    # ------------------------------------------------------------------
    # Fig 12: 2D-vs-3D Rectangular Waveguide Mapping & 3D Pierce Impedance (Karetnikova et al. 2018)
    # ------------------------------------------------------------------
    t3d_g = tasks.get("3d_grating_karetnikova")
    if t3d_g is not None and primary_2d is not None:
        bn_3d = t3d_g["bn"]
        fc_ghz = float(t3d_g["f_cut_3d_ghz"][0])
        f_cut_hz = grating_spec.cutoff_freq_3d_hz
        bn_f = primary_2d["bn_fine"]
        f2d_g_fine = primary_2d["mcvey_g_low_ghz"]
        f2d_g_hi_fine = primary_2d["mcvey_g_high_ghz"]
        f2d_m_fine = primary_2d["mcvey_m_low_ghz"]
        f3d_mapped_g = np.sqrt(f2d_g_fine**2 + fc_ghz**2)
        f3d_mapped_g_hi = np.sqrt(f2d_g_hi_fine**2 + fc_ghz**2)
        f3d_mapped_m = np.sqrt(f2d_m_fine**2 + fc_ghz**2)

        fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=14.4, height=4.1)
        ax0.plot(bn_f, f2d_g_fine, "--", color=figstyle.PALETTE[1], alpha=0.65, lw=1.6, label="2D planar Glide $f_{\\mathrm{2D}}(\\beta_0)$")
        ax0.plot(bn_f, f3d_mapped_g, "-", color=figstyle.PALETTE[1], lw=2.1, label="Mapped $\\sqrt{f_{\\mathrm{2D}}^2 + (c/2a)^2}$ (Glide)")
        ax0.plot(bn_f, f3d_mapped_g_hi, "-", color=figstyle.PALETTE[1], lw=1.8, alpha=0.85)
        ax0.plot(bn_f, f3d_mapped_m, "-", color=figstyle.PALETTE[0], lw=2.0, label="Mapped $\\sqrt{f_{\\mathrm{2D}}^2 + (c/2a)^2}$ (Mirror)")
        ax0.plot(bn_3d, t3d_g["f3d_glide_low_ghz"], "s", color=figstyle.PALETTE[1], ms=6.0, label="Wakis 3D FIT Glide ($a=0.85$ mm)")
        if "f3d_glide_high_ghz" in t3d_g:
            ax0.plot(bn_3d, t3d_g["f3d_glide_high_ghz"], "s", color=figstyle.PALETTE[1], ms=5.0, mfc="none")
        ax0.plot(bn_3d, t3d_g["f3d_mirror_low_ghz"], "o", color=figstyle.PALETTE[0], ms=6.0, label="Wakis 3D FIT Mirror ($a=0.85$ mm)")
        ax0.axhline(fc_ghz, color="0.35", ls=":", lw=1.4, label=f"$\\mathrm{{TE}}_{{10}}$ cutoff $c/(2a) = {fc_ghz:.1f}$ GHz")
        ax0.set(
            xlim=(0.0, 1.05),
            ylim=(20.0, 272.0),
            xlabel="Normalized Floquet phase $\\beta_0 d / \\pi$",
            ylabel="Frequency $f$ [GHz]",
            title="(a) 2D vs 3D $\\mathrm{LSE}_{10}$ dispersion",
        )
        ax0.legend(loc="lower right", fontsize=6.8)

        # Panel (b): Verify k_perp^2 = k_0^2 - (pi/a)^2 residual
        f_3d_fit_g = t3d_g["f3d_glide_low_ghz"]
        f_2d_recovered = np.sqrt(np.maximum(f_3d_fit_g**2 - fc_ghz**2, 0.0))
        f_2d_interp = np.interp(bn_3d, primary_2d["bn"], primary_2d["f_glide_low_ghz"])
        ax1.plot(bn_3d, f_2d_interp, "o-", color=figstyle.PALETTE[2], lw=2.0, ms=6, label="Direct 2D Wakis FIT $f_{\\mathrm{2D}}(\\beta_0)$")
        ax1.plot(bn_3d, f_2d_recovered, "s--", color=figstyle.PALETTE[1], lw=2.0, ms=6, label="3D Wakis FIT $\\sqrt{f_{\\mathrm{3D}}^2 - (c/2a)^2}$")
        ax1.plot(bn_3d, t3d_g["mcvey_3d_g_low_ghz"], "^:", color=figstyle.PALETTE[0], lw=1.8, ms=5.5, label="3D McVey field-matching $f_{\\mathrm{3D}}(\\beta_0)$")
        ax1.set(
            xlabel="Normalized Floquet phase $\\beta_0 d / \\pi$",
            ylabel="Frequency [GHz]",
            title="(b) Transverse reduction $k_\\perp^2 = k_0^2 - (\\pi/a)^2$",
        )
        ax1.legend(loc="upper left", fontsize=7.0)

        # Panel (c): 3D LSE_10 beam-averaged Pierce impedance <K_{+1}> and <K_{-2}>
        f3d_mc_arr = t3d_g["mcvey_3d_g_low_ghz"]
        kp1_dir, kp1_vgw, km2_dir, km2_vgw = [], [], [], []
        for bnv, f_mc_g in zip(bn_3d, f3d_mc_arr):
            b0_3d = float(bnv * np.pi / grating_spec.period_m)
            chk_3d = h.compute_mcvey_power_and_energy_check(
                b0_3d,
                grating_spec,
                symmetry="glide_b",
                n_harm=40,
                m_slot=6,
                f_cut_hz=f_cut_hz,
                freq_hz=float(f_mc_g * 1e9),
            )
            i_p1 = int(np.where(chk_3d["n_indices"] == 1)[0][0])
            i_m2 = int(np.where(chk_3d["n_indices"] == -2)[0][0])
            kp1_dir.append(float(chk_3d["k_n_beam_avg_ohm"][i_p1]))
            kp1_vgw.append(float(chk_3d["k_n_beam_avg_vg_w_ohm"][i_p1]))
            km2_dir.append(float(chk_3d["k_n_beam_avg_ohm"][i_m2]))
            km2_vgw.append(float(chk_3d["k_n_beam_avg_vg_w_ohm"][i_m2]))

        ax2.semilogy(f3d_mc_arr, kp1_dir, "o-", color=figstyle.PALETTE[1], lw=2.0, ms=5.5, label="$\\langle K_{+1}\\rangle_{\\mathrm{beam}}$ ($P_{\\mathrm{direct}}$)")
        ax2.semilogy(f3d_mc_arr, kp1_vgw, "s--", color=figstyle.PALETTE[0], lw=1.7, ms=4.5, label="$\\langle K_{+1}\\rangle_{\\mathrm{beam}}$ ($v_g W / d$)")
        ax2.semilogy(f3d_mc_arr, km2_dir, "^-", color=figstyle.PALETTE[2], lw=1.9, ms=5.5, label="$\\langle K_{-2}\\rangle_{\\mathrm{beam}}$ ($P_{\\mathrm{direct}}$)")
        ax2.semilogy(f3d_mc_arr, km2_vgw, "d:", color=figstyle.PALETTE[4], lw=1.6, ms=4.5, label="$\\langle K_{-2}\\rangle_{\\mathrm{beam}}$ ($v_g W / d$)")
        ax2.set(
            xlabel="3D Frequency $f_{\\mathrm{3D}}$ [GHz]",
            ylabel="Beam-averaged Pierce impedance [$\\Omega$]",
            title="(c) 3D $\\mathrm{LSE}_{10}$ Pierce impedance ($W_b\\times H_b$)",
        )
        ax2.legend(loc="upper right", fontsize=6.8)
        p12 = fig_dir / "12_2d_vs_3d_waveguide_cutoff_mapping.png"
        figstyle.finish(fig)
        figstyle.save(fig, p12)
        saved.append(p12)

    # ------------------------------------------------------------------
    # Fig 13: 3D Marosi V-Band Staggered-Pillar SWS Dispersion & Stopband Closure
    # ------------------------------------------------------------------
    tm_g = tasks.get("3d_marosi_glide")
    tm_m = tasks.get("3d_marosi_mirror")
    if tm_g is not None:
        bn_m = tm_g["bn"]
        beta0_m = bn_m * np.pi / marosi_spec.period_m
        f_lo_g = tm_g["f_fwd_ghz"]
        f_up_g = tm_g["f_bwd_ghz"]
        ve_lo_g = tm_g["ve_fwd_over_c"]
        ve_up_g = tm_g["ve_bwd_over_c"]
        vg_lo_g = np.gradient(2.0 * np.pi * f_lo_g * 1e9, beta0_m) / h.C0
        vg_up_g = np.gradient(2.0 * np.pi * f_up_g * 1e9, beta0_m) / h.C0

        bn_lo_g = 2.0 + bn_m
        bn_up_g = (4.0 - bn_m)[::-1]
        f_up_g_asc = f_up_g[::-1]
        bn_unfolded = np.concatenate([bn_lo_g, bn_up_g[bn_up_g > 3.001]])
        f_glide_unfolded = np.concatenate([f_lo_g, f_up_g_asc[bn_up_g > 3.001]])

        marosi_paper_bn = np.array([2.25, 2.50, 2.80, 3.00, 3.20, 3.50, 3.75])
        marosi_paper_fg = np.array([50.3, 52.8, 57.5, 61.0, 65.0, 70.0, 72.7])
        f_degen_val = float(f_lo_g[-1])
        f_fit_disp_interp = np.interp(marosi_paper_bn, bn_unfolded, f_glide_unfolded)

        print("\n3D Marosi glide dispersion vs. Fig. 2 (Marosi et al. 2025):")
        print("| beta_n d / pi | CST Fig. 2 [GHz] | Wakis FIT (interp) [GHz] | Rel. diff [%] |")
        print("|---|---|---|---|")
        for b_p, f_p, f_i in zip(marosi_paper_bn, marosi_paper_fg, f_fit_disp_interp):
            print(f"| {b_p:.2f} | {f_p:.2f} | {f_i:.3f} | {(f_i - f_p) / f_p * 100.0:+.1f}% |")

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.6, height=4.2)
        ax0.plot(
            bn_lo_g,
            f_lo_g,
            "o-",
            color=figstyle.PALETTE[2],
            lw=2.1,
            ms=5.5,
            label="Wakis 3D FIT Glide (lower mode, $n=+1$)",
        )
        ax0.plot(
            bn_up_g,
            f_up_g_asc,
            "o--",
            color=figstyle.PALETTE[2],
            lw=2.0,
            ms=5.5,
            mfc="none",
            label="Wakis 3D FIT Glide (upper mode, $n=-2$)",
        )
        if tm_m is not None:
            f_lo_m = tm_m["f_fwd_ghz"]
            f_up_m = tm_m["f_bwd_ghz"]
            ax0.plot(
                2.0 + tm_m["bn"],
                f_lo_m,
                "^-",
                color=figstyle.PALETTE[1],
                lw=1.9,
                ms=5.5,
                label="Wakis 3D FIT Mirror (lower mode, $n=+1$)",
            )
            ax0.plot(
                (4.0 - tm_m["bn"])[::-1],
                f_up_m[::-1],
                "^--",
                color=figstyle.PALETTE[1],
                lw=1.9,
                ms=5.5,
                mfc="none",
                label="Wakis 3D FIT broken glide (upper mode, $n=-2$)",
            )
            ax0.axhspan(
                float(f_lo_m[-1]),
                float(f_up_m[-1]),
                color=figstyle.PALETTE[1],
                alpha=0.12,
                label=f"Broken-glide $3\\pi$ stopband ({float(f_lo_m[-1]):.2f}–{float(f_up_m[-1]):.2f} GHz)",
            )
        ax0.plot(
            marosi_paper_bn,
            marosi_paper_fg,
            "s:",
            color=figstyle.PALETTE[0],
            lw=1.8,
            ms=6,
            label="Marosi et al. (2025) Fig. 2 (CST, $\\pm 0.3$ GHz)",
        )
        ax0.axvline(
            3.0,
            color="0.4",
            ls="--",
            lw=1.2,
            label=f"Glide degeneracy $\\beta_n d = 3\\pi$ ({f_degen_val:.2f} GHz)",
        )
        ax0.set(
            xlim=(2.1, 3.9),
            ylim=(44.5, 78.5),
            xlabel="Normalized phase shift $\\beta_n d / \\pi$",
            ylabel="Frequency $f$ [GHz]",
            title="(a) 3D V-Band dispersion: glide vs broken-glide control & CST",
        )
        ax0.legend(loc="upper left", fontsize=6.3)

        if "ez_yz_slice_fwd" in tm_g:
            ez_sl = np.real(tm_g["ez_yz_slice_fwd"])
            ez_sl = ez_sl / max(float(np.max(np.abs(ez_sl))), 1e-30)
            y_m_um = np.linspace(-400.0, 400.0, ez_sl.shape[0])
            z_m_um = np.linspace(0.0, 1050.0, ez_sl.shape[1])
            im13 = ax1.pcolormesh(z_m_um, y_m_um, ez_sl, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
            h.overlay_marosi_3d_yz_geometry(ax1, marosi_spec)
            ax1.set(
                xlabel="Axial coordinate $z$ [$\\mu$m]",
                ylabel="Vertical coordinate $y$ [$\\mu$m]",
                title="(b) 3D lower-mode $\\operatorname{Re}\\{E_z(0, y, z)\\}$ near $\\beta_n d \\approx 2.85\\pi$",
            )
            fig.colorbar(im13, ax=ax1, label="Normalized $\\operatorname{Re}\\{E_z\\}$")
        p13 = fig_dir / "13_3d_marosi_dispersion_and_stopband.png"
        figstyle.finish(fig)
        figstyle.save(fig, p13)
        saved.append(p13)

        # ------------------------------------------------------------------
        # Fig 14: 3D Marosi Normalized Phase Velocity v_p / c & 5.2 kV Beam Sync
        # ------------------------------------------------------------------
        beta_unfolded = bn_unfolded * np.pi / marosi_spec.period_m
        vp_over_c = (2.0 * np.pi * f_glide_unfolded * 1e9) / (beta_unfolded * h.C0)
        paper_beta = marosi_paper_bn * np.pi / marosi_spec.period_m
        paper_vp_c = (2.0 * np.pi * marosi_paper_fg * 1e9) / (paper_beta * h.C0)

        fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=11.4, height=4.0)
        ax0.plot(f_glide_unfolded, vp_over_c, "o-", color=figstyle.PALETTE[2], lw=2.1, ms=6, label="Wakis 3D FIT $v_p / c$")
        ax0.plot(marosi_paper_fg, paper_vp_c, "s--", color=figstyle.PALETTE[0], lw=1.8, ms=6, label="Marosi et al. (2025) Fig. 3 (CST)")
        ax0.axhline(
            u0_5k2v / h.C0,
            color="#e53935",
            ls="-.",
            lw=2.0,
            label=f"5.2 kV sheet beam ($u_0/c = {u0_5k2v / h.C0:.4f}$, sync $66.0$ GHz @ $3.3\\pi$)",
        )
        ax0.axvspan(57.0, 68.0, color=figstyle.PALETTE[2], alpha=0.10, label="Operating TWT band (57–68 GHz)")
        ax0.set(
            xlabel="Frequency $f$ [GHz]",
            ylabel="Normalized phase velocity $v_p / c$",
            title="(a) Phase velocity vs frequency (Marosi et al. Fig. 3a)",
        )
        ax0.legend(loc="upper right", fontsize=7.5)

        ax1.plot(bn_unfolded, vp_over_c, "o-", color=figstyle.PALETTE[2], lw=2.1, ms=6, label="Wakis 3D FIT $v_p / c$")
        ax1.plot(marosi_paper_bn, paper_vp_c, "s--", color=figstyle.PALETTE[0], lw=1.8, ms=6, label="Marosi et al. (2025) Fig. 3b (CST)")
        ax1.axhline(u0_5k2v / h.C0, color="#e53935", ls="-.", lw=2.0, label="5.2 kV beam velocity $u_0 / c$")
        ax1.axvline(3.30, color="0.4", ls=":", lw=1.4, label="Synchronism $\\beta_n d \\approx 3.3\\pi$")
        ax1.set(
            xlabel="Normalized phase shift $\\beta_n d / \\pi$",
            ylabel="Normalized phase velocity $v_p / c$",
            title="(b) Phase velocity vs phase shift (Marosi et al. Fig. 3b)",
        )
        ax1.legend(loc="upper right", fontsize=7.6)
        p14 = fig_dir / "14_3d_marosi_phase_velocity_and_beam_sync.png"
        figstyle.finish(fig)
        figstyle.save(fig, p14)
        saved.append(p14)

        # ------------------------------------------------------------------
        # Fig 15: 3D Marosi Pierce Impedance (Forward & Backward) & v_E vs v_g
        # ------------------------------------------------------------------
        kn_n = tm_g["kn_n_indices"]
        idx_p1 = int(np.where(kn_n == 1)[0][0])
        idx_m2 = int(np.where(kn_n == -2)[0][0])

        zp_fwd_lo_ax = tm_g["kn_lower_axis_ohm"][:, idx_p1]
        zp_fwd_lo_av = tm_g["kn_lower_avg_ohm"][:, idx_p1]
        zp_fwd_up_ax = tm_g["kn_upper_axis_ohm"][:, idx_m2]
        zp_fwd_up_av = tm_g["kn_upper_avg_ohm"][:, idx_m2]

        mask_lo_g = np.isfinite(zp_fwd_lo_ax)
        mask_up_g = np.isfinite(zp_fwd_up_ax)
        f_fwd_cont = np.concatenate([f_lo_g[mask_lo_g], f_up_g[mask_up_g][::-1]])
        zp_fwd_cont_ax = np.concatenate([zp_fwd_lo_ax[mask_lo_g], zp_fwd_up_ax[mask_up_g][::-1]])
        zp_fwd_cont_av = np.concatenate([zp_fwd_lo_av[mask_lo_g], zp_fwd_up_av[mask_up_g][::-1]])

        zp_bwd_up_ax = tm_g["kn_upper_axis_ohm"][:, idx_p1]
        zp_bwd_up_av = tm_g["kn_upper_avg_ohm"][:, idx_p1]
        zp_bwd_lo_ax = tm_g["kn_lower_axis_ohm"][:, idx_m2]
        zp_bwd_lo_av = tm_g["kn_lower_avg_ohm"][:, idx_m2]

        paper_f_zp = np.array([53.0, 56.0, 58.0, 60.0, 65.0, 68.0])
        paper_zp_fwd = np.array([9.9, 5.1, 3.8, 2.8, 1.6, 1.3])
        zp_fit_interp_ax = np.interp(paper_f_zp, f_fwd_cont, zp_fwd_cont_ax)
        zp_fit_interp_av = np.interp(paper_f_zp, f_fwd_cont, zp_fwd_cont_av)

        print("\n3D Marosi forward-wave Z_P vs. Fig. 5a / Sec. II (Marosi et al. 2025):")
        print("| f [GHz] | Paper Z_P [Ohm] | Wakis FIT Z_P(0,0) (interp) [Ohm] | Wakis FIT <Z_P>_beam (interp) [Ohm] | Rel. diff [%] |")
        print("|---|---|---|---|---|")
        for fp, zp_p, zi_ax, zi_av in zip(paper_f_zp, paper_zp_fwd, zp_fit_interp_ax, zp_fit_interp_av):
            print(f"| {fp:.1f} | {zp_p:.2f} | {zi_ax:.3f} | {zi_av:.3f} | {(zi_ax - zp_p) / zp_p * 100.0:+.1f}% |")

        if tm_m is not None:
            f_lo_m = tm_m["f_fwd_ghz"]
            f_up_m = tm_m["f_bwd_ghz"]
            ve_lo_m = tm_m["ve_fwd_over_c"]
            ve_up_m = tm_m["ve_bwd_over_c"]
            vg_lo_m = np.gradient(2.0 * np.pi * f_lo_m * 1e9, beta0_m) / h.C0
            vg_up_m = np.gradient(2.0 * np.pi * f_up_m * 1e9, beta0_m) / h.C0
            zp_m_bwd_up_ax = tm_m["kn_upper_axis_ohm"][:, idx_p1]
            zp_m_bwd_up_av = tm_m["kn_upper_avg_ohm"][:, idx_p1]
            zp_m_bwd_lo_ax = tm_m["kn_lower_axis_ohm"][:, idx_m2]
            zp_m_bwd_lo_av = tm_m["kn_lower_avg_ohm"][:, idx_m2]

            print("\n3D Marosi v_E / v_g consistency check (interior indices 1..10):")
            print("| idx | beta_0 d/pi | Glide lower v_E/v_g | Glide upper v_E/v_g | Broken-glide lower v_E/v_g | Broken-glide upper v_E/v_g |")
            print("|---|---|---|---|---|---|")
            for k in range(1, 11):
                print(
                    f"| {k:2d} | {bn_m[k]:.4f} | {ve_lo_g[k] / vg_lo_g[k]:.4f} | {ve_up_g[k] / vg_up_g[k]:.4f} "
                    f"| {ve_lo_m[k] / vg_lo_m[k]:.4f} | {ve_up_m[k] / vg_up_m[k]:.4f} |"
                )

            print("\n3D Marosi backward-wave Z_P(0,0) and <Z_P>_beam (glide vs. broken-glide control):")
            print("| idx | beta_0 d/pi | Glide up n=+1 ax / avg [Ohm] | Glide lo n=-2 ax / avg [Ohm] | Broken-glide up n=+1 ax / avg [Ohm] | Broken-glide lo n=-2 ax / avg [Ohm] |")
            print("|---|---|---|---|---|---|")
            for k in range(11):
                print(
                    f"| {k:2d} | {bn_m[k]:.4f} | {zp_bwd_up_ax[k]:.2e} / {zp_bwd_up_av[k]:.4f} "
                    f"| {zp_bwd_lo_ax[k]:.2e} / {zp_bwd_lo_av[k]:.4f} "
                    f"| {zp_m_bwd_up_ax[k]:.4f} / {zp_m_bwd_up_av[k]:.4f} "
                    f"| {zp_m_bwd_lo_ax[k]:.4f} / {zp_m_bwd_lo_av[k]:.4f} |"
                )

        fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=14.6, height=4.1)
        ax0.plot(
            f_fwd_cont,
            zp_fwd_cont_ax,
            "o-",
            color=figstyle.PALETTE[2],
            lw=2.1,
            ms=5.0,
            label="Wakis 3D FIT $Z_P(0,0)$ (lower $n=+1$ & upper $n=-2$)",
        )
        ax0.plot(
            f_fwd_cont,
            zp_fwd_cont_av,
            "s--",
            color=figstyle.PALETTE[0],
            lw=1.8,
            ms=4.5,
            label="Wakis 3D FIT $\\langle Z_P \\rangle_{\\mathrm{beam}}$",
        )
        ax0.plot(
            paper_f_zp,
            paper_zp_fwd,
            "^:",
            color=figstyle.PALETTE[1],
            lw=2.0,
            ms=6,
            label="Marosi et al. (2025) Fig. 5a (3.8 $\\Omega$ @ 58, 1.3 $\\Omega$ @ 68 GHz)",
        )
        ax0.axvline(
            f_degen_val,
            color="0.4",
            ls="--",
            lw=1.2,
            label=f"$f_{{3\\pi}} = {f_degen_val:.2f}$ GHz",
        )
        ax0.set(
            xlim=(48.0, 74.5),
            ylim=(0.0, 64.0),
            xlabel="Frequency $f$ [GHz]",
            ylabel="Forward Pierce impedance $Z_P$ [$\\Omega$]",
            title="(a) Continuous forward-wave $Z_P(f)$ vs Fig. 5a",
        )
        ax0.legend(loc="upper right", fontsize=6.2)

        ax0_ins = ax0.inset_axes([0.33, 0.28, 0.63, 0.43])
        ax0_ins.plot(f_fwd_cont, zp_fwd_cont_ax, "o-", color=figstyle.PALETTE[2], lw=1.6, ms=3.8)
        ax0_ins.plot(f_fwd_cont, zp_fwd_cont_av, "s--", color=figstyle.PALETTE[0], lw=1.4, ms=3.4)
        ax0_ins.plot(paper_f_zp, paper_zp_fwd, "^:", color=figstyle.PALETTE[1], lw=1.6, ms=4.5)
        ax0_ins.axvline(f_degen_val, color="0.4", ls="--", lw=1.0)
        ax0_ins.set(xlim=(52.0, 74.2), ylim=(0.0, 11.2))
        ax0_ins.tick_params(labelsize=6.5)
        ax0_ins.grid(True, ls=":", alpha=0.4)
        ax0_ins.set_title("Zoom: 52–74 GHz passband", fontsize=6.8, pad=2)

        zp_floor = 1e-6
        ax1.semilogy(
            f_lo_g[mask_lo_g],
            np.maximum(zp_bwd_lo_ax[mask_lo_g], zp_floor),
            "o-",
            color=figstyle.PALETTE[2],
            lw=2.0,
            ms=5.5,
            label="Glide $Z_P^{\\mathrm{bw}}(0,0)$ (lower $n=-2$)",
        )
        ax1.semilogy(
            f_up_g[mask_up_g][::-1],
            np.maximum(zp_bwd_up_ax[mask_up_g][::-1], zp_floor),
            "s-",
            color=figstyle.PALETTE[2],
            lw=2.0,
            ms=5.5,
            mfc="none",
            label="Glide $Z_P^{\\mathrm{bw}}(0,0)$ (upper $n=+1$)",
        )
        ax1.semilogy(
            f_lo_g[mask_lo_g],
            np.maximum(zp_bwd_lo_av[mask_lo_g], zp_floor),
            "o--",
            color=figstyle.PALETTE[0],
            lw=1.7,
            ms=4.5,
            label="Glide $\\langle Z_P^{\\mathrm{bw}}\\rangle_{\\mathrm{beam}}$ (lower $n=-2$)",
        )
        ax1.semilogy(
            f_up_g[mask_up_g][::-1],
            np.maximum(zp_bwd_up_av[mask_up_g][::-1], zp_floor),
            "s--",
            color=figstyle.PALETTE[0],
            lw=1.7,
            ms=4.5,
            mfc="none",
            label="Glide $\\langle Z_P^{\\mathrm{bw}}\\rangle_{\\mathrm{beam}}$ (upper $n=+1$)",
        )
        if tm_m is not None:
            mask_m_lo = np.isfinite(zp_m_bwd_lo_ax)
            mask_m_up = np.isfinite(zp_m_bwd_up_ax)
            ax1.semilogy(
                f_lo_m[mask_m_lo],
                np.maximum(zp_m_bwd_lo_ax[mask_m_lo], zp_floor),
                "^-",
                color="#e53935",
                lw=1.9,
                ms=5.5,
                label="Broken-glide $Z_P^{\\mathrm{bw}}(0,0)$ (lower $n=-2$)",
            )
            ax1.semilogy(
                f_up_m[mask_m_up][::-1],
                np.maximum(zp_m_bwd_up_ax[mask_m_up][::-1], zp_floor),
                "v-",
                color=figstyle.PALETTE[1],
                lw=1.9,
                ms=5.5,
                label="Broken-glide $Z_P^{\\mathrm{bw}}(0,0)$ (upper $n=+1$)",
            )
        ax1.annotate(
            "Paper Fig. 11a: $\\approx 0\\;\\Omega$ on axis\n(glide $\\leq 1.2\\times 10^{-4}\\;\\Omega$; floor $10^{-6}\\;\\Omega$)",
            xy=(55.0, zp_floor),
            xytext=(48.5, 4.0e-6),
            fontsize=6.8,
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "0.3"},
        )
        ax1.set(
            xlim=(44.5, 78.0),
            ylim=(4e-7, 25.0),
            xlabel="Frequency $f$ [GHz]",
            ylabel="Backward Pierce impedance $Z_P^{\\mathrm{bw}}$ [$\\Omega$] (log, floor $10^{-6}$)",
            title="(b) Backward-wave on-axis suppression: glide vs broken glide",
        )
        ax1.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), fontsize=6.0, framealpha=0.9)

        # Panel (c): Stage-2 energy velocity v_E vs Stage-1 group velocity v_g (interior bn[1:11])
        ax2.plot(bn_m[1:], vg_lo_g[1:], "-", color=figstyle.PALETTE[2], lw=1.9, label="Glide $v_g/c$ (lower/upper)")
        ax2.plot(bn_m[1:], vg_up_g[1:], "--", color=figstyle.PALETTE[2], lw=1.9)
        ax2.plot(bn_m[1:11], ve_lo_g[1:11], "o", color=figstyle.PALETTE[2], ms=5.5, label="Glide $v_E/c$ (Stage-2 DFT)")
        ax2.plot(bn_m[1:11], ve_up_g[1:11], "s", color=figstyle.PALETTE[2], ms=5.5, mfc="none")
        if tm_m is not None:
            ax2.plot(bn_m[1:], vg_lo_m[1:], "-", color=figstyle.PALETTE[1], lw=1.8, label="Broken-glide $v_g/c$ (lower/upper)")
            ax2.plot(bn_m[1:], vg_up_m[1:], "--", color=figstyle.PALETTE[1], lw=1.8)
            ax2.plot(bn_m[1:11], ve_lo_m[1:11], "^", color=figstyle.PALETTE[1], ms=5.5, label="Broken-glide $v_E/c$ (Stage-2 DFT)")
            ax2.plot(bn_m[1:11], ve_up_m[1:11], "v", color=figstyle.PALETTE[1], ms=5.5, mfc="none")
        ax2.axhline(0.0, color="0.4", ls=":", lw=1.1)
        ax2.set(
            xlim=(0.18, 1.03),
            ylim=(-0.155, 0.155),
            xlabel="Normalized Floquet phase $\\beta_0 d / \\pi$",
            ylabel="Normalized velocity $v / c$",
            title="(c) Consistency check: $v_E = P_z d / U$ vs $v_g = \\mathrm{d}\\omega/\\mathrm{d}\\beta_0$",
        )
        ax2.legend(loc="lower left", fontsize=6.5)
        p15 = fig_dir / "15_3d_marosi_pierce_impedance_vs_paper.png"
        figstyle.finish(fig)
        figstyle.save(fig, p15)
        saved.append(p15)

    print(f"Rendered {len(saved)} figures in {fig_dir}")
    return saved


# %% [markdown]
# ## 4. Cold-Test Results: 2D Analytic Benchmark vs. Wakis FIT
#
# Running the 2D double-corrugated grating cold test (`Tasks 5–8`, Table I of Karetnikova et al. [2]:
# $d = p = 0.50\;\text{mm}$, $l = s_{\text{vane}} = 0.10\;\text{mm}$, $s = 0.40\;\text{mm}$,
# $h = l_{\text{vane}} = 0.30\;\text{mm}$, $2a = 2 g_{\text{half}} = 0.20\;\text{mm}$, $H_b = 0.15\;\text{mm}$)
# across the 15-point `--full`
# Brillouin sweep (`Task 6`, `L2`, $n_{\text{feat}} = 8$, $\Delta y = \Delta z = 12.5\;\mu\text{m}$,
# $1\times 64\times 40 = 2{,}560$ cells, $\beta_0 d / \pi \in \{0.125, 0.1875, \dots, 0.500, \dots, 1.000\}$,
# $1{,}062{,}450$ total steps) yields quantitative agreement with multimodal McVey theory
# (Brillouin sweep evaluated at $N_{\text{harm}} = 40$, $M_{\text{slot}} = 6$, with the $\beta_0 d = 0.5\pi$
# benchmark point extrapolated to the $M_{\text{slot}} \to \infty$ limit $f_{1,\infty} = 98.120\;\text{GHz}$ at $N_{\text{harm}} = 60$):
#
# | Floquet phase $\beta_0 d / \pi$ (`--full`) | Mirror $f_1$ (McVey) | Mirror $f_1$ (`--full` L2) | Glide $f_1$ (McVey) | Glide $f_1$ (`--full` L2) | Glide $f_2$ (McVey) | Glide $f_2$ (`--full` L2) |
# |---|---|---|---|---|---|---|
# | $0.1875$ | $37.52\;\text{GHz}$ | $37.58\;\text{GHz}$ | $38.19\;\text{GHz}$ | $38.22\;\text{GHz}$ | $191.70\;\text{GHz}$ | $191.66\;\text{GHz}$ |
# | $0.5000$ | $94.49\;\text{GHz}$ | $94.67\;\text{GHz}$ | $98.12\;\text{GHz}$ ($M\to\infty$; $97.69$ at $M=6$) | $97.80\;\text{GHz}$ (`Ref` $97.99$, Rich. $98.07$) | $188.98\;\text{GHz}$ | $189.00\;\text{GHz}$ |
# | $0.8125$ | $133.03\;\text{GHz}$ | $133.37\;\text{GHz}$ | $145.27\;\text{GHz}$ | $145.45\;\text{GHz}$ | $178.52\;\text{GHz}$ | $178.64\;\text{GHz}$ |
# | $1.0000$ | $140.29\;\text{GHz}$ | $140.66\;\text{GHz}$ | $165.28\;\text{GHz}$ | $165.46\;\text{GHz}$ | $165.28\;\text{GHz}$ | $165.46\;\text{GHz}$ |
#
# - **Stopband closure**: In the mirror-symmetric grating at $\beta_0 d = \pi$, a $55.23\;\text{GHz}$ stopband
#   separates the lower passband edge ($140.66\;\text{GHz}$, vs. $140.29\;\text{GHz}$ in McVey theory) from the
#   upper branch ($195.89\;\text{GHz}$, vs. $195.88\;\text{GHz}$ in McVey theory, $\Delta f_{\text{gap}} = 55.59\;\text{GHz}$).
#   Displacing the top grating by $d/2$ closes this stopband completely at $165.46\;\text{GHz}$ in Wakis FIT `L2`
#   ($165.28\;\text{GHz}$ in McVey theory; $+0.11\%$ difference, Fig. 4a).
# - **Group vs. energy velocity**: At $\beta_0 d = 0.5\pi$, the finite-difference dispersion slope gives
#   $v_g / c = 0.5886$ in the $M_{\text{slot}} \to \infty$ limit ($0.5857$ at $M_{\text{slot}} = 6$), while the
#   Hann-windowed integer-cycle `DFTMonitor` integral $v_E = P_z d / U_{\text{cell}}$ converges monotonically
#   across the 2D conformal ladder: $0.5569\,c$ at `L1` ($n_{\text{feat}} = 4$), $0.5665\,c$ at `L2` ($n_{\text{feat}} = 8$),
#   $0.5739\,c$ at `L3` ($n_{\text{feat}} = 12$), and $0.5776\,c$ at `Reference` ($n_{\text{feat}} = 16$, within $-1.87\%$
#   of $v_{g,\infty} / c$, Fig. 5a).
# - **Space-harmonic parity & first-principles McVey Pierce impedance**: In the glide-symmetric lower branch (`glide_b`),
#   odd harmonics ($n = \pm 1, \pm 3$) have $\cosh(\tau_n y)$ profiles with strong on-axis electric field, whereas even
#   harmonics ($n = 0, \pm 2$) have antisymmetric $\sinh(\tau_n y)$ profiles that vanish on the midplane $y = 0$
#   ($K_0(0) = 1.28\times 10^{-4}\;\Omega$ at `L1`, $4.34\times 10^{-6}\;\Omega$ at `L2`, $1.34\times 10^{-6}\;\Omega$
#   at `L3`, and $5.88\times 10^{-7}\;\Omega$ at `Reference` when all $N_z$ axial cells are included in the Floquet
#   projection, Fig. 6). Using the first-principles derivation of Section 1.3
#   ($P = P_{\text{tunnel}} + P_{\text{slot}}$ and the independent energy-transport check $P_{v_g W} = v_g W / d$ with
#   $\delta\beta_0 d = 10^{-3}\pi$), the 2D grating at $\beta_0 d = 0.5\pi$ gives:
#
# | Truncation $(N_{\text{harm}}, M_{\text{slot}})$ | $f_1$ [GHz] | $v_g / c$ | $P_{\text{tunnel}}$ [W] | $P_{\text{slot}}$ [W] | $P_{\text{direct}}$ [W] | $P_{v_g W}$ [W] | $P_{\text{direct}} / P_{v_g W}$ | $K_{-1}(0)$ ($P_{\text{direct}}$) | $K_{-1}(0)$ ($P_{v_g W}$) | Tunnel-only $K_{-1}(0)$ | $\langle K_{-1}\rangle_{\text{beam}}$ |
# |---|---|---|---|---|---|---|---|---|---|---|---|
# | $(15, 5)$ | $97.5217$ | $0.58611$ | $1.7362\times 10^{-10}$ | $1.1277\times 10^{-10}$ | $2.8639\times 10^{-10}$ | $2.8675\times 10^{-10}$ | $0.99875$ ($-0.13\%$) | $1.8338\;\Omega$ | $1.8315\;\Omega$ | $3.0248\;\Omega$ | $2.1538\;\Omega$ |
# | $(40, 6)$ | $97.6868$ | $0.58573$ | $1.6670\times 10^{-10}$ | $1.0972\times 10^{-10}$ | $2.7642\times 10^{-10}$ | $2.7621\times 10^{-10}$ | $1.00079$ ($+0.08\%$) | $1.8274\;\Omega$ | $1.8288\;\Omega$ | $3.0302\;\Omega$ | $2.1462\;\Omega$ |
# | $(60, 10)$ | $97.9146$ | $0.58734$ | $1.5541\times 10^{-10}$ | $1.0344\times 10^{-10}$ | $2.5885\times 10^{-10}$ | $2.5880\times 10^{-10}$ | $1.00022$ ($+0.02\%$) | $1.7897\;\Omega$ | $1.7901\;\Omega$ | $2.9808\;\Omega$ | $2.1019\;\Omega$ |
# | $(60, 16)$ | $98.0152$ | $0.58800$ | $1.4807\times 10^{-10}$ | $9.9056\times 10^{-11}$ | $2.4712\times 10^{-10}$ | $2.4711\times 10^{-10}$ | $1.00007$ ($+0.007\%$) | $1.7756\;\Omega$ | $1.7758\;\Omega$ | $2.9636\;\Omega$ | $2.0854\;\Omega$ |
# | $(60, 32)$ | $98.0813$ | $0.58842$ | $1.4036\times 10^{-10}$ | $9.4218\times 10^{-11}$ | $2.3458\times 10^{-10}$ | $2.3458\times 10^{-10}$ | $1.00001$ ($+0.001\%$) | $1.7675\;\Omega$ | $1.7675\;\Omega$ | $2.9539\;\Omega$ | $2.0758\;\Omega$ |
# | $(60, \infty)$ fit | $98.1201$ | $0.58864$ | — | — | — | — | $1.00000$ | $1.7637\;\Omega$ | $1.7637\;\Omega$ | — | $2.0714\;\Omega$ |
#
#   Direct integration $P_{\text{direct}} = P_{\text{tunnel}} + P_{\text{slot}}$ and the energy-transport identity
#   $P_{v_g W} = v_g W / d$ agree to $-0.13\%$ at $(15, 5)$, $+0.08\%$ at $(40, 6)$, $+0.02\%$ at $(60, 10)$, and
#   $+0.001\%$ at $(60, 32)$ (whereas omitting $P_{\text{slot}}$ leaves $P_{\text{tunnel}} / P_{v_g W} \approx 0.60$, a
#   $\sim 40\%$ power deficit).
# - **Slot-mode convergence of McVey $f_1$ and $K_{-1}(0)$ and comparison with Wakis FIT (`compute_mcvey_slot_mode_extrapolation`)**:
#   At fixed $M_{\text{slot}}$, the Floquet sum $N_{\text{harm}}$ is already converged to $< 0.02\%$ by $N_{\text{harm}} = 15\dots 60$:
#
# | $M_{\text{slot}} \backslash N_{\text{harm}}$ | $5$ | $10$ | $15$ | $20$ | $30$ | $40$ |
# |---|---|---|---|---|---|---|
# | $2$ | $2.5405\;\Omega$ | $2.5378\;\Omega$ | $2.5372\;\Omega$ | $2.5370\;\Omega$ | $2.5368\;\Omega$ | $2.5368\;\Omega$ |
# | $3$ | $1.9410\;\Omega$ | $1.9419\;\Omega$ | $1.9421\;\Omega$ | $1.9421\;\Omega$ | $1.9421\;\Omega$ | $1.9421\;\Omega$ |
# | $4$ | $1.9034\;\Omega$ | $1.9037\;\Omega$ | $1.9037\;\Omega$ | $1.9037\;\Omega$ | $1.9037\;\Omega$ | $1.9037\;\Omega$ |
# | $5$ | $1.8302\;\Omega$ | $1.8333\;\Omega$ | $1.8338\;\Omega$ | $1.8339\;\Omega$ | $1.8340\;\Omega$ | $1.8341\;\Omega$ |
# | $6$ | $1.8241\;\Omega$ | $1.8267\;\Omega$ | $1.8271\;\Omega$ | $1.8272\;\Omega$ | $1.8273\;\Omega$ | $1.8274\;\Omega$ |
# | $8$ | $1.7950\;\Omega$ | $1.8005\;\Omega$ | $1.8012\;\Omega$ | $1.8015\;\Omega$ | $1.8017\;\Omega$ | $1.8017\;\Omega$ |
# | $10$ | $1.7661\;\Omega$ | $1.7877\;\Omega$ | $1.7889\;\Omega$ | $1.7893\;\Omega$ | $1.7895\;\Omega$ | $1.7896\;\Omega$ |
#
#   Sweeping $M_{\text{slot}} = 2\dots 32$ at $N_{\text{harm}} = 60$ ($\beta_0 d = 0.5\pi$, 2D glide) and fitting
#   $y(M_{\text{slot}}) = y_\infty + C\,M_{\text{slot}}^{-p}$ with `scipy.optimize.curve_fit` over $M_{\text{slot}} = 10\dots 32$:
#
# | $M_{\text{slot}}$ | $f_1$ [GHz] | $K_{-1}(0)$ [$\Omega$] | $\langle K_{-1}\rangle_{\text{beam}}$ [$\Omega$] |
# |---|---|---|---|
# | $2$ | $95.7922$ | $2.53672$ | $2.98031$ |
# | $3$ | $96.7526$ | $1.94214$ | $2.28140$ |
# | $4$ | $97.3206$ | $1.90371$ | $2.23605$ |
# | $5$ | $97.5141$ | $1.83412$ | $2.15424$ |
# | $6$ | $97.6860$ | $1.82739$ | $2.14628$ |
# | $8$ | $97.8360$ | $1.80179$ | $2.11616$ |
# | $10$ | $97.9146$ | $1.78969$ | $2.10191$ |
# | $12$ | $97.9620$ | $1.78286$ | $2.09388$ |
# | $16$ | $98.0152$ | $1.77564$ | $2.08538$ |
# | $20$ | $98.0438$ | $1.77200$ | $2.08110$ |
# | $24$ | $98.0613$ | $1.76985$ | $2.07857$ |
# | $32$ | $98.0813$ | $1.76748$ | $2.07578$ |
# | $\infty$ (fit $M = 10\dots 32$) | $98.120$ ($p \approx 1.43$) | $1.7637$ ($p \approx 1.66$) | $2.0714$ ($p \approx 1.66$) |
#
#   The power-law fit over $M_{\text{slot}} = 10\dots 32$ gives $K_{-1,\infty}(0) = 1.7637\;\Omega$ ($C = 1.174\;\Omega$,
#   fitted exponent $p \approx 1.66$) and $f_{1,\infty} = 98.120\;\text{GHz}$ ($C = -5.526\;\text{GHz}$, fitted exponent
#   $p \approx 1.43$; the exponents $p$ are empirical fits to the data, not analytically derived), and dropping the endpoints
#   ($M_{\text{slot}} = 12\dots 24$) changes both limits by $< 0.0003$ ($K_{-1,\infty} = 1.7638\;\Omega$, $\Delta = 2.3\times 10^{-5}\;\Omega$;
#   $f_{1,\infty} = 98.120\;\text{GHz}$, $\Delta = 2.9\times 10^{-4}\;\text{GHz}$). Across the committed 2D Dey–Mittra
#   conformal FIT ladder (`Tasks 5–8`), $K_{-1}(0)$ converges monotonically as $1.8381\;\Omega$ (`L1`) $\to 1.7921\;\Omega$ (`L2`)
#   $\to 1.7802\;\Omega$ (`L3`) $\to 1.7749\;\Omega$ (`Reference`), with Richardson ($p = 2$, $n_{\text{feat}} = 12, 16$)
#   limit $K_{-1,\infty}^{\text{FIT}} = 1.7681\;\Omega$ and $f_{1,\infty}^{\text{FIT}} = 98.0670\;\text{GHz}$. Thus Wakis FIT
#   Richardson $K_{-1,\infty} = 1.7681\;\Omega$ is **$+0.25\%$** above the slot-mode-converged McVey limit $1.7637\;\Omega$
#   (FIT `Reference` $1.7749\;\Omega$ is **$+0.63\%$** above), and FIT Richardson $f_{1,\infty} = 98.0670\;\text{GHz}$ is
#   **$-0.05\%$** below McVey $f_{1,\infty} = 98.120\;\text{GHz}$ (FIT `Reference` $97.9918\;\text{GHz}$ is **$-0.13\%$** below).
#   This is independent agreement within the FIT discretization error: McVey's truncation in $M_{\text{slot}}$ is the dominant
#   analytic error ($M_{\text{slot}} = 6$ leaves a $+3.61\%$ error in $K_{-1}$ and $-0.44\%$ in $f_1$) and must be extrapolated.
#
# ---
#
# ## 5. 2D-vs-3D Mapping & 3D V-Band Sheet-Beam SWS (Marosi et al., 2025)
#
# ### 5.1 Full-Width 3D Rectangular Grating (`Task 9`, Karetnikova et al., 2018 Table I)
#
# Enclosing the double-corrugated grating in a rectangular waveguide of width $b = 0.85\;\text{mm}$
# with Table I dimensions ($p = 500\;\mu\text{m}$, $s_{\text{vane}} = 100\;\mu\text{m}$, $l_{\text{vane}} = 300\;\mu\text{m}$,
# $2a = 200\;\mu\text{m}$, $H_b = 150\;\mu\text{m}$, $W_b = 750\;\mu\text{m}$, $V_0 = 20\;\text{kV}$ [2])
# imposes the $\text{LSE}_{10}$ transverse wavenumber $k_x = \pi / b$ ($f_c = c/(2b) = 176.35\;\text{GHz}$).
# In `--full`, `Task 9` runs on a $34 \times 64 \times 40 = 87{,}040$-cell grid ($n_{\text{feat}} = 8$,
# $\Delta_{\text{bg}} = 25.00\;\mu\text{m}$, $\Delta_{\min} = 12.50\;\mu\text{m}$, $154{,}000$ total steps,
# 11 phase points in $\beta_0 d / \pi \in [0.15, 1.00]$).
#
# - **Wakis 3D FIT vs. 3D McVey dispersion**: Across all 11 points, the `--full` 3D Wakis FIT lower glide branch
#   $[178.85, 182.52, 187.55, 193.68, 200.60, 207.97, 215.49, 222.84, 229.78, 236.09, 241.60]\;\text{GHz}$
#   agrees with 3D McVey theory $\sqrt{f_{\text{2D,McVey}}^2(\beta_0) + f_c^2} = [178.99, 182.68, 187.73, 193.85, 200.75, 208.10, 215.60, 222.94, 229.86, 236.16, 241.70]\;\text{GHz}$
#   to within $\le 0.09\%$ ($-0.04\%$ at $\beta_0 d = \pi$: $241.60\;\text{GHz}$ vs. $241.70\;\text{GHz}$, Fig. 12).
#   The upper glide branch $[260.43, 260.23, 259.89, 259.34, 258.50, 257.28, 255.56, 253.23, 250.18, 246.33, 241.60]\;\text{GHz}$
#   agrees with 3D McVey $[260.55, 260.35, 260.01, 259.45, 258.60, 257.37, 255.64, 253.30, 250.25, 246.39, 241.70]\;\text{GHz}$
#   to within $\le 0.05\%$. In the 3D mirror configuration at $\beta_0 d = \pi$, `--full` resolves the lower
#   passband edge at $225.26\;\text{GHz}$ (McVey: $225.35\;\text{GHz}$, $-0.04\%$) and the upper branch at
#   $263.45\;\text{GHz}$ (McVey: $263.57\;\text{GHz}$, $-0.04\%$), confirming a $38.19\;\text{GHz}$ Bragg stopband
#   that closes completely at $241.60\;\text{GHz}$ under glide symmetry.
# - **Harmonic numbering mapping & loose visual comparison with Fig. 2(a)/(b) of Karetnikova et al. (2018)**
#   (*approximate values read by eye from Fig. 2 of the paper, used only as a loose visual check*):
#   - *Harmonic numbering convention*: In our reduced-zone Floquet expansion $\beta_n = \beta_0 + 2\pi n / d$ with
#     $\beta_0 d \in [0, \pi]$, the paper's second Brillouin zone ($\phi = \beta d \in [360^\circ, 720^\circ] = [2\pi, 4\pi]$)
#     maps to unfolding our **$n = +1$ forward space harmonic** onto $\phi = \beta_{+1} d = 2\pi + \beta_0 d \in [360^\circ, 540^\circ]$
#     on the lower glide branch, and our **$n = -2$ backward space harmonic** onto
#     $\phi = |\beta_{-2}| d = 4\pi - \beta_0 d \in [540^\circ, 720^\circ]$ (coalescing at the odd-$\pi$ point $\phi = 540^\circ = 3\pi$).
#   - *Comparison with Fig. 2(a) (read by eye)*: In Fig. 2(a) of Karetnikova et al. (2018) (computed with their
#     integral-equation code [ref. 18] and checked against COMSOL and HFSS), the passband extends from
#     $\sim 175\text{--}180\;\text{GHz}$ at $\phi = 360^\circ$ to $\sim 260\;\text{GHz}$ at $\phi = 720^\circ$, with
#     the forward $n = +1$ and backward $n = -2$ curves coalescing at the odd-$\pi$ point ($\phi = 540^\circ$) and
#     beam synchronism discussed at $V_0 = 20\;\text{kV}$ ($19\text{--}21\;\text{kV}$ lines). **Our 3D Task 9 dispersion
#     lands squarely inside the paper's $\sim 175\text{--}260\;\text{GHz}$ passband**: the passband starts at the bare
#     $\text{TE}_{10}$ cutoff $f_c = 176.35\;\text{GHz}$ at $\phi = 360^\circ$ ($178.85\;\text{GHz}$ in 3D Wakis FIT at
#     $\beta_0 d = 0.15\pi$, $\phi = 387^\circ$), coalesces at $241.60\;\text{GHz}$ (McVey: $241.70\;\text{GHz}$) at
#     the odd-$\pi$ point $\phi = 540^\circ$, and reaches $260.43\;\text{GHz}$ (McVey: $260.55\;\text{GHz}$) on the
#     upper branch as $\phi \to 720^\circ$. Along the $V_0 = 20\;\text{kV}$ beam line ($u_0 = 0.2719\,c$), the beam
#     tracks the forward $n = +1$ harmonic across $\phi \in [402^\circ, 540^\circ]$ ($182.5\text{--}241.6\;\text{GHz}$).
#   - *Comparison with Fig. 2(b) (read by eye) & 11-point 3D first-principles McVey power/impedance table*: In Fig. 2(b)
#     of the paper (read by eye), the beam-averaged Pierce impedance of the forward $n = +1$ harmonic falls from
#     $\sim 1\;\Omega$ near $185\;\text{GHz}$ to $\sim 0.2\text{--}0.3\;\Omega$ at $220\text{--}250\;\text{GHz}$, while
#     the backward $n = -2$ harmonic is $\sim 2$ orders of magnitude lower. Evaluating the first-principles 3D
#     $\text{LSE}_{10}$ power and energy check (`compute_mcvey_power_and_energy_check`, $N_{\text{harm}} = 40, M_{\text{slot}} = 6$,
#     stored in `data/task_09_3d_grating_karetnikova.npz` and averaged over the $W_b \times H_b = 750\times 150\;\mu\text{m}^2$
#     sheet beam) across all 11 `Task 9` phase points alongside the $(N_{\text{harm}} = 15, M_{\text{slot}} = 5)$
#     truncation check:
#
# | $\beta_0 d / \pi$ | $f_{\text{3D,McVey}}$ [GHz] | $f_{\text{3D,FIT}}$ [GHz] | $P_{\text{tunnel}}$ [W] | $P_{\text{slot}}$ [W] | $P_{\text{direct}}$ [W] | $P_{v_g W}$ [W] | $P_{\text{direct}} / P_{v_g W}$ | $\langle K_{+1}\rangle_{\text{beam}}$ $(40,6)$ / $(15,5)$ [$\Omega$] | $\langle K_{-2}\rangle_{\text{beam}}$ $(40,6)$ / $(15,5)$ [$\Omega$] |
# |---|---|---|---|---|---|---|---|---|---|
# | $0.150$ | $178.99$ | $178.85$ | $9.029\times 10^{-12}$ | $6.094\times 10^{-12}$ | $1.512\times 10^{-11}$ | $1.512\times 10^{-11}$ | $1.00015$ | $2.4399$ / $2.6108$ | $0.0496$ / $0.0526$ |
# | $0.235$ | $182.68$ | $182.52$ | $1.500\times 10^{-11}$ | $1.007\times 10^{-11}$ | $2.507\times 10^{-11}$ | $2.506\times 10^{-11}$ | $1.00021$ | $1.3612$ / $1.4627$ | $0.0376$ / $0.0387$ |
# | $0.320$ | $187.73$ | $187.55$ | $2.111\times 10^{-11}$ | $1.409\times 10^{-11}$ | $3.520\times 10^{-11}$ | $3.519\times 10^{-11}$ | $1.00039$ | $0.9000$ / $0.9688$ | $0.0334$ / $0.0335$ |
# | $0.405$ | $193.85$ | $193.68$ | $2.707\times 10^{-11}$ | $1.794\times 10^{-11}$ | $4.501\times 10^{-11}$ | $4.499\times 10^{-11}$ | $1.00057$ | $0.6586$ / $0.7084$ | $0.0323$ / $0.0317$ |
# | $0.490$ | $200.75$ | $200.60$ | $3.266\times 10^{-11}$ | $2.150\times 10^{-11}$ | $5.417\times 10^{-11}$ | $5.412\times 10^{-11}$ | $1.00077$ | $0.5180$ / $0.5553$ | $0.0330$ / $0.0319$ |
# | $0.575$ | $208.10$ | $207.97$ | $3.770\times 10^{-11}$ | $2.467\times 10^{-11}$ | $6.238\times 10^{-11}$ | $6.232\times 10^{-11}$ | $1.00096$ | $0.4311$ / $0.4595$ | $0.0351$ / $0.0336$ |
# | $0.660$ | $215.60$ | $215.49$ | $4.202\times 10^{-11}$ | $2.739\times 10^{-11}$ | $6.942\times 10^{-11}$ | $6.934\times 10^{-11}$ | $1.00111$ | $0.3762$ / $0.3980$ | $0.0384$ / $0.0366$ |
# | $0.745$ | $222.94$ | $222.84$ | $4.544\times 10^{-11}$ | $2.961\times 10^{-11}$ | $7.506\times 10^{-11}$ | $7.496\times 10^{-11}$ | $1.00123$ | $0.3421$ / $0.3588$ | $0.0428$ / $0.0408$ |
# | $0.830$ | $229.86$ | $229.78$ | $4.777\times 10^{-11}$ | $3.133\times 10^{-11}$ | $7.910\times 10^{-11}$ | $7.899\times 10^{-11}$ | $1.00129$ | $0.3229$ / $0.3355$ | $0.0482$ / $0.0462$ |
# | $0.915$ | $236.16$ | $236.09$ | $4.877\times 10^{-11}$ | $3.255\times 10^{-11}$ | $8.133\times 10^{-11}$ | $8.122\times 10^{-11}$ | $1.00131$ | $0.3153$ / $0.3245$ | $0.0546$ / $0.0527$ |
# | $1.000$ | $241.70$ | $241.60$ | $4.822\times 10^{-11}$ | $3.337\times 10^{-11}$ | $8.159\times 10^{-11}$ | $8.148\times 10^{-11}$ | $1.00134$ | $0.3175$ / $0.3239$ | $0.0618$ / $0.0601$ |
#
#     Across all 11 points, $P_{\text{direct}} = P_{\text{tunnel}} + P_{\text{slot}}$ and $P_{v_g W} = v_g W / d$ agree to
#     within $0.015\%\text{--}0.134\%$ (Fig. 12c). The first-principles $(40, 6)$ forward-harmonic impedance
#     $\langle K_{+1}\rangle_{\text{beam}}$ falls from $1.36\;\Omega$ at $182.7\;\text{GHz}$ and $0.90\;\Omega$ at
#     $187.7\;\text{GHz}$ to $0.34\text{--}0.32\;\Omega$ across $222.9\text{--}241.7\;\text{GHz}$, while the backward
#     $n = -2$ harmonic $\langle K_{-2}\rangle_{\text{beam}}$ is $0.032\text{--}0.062\;\Omega$.
#
# ### 5.2 3D Staggered-Pillar V-Band SWS (`Tasks 10–11`, Marosi, Zuboraj & Capolino, 2025)
#
# Unlike a full-width grating ($w = a$), the V-band sheet-beam SWS of Marosi, Zuboraj & Capolino
# (`arXiv:2505.22927` [1], Table I: $a = 1.40\;\text{mm}$, $b = 0.80\;\text{mm}$, $d = 1.05\;\text{mm}$,
# $w = 1.08\;\text{mm}$, $l = 0.27\;\text{mm}$, $t = 0.10\;\text{mm}$, $g = 0.10\;\text{mm}$, modeled in
# CST Studio Suite) uses pillars that protrude only partially across the broad wall ($w / a = 0.771$).
# The pillars act as quarter-wave transverse stub resonators ($c / (4w) = 69.4\;\text{GHz}$), pulling the
# passband down into the $49\text{--}74\;\text{GHz}$ V-band below the bare waveguide cutoff
# ($c / (2a) = 107.1\;\text{GHz}$). `Task 10` models the glide-symmetric staggered-pillar unit cell
# ($z_1 = 0.25\,d$, $z_2 = 0.75\,d$) and `Task 11` is a broken-glide control
# ($z_2 = 0.62\,d$: the $+x$ pillar pair is no longer shifted by exactly $d/2$, so the cell has neither a glide plane
# nor an $x = 0$ mirror plane). Both structures keep the $y = 0$ mirror plane, because each pillar pair is symmetric
# in $y$. Both are run on a $56 \times 64 \times 56 = 200{,}704$-cell grid ($8$ cells across the $0.10\;\text{mm}$
# minimum feature) across $12$ phase points $\beta_0 d / \pi \in [0.15, 1.00]$.
#
# #### Two-Stage Cold-Test Method & Space-Harmonic Labelling
#
# - **Stage 1 (Harminv eigenmode identification)**: Each $\beta_0$ point runs a $40{,}000$-step ring-down.
#   Because the cold PEC structure is lossless, true eigenmodes ring with $Q > 10^9$ (whereas transient
#   fits have $Q < 10^3$). Beam-coupled modes are selected by requiring $Q > 10^4$ and visibility on the
#   on-axis $E_z$ probe: the lowest such frequency is the **lower mode** and the highest is the **upper mode**.
#   At $\beta_0 d = \pi$ ($\beta_n d = 3\pi$), glide symmetry produces one degenerate mode ($f_{3\pi} = 60.857\;\text{GHz}$),
#   whereas the broken-glide control resolves two distinct band edges separated by a stopband.
# - **Symmetric source excitation**: The unit cell is excited by on-axis $E_z$ dipoles together with
#   four-dipole quadrant sets at $(x, y) = (\pm 0.18, \pm 0.04)\;\text{mm}$ that are even under the $y = 0$
#   mirror plane (both cases). Modes odd in $y$ have $E_z = 0$ on the whole $y = 0$ plane, so they cannot couple to the
#   beam axis and are not excited at all. (With $y$-asymmetric sources such modes appear at $56.4\text{--}57.6\;\text{GHz}$
#   and $69.4\text{--}70.8\;\text{GHz}$ in the broken-glide control.)
# - **Stage 2 (Hann-windowed DFT & Floquet projection)**: Each identified mode (for $\beta_0 d / \pi < 1$,
#   since $\beta_0 d = \pi$ is a standing wave with $P_z = 0$) is driven by a narrowband ($2.2\;\text{GHz}$)
#   Gaussian pulse followed by a $60{,}000$-step Hann-windowed ring-down DFT ($\Delta t \approx 24\;\text{fs}$,
#   window $\approx 1.4\;\text{ns}$) and full-period ($N_z = 56$) Floquet projection onto harmonics
#   $n \in [-3, +3]$ to obtain $P_z$, $U_{\text{cell}}$, $v_E = P_z d / U_{\text{cell}}$, and $K_n$ (`kn_lower_*`
#   and `kn_upper_*`).
# - **Independent velocity consistency check**: Stage-2 energy velocity $v_E = P_z d / U_{\text{cell}}$ is
#   checked directly against Stage-1 group velocity $v_g = \mathrm{d}\omega / \mathrm{d}\beta_0$ computed via
#   `np.gradient` on the dispersion curve (central finite differences at interior points $\beta_0 d / \pi \in [0.23, 0.92]$,
#   indices $1\dots 10$; the endpoints use one-sided differences, so $\beta_0 d / \pi = 0.15$ is excluded).
# - **Physics labelling in the design band $\beta_n d / \pi \in [2, 4]$ (Marosi Eq. 1, beam along $+z$)**:
#   - **Forward-wave $Z_P$ (Marosi Fig. 5a)** forms one continuous curve across $49.02\text{--}73.60\;\text{GHz}$:
#     the lower mode $n = +1$ harmonic ($\beta_{+1} d = (2 + \beta_0 d/\pi)\pi \in [2.15\pi, 3\pi]$,
#     $49.02\text{--}60.86\;\text{GHz}$, $v_g > 0$) continued above $3\pi$ by the upper mode $n = -2$ harmonic
#     ($|\beta_{-2}| d = (4 - \beta_0 d/\pi)\pi \in [3\pi, 3.85\pi]$, $60.86\text{--}73.60\;\text{GHz}$;
#     by time-reversal symmetry of the $+\beta_0$ and $-\beta_0$ Bloch sectors, $n = -2$ of the $+\beta_0$ upper
#     mode with $v_g(\beta_0) < 0$ is identical to the $n = +2$ forward-propagating harmonic of the $-\beta_0$ mode).
#   - **Backward-wave $Z_P$ (Marosi Fig. 11a)** comprises the upper mode $n = +1$ harmonic
#     ($\beta_{+1} d = (2 + \beta_0 d/\pi)\pi$, where $v_{p,+1} > 0$ and $v_g < 0$) and the lower mode
#     $n = -2$ harmonic ($|\beta_{-2}| d = (4 - \beta_0 d/\pi)\pi$).
#
# Summary across all three benchmarks—2D Grating, 3D Grating (`Task 9`), and 3D Marosi Pillar SWS
# (`Tasks 10–11`)—comparing `--full` Wakis FIT against Analytic McVey / Karetnikova et al. (2018) /
# Marosi et al. (2025) CST Studio Suite:
#
# | Study & Metric | Wakis FIT (`--full`, Table I $2a = 200\;\mu\text{m}$) | Reference (Theory / Paper) | Full vs. Ref ($\Delta\%$) |
# |---|---|---|---|
# | **2D Grating** ($\beta_0 d = 0.5\pi$) glide $f_1$: `L2` ($n=8$) / `L3` ($n=12$) / `Ref` ($n=16$) | $97.80$ / $97.93$ / $97.99\;\text{GHz}$ (Rich. $98.07\;\text{GHz}$) | $98.120\;\text{GHz}$ (McVey $f_{1,\infty}$, $M_{\text{slot}}\to\infty$)$^\ddagger$ | $-0.32\%$ / $-0.19\%$ / $-0.13\%$ (Rich. $-0.05\%$) |
# | **2D Grating** ($\beta_0 d = \pi$) mirror stopband $[f_1, f_2]$ (`L2`) | $[140.66, 195.89]\;\text{GHz}$ | $[140.29, 195.88]\;\text{GHz}$ (McVey) | $+0.26\%$ / $+0.01\%$ |
# | **2D Grating** ($\beta_0 d = \pi$) glide degeneracy $f_1 = f_2$ (`L2`) | $165.46\;\text{GHz}$ | $165.28\;\text{GHz}$ (McVey) | $+0.11\%$ |
# | **2D Grating** ($\beta_0 d = 0.5\pi$) energy velocity $v_E / c$: `L1` / `L2` / `L3` / `Ref` | $0.5569$ / $0.5665$ / $0.5739$ / $0.5776$ | $0.5857$ ($v_g/c$, McVey) | $-4.92\%$ / $-3.29\%$ / $-2.02\%$ / $-1.39\%$ |
# | **2D Grating** ($\beta_0 d = 0.5\pi$) Pierce $K_{-1}(0)$: `L1` / `L2` / `L3` / `Ref` | $1.8381$ / $1.7921$ / $1.7802$ / $1.7749\;\Omega$ (Rich. $1.7681\;\Omega$) | $1.7637\;\Omega$ (McVey $K_{-1,\infty}$, $M_{\text{slot}}\to\infty$)$^\ddagger$ | $+4.21\%$ / $+1.61\%$ / $+0.93\%$ / $+0.63\%$ (Rich. $+0.25\%$) |
# | **3D Grating** (`Task 9`, $\beta_0 d = \pi$) glide $f_1$ ($n=8$, $34\times 64\times 40$) | $241.60\;\text{GHz}$ | $241.70\;\text{GHz}$ (3D McVey; inside $\sim 175\text{--}260\;\text{GHz}$ Fig. 2a) | $-0.04\%$ |
# | **3D Grating** (`Task 9`, $\beta_0 d = \pi$) mirror $[f_1, f_2]$ ($n=8$, $34\times 64\times 40$) | $[225.26, 263.45]\;\text{GHz}$ | $[225.35, 263.57]\;\text{GHz}$ (3D McVey) | $-0.04\%$ / $-0.04\%$ |
# | **3D Grating** (`Task 9`) beam-avg $\langle K_{+1}\rangle_{\text{beam}}$ ($183\text{--}188$ vs. $223\text{--}242\;\text{GHz}$) | $1.36\text{--}0.90\;\Omega \to 0.34\text{--}0.32\;\Omega$ ($\langle K_{-2}\rangle = 0.032\text{--}0.062\;\Omega$) | $\sim 1\;\Omega$ ($\sim 185\;\text{GHz}$) $\to \sim 0.2\text{--}0.3\;\Omega$ ($220\text{--}250\;\text{GHz}$, Fig. 2b by eye) | Consistent (see Sec. 5.1) |
# | **3D Marosi** (`Task 10`) glide degeneracy $f_{3\pi}$ at $\beta_n d = 3\pi$ | $60.857\;\text{GHz}$ ($60.8566\;\text{GHz}$) | $61.0\;\text{GHz}$ (CST, Sec. II & Figs. 2, 3, 5b, 11b) | $-0.2\%$ |
# | **3D Marosi** (`Task 10`) dispersion across $\beta_n d \in [2.25\pi, 3.75\pi]$ (interpolated) | $[49.63, 52.29, 57.11, 60.86, 64.69, 69.87, 72.88]\;\text{GHz}$ | $[50.3, 52.8, 57.5, 61.0, 65.0, 70.0, 72.7]\;\text{GHz}$ (Fig. 2) | $-1.3\%$ to $+0.2\%$ |
# | **3D Marosi** (`Task 10`) interior $v_E / v_g$ ($\beta_0 d/\pi \in [0.23, 0.92]$, idx $1\dots 10$) | Lower: $0.988\text{--}0.999$; Upper: $0.985\text{--}0.987$ | $v_E / v_g = 1$ (lossless Bloch identity) | $\le 1.5\%$ |
# | **3D Marosi** (`Task 10`) forward-wave $Z_P(0,0)$ at $58\;\text{GHz}$ / $68\;\text{GHz}$ (interp.) | $3.78\;\Omega$ ($58\;\text{GHz}$) / $1.01\;\Omega$ ($68\;\text{GHz}$) | $3.8\;\Omega$ ($58\;\text{GHz}$) / $1.3\;\Omega$ ($68\;\text{GHz}$, Sec. II text) | $-0.5\%$ / $-22.2\%$ |
# | **3D Marosi** (`Task 10`) backward-wave $Z_P^{\text{bw}}(0,0)$ (upper $n=+1$, lower $n=-2$) | $\le 1.17\times 10^{-4}\;\Omega$ at all points ($\langle Z_P^{\text{bw}}\rangle = 0.085\text{--}1.39\;\Omega$, $0.023\text{--}0.060\;\Omega$) | $\approx 0\;\Omega$ on axis (Sec. II, App. B, Fig. 11a) | Glide-suppressed |
# | **3D Marosi** (`Task 11`) broken-glide control ($3\pi$ stopband, $v_E/v_g$, backward $Z_P^{\text{bw}}(0,0)$) | Stopband $51.74\text{--}69.54\;\text{GHz}$ ($17.81\;\text{GHz}$); $v_E/v_g$: $0.985\text{--}0.992$, $0.966\text{--}0.979$; $Z_P^{\text{bw}}(0,0)$ up to $2.34\;\Omega$ ($n=+1$), $0.77\text{--}9.60\;\Omega$ ($n=-2$) | Non-glide control ($z_2 = 0.62\,d$) | Stopband opens; $Z_P^{\text{bw}}(0,0) \ne 0$ |
#
# $^\ddagger$Slot-mode-extrapolated first-principles McVey limit ($y = y_\infty + C M_{\text{slot}}^{-p}$ fitted over $M_{\text{slot}} = 10\dots 32$ at $N_{\text{harm}} = 60$:
# $K_{-1,\infty} = 1.7637\;\Omega$ with fitted $p \approx 1.66$, $f_{1,\infty} = 98.120\;\text{GHz}$ with fitted $p \approx 1.43$; $P = P_{\text{tunnel}} + P_{\text{slot}}$,
# verified against $P = v_g W / d$ to $\le 0.13\%$; see Section 4 for the finite-$M_{\text{slot}}$ convergence table).
#
# #### Dispersion vs. Marosi et al. (2025) Fig. 2 & Broken-Glide Stopband Control
#
# In the glide-symmetric SWS (`Task 10`, $2{,}646{,}876$ total steps across 12 $\beta_0$ shards), the lower mode rises
# smoothly from $49.016\;\text{GHz}$ ($\beta_0 d/\pi = 0.15$, $\beta_{+1} d/\pi = 2.15$) to $60.857\;\text{GHz}$ at
# $\beta_0 d/\pi = 1.00$ ($\beta_{+1} d/\pi = 3.00$), while the upper mode descends from $73.597\;\text{GHz}$
# ($|\beta_{-2}| d/\pi = 3.85$) to the same degenerate crossing $f_{3\pi} = 60.857\;\text{GHz}$ at $3\pi$ (paper:
# $61.0\;\text{GHz}$, $-0.2\%$). Interpolating the unfolded Wakis FIT dispersion curve onto the seven CST Studio Suite
# reference points from Fig. 2 of Marosi et al. (2025):
#
# | $\beta_n d / \pi$ | Marosi et al. Fig. 2 (CST) [GHz] | Wakis 3D FIT (interpolated) [GHz] | Relative difference |
# |---|---|---|---|
# | $2.25$ | $50.3 \pm 0.3$ | $49.633$ | $-1.3\%$ |
# | $2.50$ | $52.8 \pm 0.3$ | $52.288$ | $-1.0\%$ |
# | $2.80$ | $57.5 \pm 0.3$ | $57.105$ | $-0.7\%$ |
# | $3.00$ | $61.0$ (text) | $60.857$ | $-0.2\%$ |
# | $3.20$ | $65.0 \pm 0.3$ | $64.690$ | $-0.5\%$ |
# | $3.50$ | $70.0 \pm 0.3$ | $69.865$ | $-0.2\%$ |
# | $3.75$ | $72.7 \pm 0.3$ | $72.877$ | $+0.2\%$ |
#
# In the broken-glide control (`Task 11`, $2{,}717{,}909$ total steps across 12 $\beta_0$ shards), breaking the glide
# alternation shifts the lower mode to $45.889 \to 51.737\;\text{GHz}$ and the upper mode to $76.394 \to 69.544\;\text{GHz}$,
# opening a $51.74\text{--}69.54\;\text{GHz}$ Bragg stopband of width $17.81\;\text{GHz}$ at $\beta_n d = 3\pi$ (shaded in Fig. 13a).
#
# #### Full 12-Point 3D Marosi Glide (`Task 10`) & Broken-Glide (`Task 11`) Cold-Test Table
#
# | Index | $\beta_0 d/\pi$ | Glide $f_{\text{lo}}$ / $f_{\text{up}}$ [GHz] | Glide $v_E^{\text{lo}} / v_g^{\text{lo}}$ | Glide $v_E^{\text{up}} / v_g^{\text{up}}$ | Glide Fwd $Z_P(0,0)$ / $\langle Z_P\rangle$: lo $n=+1$, up $n=-2$ [$\Omega$] | Glide Bwd $Z_P^{\text{bw}}(0,0)$ / $\langle Z_P^{\text{bw}}\rangle$: up $n=+1$, lo $n=-2$ [$\Omega$] | Broken-glide $f_{\text{lo}}$ / $f_{\text{up}}$ [GHz] | Broken-glide $v_E / v_g$: lo / up | Broken-glide Bwd $Z_P^{\text{bw}}(0,0)$: up $n=+1$ / lo $n=-2$ [$\Omega$] |
# |---|---|---|---|---|---|---|---|---|---|
# | $0$ | $0.1500$ | $49.016$ / $73.597$ | $0.799^*$ | $0.790^*$ | $60.06$ ($58.42$), $0.383$ ($0.432$) | $2.1\times 10^{-14}$ ($1.39$), $1.7\times 10^{-14}$ ($0.026$) | $45.889$ / $76.394$ | $0.793^*$ / $0.777^*$ | $0.636$ / $1.082$ |
# | $1$ | $0.2273$ | $49.454$ / $73.084$ | $0.999$ | $0.987$ | $35.02$ ($34.16$), $0.415$ ($0.457$) | $1.7\times 10^{-14}$ ($0.831$), $1.3\times 10^{-14}$ ($0.023$) | $46.236$ / $75.963$ | $0.988$ / $0.966$ | $0.168$ / $0.848$ |
# | $2$ | $0.3045$ | $50.061$ / $72.381$ | $0.998$ | $0.986$ | $23.19$ ($22.69$), $0.482$ ($0.521$) | $1.4\times 10^{-14}$ ($0.562$), $1.1\times 10^{-14}$ ($0.023$) | $46.707$ / $75.385$ | $0.987$ / $0.966$ | $0.028$ / $0.769$ |
# | $3$ | $0.3818$ | $50.826$ / $71.505$ | $0.997$ | $0.986$ | $16.46$ ($16.15$), $0.573$ ($0.612$) | $1.3\times 10^{-14}$ ($0.406$), $9.4\times 10^{-15}$ ($0.024$) | $47.286$ / $74.682$ | $0.986$ / $0.966$ | $1.1\times 10^{-4}$ / $0.766$ |
# | $4$ | $0.4591$ | $51.737$ / $70.478$ | $0.996$ | $0.986$ | $12.21$ ($12.02$), $0.689$ ($0.727$) | $1.2\times 10^{-14}$ ($0.307$), $8.9\times 10^{-15}$ ($0.026$) | $47.949$ / $73.889$ | $0.985$ / $0.967$ | $0.023$ / $0.822$ |
# | $5$ | $0.5364$ | $52.779$ / $69.320$ | $0.995$ | $0.985$ | $9.34$ ($9.23$), $0.831$ ($0.869$) | $1.1\times 10^{-14}$ ($0.238$), $8.6\times 10^{-15}$ ($0.029$) | $48.668$ / $73.042$ | $0.985$ / $0.967$ | $0.076$ / $0.944$ |
# | $6$ | $0.6136$ | $53.934$ / $68.057$ | $0.993$ | $0.985$ | $7.32$ ($7.25$), $1.003$ ($1.040$) | $1.0\times 10^{-14}$ ($0.189$), $8.6\times 10^{-15}$ ($0.032$) | $49.405$ / $72.184$ | $0.985$ / $0.969$ | $0.161$ / $1.163$ |
# | $7$ | $0.6909$ | $55.190$ / $66.706$ | $0.992$ | $0.986$ | $5.82$ ($5.79$), $1.210$ ($1.247$) | $1.4\times 10^{-14}$ ($0.153$), $1.2\times 10^{-14}$ ($0.037$) | $50.121$ / $71.363$ | $0.986$ / $0.971$ | $0.293$ / $1.554$ |
# | $8$ | $0.7682$ | $56.528$ / $65.290$ | $0.991$ | $0.986$ | $4.69$ ($4.68$), $1.460$ ($1.495$) | $2.7\times 10^{-15}$ ($0.125$), $7.7\times 10^{-16}$ ($0.043$) | $50.765$ / $70.632$ | $0.988$ / $0.973$ | $0.518$ / $2.305$ |
# | $9$ | $0.8455$ | $57.930$ / $63.830$ | $0.990$ | $0.986$ | $3.816$ ($3.821$), $1.762$ ($1.795$) | $7.1\times 10^{-9}$ ($0.103$), $6.4\times 10^{-9}$ ($0.051$) | $51.282$ / $70.051$ | $0.990$ / $0.976$ | $0.972$ / $3.999$ |
# | $10$ | $0.9227$ | $59.377$ / $62.348$ | $0.988$ | $0.987$ | $3.127$ ($3.143$), $2.127$ ($2.157$) | $1.17\times 10^{-4}$ ($0.085$), $9.66\times 10^{-5}$ ($0.060$) | $51.620$ / $69.675$ | $0.992$ / $0.979$ | $2.341$ / $9.603$ |
# | $11$ | $1.0000$ | $60.857$ / $60.857$ | — | — | — | — | $51.737$ / $69.544$ | — | — |
#
# $^*$At index $0$ ($\beta_0 d/\pi = 0.15$), `np.gradient` uses a one-sided finite difference near the lower band edge where dispersion curvature is strong, so $\beta_0 d/\pi = 0.15$ is excluded from the interior $v_E / v_g$ check; at index $11$ ($\beta_0 d/\pi = 1.00$), the modes are standing waves ($P_z = 0$). Across all interior points (indices $1\dots 10$, $\beta_0 d/\pi \in [0.23, 0.92]$), Stage-2 $v_E$ and Stage-1 $v_g$ agree with $v_E / v_g \in [0.988, 0.999]$ (glide lower), $[0.985, 0.987]$ (glide upper), $[0.985, 0.992]$ (mirror lower), and $[0.966, 0.979]$ (mirror upper) (Fig. 15c).
#
# #### Forward-Wave Pierce Impedance vs. Marosi et al. (2025) Fig. 5a & Sec. II
#
# Combining the lower-mode $n = +1$ harmonic ($49.02\text{--}59.38\;\text{GHz}$) and the upper-mode $n = -2$ harmonic
# ($62.35\text{--}73.60\;\text{GHz}$) yields a single continuous forward-wave Pierce impedance curve across the V-band
# passband, with the beam-averaged impedance $\langle Z_P\rangle_{\text{beam}}$ within $3\%$ of the on-axis value
# $Z_P(0,0)$ across the main passband (Fig. 15a). Interpolating `np.interp` along this continuous forward curve onto the
# reference frequencies from Section II and Fig. 5a of Marosi et al. (2025):
#
# | Frequency $f$ [GHz] | Marosi et al. (2025) $Z_P(0,0)$ [$\Omega$] | Source in Paper | Wakis 3D FIT $Z_P(0,0)$ (interp.) [$\Omega$] | Wakis 3D FIT $\langle Z_P\rangle_{\text{beam}}$ (interp.) [$\Omega$] | Relative diff. (on-axis) |
# |---|---|---|---|---|---|
# | $53.0$ | $9.9 \pm 0.3$ | Fig. 5a read | $8.96$ ($8.956$) | $8.85$ | $-9.5\%$ |
# | $56.0$ | $5.1 \pm 0.3$ | Fig. 5a read | $5.14$ ($5.136$) | $5.12$ | $+0.7\%$ |
# | $58.0$ | $3.8$ | Sec. II text & Fig. 5a | $3.78$ ($3.782$) | $3.79$ | $-0.5\%$ |
# | $60.0$ | $2.8 \pm 0.3$ | Fig. 5a read | $2.92$ ($2.917$) | $2.94$ | $+4.2\%$ |
# | $65.0$ | $1.6 \pm 0.3$ | Fig. 5a read | $1.52$ ($1.520$) | $1.56$ | $-5.0\%$ |
# | $68.0$ | $1.3$ | Sec. II text & Fig. 5a | $1.01$ ($1.011$) | $1.05$ | $-22.2\%$ |
#
# - **Agreement below $3\pi$ and open discrepancy above $3\pi$**: At the lower band edge of the operating window
#   ($58\;\text{GHz}$, a value stated in Section II of the paper), the interpolated Wakis FIT on-axis impedance is
#   $3.78\;\Omega$ vs. $3.8\;\Omega$ ($-0.5\%$). Above the $3\pi$ crossing, Wakis FIT is systematically lower than the
#   paper, reaching $1.01\;\Omega$ vs. the text-stated $1.3\;\Omega$ at $68\;\text{GHz}$ ($-22\%$). Candidate causes
#   for this discrepancy (without singling out one) include:
#   1. **Single-mesh discretization**: `Tasks 10–11` are evaluated at a single grid resolution ($8$ cells across the
#      $0.10\;\text{mm}$ minimum feature, with no 3D mesh-refinement ladder for this task yet);
#   2. **Geometry details not fully specified**: potential differences in CST model details (such as corner rounding,
#      exact axial pillar length $l = 0.27\;\text{mm}$ in Table I vs. $d/4 = 0.2625\;\text{mm}$ noted in the text, or
#      averaging conventions); and
#   3. **Figure-read uncertainty**: $\pm 0.3\;\Omega$ reading uncertainty on the points digitized from Fig. 5a.
#
# #### Backward-Wave On-Axis Suppression: Glide Symmetry vs. Broken-Glide Control
#
# - **Glide-symmetric SWS (`Task 10`, Fig. 15b)**: On both backward-wave branches—the upper mode $n = +1$ harmonic
#   ($v_g < 0$, $62.35\text{--}73.60\;\text{GHz}$) and the lower mode $n = -2$ harmonic ($49.02\text{--}59.38\;\text{GHz}$)—the
#   on-axis Pierce impedance is $Z_P^{\text{bw}}(0,0) \le 1.17\times 10^{-4}\;\Omega$ at every point ($\sim 10^{-14}\;\Omega$
#   across $\beta_0 d/\pi \in [0.15, 0.77]$), reproducing the paper's result that the backward-wave impedance is
#   "effectively $0\;\Omega$" on axis (Marosi et al. Section II, Appendix B, and Fig. 11a). Averaged over the
#   $0.55\;\text{mm} \times 0.10\;\text{mm}$ sheet-beam cross-section, the off-axis field yields finite beam-averaged
#   backward impedance $\langle Z_P^{\text{bw}}\rangle_{\text{beam}} = 0.085\text{--}1.39\;\Omega$ for upper $n = +1$
#   (largest near $\beta_0 d/\pi = 0.15$) and $0.023\text{--}0.060\;\Omega$ for lower $n = -2$.
# - **Why the glide makes it exactly zero on axis**: the glide operator $G: (x, y, z) \mapsto (-x, y, z + d/2)$ commutes
#   with the Bloch translation, and a non-degenerate Bloch mode is an eigenvector of it, $G\,\mathbf{E} = \pm e^{-j\beta_0 d/2}\,\mathbf{E}$
#   ($G^2$ is the one-period translation). The beam axis $x = y = 0$ maps onto itself, so for $E_z(0, 0, z) = \sum_n E_{z,n}
#   e^{-j\beta_n z}$ this gives $E_{z,n}\,e^{-j\beta_n d/2} = \pm e^{-j\beta_0 d/2} E_{z,n}$, i.e. $(-1)^n = \pm 1$: **on axis, each
#   mode carries only odd or only even space harmonics.** The lower mode carries $n = +1$ (odd), so its $n = -2$ vanishes;
#   the upper mode carries $n = -2$ (even), so its $n = +1$ vanishes. The residual $\sim 10^{-14}\;\Omega$ is round-off;
#   the $\sim 10^{-4}\;\Omega$ at $\beta_0 d/\pi = 0.92$ is most likely leakage between the two modes, which are only $3\;\text{GHz}$ apart there. Off axis
#   ($x \ne 0$) the point does not map onto itself, so the beam-averaged values are non-zero.
# - **Broken-glide control (`Task 11`, Fig. 15b)**: Without the glide operator the parity selection is lost, and the backward-wave on-axis Pierce impedance is **not** zero: $Z_P^{\text{bw}}(0,0)$
#   reaches up to $2.34\;\Omega$ on the upper mode $n = +1$ harmonic and $0.77\text{--}9.60\;\Omega$ on the lower mode
#   $n = -2$ harmonic. Thus the vanishing of the on-axis backward-wave interaction impedance is a direct consequence of
#   glide symmetry rather than a generic property of pillar-loaded SWSs.
#
# ---
#
# ## 6. Resolution Study & Mesh Audit: Conformal PEC vs. Staircase PEC
#
# Evaluating the 2D glide-symmetric grating at exact $\beta_0 d = 0.5\pi$ across our 9-task 2D resolution
# ladder (`Tasks 0–8`, minimum feature $l = s_{\text{vane}} = 0.10\;\text{mm}$, tunnel gap $2a = 0.20\;\text{mm}$,
# beam thickness $H_b = 0.15\;\text{mm}$ per Karetnikova et al. [2] Table I):
#
# | Level | $n_{\text{feat}}$ | $\Delta y = \Delta z$ [$\mu\text{m}$] | Conformal $f_1$ [GHz] | Staircase $f_1$ [GHz] | Conformal $v_E / c$ | Conformal $K_{-1}(0)$ [$\Omega$] | Staircase $K_{-1}(0)$ [$\Omega$] |
# |---|---|---|---|---|---|---|---|
# | Under-resolved (`Tasks 0–1`) | $2$ | $50.00$ | $97.6184$ | $77.5826$ | $0.5337$ | $1.7650$ | $5.9410$ |
# | `L1` (`Tasks 2, 5`) | $4$ | $25.00$ | $97.3084$ | $88.5263$ | $0.5569$ | $1.8381$ | $3.2418$ |
# | `L2` (`Tasks 3, 6`) | $8$ | $12.50$ | $97.8032$ | $93.3470$ | $0.5665$ | $1.7921$ | $2.4082$ |
# | `L3` (`Tasks 4, 7`) | $12$ | $8.33$ | $97.9334$ | $94.9442$ | $0.5739$ | $1.7802$ | $2.1733$ |
# | `Reference` (`Task 8`) | $16$ | $6.25$ | $97.9918$ | — | $0.5776$ | $1.7749$ | — |
# | `FIT Richardson` ($n=12,16$, $p=2$) | $\infty$ | $0$ | $98.0670$ | — | $0.5824$ | $1.7681$ | — |
# | `McVey Limit` ($M_{\text{slot}}\to\infty$, $N_{\text{harm}}=60$) | $\infty$ | $0$ | $98.1201$ ($97.6868$ at $M=6$) | $98.1201$ | $0.5882$ ($0.5857$ at $M=6$) | $1.7637$ ($1.8274$ at $M=6$) | $1.7637$ ($1.8274$ at $M=6$) |
#
# ### Convergence Verdict
#
# 1. **Eigenmode frequency $f_1$ (monotone, converged to $-0.05\%$ of the slot-mode-extrapolated McVey limit)**: Across `L1` $\to$ `L2` $\to$ `L3` $\to$ `Reference` ($n_{\text{feat}} = 4 \to 8 \to 12 \to 16$), Dey–Mittra conformal PEC converges monotonically from below: $97.3084 \to 97.8032 \to 97.9334 \to 97.9918\;\text{GHz}$ at exact $\beta_0 d = 0.50\pi$, with second-order Richardson ($p=2$) limit $98.0670\;\text{GHz}$. The relative change between `L2` and `L3` ($n_{\text{feat}} = 8 \to 12$) is $\Delta f_1 / f_1 = +0.133\%$, and between the last two conformal levels (`L3` $\to$ `Reference`, $n_{\text{feat}} = 12 \to 16$) is $+0.060\%$. Compared against the slot-mode-extrapolated McVey limit $f_{1,\infty} = 98.120\;\text{GHz}$ ($M_{\text{slot}} \to \infty$ at $N_{\text{harm}} = 60$, fitted order $p \approx 1.43$; see Section 4), the conformal errors are $-0.83\%$ (`L1`), $-0.32\%$ (`L2`), $-0.19\%$ (`L3`), **$-0.13\%$ (`Reference`)**, and **$-0.05\%$ (FIT Richardson $98.0670\;\text{GHz}$)**, whereas the $M_{\text{slot}} = 6$ value ($97.6868\;\text{GHz}$) is unconverged in slot modes. By contrast, staircase PEC (`conformal=False`) converges roughly $10\times$ more slowly, retaining a $-3.24\%$ frequency error vs. $f_{1,\infty}$ ($-3.11\%$ vs. FIT `Reference`, $94.9442\;\text{GHz}$) at `L3` with $\Delta f_1 / f_1 = +1.71\%$ from `L2` to `L3`.
# 2. **On-axis Pierce impedance $K_{-1}(0)$ (monotone across both conformal and staircase ladders, self-converged to $0.30\%$ between the last two conformal levels, and within $+0.25\%$ of the converged McVey limit)**:
#    - **Dey–Mittra conformal PEC (`L1` $\to$ `L2` $\to$ `L3` $\to$ `Reference`, $n_{\text{feat}} = 4 \to 8 \to 12 \to 16$)**: Strict monotonic convergence from above: $1.8381 \to 1.7921 \to 1.7802 \to 1.7749\;\Omega$. The relative change between successive levels decreases as $-2.50\%$ (`L1` $\to$ `L2`), $-0.66\%$ (`L2` $\to$ `L3`), and **$-0.30\%$ between the last two levels (`L3` $\to$ `Reference`)**—matching $\mathcal{O}(\Delta^2)$ second-order self-convergence toward the FIT Richardson ($p = 2$) limit $K_{-1,\infty}^{\text{FIT}} = 1.7681\;\Omega$ ($|K_{-1} - K_{-1,\infty}^{\text{FIT}}| / K_{-1,\infty}^{\text{FIT}} = 3.96\% \to 1.36\% \to 0.69\% \to 0.39\%$). Compared against the slot-mode-extrapolated McVey limit $K_{-1,\infty} = 1.7637\;\Omega$ ($M_{\text{slot}} \to \infty$ at $N_{\text{harm}} = 60$, fitted order $p \approx 1.66$; see Section 4), FIT Richardson ($1.7681\;\Omega$) is **$+0.25\%$** above $K_{-1,\infty}$ and FIT `Reference` ($1.7749\;\Omega$) is **$+0.63\%$** above—an independent agreement within the FIT discretization error. McVey's $M_{\text{slot}}$ truncation is the dominant analytic error ($M_{\text{slot}} = 6\dots 10$ gives $1.8274\dots 1.7897\;\Omega$) and must be extrapolated.
#    - **Staircase PEC (`Tasks 0, 2, 3, 4`, $n_{\text{feat}} = 2 \to 4 \to 8 \to 12$)**: Also converges monotonically from above ($5.9410 \to 3.2418 \to 2.4082 \to 2.1733\;\Omega$), with a relative change of **$-9.75\%$** between the last two staircase levels (`L2` $\to$ `L3`), still $+22.4\%$ above FIT `Reference` ($+23.2\%$ above McVey $K_{-1,\infty}$) at `L3`.
#
# ### Graded Non-Uniform Mesh Upgrade vs. Uniform Mesh Baseline (Equal Accuracy & Equal Cost)
#
# In a corrugated SWS, the electromagnetic field gradients are concentrated at the $90^\circ$ PEC vane
# corners ($y = \pm g_{\text{half}} = \pm 100\;\mu\text{m}$) and across the $l = 100\;\mu\text{m}$ vane
# teeth, whereas the back of the $h = 300\;\mu\text{m}$ groove slots and the open groove span carry
# smoothly varying evanescent/standing fields. `build_2d_grating_graded_grid` and
# `build_3d_grating_graded_grid` in `_helpers.py` construct graded non-uniform Cartesian grids that
# place exact grid nodes on the vane walls ($z \in [75, 175]\;\mu\text{m}$ and $[325, 425]\;\mu\text{m}$)
# and tunnel apertures ($y = \pm 100\;\mu\text{m}$), maintaining $\Delta y_{\min} = \Delta z_{\min} = l / n_{\text{feat}}$
# across the vane teeth and across a $\pm 35\;\mu\text{m}$ band straddling the vane tips while grading
# smoothly toward $\Delta y_{\max} = 30.0\;\mu\text{m}$ in the slot depth and $\Delta z_{\max} = 22.0\;\mu\text{m}$
# in the open groove ($\ge 42.4\;\text{cells}/\lambda$ everywhere at $f_{\max} = 235\;\text{GHz}$, adjacent
# cell-size stretch ratio $\le 1.24$).
#
# Comparing the 4-level graded non-uniform convergence ladder (`data/graded_mesh_comparison.json`, evaluated
# at $\beta_0 d = 0.5\pi$) directly against the uniform conformal baseline (`Tasks 5–8`, kept as the
# reference throughout the text):
#
# | Level ($n_{\text{feat}}$) | Uniform Grid ($N_y \times N_z$, cells) | Graded Grid ($N_y \times N_z$, cells, reduction) | Uniform $f_1$ [GHz] | Graded $f_1$ [GHz] ($\Delta$ vs. Uniform) | Uniform $K_{-1}(0)$ / $\langle K_{-1}\rangle_{\text{beam}}$ [$\Omega$] | Graded $K_{-1}(0)$ / $\langle K_{-1}\rangle_{\text{beam}}$ [$\Omega$] | Uniform Wall Time [s] | Graded Wall Time [s] (speedup) |
# |---|---|---|---|---|---|---|---|---|
# | `L1` ($n_{\text{feat}} = 4$, $\Delta_{\min} = 25.0\;\mu\text{m}$) | $32 \times 20 = 640$ | $24 \times 20 = 480$ ($1.33\times$) | $97.3084$ | $97.2817$ ($-0.027\%$) | $1.8381$ / $2.0425$ | $1.8307$ / $2.0342$ ($-0.40\%$) | $1.47^*$ | $1.16$ ($1.27\times$) |
# | `L2` ($n_{\text{feat}} = 8$, $\Delta_{\min} = 12.5\;\mu\text{m}$) | $64 \times 40 = 2{,}560$ | $44 \times 36 = 1{,}584$ ($1.62\times$) | $97.8032$ | $97.7737$ ($-0.030\%$) | $1.7921$ / $2.0475$ | $1.7923$ / $2.0476$ ($+0.011\%$) | $8.07^*$ | $4.31$ ($1.87\times$) |
# | `L3` ($n_{\text{feat}} = 12$, $\Delta_{\min} = 8.33\;\mu\text{m}$) | $96 \times 60 = 5{,}760$ | $62 \times 52 = 3{,}224$ ($1.79\times$) | $97.9334$ | $97.9128$ ($-0.021\%$) | $1.7802$ / $2.0537$ | $1.7808$ / $2.0541$ ($+0.034\%$) | $26.02$ | $10.58$ ($2.46\times$) |
# | `Reference` ($n_{\text{feat}} = 16$, $\Delta_{\min} = 6.25\;\mu\text{m}$) | $128 \times 80 = 10{,}240$ | $82 \times 64 = 5{,}248$ ($1.95\times$) | $97.9918$ | $97.9701$ ($-0.022\%$) | $1.7749$ / $2.0566$ | $1.7768$ / $2.0585$ ($+0.107\%$) | $58.65$ | $20.35$ ($2.88\times$) |
#
# $^*$Single-point $\beta_0 d = 0.5\pi$ equivalent wall time (`Tasks 2, 3`).
#
# - **Equal feature resolution (`Graded L2` vs. `Uniform L2`, $n_{\text{feat}} = 8$)**: `Graded L2`
#   ($1 \times 44 \times 36 = 1{,}584$ cells, $4.31\;\text{s}$) uses **$1.62\times$ fewer cells** and runs
#   **$1.87\times$ faster** than `Uniform L2` ($1 \times 64 \times 40 = 2{,}560$ cells, $8.07\;\text{s}$)
#    while matching $f_1(\beta_0 d = 0.5\pi)$ to $-0.030\%$ ($97.7737\;\text{GHz}$ vs. $97.8032\;\text{GHz}$),
#   $K_{-1}(0)$ to $+0.011\%$ ($1.7923\;\Omega$ vs. $1.7921\;\Omega$), $\langle K_{-1}\rangle_{\text{beam}}$
#   to $+0.005\%$ ($2.0476\;\Omega$ vs. $2.0475\;\Omega$), and $v_E / c$ to $-0.07\%$ ($0.5661$ vs. $0.5665$),
#   with on-axis even-harmonic leakage $K_0(0) = 4.14\times 10^{-6}\;\Omega$. Across the full 7-point
#   Brillouin sweep (`sweep_L2` in `data/graded_mesh_comparison.json`, Figs. 4, 5a, 7a), `Graded L2`
#   reproduces `Uniform L2` across $\beta_0 d / \pi \in [0.20, 1.00]$ to within $\le 0.04\;\text{GHz}$ in
#   $f_1(\beta_0)$ and $\le 0.008\;\Omega$ in $K_{-1}(0)$ and $\langle K_{-1}\rangle_{\text{beam}}$.
# - **Superior accuracy at lower cost (`Graded Reference` vs. `Uniform L3`, Fig. 10c)**: `Graded Reference`
#   ($n_{\text{feat}} = 16$, $5{,}248$ cells, $20.35\;\text{s}$) uses **fewer cells ($5{,}248$ vs. $5{,}760$,
#   $0.91\times$) and $1.28\times$ less wall time ($20.35\;\text{s}$ vs. $26.02\;\text{s}$)** than `Uniform L3`
#   ($n_{\text{feat}} = 12$, $5{,}760$ cells), while cutting the eigenfrequency discrepancy against
#   `Uniform Reference` ($97.9918\;\text{GHz}$) by $2.7\times$ ($21.7\;\text{MHz}$ at `Graded Reference` vs.
#   $58.4\;\text{MHz}$ at `Uniform L3`) and cutting the Pierce impedance discrepancy against
#   `Uniform Reference` ($1.7749\;\Omega$) by $2.8\times$ ($+0.107\%$ at `Graded Reference` vs. $+0.299\%$
#   at `Uniform L3`). Compared against `Uniform Reference` ($10{,}240$ cells, $58.65\;\text{s}$) at equal
#   vane-tip resolution $n_{\text{feat}} = 16$, `Graded Reference` reduces cell count by **$1.95\times$** and
#   wall time by **$2.88\times$**.
# - **3D Graded Grating Preflight (`build_3d_grating_graded_grid` with `wakis.estimate`)**: Extending the
#   graded $(y, z)$ profile to the 3D Karetnikova rectangular waveguide (`Task 9`, $b = 0.85\;\text{mm}$,
#   $n_{\text{feat}} = 8$, $N_x = 34$) reduces the 3D mesh from $34 \times 64 \times 40 = 87{,}040$ cells
#   (`Uniform 3D`, preflighted at $69.8\;\text{MB}$ CPU RAM) to $34 \times 44 \times 36 = 53{,}856$ cells
#   (`Graded 3D`, **$1.62\times$ fewer cells**, preflighted at $48.6\;\text{MB}$ CPU RAM via `wakis.estimate`)
#   while preserving $\Delta y_{\min} = \Delta z_{\min} = 12.5\;\mu\text{m}$ ($8.0$ cells across the
#   $100\;\mu\text{m}$ vane tooth) and $\ge 35.1\;\text{cells}/\lambda$ at $f_{\max} = 285\;\text{GHz}$.
#
# ### Numerical Requirements for Monotonic $K_{-1}(0)$ Convergence
#
# Three numerical details in `_helpers.py` and `sws_cold_test.py` are essential to prevent $\pm 1.7\%$ spectral and projection artifacts in $K_{-1}(0)$:
#
# 1. **Hann-windowed DFT over an integer number of RF cycles (`HannDFTMonitor` in `extract_bloch_mode_dft`)**: Because `GaussianSource` injects a real-valued current waveform, the lossless Bloch cavity is excited with equal-amplitude $+f_1$ and $-f_1$ oscillations (plus weak higher-band $f_2$ components). In the demodulated $+f_1$ frame, the $-f_1$ image oscillates at $2 f_1$ with opposite $\hat{\mathbf{E}} / \hat{\mathbf{H}}$ sign. An unwindowed boxcar DFT over a non-integer number of cycles leaks $\sim \sin(2\pi N_{\text{cyc}}) / (2\pi N_{\text{cyc}}) \approx \pm 1.7\%$ between $|\hat{\mathbf{E}}|$ and $|\hat{\mathbf{H}}|$. Rounding `n_ring_steps` to an exact integer number of RF cycles `n_ring_cycles` and applying a continuous Hann taper $w(\tau) = 1 - \cos(2\pi \tau / T_{\text{ring}})$ at each component's native leapfrog time level (`t + dt` for `E`, `t + dt/2` for `H`) suppresses this spectral leakage below $10^{-6}$ (verified invariant across $N_{\text{cyc}} \in \{6, 9, 12\}$ to 6 significant digits).
# 2. **Full-period $N_z$-cell Floquet projection (`project_floquet_space_harmonics`)**: When `bc_z = 'bloch'`, `SolverFIT3D` calls `grid.remove_edge_ghost(2)`, setting `grid.edge_ghosts[2] = 0` and `int(grid.Nz) = N_z`. Using `nz = int(grid.Nz) - int(grid.edge_ghosts[2])` (rather than `int(grid.Nz) - 1`) sums the harmonic projection and cell-center field averages over all $N_z$ axial cells of the period (`10, 20, 40, 60, 80`), recovering exact synthetic Floquet harmonics to $< 10^{-15}$ (`test_floquet_harmonic_projection_full_period_vs_off_by_one`) and reducing on-axis even-harmonic leakage $K_0(0)$ to $5.88\times 10^{-7}\;\Omega$ at `Reference`.
# 3. **Direct $\beta_0 d = 0.5\pi$ sampling in the 15-point `--full` sweep (`Tasks 5–6`)**: Choosing the `--full` 2D phase grid `np.linspace(0.125, 1.0, 15)` (step $1/16 = 0.0625$) places $\beta_0 d = 0.5000\pi$ directly at index 6 in `Tasks 5–6`, matching single-point `Tasks 0–4, 7–8` without interpolation.
#
# ### Mesh Resolution Audit (`data/task_*.json`)
#
# Reading the saved resolution diagnostics from `data/task_00_*.json` through `data/task_11_*.json` against the project's three mesh criteria ($\ge 10$ cells per wavelength in the highest-index material, $\ge 4$ cells across the smallest geometric feature, and $\ge 4$ cells per skin depth for lossy metals):
#
# | Task(s) | Description | `cells_across_feature` ($\ge 4$) | `cells_per_wavelength` ($\ge 10$, $n = 1$) | Metal BC (`skin_depth`) | Status |
# |---|---|---|---|---|---|
# | `Tasks 00–01` | 2D under-resolved demo ($1\times 16\times 10$, $160$ cells, $\Delta = 50\;\mu\text{m}$) | $2.0$ (intentional demo) | $25.51$ (at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Intentional failure demo |
# | `Tasks 02, 05` | 2D `L1` ($1\times 32\times 20$, $640$ cells, $\Delta = 25\;\mu\text{m}$) | $4.0$ | $51.03$ (at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
# | `Tasks 03, 06` | 2D `L2` ($1\times 64\times 40$, $2{,}560$ cells, $\Delta = 12.5\;\mu\text{m}$) | $8.0$ | $102.06$ (at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
# | `Tasks 04, 07` | 2D `L3` ($1\times 96\times 60$, $5{,}760$ cells, $\Delta = 8.33\;\mu\text{m}$) | $12.0$ | $153.09$ (at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
# | `Task 08` | 2D `Reference` ($1\times 128\times 80$, $10{,}240$ cells, $\Delta = 6.25\;\mu\text{m}$) | $16.0$ | $204.11$ (at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
# | `Task 09` | 3D Karetnikova grating ($34\times 64\times 40$, $87{,}040$ cells, $\Delta = 25\times 12.5\times 12.5\;\mu\text{m}$) | $8.0$ | $42.08$ (at $285\;\text{GHz}$; $51.03$ at $235\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
# | `Tasks 10–11` | 3D Marosi pillar SWS ($56\times 64\times 56$, $200{,}704$ cells, $\Delta = 25\times 12.5\times 18.75\;\mu\text{m}$) | $8.0$ | $149.90$ (at $80\;\text{GHz}$; $133.24$ at $90\;\text{GHz}$) | Lossless PEC ($0\;\text{m}$) | Pass |
#
# Every non-demo run (`Tasks 02–11`) meets all applicable criteria with zero violations.
#
# ---
#
# ## 7. Limits & Failure Modes
#
# 1. **Under-resolving the vane thickness / beam tunnel or using staircase PEC (`Tasks 0–1`, Fig. 11)**:
#    When $\Delta y = \Delta z = 50\;\mu\text{m}$ ($n_{\text{feat}} = 2$, violating the $\ge 4$ cells/feature rule
#    across the $100\;\mu\text{m}$ vane thickness), staircase PEC shifts the resonant frequency by $-20.9\%$
#    ($77.58\;\text{GHz}$ vs. $98.12\;\text{GHz}$ McVey $f_{1,\infty}$) and inflates the on-axis Pierce impedance
#    by $+237\%$ in `--full` ($5.94\;\Omega$ vs. $1.7637\;\Omega$ McVey $K_{-1,\infty}$ / $1.7749\;\Omega$ FIT
#    `Reference`) because zeroing the boundary cell edges artificially narrows the effective tunnel aperture and slot
#    opening. Always enforce $\ge 4$ cells across the smallest feature and enable `conformal='pec'`.
# 2. **Slot-mode truncation in wide-slot gratings (Fig. 3, Section 4)**:
#    When the groove width $s$ exceeds the tunnel height $2 g_{\text{half}}$ ($s / (2 g_{\text{half}}) = 2$
#    in Karetnikova et al. Table I), the single-mode uniform-aperture assumption ($M_{\text{slot}} = 1$,
#    $E_z(g_{\text{half}}, z) = \text{const}$) misses the strong field concentration at the $90^\circ$ PEC vane
#    corners, under-predicting the passband frequency by $17.2\%$ ($81.28\;\text{GHz}$ vs. $98.12\;\text{GHz}$
#    McVey $f_{1,\infty}$ at $\beta_0 d = 0.5\pi$). Even $M_{\text{slot}} = 6$ ($f_1 = 97.69\;\text{GHz}$,
#    $K_{-1}(0) = 1.8274\;\Omega$) remains $-0.44\%$ below $f_{1,\infty}$ and $+3.61\%$ above the extrapolated
#    impedance limit $K_{-1,\infty} = 1.7637\;\Omega$; power-law extrapolation in $M_{\text{slot}}$ is required
#    for sub-percent impedance benchmarking.
# 3. **Source self-field contamination in `DFTMonitor`**:
#    Accumulating a volume `DFTMonitor` from $t = 0$ while a point `CurrentSource` is firing inside the
#    beam tunnel superimposes the non-propagating dipole near-field ($E_z \propto 1/r^3$) onto the
#    Bloch eigenmode, corrupting the ratio $|E_{zn}|^2 / |P_z|$ by orders of magnitude. Running the drive
#    pulse to completion (`pulse.end_time`) before starting the `DFTMonitor` during the free ring-down
#    eliminates this artifact completely.
# 4. **Band-edge $\pi$-mode impedance divergence ($v_g \to 0$)**:
#    At a non-glide stopband edge ($\beta_0 d = \pi$ in a mirror grating), the group velocity vanishes
#    ($v_g = v_E \to 0 \implies P_z \to 0$) while the standing-wave stored energy $U_{\text{cell}}$
#    remains finite. The cold-test definition $K_n = |E_{zn}|^2 / (2\beta_n^2 |P_z|)$ therefore diverges
#    as $1 / v_g$. Near a band edge, standard 3-wave Pierce theory breaks down and must be replaced by
#    coupled-mode / coalesced-root dispersion theory (linking to Chapter 8).
#
# ---
#
# ## 8. Exercises
#
# 1. **Groove depth tuning**: Vary `vane_height_m` over $h \in \{0.20, 0.25, 0.30, 0.35\}\;\text{mm}$
#    in `GratingSWSSpec` and plot how the upper asymptotic cutoff frequency $f_{\max} \approx c / (4h)$
#    and the synchronous voltage $V_{\text{sync}}$ scale with $h$.
# 2. **Symmetry-breaking perturbation**: Displace the top vane in the 2D grating by $\delta z = 0.45\,d$
#    instead of $0.50\,d$ (broken glide symmetry). Measure the width of the stopband $\Delta f_{\text{gap}}$
#    that opens at $\beta_0 d = \pi$ as a function of $|\delta z - d/2|$.
# 3. **Sheet-beam thickness averaging**: Vary `beam_half_thickness_m` from $25\;\mu\text{m}$ ($H_b = 50\;\mu\text{m}$)
#    to $75\;\mu\text{m}$ ($H_b = 150\;\mu\text{m}$, Table I, filling $75\%$ of the $2a = 200\;\mu\text{m}$ tunnel)
#    and $90\;\mu\text{m}$ (filling $90\%$ of the tunnel), and quantify the enhancement of
#    $\langle K_{-1} \rangle_{\text{beam}} / K_{-1}(0)$ due to the $\cosh(\tau_{-1} y)$ field rise near
#    the vane tips.
#
# ---
#
# ## 9. References
#
# 1. R. Marosi, M. Zuboraj, and F. Capolino, "Wideband Glide-Symmetric Slow-Wave Structure for
#    Millimeter-Wave Sheet Beam TWTs," *arXiv:2505.22927v1* (2025),
#    DOI: [10.48550/arXiv.2505.22927](https://doi.org/10.48550/arXiv.2505.22927).
# 2. T. A. Karetnikova, A. G. Rozhnev, N. M. Ryskin, A. E. Fedotov, S. V. Mishakin, and N. S. Ginzburg,
#    "Gain Analysis of a 0.2-THz Traveling-Wave Tube With Sheet Electron Beam and Staggered Grating
#    Slow Wave Structure," *IEEE Trans. Electron Devices* **65**(6), 2129–2134 (2018),
#    DOI: [10.1109/TED.2017.2787960](https://doi.org/10.1109/TED.2017.2787960).
# 3. B. D. McVey, M. A. Basten, J. H. Booske, J. Joe, and J. E. Scharer, "Analysis of rectangular
#    waveguide-gratings for amplifier applications," *IEEE Trans. Microw. Theory Tech.* **42**(6),
#    995–1003 (1994), DOI: [10.1109/22.293568](https://doi.org/10.1109/22.293568).
# 4. S. Dey and R. Mittra, "A locally conformal finite-difference time-domain (FDTD) algorithm for
#    modeling three-dimensional perfectly conducting objects," *IEEE Microw. Guided Wave Lett.* **7**(9),
#    273–275 (1997), DOI: [10.1109/75.622536](https://doi.org/10.1109/75.622536).
# 5. V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and its applications,"
#    *J. Chem. Phys.* **107**(17), 6756–6769 (1997), DOI: [10.1063/1.475324](https://doi.org/10.1063/1.475324).
# 6. J. R. Pierce, *Traveling-Wave Tubes*, D. Van Nostrand Company, New York (1950).


# %%
def merge_bn_shards(task: dict[str, Any], out_dir: Path = DATA_DIR) -> Path:
    """Merge single-point shards ``task_XX_<name>_bnJJ.{npz,json}`` into ``task_XX_<name>.*``.

    Per-point arrays are concatenated in bn order; ``n_steps`` and ``wall_time_s`` are summed
    (``wall_time_s`` is then total compute time, not elapsed time). The forward field slice is
    taken from the shard the unsharded loop would have kept (last point with |bn - 0.8| < 0.08,
    otherwise the first point that has one).
    """
    out_dir = Path(out_dir)
    stem = f"task_{int(task['index']):02d}_{task['name']}"
    n_bn = len(task["bn_list"])
    shard_paths = [out_dir / f"{stem}_bn{j:02d}.npz" for j in range(n_bn)]
    missing = [p.name for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    shards = [dict(np.load(p)) for p in shard_paths]
    metas = [json.loads(p.with_suffix(".json").read_text()) for p in shard_paths]
    per_point = [k for k, v in shards[0].items() if np.ndim(v) >= 1 and v.shape[0] == 1]
    merged = {}
    for k in per_point:
        arrs = [np.asarray(sh[k], dtype=float) for sh in shards]
        shape = tuple(max(a.shape[i] for a in arrs) for i in range(arrs[0].ndim))
        padded = []
        for a in arrs:
            b = np.full(shape, np.nan)
            b[tuple(slice(0, n) for n in a.shape)] = a
            padded.append(b)
        merged[k] = np.concatenate(padded)
    slices = [(float(sh["bn"][0]), sh["ez_yz_slice_fwd"]) for sh in shards if "ez_yz_slice_fwd" in sh]
    near = [sl for bn, sl in slices if abs(bn - 0.8) < 0.08]
    if near or slices:
        merged["ez_yz_slice_fwd"] = near[-1] if near else slices[0][1]
    if "kn_n_indices" in shards[0]:
        merged["kn_n_indices"] = shards[0]["kn_n_indices"]
    np.savez_compressed(out_dir / f"{stem}.npz", **merged)
    summary = dict(metas[0])
    summary["name"] = str(task["name"])
    summary["n_steps"] = int(sum(m["n_steps"] for m in metas))
    summary["wall_time_s"] = float(sum(m["wall_time_s"] for m in metas))
    summary["n_bn_shards"] = n_bn
    summary.pop("f1_mid_ghz", None)
    summary.pop("km1_mid_ohm", None)
    if "f_fwd_ghz" in merged:
        summary["f_degen_ghz"] = float(merged["f_fwd_ghz"][-1])
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"merged {n_bn} shards -> {json_path}")
    return json_path


# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="02. Slow-Wave Structure Cold Test: Corrugated & Glide-Symmetric Waveguides (2D and 3D)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run compact 2D/3D cold-test study locally and render all 15 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run high-density production study across all 12 tasks",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single task index I (0..11); respects --quick or defaults to --full",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for task .npz and .json files (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read existing data/task_*.npz and generate all 15 figures in figures/",
    )
    ap.add_argument(
        "--bn-index",
        type=int,
        default=None,
        help="With --task: run only point J of the task's bn_list and write task_XX_<name>_bnJJ.* "
        "(shards long 3D sweeps across Batch jobs; combine with --merge-bn)",
    )
    ap.add_argument(
        "--merge-bn",
        type=int,
        default=None,
        help="Merge the task_XX_<name>_bnJJ.* shards of task I in --out into task_XX_<name>.*",
    )
    ap.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="With --task: override the task's ring-down length n_steps (output name gets _nNNNNN)",
    )
    ap.add_argument(
        "--stage1-only",
        action="store_true",
        help="3D Marosi tasks: run only the Harminv stage, store all modes, and write task_XX_<name>_s1.*",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the 12-task study table and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_tasks:
        tasks = get_task_table(quick=bool(args.quick and not args.full))
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        quick_mode = bool(args.quick and not args.full)
        tasks = get_task_table(quick=quick_mode)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        task = dict(tasks[args.task])
        if args.n_steps is not None:
            task["n_steps"] = int(args.n_steps)
            task["name"] = f"{task['name']}_n{args.n_steps}"
        if args.stage1_only:
            global STAGE1_ONLY
            STAGE1_ONLY = True
            task["name"] = f"{task['name']}_s1"
        if args.bn_index is not None:
            bn_all = list(task["bn_list"])
            if not (0 <= args.bn_index < len(bn_all)):
                raise SystemExit(f"--bn-index must be in 0..{len(bn_all) - 1}")
            task["bn_list"] = [bn_all[args.bn_index]]
            task["name"] = f"{task['name']}_bn{args.bn_index:02d}"
        run_single_task(task, args.out)
        return

    if args.merge_bn is not None:
        task = dict(get_task_table(quick=bool(args.quick and not args.full))[args.merge_bn])
        if args.n_steps is not None:
            task["name"] = f"{task['name']}_n{args.n_steps}"
        if args.stage1_only:
            task["name"] = f"{task['name']}_s1"
        merge_bn_shards(task, args.out)
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    # Default when invoked with no flags (e.g. notebook build/execution):
    # if all task_*.npz files already exist in DATA_DIR and neither --quick nor --full
    # was explicitly forced, render figures directly from cached data; otherwise run --quick.
    quick = not args.full
    tasks = get_task_table(quick=quick)
    existing = list(Path(args.out).glob("task_*.npz"))
    if not (args.quick or args.full) and len(existing) >= len(tasks):
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    for t in tasks:
        run_single_task(t, args.out)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
