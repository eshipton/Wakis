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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/08_vacuum_electronics/08-07_metamaterial_sws_reversed_cherenkov/metamaterial_sws.ipynb)
#
# # Tutorial 8.7 · A Metamaterial Slow-Wave Structure: Reversed Cherenkov Radiation
#
# In 1968, Viktor Veselago predicted that a charge moving uniformly through an isotropic
# medium with simultaneously negative permittivity ($\varepsilon < 0$) and permeability
# ($\mu < 0$)—a **double-negative (DNG)** or **left-handed** medium—radiates
# **reversed Cherenkov radiation (RCR)**: electromagnetic phase fronts advance forward
# with the charge ($v_p = \omega / \beta_0 > 0$), while Poynting energy flows backward
# upstream toward the beam entrance ($v_g = d\omega / d\beta_0 < 0$) forming an obtuse
# angle with the particle velocity (Veselago 1968).
# Because high-current relativistic electron beams require vacuum propagation and high
# thermal/breakdown thresholds, Duan *et al.* (2017)
# realized a planar **all-metal metamaterial slow-wave structure (MTM-SWS)** by suspending a
# periodic copper sheet perforated with **complementary electric split-ring resonators
# (CeSRRs)** down the midplane of a below-cutoff square copper waveguide.
#
# In this tutorial, we model the S-band all-metal CeSRR-loaded square waveguide of
# Duan *et al.* (2017) in 3D Finite Integration Technique (`SolverFIT3D`):
#
# 1. **Unit-cell cold test & effective-medium interpretation**: we present
#    Duan *et al.*'s (2017, Supplementary Note 1, Eqs. 2–3, SI l. 340–350) Drude-like
#    effective permeability $\mu_{\text{eff}}^{\text{model}}(f) = 1 - (f_{c,\text{TM}_{11}}/f)^2 < 0$
#    of the below-cutoff square waveguide alongside a Lorentz illustration of
#    $\varepsilon_{xx,\text{eff}}^{\text{model}}(f) < 0$ (parameters assumed by us; SI Note 1 Eq. 1,
#    SI l. 326–334, gives only the diagonal tensor form while the paper's $\varepsilon_{xx}$ curve
#    comes from HFSS retrieval), compute the full 3D FIT Floquet–Bloch dispersion $\omega(\beta_0)$,
#    verify $v_E = P_z p / U_{\text{cell}} = v_g < 0$ while $v_p > 0$, and perform a systematic
#    numerical diagnosis of the Pierce interaction impedance $K_c$ across candidate power-flow,
#    phasor, space-harmonic, transverse-position, and sub-cell geometry variations.
# 2. **Reversed Cherenkov radiation with a distributed sheet-beam bunch (`wakis.sources.Beam`)**:
#    a 6-filament relativistic sheet-beam bunch at the paper's space-charge-depressed
#    entrance velocity ($\beta = 0.5474$, $V_0 = 99.7\text{ keV}$, main text l. 284; SI Eq. 5, l. 513–514)
#    traverses a finite CeSRR structure terminated by matched electric–magnetic adiabatic tapers
#    (`termination='matched_taper'`), comparing the direct upstream-to-downstream Poynting wake
#    ratio (both ends absorbed) against the paper's simulated (main text l. 288) and measured
#    (main text l. 364) ratios where Port 1 records the Coupler 2 reflection (SI l. 479–481).
# 3. **Falsifiable physics checks**: (i) Poynting flux sign $\text{sgn}(P_z) = \text{sgn}(v_g) < 0$
#    and Bloch virial equipartition $U_E / U_H = 1$; (ii) positive synchronous frequency slope
#    $df_{\text{sync}}/d\beta > 0$ across beam energies even though $v_g < 0$, plus computation of
#    the 1D slow space-charge wave intersection $\omega = \beta_0 v_0 - \omega_q$ at the paper's
#    particle-in-cell simulation current $I_0 = 1.55\text{ kA}$ (main text l. 199; SI l. 474) vs.
#    the particle-in-cell Port 2 peak (main text l. 289); (iii) removing the CeSRR slots
#    (`plain_plate=True`) eliminates the left-handed passband and suppresses upstream wake energy by
#    the plain-plate suppression factor rendered from JSON below; and (iv) discrete FIT leapfrog Poynting
#    and inner-taper volumetric absorber energy conservation $\sum_t \sum_{\text{inner taper}} (\sigma |\mathbf{E}|^2 + \sigma_m |\mathbf{H}|^2)\,\Delta t\,\Delta V$.
# 4. **Backward-wave oscillator (BWO) start-oscillation condition**: we evaluate the Johnson
#    space-charge-corrected start current (Johnson 1955) for the $N = 20$ cell SWS ($L = 290\text{ mm}$,
#    main text l. 411, 413; SI l. 117, 455) with consistent small-signal space-charge parameters
#    $QC(I_{\text{st}})$ and couple the 3D CeSRR mode to `wakis.beam_fluid.LinearizedBeam`.
#
# ## Learning Goals
#
# - Distinguish the primary dimensions reported in Duan *et al.* (2017) main text and Supplementary
#   Information ($a = 14.5\text{ mm}$, $p = 14.5\text{ mm}$, plate thickness $t = 1.2\text{ mm}$,
#   $N = 20$, $L = 290\text{ mm}$, $12\times 2\text{ mm}$ sheet beam, $0.5\text{ mm}$ clearance)
#   from the sub-cell CeSRR slot dimensions $(b, d, g, h_1, h_2, j)$ assumed by us, and distinguish
#   Duan *et al.*'s (2017, SI Note 1 Eqs. 2–3) Drude $\mu_{\text{eff}}^{\text{model}}$ from our
#   Lorentz illustration of $\varepsilon_{xx,\text{eff}}^{\text{model}}$.
# - Run a 3D FIT Bloch cold test (`SolverFIT3D` with complex periodic boundaries, `ResonanceMonitor`,
#   and `HannDFTMonitor`) to measure backward-wave dispersion ($v_g < 0, v_p > 0$), verify
#   $v_E = P_z p / U_{\text{cell}} = v_g$ using staggered FIT energy and colocated Poynting quadrature,
#   and diagnose the gap between the 3D FIT Pierce interaction impedance $\langle K_{c,0}\rangle$ and
#   SI Note 2 ("averaged interaction impedance is greater than $\sim 750\;\Omega$", SI l. 395–396;
#   note that $0.72\text{--}1.32\text{ k}\Omega$ in SI Fig. 2b is the y-axis tick range [SI l. 28, 50], not data).
# - Measure passive port termination reflection $|\Gamma(f)|$ across the cold-synchronism evaluation band
#   via a two-probe Bloch-mode decomposition, perform a coarse band-median consistency check on a partial
#   reflector against cross-sectional energy flux (which checks only the band median and does not validate
#   $\phi(f)$ per frequency), quantify the per-frequency estimator error on a smooth below-cutoff air
#   waveguide standing-wave sanity check, simulate reversed Cherenkov radiation from a 6-filament
#   sheet-beam discretization vs. a single line beam, and compare the both-ends-absorbed Port 2 / Port 1
#   direct leakage ratio with Duan *et al.*'s Coupler-2-reflected setup (SI l. 479–481).
# - Compute the BWO start-oscillation current using the Johnson (1955) formalism with consistent
#   $QC(I_{\text{st}})$ and `wakis.beam_fluid.LinearizedBeam`, compute the slow space-charge wave
#   intersection $\omega = \beta_0 v_0 - \omega_q$ at the paper's particle-in-cell simulation current
#   $1.55\text{ kA}$ (main text l. 199; SI l. 474), and explain why $f_q / f \approx 0.5$ lies outside
#   the regime of validity of linear space-charge wave theory.
#
# ---
#
# ## Theory and Physical Formulation
#
# ### 1. All-Metal CeSRR Geometry: Primary-Source Dimensions vs. Assumed Sub-Cell Cutouts
#
# We separate the dimensions explicitly given in Duan *et al.* (2017) main text and Supplementary
# Information (SI) from the unquoted sub-cell CeSRR slot dimensions assumed by us:
#
# 1. **Dimensions and beam parameters given in Duan *et al.* (2017) main text and SI**:
#    - **Square waveguide inner width, plate thickness, and period**: $a \times a = 14.5\text{ mm} \times 14.5\text{ mm}$
#      (main text l. 96, 98, 411; SI l. 44, 50, 324, 326), axial period $p = 14.5\text{ mm}$
#      (main text l. 98; SI l. 50, 326), and copper CeSRR layer thickness $t = 1.2\text{ mm}$
#      ($y \in [-0.6, +0.6]\text{ mm}$, given in main text l. 413 as $14.5\text{ mm} \times 1.2\text{ mm} \times 290\text{ mm}$),
#      with $N = 20$ unit cells of total interaction length $L = 20 p = 290\text{ mm}$ (main text l. 411, 413;
#      SI l. 117, 455; compare the $17$-cell CSWG prototype of Hummelt *et al.* 2014 and Duan *et al.* 2014).
#    - **Sheet electron beam and clearance**: nominal cross-section $W_b \times H_b = 12.0\text{ mm} \times 2.0\text{ mm}$
#      (main text l. 199, 424; SI l. 58, 474–475; $11\text{ mm} \times 1.7\text{ mm}$ measured on witness
#      target, main text l. 278; SI l. 577–578), propagating with a $0.5\text{ mm}$ vertical clearance above
#      the CeSRR plate surface (main text l. 280; SI l. 56, 475; so the $2.0\text{ mm}$-thick sheet beam
#      occupies $y \in [+1.1, +3.1]\text{ mm}$ above the $t = 1.2\text{ mm}$ plate), driven at a
#      particle-in-cell simulation beam current $I_0 = 1.55\text{ kA}$ (main text l. 199; SI l. 474) and
#      confined by an axial magnetic flux density $B_z = 1.06\text{ T}$ (main text l. 287; SI l. 466, 551).
#    - **Cathode vs. space-charge-depressed beam velocity**: the diode cathode voltage is $160\text{ kV}$
#      ($v_0 = 1.94\times 10^8\text{ m/s}$, $\beta = 0.6471$, main text l. 142, 281–284), while DC electrostatic
#      potential depression inside the $14.5\text{ mm}$ tunnel reduces the entrance kinetic energy to
#      $99.7\text{ keV}$ ($v_0 = 1.64\times 10^8\text{ m/s}$ in main text l. 284;
#      $v_0 = 1.6410\times 10^8\text{ m/s}$, $\beta = 0.5474$ in SI Eq. 5, l. 513–514).
# 2. **Sub-cell CeSRR cutout dimensions (assumed by us)**:
#    Neither Duan *et al.* (2017) main text nor the Supplementary Information tabulates the internal
#    in-plane cutout dimensions $(b, d, g, h_1, h_2, j)$ labeled in Main Text Fig. 1a and SI Fig. 1a.
#    We therefore adopt a self-consistent set of in-plane cutout dimensions **assumed by us** to satisfy
#    the geometric partition of the $a = p = 14.5\text{ mm}$ cell and place the fundamental left-handed
#    resonance near $2.84\text{ GHz}$:
#    - Outer CeSRR cutout span $b = 13.5\text{ mm}$ (assumed by us to leave a $0.5\text{ mm}$ wall frame
#      on each side of the $14.5\text{ mm}$ period), slot width $d = 1.0\text{ mm}$ and central split gap
#      $g = 1.0\text{ mm}$ (assumed by us as standard $1\text{ mm}$ CNC-milled slot/gap widths), central
#      axial bar width $j = 1.5\text{ mm}$ and transverse paddle width $h_2 = 4.0\text{ mm}$ (assumed by us
#      so that $2 h_2 + 4 d + j = b = 13.5\text{ mm}$), and axial paddle length $h_1 = 5.25\text{ mm}$
#      (assumed by us so that $2 h_1 + 2 d + g = b = 13.5\text{ mm}$ and the half-slot perimeter
#      $L_{\text{slot}} = 2 b + 2 h_1 + 2 h_2 - g/2 = 45.0\text{ mm}$ yields a fundamental slot resonance
#      near $c / (2 L_{\text{slot}}) \approx 3.33\text{ GHz}$).
#
# Along the transverse ($x$) and axial ($z$) coordinates within the $b = 13.5\text{ mm}$ cutout,
# these assumed sub-cell dimensions satisfy exact geometric partition identities (*derived here*):
#
# $$\underbrace{2 h_2}_{8.0\text{ mm}} + \underbrace{4 d}_{4.0\text{ mm}} + \underbrace{j}_{1.5\text{ mm}} = 13.5\text{ mm} = b, \qquad \underbrace{2 h_1}_{10.5\text{ mm}} + \underbrace{2 d}_{2.0\text{ mm}} + \underbrace{g}_{1.0\text{ mm}} = 13.5\text{ mm} = b.$$
#
# Together with the $0.5\text{ mm}$ outer frame on each end ($a - b = 1.0\text{ mm}$) and the paper's
# $t = 1.2\text{ mm}$ plate (main text l. 413) and $0.5\text{ mm}$ beam clearance (SI l. 56), the
# interface coordinates on each half-axis are:
#
# - $|x| \in \{0,\; 0.75,\; 1.75,\; 5.75,\; 6.75,\; 7.25\}\text{ mm}$,
# - $|z| \in \{0,\; 0.50,\; 1.50,\; 5.75,\; 6.75,\; 7.25\}\text{ mm}$,
# - $y \in \{-7.25,\; -0.60,\; +0.60,\; +1.10,\; +3.10,\; +7.25\}\text{ mm}$.
#
# ### 2. Effective-Medium Interpretation: Duan *et al.*'s Drude $\mu_{\text{eff}}$ (SI Eqs. 2–3) & Our Lorentz $\varepsilon_{xx}$ Illustration
#
# Why does an all-metal CeSRR-loaded waveguide support a backward-wave ($\mathbf{v}_g \cdot \mathbf{v}_p < 0$)
# passband far below the hollow waveguide cutoff? Duan *et al.* (2017, Supplementary Note 1, Eqs. 1–3,
# SI l. 326–350) and Marqués *et al.* (2002) interpret the structure via an effective-medium framework
# (all quantities from this framework carry the superscript/label `model` to distinguish them from 3D FIT
# simulation results):
#
# 1. **Drude effective permeability ($\mu_{\text{eff}}^{\text{model}} < 0$) from Duan *et al.* (2017, SI Note 1 Eqs. 2–3, SI l. 340–350)**:
#    For a hollow square waveguide of width $a = 14.5\text{ mm}$ (main text l. 96; SI l. 44, 326), the lowest
#    transverse-magnetic mode is $\text{TM}_{11}$, with transverse wavenumber
#    $k_{c,\text{TM}_{11}} = \sqrt{(\pi/a)^2 + (\pi/a)^2} = \sqrt{2}\pi / a$ and cutoff frequency
#    (SI Eq. 3, l. 349–350):
#
#    $$f_{c,\text{TM}_{11}} = \frac{c}{\sqrt{2}\,a} = \frac{2.99792\times 10^8\text{ m/s}}{\sqrt{2}\times 14.5\times 10^{-3}\text{ m}} \approx 14.620\text{ GHz}.$$
#
#    (Even the fundamental $\text{TE}_{10}$ mode has cutoff $f_{c,\text{TE}_{10}} = c/(2a) \approx 10.338\text{ GHz}$.)
#    In the 1D transmission-line equivalence of a $\text{TM}$ waveguide mode propagating as $e^{i(\omega t - \beta_0 z)}$,
#    eliminating $E_z$ from Faraday's and Ampère's laws yields (*derived here*):
#
#    $$\beta_0^2 = k_0^2 - k_{c,\text{TM}_{11}}^2 = k_0^2 \left[1 - \left(\frac{f_{c,\text{TM}_{11}}}{f}\right)^2\right] \equiv k_0^2\,\frac{\varepsilon_{xx,\text{eff}}^{\text{model}}}{\varepsilon_0}\,\mu_{\text{eff}}^{\text{model}}(f),$$
#
#    where the effective shunt magnetic response is given in lossless Drude form by Duan *et al.*
#    (Supplementary Note 1 Eqs. 2–3, SI l. 340–350):
#
#    $$\mu_{\text{eff}}^{\text{model}}(f) = 1 - \left(\frac{f_{c,\text{TM}_{11}}}{f}\right)^2.$$
#
#    Across the paper's left-handed passband ($f \in [2.83, 3.05]\text{ GHz}$, main text l. 195, 203, 376;
#    SI l. 23, 352), $\mu_{\text{eff}}^{\text{model}}(f)$ is strongly **negative**. In an unloaded waveguide
#    ($\varepsilon = \varepsilon_0 > 0, \mu_{\text{eff}}^{\text{model}} < 0$), $\beta_0 = -i|\beta_0|$ is
#    purely imaginary and waves are evanescent.
#
# 2. **Lorentz illustration of $\varepsilon_{xx,\text{eff}}^{\text{model}} < 0$ (parameters assumed by us)**:
#    In Duan *et al.* (2017, Supplementary Note 1 Eq. 1, SI l. 326–334), Eq. 1 specifies only the
#    anisotropic diagonal tensor structure $\bar{\bar{\varepsilon}} = \operatorname{diag}(\varepsilon_{xx}, \varepsilon_0, \varepsilon_0)$,
#    while the paper's $\varepsilon_{xx}(f)$ curve in SI Fig. 1c is obtained by HFSS S-parameter retrieval
#    (SI l. 336–339). To provide a closed-form pedagogical **Lorentz illustration (parameters assumed by us)**
#    spanning the paper's left-handed passband $[f_{e0}, f_{ep}] = [2.83, 3.05]\text{ GHz}$ (main text
#    l. 195, 203; with damping $\gamma_e / 2\pi = 8\text{ MHz}$ assumed by us), we write:
#
#    $$\frac{\varepsilon_{xx,\text{eff}}^{\text{model}}(\omega)}{\varepsilon_0} = 1 - \frac{\omega_{ep}^2 - \omega_{e0}^2}{\omega^2 - \omega_{e0}^2 - i\gamma_e\omega}, \qquad n_{\text{eff}}^{\text{model}}(\omega) = -\sqrt{\frac{\varepsilon_{xx,\text{eff}}^{\text{model}}(\omega)}{\varepsilon_0}\,\mu_{\text{eff}}^{\text{model}}(\omega)}.$$
#
#    Because $\varepsilon_{xx,\text{eff}}^{\text{model}}$ and $n_{\text{eff}}^{\text{model}}$ use illustrative
#    Lorentz parameters assumed by us rather than an S-parameter retrieval, we label them as a
#    `"Lorentz illustration, parameters assumed by us"` in Figure 1b and include only Duan *et al.*'s
#    Drude $\mu_{\text{eff}}^{\text{model}}$ (SI Eqs. 2–3) in the Quantitative Summary table. Moreover,
#    at the synchronous operating point $\phi = \beta_0 p = 95.7^\circ$ ($p \approx \lambda_g / 3.76$),
#    the unit cell is not deep-subwavelength relative to the guided wavelength $\lambda_g$, so quantitative
#    beam–wave design must rely on the exact 3D Floquet–Bloch dispersion $\omega(\beta_0)$ and
#    space-harmonic Pierce impedance $K_{c,0}(\beta_0)$ computed directly in 3D FIT below.
#
# ### 3. Floquet–Bloch Dispersion, Energy Velocity $v_E = v_g$, and Pierce Impedance $K_c$ Diagnosis
#
# By Floquet's theorem, an eigenmode of the periodic CeSRR waveguide with period $p$ satisfies
# $\mathbf{E}(\mathbf{r} + p\hat{\mathbf{z}}, t) = e^{-i\beta_0 p}\mathbf{E}(\mathbf{r}, t)$, where
# $\phi = \beta_0 p \in [0, \pi]$ is the inter-cell phase shift. Expanding the axial electric field
# in space harmonics $\beta_n = \beta_0 + 2\pi n / p$ ($n \in \mathbb{Z}$):
#
# $$E_z(x, y, z) = \sum_{n=-\infty}^{\infty} E_{z,n}(x, y)\,e^{-i\beta_n z}, \qquad E_{z,n}(x, y) = \frac{1}{p}\int_{-p/2}^{p/2} E_z(x, y, z)\,e^{i\beta_n z}\,dz.$$
#
# In the CeSRR slow-wave structure, the **fundamental ($n = 0$) space harmonic** itself is backward
# ($v_g = d\omega / d\beta_0 < 0$ while $v_p = \omega / \beta_0 > 0$). By time-reversal symmetry and
# $z \leftrightarrow -z$ mirror symmetry of the unit cell, $\omega(\phi) = \sum_{m=0}^M c_m \cos(m\phi)$
# is an even periodic function of $\phi = \beta_0 p$ with analytically vanishing group velocity at both
# Brillouin-zone boundaries ($v_g(0) = v_g(\pi) = 0$). By Poynting's theorem on the staggered FIT
# primary/dual grid (Pierce 1950), the cycle-averaged axial electromagnetic power flow $P_z$ and
# stored energy per unit cell $U_{\text{cell}} = U_E + U_H$ (with $U_E = U_H$ by the Bloch virial theorem)
# determine the **energy velocity** $v_E$, which equals the dispersion group velocity $v_g$:
#
# $$P_z = \frac{1}{2}\text{Re}\iint_{A_\perp} (\mathbf{E} \times \mathbf{H}^*) \cdot \hat{\mathbf{z}}\,dx\,dy, \qquad U_{\text{cell}} = \frac{1}{4}\sum_{e} \varepsilon_e L_e \tilde{A}_e |E_e|^2 + \frac{1}{4}\sum_{f} \mu_f \tilde{L}_f A_f |H_f|^2, \qquad v_E = \frac{P_z\,p}{U_{\text{cell}}} = \frac{d\omega}{d\beta_0} = v_g < 0.$$
#
# The local and 2D sheet-beam-averaged **Pierce interaction impedance** of the $n = 0$ space harmonic
# are (Pierce 1950; Duan *et al.* 2017 SI l. 53–58, 375–402 & SI Fig. 2b):
#
# $$K_{c,0}(x, y) = \frac{|E_{z,0}(x, y)|^2}{2\beta_0^2 P}, \qquad \langle K_{c,0} \rangle_{\text{beam}} = \frac{1}{W_b H_b}\int_{-W_b/2}^{W_b/2}dx \int_{y_{\min}}^{y_{\max}}dy\; K_{c,0}(x, y).$$
#
# Duan *et al.* (2017, SI l. 53–58, 375–402) state that $K_c$ is averaged over the sheet-beam cross-section
# and report in SI Note 2 that the averaged interaction impedance is "greater than $\sim 750\;\Omega$"
# (SI l. 395–396; note that $0.72\text{--}1.32\text{ k}\Omega$ in SI Fig. 2b is the y-axis tick range
# [SI l. 28, 50], not data). In the code cells below, we evaluate each candidate $K_c$ normalisation on
# the 3D FIT eigenmode at $\phi = 95.7^\circ$ (two-channel total power $|P_z|$ vs. single-channel upper
# power $|P_z|/2$, peak vs. RMS phasor factor of $1/2$, $n = 0$ vs. $n = \pm 1$ space harmonics, on-axis
# transverse height $y \in \{2.1, 1.1, 0.6\}\text{ mm}$, and sub-cell cutout geometry sensitivity),
# showing that the remaining gap relative to $\sim 750\;\Omega$ is unexplained: slot width $d$ ($\pm 10\%$
# with $h_1, h_2$ adjusted to preserve the $b$ partitions) is ruled out, while independent variations of
# the unquoted cutout parameters $(h_1, h_2, j)$ (as well as $b, g$) are untested.
#
# ### 4. Reversed Cherenkov Radiation, Slow Space-Charge Wave Shift, and Port Asymmetry
#
# When a relativistic sheet-beam bunch of total charge $q$ and velocity $\mathbf{v} = \beta c \hat{\mathbf{z}}$
# ($\beta = \sqrt{1 - (1 + e V_0 / m_e c^2)^{-2}}$) traverses an $N$-cell CeSRR waveguide, it
# excites the synchronous mode at the intersection of the beam line $\omega = \beta c \beta_0$
# with the backward dispersion curve $\omega(\beta_0)$:
#
# $$\omega_{\text{sync}} = \beta c\,\beta_0(\omega_{\text{sync}}), \qquad v_p(\omega_{\text{sync}}) = \frac{\omega_{\text{sync}}}{\beta_0} = \beta c > 0, \qquad v_g(\omega_{\text{sync}}) = \left.\frac{d\omega}{d\beta_0}\right|_{\omega_{\text{sync}}} < 0.$$
#
# Because $v_g(\omega_{\text{sync}}) < 0$, the Cherenkov wave packet generated by the forward-moving
# bunch propagates **backward** toward the upstream port ($-\hat{\mathbf{z}}$, Port 2 in
# Duan *et al.* 2017 Fig. 1b). Why does Duan *et al.* (2017) report three different frequencies at the
# $99.7\text{ keV}$ ($\beta = 0.5474$, main text l. 284; SI Eq. 5, l. 513–514) operating point?
#
# - **Cold dispersion intersection ("slightly below $2.98\text{ GHz}$" in main text l. 294–295)**:
#   In the zero-current cold dispersion diagram (Main Text Fig. 1c and l. 294–295), the kinematic line
#   $\omega = \beta_0 v_0$ for $v_0 = 1.6410\times 10^8\text{ m/s}$ ($99.7\text{ keV}$, SI Eq. 5, l. 513–514)
#   intersects the fundamental backward branch "slightly below $2.98\text{ GHz}$".
# - **Slow space-charge wave intersection vs. hot particle-in-cell ($2.836\text{ GHz}$, main text l. 289) and measured ($2.850\text{ GHz}$, main text l. 297, 364) RCR peaks**:
#   At the paper's particle-in-cell simulation beam current $I_0 = 1.55\text{ kA}$ (main text l. 199;
#   SI l. 474) across $12\times 2\text{ mm}^2$ (main text l. 199, 424; SI l. 58, 474–475), the 1D slow
#   space-charge wave dispersion line $\omega = \beta_0 v_0 - \omega_q$ (where
#   $\omega_q = R_{\text{sc}}\,\omega_p / \gamma_0^{3/2}$) shifts the synchronous intersection downward
#   to a larger Bloch phase $\phi_{\text{SCW}}$ and lower frequency $f_{\text{SCW}}$ (rendered from JSON
#   below). Crucially, at $I_0 = 1.55\text{ kA}$, $f_q / f \approx 0.5$ and $\omega_p / 2\pi > f$
#   violate the weak-coupling assumptions of 1D linear space-charge wave theory ($f_q / f \ll 1$,
#   $\omega_p \ll \omega$, and $QC \ll 1$), so quantitative prediction of the hot shift requires
#   self-consistent 3D relativistic particle-in-cell simulation.
#
# In our finite-structure 3D FIT `Beam` simulation, both ends of the CeSRR structure are terminated with
# matched adiabatic absorbers (`termination='matched_taper'`), and we measure the outgoing cross-sectional
# Poynting energy through the upstream (Port 2) and downstream (Port 1) reference planes:
#
# $$W_{\text{rad, Port 2}} = \int_{t_{\text{clear}}}^T dt \iint_{z = z_{\text{Port 2}}} (-\mathbf{E}\times\mathbf{H})\cdot\hat{\mathbf{z}}\,dx\,dy, \qquad W_{\text{rad, Port 1}} = \int_{t_{\text{clear}}}^T dt \iint_{z = z_{\text{Port 1}}} (+\mathbf{E}\times\mathbf{H})\cdot\hat{\mathbf{z}}\,dx\,dy.$$
#
# Note the physical difference between our both-ends-absorbed port ratio and Duan *et al.*'s (2017)
# Port 2 / Port 1 ratio: in Duan *et al.* (main text l. 288–291, 364–375; SI l. 478–482, 599–608), the
# Port 1 signal is the backward RCR wave reflected from Coupler 2 after a $\sim 5.5\text{ ns}$ transit delay
# (giving a **simulated** Port-2/Port-1 ratio of $16.9\text{ dB}$ at $2.829\text{ GHz}$ [main text l. 288;
# SI l. 478–479] and a **measured** Port-2/Port-1 ratio of $17.6\text{ dB}$ at $2.847\text{ GHz}$
# [main text l. 297, 364]). Because our 3D FIT run absorbs outgoing waves at both ends without a
# reflecting coaxial coupler, our Port 2 / Port 1 ratio measures **direct downstream wake leakage** rather
# than the Coupler 2 reflection.
#
# Both during bunch transit ($t \in [0, t_{\text{clear}}]$) and during post-transit radiative wake decay
# ($t \in [t_{\text{clear}}, T]$), discrete leapfrog FIT Poynting energy conservation closes across the
# control volume $[z_{\text{Port 2}}, z_{\text{Port 1}}]$. Because the flux planes $z_{\text{Port 2}}, z_{\text{Port 1}}$
# sit $6$ grid cells inside the cubic adiabatic tapers, $W_{\text{abs,wake}}$ accounts for the volumetric
# Ohmic and magnetic dissipation inside those $6$ inner taper cells ($\sim 0.03\%$ of the wake), whereas
# the radiated wake energy $W_{\text{rad,Port 2}} + W_{\text{rad,Port 1}}$ crosses the flux planes and is
# subsequently absorbed in the outer buffer taper beyond the flux planes (*derived here*):
#
# $$W_{\text{beam,loss}}(t_{\text{clear}}) = U_{\text{EM}}(t_{\text{clear}}) + W_{\text{rad,transit}}, \qquad W_{\text{abs,wake}} = \sum_{t \ge t_{\text{clear}}} \sum_{\text{inner taper cells}} \left(\sigma_e |\mathbf{E}|^2 + \sigma_m |\mathbf{H}|^2\right) \Delta t\,\Delta V.$$
#
# ### 5. Backward-Wave Oscillator (BWO) Start-Oscillation Condition
#
# For a continuous sheet electron beam of DC current $I_0$ and voltage $V_0$ interacting with an
# $N$-cell backward-wave structure of length $L = N p$ ($N = 20, L = 290\text{ mm}$ in Duan *et al.* 2017,
# main text l. 411, 413; SI l. 117, 455), the **Pierce gain parameter** $C$ and **space-charge parameter**
# $QC$ evaluated consistently at the start-oscillation threshold $I = I_{\text{st}}$ are
# (Pierce 1950; Johnson 1955):
#
# $$C_{\text{st}} = \frac{(CN)_{\text{st}}}{N_\lambda}, \qquad I_{\text{st}} = \frac{4 V_{\text{eff}}}{\langle K_{c,0} \rangle}\,C_{\text{st}}^3, \qquad V_{\text{eff}} = \frac{\gamma_0(\gamma_0 + 1)}{2}\,V_0, \qquad QC_{\text{st}} = \frac{1}{4}\left[\frac{\omega_q(I_{\text{st}})}{\omega C_{\text{st}}}\right]^2,$$
#
# where $\omega_q(I_{\text{st}}) = R_{\text{sc}}\,\omega_p(I_{\text{st}}) / \gamma_0^{3/2}$ is the
# geometry-reduced relativistic plasma frequency (`reduction_factor_rectangular` returns the linear
# reduction factor $R_{\text{sc}} \in (0, 1)$ directly) at the start current $I_{\text{st}}$
# (where $f_q(I_{\text{st}}) \ll f_{\text{sync}}$ and $QC_{\text{st}} < 1$, well within the small-signal
# Pierce–Johnson validity regime).
#
# ---
#
# ## Setup: CeSRR Unit Cell & Finite Reversed-Cherenkov Waveguide
#
# We first inspect the CeSRR coordinate partition identities (`verify_cesrr_coordinate_identities`),
# preflight both 3D FIT geometries with `wakis.estimate.estimate` and `preflight`, and generate
# the setup schematics (`00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png` for the
# unit-cell Bloch cold test, and `00a_setup_geometry_rcr.png`, `00b_setup_mesh_and_excitation_rcr.png`
# for the finite reversed-Cherenkov waveguide with `wakis.sources.Beam`).

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from IPython.display import Image as IPyImage
from IPython.display import Markdown, display

from _helpers import (
    DATA_DIR,
    FIG_DIR,
    TUTORIAL_DIR,
    CeSRRSWSSpec,
    build_cesrr_unit_cell_nodes,
    build_finite_cesrr_grid,
    build_quantitative_summary_markdown,
    effective_medium_cesrr_model,
    generate_all_figures,
    generate_setup_figures,
    run_task_bwo_start_current,
    run_task_cold_test,
    run_task_reversed_cherenkov,
    run_task_sparam_retrieval,
    verify_cesrr_coordinate_identities,
)

# %%
spec = CeSRRSWSSpec()
coord_check = verify_cesrr_coordinate_identities(spec)
_, _, _, mesh_meta = build_cesrr_unit_cell_nodes(spec, n_slot=4, graded=True)
_, finite_meta = build_finite_cesrr_grid(spec, n_cells=5, n_slot=4, n_buffer_cells=2)
print(
    f"CeSRR partition check: x_span = {coord_check['x_partition_mm']:.2f} mm, "
    f"z_span = {coord_check['z_partition_mm']:.2f} mm, "
    f"l_cir = {coord_check['slot_perimeter_mm']:.1f} mm"
)
print(
    f"Production unit-cell mesh: {mesh_meta['nx']}x{mesh_meta['ny']}x{mesh_meta['nz']} "
    f"= {mesh_meta['n_cells']} cells | "
    f"d_min = {mesh_meta['d_min_mm']:.3f} mm "
    f"({mesh_meta['cells_across_slot']:.1f} cells/slot, "
    f"{mesh_meta['cells_across_plate']:.1f} cells/plate, "
    f"PPW_min = {mesh_meta['ppw_min_at_3g5']:.1f})"
)
print(
    f"Production finite RCR/S-param mesh: {finite_meta['nx']}x{finite_meta['ny']}x{finite_meta['nz']} "
    f"= {finite_meta['n_cells']} cells | "
    f"d_min = {finite_meta['d_min_mm']:.3f} mm "
    f"({finite_meta['cells_across_slot']:.1f} cells/slot, "
    f"{finite_meta['cells_across_plate']:.1f} cells/plate)"
)

# %% [markdown]
# ### Geometry 1: CeSRR Unit-Cell Cold Test (`00a` / `00b`)
#
# - **Geometry (`00a_setup_geometry.png`)**: One $14.5\text{ mm} \times 14.5\text{ mm} \times 14.5\text{ mm}$
#   unit cell ($a = p = 14.5\text{ mm}$, main text l. 96, 98, 411; SI l. 44, 50, 324, 326) bounded
#   transversely by PEC square waveguide walls ($x, y = \pm 7.25\text{ mm}$) and axially by complex
#   Floquet–Bloch periodic boundaries ($\mathbf{E}(z + p) = e^{-i\phi}\mathbf{E}(z)$), loaded across
#   $y \in [-0.6, +0.6]\text{ mm}$ ($t = 1.2\text{ mm}$, main text l. 413) by the 4-paddle copper CeSRR plate.
# - **Mesh & Excitation (`00b_setup_mesh_and_excitation.png`)**: Interface-snapped graded Cartesian grid
#   (`n_slot=4`) resolving every $1.0\text{ mm}$ slot with $4.0$ cells ($\Delta_{\min} = 0.250\text{ mm}$)
#   and the $1.2\text{ mm}$ plate thickness with $4.0$ cells ($\Delta_{y,\text{plate}} = 0.300\text{ mm}$),
#   excited by an off-axis Gaussian current pulse $J_z(t), J_x(t)$ in the sheet-beam channel.
#
# ![Unit-cell cold-test setup geometry](figures/00a_setup_geometry.png)
# ![Unit-cell cold-test setup mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# | Section | Parameter | Value & Attribution |
# | :--- | :--- | :--- |
# | **Domain** | Extent $L_x \times L_y \times L_z$ | $14.5\text{ mm} \times 14.5\text{ mm} \times 14.5\text{ mm}$ ($a = 14.5\text{ mm}$ [main text l. 96, 411; SI l. 44, 324, 326], $p = 14.5\text{ mm}$ [main text l. 98; SI l. 50, 326]) |
# | | Grid cells & resolution | Snapped graded mesh (`n_slot=4`, $4.0$ cells/$1.0\text{ mm}$ slot, $4.0$ cells/$1.2\text{ mm}$ plate) |
# | **Boundaries** | $(x_{\pm}, y_{\pm}, z_{\pm})$ | $(\text{PEC}, \text{PEC}, \text{Bloch } e^{-i\phi})$ |
# | **Materials** | CeSRR plate ($t$ from paper; in-plane cutouts **assumed by us**) | PEC copper ($t = 1.2\text{ mm}$ [main text l. 413]; $b = 13.5\text{ mm}, h_1 = 5.25\text{ mm}, h_2 = 4.0\text{ mm}, d = g = 1.0\text{ mm}, j = 1.5\text{ mm}$ assumed by us; $0.5\text{ mm}$ beam clearance [main text l. 280; SI l. 56, 475]) |
# | **Sources** | Bloch driver (`J_z`, `J_x`) | Off-axis Gaussian pulse ($f_0 = 3.1\text{ GHz}, \Delta f = 1.8\text{ GHz}$) |
# | **Monitors** | `ResonanceMonitor` + `HannDFTMonitor` | Time-domain harminv probes + 3D staggered FIT mode field extraction |
#
# ### Geometry 2: Finite CeSRR Waveguide for Termination Reflection & Reversed Cherenkov Radiation (`00a_rcr` / `00b_rcr`)
#
# - **Termination design and passive reflection**: Because the $14.5\text{ mm} \times 14.5\text{ mm}$ hollow
#   square waveguide is below cutoff at $3.0\text{ GHz}$ ($f_{c,\text{TE}_{10}} = 10.34\text{ GHz}$),
#   terminating the CeSRR array into an empty below-cutoff waveguide section (`termination='smooth_air'`)
#   produces a pure standing wave ($|\Gamma_{\text{air}}| = 1$ in theory; our two-probe estimator recovers a
#   near-unity band median with per-frequency scatter quantified from JSON in Task 2 below).
#   In the laboratory experiment of Duan *et al.* (2017, SI Figs. 3–7; note SI Fig. 6a,b is an $S_{12}$ phase
#   plot [SI l. 146–149, 450–458]), coaxial monopole couplers transition the slow Bloch wave into $50\;\Omega$ ports.
#   In our 3D FIT finite-structure simulations, we continue the periodic CeSRR cells into buffer periods
#   on each end with a **matched electric–magnetic cubic adiabatic conductivity taper**
#   ($\sigma_e(z)/\varepsilon_0 = \sigma_m(z)/\mu_0 = \gamma(z)$, `termination='matched_taper'`),
#   absorbing the slow backward Bloch wave so that the post-bunch wake exits through the ports.
#
# ![Finite RCR setup geometry](figures/00a_setup_geometry_rcr.png)
# ![Finite RCR setup mesh and excitation](figures/00b_setup_mesh_and_excitation_rcr.png)
#
# | Section | Parameter | Value & Attribution |
# | :--- | :--- | :--- |
# | **Domain** | Extent $L_x \times L_y \times L_z$ | $14.5\text{ mm} \times 14.5\text{ mm} \times 130.5\text{ mm}$ ($N=5$ active CeSRR cells + $2$ adiabatic buffer cells/end; experiment has $N=20, L=290\text{ mm}$ [main text l. 411, 413; SI l. 117, 455]) |
# | | Grid cells & resolution | Snapped graded mesh (`n_slot=4`, $4.0$ cells/$1.0\text{ mm}$ slot, $4.0$ cells/$1.2\text{ mm}$ plate) |
# | **Boundaries** | $(x_{\pm}, y_{\pm}, z_{\pm})$ | $(\text{PEC}, \text{PEC}, \text{Matched electric--magnetic adiabatic CeSRR taper})$ |
# | **Sources** | `wakis.sources.Beam` (6-filament sheet vs. 1-filament line) | $3\times 2 = 6$ filaments at $x \in \{-4, 0, +4\}\text{ mm}, y \in \{1.6, 2.6\}\text{ mm}$ discretizing the $12\times 2\text{ mm}$ sheet beam [main text l. 199, 424; SI l. 58, 474–475], $0.5\text{ mm}$ clearance [main text l. 280; SI l. 56, 475], $\beta = 0.5474$ ($99.7\text{ keV}$ [main text l. 284; SI Eq. 5, l. 513–514]), $q = 1.0\text{ nC}, \sigma_z = 8.0\text{ mm}$ |
# | **Monitors** | Upstream Port 2 ($-z$) & Downstream Port 1 ($+z$) | Cross-sectional Poynting flux $P_{\text{Port 2,1}}(t)$, radiated wake energies $W_{\text{rad}}$, volumetric taper dissipation $W_{\text{abs}}$, and $yz$-plane $E_z(y,z,t)$ wavefield |

# %% [markdown]
# ---
#
# ## Results
#
# ### Task 1: Unit-Cell Cold Test — Bloch Dispersion, Exact $v_E = v_g < 0$, Pierce Impedance $K_c$ Diagnosis, and Mesh Convergence
#
# We sweep the inter-cell Bloch phase shift $\phi = \beta_0 p \in [15^\circ, 180^\circ]$ in 3D FIT
# and render all numerical results directly from `data/task_cold_test.json`:
#
# 1. **Stage 1 (Harminv Bloch sweep)**: At each $\phi$, a broadband Gaussian $J_z$ kick excites both
#    the fundamental **left-handed backward passband** (Mode 1, $v_g < 0, v_p > 0$, compared with the
#    paper's stated left-handed passband $2.83\text{--}3.05\text{ GHz}$ [main text l. 195, 203, 376;
#    SI l. 23, 352]; note that $2.82\text{--}3.32\text{ GHz}$ in SI Fig. 6a,b is the axis tick range
#    [SI l. 146–149, 450–458], not a passband) and the second **right-handed forward passband**
#    (Mode 2, $v_g > 0$, SI Fig. 2a). Harminv poles are filtered by quality factor $|Q| > 25$, finite
#    complex amplitude, and relative amplitude $|a_k| \ge 10^{-3}\max_{|Q|>25}|a|$ (preventing low-$|Q|$
#    spurious poles whose $\exp(-2\pi i F t_0)$ factor overflows at $t_0 > 0$ from affecting mode selection).
# 2. **Cold vs. slow space-charge wave synchronous frequency**: At $V_0 = 99.7\text{ keV}$ ($\beta = 0.5474$,
#    main text l. 284; SI Eq. 5, l. 513–514), the cold beam line $\omega = \beta c \beta_0$ intersects Mode 1
#    near the paper's cold-test intersection ("slightly below $2.98\text{ GHz}$", main text l. 294–295).
#    At the paper's particle-in-cell simulation current $I_0 = 1.55\text{ kA}$ (main text l. 199; SI l. 474),
#    the 1D slow space-charge wave line $\omega = \beta_0 v_0 - \omega_q$ shifts the intersection downward
#    toward the paper's hot particle-in-cell Port 2 peak at $2.836\text{ GHz}$ (main text l. 289; Port 1 at
#    $2.829\text{ GHz}$, l. 288; measured Port 2 at $2.850\text{ GHz}$ and Port 1 at $2.847\text{ GHz}$,
#    l. 297, 364). Because $f_q / f \approx 0.5$ and $\omega_p / 2\pi > f$ at $1.55\text{ kA}$, this
#    intense-beam operating point lies outside the regime of validity of linear space-charge wave theory.
# 3. **Stage 2 (`HannDFTMonitor` eigenmode extraction, exact $v_E = v_g$, and $K_c$ diagnosis)**:
#    Evaluating $U_{\text{cell}}$ on the native staggered FIT primal/dual cells confirms virial
#    equipartition $U_E / U_H \approx 1$ and $v_E = P_z p / U_{\text{cell}} = v_g < 0$. Checking candidate
#    $K_c$ normalisations (2-channel vs. 1-channel power, phasor factor of $1/2$, $n = 0$ vs. $n = \pm 1$
#    space harmonics, and transverse height $y$) against SI Note 2 ("averaged interaction impedance is
#    greater than $\sim 750\;\Omega$", SI l. 395–396; note that $0.72\text{--}1.32\text{ k}\Omega$ in
#    SI Fig. 2b is the y-axis tick range [SI l. 28, 50], not data) shows that the remaining gap to the
#    paper's $\sim 750\;\Omega$ is unexplained: slot width $d$ is ruled out, while independent variations
#    of $h_1, h_2, j$ are untested.
# 4. **Plain-plate control (`plain_plate=True`) & 3-level mesh convergence (`n_slot = [4, 6, 8]`)**:
#    Removing the CeSRR slots eliminates the left-handed passband ($0$ eigenmodes in $2.5\text{--}3.8\text{ GHz}$),
#    and refining the slot-snapped mesh across `n_slot = [4, 6, 8]` (all $\ge 4$ cells per feature) converges
#    both $f_1(95.7^\circ)$ and $\langle K_{c,0}\rangle_{\text{beam}}$ monotonically.

# %%
def load_or_run_all(quick: bool = False, force: bool = False) -> dict[str, dict]:
    """Load cached JSON task results or run the 3D FIT tasks sequentially."""
    cold_path = DATA_DIR / "task_cold_test.json"
    sp_path = DATA_DIR / "task_sparam_retrieval.json"
    rcr_path = DATA_DIR / "task_reversed_cherenkov.json"
    bwo_path = DATA_DIR / "task_bwo_start_current.json"

    if force or not cold_path.exists():
        run_task_cold_test(quick=quick)
    if force or not sp_path.exists():
        run_task_sparam_retrieval(quick=quick)
    if force or not rcr_path.exists():
        run_task_reversed_cherenkov(quick=quick)
    if force or not bwo_path.exists():
        run_task_bwo_start_current(quick=quick)

    return {
        "cold": json.loads(cold_path.read_text(encoding="utf-8")),
        "sparam": json.loads(sp_path.read_text(encoding="utf-8")),
        "rcr": json.loads(rcr_path.read_text(encoding="utf-8")),
        "bwo": json.loads(bwo_path.read_text(encoding="utf-8")),
    }


results = load_or_run_all(quick=False, force=False)
cold = results["cold"]
sparam = results["sparam"]
rcr = results["rcr"]
bwo = results["bwo"]
idx_sync_dft = int(np.argmin(np.abs(np.asarray(cold["dft_phi_deg"], dtype=float) - 95.7)))
kc_d_rel_shift_pct = (
    0.5
    * abs(cold["kc_geom_sensitivity_d_plus10_ohm"] - cold["kc_geom_sensitivity_d_minus10_ohm"])
    / cold["sync_99k7_kc_beam_avg_ohm"]
    * 100.0
)
kc_mesh_err_pct = cold["conv_kc_rel_err"][0] * 100.0
kc_mesh_to_d_ratio = kc_mesh_err_pct / max(kc_d_rel_shift_pct, 1e-12)

display(
    Markdown(
        f"**Task 1 Quantitative Highlights (rendered from `data/task_cold_test.json`):**\n"
        f"- **Fundamental LH Mode 1 passband**: `{cold['passband_low_fit_ghz']:.4f}`–`{cold['passband_high_fit_ghz']:.4f}` GHz "
        f"(paper LH passband `{cold['passband_low_paper_ghz']:.2f}`–`{cold['passband_high_paper_ghz']:.2f}` GHz, main text l. 195, 203, 376; SI l. 23, 352; "
        f"lower-edge rel. diff `{cold['passband_low_rel_err']*100:.2f}%`); **RH Mode 2**: `{min(cold['f2_rh_ghz']):.4f}`–`{max(cold['f2_rh_ghz']):.4f}` GHz.\n"
        f"- **Cold vs. slow space-charge wave synchronous frequency ($99.7\\text{{ keV}}, \\beta = 0.5474$)**: "
        f"computed intersection $\\phi_{{\\text{{sync}}}} = {cold['sync_99k7_phi_deg']:.2f}^\\circ$, $f_{{\\text{{sync,cold}}}} = {cold['sync_99k7_f_ghz']:.4f}\\text{{ GHz}}$ "
        f"(`{cold['sync_99k7_vs_cold_paper_rel_err']*100:.2f}%` above paper's cold intersection $\\approx {cold['sync_99k7_cold_paper_ghz']:.2f}\\text{{ GHz}}$, main text l. 294–295; "
        f"offset from nearest Stage-2 DFT point $\\phi_{{\\text{{DFT}}}} = {cold['sync_99k7_dft_phi_deg']:.1f}^\\circ$, $f_{{\\text{{DFT}}}} = {cold['sync_99k7_dft_f_ghz']:.4f}\\text{{ GHz}}$ is "
        f"$\\Delta\\phi = {cold['sync_99k7_dft_phi_offset_deg']:+.2f}^\\circ$, $\\Delta f = {cold['sync_99k7_dft_f_offset_ghz']*1e3:+.2f}\\text{{ MHz}}$). "
        f"At the paper's particle-in-cell simulation current $I_0 = 1.55\\text{{ kA}}$ (main text l. 199; SI l. 474), the 1D slow space-charge wave "
        f"($f_q = {cold['sync_99k7_scw_fq_ghz']:.3f}\\text{{ GHz}}$) intersects Mode 1 at $\\phi_{{\\text{{SCW}}}} = {cold['sync_99k7_scw_phi_deg']:.1f}^\\circ$, "
        f"$f_{{\\text{{SCW}}}} = {cold['sync_99k7_scw_f_ghz']:.4f}\\text{{ GHz}}$ (`{cold['sync_99k7_scw_vs_hot_sim_rel_diff']*100:.2f}%` from paper's "
        f"${cold['sync_99k7_rcr_sim_paper_ghz']:.3f}\\text{{ GHz}}$ particle-in-cell peak, main text l. 289; measured ${cold['sync_99k7_rcr_meas_paper_ghz']:.3f}\\text{{ GHz}}$, l. 297, 364).\n"
        f"- **Energy velocity vs. group velocity**: at $\\phi = 95.7^\\circ$, $v_E / c = {cold['sync_99k7_ve_over_c']:.4f}$ vs. "
        f"$v_g / c = {cold['sync_99k7_vg_over_c']:.4f}$ ($U_E / U_H = {cold['dft_ue_over_uh'][idx_sync_dft]:.4f}$, rel. diff `{cold['dft_ve_vg_rel_diff'][idx_sync_dft]*100:.2f}%`, "
        f"max across passband `{max(cold['dft_ve_vg_rel_diff'])*100:.2f}%`).\n"
        f"- **Pierce impedance $K_c$ diagnosis at $\\phi = 95.7^\\circ$**: 2-channel sheet average $\\langle K_{{c,0}}\\rangle_{{\\text{{2ch}}}} = {cold['sync_99k7_kc_beam_avg_ohm']:.2f}\\;\\Omega$ "
        f"(`{cold['sync_99k7_kc_beam_avg_from_ucell_vg_ohm']:.2f}` $\\Omega$ from $U_{{\\text{{cell}}}}|v_g|/p$), "
        f"1-channel upper-half power $\\langle K_{{c,0}}\\rangle_{{\\text{{1ch}}}} = {cold['sync_99k7_kc_beam_avg_upper_channel_ohm']:.2f}\\;\\Omega$, "
        f"no-$1/2$ phasor convention `{cold['sync_99k7_kc_beam_avg_no_half_2ch_ohm']:.2f}` / `{cold['sync_99k7_kc_beam_avg_no_half_1ch_ohm']:.2f}` $\\Omega$, "
        f"space harmonics $\\langle K_{{c,-1}}\\rangle = {cold['sync_99k7_kc_beam_avg_harm_minus1_ohm']:.2f}\\;\\Omega$ and "
        f"$\\langle K_{{c,+1}}\\rangle = {cold['sync_99k7_kc_beam_avg_harm_plus1_ohm']:.2f}\\;\\Omega$, "
        f"beam center $K_{{c,0}}(0, 2.1\\text{{ mm}}) = {cold['sync_99k7_kc_center_ohm']:.2f}\\;\\Omega$, "
        f"beam bottom edge $K_{{c,0}}(0, 1.1\\text{{ mm}}) = {cold['sync_99k7_kc_surface_ohm']:.2f}\\;\\Omega$ "
        f"(`{cold['sync_99k7_kc_upper_channel_surface_ohm']:.2f}` $\\Omega$ 1-ch), and plate surface "
        f"$K_{{c,0}}(0, 0.6\\text{{ mm}}) = {cold['sync_99k7_kc_plate_surface_ohm']:.2f}\\;\\Omega$ "
        f"(varying slot width $d$ by $\\pm 10\\%$ while preserving $2h_2 + 4d + j = b$ and $2h_1 + 2d + g = b$ on a slot-snapped grid shifts $\\langle K_{{c,0}}\\rangle_{{\\text{{2ch}}}}$ from "
        f"`{cold['kc_geom_sensitivity_d_minus10_ohm']:.2f}` to `{cold['kc_geom_sensitivity_d_plus10_ohm']:.2f}` $\\Omega$, or $\\pm {kc_d_rel_shift_pct:.2f}\\%$, which is $\\sim {kc_mesh_to_d_ratio:.0f}\\times$ smaller than the `{kc_mesh_err_pct:.2f}%` `n_slot=4` mesh discretization error; "
        f"the `{cold['kc_gap_factor_1ch_vs_si_min']:.1f}`–`{cold['kc_gap_factor_2ch_vs_si_min']:.1f}x` (`{cold['kc_gap_factor_1ch_vs_si_min']:.2f}x` 1-ch to `{cold['kc_gap_factor_2ch_vs_si_min']:.2f}x` 2-ch) gap to "
        f"the paper's $\\sim {cold['kc_si_note2_avg_min_ohm']:.0f}\\;\\Omega$ in SI Note 2, SI l. 395–396, is unexplained: $d$ is ruled out, while $h_1, h_2, j$ are untested).\n"
        f"- **3-level mesh convergence (`n_slot = {cold['conv_n_slot']}`, $\\Delta_{{\\min}} = {cold['conv_d_min_mm']}$ mm, cells $= {cold['conv_cells']}$)**: "
        f"$f_1(95.7^\\circ) = {[round(x, 4) for x in cold['conv_f1_ghz']]}\\text{{ GHz}}$ "
        f"(fine-grid difference `{cold['conv_f1_rel_err'][1]*100:.4f}%` between `n_slot=6` and `8`; Richardson extrapolation on $f_1$ is omitted because $p_{{\\text{{obs}}}}(f_1) = {cold['conv_f1_observed_order']:.2f}$ is below the FFT interpolation floor), "
        f"$\\langle K_{{c,0}}\\rangle_{{\\text{{beam}}}} = {[round(x, 2) for x in cold['conv_kc_ohm']]}\\;\\Omega$ "
        f"(observed order $p_{{\\text{{obs}}}}(K_c) = {cold['conv_kc_observed_order']:.2f}$, observed-order Richardson `{cold['conv_kc_richardson_ohm']:.2f}` $\\Omega$; "
        f"plain-plate LH modes in $[2.5, 3.8]\\text{{ GHz}}$: `{cold['plain_plate_lh_mode_count']}`)."
    )
)

# %% [markdown]
# ![Unit-cell Bloch dispersion and Pierce interaction impedance](figures/02_bloch_dispersion_and_pierce_impedance.png)
#
# ### Task 2: Passive Termination Reflection ($|\Gamma(f)| \le 1$), Band-Median Consistency Check, Standing-Wave Scatter & Effective-Medium Model Comparison
#
# First, we evaluate the finite-structure port reflection $|\Gamma(f)|$ across the cold-synchronism
# evaluation band (spanning the cold-beam synchronous region and excluding
# the lower band-edge region containing the 1D slow-SCW intersection and the paper's hot frequencies
# where $v_g \to 0$, with all band edges, code parameters, and error bounds rendered from JSON below)
# for three downstream terminations using a **two-probe Bloch-mode decomposition**. In the $7$-period
# grid (`n_cells = 3` central cells with `2` nominal buffer periods per end), cell $0$ ($z < -2.5p$) is
# the $1$-period low-side adiabatic taper, cell $1$ (the inner low-side buffer period) holds the source,
# cells $2\dots 4$ are the $3$ undamped active periods downstream of the source housing the two homologous
# probes at cells $3$ and $4$ ($2p$ and $3p$ downstream of the source in cell $1$, with both probe cells
# populated by CeSRR boxes across all terminations), and cells $5\dots 6$ ($z > +1.5p$) form the $2$-period
# downstream termination under test. We solve the $2\times 2$ forward/backward Bloch phase system with
# the 3D FIT cold-test phase $\phi_{\text{Bloch}}(f)$ after projecting $E_z(x, y, z_m, t)$ onto the
# assumed upper-channel transverse trial profile $\psi_{xy}(x,y) = \cos(\pi x/a)\exp(-(y - t/2)/(3\text{ mm}))$
# (any common frequency-dependent scale factor such as $\sqrt{\frac{1}{2}\phi^2 K_c}$ divides both probes
# identically and cancels in $\Gamma(f) = a_{\text{bwd}}(f)/a_{\text{fwd}}(f)$).
#
# Second, we plot Duan *et al.*'s (2017, SI Note 1 Eqs. 2–3, SI l. 340–350) Drude waveguide-plasma
# permeability $\mu_{\text{eff}}^{\text{model}}(f) < 0$ alongside our Lorentz illustration of
# $\varepsilon_{xx,\text{eff}}^{\text{model}}(f) < 0$ (parameters assumed by us).

# %%
sp_meta = sparam["mesh_meta"]
display(
    Markdown(
        f"**Task 2 Methodological Breakdown & Quantitative Highlights (rendered from `data/task_sparam_retrieval.json`):**\n"
        f"- **Evaluation band & 7-period layout (`n_cells = {sparam['n_cells']}` downstream of source, `{sp_meta['n_cells_buffer_each_end']}` nominal buffer cells/end, `{sp_meta['n_cells_total']}` total cells)**: "
        f"`{min(sparam['freqs_ghz']):.2f}`–`{max(sparam['freqs_ghz']):.2f}` GHz, spanning the `{min(rcr['beta_sweep_f_cold_ghz']):.3f}`–`{max(rcr['beta_sweep_f_cold_ghz']):.3f}` GHz cold-beam synchronous region "
        f"(and `{min(rcr['beta_sweep_f_rcr_ghz']):.3f}`–`{max(rcr['beta_sweep_f_rcr_ghz']):.3f}` GHz finite-structure RCR peaks) across $\\beta \\in [{min(rcr['beta_sweep']):.4f}, {max(rcr['beta_sweep']):.4f}]$, "
        f"while excluding `{cold['passband_low_fit_ghz']:.2f}`–`{min(sparam['freqs_ghz']):.2f}` GHz near the lower band edge (where $v_g \\to 0$ and the 1D slow-SCW intersection lies at `{cold['sync_99k7_scw_f_ghz']:.3f}` GHz).\n"
        f"- **Partial-reflector band-median consistency check (`termination='partial_taper'`, `{min(sparam['freqs_ghz']):.2f}`–`{max(sparam['freqs_ghz']):.2f}` GHz — does NOT validate $\\phi(f)$ per frequency)**: "
        f"Because a lossless standing wave has in-phase probe signals (real ratio $r = b_{{-1}}/b_0 \\in \\mathbb{{R}}$) so that $|a_{{\\text{{bwd}}}}/a_{{\\text{{fwd}}}}| = |r - e^{{-i\\phi}}|/|r - e^{{i\\phi}}| = 1$ identically for *any* phase $\\phi$, "
        f"we compare the two-probe estimator on a partial reflector (downstream taper with reduced peak conductivity $\\gamma_{{\\max}} = {sp_meta['gamma_max_partial_taper_s']:.2e}\\text{{ s}}^{{-1}}$ over a `{sp_meta['flux_single_pass_window_ns']:.1f}` ns post-pulse run) "
        f"against a cross-sectional Poynting energy-flux estimate $|\\Gamma_{{\\text{{flux}}}}(f)| = \\sqrt{{\\max(0, 1 - P_{{\\text{{net}}}}(f)/P_{{\\text{{inc}}}}(f))}}$ integrated over an assumed `{sp_meta['flux_single_pass_window_ns']:.1f}` ns causal single-pass round-trip window. "
        f"Crucially, this comparison agrees **only on the band median**: two-probe band-median $|\\Gamma_{{\\text{{2probe}}}}| = {sparam['partial_reflector_band_median_2probe']:.4f}$ (`{sparam['termination_partial_taper_refl_median_db']:.4f}` dB, with per-frequency estimator uncertainty $\\pm {sparam['standing_wave_gamma_air_max_abs_err']:.4f}$ from the standing-wave check below) vs. "
        f"flux band-median $|\\Gamma_{{\\text{{flux}}}}| = {sparam['partial_reflector_band_median_flux']:.4f}$ (`{sparam['termination_partial_taper_flux_refl_median_db']:.4f}` dB; band-median absolute difference `{sparam['partial_reflector_band_median_abs_diff']:.4f} <= {sparam['partial_reflector_band_median_tol']:.2f}`, `partial_reflector_band_median_consistency_within_tol`: `{sparam['partial_reflector_band_median_consistency_within_tol']}`). "
        f"**Per frequency, the two $|\\Gamma(f)|$ curves disagree**: $|\\Gamma_{{\\text{{2probe}}}}(f)|$ rises from `{sparam['partial_reflector_2probe_min']:.4f}` to `{sparam['partial_reflector_2probe_max']:.4f}` across `{min(sparam['freqs_ghz']):.2f}`–`{max(sparam['freqs_ghz']):.2f}` GHz, whereas $|\\Gamma_{{\\text{{flux}}}}(f)|$ is clipped to `0` at `{sparam['partial_reflector_flux_clipped_count']}` frequencies (`{sparam['partial_reflector_flux_clipped_f_min_ghz']:.2f}`–`{sparam['partial_reflector_flux_clipped_f_max_ghz']:.2f}` GHz where $P_{{\\text{{net}}}} \\ge P_{{\\text{{inc}}}}$), peaks at `{sparam['partial_reflector_flux_max']:.4f}` near `{sparam['partial_reflector_flux_peak_freq_ghz']:.2f}` GHz, and falls to `{sparam['partial_reflector_flux_end_high_freq']:.4f}` at `{max(sparam['freqs_ghz']):.2f}` GHz "
        f"(per-frequency $\\max|\\Delta|\\Gamma|| = {sparam['partial_reflector_per_freq_max_abs_diff']:.4f}$, RMS difference `{sparam['partial_reflector_per_freq_rms_diff']:.4f}`, Pearson correlation $r = {sparam['partial_reflector_per_freq_corr']:.4f}$). "
        f"Moreover, $P_{{\\text{{inc}}}}(f)$ is taken as $P_{{\\text{{net}}}}(f)$ of the `matched_taper` run, which itself carries residual reflections up to $|\\Gamma_{{\\text{{taper}}}}| = {sparam['matched_taper_peak_gamma_mag']:.4f}$ (`{sparam['termination_matched_taper_refl_peak_db']:.2f}` dB), biasing $|\\Gamma_{{\\text{{flux}}}}|^2 = 1 - P_{{\\text{{net}}}}/P_{{\\text{{inc}}}}$ by $|\\Gamma_{{\\text{{taper}}}}|^2(1 - |\\Gamma|^2)/(1 - |\\Gamma_{{\\text{{taper}}}}|^2)$, i.e. of order $|\\Gamma_{{\\text{{taper}}}}|^2 = {sparam['matched_taper_peak_gamma_sq_bias']:.4f}$ at the taper's peak and producing the `{sparam['partial_reflector_flux_clipped_count']}` clipped points. "
        f"A phase-sensitivity scan across $\\Delta\\phi \\in [{min(sparam['partial_reflector_dphi_grid_rad']):+.1f}, {max(sparam['partial_reflector_dphi_grid_rad']):+.1f}]\\text{{ rad}}$ and scale factors $s \\in [{min(sparam['partial_reflector_phi_scale_grid']):.1f}, {max(sparam['partial_reflector_phi_scale_grid']):.1f}]$ confirms that the band-median check (`tol = {sparam['partial_reflector_band_median_tol']:.2f}`) does **not** validate $\\phi(f)$ per frequency: "
        f"it still passes for $\\Delta\\phi \\in [{sparam['partial_reflector_dphi_pass_min_rad']:+.1f}, {sparam['partial_reflector_dphi_pass_max_rad']:+.1f}]\\text{{ rad}}$ (symmetric range $|\\Delta\\phi| \\le {sparam['partial_reflector_dphi_sym_pass_max_rad']:.1f}\\text{{ rad}}$), for $\\phi \\to s\\phi$ with $s \\in [{sparam['partial_reflector_phi_scale_pass_min']:.1f}, {sparam['partial_reflector_phi_scale_pass_max']:.1f}]$, and even for a constant frequency-independent phase $\\phi(f) = \\mathrm{{median}}(\\phi)$ (giving band-median $|\\Gamma| = {sparam['partial_reflector_const_median_phi_gamma_med']:.4f}$, passing: `{sparam['partial_reflector_const_median_phi_passes']}`), failing first at $\\Delta\\phi = {sparam['partial_reflector_dphi_first_failing_signed_rad']:+.1f}\\text{{ rad}}$ (one-sided) and for both signs only at $|\\Delta\\phi| \\ge {sparam['partial_reflector_dphi_both_signs_fail_abs_rad']:.1f}\\text{{ rad}}$.\n"
        f"- **Standing-wave sanity check & per-frequency estimator error (`termination='smooth_air'`, `{min(sparam['freqs_ghz']):.2f}`–`{max(sparam['freqs_ghz']):.2f}` GHz)**: "
        f"Leaving the `{sp_meta['n_cells_buffer_each_end']}` downstream buffer periods (`ic = 5, 6`, $z > +1.5p$, while cells `ic = 0..4` including both probes `c_probes = [3, 4]` contain CeSRR boxes) as empty $14.5\\text{{ mm}}$ square waveguide ($f_{{c,\\text{{TE}}_{{10}}}} = 10.34\\text{{ GHz}}$) forms a lossless below-cutoff standing-wave reflector. "
        f"While the band-median reflection is $|\\Gamma_{{\\text{{air}}}}|_{{\\text{{med}}}} = {sparam['standing_wave_gamma_air_median']:.4f}$ (`{sparam['termination_smooth_air_refl_median_db']:.4f}` dB, passing the weak band-median check $|\\mathrm{{med}}(|\\Gamma_{{\\text{{air}}}}|) - 1| \\le {sparam['standing_wave_band_median_tol']:.2f}$: `{sparam['standing_wave_sanity_check_gamma_unity_within_tol']}`), "
        f"**per-frequency $|\\Gamma_{{\\text{{air}}}}(f)|$ ranges from `{sparam['standing_wave_gamma_air_min']:.4f}` to `{sparam['standing_wave_gamma_air_max']:.4f}`** (peak `{sparam['termination_smooth_air_refl_peak_db']:+.2f}` dB, exceeding passivity $|\\Gamma| = 1$ in Fig. 01(c)), "
        f"with `{sparam['standing_wave_gamma_air_frac_above_1p04']*100:.1f}%` of frequencies exceeding `{1.0 + sparam['standing_wave_band_median_tol']:.2f}` (and `{sparam['standing_wave_gamma_air_frac_outside_4pct']*100:.1f}%` outside $1 \\pm {sparam['standing_wave_band_median_tol']:.2f}$). "
        f"Because an ideal single-mode standing wave yields $|\\Gamma| = 1$ identically for any $\\phi$, this scatter is the two-probe estimator's own per-frequency error ($\\max||\\Gamma_{{\\text{{air}}}}(f)| - 1| = {sparam['standing_wave_gamma_air_max_abs_err']:.4f}$), "
        f"driven by higher-order/evanescent modes excited at the CeSRR-to-air discontinuity `{sp_meta['probe4_to_interface_grid_periods']:.2f}`$p$ downstream of probe 4 and by the finite record (pulse plus `{sp_meta['t_post_used_ns']:.1f}` ns, rectangular with a 1.5 ns $\\sin^2$ end taper). "
        f"Consequently, the band-median test is strictly weaker than a per-frequency tolerance test, which cannot pass because `{sparam['standing_wave_gamma_air_frac_above_1p04']*100:.1f}%` of evaluated frequencies exceed `{1.0 + sparam['standing_wave_band_median_tol']:.2f}`; we carry this per-frequency estimator uncertainty ($\\pm {sparam['standing_wave_gamma_air_max_abs_err']:.4f}$ in $|\\Gamma|$) into every termination-reflection claim.\n"
        f"- **Matched adiabatic CeSRR taper (`termination='matched_taper'`, `{min(sparam['freqs_ghz']):.2f}`–`{max(sparam['freqs_ghz']):.2f}` GHz, excluding `{cold['passband_low_fit_ghz']:.2f}`–`{min(sparam['freqs_ghz']):.2f}` GHz)**: "
        f"Extending the periodic CeSRR geometry into buffer cells per end with balanced conductivity $\\sigma_e(z)/\\varepsilon_0 = \\sigma_m(z)/\\mu_0 = \\gamma(z)$ ($\\gamma_{{\\max}} = {sp_meta['gamma_max_matched_taper_s']:.2e}\\text{{ s}}^{{-1}}$ over `{sp_meta['t_post_used_ns']:.1f}` ns) gives "
        f"band-median $|\\Gamma_{{\\text{{taper}}}}| = {sparam['termination_matched_taper_refl_median_db']:.4f}\\text{{ dB}}$ and peak $|\\Gamma_{{\\text{{taper}}}}| = {sparam['matched_taper_peak_gamma_mag']:.4f}$ (`{sparam['termination_matched_taper_refl_peak_db']:.2f}` dB, subject to the two-probe per-frequency estimator uncertainty $\\pm {sparam['standing_wave_gamma_air_max_abs_err']:.4f}$; passivity $|\\Gamma| \\le 1$: `{sparam['passivity_gamma_taper_le_one']}`; note that SI Fig. 6a,b of Duan *et al.* 2017 is an $S_{{12}}$ phase plot [SI l. 146–149, 450–458], not a coupler $|S_{{11}}|$ curve).\n"
        f"- **Effective-medium values at $f = 3.00\\text{{ GHz}}$**: Duan *et al.* (SI Eqs. 2–3) Drude permeability "
        f"$\\mathrm{{Re}}(\\mu_{{\\text{{eff}}}}^{{\\text{{model}}}}) = {sparam['mu_eff_model_real_at_3g0']:.4f} < 0$; "
        f"Lorentz illustration (parameters assumed by us) $\\mathrm{{Re}}(\\varepsilon_{{xx,\\text{{eff}}}}^{{\\text{{model}}}}/\\varepsilon_0) = {sparam['eps_xx_eff_model_real_at_3g0']:.4f} < 0$, "
        f"$\\mathrm{{Re}}(n_{{\\text{{eff}}}}^{{\\text{{model}}}}) = {sparam['n_eff_model_real_at_3g0']:.4f} < 0$."
    )
)

# %% [markdown]
# ![CeSRR geometry, Duan et al. (2017) Drude model & Lorentz illustration, and termination reflection](figures/01_cesrr_geometry_and_effective_medium.png)
#
# ### Task 3: Reversed Cherenkov Radiation (`wakis.sources.Beam`), Radiative Port Exit, Sheet-vs.-Line Beam & Energy Bookkeeping
#
# We propagate a distributed **6-filament relativistic sheet-beam bunch** (`wakis.sources.Beam`
# with $3 \times 2 = 6$ filaments at $x \in \{-4.0, 0.0, +4.0\}\text{ mm}$ and $y \in \{1.6, 2.6\}\text{ mm}$
# discretizing the $12\times 2\text{ mm}$ sheet beam [main text l. 199, 424; SI l. 58, 474–475],
# $\beta = 0.5474$, $V_0 = 99.7\text{ keV}$ [main text l. 284; SI Eq. 5, l. 513–514], total charge
# $q = 1.0\text{ nC}$) in the $+\hat{\mathbf{z}}$ direction through the $N = 5$ active CeSRR waveguide
# terminated at both ends by matched adiabatic tapers, and render all numerical outputs from
# `data/task_reversed_cherenkov.json`:
#
# - **Radiative wake decay through the ports**: Once the bunch clears the active CeSRR region at
#   $t_{\text{clear}}$, the stored electromagnetic energy $U_{\text{EM}}(t)$ decays as the backward
#   Cherenkov wake exits through the ports into the adiabatic tapers.
# - **Cross-sectional Poynting port ratio (direct downstream leakage vs. paper's Coupler 2 reflection)**:
#   Integrating the outgoing cross-sectional Poynting flux $P_{\text{Port 2}}(t)$ and $P_{\text{Port 1}}(t)$
#   over the post-bunch window yields the direct radiative port energy ratio $10\log_{10}(W_{\text{Port 2}}/W_{\text{Port 1}})$.
#   In Duan *et al.* (2017, main text l. 288–291, 364–375; SI l. 478–482, 599–608), the Port 1 signal
#   is the backward wave reflected from Coupler 2 after a $\sim 5.5\text{ ns}$ delay, whereas our
#   both-ends-absorbed FIT simulation has no reflecting coaxial coupler and therefore measures direct
#   downstream wake leakage at Port 1.
# - **Spectral resolution & velocity sweep**: Across $\beta \in \{0.5000, 0.5474, 0.6000, 0.6471\}$
#   (compare SI Fig. 15 voltage tuning, main text l. 379–386; SI l. 311–314, 610–613), the error bars in
#   Figure 3c show the half-Rayleigh window bandwidth $\pm \Delta f_{\text{Rayleigh}}/2 = \pm 1/(2\Delta T_{\text{wake}})$
#   (a deterministic window bandwidth, not a statistical peak-fit uncertainty; individual step-to-step
#   increments are smaller than $\Delta f_{\text{Rayleigh}}/2$, while only the full $\beta \in [0.50, 0.6471]$
#   span is comparable to $\Delta f_{\text{Rayleigh}}/2$).
# - **6-filament sheet beam vs. single line-charge filament & plain-plate control**: Replacing the
#   6-filament sheet beam by a single on-axis line charge increases the upstream wake energy because the
#   sheet spreads charge into weaker-field regions near $x = \pm 4\text{ mm}$ and $y = 2.6\text{ mm}$;
#   replacing the CeSRR plate with a solid plain plate (`plain_plate=True`) suppresses upstream radiation
#   by the plain-plate suppression factor rendered from JSON below.
# - **Discrete FIT Poynting & inner-taper volumetric absorber energy conservation**: Both the transit-phase
#   leapfrog work balance and the post-clear energy budget across the control volume $[z_{k_1}, z_{k_2}]$
#   close within the discrete time-centring tolerance. Because the flux planes $z_{k_1}, z_{k_2}$ sit $6$
#   grid cells into the adiabatic tapers, $W_{\text{abs,wake}} = \sum_t \sum_{\text{inner taper}} (\sigma_e |\mathbf{E}|^2 + \sigma_m |\mathbf{H}|^2)\,\Delta t\,\Delta V$
#   accounts for the $6$ inner taper cells ($\sim 0.03\%$ of the wake), whereas the radiated wake energy
#   ($W_{\text{rad,Port 2}} + W_{\text{rad,Port 1}}$) crosses the flux planes and is absorbed in the outer
#   buffer taper beyond the flux planes.

# %%
asym_vals_db = [float(x) for x in rcr["asymmetry_plane_values_db"]]
asym_min_db = min(asym_vals_db)
asym_max_db = max(asym_vals_db)
asym_diff_sim_min_db = asym_min_db - float(rcr["asymmetry_paper_sim_db"])
asym_diff_sim_max_db = asym_max_db - float(rcr["asymmetry_paper_sim_db"])
asym_diff_meas_min_db = asym_min_db - float(rcr["asymmetry_paper_meas_db"])
asym_diff_meas_max_db = asym_max_db - float(rcr["asymmetry_paper_meas_db"])

display(
    Markdown(
        f"**Task 3 Quantitative Highlights (rendered from `data/task_reversed_cherenkov.json`):**\n"
        f"- **Post-bunch wake window**: $\\Delta T_{{\\text{{wake}}}} = {rcr['wake_window_ns']:.2f}\\text{{ ns}}$, "
        f"Rayleigh resolution $\\Delta f_{{\\text{{Rayleigh}}}} = {rcr['rayleigh_df_ghz']:.4f}\\text{{ GHz}}$ "
        f"(half-Rayleigh window bandwidth $\\pm \\Delta f_{{\\text{{Rayleigh}}}}/2 = \\pm {0.5*rcr['rayleigh_df_ghz']*1e3:.1f}\\text{{ MHz}}$).\n"
        f"- **Upstream RCR spectral peak ($\beta = 0.5474$)**: $f_{{\\text{{RCR}}}} = {rcr['f_peak_99k7_ghz']:.4f}\\text{{ GHz}}$ "
        f"(`{rcr['f_peak_vs_cold_rel_err']*100:.2f}%` from cold-test synchronous frequency `{rcr['f_cold_sync_99k7_ghz']:.4f}` GHz; "
        f"paper cold intersection $\\approx {rcr['f_cold_paper_99k7_ghz']:.2f}\\text{{ GHz}}$ [main text l. 294–295]; "
        f"paper $1.55\\text{{ kA}}$ particle-in-cell peak `{rcr['f_hot_rcr_sim_paper_ghz']:.3f}` GHz [main text l. 289]).\n"
        f"- **Radiative port wake energies, systematic flux-plane dependence & direct-leakage vs. Coupler-2-reflection asymmetry**: "
        f"at $6$ cells into the taper, $W_{{\\text{{rad,Port 2}}}} = {rcr['w_wake_up_j']*1e6:.4f}\\;\\mu\\text{{J}}$ upstream vs. "
        f"$W_{{\\text{{rad,Port 1}}}} = {rcr['w_wake_dn_j']*1e6:.4f}\\;\\mu\\text{{J}}$ downstream (`{rcr['asymmetry_99k7_db']:.2f}` dB), "
        f"with a systematic flux-plane dependence across plane offsets `{rcr['asymmetry_plane_offsets_cells']}` cells into the taper: "
        f"`{' -> '.join(f'{x:.2f}' for x in asym_vals_db)}` dB, spanning `{asym_min_db:.1f}`–`{asym_max_db:.1f}` dB "
        f"(`{asym_min_db:.2f}`–`{asym_max_db:.2f}` dB, spread `{rcr['asymmetry_plane_spread_db']:.2f}` dB). "
        f"Across this `{asym_min_db:.1f}`–`{asym_max_db:.1f}` dB range, our both-ends-absorbed ratio differs by "
        f"`{asym_diff_sim_min_db:+.2f}` to `{asym_diff_sim_max_db:+.2f}` dB (`{abs(asym_diff_sim_max_db):.1f}`–`{abs(asym_diff_sim_min_db):.1f}` dB below) "
        f"vs. the paper's `{rcr['asymmetry_paper_sim_db']:.1f}` dB simulated ratio [main text l. 288] "
        f"and by `{asym_diff_meas_min_db:+.2f}` to `{asym_diff_meas_max_db:+.2f}` dB (`{abs(asym_diff_meas_max_db):.1f}`–`{abs(asym_diff_meas_min_db):.1f}` dB below) "
        f"vs. the paper's `{rcr['asymmetry_paper_meas_db']:.1f}` dB measured ratio [main text l. 364], "
        f"because the paper's Port 1 signal is the Coupler 2 reflection [SI l. 479–481] whereas our value is direct downstream leakage.\n"
        f"- **Radiative exit & inner-taper volumetric absorption**: $U_{{\\text{{EM}}}}(t_{{\\text{{clear}}}}) = {rcr['energy_stored_at_clear_j']*1e6:.4f}\\;\\mu\\text{{J}} \\to "
        f"U_{{\\text{{EM}}}}(T) = {rcr['energy_stored_em_j']*1e6:.4f}\\;\\mu\\text{{J}}$ (`{rcr['fraction_wake_radiated']*100:.2f}%` radiated out ports and absorbed beyond the flux planes), "
        f"inner-6-cell taper-absorbed energy inside the control volume $W_{{\\text{{abs,wake}}}} = \\sum_t \\sum_{{\\text{{inner taper}}}} (\\sigma|\\mathbf{{E}}|^2 + \\sigma_m|\\mathbf{{H}}|^2)\\,dt\\,dV = {rcr['energy_absorber_removed_wake_j']*1e6:.4f}\\;\\mu\\text{{J}}$ "
        f"(transit balance error `{rcr['energy_bookkeeping_rel_err']*100:.4f}%`, wake decay closure error `{rcr['energy_wake_decay_rel_err']*100:.4f}%`).\n"
        f"- **6-filament sheet beam vs. line beam & plain-plate suppression**: sheet-to-line upstream wake energy ratio "
        f"`{rcr['sheet_vs_line_wake_energy_ratio']:.4f}` (`{rcr['w_wake_up_j']*1e6:.4f}` vs. `{rcr['w_wake_up_line_beam_j']*1e6:.4f}` $\\mu\\text{{J}}$); "
        f"plain-plate upstream suppression `{rcr['plain_plate_suppression_db']:.2f}` dB.\n"
        f"- **Velocity sweep $f_{{\\text{{RCR}}}}(\\beta)$**: "
        + ", ".join(
            f"$\\beta={b:.4f} \\to {f:.4f}\\text{{ GHz}}$ (cold ${fc:.4f}\\text{{ GHz}}$)"
            for b, f, fc in zip(
                rcr["beta_sweep"], rcr["beta_sweep_f_rcr_ghz"], rcr["beta_sweep_f_cold_ghz"]
            )
        )
        + "."
    )
)

# %% [markdown]
# ![Reversed Cherenkov spectra, upstream/downstream asymmetry, and velocity tuning (error bars in panel (c) show only the deterministic half-Rayleigh window bandwidth ±Δf_Rayleigh/2, not statistical uncertainty)](figures/03_reversed_cherenkov_spectra_and_velocity_tuning.png)
# ![Reversed Cherenkov wavefronts and discrete FIT energy bookkeeping](figures/04_rcr_wavefronts_and_energy_bookkeeping.png)
# ![Animated reversed Cherenkov wavefronts: phase advances forward (+z) while energy flows backward (-z)](figures/reversed_cherenkov_wavefronts.gif)
#
# ### Task 4: Backward-Wave Oscillator (BWO) Start Current & Consistent `LinearizedBeam` Small-Signal Regime
#
# Finally, we evaluate the Johnson (1955) start-oscillation current $I_{\text{st}}(N)$ for the
# $N = 20$ CeSRR interaction circuit of Duan *et al.* (2017, main text l. 411, 413; SI l. 117, 455:
# $L = 290.0\text{ mm}$, alongside $N = 17$ from Duan *et al.* 2014 and Hummelt *et al.* 2014) and
# clearly separate the **self-consistent small-signal start-oscillation regime** ($I_0 = I_{\text{st}}$)
# from the **intense-beam relativistic particle-in-cell simulation regime** ($I_0 = 1.55\text{ kA}$,
# main text l. 199; SI l. 474), rendering all numerical values from `data/task_bwo_start_current.json`:

# %%
display(
    Markdown(
        f"**Task 4 Quantitative Highlights (rendered from `data/task_bwo_start_current.json`):**\n"
        f"- **Self-consistent start-oscillation regime ($I_0 = I_{{\\text{{st}}}}$)**: for $N = {bwo['n_cells_paper']}$ cells "
        f"($L = {bwo['length_paper_mm']:.1f}\\text{{ mm}}$) at $f_{{\\text{{sync}}}} = {bwo['f_sync_ghz']:.4f}\\text{{ GHz}}$ with "
        f"$\\langle K_{{c,0}}\\rangle_{{\\text{{beam}}}} = {bwo['kc_beam_avg_sync_ohm']:.2f}\\;\\Omega$, Johnson's start current is "
        f"$I_{{\\text{{st}}}} = {bwo['i_start_beam_avg_a']:.4f}\\text{{ A}}$ (`{bwo['i_start_surface_a']:.4f}` A at the bottom-edge impedance "
        f"$K_{{c,\\text{{surf}}}} = {bwo['kc_surface_sync_ohm']:.2f}\\;\\Omega$, and `{bwo['i_start_17cells_beam_avg_a']:.4f}` A for $N = 17$). "
        f"At $I_0 = I_{{\\text{{st}}}}$, the unreduced plasma frequency is $f_p(I_{{\\text{{st}}}}) = {bwo['plasma_freq_ghz']:.4f}\\text{{ GHz}}$, "
        f"the reduced plasma frequency is $f_q(I_{{\\text{{st}}}}) = {bwo['reduced_plasma_freq_ghz']:.4f}\\text{{ GHz}} \\ll f_{{\\text{{sync}}}}$ "
        f"(linear reduction factor $R_{{\\text{{sc}}}} = {bwo['reduction_factor_r']:.4f}$), Pierce's space-charge parameter is "
        f"$QC(I_{{\\text{{st}}}}) = {bwo['qc_space_charge_at_ist']:.4f} < 1$, and coupling `SolverFIT3D` to `LinearizedBeam` yields bounded "
        f"small-signal modulation across `{bwo['linearized_beam_active_steps']}` active steps with peak $|J_{{1z}}| = {bwo['linearized_beam_max_j1_abs']:.2f}\\text{{ A/m}}^2$.\n"
        f"- **Intense-beam particle-in-cell simulation regime ($I_0 = {bwo['i_beam_paper_a']:.0f}\\text{{ A}}$, `{bwo['overdrive_ratio_beam_avg']:.0f}x` above $I_{{\\text{{st}}}}$)**: "
        f"at the paper's particle-in-cell simulation current $I_0 = 1.55\\text{{ kA}}$ (main text l. 199; SI l. 474), "
        f"$f_p = {bwo['pic_1550a_plasma_freq_ghz']:.4f}\\text{{ GHz}} > f$ and $f_q = {bwo['pic_1550a_reduced_plasma_freq_ghz']:.4f}\\text{{ GHz}}$ "
        f"($f_q / f \\approx 0.5$), outside the regime of validity of linear space-charge wave theory."
    )
)

# %% [markdown]
# ![Mesh convergence and Johnson BWO start-oscillation current](figures/05_mesh_convergence_and_bwo_start_current.png)
#
# ---
#
# ## Quantitative Summary
#
# Every numerical value in the table below is rendered dynamically at runtime from `data/*.json` via
# `build_quantitative_summary_markdown()` and verified by `tests/test_075_metamaterial_sws_reversed_cherenkov.py`.

# %%
display(Markdown(build_quantitative_summary_markdown(DATA_DIR)))

# %% [markdown]
# ---
#
# ## Limits and Failure Modes
#
# 1. **Effective-medium homogenization breakdown at $\beta_0 p = 95.7^\circ$ ($p \approx \lambda_g / 3.76$, Bragg spatial dispersion)**:
#    - *Assumption*: Analytic continuum models treat the periodic CeSRR waveguide as a locally homogeneous
#      medium ($k_0 p \ll 1$ and $\beta_0 p \ll \pi$).
#    - *Counter-case / failure mode*: Even though the free-space electrical size at $f_{\text{sync}} \approx 3.00\text{ GHz}$
#      is sub-wavelength ($p = 14.5\text{ mm} \approx \lambda_0 / 6.9$), the guided wave is slowed to
#      $v_p \approx 0.547 c$, so the synchronous guided wavelength is $\lambda_g = 2\pi / \beta_0 \approx 54.6\text{ mm}$
#      and the inter-cell phase shift is $\phi_{\text{sync}} = \beta_0 p = 95.7^\circ = 0.532\pi$
#      ($p \approx \lambda_g / 3.76$). A phase advance of $\sim \pi / 2$ per unit cell strictly
#      **rules out local effective-medium homogenization**: spatial dispersion and periodic Bragg scattering
#      bend the dispersion curve $\omega(\beta_0)$ so that $v_g(\pi) = d\omega / d\beta_0|_{\pi} = 0$
#      at the Brillouin-zone edge, whereas a local Lorentz–Drude continuum model predicts a non-zero slope
#      and a resonant divergence $\varepsilon_{xx,\text{eff}}^{\text{model}} \to -\infty$. Therefore,
#      **exact Floquet–Bloch dispersion $\omega(\beta_0)$ and space-harmonic Pierce impedance
#      $K_{c,0}(\beta_0)$ are mandatory** for quantitative SWS design.
#
# 2. **Below-cutoff waveguide port termination vs. matched adiabatic taper or coaxial coupler**:
#    - *Assumption*: Smooth square waveguide buffer sections absorb outgoing signals without reflection.
#    - *Counter-case / failure mode*: Because the $14.5\text{ mm}$ square waveguide is far below cutoff
#      at $3\text{ GHz}$ ($f_{c,\text{TE}_{10}} = 10.34\text{ GHz}$), a lossless transition from the
#      propagating CeSRR metamaterial ($Z_{\text{Bloch}} > 0$) to an empty below-cutoff waveguide tunnel
#      has a purely reactive wave impedance ($Z_{\text{tunnel}} = +i|Z_{\text{tunnel}}|$), acting as a
#      standing-wave reflector ($|\Gamma_{\text{air}}| = 1$ in theory; see item 4 below for the two-probe
#      estimator's per-frequency error on this termination).
#      Moreover, in the experiment and particle-in-cell simulation of Duan *et al.* (2017, main text
#      l. 288–291, 364–375; SI l. 478–482, 599–608), partial reflection at the upstream coaxial coupler
#      (Coupler 2) sends a delayed forward wave to Port 1, whereas our both-ends-absorbed adiabatic taper
#      (`termination='matched_taper'`) measures only the direct upstream-to-downstream wake leakage.
#
# 3. **Rigid-bunch (`Beam`) and linear space-charge wave (`LinearizedBeam`) models vs. intense-beam particle-in-cell**:
#    - *Assumption*: A rigid bunch (`wakis.sources.Beam`) moves at constant velocity $\mathbf{v} = \beta c\hat{\mathbf{z}}$
#      without space-charge depression or energy depletion, and 1D linear space-charge wave theory
#      (`LinearizedBeam`, $\omega = \beta_0 v_0 - \omega_q$) assumes weak space charge ($f_q / f \ll 1$,
#      $f_p < f$, $QC \ll 1$) around a uniform DC velocity $v_0$, which holds near the BWO start threshold
#      $I_{\text{st}} \sim 1\text{ A}$ ($f_q / f \approx 1.3\%$, $QC_{\text{st}} \approx 0.29$).
#    - *Counter-case / failure mode*: In the intense-beam $I_0 = 1.55\text{ kA}$ particle-in-cell simulation
#      of Duan *et al.* (2017, main text l. 199, 288–289; SI l. 474), the unreduced plasma frequency
#      exceeds the wave frequency ($f_p > f$) and the tunnel-reduced plasma frequency reaches
#      $f_q / f \approx 0.5$. Consequently, $I_0 = 1.55\text{ kA}$ lies outside the regime of validity
#      of linear space-charge wave theory and requires self-consistent 3D relativistic particle-in-cell
#      simulation.

# %%
display(
    Markdown(
        f"4. **Two-probe Bloch reflection estimator and partial-reflector band-median consistency check**:\n"
        f"   - *Assumptions*: The two-probe estimator (`decompose_two_mode_bloch_reflection`) projects $E_z(x, y, z_m, t)$ at two homologous cell planes (`c_probes = {sp_meta['probe_cell_indices']}`) onto a **single assumed frequency-independent transverse trial profile** $\\psi_{{xy}}(x,y) = \\cos(\\pi x/a)\\exp(-(y - t/2)/(3\\text{{ mm}}))$ and inverts the $2\\times 2$ forward/backward fundamental Bloch system over a finite record of the pulse plus `{sp_meta['t_post_used_ns']:.1f}` ns (`{sp_meta['flux_single_pass_window_ns']:.1f}` ns for the partial reflector), rectangular with a 1.5 ns $\\sin^2$ end taper; the cross-sectional energy-flux reference $|\\Gamma_{{\\text{{flux}}}}(f)| = \\sqrt{{\\max(0, 1 - P_{{\\text{{net}}}}(f)/P_{{\\text{{inc}}}}(f))}}$ assumes a reflectionless incident reference $P_{{\\text{{inc}}}}(f) = P_{{\\text{{net,taper}}}}(f)$ and a `{sp_meta['flux_single_pass_window_ns']:.1f}` ns single-pass window.\n"
        f"   - *Counter-case / failure mode*: Near the CeSRR-to-termination transition (`{sp_meta['probe4_to_interface_grid_periods']:.2f}`$p$ downstream of probe 4), higher-order and evanescent cutoff modes contaminate the single-profile projection and finite-window FFT. On the lossless smooth-air standing wave (`termination='smooth_air'`, where theory gives $|\\Gamma| = 1$ identically for any $\\phi$), the two-probe estimator produces **per-frequency $|\\Gamma_{{\\text{{air}}}}(f)| \\in [{sparam['standing_wave_gamma_air_min']:.4f}, {sparam['standing_wave_gamma_air_max']:.4f}]$** (peak `{sparam['termination_smooth_air_refl_peak_db']:+.2f}` dB above passivity, with `{sparam['standing_wave_gamma_air_frac_above_1p04']*100:.1f}%` of frequencies exceeding `{1.0 + sparam['standing_wave_band_median_tol']:.2f}` and `{sparam['standing_wave_gamma_air_frac_outside_4pct']*100:.1f}%` outside $1 \\pm {sparam['standing_wave_band_median_tol']:.2f}$), setting a per-frequency estimator uncertainty of $\\pm\\max||\\Gamma_{{\\text{{air}}}}| - 1| = \\pm {sparam['standing_wave_gamma_air_max_abs_err']:.4f}$ across all termination-reflection claims (a heuristic transfer: the 2×2 conditioning on a near-total standing wave differs from that on the matched and partial terminations, so the matched-taper median is a regression bound, not an accuracy claim). Simultaneously, $P_{{\\text{{inc}}}}(f)$ carries the matched taper's own residual reflection up to $|\\Gamma_{{\\text{{taper}}}}| = {sparam['matched_taper_peak_gamma_mag']:.4f}$ (`{sparam['termination_matched_taper_refl_peak_db']:.2f}` dB), biasing $|\\Gamma_{{\\text{{flux}}}}|^2$ by of order $|\\Gamma_{{\\text{{taper}}}}|^2$ = `{sparam['matched_taper_peak_gamma_sq_bias']:.4f}` (exactly $|\\Gamma_{{\\text{{taper}}}}|^2(1 - |\\Gamma|^2)/(1 - |\\Gamma_{{\\text{{taper}}}}|^2)$) and clipping `{sparam['partial_reflector_flux_clipped_count']}` frequency points (`{sparam['partial_reflector_flux_clipped_f_min_ghz']:.2f}`–`{sparam['partial_reflector_flux_clipped_f_max_ghz']:.2f}` GHz) to zero. As a result, the partial-reflector comparison agrees only on the band median (`{sparam['partial_reflector_band_median_2probe']:.4f}` vs. `{sparam['partial_reflector_band_median_flux']:.4f}`, diff `{sparam['partial_reflector_band_median_abs_diff']:.4f} <= {sparam['partial_reflector_band_median_tol']:.2f}`), disagrees per frequency ($\\max|\\Delta|\\Gamma|| = {sparam['partial_reflector_per_freq_max_abs_diff']:.4f}$, RMS `{sparam['partial_reflector_per_freq_rms_diff']:.4f}`, correlation $r = {sparam['partial_reflector_per_freq_corr']:.4f}$), and passes for any phase perturbation in $\\Delta\\phi \\in [{sparam['partial_reflector_dphi_pass_min_rad']:+.1f}, {sparam['partial_reflector_dphi_pass_max_rad']:+.1f}]\\text{{ rad}}$, scale $s \\in [{sparam['partial_reflector_phi_scale_pass_min']:.1f}, {sparam['partial_reflector_phi_scale_pass_max']:.1f}]$, or constant $\\phi = \\mathrm{{median}}(\\phi)$ (first fails at $\\Delta\\phi = {sparam['partial_reflector_dphi_first_failing_signed_rad']:+.1f}\\text{{ rad}}$; fails for both signs only at $|\\Delta\\phi| \\ge {sparam['partial_reflector_dphi_both_signs_fail_abs_rad']:.1f}\\text{{ rad}}$)."
    )
)
#
# ---
#
# ## References
#
# 1. V. G. Veselago, "The electrodynamics of substances with simultaneously negative values of $\epsilon$ and $\mu$," *Sov. Phys. Usp.* **10**(4), 509–514 (1968), <https://doi.org/10.1070/PU1968v010n04ABEH003699>.
# 2. Z. Duan, X. Tang, Z. Wang, Y. Zhang, X. Chen, M. Chen, and Y. Gong, "Observation of the reversed Cherenkov radiation," *Nat. Commun.* **8**, 14901 (2017), <https://doi.org/10.1038/ncomms14901> (and Supplementary Information).
# 3. Z. Duan, J. S. Hummelt, M. A. Shapiro, and R. J. Temkin, "Sub-wavelength waveguide loaded by a complementary electric metamaterial for vacuum electron devices," *Phys. Plasmas* **21**(10), 103301 (2014), <https://doi.org/10.1063/1.4897392>.
# 4. J. S. Hummelt, S. M. Lewis, M. A. Shapiro, and R. J. Temkin, "Design of a metamaterial-based backward-wave oscillator," *IEEE Trans. Plasma Sci.* **42**(4), 930–936 (2014), <https://doi.org/10.1109/TPS.2014.2309597>.
# 5. R. Marqués, J. Martel, F. Mesa, and F. Medina, "Left-handed-media simulation and transmission of EM waves in subwavelength split-ring-resonator-loaded metallic waveguides," *Phys. Rev. Lett.* **89**(18), 183901 (2002), <https://doi.org/10.1103/PhysRevLett.89.183901>.
# 6. J. R. Pierce, "Theory of the beam-type traveling-wave tube," *Proc. IRE* **35**(2), 111–123 (1947), <https://doi.org/10.1109/JRPROC.1947.226217>; *Traveling-Wave Tubes*, D. Van Nostrand, New York (1950).
# 7. H. R. Johnson, "Backward-wave oscillators," *Proc. IRE* **43**(6), 684–697 (1955), <https://doi.org/10.1109/JRPROC.1955.278054>.

# %%
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tutorial 8.7: Metamaterial SWS & Reversed Cherenkov Radiation"
    )
    parser.add_argument(
        "--task",
        choices=["all", "cold_test", "sparam_retrieval", "reversed_cherenkov", "bwo_start_current"],
        default="all",
        help="Which simulation task to execute.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced-resolution / fast verification settings.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full production resolution settings (default when --quick is not set).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate setup schematics (00a/00b) and all result figures (01-05 + GIF).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run FIT simulations even when cached data/*.json files exist.",
    )
    args, _ = parser.parse_known_args()
    quick = bool(args.quick and not args.full)

    if args.plot and (DATA_DIR / "task_cold_test.json").exists() and not args.force:
        generate_all_figures()
        return

    if args.task in ("all", "cold_test") and (
        args.force or not (DATA_DIR / "task_cold_test.json").exists()
    ):
        run_task_cold_test(quick=quick)
    if args.task in ("all", "sparam_retrieval") and (
        args.force or not (DATA_DIR / "task_sparam_retrieval.json").exists()
    ):
        run_task_sparam_retrieval(quick=quick)
    if args.task in ("all", "reversed_cherenkov") and (
        args.force or not (DATA_DIR / "task_reversed_cherenkov.json").exists()
    ):
        run_task_reversed_cherenkov(quick=quick)
    if args.task in ("all", "bwo_start_current") and (
        args.force or not (DATA_DIR / "task_bwo_start_current.json").exists()
    ):
        run_task_bwo_start_current(quick=quick)

    if args.plot or not (FIG_DIR / "05_mesh_convergence_and_bwo_start_current.png").exists():
        generate_all_figures()


if __name__ == "__main__":
    main()
