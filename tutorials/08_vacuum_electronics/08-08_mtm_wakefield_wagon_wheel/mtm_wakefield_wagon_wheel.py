# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 08-08 — Reversed-Cherenkov Wakefields in an All-Metal Metamaterial (Wagon-Wheel) Structure
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/08_vacuum_electronics/08-08_mtm_wakefield_wagon_wheel/mtm_wakefield_wagon_wheel.ipynb)
#
# This tutorial reproduces with Wakis's 3D finite-integration-technique (FIT) time-domain solver
# (`SolverFIT3D`, `WakeSolver`, subcell conformal PEC, `wakis.bloch`, and `ResonanceMonitor`) the
# all-metal metamaterial (MTM) reversed-Cherenkov wakefield experiment of X. Lu *et al.*,
# *Phys. Rev. Lett.* **122**, 014801 (2019), performed at the Argonne Wakefield Accelerator (AWA) facility.
# Because the $65.0\;\mathrm{MeV}$ (PRL line 49; thesis line 2299) drive electron bunches are ultrarelativistic,
# no particle-in-cell plasma tracking is needed: the beam is modeled as a rigid relativistic Gaussian bunch
# (`wakis.sources.Beam`) coupled to the full 3D Maxwell equations on a conformal PEC grid.
# All simulation metrics, tables, and figure annotations are formatted directly from `data/*.json` at render time.
#
# ---
#
# ## Learning Goals
#
# 1. **All-metal wagon-wheel metamaterial geometry & port reflectometry (Part 1):** Construct a
#    subwavelength $p = 2.0\;\mathrm{mm}$ (PRL lines 140, 145–146, 166; thesis lines 1978, 1985, 1994) periodic
#    wagon-wheel slow-wave model inspired by Lu *et al.* (2019), tabulate every dimension stated in the paper and
#    thesis alongside every engineering dimension assumed by us (`ASSUMED`), justify the lossless PEC model from the
#    cold-test quality factor $Q_0 = 1050$ (Lu thesis Table 2.4, line 2173), and characterize a 2-cell coaxial TEM
#    extraction port against an abrupt circular pipe below the $14.2\;\mathrm{GHz}$ (PRL lines 150–151) $\mathrm{TM}_{01}$ cutoff.
# 2. **3D Bloch unit-cell cold test & mesh convergence (Part 2):** Use `wakis.bloch` and
#    `ResonanceMonitor` (`harminv` + single-frequency running DFT) to extract the fundamental backward
#    $\mathrm{TM}_{01}$-like passband below the empty circular pipe $\mathrm{TM}_{01}$ cutoff for both the primary
#    waveguide radius $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (`16 mm` ID in PRL lines 121, 152 and thesis line 1968)
#    and the thesis variant $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ (Lu thesis lines 2075, 2208), its negative group/energy
#    velocity vs the paper's $v_g = -0.158\,c$ (PRL line 147; thesis line 1992 and Table 2.4 line 2175), the model's own
#    $65.0\;\mathrm{MeV}$ (PRL line 49) beam-line crossing alongside the eigenmode frequency at the paper's synchronous
#    phase advance $\phi_s = 27.43^\circ$ (derived via $\phi_s = \omega_s p / u_0$ from $f_s = 11.42\;\mathrm{GHz}$, PRL lines 137, 149,
#    and $p = 2.0\;\mathrm{mm}$, PRL line 140), shunt
#    impedance per unit length $r/Q$ vs the paper's $21.0\;\mathrm{k\Omega/m}$ (PRL line 250; thesis Eq. 2.18 lines
#    2128–2135 and Table 2.4 line 2172), and 3-level mesh convergence across $n_{\mathrm{feat}} \in \{4, 6, 8\}$.
# 3. **Finite-structure wakefield simulation, 3-level transverse mesh GCI, affine length scaling & reversed-Cherenkov directionality (Part 3):**
#    Drive $N \in \{6, 8, 12\}$ periods at `n_xy = 64` ($4$ cells across all $1.0\;\mathrm{mm}$ features, PRL lines 112–113, 119;
#    with 3-level transverse mesh convergence across `n_xy in {48, 64, 96}`) with a $65.0\;\mathrm{MeV}$ (PRL line 49),
#    $q = 45.0\;\mathrm{nC}$ (PRL lines 49, 144, 149, 214, 224, 233, 291),
#    $\sigma_z = 1.2\;\mathrm{mm}$ (PRL lines 151, 212, 248–250, 291) rigid Gaussian bunch; compare the cycle-averaged RF output power
#    against the model's own Eq. (2) analytical prediction and the paper's "up to $25.0\;\mathrm{MW}$" (PRL lines 50, 232, 291) $N = 40$ benchmark
#    within the combined finest-level mesh uncertainty $U_P$ and two-port reflection bound $2|\Gamma|^2$;
#    fit the affine length-scaling laws $W_{\mathrm{beam,bp,cv}}(L) = a_W L + b_W$ and $E_{\mathrm{up}}(L) = a_{E,\mathrm{up}} L + b_{E,\mathrm{up}}$
#    and compare $a_W$ against the cold-test analytic loss per unit length $q^2 k_L \Phi^2 / (1 - v_g / u_0)$ within the two-level slope GCI uncertainty;
#    and contrast upstream ($-z$) extraction in the wagon-wheel metamaterial ($v_g < 0$) against downstream ($+z$) extraction
#    in a non-inductive disk-loaded forward-wave control structure ($v_g > 0$).
# 4. **Two-bunch coherent superposition & charge-scaled single-bunch estimate (Part 4):** Verify that 3D FIT satisfies
#    two-bunch linear superposition $\mathbf{E}_{1+2} = \mathbf{E}_1 + \mathbf{E}_2$ to double-precision round-off,
#    demonstrate finite-length passband wake interference, explain why $N=8$ cannot overlap two bunches at the experimental
#    $1.3\;\mathrm{GHz}$ (PRL lines 143, 153–154, 255–257, 329–331) linac spacing ($T_b = 769.23\;\mathrm{ps}$, PRL line 143, where
#    exact $1 / 1.3\;\mathrm{GHz}$ spacing is not in phase without laser splitter delay tuning to the 9th-harmonic
#    $1.27\;\mathrm{GHz}$, $T_b = 0.79\;\mathrm{ns}$ bunch train in Lu thesis lines 2340–2341, 2365), and
#    compare the **charge-scaled single-bunch estimate** for $Q_1 + Q_2 = 85.0\;\mathrm{nC}$ (PRL lines 51, 154, 297–298, 323, 337–338)
#    against the paper's $80.0\;\mathrm{MW}$ (PRL lines 51, 77, 298, 338) two-bunch benchmark.
# 5. **Analytic backward-wave power & unified control-volume Poynting energy balance (Parts 5 & 6):** Derive the closed-form
#    backward-wave wakefield pulse length $t_p$ (PRL Eq. 1, line 234) and power formula $P$ (PRL Eq. 2, lines 240–245, attributed on
#    PRL line 238 to T. P. Wangler, *RF Linear Accelerators*, 2008, Ref. 24 at lines 370–371) from
#    energy conservation, compare Wakis's 3D Bloch cold-test analytical power against the full 3D FIT wakefield simulation,
#    quantify the full-band ($|F(f_{\max})|^2 < 10^{-3}$ for $\sigma_z = 1.2\;\mathrm{mm}$, PRL line 151) vs passband
#    bunch energy loss and out-of-band share, and close the discrete FIT Poynting theorem
#    over the single control volume $[z_{\mathrm{up}}, z_{\mathrm{dn}}]$ (both unfiltered and causally bandpass-filtered,
#    with $U_{\mathrm{bp}}(T_{\mathrm{end}}) / W_{\mathrm{beam,bp}} < 0.02$) across $N \in \{6, 8, 12\}$.

# %% [markdown]
# ---
#
# ## Physical Background & Theory
#
# ### 1. Subwavelength Wagon-Wheel Metamaterial Passband Below the Circular Waveguide Cutoff
#
# An empty circular metallic waveguide of inner diameter $2 R_{\mathrm{wg}} = 16.0\;\mathrm{mm}$ ($R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$, PRL lines 121, 152;
# Lu 2019 MIT PhD thesis, line 1968) has its lowest transverse-magnetic
# $\mathrm{TM}_{01}$ cutoff at
#
# $$
# f_{c,\mathrm{TM}_{01}} = \frac{\chi_{01} c}{2 \pi R_{\mathrm{wg}}},
# $$
#
# where $\chi_{01}$ is the first root of $J_0(x) = 0$. In Lu *et al.* (2019, PRL lines 150–151), the empty-waveguide
# $\mathrm{TM}_{01}$ cutoff is quoted as $14.2\;\mathrm{GHz}$ (PRL lines 150–151); X. Lu's companion PhD thesis (MIT, 2019, Section 2.1,
# lines 2075, 2208) gives the inner waveguide radius to two decimal places as $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$
# ($2 R_{\mathrm{wg}} = 16.2\;\mathrm{mm}$, thesis line 2075), whose exact $\mathrm{TM}_{01}$ cutoff rounds to
# $14.2\;\mathrm{GHz}$ (PRL line 150). We adopt $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL lines 121, 152) as our primary
# geometry and also evaluate the $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ (thesis lines 2075, 2208) variant in Part 2.
# Below $f_{c,\mathrm{TM}_{01}}$, a hollow circular pipe acts as an overdense electric plasma with negative effective
# axial permittivity $\varepsilon_{\mathrm{eff}}(\omega) < 0$ and positive permeability $\mu_{\mathrm{eff}} > 0$, so
# $\mathrm{TM}$ waves are evanescent.
#
# In the experimental hardware (Lu *et al.*, 2019, PRL lines 112–125, 139–146, Fig. 2; Lu 2019 MIT PhD thesis,
# Section 2.1, lines 1950–1996, 2064–2095, Fig. 2-2), each $p = 2.0\;\mathrm{mm} \approx \lambda_0 / 13$ (PRL lines
# 140, 145–146, 166) period comprises a $1.0\;\mathrm{mm}$ (PRL lines 112–113; thesis lines 1950–1951, 1967, 2037–2038)
# stainless-steel wagon-wheel plate with radial spokes alternated with a $1.0\;\mathrm{mm}$ (PRL lines 112–113; thesis
# lines 1951, 1967, 2038) copper circular spacer plate.
#
# In our 3D FIT model (`WagonWheelMTMSpec`), we represent the $p = 2.0\;\mathrm{mm}$ (PRL line 140) unit cell using the
# dimensions stated in the paper supplemented by three explicit geometric assumptions (`ASSUMED`, listed in Table 1B):
#
# 1. **Wagon-wheel plate ($t_{\mathrm{ww}} = 1.0\;\mathrm{mm}$, $z \in [0, 1.0]\;\mathrm{mm}$, PRL lines 112–113):** An
#    inner PEC hub ring surrounding the $2 r_b = 6.0\;\mathrm{mm}$ (PRL line 122) beam aperture ($r_b = 3.0\;\mathrm{mm}$ to
#    $r_{\mathrm{hub}} = 4.0\;\mathrm{mm}$, PRL lines 122–123, 145, where the $1.0\;\mathrm{mm}$ ring wall thickness is
#    `ASSUMED` from the `1 mm` callout in PRL Fig. 2(b), line 119) is connected directly to the outer circular PEC
#    waveguide wall ($R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$, PRL line 121) by **four radial PEC spokes**
#    ($N_{\mathrm{spokes}} = 4$ along $\pm x, \pm y$, `ASSUMED`) of width $w_{\mathrm{spoke}} = 1.0\;\mathrm{mm}$ (PRL line 119;
#    Lu thesis line 1968). The four open quadrant regions between the spokes extend to the
#    outer waveguide wall ($R_{\mathrm{slot}} = R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$, PRL line 121, `ASSUMED`).
# 2. **Spacer plate ($t_{\mathrm{sp}} = 1.0\;\mathrm{mm}$, $z \in [1.0, 2.0]\;\mathrm{mm}$, PRL lines 112–113):** An
#    open circular vacuum region from the beam aperture $r_b = 3.0\;\mathrm{mm}$ (PRL line 122) to the outer PEC wall
#    $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL line 121), forming a $1.0\;\mathrm{mm}$ (PRL line 113) axial capacitive
#    gap between adjacent wagon-wheel hub/spoke plates.
#
# Radial currents along the four spokes provide shunt inductance below the circular pipe cutoff while the axial gap
# between adjacent hub rings and spokes provides series periodic capacitance. Together, they create a propagating
# fundamental backward $\mathrm{TM}_{01}$-like space harmonic below $f_{c,\mathrm{TM}_{01}}$ with positive phase velocity
# $v_p = \omega / k_z \approx c > 0$ synchronized to the $65.0\;\mathrm{MeV}$ (PRL line 49) electron beam, but **negative
# group and energy velocity**:
#
# $$
# v_g(\omega) = \frac{\mathrm{d}\omega}{\mathrm{d}k_z} = v_E = \frac{\langle P_z \rangle p}{U_{\mathrm{cell}}} < 0.
# $$
#
# Because $\mathbf{v}_g \cdot \mathbf{v}_p < 0$, the Cherenkov cone angle reverses: electromagnetic energy
# radiated by the $+z$-moving bunch streams **backward** toward the upstream ($-z$) beam entrance port
# (**reversed-Cherenkov wakefield radiation**, Lu *et al.*, 2019, PRL lines 56–67, 155–160, Fig. 1).
#
# ---
#
# ### 2. Derivation of Backward-Wave Pulse Duration, Energy Loss, and Output Power (*Derived Here*)
#
# Consider a lossless periodic slow-wave structure of active length $L = N p$ traversed along $+z$ by a rigid
# relativistic bunch of total charge $q$, velocity $u_0 = \beta c$ ($\beta = \sqrt{1 - \gamma^{-2}}$), and
# longitudinal charge line density $\lambda_q(\zeta)$ with RMS length $\sigma_z$.
#
# #### Step 1 — Longitudinal wake potential, shunt impedance, and modal loss factor (*derived here*)
#
# For a single dominant synchronous mode at angular frequency $\omega_s = 2 \pi f_s$ and axial wavenumber
# $k_z = \omega_s / u_0$ (phase advance per period $\phi_s = k_z p$), we use the accelerator/linac convention for
# the shunt impedance per unit length $r/Q$ (matching X. Lu, MIT PhD thesis, 2019, Eq. 2.18, lines 2128–2135,
# and Lu *et al.*, 2019, PRL lines 146, 244, 250):
#
# $$
# \frac{r}{Q} = \frac{|E_{z,0}|^2}{\omega_s u_{\mathrm{stored}}} = \frac{|E_{z,0}|^2 p}{\omega_s U_{\mathrm{cell}}}
# \quad [\Omega/\mathrm{m}], \qquad
# E_{z,0} = \frac{1}{p} \int_0^p E_z(0, 0, z)\,e^{+\mathrm{i} k_z z}\,\mathrm{d}z,
# $$
#
# where $|E_{z,0}|$ is the peak synchronous space-harmonic amplitude on axis and
# $U_{\mathrm{cell}} = u_{\mathrm{stored}} p = \frac{1}{4}\int_{\mathrm{cell}} (\varepsilon_0 |\tilde{\mathbf{E}}|^2 + \mu_0 |\tilde{\mathbf{H}}|^2)\,\mathrm{d}V$
# is the cycle-averaged electromagnetic energy stored in one period $p$ (`evaluate_bloch_unit_cell` computes
# `r_over_q_ohm_per_m` directly from this single expression with no rescaling factors).
# Under this definition, the point-charge modal loss factor per unit length is
#
# $$
# k_L = \frac{\omega_s}{4}\left(\frac{r}{Q}\right) = \frac{|E_{z,0}|^2 p}{4 U_{\mathrm{cell}}} \quad [\mathrm{V/(C \cdot m)}],
# $$
#
# and a Gaussian bunch of RMS length $\sigma_z$ has the longitudinal form factor (Lu *et al.*, 2019, PRL lines 248–250)
#
# $$
# \Phi(\omega_s, \sigma_z) = \exp\!\left(-\frac{\omega_s^2 \sigma_z^2}{2 u_0^2}\right).
# $$
#
# #### Step 2 — Wave-packet drainage time $t_p$ for a backward wave (*derived here*)
#
# Let the bunch enter the upstream end $z = 0$ at $t = 0$ and exit the downstream end $z = L$ at
# $t_{\mathrm{exit}} = L / u_0$.
# - The first wavelet emitted at $z = 0, t = 0$ immediately exits the upstream port $z = 0$ at $t_{\mathrm{start}} = 0$.
# - The last wavelet emitted just as the bunch reaches $z = L$ at $t = L / u_0$ must propagate **backward**
#   a distance $L$ at group speed $|v_g| = -v_g > 0$, arriving at the upstream port $z = 0$ at
#
# $$
# t_{\mathrm{end}} = \frac{L}{u_0} + \frac{L}{|v_g|} = \frac{L}{|v_g|}\left(1 + \frac{|v_g|}{u_0}\right)
# = \frac{L}{|v_g|}\left(1 - \frac{v_g}{u_0}\right), \qquad (v_g < 0).
# $$
#
# Therefore, the rectangular RF pulse envelope at the upstream ($-z$) output port has duration (Lu *et al.*, 2019,
# PRL Eq. 1, line 234)
#
# $$
# \boxed{\;t_p = \frac{L}{|v_g|}\left(1 - \frac{v_g}{u_0}\right) = L\left(\frac{1}{|v_g|} + \frac{1}{u_0}\right)\;}
# \quad \text{(derived here)}.
# $$
#
# By contrast, in a forward-wave structure ($v_g > 0$), the upstream-emitted wavelet travels in the same direction
# as the bunch, compressing the pulse at the downstream port to $t_p^{(\mathrm{fwd})} = L\left(\frac{1}{v_g} - \frac{1}{u_0}\right)$.
#
# #### Step 3 — Group-front slippage and total bunch energy loss $\Delta U_{\mathrm{bunch}}$ (*derived here*)
#
# Behind the bunch at position $z_b = u_0 t$, the radiated wave packet front moves at velocity $v_g < 0$, so the
# spatial length of the wake column stretched between the bunch and the backward-moving trailing edge grows at rate
# $u_0 - v_g = u_0(1 - v_g / u_0) > u_0$. Because the wake energy deposited per unit wake length is
# $u_{\mathrm{wake}} = (q \Phi)^2 k_L / (1 - v_g / u_0)^2$, integrating the work done against the bunch over
# $z_b \in [0, L]$ (or integrating the energy stored across the stretched wake column of length $(1 - v_g / u_0) L$)
# yields the total passband energy lost by the bunch (*derived here*):
#
# $$
# \boxed{\;\Delta U_{\mathrm{bunch}}(L) = \frac{q^2 k_L L\,\Phi^2}{1 - v_g / u_0}\;} \quad [\mathrm{J}]
# \quad \text{(derived here)}.
# $$
#
# #### Step 4 — Length-independent steady-state output RF power $P$ (*derived here*)
#
# In a lossless PEC structure with matched ends ($|\Gamma| \to 0$), conservation of electromagnetic energy requires
# that the total passband energy $\Delta U_{\mathrm{bunch}}(L)$ lost by the bunch equals the time-integrated
# Poynting flux $P \, t_p$ exiting the upstream port over the flat-top pulse duration $t_p$:
#
# $$
# P = \frac{\Delta U_{\mathrm{bunch}}(L)}{t_p}
# = \frac{\dfrac{q^2 k_L L\,\Phi^2}{1 - v_g / u_0}}{\dfrac{L}{|v_g|}\left(1 - \dfrac{v_g}{u_0}\right)}
# \implies
# \boxed{\;P = \frac{q^2 k_L |v_g| \Phi^2}{(1 - v_g / u_0)^2}
# = q^2 \Phi^2 \left[\frac{\omega_s}{4}\left(\frac{r}{Q}\right)\right] \frac{|v_g|}{(1 - v_g / u_0)^2}\;}
# \quad \text{(derived here; Lu et al., 2019, PRL Eq. 2, lines 240–245, citing Wangler, RF Linear Accelerators, Ref. 24 at lines 238, 370–371)}.
# $$
#
# Notice three key physical consequences of this derivation (*derived here*):
#
# 1. **Exact length invariance of peak power ($P \propto L^0$):** The active length $L$ cancels identically between
#    $\Delta U_{\mathrm{bunch}} \propto L^1$ and $t_p \propto L^1$. Increasing the number of periods $N$ stretches
#    the pulse duration $t_p \propto N$ and total energy $E_{\mathrm{rad}} \propto N$, and narrows the spectral
#    bandwidth $\Delta f \approx 1 / t_p \propto N^{-1}$, while the exact $\mathrm{sinc}^2$ peak longitudinal impedance scales as
#    $\operatorname{Re} Z_\parallel(f_s) = \frac{k_L L^2}{2 |v_g|} \propto \frac{k_L L}{\Delta f} \propto L^2$.
# 2. **Assumptions required for Eq. (2):** (i) Single dominant synchronous mode ($f_s$); (ii) rigid relativistic
#    bunch ($\Delta\gamma / \gamma \ll 1$, constant $u_0$); (iii) negligible ohmic attenuation over one drain time
#    ($\alpha_c L \ll 1$); (iv) matched port extraction ($|\Gamma|^2 \ll 1$); and (v) structure length exceeding
#    the bunch slippage length ($L > \sigma_z |v_g| / u_0$).
# 3. **Two-bunch coherent superposition (*derived here*):** Because Maxwell's equations in vacuum with PEC boundaries
#    are linear, two bunches with charges $Q_1, Q_2$ separated by time delay $\Delta t$ (phase separation
#    $\Delta\psi = \omega_s \Delta t$) produce the superposed electric field
#    $\mathbf{E}_{1+2}(\mathbf{r}, t) = (Q_1 / q_0)\mathbf{E}_1(\mathbf{r}, t) + (Q_2 / q_0)\mathbf{E}_1(\mathbf{r}, t - \Delta t)$.
#    In the temporal overlap region of the two wave packets ($t \in [\Delta t, t_p]$), the cycle-averaged output RF
#    power is
#
# $$
# \boxed{\;P_{1+2}(\Delta\psi) = P_1 \frac{Q_1^2 + Q_2^2 + 2 Q_1 Q_2 \cos(\Delta\psi)}{q_0^2}
# \;\xrightarrow{Q_1 = Q_2 = q_0}\; 4 P_1 \cos^2\!\left(\frac{\Delta\psi}{2}\right)\;} \quad \text{(derived here)},
# $$
#
# yielding $4\times$ the single-bunch power when $\Delta\psi = 2\pi m$ ($\Delta t = m / f_s$, in phase) and
# destructive cancellation ($P_{1+2} = 0$) when $\Delta\psi = (2m + 1)\pi$ (out of phase).

# %% [markdown]
# ---
#
# ## Setup, Paper Parameters vs Assumed Parameters, and Port Reflectometry (Part 1)
#
# Every physical and geometric quantity from X. Lu *et al.*, *Phys. Rev. Lett.* **122**, 014801 (2019), and X. Lu,
# MIT PhD thesis (2019), is quoted below with its exact 1-indexed line numbers in `.ref/mtm/`, followed by Table 1B listing
# every dimension and modeling choice not stated in the paper text (`ASSUMED`) and the physical justification for each.
#
# ### Table 1A — Parameters Stated Explicitly in Lu *et al.*, *Phys. Rev. Lett.* 122, 014801 (2019) & Lu (2019) Thesis
#
# | Parameter | Symbol | Paper Value | Unit | `.ref` Line Numbers | Paper Context |
# | :--- | :--- | :---: | :---: | :--- | :--- |
# | Unit-cell axial period | $p$ | `2.0` | mm | PRL lines 140, 145–146, 166; thesis lines 1978, 1985, 1994 | One 1 mm wagon-wheel plate + one 1 mm spacer plate |
# | Wagon-wheel plate thickness | $t_{\mathrm{ww}}$ | `1.0` | mm | PRL lines 112–113, 163–165; thesis lines 1950–1951, 1967, 2037–2038 | 1-mm-thick stainless-steel wagon-wheel plates in Fig. 2(b) |
# | Spacer plate thickness | $t_{\mathrm{sp}}$ | `1.0` | mm | PRL lines 112–113, 163–165; thesis lines 1951, 1967, 2038 | 1-mm-thick copper circular spacer plate between wagon-wheel plates |
# | Radial spoke width | $w_{\mathrm{spoke}}$ | `1.0` | mm | PRL Fig. 2(b) line 119; thesis line 1968 | Dimension callout `1 mm` in Fig. 2(b) and "thickness of spokes as 1 mm" in thesis |
# | Beam hole diameter | $2 r_b$ | `6.0` | mm | PRL lines 122–123, 145; thesis lines 1967, 1969 | Central electron beam aperture ($r_b = 3.0\;\mathrm{mm}$) |
# | Enclosing circular waveguide ID | $2 R_{\mathrm{wg}}$ | `16.0` (`16.2` thesis) | mm | PRL lines 121, 152 (`16 mm`); thesis line 1968 (`16 mm`) & lines 2075, 2208 (`R_wg = 8.1 mm`) | Circular waveguide ID ($R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ in PRL; $8.1\;\mathrm{mm}$ in thesis) |
# | Empty waveguide $\mathrm{TM}_{01}$ cutoff | $f_{c,\mathrm{TM}_{01}}$ | `14.2` | GHz | PRL lines 150–151; thesis line 2075 | Quoted cutoff of empty circular pipe |
# | Experimental period count (active length) | $N_{\mathrm{exp}}$ ($L_{\mathrm{exp}}$) | `40` (`80.0`) | periods (mm) | PRL lines 134, 166; thesis lines 1951–1952, 2201, 2237, 2244, 2376 | 40 periods for an $8\;\mathrm{cm}$ ($80\;\mathrm{mm}$) interaction region |
# | Unloaded cold-test quality factor | $Q_0$ | `1050` | — | Lu thesis Table 2.4, line 2173 (PRL lines 163–165) | Stainless-steel wagon-wheel + copper spacer assembly |
# | Drive electron beam kinetic energy | $E_k$ | `65.0` | MeV | PRL lines 49, 105, 141–143, 148; thesis line 2299 | Relativistic bunches from the $1.3\;\mathrm{GHz}$ (PRL line 143) AWA RF photoinjector |
# | Single drive bunch charge | $q$ | `45.0` | nC | PRL lines 49, 143, 149, 214, 224, 232–233, 291; thesis line 2299 | Single high-charge bunch generating up to `25.0` MW (PRL lines 50, 232, 291) peak power |
# | RMS longitudinal bunch length | $\sigma_z$ | `1.2` | mm | PRL lines 151, 212, 248, 291 | Gaussian bunch distribution with form factor $\Phi = 0.96$ (PRL line 248) |
# | Cold-test synchronous frequency | $f_s$ | `11.42` | GHz | PRL lines 137, 149 (11.4 GHz at lines 50, 78, 145, 155, 249, 293); thesis lines 1960, 1991, 2340 | Intersection of fundamental backward mode with $65\;\mathrm{MeV}$ beam line |
# | Cold-test synchronous phase advance | $\phi_s$ | `27.43` | deg | Derived ($\phi_s = \omega_s p / u_0$) from PRL lines 137, 140, 147 | Phase advance per $2.0\;\mathrm{mm}$ (PRL line 140) period at $11.42\;\mathrm{GHz}$ (PRL line 147), derived here |
# | Cold-test group velocity | $v_g / c$ | `-0.158` | — | PRL line 147; thesis line 1992, Table 2.4 (line 2175) | Negative group velocity of fundamental backward mode at synchronism |
# | Cold-test shunt impedance per unit length | $r/Q$ | `21.0` | $\mathrm{k\Omega/m}$ | PRL line 250; thesis Eq. 2.18 (lines 2128–2135), Table 2.4 (line 2172) | Fundamental mode interaction impedance per unit length $|E_{z,0}|^2 p / (\omega_s U_{\mathrm{cell}})$ |
# | Two-bunch total combined charge & spacing | $Q_1 + Q_2, T_b$ | `85.0` nC, `769.23` ps (`0.79` ns) | nC, ps | PRL lines 51, 77, 143, 153–154, 255–257, 297–298, 323, 329–331, 337–338; thesis lines 2337–2341, 2365 | Two bunches spaced around $1 / (1.3\;\mathrm{GHz}) = 769.23\;\mathrm{ps}$ (PRL line 143) and fine-tuned in phase with laser splitters (PRL lines 153–154, 256–257, 329–331; thesis lines 2338–2341: $11.42\;\mathrm{GHz}$ is the 9th harmonic of a $1.27\;\mathrm{GHz}$ bunch train with $T_b = 0.79\;\mathrm{ns}$ at line 2365) with combined charge `85.0` nC (PRL line 51) generating `80.0` MW (PRL lines 51, 77, 298, 338) |
#
# ### Table 1B — Dimensions and Modeling Choices Assumed by Us (`ASSUMED`, Not Given in Paper Text)
#
# | Parameter | Symbol | Assumed Value | Reason / Physical Justification |
# | :--- | :--- | :---: | :--- |
# | Inner hub ring outer radius | $r_{\mathrm{hub}}$ | `4.0` mm (`ASSUMED`: $r_b + 1.0\;\mathrm{mm}$, PRL lines 119, 122) | PRL Fig. 2(b) (line 119) shows a `1 mm` dimension callout and thesis line 1968 states "thickness of spokes as 1 mm", but neither text explicitly prints $r_{\mathrm{hub}} = 4.0\;\mathrm{mm}$ |
# | Number, span & orientation of spokes | $N_{\mathrm{spokes}}, R_{\mathrm{slot}}$ | `4` spokes along $\pm x, \pm y$, $R_{\mathrm{slot}} = R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL line 121, `ASSUMED`) | Counted from PRL Fig. 2(b) (line 119); our model connects the 4 spokes directly to the $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL line 121) outer wall with no separate outer slot rim |
# | Spoke & hub corner fillet radius | $r_{\mathrm{fillet}}$ | `0.0` mm (`ASSUMED`: sharp rectangular, PRL line 119) | Machined corner radii are not specified in PRL or thesis; represented via subcell conformal PEC (`conformal="pec"`) |
# | Conductor wall model | $\sigma_{\mathrm{wall}}$ | Lossless PEC ($\sigma \to \infty$, `ASSUMED`) | Justified quantitatively in code cell below from $Q_0 = 1050$ (Lu thesis Table 2.4, line 2173) |
# | End output port couplers | $r_{\mathrm{coax}} = 3.30\;\mathrm{mm}$ coaxial TEM coupler + 2-cell taper (`ASSUMED`, PRL line 135) | Replaces WR-90 side ports | An empty $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL line 121) circular pipe is cut off below $14.2\;\mathrm{GHz}$ (PRL line 150); the 2-cell coaxial TEM coupler transmits the backward wave while preserving axial symmetry |
# | Simulated active period counts | $N_{\mathrm{sim}}$ | $N \in \{6, 8, 12\}$ (`ASSUMED`) | Spans a $2\times$ range in active interaction length ($L \in \{12, 16, 24\}\;\mathrm{mm}$) with full passband cavity drainage ($U_{\mathrm{bp}}(T_{\mathrm{end}})/W_{\mathrm{beam,bp}} < 0.02$) to isolate the active-length affine slope $a_W = \mathrm{d}W_{\mathrm{beam,bp,cv}}/\mathrm{d}L$ from the 2-cell taper intercept $b_W$, while pre-asymptotic short-$N$ deviations from $N = 40$ (PRL lines 134, 166) are quantified in Part 3 |

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

from IPython.display import Markdown, display

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (TUTORIAL_DIR / "_helpers.py").is_file():
    TUTORIAL_DIR = (
        Path.cwd() / "tutorials" / "08_vacuum_electronics" / "08-08_mtm_wakefield_wagon_wheel"
    )
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (
    C0,
    derive_pre_asymptotic_scaling_slopes,
    format_gci_line,
    gaussian_bunch_form_factor,
    reflection_induced_energy_error_bound,
    render_cold_test_table_markdown,
    render_quantitative_summary_markdown,
)

DATA_DIR = TUTORIAL_DIR / "data"
FIGURES_DIR = TUTORIAL_DIR / "figures"

t1 = json.loads((DATA_DIR / "task_01_geometry_and_port_reflection.json").read_text(encoding="utf-8"))
t2 = json.loads((DATA_DIR / "task_02_cold_test_and_convergence.json").read_text(encoding="utf-8"))
t3 = json.loads((DATA_DIR / "task_03_wakefield_and_directionality.json").read_text(encoding="utf-8"))
t4 = json.loads((DATA_DIR / "task_04_two_bunch_superposition.json").read_text(encoding="utf-8"))
t5 = json.loads((DATA_DIR / "task_05_analytic_power_and_energy_balance.json").read_text(encoding="utf-8"))
sm = json.loads((DATA_DIR / "summary_metrics.json").read_text(encoding="utf-8"))

pr = t1["port_reflection"]
pj = t1["pec_ohmic_justification"]
setup_md_path = FIGURES_DIR / "00_setup_table.md"
if setup_md_path.is_file():
    display(Markdown(setup_md_path.read_text(encoding="utf-8")))

print(
    f"Empty circular waveguide TM01 cutoff:\n"
    f"  R_wg = 8.0 mm (PRL line 121, 16 mm ID): analytic = {t1['empty_waveguide_tm01_cutoff_analytic_ghz']:.4f} GHz, "
    f"3D FIT = {t2['empty_waveguide_tm01_cutoff_fit_ghz']:.4f} GHz\n"
    f"  R_wg = 8.1 mm (thesis line 2075, 16.2 mm ID): analytic = {t1['empty_waveguide_tm01_cutoff_8p1mm_analytic_ghz']:.4f} GHz "
    f"(paper quotes {t1['empty_waveguide_tm01_cutoff_paper_ghz']:.1f} GHz, PRL line 150)"
)
print(
    f"Upstream (-z) port reflection evaluated at FFT bin f_eval = {pr['f_sync_eval_ghz']:.4f} GHz "
    f"(target f_s = {pr['f_sync_target_ghz']:.2f} GHz, n_xy = {pr['n_xy_port']}):\n"
    f"  Tapered 2-cell coax: |Gamma| = {pr['gamma_tapered_at_sync']:.4f} "
    f"(raw unclipped = {pr['gamma_tapered_raw_at_sync']:.4f}, "
    f"power transmission 1 - |Gamma|^2 = {100 * pr['power_transmission_tapered_at_sync']:.2f}%, "
    f"standing-wave power error bound = +/-{100 * pr['rel_error_bound_tapered']:.1f}%)\n"
    f"  Abrupt empty circular pipe: |Gamma| = {pr['gamma_abrupt_at_sync_label']} "
    f"(raw unclipped = {pr['gamma_abrupt_unclipped_at_sync']:.4f})"
)
print(
    f"Lossless PEC justification (stainless-steel + copper assembly, Q_0 = {pj['unloaded_q0_thesis']:.0f}, Lu thesis Table 2.4 line 2173):\n"
    f"  Ohmic field decay constant tau_0 = 2*Q_0/omega_s = {pj['ohmic_field_decay_time_2q_over_omega_ns']:.2f} ns\n"
    f"  tau_0 / t_p(N=40) = {pj['ratio_field_decay_over_pulse_n40']:.2f}x (mean one-way power loss = {100 * pj['mean_ohmic_power_loss_fraction_n40']:.2f}%)\n"
    f"  tau_0 / t_p(N=8)  = {pj['ratio_field_decay_over_pulse_n8']:.2f}x (mean one-way power loss = {100 * pj['mean_ohmic_power_loss_fraction_n8']:.2f}%)"
)

# %% [markdown]
# ![Wagon-Wheel Metamaterial Geometry & Setup Schematic](figures/00a_setup_geometry.png)
#
# ![Computational Mesh, Relativistic Bunch Excitation & Spectrum](figures/00b_setup_mesh_and_excitation.png)
#
# ![Wagon-Wheel Plate Cross-Section and Upstream Port Reflectometry](figures/01_wagon_wheel_geometry_and_port_reflection.png)
#
# **Physical interpretation of Figure 01:**
# - **Why an empty circular beam pipe cannot extract the $11.42\;\mathrm{GHz}$ (PRL lines 137, 149) wakefield:** Because the empty
#   $2 R_{\mathrm{wg}} = 16.0\;\mathrm{mm}$ (PRL line 121) circular waveguide has its fundamental $\mathrm{TM}_{01}$ cutoff
#   above the passband ($14.2\;\mathrm{GHz}$ in PRL lines 150–151, corresponding to $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ in
#   thesis line 2075), terminating the metamaterial directly into an empty circular pipe is evanescent below cutoff,
#   trapping the wakefield inside the cavity as a standing wave like a shorted PEC cap ($|\Gamma| \equiv 1$). In
#   `measure_upstream_port_reflection`, the inner beam-tunnel shield of `abrupt_empty_pipe` terminates at the exact
#   reference plane $z = n_{\mathrm{up}} p$ matching `pec_cap` so that no unshielded beam gap in cell $k = n_{\mathrm{up}} - 1$
#   contaminates the scattered-field subtraction, yielding passive reflection $|\Gamma_{\mathrm{abrupt,raw}}| < 1$ across the
#   passband without clipping.
# - **Adiabatic coaxial TEM extraction coupler & standing-wave error bound:** In the AWA experiment, two side-wall WR-90
#   rectangular waveguides coupled out the backward wave (Lu *et al.*, 2019, PRL lines 134–138, 166–170). In our axially
#   symmetric 3D model, extending the inner $r_b = 3.0\;\mathrm{mm}$ (PRL line 122) beam tunnel as a thin coaxial inner shield
#   ($r \in [3.0, 3.3]\;\mathrm{mm}$, PRL line 122, `ASSUMED`) through a 2-cell linear spoke/hub taper converts the backward
#   metamaterial mode into a cutoff-free coaxial TEM wave in the outer annulus $r \in [3.3, 8.0]\;\mathrm{mm}$ (PRL line 121),
#   with the exact reflection $|\Gamma(f_{\mathrm{eval}})|$, power transmission $1 - |\Gamma|^2$, and standing-wave bound
#   printed from `data/task_01_geometry_and_port_reflection.json` above.

# %% [markdown]
# ---
#
# ## Part 2 — 3D Bloch Unit-Cell Cold Test, Negative Group Velocity & Mesh Convergence
#
# Using `wakis.bloch.install_bloch_boundaries` on a single $p = 2.0\;\mathrm{mm}$ (PRL line 140) period (one $1.0\;\mathrm{mm}$
# 4-spoke wagon-wheel plate with $r_{\mathrm{hub}} = 4.0\;\mathrm{mm}, R_{\mathrm{slot}} = R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$, PRL lines 112, 119, 121,
# and one $1.0\;\mathrm{mm}$ spacer plate, PRL line 113) and `ResonanceMonitor` (`harminv` pole
# extraction with FFT fallback, followed by a single-frequency running DFT pass), we map the dispersion curve $f(\phi)$ across
# $\phi = k_z p \in [12^\circ, 72^\circ]$ and perform a 3-level mesh convergence study across $n_{\mathrm{feat}} \in \{4, 6, 8\}$
# cells per $1.0\;\mathrm{mm}$ (PRL line 112) feature for both uniform isotropic and feature-snapped graded conformal PEC grids.
# Because the unrounded $90^\circ$ PEC spoke and hub corners (`r_fillet = 0.0 mm`, `ASSUMED`) produce local field
# singularities, the formal spatial convergence order of the 3D FIT discretization is first order ($p_{\mathrm{formal}} = 1$).
# Following Roache (1998), Celik *et al.* (*J. Fluids Eng.* **130**, 078001, 2008), and Stern *et al.* (*J. Fluids Eng.* **123**,
# 793, 2001), `compute_three_level_gci` classifies each observable from the increment ratio $R = e_{32} / e_{21}$
# ($e_{21} = f_{\mathrm{med}} - f_{\mathrm{coarse}}$, $e_{32} = f_{\mathrm{fine}} - f_{\mathrm{med}}$) as monotone ($0 < R < 1$),
# oscillatory ($R < 0$), or divergent ($R \ge 1$), solves the observed order $p_{\mathrm{obs}}$ and Richardson extrapolate only
# for monotone data (noting that $p_{\mathrm{obs}}$ can be unsolvable even for monotone data with $0 < R < 1$ when no positive
# root $p_{\mathrm{obs}} > 0$ exists on a non-uniform 3-grid ratio $r_{12} \neq r_{23}$, as occurs for uniform cold-test $r/Q$,
# in which case $F_s = 3, p = 1$ is then used), uses $F_s = 1.25$ with $p_{\mathrm{used}} = p_{\mathrm{obs}}$ only when $p_{\mathrm{obs}}$ is within $\pm 25\%$
# of $p_{\mathrm{formal}} = 1$ (otherwise $F_s = 3$ with $p_{\mathrm{used}} = 1$), and reports the finest-level uncertainty
# $U_{\mathrm{fine}} = \max(\mathrm{GCI}(F_s, p_{\mathrm{used}}), (\max f - \min f) / (2 |f_{\mathrm{fine}}|))$.
#
# We distinguish three related frequencies near synchronism (all values printed and tabulated dynamically from
# `data/task_02_cold_test_and_convergence.json` and `data/task_03_wakefield_and_directionality.json` below):
# 1. **Model's own $65.0\;\mathrm{MeV}$ (PRL line 49) beam-line crossing (`nf4` dispersion sweep):** The beam line
#    $f_{\mathrm{beam}}(\phi) = (\beta c / p)(\phi / 2\pi)$ intersects the model's `nf4` dispersion curve at
#    $(\phi_{\mathrm{cross}}, f_{\mathrm{cross,nf4}})$.
# 2. **Bloch eigenmode frequency at the synchronous phase advance $\phi_s = 27.43^\circ$ (derived via $\omega_s p / u_0$ from PRL lines 137, 140):** Evaluated at fixed
#    $\phi_s = 27.43^\circ$ (derived from $\omega_s p / u_0$, PRL lines 137, 140) across `nf4`, `nf6`, and `nf8` for both the **primary model radius**
#    $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ ($16.0\;\mathrm{mm}$ inner diameter in PRL lines 121, 152 and thesis line 1968 —
#    chosen as primary because it is the dimension printed in the primary PRL publication and places $1.0\;\mathrm{mm}$ (PRL line 112)
#    spoke/hub edges on exact grid nodes of power-of-two grids `n_xy = 64` and `n_feat = 8`) and the thesis
#    engineering-drawing variant $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ (Lu 2019 thesis lines 2075, 2208 across `nf4`, `nf6`,
#    and `nf8`), which bracket the paper's $f_s = 11.42\;\mathrm{GHz}$ (PRL line 137).
# 3. **Finite-structure ($N=8$) wake impedance peak frequency (Part 3):** In the finite $N=8$ structure with 2-cell
#    coaxial tapers at `n_xy = 64`, $\operatorname{Re} Z_\parallel(f)$ peaks at $f_{\mathrm{pk}}(N=8)$.

# %%
disp = t2["dispersion"]
wb = t2["wakis_finest_nf8_benchmarks"]
pb = t2["paper_cold_test_benchmarks"]
cu = t2["convergence_uniform"]
cg = t2["convergence_graded"]
v81 = t2["variant_rwg_8p1mm_thesis"]
v81_finest = v81["nf8"] if "nf8" in v81 else v81["nf6"]
v81_finest_lbl = "nf=8" if "nf8" in v81 else "nf=6"

print(
    f"Model's own 65.0 MeV (PRL line 49) beam-line crossing (nf=4 dispersion): "
    f"phi_cross = {wb['phi_beam_crossing_nf4_deg']:.2f} deg, "
    f"f_cross (nf=4) = {wb['f_beam_crossing_nf4_ghz']:.4f} GHz vs paper {pb['f_sync_ghz']:.2f} GHz "
    f"(rel. diff = {100 * wb['rel_error_f_beam_crossing_nf4_vs_paper']:.2f}%)"
)
print(
    f"Primary radius R_wg = 8.0 mm (PRL line 121, 16.0 mm ID) Bloch frequency at phi_s = {pb['phi_sync_deg']:.2f} deg:\n"
    f"  f_s (nf=4) = {cu['f_mode_ghz']['values'][0]:.4f} GHz, "
    f"f_s (nf=6) = {cu['f_mode_ghz']['values'][1]:.4f} GHz, "
    f"f_s (nf=8) = {wb['f_sync_ghz']:.4f} GHz vs paper {pb['f_sync_ghz']:.2f} GHz "
    f"(rel. diff = {100 * wb['rel_error_f_sync_vs_paper']:.2f}%)\n"
    f"Thesis R_wg = 8.1 mm (thesis line 2075) Bloch variant at phi_s = {pb['phi_sync_deg']:.2f} deg "
    f"(TM01 cutoff = {v81['tm01_cutoff_analytic_ghz']:.4f} GHz):\n"
    f"  f_s (nf=4) = {v81['nf4']['f_mode_ghz']:.4f} GHz, "
    f"f_s (nf=6) = {v81['nf6']['f_mode_ghz']:.4f} GHz, "
    f"f_s ({v81_finest_lbl}) = {v81_finest['f_mode_ghz']:.4f} GHz "
    f"(rel. diff vs paper = {100 * abs(v81_finest['f_mode_ghz'] - pb['f_sync_ghz']) / pb['f_sync_ghz']:.2f}%), "
    f"v_E/c ({v81_finest_lbl}) = {v81_finest['ve_over_c']:.4f}, r/Q ({v81_finest_lbl}) = {v81_finest['r_over_q_kohm_per_m']:.3f} kOhm/m"
)
print(
    f"Negative group/energy velocity at phi_s = {pb['phi_sync_deg']:.2f} deg:\n"
    f"  v_g/c (dispersion slope) = {wb['vg_disp_over_c']:.4f} (rel. diff = {100 * wb['rel_error_vg_vs_paper']:.2f}%)\n"
    f"  v_E/c (DFT Poynting flux, nf=8) = {wb['ve_dft_over_c']:.4f} vs paper {pb['vg_over_c']:.3f} "
    f"(rel. diff = {100 * wb['rel_error_ve_vs_paper']:.2f}%)"
)
print(
    f"Shunt impedance & loss factor (single convention |E_z0|^2*p/(omega_s*U_cell), thesis Eq. 2.18 line 2128, nf=8):\n"
    f"  r/Q = {wb['r_over_q_kohm_per_m']:.3f} kOhm/m vs paper {pb['r_over_q_kohm_per_m']:.1f} kOhm/m "
    f"(rel. diff = {100 * wb['rel_error_r_over_q_vs_paper']:.2f}%)\n"
    f"  k_L = {wb['k_loss_v_per_c_m']:.3e} V/(C*m) vs derived paper {pb['k_loss_v_per_c_m']:.3e} V/(C*m) "
    f"(rel. diff = {100 * wb['rel_error_k_loss_vs_paper']:.2f}%)\n"
    f"  Unit-cell energy equipartition (nf=8): U_e/U_m = {wb['u_elec_over_u_mag']:.4f} "
    f"({abs(wb['u_elec_over_u_mag'] - 1.0) * 100:.2f} % from unity)"
)
print(
    f"Uniform mesh convergence at phi_s = {pb['phi_sync_deg']:.2f} deg (nf = 4, 6, 8 -> h = {cu['h_mm']} mm):\n"
    f"  f_s [GHz]         : {format_gci_line(cu['f_mode_ghz'], 'GHz', 4)}\n"
    f"  v_E/c [—]         : {format_gci_line(cu['ve_over_c'], '', 4)}\n"
    f"  r/Q [kOhm/m]      : {format_gci_line(cu['r_over_q_kohm_per_m'], 'kOhm/m', 3)}\n"
    f"  k_L [V/(C*m)]     : {format_gci_line(cu['k_loss_v_per_c_m'], 'V/(C*m)', 0)}\n"
    f"Graded mesh convergence (nf = 4, 6, 8):\n"
    f"  f_s [GHz]         : {format_gci_line(cg['f_mode_ghz'], 'GHz', 4)}\n"
    f"  r/Q [kOhm/m]      : {format_gci_line(cg['r_over_q_kohm_per_m'], 'kOhm/m', 3)}"
)
display(Markdown("### Table 2 — 3D Bloch Unit-Cell Cold-Test Mesh Convergence\n\n" + render_cold_test_table_markdown(t2)))

# %% [markdown]
# ![3D Bloch Dispersion Diagram and Transverse Eigenmode Field](figures/02_bloch_dispersion_and_eigenmode_fields.png)
#
# ![Cold-Test 3-Level Mesh Convergence](figures/03_cold_test_mesh_convergence.png)
#
# **Physical interpretation of Figures 02 & 03:**
# - **Backward-wave passband below cutoff:** Across $\phi \in [12^\circ, 72^\circ]$, the fundamental wagon-wheel
#   metamaterial mode decreases monotonically with phase advance $\phi = k_z p$ entirely below the empty-pipe
#   $\mathrm{TM}_{01}$ cutoff ($14.2\;\mathrm{GHz}$ in PRL lines 150–151), whereas the non-inductive disk-loaded control
#   mode propagates above cutoff with positive group velocity $v_g > 0$.
# - **Sensitivity to outer waveguide radius ($R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$, PRL line 121, vs $8.1\;\mathrm{mm}$, thesis line 2075):** Increasing the
#   outer pipe radius from the PRL rounded dimension $R_{\mathrm{wg}} = 8.0\;\mathrm{mm}$ (PRL lines 121, 152) to the
#   thesis engineering dimension $R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ (Lu 2019 thesis lines 2075, 2208) lowers both the
#   empty-pipe $\mathrm{TM}_{01}$ cutoff and the Bloch synchronous frequency at $\phi_s = 27.43^\circ$ (derived from $\omega_s p / u_0$, PRL lines 137, 140),
#   bracketing the paper's $11.42\;\mathrm{GHz}$ (PRL line 137) while keeping $v_E / c$ and $r/Q$ nearly unchanged.
# - **Energy equipartition, velocity consistency & Figure 03(a) normalization:** At $\phi_s = 27.43^\circ$ (derived from $\omega_s p / u_0$, PRL lines 137, 140), the electric-to-magnetic
#   stored energy ratio $U_e / U_m$ in the unit cell (`wb["u_elec_over_u_mag"]`, formatted from JSON in the code cells
#   above and below) satisfies time-harmonic energy equipartition, and the independent DFT
#   energy transport velocity $v_E = \langle P_z \rangle p / U_{\mathrm{cell}}$ agrees closely with both the dispersion
#   slope $\mathrm{d}\omega / \mathrm{d}k_z$ and the paper's $-0.158\,c$ (PRL line 147). In Figure 03(a), each relative
#   discretization error curve is normalized by its $p_{\mathrm{obs}}$-based Richardson extrapolate (allowed for monotone
#   data, with `observed_order` and `richardson_estimate` printed from `t2["convergence_uniform"]` and
#   `t2["convergence_graded"]` above), whereas for uniform cold-test $r/Q$ (`convergence_class = "monotone"`, where
#   $p_{\mathrm{obs}}$ is unsolvable on the non-uniform 3-grid ratios), `compute_three_level_gci` uses $F_s = 3, p = 1$.

# %%
display(
    Markdown(
        f"*Formatted from `data/task_02_cold_test_and_convergence.json`:* At `nf=8`, "
        f"`U_e/U_m = {wb['u_elec_over_u_mag']:.4f} ({abs(wb['u_elec_over_u_mag'] - 1.0) * 100:.2f} % from unity)`, "
        f"Figure 03(a) normalizes monotone $f_s$ and $|v_E|/c$ by their $p_{{\\mathrm{{obs}}}}$-based Richardson extrapolates "
        f"($p_{{\\mathrm{{obs}}}} = {cu['f_mode_ghz']['observed_order']:.2f}, {cg['f_mode_ghz']['observed_order']:.2f}, {cu['ve_over_c']['observed_order']:.2f}$), "
        f"and uniform $r/Q$ is `{cu['r_over_q_kohm_per_m']['convergence_class']}` ($R = {cu['r_over_q_kohm_per_m']['R_ratio']:.3f}$) "
        f"with `p_obs = {cu['r_over_q_kohm_per_m']['observed_order']}` ($F_s = {cu['r_over_q_kohm_per_m']['fs_used']:.0f}, p = {cu['r_over_q_kohm_per_m']['p_used']:.0f}$ used)."
    )
)

# %% [markdown]
# ---
#
# ## Part 3 — Finite-Structure Wakefield Simulation, 3-Level Transverse Mesh GCI, Affine Length Scaling & Reversed-Cherenkov Directionality
#
# We now simulate the finite wagon-wheel metamaterial structure between coaxial TEM extraction ports for
# $N \in \{6, 8, 12\}$ periods at `n_xy = 64` (providing $4$ cells across every $1.0\;\mathrm{mm}$ spoke, hub wall, PRL lines 112–113, 119,
# plate, and axial gap), excited by a $65.0\;\mathrm{MeV}$ (PRL line 49), $q = 45.0\;\mathrm{nC}$
# (PRL line 49), $\sigma_z = 1.2\;\mathrm{mm}$ (PRL line 151) rigid Gaussian bunch, and evaluate a 3-level transverse
# mesh hierarchy at $N = 8$ across `n_xy in {48, 64, 96}` ($3, 4, 6$ cells per $1.0\;\mathrm{mm}$ feature, PRL line 112)
# to classify each observable's convergence from $R = e_{32} / e_{21}$ ($e_{21} = f_{64} - f_{48}$, $e_{32} = f_{96} - f_{64}$;
# monotone if $0 < R < 1$, oscillatory if $R < 0$ — with $|R| > 1$ flagged as "oscillatory, $|R| > 1$ (amplitude not decreasing)"
# from `R_ratio`, as occurs for $W_{\mathrm{beam,bp,cv}}$, $E_{\mathrm{up,coax}}$, and $|\Gamma_{\mathrm{tap}}(f_s)|$ — and
# divergent if $R \ge 1$) and evaluate the finest-level uncertainty
# $U_{\mathrm{fine}} = \max(\mathrm{GCI}(F_s, p_{\mathrm{used}}), (\max f - \min f)/(2|f_{96}|))$ with formal order
# $p_{\mathrm{formal}} = 1$ (governed by the singular fields at the $90^\circ$ PEC spoke/hub corners; $F_s = 1.25$ only
# when $p_{\mathrm{obs}}$ is solved and within $\pm 25\%$ of $p_{\mathrm{formal}} = 1$, otherwise $F_s = 3, p_{\mathrm{used}} = 1$).
# We also compare the $N = 8$
# backward-wave metamaterial structure directly against an $N = 8$ non-inductive disk-loaded forward-wave control
# structure ($v_g > 0$). To isolate the structure-induced scattered Cherenkov wave packet
# $\mathbf{E}_{\mathrm{scat}}(\mathbf{r}, t) = \mathbf{E}_{\mathrm{SWS}}(\mathbf{r}, t) - \mathbf{E}_{\mathrm{pipe}}(\mathbf{r}, t)$
# from the bunch's direct Coulomb self-field, each run performs a paired smooth-pipe reference subtraction.

# %%
ls = t3["length_scaling"]
ls48 = t3["length_scaling_nxy48"]
dc = t3["directionality_comparison_N8"]
mc = t3["n8_transverse_mesh_convergence"]
mc_gci = mc["gci"]
r6 = t3["runs_mtm"]["6"]
r8 = t3["runs_mtm"]["8"]
r12 = t3["runs_mtm"]["12"]

print(
    f"3-level transverse mesh convergence at N = 8 (levels n_xy = {mc['levels_n_xy']}, h_xy = {[round(x, 4) for x in mc['h_xy_mm']]} mm, "
    f"cells across 1 mm spoke = {mc['cells_across_1mm_spoke']}, cells across 0.3 mm ASSUMED coax shield = {mc['cells_across_0p3mm_coax_shield']}):\n"
    f"  W_beam,bp,cv [mJ] : {format_gci_line(mc_gci['w_beam_passband_cv_mj'], 'mJ', 4)}\n"
    f"  P_up,max [MW]     : {format_gci_line(mc_gci['p_up_cycle_avg_max_mw'], 'MW', 4)}\n"
    f"  E_up [mJ]         : {format_gci_line(mc_gci['e_up_mj'], 'mJ', 4)}\n"
    f"  E_up,coax [mJ]    : {format_gci_line(mc_gci['e_up_coax_mj'], 'mJ', 4)}\n"
    f"  Port |Gamma(f_s)| : {format_gci_line(mc_gci['port_gamma_tapered_at_sync'], '', 4)}\n"
    f"  f_pk [GHz]        : {format_gci_line(mc_gci['f_pk_ghz'], 'GHz', 4)} "
    f"(n_xy=64 vs paper {pb['f_sync_ghz']:.2f} GHz = {100 * abs(mc['f_pk_ghz_values'][1] - pb['f_sync_ghz']) / pb['f_sync_ghz']:.2f}%, "
    f"vs model nf8 f_s {wb['f_sync_ghz']:.4f} GHz = {100 * abs(mc['f_pk_ghz_values'][1] - wb['f_sync_ghz']) / wb['f_sync_ghz']:.2f}%)\n"
    f"  Re Z_pk [Ohm]     : {format_gci_line(mc_gci['re_z_pk_ohm'], 'Ohm', 2)}\n"
    f"  Coax shield note  : {mc['coax_shield_resolution_note']}"
)
pre_slopes = derive_pre_asymptotic_scaling_slopes(
    [float(x) * 1e-3 for x in ls["active_length_mm"]],
    float(wb["ve_dft_over_c"]) * C0,
    float(t1["beam_u0_m_s"]),
    taper_length_total_m=4.0 * float(t1["spec"]["period_m"]),
    passband_bw_hz=float(t1["spec"]["passband_high_hz"]) - float(t1["spec"]["passband_low_hz"]),
)
print(
    f"Finite-structure length scaling across short N in {t3['n_periods_list']} at n_xy = 64 (L = {ls['active_length_mm']} mm, pre-asymptotic regime):\n"
    f"  f_pk [GHz]          : {[round(t3['runs_mtm'][str(n)]['f_pk_ghz'], 4) for n in t3['n_periods_list']]} "
    f"(N=8 f_pk = {r8['f_pk_ghz']:.4f} GHz vs model nf8 f_s {wb['f_sync_ghz']:.4f} GHz, nf4 beam crossing {wb['f_beam_crossing_nf4_ghz']:.4f} GHz, and paper {pb['f_sync_ghz']:.2f} GHz)\n"
    f"  Re Z_pk [kOhm]      : {[round(z * 1e-3, 3) for z in ls['re_z_pk_ohm']]} "
    f"(single-mode cold-test sinc^2 peak = {[round(z * 1e-3, 3) for z in ls['cold_sinc2_re_z_pk_ohm']]} kOhm, "
    f"FIT/sinc^2 peak ratios = {[round(r, 2) for r in ls['re_z_pk_over_cold_sinc2_ratio']]}x [N=8: {ls['n8_re_z_pk_over_cold_sinc2_ratio']:.2f}x, unexplained peak excess above the +/-{100 * pr['rel_error_bound_tapered']:.1f}% coherent standing-wave bound at |Gamma_tap| = {pr['gamma_tapered_at_sync']:.3f}]; "
    f"wake-passband integrated active-loss ratios int Re Z_FIT / int Re Z_sinc^2 = {[round(r, 2) for r in ls['passband_integrated_re_z_fit_over_cold_sinc2_ratio']]}, "
    f"passband integrated excess int Re Z_FIT - int Re Z_sinc^2 = {[round(x, 1) for x in ls['passband_integrated_re_z_excess_ohm_ghz']]} Ohm*GHz, "
    f"per-period slopes of int Re Z vs N: FIT = {ls['passband_integrated_re_z_fit_slope_ohm_ghz_per_period']:.1f} vs sinc^2 = {ls['passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period']:.1f} Ohm*GHz/period "
    f"(rel. diff = {100 * ls['rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2']:.2f}%; intercepts {ls['passband_integrated_re_z_fit_intercept_ohm_ghz']:.1f} vs {ls['passband_integrated_re_z_cold_sinc2_intercept_ohm_ghz']:.1f} Ohm*GHz, so excess is in the end/taper intercept while differential slope a_W agrees to {100 * ls['rel_error_a_w_vs_cold_analytic']:.2f}%), "
    f"and defined effective widths Delta_f_eff := (int Re Z)/Re Z_pk = {[round(w, 2) for w in ls['effective_width_fit_ghz']]} vs {[round(w, 2) for w in ls['effective_width_cold_sinc2_ghz']]} GHz "
    f"[ratios = {[round(r, 2) for r in ls['effective_width_fit_over_cold_sinc2_ratio']]}, FWHM ratios = {[round(r, 2) for r in ls['fwhm_fit_over_cold_sinc2_ratio']]}]; "
    f"fitted log-log slope = {ls['slope_re_z_pk_vs_L_fit']:.2f}; derived pre-asymptotic interval [{pre_slopes['slope_z_min']:.2f}, {pre_slopes['slope_z_max']:.2f}] vs asymptotic L^2 = 2)\n"
    f"  FWHM [GHz]          : {[round(w, 3) for w in ls['fwhm_ghz']]} "
    f"(cold sinc^2 FWHM = {[round(w, 3) for w in ls['fwhm_cold_sinc2_ghz']]} GHz; fitted log-log slope = {ls['slope_fwhm_vs_L_fit']:.2f}; derived pre-asymptotic interval [{pre_slopes['slope_fwhm_min']:.2f}, {pre_slopes['slope_fwhm_max']:.2f}] vs asymptotic L^-1 = -1)\n"
    f"  W_beam,bp,cv [mJ]   : {[round(w, 3) for w in ls['w_beam_passband_cv_mj']]} "
    f"(affine fit W(L) = a_W*L + b_W: a_W(n_xy=64) = {ls['affine_a_w_j_per_m']:.4f} J/m, a_W(n_xy=48) = {ls48['affine_a_w_j_per_m']:.4f} J/m, "
    f"b_W = {ls['affine_b_w_mj']:.3f} mJ vs cold-test analytic a_W,analytic = {ls['cold_analytic_a_w_j_per_m']:.4f} J/m, "
    f"rel. diff = {100 * ls['rel_error_a_w_vs_cold_analytic']:.2f}% within +/-{100 * ls['a_w_uncertainty_rel']:.2f}% combined uncertainty: "
    f"two-level slope GCI {100 * ls['gci_a_w_two_level_rel']:.2f}% + cold k_L GCI {100 * ls['cold_k_loss_gci_fine_rel']:.2f}% + |Gamma_tap|^2 {100 * (pr['gamma_tapered_at_sync']**2):.2f}%)\n"
    f"  k_loss,bp (= W_bp,cv/q^2) [1e12 V/C]: {[round(k * 1e-12, 3) for k in ls['k_loss_passband_v_per_c']]} "
    f"(affine fit k_L(L) = a_k*L + b_k: a_k = {ls['affine_a_k_loss_v_per_c_m']:.3e} V/(C*m), b_k = {ls['affine_b_k_loss_v_per_c']:.3e} V/C)\n"
    f"  E_up [mJ]           : {[round(e, 3) for e in ls['e_up_mj']]} "
    f"(affine fit E_up(L) = a_E*L + b_E: a_E = {ls['affine_a_e_up_j_per_m']:.4f} J/m, b_E = {ls['affine_b_e_up_mj']:.3f} mJ)\n"
    f"  t_p (20% env) [ns]  : {[round(tp, 4) for tp in ls['pulse_duration_20pct_ns']]} "
    f"(fitted log-log slope = {ls['slope_tp_vs_L_fit']:.2f}; derived pre-asymptotic interval [{pre_slopes['slope_tp_min']:.2f}, {pre_slopes['slope_tp_max']:.2f}] vs asymptotic L^1 = 1; N=8 FIT = {ls['n8_pulse_duration_20pct_fit_ns']:.4f} ns vs analytic {ls['n8_pulse_duration_analytic_ns']:.4f} ns, filter 10-90% rise = {ls['filter_step_rise_time_10_90_ns']:.3f} ns, tol = {ls['pulse_duration_tolerance_ns']:.3f} ns)\n"
    f"  P_up,max [MW]       : {[round(p, 2) for p in ls['p_up_cycle_avg_max_mw']]} "
    f"(fitted log-log slope = {ls['slope_p_up_vs_L_fit']:.2f} vs asymptotic L^0 = 0; paper = {sm['single_bunch_45nc_power_paper_quoted_mw']:.1f} MW, PRL lines 50, 232, 291)"
)
display(
    Markdown(
        f"**Unexplained $\\operatorname{{Re}} Z_{{\\mathrm{{pk}}}}$ excess vs integrated active-loss conservation (from `data/task_03_wakefield_and_directionality.json`):** "
        f"Across $N \\in \\{{6, 8, 12\\}}$, the 3D FIT peak impedance ratios over the reflectionless cold-test $\\mathrm{{sinc}}^2$ formula are "
        f"$\\operatorname{{Re}} Z_{{\\mathrm{{pk,FIT}}}} / \\operatorname{{Re}} Z_{{\\mathrm{{pk,sinc^2}}}} = "
        f"{ls['re_z_pk_over_cold_sinc2_ratio'][0]:.2f}, {ls['re_z_pk_over_cold_sinc2_ratio'][1]:.2f}, {ls['re_z_pk_over_cold_sinc2_ratio'][2]:.2f}$ "
        f"(`{ls['n8_re_z_pk_over_cold_sinc2_ratio']:.2f}` at $N=8$). Fabry–Pérot peaking, bounded by "
        f"`reflection_induced_peak_error_bound` $= 2|\\Gamma_{{\\mathrm{{tap}}}}|/(1-|\\Gamma_{{\\mathrm{{tap}}}}|^2) = "
        f"{100 * pr['rel_error_bound_tapered']:.1f}\\%$ at $|\\Gamma_{{\\mathrm{{tap}}}}| = {pr['gamma_tapered_at_sync']:.3f}$ "
        f"(factor `{1.0 + pr['rel_error_bound_tapered']:.3f}`), cannot account for the $\\times{ls['n8_re_z_pk_over_cold_sinc2_ratio']:.2f}$ peak excess; "
        f"its origin is not identified here (**unexplained**). "
        f"Integrating over the wake passband (`{t1['spec']['wake_passband_low_hz']*1e-9:.1f}–{t1['spec']['wake_passband_high_hz']*1e-9:.1f}` GHz), "
        f"$\\int \\operatorname{{Re}} Z_{{\\mathrm{{FIT}}}}\\,\\mathrm{{d}}f = "
        f"{ls['passband_integrated_re_z_fit_ohm_ghz'][0]:.1f}, {ls['passband_integrated_re_z_fit_ohm_ghz'][1]:.1f}, {ls['passband_integrated_re_z_fit_ohm_ghz'][2]:.1f}\\;\\Omega\\cdot\\mathrm{{GHz}}$ "
        f"vs $\\int \\operatorname{{Re}} Z_{{\\mathrm{{sinc^2}}}}\\,\\mathrm{{d}}f = "
        f"{ls['passband_integrated_re_z_cold_sinc2_ohm_ghz'][0]:.1f}, {ls['passband_integrated_re_z_cold_sinc2_ohm_ghz'][1]:.1f}, {ls['passband_integrated_re_z_cold_sinc2_ohm_ghz'][2]:.1f}\\;\\Omega\\cdot\\mathrm{{GHz}}$ "
        f"(ratios `{ls['passband_integrated_re_z_fit_over_cold_sinc2_ratio'][0]:.2f}, {ls['passband_integrated_re_z_fit_over_cold_sinc2_ratio'][1]:.2f}, {ls['passband_integrated_re_z_fit_over_cold_sinc2_ratio'][2]:.2f}`; "
        f"defining the effective spectral width as $\\Delta f_{{\\mathrm{{eff}}}} := \\int \\operatorname{{Re}} Z\\,\\mathrm{{d}}f / \\operatorname{{Re}} Z_{{\\mathrm{{pk}}}}$ gives "
        f"`{ls['effective_width_fit_ghz'][0]:.2f}` vs `{ls['effective_width_cold_sinc2_ghz'][0]:.2f}` GHz, "
        f"`{ls['effective_width_fit_ghz'][1]:.2f}` vs `{ls['effective_width_cold_sinc2_ghz'][1]:.2f}` GHz, and "
        f"`{ls['effective_width_fit_ghz'][2]:.2f}` vs `{ls['effective_width_cold_sinc2_ghz'][2]:.2f}` GHz, with ratios "
        f"`{ls['effective_width_fit_over_cold_sinc2_ratio'][0]:.2f}, {ls['effective_width_fit_over_cold_sinc2_ratio'][1]:.2f}, {ls['effective_width_fit_over_cold_sinc2_ratio'][2]:.2f}` "
        f"and FWHM ratios `{ls['fwhm_fit_over_cold_sinc2_ratio'][0]:.2f}, {ls['fwhm_fit_over_cold_sinc2_ratio'][1]:.2f}, {ls['fwhm_fit_over_cold_sinc2_ratio'][2]:.2f}`). "
        f"Crucially, the passband integrated excess $\\int \\operatorname{{Re}} Z_{{\\mathrm{{FIT}}}}\\,\\mathrm{{d}}f - \\int \\operatorname{{Re}} Z_{{\\mathrm{{sinc^2}}}}\\,\\mathrm{{d}}f = "
        f"{ls['passband_integrated_re_z_excess_ohm_ghz'][0]:.1f}, {ls['passband_integrated_re_z_excess_ohm_ghz'][1]:.1f}, {ls['passband_integrated_re_z_excess_ohm_ghz'][2]:.1f}\\;\\Omega\\cdot\\mathrm{{GHz}}$ "
        f"for $N = 6, 8, 12$ is roughly constant in $L$, and an affine fit of $\\int \\operatorname{{Re}} Z\\,\\mathrm{{d}}f$ vs $N$ yields per-period slopes of "
        f"`{ls['passband_integrated_re_z_fit_slope_ohm_ghz_per_period']:.1f}` vs `{ls['passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period']:.1f}` $\\Omega\\cdot\\mathrm{{GHz/period}}$ "
        f"(FIT vs $\\mathrm{{sinc}}^2$, agreeing within `{100 * ls['rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2']:.2f}%`, with intercepts "
        f"`{ls['passband_integrated_re_z_fit_intercept_ohm_ghz']:.1f}` vs `{ls['passband_integrated_re_z_cold_sinc2_intercept_ohm_ghz']:.1f}` $\\Omega\\cdot\\mathrm{{GHz}}$). "
        f"Thus the integrated excess resides mainly in the $L$-independent intercept rather than the per-period slope; we hypothesise end/taper effects, "
        f"although the fixed passband window also truncates the broad short-$N$ $\\mathrm{{sinc}}^2$ curves and lowers their intercept. This is consistent with "
        f"the agreement of the differential loss slope $a_W = {ls['affine_a_w_j_per_m']:.4f}\\;\\mathrm{{J/m}}$ "
        f"with the cold-test analytical slope $a_{{W,\\mathrm{{analytic}}}} = {ls['cold_analytic_a_w_j_per_m']:.4f}\\;\\mathrm{{J/m}}$ within `{100 * ls['rel_error_a_w_vs_cold_analytic']:.2f}%`, but does not by itself explain it, since $a_W$ is weighted by the bunch form factor $|F|^2$ and the integral is not. "
        f"Candidate hypotheses for the peak excess beyond the `{100 * pr['rel_error_bound_tapered']:.1f}%` port-reflection bound include: "
        f"(i) a smaller local $|v_g|$ at the finite-structure peak $f_{{\\mathrm{{pk}}}} = {r8['f_pk_ghz']:.2f}\\;\\mathrm{{GHz}}$ than the cold-test value at "
        f"$f_s = {wb['f_sync_ghz']:.2f}\\;\\mathrm{{GHz}}$ (since $\\operatorname{{Re}} Z_{{\\mathrm{{pk}}}} \\propto 1/|v_g|$), "
        f"(ii) the frequency shift between $f_{{\\mathrm{{pk}}}} = {r8['f_pk_ghz']:.2f}\\;\\mathrm{{GHz}}$ and $f_s = {wb['f_sync_ghz']:.2f}\\;\\mathrm{{GHz}}$, "
        f"and (iii) reactive loading and coupling in the two 2-cell adiabatic tapers."
    )
)
print(
    f"Directionality across N in {{6, 8, 12}} at n_xy = 64 (q = 45.0 nC, PRL line 49):\n"
    f"  Wagon-wheel MTM upstream fractions E_up / E_rad: "
    f"N = 6 -> {100 * r6['upstream_energy_fraction']:.2f}%, "
    f"N = 8 -> {100 * dc['mtm_backward_upstream_fraction']:.2f}% "
    f"(E_up = {dc['mtm_backward_e_up_mj']:.2f} mJ vs E_dn = {dc['mtm_backward_e_dn_mj']:.2f} mJ, peak P_up = {dc['mtm_backward_p_up_max_mw']:.2f} MW), "
    f"N = 12 -> {100 * r12['upstream_energy_fraction']:.2f}%\n"
    f"  Forward control at N = 8 (v_g > 0, f_pk = {dc['control_forward_f_pk_ghz']:.2f} GHz): E_dn = {dc['control_forward_e_dn_mj']:.2f} mJ "
    f"({100 * dc['control_forward_downstream_fraction']:.2f}% downstream) vs "
    f"E_up = {dc['control_forward_e_up_mj']:.2f} mJ ({100 * (1 - dc['control_forward_downstream_fraction']):.2f}% upstream); "
    f"peak P_dn = {dc['control_forward_p_dn_max_mw']:.2f} MW\n"
    f"  Scaling note: {ls['scaling_regime_note']}"
)

# %% [markdown]
# ![Longitudinal Wake Impedance and Finite-Length Scaling Laws](figures/04_wake_potential_and_impedance_scaling.png)
#
# ![Reversed-Cherenkov Upstream vs Downstream Poynting Flux and Forward Control](figures/05_reversed_cherenkov_poynting_flux.png)
#
# ![Reversed-Cherenkov vs Forward-Cherenkov Wave-Packet Snapshots](figures/06_reversed_cherenkov_wave_packet.png)
#
# ![Animated Reversed-Cherenkov vs Forward-Cherenkov Wave-Packet Propagation](figures/07_reversed_cherenkov_wave_packet.gif)
#
# **Physical interpretation of Figures 04–07:**
# - **3-level transverse mesh convergence (`n_xy in {48, 64, 96}`):** In the code cell above, `compute_three_level_gci`
#   evaluates the increment ratio $R = e_{32} / e_{21}$, convergence classification (`convergence_class`), and finest-grid
#   uncertainty $U_{\mathrm{fine}} = \max(\mathrm{GCI}(F_s, p_{\mathrm{used}}), (\max f - \min f) / (2|f_{96}|))$ directly
#   from `data/task_03_wakefield_and_directionality.json` for every observable ($W_{\mathrm{beam,bp,cv}}$, $P_{\mathrm{up,max}}$,
#   $E_{\mathrm{up}}$, $E_{\mathrm{up,coax}}$, $|\Gamma_{\mathrm{tap}}(f_s)|$, $f_{\mathrm{pk}}$, and $\operatorname{Re} Z_{\mathrm{pk}}$,
#   with `k_loss_passband_v_per_c` $= W_{\mathrm{beam,bp,cv}} / q^2$). Because the unrounded $90^\circ$ PEC spoke and hub
#   corners (`r_fillet = 0.0 mm`, `ASSUMED`) create field singularities, the formal spatial order is first order
#   ($p_{\mathrm{formal}} = 1$). Furthermore, because the `ASSUMED` coaxial inner shield thickness
#   $r_{\mathrm{coax}} - r_b = 0.30\;\mathrm{mm}$ (PRL line 122, `ASSUMED`) spans fewer than two transverse cells across
#   `n_xy = 48, 64, 96` (exact cell counts and port reflections printed in `mc["coax_shield_resolution_note"]` above),
#   subcell conformal PEC cut-cell fractions vary non-monotonically across the three grids; observables that depend on
#   the coaxial port boundary ($W_{\mathrm{beam,bp,cv}}$, $P_{\mathrm{up,max}}$, $E_{\mathrm{up}}$, and $|\Gamma_{\mathrm{tap}}|$)
#   are therefore classified from $R$ without Richardson extrapolation, using $F_s = 3, p_{\mathrm{used}} = 1$ and the
#   Stern half-range when $R < 0$, while $f_{\mathrm{pk}}$ and $\operatorname{Re} Z_{\mathrm{pk}}$ have their solved
#   $p_{\mathrm{obs}}$ and $F_s$ printed from JSON above.
# - **Near-constant backward-wave peak power ($P_{\mathrm{up,max}} \propto L^0$):** Across $N \in \{6, 8, 12\}$, the
#   cycle-averaged upstream output power remains nearly length-invariant; in Part 5 we compare $P_{\mathrm{up,max}}$ first
#   against the model's own Eq. (2) analytical prediction and then show that it is consistent with the paper's measured
#   "up to $25.0\;\mathrm{MW}$" (PRL lines 50, 232, 291) at $N = 40$ ($L = 80.0\;\mathrm{mm}$, PRL lines 134, 166) within the combined finest-grid
#   uncertainty and two-port reflection band $\pm(U_P + 2|\Gamma|^2)$ printed from JSON in Part 5.
# - **Affine length scaling of passband beam work $W_{\mathrm{beam,bp,cv}}(L)$ and upstream energy $E_{\mathrm{up}}(L)$:**
#   1. *Affine energy scaling ($y = a L + b$ with taper intercept):* Because the control volume $[z_{\mathrm{up}}, z_{\mathrm{dn}}]$
#      encloses $N$ active wagon-wheel cells of length $L = N p$ plus two 2-cell ($4\;\mathrm{mm}$, `ASSUMED`) adiabatic
#      coaxial tapers ($L_{\mathrm{tap}} = 8\;\mathrm{mm}$, `ASSUMED`), both $W_{\mathrm{beam,bp,cv}}(L) = a_W L + b_W$ and
#      $E_{\mathrm{up}}(L) = a_{E,\mathrm{up}} L + b_{E,\mathrm{up}}$ are **affine** in $L$ with positive end/taper intercepts
#      $b_W > 0$ and $b_{E,\mathrm{up}} > 0$ rather than pure power laws through the origin. The fitted active-section
#      slope $a_W = \mathrm{d}W_{\mathrm{beam,bp,cv}} / \mathrm{d}L$ agrees with the cold-test analytic loss per unit length
#      $a_{W,\mathrm{analytic}} = q^2 k_L \Phi^2 / (1 - v_g / u_0)$ within the combined uncertainty
#      (two-level slope GCI between `n_xy = 48` and `64` with $F_s = 3, p = 1$, plus cold-test $k_L$ GCI, plus $|\Gamma_{\mathrm{tap}}|^2$)
#      printed in `ls["scaling_regime_note"]` above.
#   2. *Pre-asymptotic pulse-duration and bandwidth exponents & unexplained $\operatorname{Re} Z_{\mathrm{pk}}$ excess over cold-test $\mathrm{sinc}^2$:*
#      The Gaussian bunch ($\sigma_z = 1.2\;\mathrm{mm}$, PRL line 151), taper transit length, and causal passband filter add an
#      $L$-independent offset to $t_p^{(20\%)}(L)$, placing the log-log slopes of $t_p(L)$, $\mathrm{FWHM}(L)$, and
#      $\operatorname{Re} Z_{\mathrm{pk}}(L)$ in the derived pre-asymptotic intervals printed in `ls["scaling_regime_note"]`
#      above. In Figure 04(a), where the cold-test $\mathrm{sinc}^2(\Delta k_z L / 2)$ curves use the `nf8` Bloch cold-test
#      parameters ($k_L$, $v_E$, and $f_s$ from `t2["wakis_finest_nf8_benchmarks"]`, plus a dotted $N=8$ reference shifted
#      to the FIT $f_{\mathrm{pk}}$), the 3D FIT peak resistance $\operatorname{Re} Z_{\mathrm{pk}}$ exceeds the
#      reflectionless single-mode $\mathrm{sinc}^2$ peak (`ls["cold_sinc2_re_z_pk_ohm"]`) by the per-$N$ ratios
#      `ls["re_z_pk_over_cold_sinc2_ratio"]` (`ls["n8_re_z_pk_over_cold_sinc2_ratio"]` at $N=8$) printed from JSON above.
#      Because the tutorial's own coherent standing-wave bound `reflection_induced_peak_error_bound`
#      $= 2|\Gamma_{\mathrm{tap}}| / (1 - |\Gamma_{\mathrm{tap}}|^2)$ (`pr["rel_error_bound_tapered"]` at
#      $|\Gamma_{\mathrm{tap}}| = $ `pr["gamma_tapered_at_sync"]`, printed above) is much smaller than this peak ratio,
#      Fabry–Pérot standing-wave peaking cannot explain the $\operatorname{Re} Z_{\mathrm{pk}}$ excess, and we call the peak
#      excess **unexplained**. At the same time, defining the effective spectral width as
#      $\Delta f_{\mathrm{eff}} := (\int \operatorname{Re} Z_\parallel\,\mathrm{d}f) / \operatorname{Re} Z_{\mathrm{pk}}$
#      (`ls["effective_width_fit_ghz"]`, `ls["effective_width_cold_sinc2_ghz"]`, and `ls["fwhm_fit_over_cold_sinc2_ratio"]` printed above),
#      the passband integrated excess $\int \operatorname{Re} Z_{\mathrm{FIT}}\,\mathrm{d}f - \int \operatorname{Re} Z_{\mathrm{sinc}^2}\,\mathrm{d}f$
#      (`ls["passband_integrated_re_z_excess_ohm_ghz"]`, printed above) is roughly constant across $N \in \{6, 8, 12\}$,
#      and the affine per-period slopes of $\int \operatorname{Re} Z\,\mathrm{d}f$ vs $N$
#      (`ls["passband_integrated_re_z_fit_slope_ohm_ghz_per_period"]` vs `ls["passband_integrated_re_z_cold_sinc2_slope_ohm_ghz_per_period"]`,
#      agreeing to `ls["rel_diff_passband_integrated_re_z_slope_fit_vs_cold_sinc2"]`) show that the integrated excess is in the
#      $L$-independent intercept (hypothesis: end/taper effects; the fixed passband window also truncates the broad
#      short-$N$ sinc² curves) rather than the per-period slope, consistent with the close agreement between the differential loss slope $a_W$ and $a_{W,\mathrm{analytic}}$. Candidate hypotheses for the
#      unexplained peak excess include a smaller local $|v_g|$ at $f_{\mathrm{pk}}$ (`r8["f_pk_ghz"]`) than the cold-test value
#      at $f_s$ (`wb["f_sync_ghz"]`, since $\operatorname{Re} Z_{\mathrm{pk}} \propto 1/|v_g|$), the shift between
#      $f_{\mathrm{pk}}$ and $f_s$, and taper coupling.
# - **Unmistakable reversed-Cherenkov signature:** In the wagon-wheel metamaterial ($v_g < 0$), the majority of the
#   passband radiated energy exits through the **upstream** ($-z$) beam entrance port (with exact per-$N$ fractions
#   `r6["upstream_energy_fraction"]`, `r8["upstream_energy_fraction"]`, and `r12["upstream_energy_fraction"]` printed
#   from JSON in the code cell above), whereas in the non-inductive disk-loaded forward-wave control structure ($v_g > 0$),
#   the energy flow reverses to the **downstream** ($+z$) beam exit port.

# %% [markdown]
# ---
#
# ## Part 4 — Two-Bunch Coherent Superposition, Finite-Length Phase Scan & Charge-Scaled Single-Bunch Estimate
#
# In Lu *et al.* (2019, PRL lines 51, 77, 143, 153–154, 255–257, 297–298, 323, 329–331, 337–338, Fig. 5; Lu thesis lines
# 2337–2341, 2365), two $65.0\;\mathrm{MeV}$ (PRL line 49) bunches spaced around one $1.3\;\mathrm{GHz}$ (PRL line 143)
# linac RF period ($T_b = 1 / 1.3\;\mathrm{GHz} = 769.23\;\mathrm{ps}$, PRL line 143) and fine-tuned in relative delay
# with laser splitters before the photocathode gun (PRL lines 153–154, 256–257, 329–331; thesis lines 2338–2341) with a combined charge of
# $Q_1 + Q_2 = 85.0\;\mathrm{nC}$ (PRL line 51) generated up to $80.0\;\mathrm{MW}$ (PRL lines 51, 77, 298, 338) peak output power in
# the $N = 40$ ($L = 80.0\;\mathrm{mm}$, PRL lines 134, 166) structure. Notice that an exact nominal spacing of
# $T_b = 1 / 1.3\;\mathrm{GHz}$ (PRL line 143) equals $m = T_b f_s$ non-integer cycles of the synchronous mode (printed from
# `t4["harmonic_number_m_at_1p3ghz"]` and `t4["phase_mod_360_deg_at_1p3ghz"]` below) and is **not** in phase without
# optical delay tuning; as stated in Lu's thesis (lines 2340–2341, 2365), $11.42\;\mathrm{GHz}$ (PRL lines 137, 149) is the 9th harmonic
# of a $f_s / 9 = 1.27\;\mathrm{GHz}$ (thesis line 2341) bunch train with tuned spacing $T_b = 9 / f_s = 0.79\;\mathrm{ns}$
# (thesis line 2365; exact values `t4["ninth_harmonic_bunch_train_freq_ghz"]` and `t4["ninth_harmonic_bunch_spacing_ns"]`
# printed from JSON below). Below we:
#
# 1. **Verify 3D FIT operator linearity (`verify_two_bunch_fit_superposition`):** Running bunch 1 alone
#    ($\mathbf{E}_1, \mathbf{H}_1$), bunch 2 alone ($\mathbf{E}_2, \mathbf{H}_2$), and both bunches together
#    ($\mathbf{E}_{1+2}, \mathbf{H}_{1+2}$) in 3D conformal FIT satisfies $\mathbf{E}_{1+2} = \mathbf{E}_1 + \mathbf{E}_2$
#    to IEEE-754 double-precision round-off, whereas shifting bunch 2 by half a time step breaks the identity.
# 2. **Evaluate passband-filtered 3D FIT wake interference (`t4["equal_bunch_superposition"]`):** Rather than projecting
#    a tautological pure sinusoid, we superpose the passband-filtered 3D FIT longitudinal wake potential $W_\parallel(s)$
#    from the $N = 8$ (`n_xy = 64`) simulation for two equal bunches ($Q_1 = Q_2$) across in-phase ($\Delta\psi = 360^\circ$)
#    and out-of-phase ($\Delta\psi = 540^\circ$) separations.
# 3. **Why $N = 8$ cannot overlap two bunches at the experimental $769.23\;\mathrm{ps}$ (PRL line 143) spacing, and
#    charge-scaled single-bunch estimate:**
#    - In the $N = 40$ ($L = 80.0\;\mathrm{mm}$, PRL lines 134, 166) AWA experiment, $t_p(N=40) > T_b = 769.23\;\mathrm{ps}$
#      (PRL line 143; $T_b = 0.79\;\mathrm{ns}$ in thesis line 2365), so the two wave packets overlap inside the structure when tuned in phase by the laser splitters
#      (PRL lines 153–154, 256–257, 329–331; thesis lines 2338–2341).
#    - In our short $N = 8$ simulation, $t_p^{(20\%)} < T_b = 769.23\;\mathrm{ps}$ (PRL line 143): bunch 1's wave packet
#      exits before bunch 2 arrives at $T_b = 769.23\;\mathrm{ps}$ (PRL line 143), producing two non-overlapping pulses.
#    - Therefore, to estimate the peak power when two bunches of combined charge $Q_1 + Q_2 = 85.0\;\mathrm{nC}$ (PRL line 51;
#      $Q_1 = Q_2 = 42.5\;\mathrm{nC}$ under equal division of $85.0\;\mathrm{nC}$, PRL line 51) overlap
#      constructively, we report a **charge-scaled single-bunch estimate** obtained by scaling the $q = 45.0\;\mathrm{nC}$ (PRL line 49)
#      single-bunch result by $((Q_1 + Q_2)/q)^2$, alongside a 1-RF-cycle ($\Delta t = 1/f_s$) shifted
#      single-bunch envelope superposition (all values printed from `data/task_04_two_bunch_superposition.json` below).

# %%
sup = t4["fit_3d_superposition_verification"]
eq_b = t4["equal_bunch_superposition"]
c85 = t4["two_bunch_85nc_comparison"]
c85_tot = c85["convention_total_85nc_q1_42p5_q2_42p5"]
c85_each = c85["convention_each_85nc_q1_85_q2_85"]

p_cs_1rf_mw = float(c85_tot["charge_scaled_1rf_shifted_envelope_peak_mw"])
p_cs_4x_mw = float(c85_tot["charge_scaled_single_bunch_4x_42p5nc_mw"])
p_cs_85_1_mw = float(c85_each["charge_scaled_single_bunch_85nc_mw"])
p_cs_85_4x_mw = float(c85_each["charge_scaled_single_bunch_4x_85nc_mw"])
p_cs_85_4x_an_mw = float(c85_each["charge_scaled_single_bunch_4x_85nc_analytic_mw"])

print(
    f"3D FIT two-bunch linear superposition verification (N = {sup['n_periods']}, {sup['n_steps']} steps):\n"
    f"  ||E_12 - (E_1 + E_2)||_2 / ||E_12||_2 = {sup['rel_l2_error_E_3d']:.3e}\n"
    f"  ||H_12 - (H_1 + H_2)||_2 / ||H_12||_2 = {sup['rel_l2_error_H_3d']:.3e}\n"
    f"  Mutated (+dt/2 shift on bunch 2)      = {sup['mutated_half_dt_shift_rel_l2_error_E']:.3e}"
)
print(
    f"Passband-filtered 3D FIT wake interference (lambda_rf = {t4['rf_wavelength_mm']:.2f} mm, T_rf = {t4['rf_period_ps']:.2f} ps):\n"
    f"  In-phase (ds = {eq_b['in_phase_spacing_mm']:.2f} mm)     : wake ratio = {eq_b['in_phase_wake_ratio_over_single']:.3f}x, "
    f"power ratio = {eq_b['in_phase_power_ratio_over_single']:.3f}x (ideal infinite-N = 4x)\n"
    f"  Out-of-phase (ds = {eq_b['out_of_phase_spacing_mm']:.2f} mm) : wake ratio = {eq_b['out_of_phase_wake_ratio_over_single']:.3f}x, "
    f"power ratio = {eq_b['out_of_phase_power_ratio_over_single']:.4f}x (finite-N envelope decay residual)"
)
print(
    f"Experimental bunch spacing vs N=8 pulse duration:\n"
    f"  Linac spacing T_b = 1/(1.3 GHz, PRL line 143) = {t4['linac_period_ps']:.2f} ps "
    f"(exact 1/(1.3 GHz, PRL line 143) harmonic number m = T_b*f_s = {t4['harmonic_number_m_at_1p3ghz']:.3f} cycles, "
    f"phase mod 360 deg = {t4['phase_mod_360_deg_at_1p3ghz']:.1f} deg at {pb['f_sync_ghz']:.2f} GHz, PRL lines 137, 149; "
    f"model m = {t4['harmonic_number_m_at_1p3ghz_model']:.3f} cycles, {t4['phase_mod_360_deg_at_1p3ghz_model']:.1f} deg)\n"
    f"  9th-harmonic tuned spacing (Lu thesis lines 2340-2341, 2365): f_s/9 = {t4['ninth_harmonic_bunch_train_freq_ghz']:.4f} GHz "
    f"(thesis: 1.27 GHz, line 2341), T_b = 9/f_s = {t4['ninth_harmonic_bunch_spacing_ns']:.4f} ns "
    f"(thesis: 0.79 ns, line 2365; model f_pk/9 = {t4['ninth_harmonic_bunch_train_freq_ghz_model']:.4f} GHz, "
    f"9/f_pk = {t4['ninth_harmonic_bunch_spacing_ns_model']:.4f} ns)\n"
    f"  N=8 pulse t_p^(20%) = {t4['n8_pulse_duration_20pct_ps']:.2f} ps (N=40 analytic t_p = {t4['n40_experiment_pulse_duration_ps']:.1f} ps)\n"
    f"  Can N=8 overlap two bunches at 769.23 ps (PRL line 143) spacing? {t4['n8_can_overlap_at_experiment_spacing']} "
    f"(non-overlapping peak in N=8 = {c85_tot['non_overlapping_769ps_spacing_n8_peak_mw']:.2f} MW)\n"
    f"  Note: {t4['bunch_spacing_limitation_note']}"
)
print(
    f"Charge-scaled single-bunch estimates for Q1 + Q2 = 85.0 nC (PRL line 51) vs Lu et al. (2019) {c85['paper_reported_two_bunch_peak_power_mw']:.1f} MW (PRL lines 51, 77, 298, 338) benchmark:\n"
    f"  Convention A (Q1 + Q2 = 85.0 nC, PRL line 51): "
    f"Eq. (2) = {c85_tot['in_phase_eq2_analytic_mw']:.2f} MW (rel. diff {100 * c85_tot['rel_diff_eq2_vs_paper_80mw']:.2f}%), "
    f"charge-scaled 1-RF-shifted (dt = {t4['rf_period_ps']:.2f} ps) single-bunch estimate = {p_cs_1rf_mw:.2f} MW "
    f"(zero-delay 4*P1 charge-scaled single-bunch estimate = {p_cs_4x_mw:.2f} MW)\n"
    f"  Convention B (Q1 = Q2 = 85.0 nC, PRL line 51): "
    f"charge-scaled single-bunch 85 nC estimate = {p_cs_85_1_mw:.2f} MW, "
    f"4x charge-scaled single-bunch estimate = {p_cs_85_4x_mw:.2f} MW (Eq. 2: {p_cs_85_4x_an_mw:.2f} MW)"
)

# %% [markdown]
# ![Two-Bunch Coherent RF Power and Phase Interference](figures/08_two_bunch_coherent_superposition.png)

# %% [markdown]
# ---
#
# ## Parts 5 & 6 — Analytic Backward-Wave Power Checks & Unified Control-Volume Poynting Energy Balance
#
# Substituting our 3D Bloch cold-test eigenmode parameters (`nf8`) into the backward-wave power formula Eq. (2)
# (PRL lines 240–245, citing T. P. Wangler, *RF Linear Accelerators*, Ref. 24 at lines 238, 370–371) predicts the
# single-bunch ($q = 45.0\;\mathrm{nC}$, PRL line 49) steady-state upstream output power
# $P_{\mathrm{cold,Wakis}}$, which we compare below first against the cycle-averaged 3D FIT wakefield peak power
# $P_{\mathrm{FIT},N=8}$ (at `n_xy = 64` and `96`) and then against the paper's measured "up to $25.0\;\mathrm{MW}$"
# at $N = 40$ (PRL lines 50, 232, 291) within the combined finest-level mesh uncertainty and two-port reflection band
# $\pm(U_P + 2|\Gamma|^2)$ printed from JSON below.
#
# Notice how critical the backward-wave group-front slippage factor $(1 - v_g / u_0)^{-2} = (1 + |v_g| / u_0)^{-2}$
# is: dropping $(1 - v_g / u_0)^{-2}$ or flipping the sign of $v_g \to +|v_g|$ (treating the structure as a forward
# wave with $(1 - |v_g| / u_0)^{-2}$) substantially overpredicts the output power, as quantified in the code cell below.
#
# ### Unified Control-Volume Accounting & Causal Filtering (Part 6)
#
# To close the discrete FIT Poynting theorem across all three structure lengths $N \in \{6, 8, 12\}$, all energy terms
# in `run_finite_mtm_wakefield` are evaluated over the **single control volume** $z \in [z_{\mathrm{up}}, z_{\mathrm{dn}}]$
# bounded by the upstream and downstream port planes ($z_{\mathrm{up}} = z_{\mathrm{up,taper}} - p$ and
# $z_{\mathrm{dn}} = z_{\mathrm{dn,taper}} + p$, enclosing the $N$ active wagon-wheel cells and both 2-cell adiabatic
# coaxial tapers) over the drainage window $T_{\mathrm{end}} = t_i + (z_{\mathrm{tot}} + L_{\mathrm{wake}}(N)) / u_0$
# where $L_{\mathrm{wake}}(N)$ is set by `estimate_drainage_wake_length_m(N)` from dispersive passband transit across
# $(N + 4)p$ plus Fabry–Pérot cavity ring-down so that $U_{\mathrm{bp}}(T_{\mathrm{end}}) / W_{\mathrm{beam,bp}} < 0.02$
# at every $N \in \{6, 8, 12\}$ (exact values printed from JSON below):
#
# $$
# W_{\mathrm{beam}} = -\int_0^{T_{\mathrm{end}}} \int_{z_{\mathrm{up}}}^{z_{\mathrm{dn}}} J_z E_z\,\mathrm{d}V\,\mathrm{d}t
# \;=\; E_{\mathrm{up}} + E_{\mathrm{dn}} + U(T_{\mathrm{end}}),
# $$
#
# where:
# 1. **Unfiltered discrete FIT closure, unexplained late-time unfiltered energy growth on the coarse (`3 cells/spoke`) mesh,
#    & unexplained smooth-pipe reference-run artifact ($W_{\mathrm{ref,unf,cv}}$):** Using
#    conformal FIT primal/dual volume weights (`Ds`, `itDa`, `tDs`, and stabilized `iDa`) and half-step leapfrog
#    time-averaging ($\mathbf{E}^{n}$ paired with $\frac{1}{2}(\mathbf{H}^{n-1/2} + \mathbf{H}^{n+1/2})$ at $t_n = n\,\Delta t$),
#    the unfiltered control-volume energy balance
#    $|E_{\mathrm{rad,unf}} + U_{\mathrm{unf}}(T_{\mathrm{end}}) - W_{\mathrm{beam,unf}}| / \max(|W_{\mathrm{beam,unf}}|, |E_{\mathrm{rad,unf}}|, |U_{\mathrm{unf}}(T_{\mathrm{end}})|)$
#    serves as a discrete bookkeeping check. At `n_xy = 64` ($4\;\mathrm{cells/spoke}$) and `n_xy = 96` ($6\;\mathrm{cells/spoke}$),
#    the metamaterial runs ($N \in \{6, 8, 12\}$) and the `n_xy = 64` forward-wave control run (`forward_control_N8`) close to
#    `rel_error_unfiltered_cv_closure` (with the exact per-$N$ values and maximum formatted from JSON in the code cell below). By contrast,
#    on the coarse `n_xy = 48` ($3\;\mathrm{cells/spoke}$) mesh (`t3["runs_mtm_nxy48"]` and `t3["forward_control_N8_nxy48"]`),
#    the unfiltered stored energy `u_stored_unfiltered_cv_mj` exhibits **unexplained late-time unfiltered energy growth**
#    (accompanied at $N = 8, 12$ and in `forward_control_N8_nxy48` by large negative `e_rad_unfiltered_cv_mj`, and at $N = 12$
#    reaching more than three orders of magnitude above the beam work `w_beam_unfiltered_cv_mj` with a closure error above unity;
#    all per-$N$ values formatted from JSON in the prose and print cell below). This coarse-mesh (`3 cells/spoke`) late-time
#    unfiltered growth is distinct from the separate **unexplained** late-time unfiltered artifact in the empty smooth-pipe
#    reference run (`solver_ref`): across $N \in \{6, 8, 12\}$ at `n_xy = 64`, the unfiltered reference-run control-volume work
#    is $W_{\mathrm{ref,unf,cv}} = W_{\mathrm{beam,unf,cv}} - W_{\mathrm{beam,scat,full,cv}}$ (`w_ref_unfiltered_cv_mj`),
#    accompanied at $N = 8, 12$ by late-time unfiltered stored energy (`u_stored_ref_unfiltered_cv_mj`) and net Poynting inflow
#    (`e_rad_ref_unfiltered_cv_mj < 0`, all printed per $N$ from JSON below), which enters all unfiltered and full-band
#    reference-subtracted numbers (`w_beam_unfiltered_cv_mj`, `w_beam_scat_fullband_cv_mj`, `w_loss_wake_fullband_cv_mj`,
#    `w_loss_wake_fullband_active_mj`, and `w_loss_z_fullband_active_mj`). Crucially, passband-filtered quantities are
#    **unaffected** by both artifacts: in the smooth-pipe reference run, `u_stored_ref_passband_cv_mj` and `w_ref_passband_cv_mj`
#    are five orders of magnitude smaller, and in the coarse `n_xy = 48` metamaterial runs ($N \in \{6, 8, 12\}$), the passband
#    residual stored energy fraction `u_stored_over_w_beam_passband` ($< 0.02$) and passband control-volume closure
#    `rel_error_total_accounted_vs_bunch_loss` ($< 10^{-3}$, both formatted per $N$ from `t3["runs_mtm_nxy48"]` below) remain
#    clean, confirming that the passband slope $a_W(48)$ and the coarse GCI level are not contaminated by the high-frequency
#    late-time unfiltered growth.
# 2. **Two passband definitions & why passband closure holds with scattered $E_{\mathrm{rad,bp}}, U_{\mathrm{bp}}$:**
#    We distinguish two passband windows (`t1["spec"]`, printed below):
#    - **Causal time-domain field filter (`9.8–13.0 GHz`, `ASSUMED`):** A 3rd-order causal Butterworth bandpass impulse
#      response $h_{\mathrm{bp}}(t)$ (with a $5\%$ cosine tail taper) applied via zero-padded ($\ge 2\times$) linear FFT
#      convolution to the time-domain fields entering $W_{\mathrm{beam,bp}}$, $E_{\mathrm{up}}$, $E_{\mathrm{dn}}$, and
#      $U_{\mathrm{bp}}(T_{\mathrm{end}})$. Note that $W_{\mathrm{beam,bp}}$ integrates the total SWS electric field
#      $E_{z,\mathrm{SWS}}$ against $J_z$, whereas $E_{\mathrm{rad,bp}}$ and $U_{\mathrm{bp}}(T_{\mathrm{end}})$ use the
#      scattered field $\mathbf{E}_{\mathrm{scat}} = \mathbf{E}_{\mathrm{SWS}} - \mathbf{E}_{\mathrm{pipe}}$. Because the
#      smooth reference pipe of radius $r_b = 3.0\;\mathrm{mm}$ (PRL line 122) is deep below its fundamental $\mathrm{TM}_{01}$
#      cutoff in the `9.8–13.0 GHz` (`ASSUMED`) band, its passband-filtered residual stored energy and radiation vanish
#      ($U_{\mathrm{ref,bp}}(T_{\mathrm{end}}) \approx 0$ and $W_{\mathrm{ref,bp,cv}} \approx 0$, printed per $N$ below),
#      which is why the passband control-volume identity closes to $O((\omega_s \Delta t)^2)$. Shrinking the beam-work
#      integration domain back to only the active $N$-cell interval $[z_{\mathrm{act,start}}, z_{\mathrm{act,end}}]$ while
#      keeping the port planes at $[z_{\mathrm{up}}, z_{\mathrm{dn}}]$ omits the beam work performed in the two 2-cell
#      tapers and breaks closure.
#    - **Spectral impedance integration window (`9.5–13.5 GHz`, `ASSUMED`):** A slightly wider frequency window used when
#      integrating $2 q^2 \int \operatorname{Re} Z_\parallel(f) |F(f)|^2\,\mathrm{d}f$ from `WakeSolver`
#      (`w_loss_wake_passband_active_mj` and `w_loss_wake_passband_cv_mj`, whereas `k_loss_passband_v_per_c`
#      $= W_{\mathrm{beam,bp,cv}} / q^2$ is the causally filtered time-domain control-volume beam work normalized by $q^2$)
#      to capture the sinc-broadened main lobe of short-$N$ structures.
# 3. **Full-band vs passband bunch energy loss and out-of-band share:** In `WakeSolver.calc_long_Z`, we evaluate the
#    impedance spectrum up to $f_{\max}$ (printed from JSON below), where the Gaussian bunch form factor squared for
#    $\sigma_z = 1.2\;\mathrm{mm}$ (PRL line 151) has decayed to $|F(f_{\max})|^2 < 10^{-3}$ (whereas truncating at
#    $25\;\mathrm{GHz}$ leaves a large unattenuated form factor $|F(25\;\mathrm{GHz})|^2$, both printed from JSON below).
#    Below we report both the full-band loss ($-q^2 \int \lambda(s) W_\parallel(s)\,\mathrm{d}s$ and
#    $2 q^2 \int_0^{f_{\max}} \operatorname{Re} Z_\parallel(f) |F(f)|^2\,\mathrm{d}f$, without clipping
#    $\operatorname{Re} Z_\parallel$ at zero) and the passband share, quantifying the out-of-band energy radiated
#    outside the `9.5–13.5 GHz` (`ASSUMED`) band (with the coaxial extraction ports carrying a cutoff-free TEM mode at all
#    frequencies).
# 4. **Role of port reflection $|\Gamma|$ and residual stored energy $U_{\mathrm{bp}}(T_{\mathrm{end}})$:** Partial reflection
#    $|\Gamma|$ at the coaxial couplers creates a standing-wave peak-power bound `reflection_induced_peak_error_bound`
#    $= 2|\Gamma| / (1 - |\Gamma|^2)$ and single-pass reflection fraction `reflection_induced_energy_error_bound` $= |\Gamma|^2$
#    when comparing extracted port power against the reflectionless cold-test formula Eq. (2). Inside the closed control
#    volume $[z_{\mathrm{up}}, z_{\mathrm{dn}}]$, port reflection and slow group velocity near the passband edges
#    leave a small residual stored fraction $U_{\mathrm{bp}}(T_{\mathrm{end}}) / W_{\mathrm{beam,bp}} < 0.02$ at
#    $T_{\mathrm{end}}$ (printed per $N$ below) while preserving net control-volume closure.

# %%
wa = t5["wakis_cold_test_analytic"]
pa = t5["paper_cold_test_analytic"]
wc = t5["wakefield_vs_analytic_comparison"]
eb = t5["energy_balance_N8"]
eb_all = t5["energy_balance_all_N"]
e_refl_bound = reflection_induced_energy_error_bound(float(pr["gamma_tapered_at_sync"]))
u0_beam = float(t1["beam_u0_m_s"])
sigmaz_m = float(t1["spec"]["bunch_length_rms_m"])
f_bp_lo_ghz = float(t1["spec"]["passband_low_hz"]) * 1e-9
f_bp_hi_ghz = float(t1["spec"]["passband_high_hz"]) * 1e-9
f_wk_lo_ghz = float(t1["spec"]["wake_passband_low_hz"]) * 1e-9
f_wk_hi_ghz = float(t1["spec"]["wake_passband_high_hz"]) * 1e-9

print(
    f"Single-bunch (q = 45.0 nC, PRL line 49) backward-wave RF power comparison:\n"
    f"  Wakis cold-test Eq. (2)  : {wa['single_bunch_45nc_power_mw']:.2f} MW\n"
    f"  Wakis 3D FIT wake (N=8)  : n_xy=64 = {wc['wakis_3d_fit_power_N8_mw']:.2f} MW "
    f"(rel. diff vs Wakis Eq. 2 = {100 * wc['rel_error_fit_vs_wakis_cold_analytic']:.2f}%), "
    f"finest n_xy=96 = {wc['wakis_3d_fit_power_N8_nxy96_mw']:.2f} MW "
    f"(rel. diff vs Wakis Eq. 2 = {100 * wc['rel_error_fit_nxy96_vs_wakis_cold_analytic']:.2f}%; "
    f"convergence class = {wc['p_up_convergence_class']}, finest-level U_P = {100 * wc['p_up_u_fine_nxy96_rel']:.2f}%, "
    f"GCI_med = {100 * wc['p_up_gci_med_nxy64_rel']:.2f}%, two-port reflection bound 2|Gamma|^2 = {100 * wc['two_port_reflection_power_bound']:.2f}%, "
    f"standing-wave peak bound = +/-{100 * pr['rel_error_bound_tapered']:.1f}%)\n"
    f"  Paper cold-test Eq. (2)  : {pa['single_bunch_45nc_power_mw']:.2f} MW; "
    f"paper N=40 experiment quotes up to {pa['paper_quoted_power_mw']:.1f} MW (PRL lines 50, 232, 291): "
    f"n_xy=64 rel. diff = {100 * wc['rel_error_fit_vs_paper_25mw']:.2f}%, "
    f"n_xy=96 rel. diff = {100 * wc['rel_error_fit_nxy96_vs_paper_25mw']:.2f}% "
    f"(consistent within +/-(U_P + 2|Gamma|^2) = +/-{100 * wc['consistency_band_u_p_plus_2gamma_sq']:.2f}%)\n"
    f"  Mutated (no slippage)    : {wa['mutated_no_slippage_power_mw']:.2f} MW\n"
    f"  Mutated (flipped +|v_g|) : {wa['mutated_flipped_vg_sign_power_mw']:.2f} MW"
)
print(
    f"Unified control-volume [z_up, z_dn] Poynting energy balance & full-band vs passband breakdown across N in {{6, 8, 12}} "
    f"(q = 45.0 nC, PRL line 49, sigma_z = 1.2 mm, PRL line 151; "
    f"time-domain field filter = [{f_bp_lo_ghz:.1f}, {f_bp_hi_ghz:.1f}] GHz, WakeSolver spectral passband = [{f_wk_lo_ghz:.1f}, {f_wk_hi_ghz:.1f}] GHz):"
)
unf_48_bound = 1.0e-3
for n_key, ebn in eb_all.items():
    rn_t3 = t3["runs_mtm"][n_key]
    rn_48 = t3["runs_mtm_nxy48"][n_key]
    u_st_fit = float(ebn["residual_stored_energy_fit_mj"])
    w_cv_bp = float(ebn["bunch_energy_loss_wake_passband_mj"])
    w_act_bp = float(ebn["w_loss_wake_passband_active_mj"])
    w_act_full = float(ebn["w_loss_wake_fullband_active_mj"])
    w_cv_full_scat = float(ebn["w_beam_scat_fullband_cv_mj"])
    w_cv_unf = float(ebn["w_beam_unfiltered_cv_mj"])
    w_ref_unf = float(ebn["w_ref_unfiltered_cv_mj"])
    e_rad_ref_unf = float(ebn["e_rad_ref_unfiltered_cv_mj"])
    u_ref_unf = float(ebn["u_stored_ref_unfiltered_cv_mj"])
    err_ref_unf = float(ebn["rel_error_ref_unfiltered_cv_closure"])
    u_ref_bp = float(ebn["u_stored_ref_passband_cv_mj"])
    oob_act = float(ebn["out_of_band_wake_loss_fraction_active"])
    oob_cv = float(ebn["out_of_band_beam_work_fraction_cv"])
    fmax_z = float(ebn["fmax_impedance_ghz"])
    ff_fmax = float(ebn["form_factor_sq_at_fmax"])
    ff_25 = float(ebn["form_factor_sq_at_25ghz"])
    err_unf = float(ebn["rel_error_unfiltered_cv_closure"])
    err_unf_48 = float(ebn["nxy48_rel_error_unfiltered_cv_closure"])
    unf_48_tag = "exceeds" if err_unf_48 > unf_48_bound else "within"
    u_unf_48 = float(rn_48["u_stored_unfiltered_cv_mj"])
    e_rad_unf_48 = float(rn_48["e_rad_unfiltered_cv_mj"])
    w_unf_48 = float(rn_48["w_beam_unfiltered_cv_mj"])
    u_over_w_48 = float(rn_48["u_stored_over_w_beam_passband"])
    err_bp_48 = float(rn_48["rel_error_total_accounted_vs_bunch_loss"])
    w_ref_bp = float(ebn["w_ref_passband_cv_mj"])
    u_over_w = float(ebn["u_stored_over_w_beam_passband"])
    print(
        f"  N = {int(n_key)} (L = {ebn['active_length_mm']:.1f} mm, wake_len = {ebn['wake_length_mm']:.1f} mm): "
        f"CV passband W_beam = {w_cv_bp:.2f} mJ | "
        f"E_up = {ebn['upstream_radiated_poynting_mj']:.2f} mJ, "
        f"E_dn = {ebn['downstream_radiated_poynting_mj']:.2f} mJ, "
        f"U_bp(T_end) = {u_st_fit:.3f} mJ (U_bp/W_beam,bp = {100 * u_over_w:.2f}% < 2%, "
        f"passband ref U_ref,bp(T_end) = {u_ref_bp:.3e} mJ, W_ref,bp = {w_ref_bp:.3e} mJ [unaffected]) -> "
        f"E_tot = {ebn['total_accounted_em_energy_mj']:.2f} mJ "
        f"(passband CV closure error = {ebn['rel_error_total_accounted_vs_bunch_loss']:.2e}, "
        f"n_xy=64 unfiltered FIT closure error = {err_unf:.2e}; "
        f"coarse n_xy=48 [3 cells/spoke] unexplained late-time unfiltered growth: U_unf = {u_unf_48:.2f} mJ, "
        f"E_rad,unf = {e_rad_unf_48:.2f} mJ, W_beam,unf = {w_unf_48:.2f} mJ, "
        f"unfiltered closure error = {err_unf_48:.2e} [{unf_48_tag} {unf_48_bound:.0e} bound], "
        f"whereas n_xy=48 passband U_bp/W_bp = {100 * u_over_w_48:.2f}% < 2% and passband closure = {err_bp_48:.2e} < 1e-3 [passband unaffected]; "
        f"n_xy=64 unfiltered CV W_beam,unf = {w_cv_unf:.2f} mJ [including a reference-run artifact of W_ref,unf,cv = {w_ref_unf:.2f} mJ, "
        f"net inflow E_rad,ref = {e_rad_ref_unf:.2f} mJ, late-time stored U_ref = {u_ref_unf:.2f} mJ, ref closure = {err_ref_unf:.2e}], "
        f"scattered full-band CV work = {w_cv_full_scat:.2f} mJ; "
        f"active wake loss: passband = {w_act_bp:.2f} mJ vs full-band [0, {fmax_z:.0f} GHz, |F(fmax)|^2={ff_fmax:.2e} vs |F(25GHz)|^2={ff_25:.3f}, "
        f"|F(f)|^2=1e-3 crossing f = {wa['f_form_factor_sq_1e3_ghz']:.2f} GHz] = {w_act_full:.2f} mJ, "
        f"out-of-band active share = {100 * oob_act:.1f}%, CV share = {100 * oob_cv:.1f}%, "
        f"peak on-axis decelerating gradient = {rn_t3['max_decelerating_gradient_mv_per_m']:.2f} MV/m)"
    )

r48_6 = t3["runs_mtm_nxy48"]["6"]
r48_8 = t3["runs_mtm_nxy48"]["8"]
r48_12 = t3["runs_mtm_nxy48"]["12"]
rfwd_48 = t3["forward_control_N8_nxy48"]
rfwd_64 = t3["forward_control_N8"]
r96_8 = t3["runs_mtm_nxy96_N8"]
max_unf_64_96 = max(
    float(t3["runs_mtm"]["6"]["rel_error_unfiltered_cv_closure"]),
    float(t3["runs_mtm"]["8"]["rel_error_unfiltered_cv_closure"]),
    float(t3["runs_mtm"]["12"]["rel_error_unfiltered_cv_closure"]),
    float(r96_8["rel_error_unfiltered_cv_closure"]),
    float(rfwd_64["rel_error_unfiltered_cv_closure"]),
)
display(
    Markdown(
        f"**Coarse-mesh (`n_xy = 48`, 3 cells/spoke) unexplained late-time unfiltered energy growth vs clean passband closure (from `data/task_03_wakefield_and_directionality.json`):** "
        f"In `t3[\"runs_mtm_nxy48\"]`, the unfiltered stored energy grows late in the run: "
        f"at $N = 6$, $U_{{\\mathrm{{unf}}}} = {r48_6['u_stored_unfiltered_cv_mj']:.1f}\\;\\mathrm{{mJ}}$ "
        f"(unfiltered closure `{r48_6['rel_error_unfiltered_cv_closure']:.2e}`); "
        f"at $N = 8$, $U_{{\\mathrm{{unf}}}} = {r48_8['u_stored_unfiltered_cv_mj']:.1f}\\;\\mathrm{{mJ}}$ and "
        f"$E_{{\\mathrm{{rad,unf}}}} = {r48_8['e_rad_unfiltered_cv_mj']:.1f}\\;\\mathrm{{mJ}}$ "
        f"(closure `{r48_8['rel_error_unfiltered_cv_closure']:.2e}`); "
        f"at $N = 12$, $U_{{\\mathrm{{unf}}}} = {r48_12['u_stored_unfiltered_cv_mj']*1e-3:.1f}\\;\\mathrm{{J}}$ "
        f"(`{r48_12['u_stored_unfiltered_cv_mj']:.1f}` mJ) and $E_{{\\mathrm{{rad,unf}}}} = {r48_12['e_rad_unfiltered_cv_mj']*1e-3:.1f}\\;\\mathrm{{J}}$ "
        f"(closure `{r48_12['rel_error_unfiltered_cv_closure']:.2f}`, "
        f"$\\approx {r48_12['u_stored_unfiltered_cv_mj']/r48_12['w_beam_unfiltered_cv_mj']:.0f}\\times$ the unfiltered beam work "
        f"$W_{{\\mathrm{{beam,unf}}}} = {r48_12['w_beam_unfiltered_cv_mj']:.1f}\\;\\mathrm{{mJ}}$); "
        f"and in `forward_control_N8_nxy48`, $U_{{\\mathrm{{unf}}}} = {rfwd_48['u_stored_unfiltered_cv_mj']:.0f}\\;\\mathrm{{mJ}}$ "
        f"($E_{{\\mathrm{{rad,unf}}}} = {rfwd_48['e_rad_unfiltered_cv_mj']:.1f}\\;\\mathrm{{mJ}}$, closure `{rfwd_48['rel_error_unfiltered_cv_closure']:.2e}`). "
        f"At `n_xy = 64` and `n_xy = 96`, the same structures close to $\\le {max_unf_64_96:.2e}$ "
        f"(`{t3['runs_mtm']['6']['rel_error_unfiltered_cv_closure']:.2e}`, `{t3['runs_mtm']['8']['rel_error_unfiltered_cv_closure']:.2e}`, "
        f"`{t3['runs_mtm']['12']['rel_error_unfiltered_cv_closure']:.2e}` at `n_xy = 64`; `{r96_8['rel_error_unfiltered_cv_closure']:.2e}` at `n_xy = 96` $N=8$; "
        f"`{rfwd_64['rel_error_unfiltered_cv_closure']:.2e}` for `forward_control_N8`). "
        f"Crucially, the passband-filtered observables on `n_xy = 48` are **unaffected** by this late-time high-frequency growth: "
        f"for $N = 6, 8, 12$, the `n_xy = 48` passband stored-energy fractions `u_stored_over_w_beam_passband` are "
        f"`{100*r48_6['u_stored_over_w_beam_passband']:.2f}%`, `{100*r48_8['u_stored_over_w_beam_passband']:.2f}%`, and "
        f"`{100*r48_12['u_stored_over_w_beam_passband']:.2f}%` (all $< 2\\%$; `{100*rfwd_48['u_stored_over_w_beam_passband']:.2f}%` for `forward_control_N8_nxy48`), "
        f"and the `n_xy = 48` passband control-volume closure errors `rel_error_total_accounted_vs_bunch_loss` are "
        f"`{r48_6['rel_error_total_accounted_vs_bunch_loss']:.2e}`, `{r48_8['rel_error_total_accounted_vs_bunch_loss']:.2e}`, and "
        f"`{r48_12['rel_error_total_accounted_vs_bunch_loss']:.2e}` (all $< 10^{{-3}}$), confirming that $a_W(48)$ and the coarse GCI level are clean."
    )
)

# %% [markdown]
# ![Analytic Backward-Wave Power Formula and Unified Control-Volume Poynting Energy Balance](figures/09_analytic_power_and_energy_balance.png)

# %% [markdown]
# ---
#
# ## Quantitative Summary
#
# Every numerical value in the table below is formatted at render time directly from `data/summary_metrics.json`
# (and the task JSON files `data/task_01_*.json` through `data/task_05_*.json`) and verified automatically by the
# test suite in `tests/test_081_mtm_wakefield_wagon_wheel.py`.

# %%
display(Markdown(render_quantitative_summary_markdown(sm, t2, t3, t4, t5)))

# %% [markdown]
# ---
#
# ## Limits & Failure Modes
#
# Our derivations and 3D conformal FIT simulations rely on explicit physical approximations that break down in the
# following counter-cases (all model metrics below are printed from `data/*.json` in the code cell below):
#
# 1. **Rigid-bunch approximation ($\Delta\gamma / \gamma \ll 1$, constant $u_0 \approx c$) vs low-energy beam
#    deceleration & microbunching:**
#    - *Assumption:* Throughout this tutorial, the $65.0\;\mathrm{MeV}$ (PRL line 49) drive bunch is treated
#      as a rigid current source (`wakis.sources.Beam`) moving at constant velocity $u_0 = \beta c$. For a
#      $q = 45.0\;\mathrm{nC}$ (PRL line 49) ultrarelativistic bunch over $L = 80.0\;\mathrm{mm}$ (PRL lines 134, 166),
#      the fractional kinetic energy change is only a few percent, keeping the change in relativistic speed
#      $\Delta\beta \approx (\Delta\gamma / \gamma) / \gamma^2 \ll 10^{-5}$ completely negligible.
#    - *Counter-case (when the assumption fails):* If the same $q = 45.0\;\mathrm{nC}$ (PRL line 49) bunch is injected at low
#      kinetic energy ($E_k \lesssim 3\;\mathrm{MeV}$, $\gamma \lesssim 7$) or propagated through a meter-scale
#      decelerator ($L \gtrsim 1\;\mathrm{m}$), the tail of the bunch loses a large fraction of its kinetic energy
#      ($\Delta\gamma / \gamma \sim 1$), slows down below the synchronous phase velocity $v_p = \omega_s / k_z$,
#      slips out of phase with the $11.42\;\mathrm{GHz}$ (PRL lines 137, 149) wave, and undergoes longitudinal microbunching or radial
#      defocusing into the $2 r_b = 6.0\;\mathrm{mm}$ (PRL line 122) hub aperture. That regime requires self-consistent
#      relativistic particle-in-cell tracking rather than a rigid-bunch wake solver.
# 2. **Lossless PEC conductor model ($\alpha_c L \ll 1$) vs finite-conductivity ohmic attenuation & RF surface
#    fields:**
#    - *Assumption:* In the AWA experiment, the wagon-wheel plates were fabricated from stainless steel and the spacer
#      plates from copper (Lu *et al.*, 2019, PRL lines 163–165), with a cold-test unloaded quality factor of
#      $Q_0 = 1050$ (X. Lu, MIT PhD thesis, 2019, Table 2.4, line 2173). From `analytic_ohmic_pec_justification`,
#      the ohmic field decay time constant $\tau_0 = 2 Q_0 / \omega_s$ is much longer than the pulse drain time $t_p$
#      (ratios printed in Part 1), so the mean one-way ohmic power attenuation across the structure is only a few percent,
#      justifying our lossless PEC boundary model ($\sigma \to \infty$).
#    - *Counter-case (when the assumption fails):* If the structure is scaled to sub-millimeter terahertz dimensions
#      (where surface resistance grows as $\sqrt{f}$) or designed with an extremely small group velocity, the conductor
#      attenuation length $\alpha_c^{-1} = 2 Q_0 |v_g| / \omega_s = |v_g| \tau_0$ becomes shorter than $L$, causing
#      output power to saturate below Eq. (2) (PRL lines 240–245) and converting a substantial fraction of the wake
#      energy into wall heating. In addition, Lu *et al.* (2019, PRL line 267) note from CST simulations that at
#      $80.0\;\mathrm{MW}$ (PRL lines 51, 77, 298, 338) output power the peak surface electric field reaches $130\;\mathrm{MV/m}$
#      (PRL line 267), while the single-bunch peak on-axis decelerating gradient `max_decelerating_gradient_mv_per_m`
#      in our 3D FIT runs is printed from JSON above.
# 3. **Single-mode passband formula Eq. (2) vs substantial out-of-band radiation for $\sigma_z = 1.2\;\mathrm{mm}$:**
#    - *Reality in the $\sigma_z = 1.2\;\mathrm{mm}$ (PRL line 151) regime:* For $\sigma_z = 1.2\;\mathrm{mm}$ (PRL line 151),
#      the Gaussian bunch form factor at $f_s = 11.42\;\mathrm{GHz}$ (PRL lines 137, 149) is $\Phi = 0.96$ (PRL line 248),
#      but frequencies above the fundamental passband are **not** negligible: at $25\;\mathrm{GHz}$, the form factor
#      squared `form_factor_sq_at_25ghz` in `data/task_05_analytic_power_and_energy_balance.json` is still large,
#      and $|F(f)|^2 = \exp(-(2\pi f \sigma_z / u_0)^2)$ crosses $10^{-3}$ only at the frequency `f_form_factor_sq_1e3_ghz`
#      computed in code and printed in the cell below (with `form_factor_sq_at_fmax` at `fmax_impedance_ghz` also printed below).
#      Consequently, the out-of-band share of the bunch energy loss (`out_of_band_wake_loss_fraction_active` and
#      `out_of_band_beam_work_fraction_cv`, printed per $N$ in the Part 6 code cell above) accounts for a substantial
#      fraction of the total broadband wake loss, and the single-mode analytical formula Eq. (2) (PRL lines 240–245)
#      describes **only the fundamental passband share** of the wakefield radiation.
#    - *Counter-case (ultrashort bunches):* For sub-millimeter bunches ($\sigma_z \ll 1.0\;\mathrm{mm}$, PRL line 112),
#      $\Phi(\omega) \approx 1$ across hundreds of gigahertz, so broadband multi-mode radiation outside the fundamental
#      passband dominates over the synchronous mode.
# 4. **Sub-millimeter coaxial inner-shield discretization ($t_{\mathrm{shield}} = 0.3\;\mathrm{mm}$ across 3 mesh levels)
#    as the likely source of oscillatory mesh behavior:**
#    - *Limitation & 3-level mesh study:* Our axially symmetric coaxial TEM extraction coupler (`ASSUMED`, replacing the
#      side-wall WR-90 waveguides of PRL lines 134–138) uses a thin cylindrical PEC inner shield extending from
#      $r_b = 3.0\;\mathrm{mm}$ (PRL line 122) to $r_{\mathrm{coax}} = 3.30\;\mathrm{mm}$ (`ASSUMED`, radial thickness
#      $t_{\mathrm{shield}} = 0.3\;\mathrm{mm}$). Because $0.3\;\mathrm{mm}$ (`ASSUMED`) is sub-millimeter, it spans fewer
#      than 4 Cartesian cells across our 3-level transverse mesh hierarchy (`cells_across_0p3mm_coax_shield` printed below).
#      Across $n_{xy} \in \{48, 64, 96\}$, the upstream port reflection $|\Gamma(f_{\mathrm{eval}})|$
#      (`port_gamma_tapered_at_sync_values` printed below) exhibits oscillatory mesh variation with $|R| > 1$
#      (oscillatory, $|R| > 1$ (amplitude not decreasing), from `R_ratio`) rather than remaining invariant,
#      making the under-resolved $0.3\;\mathrm{mm}$ (`ASSUMED`) shield wall the likely source of the oscillatory convergence
#      observed for $W_{\mathrm{beam,bp,cv}}$ and $E_{\mathrm{up,coax}}$ (both oscillatory, $|R| > 1$ (amplitude not decreasing),
#      from `R_ratio`) as well as $P_{\mathrm{up,max}}$ and $E_{\mathrm{up}}$ (oscillatory with $|R| < 1$). Widening the coaxial inner shield
#      outer radius to $r_{\mathrm{coax}} = r_{\mathrm{hub}} = 4.0\;\mathrm{mm}$ (`ASSUMED`, matching the $1.0\;\mathrm{mm}$
#      hub/spoke wall thickness of PRL Fig. 2(b) line 119 so that every metallic feature is $\ge 1.0\;\mathrm{mm}$ thick)
#      is an important future improvement.
# 5. **Affine loss slope $a_W$ evaluated from two mesh levels only ($N = 6, 12$ not run at $n_{xy} = 96$):**
#    - *Limitation:* While the $N = 8$ wakefield observables are evaluated on all three transverse mesh levels
#      $n_{xy} \in \{48, 64, 96\}$, the $N = 6$ and $N = 12$ length-scaling runs are evaluated on two transverse mesh levels
#      ($n_{xy} \in \{48, 64\}$) only. Consequently, the discretization uncertainty of the affine passband loss slope
#      $a_W = \mathrm{d}W_{\mathrm{beam,bp,cv}} / \mathrm{d}L$ (`gci_a_w_two_level_rel` and `a_w_uncertainty_rel` printed below)
#       comes from a conservative two-level Roache ($F_s = 3, p = 1$) GCI between $n_{xy} = 48$ and $n_{xy} = 64$ rather than
#      a three-level $N \in \{6, 8, 12\}$ sweep at $n_{xy} = 96$.
# 6. **Unexplained late-time unfiltered energy growth on the coarse (`3 cells/spoke`) mesh & unexplained smooth-pipe
#    reference-run artifact:**
#    - *Limitation:* On the coarse $n_{xy} = 48$ ($3\;\mathrm{cells/spoke}$) mesh (`t3["runs_mtm_nxy48"]` across
#      $N \in \{6, 8, 12\}$ and `t3["forward_control_N8_nxy48"]`), the unfiltered stored energy `u_stored_unfiltered_cv_mj`
#      exhibits **unexplained late-time unfiltered energy growth** (accompanied at $N = 8, 12$ and in `forward_control_N8_nxy48`
#      by large negative `e_rad_unfiltered_cv_mj`, and at $N = 12$ reaching more than three orders of magnitude above the beam
#      work `w_beam_unfiltered_cv_mj`), whereas at $n_{xy} = 64$ ($4\;\mathrm{cells/spoke}$) and $n_{xy} = 96$
#      ($6\;\mathrm{cells/spoke}$) the same structures close to `rel_error_unfiltered_cv_closure` (all per-$N$ numbers and
#      their maximum formatted from JSON below). Distinct from this coarse-mesh late-time growth, the empty $r_b = 3.0\;\mathrm{mm}$ (PRL line 122) smooth-pipe
#      reference run (`solver_ref`) exhibits a separate unexplained late-time unfiltered electromagnetic artifact
#      (`u_stored_ref_unfiltered_cv_mj` and net inflow `e_rad_ref_unfiltered_cv_mj` at $N = 6, 8, 12$, printed below), and all
#      unfiltered or full-band reference-subtracted control-volume numbers carry this reference-pipe artifact. By contrast,
#      the passband-filtered quantities are **unaffected** by both artifacts: in the smooth-pipe reference run,
#      `u_stored_ref_passband_cv_mj` vanishes to five orders of magnitude, and in every $N \in \{6, 8, 12\}$ coarse $n_{xy} = 48$
#      run, both `u_stored_over_w_beam_passband` ($< 0.02$) and the passband closure `rel_error_total_accounted_vs_bunch_loss`
#      ($< 10^{-3}$, printed per $N$ from JSON below) remain clean, so $a_W(48)$ and the coarse GCI level are unaffected.

# %%
g_port_gci = mc_gci["port_gamma_tapered_at_sync"]
print(
    f"Limits & Failure Modes quantitative diagnostics (from data/*.json):\n"
    f"  3. Form factor |F(f)|^2 = 1e-3 crossing frequency : {wa['f_form_factor_sq_1e3_ghz']:.2f} GHz "
    f"(|F(25 GHz)|^2 = {eb['form_factor_sq_at_25ghz']:.3f}, |F({eb['fmax_impedance_ghz']:.0f} GHz)|^2 = {eb['form_factor_sq_at_fmax']:.2e})\n"
    f"  4. 0.3 mm ASSUMED coax shield cells (n_xy=48..96) : {mc['cells_across_0p3mm_coax_shield']} cells; "
    f"port |Gamma(f_s)| = {[round(x, 4) for x in g_port_gci['values']]} "
    f"({'oscillatory, |R| > 1 (amplitude not decreasing)' if abs(g_port_gci['R_ratio']) > 1.0 else g_port_gci['convergence_class']}, "
    f"R = {g_port_gci['R_ratio']:.3f}, rel change 64->96 = {100 * abs(g_port_gci['values'][2] - g_port_gci['values'][1]) / g_port_gci['values'][2]:.2f}%, "
    f"U_fine = {100 * g_port_gci['u_fine_rel']:.2f}%)\n"
    f"  5. Two-level slope a_W (N=6,8,12 at n_xy=48,64)   : a_W(48) = {ls48['affine_a_w_j_per_m']:.4f} J/m, "
    f"a_W(64) = {ls['affine_a_w_j_per_m']:.4f} J/m, two-level GCI (Fs=3, p=1) = {100 * ls['gci_a_w_two_level_rel']:.2f}%, "
    f"total a_W uncertainty = +/-{100 * ls['a_w_uncertainty_rel']:.2f}%\n"
    f"  6a. Coarse n_xy=48 (3 cells/spoke) unexplained late-time unfiltered growth (N = 6, 8, 12 & fwd control):\n"
    f"      N=6  : U_unf = {r48_6['u_stored_unfiltered_cv_mj']:.1f} mJ, E_rad,unf = {r48_6['e_rad_unfiltered_cv_mj']:.1f} mJ, closure = {r48_6['rel_error_unfiltered_cv_closure']:.2e} | passband U_bp/W_bp = {100*r48_6['u_stored_over_w_beam_passband']:.2f}% (<2%), passband closure = {r48_6['rel_error_total_accounted_vs_bunch_loss']:.2e} (<1e-3)\n"
    f"      N=8  : U_unf = {r48_8['u_stored_unfiltered_cv_mj']:.1f} mJ, E_rad,unf = {r48_8['e_rad_unfiltered_cv_mj']:.1f} mJ, closure = {r48_8['rel_error_unfiltered_cv_closure']:.2e} | passband U_bp/W_bp = {100*r48_8['u_stored_over_w_beam_passband']:.2f}% (<2%), passband closure = {r48_8['rel_error_total_accounted_vs_bunch_loss']:.2e} (<1e-3)\n"
    f"      N=12 : U_unf = {r48_12['u_stored_unfiltered_cv_mj']*1e-3:.1f} J ({r48_12['u_stored_unfiltered_cv_mj']:.1f} mJ), E_rad,unf = {r48_12['e_rad_unfiltered_cv_mj']*1e-3:.1f} J ({r48_12['e_rad_unfiltered_cv_mj']:.1f} mJ), closure = {r48_12['rel_error_unfiltered_cv_closure']:.2f} (~{r48_12['u_stored_unfiltered_cv_mj']/r48_12['w_beam_unfiltered_cv_mj']:.0f}x W_beam,unf = {r48_12['w_beam_unfiltered_cv_mj']:.1f} mJ) | passband U_bp/W_bp = {100*r48_12['u_stored_over_w_beam_passband']:.2f}% (<2%), passband closure = {r48_12['rel_error_total_accounted_vs_bunch_loss']:.2e} (<1e-3)\n"
    f"      forward_control_N8_nxy48: U_unf = {rfwd_48['u_stored_unfiltered_cv_mj']:.1f} mJ, E_rad,unf = {rfwd_48['e_rad_unfiltered_cv_mj']:.1f} mJ, closure = {rfwd_48['rel_error_unfiltered_cv_closure']:.2e} | passband U_bp/W_bp = {100*rfwd_48['u_stored_over_w_beam_passband']:.2f}% (<2%)\n"
    f"      n_xy=64 & 96 unfiltered closures: N=6,8,12 (n_xy=64) = {[round(t3['runs_mtm'][str(n)]['rel_error_unfiltered_cv_closure'], 7) for n in t3['n_periods_list']]}, N=8 (n_xy=96) = {r96_8['rel_error_unfiltered_cv_closure']:.2e}, forward_control_N8 (n_xy=64) = {rfwd_64['rel_error_unfiltered_cv_closure']:.2e} (all <= {max_unf_64_96:.2e})\n"
    f"  6b. Smooth-pipe reference artifact (N = 6, 8, 12) : "
    f"U_ref,unf(T_end) = {[round(eb_all[str(n)]['u_stored_ref_unfiltered_cv_mj'], 2) for n in t3['n_periods_list']]} mJ, "
    f"E_rad,ref,unf = {[round(eb_all[str(n)]['e_rad_ref_unfiltered_cv_mj'], 2) for n in t3['n_periods_list']]} mJ, "
    f"passband U_ref,bp(T_end) = {[round(eb_all[str(n)]['u_stored_ref_passband_cv_mj'], 8) for n in t3['n_periods_list']]} mJ"
)
display(
    Markdown(
        f"*Formatted from `data/task_03_wakefield_and_directionality.json`:* Across `n_xy = 64` ($N \\in \\{{6, 8, 12\\}}$ and `forward_control_N8`) "
        f"and `n_xy = 96` ($N = 8$), the unfiltered control-volume closure `rel_error_unfiltered_cv_closure` is "
        f"$\\le {max_unf_64_96:.2e}$ (`{t3['runs_mtm']['6']['rel_error_unfiltered_cv_closure']:.2e}`, "
        f"`{t3['runs_mtm']['8']['rel_error_unfiltered_cv_closure']:.2e}`, and "
        f"`{t3['runs_mtm']['12']['rel_error_unfiltered_cv_closure']:.2e}` for $N = 6, 8, 12$ at `n_xy = 64`; "
        f"`{r96_8['rel_error_unfiltered_cv_closure']:.2e}` at `n_xy = 96`; and "
        f"`{rfwd_64['rel_error_unfiltered_cv_closure']:.2e}` for `forward_control_N8`)."
    )
)

# %% [markdown]
# ---
#
# ## References
#
# 1. **X. Lu, M. A. Shapiro, I. Mastovsky, R. J. Temkin, M. Conde, J. G. Power, J. Shao, E. E. Wisniewski, and C. Jing**, "Generation of High-Power, Reversed-Cherenkov Wakefield Radiation in a Metamaterial Structure," *Physical Review Letters* **122**(1), 014801 (2019), [`doi:10.1103/PhysRevLett.122.014801`](https://doi.org/10.1103/PhysRevLett.122.014801) — Experimental demonstration of $25.0\;\mathrm{MW}$ (PRL lines 50, 232, 291) single-bunch and $80.0\;\mathrm{MW}$ (PRL lines 51, 77, 298, 338) two-bunch reversed-Cherenkov wakefield power at $11.42\;\mathrm{GHz}$ (PRL lines 137, 149) in an all-metal wagon-wheel metamaterial structure driven by $65.0\;\mathrm{MeV}$ (PRL line 49) bunches at the AWA facility.
# 2. **X. Lu**, *Metamaterial Structures for Wakefield Acceleration and High-Power Microwave Generation*, Ph.D. Thesis, Department of Physics, Massachusetts Institute of Technology (2019) — Detailed cold-test tables (Tables 2.2 and 2.4, $Q_0 = 1050$ at line 2173), shunt impedance definition (Eq. 2.18 at lines 2128–2135), and engineering drawings ($R_{\mathrm{wg}} = 8.1\;\mathrm{mm}$ at lines 2075, 2208) for the X-band wagon-wheel metamaterial structure.
# 3. **X. Lu, M. A. Shapiro, and R. J. Temkin**, "Modeling of the interaction of a volumetric metallic metamaterial structure with a relativistic electron beam," *Physical Review Special Topics - Accelerators and Beams* **18**(8), 081303 (2015), [`doi:10.1103/PhysRevSTAB.18.081303`](https://doi.org/10.1103/PhysRevSTAB.18.081303) — Analytical and numerical modeling of spatial dispersion, effective-medium permittivity tensor, and reversed Cherenkov wakefield radiation in a volumetric cubic metallic metamaterial.
# 4. **J. S. Hummelt, X. Lu, H. Xu, I. Mastovsky, M. A. Shapiro, and R. J. Temkin**, "Coherent Cherenkov-Cyclotron Radiation Excited by an Electron Beam in a Metamaterial Waveguide," *Physical Review Letters* **117**(23), 237701 (2016), [`doi:10.1103/PhysRevLett.117.237701`](https://doi.org/10.1103/PhysRevLett.117.237701) — Demonstration of negative-index backward-wave dispersion and beam-wave interaction in a loaded metamaterial waveguide.
# 5. **V. G. Veselago**, "The electrodynamics of substances with simultaneously negative values of $\varepsilon$ and $\mu$," *Soviet Physics Uspekhi* **10**(4), 509–514 (1968), [`doi:10.1070/PU1968v010n04ABEH003699`](https://doi.org/10.1070/PU1968v010n04ABEH003699) — Foundational electrodynamics of left-handed media with antiparallel phase and group velocities ($\mathbf{v}_g \cdot \mathbf{v}_p < 0$) and reversed Cherenkov radiation.
# 6. **T. Weiland**, "A discretization method for the solution of Maxwell's equations for six-component fields," *Archiv für Elektronik und Übertragungstechnik (AEÜ)* **31**(3), 116–120 (1977) — Original formulation of the Finite Integration Technique (FIT) on dual orthogonal grids.
# 7. **T. P. Wangler**, *RF Linear Accelerators*, 2nd ed. (Wiley-VCH, Weinheim, 2008) — Travelling-wave wakefield power relation cited as Ref. [24] in Lu *et al.* (2019, PRL lines 238, 370–371) for Eq. (2) (lines 240–245).
