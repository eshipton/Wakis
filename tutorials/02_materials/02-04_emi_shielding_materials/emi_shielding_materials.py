# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 2.4 · EMI Shielding and Absorbing Materials
#
# **Chapter 2 · Materials**
#
# Electromagnetic interference (EMI) control in RF/microwave enclosures, printed
# circuit boards, and cable harnesses relies on two distinct physical mechanisms:
# **reflective shielding** (bouncing incident energy back via a large wave-impedance
# mismatch $Z \ll Z_0$ or $Z \gg Z_0$) and **absorptive suppression** (converting
# electromagnetic energy into heat inside the slab via magnetic $\mu''(\omega)$
# and/or dielectric/conductive $\varepsilon''(\omega)$ losses while keeping the
# wave impedance $|Z| = Z_0|\sqrt{\mu/\varepsilon}|$ closer to free space).
#
# In this tutorial we build, validate, and compare **six commercial EMI shielding
# and absorbing material classes** registered in `wakis.material_library` across
# $10\;\text{MHz}\text{–}6\;\text{GHz}$:
#
# 1. **Magnetic-particle-loaded silicone (`magnetic particle silicone`):** manufacturer
#    datasheet of a commercially available magnetic-particle-loaded silicone
#    absorber (digitized curves in `data/`) [1] — an electrically insulating
#    ($\rho_v = 7.45 \times 10^{11}\;\Omega\cdot\text{cm}$) elastomer whose
#    complex permeability $\mu'(f) - i\mu''(f)$ is digitized from $10\;\text{MHz}$
#    to $1\;\text{GHz}$, fitted with passive magnetic Debye + Lorentz poles,
#    combined with an inferred permittivity $\varepsilon'$ over the $0.1\text{–}1.0\;\text{GHz}$
#    overlap band, and extrapolated to $1\text{–}6\;\text{GHz}$ to predict the
#    datasheet attenuation curve ($\approx 27.8\;\text{dB/cm}$ at $1\;\text{GHz}$
#    to $\approx 81.8\;\text{dB/cm}$ at $6\;\text{GHz}$).
# 2. **Magnetic-flake-filled noise-suppression sheet (`sendust flake sheet`):**
#    aligned $\text{Fe}_{85}\text{Si}_{9.6}\text{Al}_{5.4}$ (Sendust) flaky-filler
#    polymer composite (**Yoshida *et al.*, 1999** [2]; **Whalen *et al.*, 2025** [3]),
#    where planar flake anisotropy pushes $\mu_s \approx 70$ above the spherical
#    Snoek limit [4] while inter-flake capacitance yields $\varepsilon' \approx 116$.
# 3. **High-permittivity low-loss ceramic (`tio2 ceramic`):** sintered
#    polycrystalline rutile $\text{TiO}_2$ microwave resonator ceramic with
#    $\varepsilon_r \approx 100$ and $\tan\delta \approx 2.1 \times 10^{-4}$ at
#    $3\;\text{GHz}$ (**Templeton *et al.*, 2000** [5]; **Tobar *et al.*, 1998** [6]).
# 4. **Carbon-nanotube loaded silicone rubber (`cnt silicone`):** $3\;\text{wt}\%$
#    MWCNT/PDMS nanocomposite elastomer above percolation ($\sigma_{\text{dc}} = 25\;\text{S/m}$,
#    $\varepsilon_\infty = 14$, interfacial Debye relaxation $\Delta\varepsilon = 8$;
#    **Al-Saleh & Sundararaj, 2009** [7]; **Ananthasubramanian *et al.*, 2025** [8];
#    **Galyaltdinov *et al.*, 2022** [9]).
# 5. **Conductive elastomer gasket (`ni-graphite silicone`):** nickel-coated-graphite
#    silicone EMI gasket with volume resistivity $\rho_v = 0.50\;\Omega\cdot\text{cm}
#    \implies \sigma = 200\;\text{S/m}$ (**Ott, 2009** [10]).
# 6. **Thin copper film (`copper film, 50 nm`) and PCB copper foils ($18\;\mu\text{m}$, $35\;\mu\text{m}$):**
#    annealed copper ($\sigma_{\text{Cu}} = 5.8 \times 10^7\;\text{S/m}$, library key
#    `'copper foil'`) for the classical **Schelkunoff (1943)** shielding effectiveness
#    decomposition [11]. The sub-skin-depth **copper film, 50 nm** ($t \ll \delta \ll \Delta z$)
#    is simulated in 3D FIT via Wakis's subcell zero-thickness sheet model
#    (`solver.add_sheet`), and then compared with real PCB copper foils ($18\;\mu\text{m}$
#    and $35\;\mu\text{m}$, standard $0.5\;\text{oz/ft}^2$ and $1\;\text{oz/ft}^2$ weights)
#    computed with both the thin-sheet model and the full Schelkunoff absorption term
#    ($t \gg \delta$: $\text{SE}_A = 8.686\,t/\delta$) [10, 11].
#
# > **Time-harmonic sign convention (`exp(+i*omega*t)`):** Throughout Wakis,
# > monochromatic fields vary as $\operatorname{Re}[\hat{\mathbf{E}}\,e^{+i\omega t}]$.
# > Passive media satisfy $\operatorname{Im}\varepsilon(\omega) \le 0$ and
# > $\operatorname{Im}\mu(\omega) \le 0$, and the forward-propagating refractive
# > index $n(\omega) = \sqrt{\varepsilon(\omega)\mu(\omega)} = n'(\omega) - i\kappa(\omega)$
# > has $n'(\omega) \ge 0$ and $\kappa(\omega) = -\operatorname{Im}n(\omega) \ge 0$.

# %%
# --- Colab setup (runs only on Google Colab) ---
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running on Google Colab — installing dependencies...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "xvfb", "libgl1-mesa-glx"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/eshipton/Wakis.git@main",
            "pyvista",
            "scikit-rf",
            "tqdm",
        ],
        check=True,
    )
    print("Colab setup complete.")

# %% [markdown]
# ## Learning Goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Fit passive magnetic poles and infer permittivity from datasheet curves:**
#    fit digitized $\mu'(f), \mu''(f)$ of a commercially available magnetic-particle-loaded
#    silicone absorber ($10\;\text{MHz}\text{–}1\;\text{GHz}$)
#    with passive Debye and Lorentz magnetic poles (`fit_mu_poles`, `is_stable`),
#    infer the unstated relative permittivity $\varepsilon'$ over the $0.1\text{–}1.0\;\text{GHz}$
#    overlap band from $\alpha(f) = \frac{20}{\ln 10}\frac{\omega}{c_0}(-\operatorname{Im}\sqrt{\varepsilon\mu})\times 10^{-2}$,
#    quantify its sensitivity to dielectric loss $\tan\delta_e$, and test the
#    $1\text{–}6\;\text{GHz}$ extrapolation against the datasheet attenuation curve.
# 2. **Simulate all six EMI material classes in 3D FIT (`SolverFIT3D`):**
#    validate normal-incidence reflectance $|R(f)|^2$, transmittance $|T(f)|^2$,
#    and absorbed fraction $A(f) = 1 - |R(f)|^2 - |T(f)|^2$ across multiple slab
#    thicknesses against the exact transfer matrix using locally refined
#    non-uniform $z$-meshes ($\ge 20\;\text{cells}/\lambda_{\text{mat}}$,
#    $\ge 4\;\text{cells}$ across the slab, $\ge 4\;\text{cells}$ per skin depth
#    $\delta$) and `solver.add_sheet` for subcell metallic films.
# 3. **Decompose shielding effectiveness with Schelkunoff's theorem:**
#    write $\text{SE}(f, d) = -10\log_{10}|T(f, d)|^2 = \text{SE}_R(f) + \text{SE}_A(f, d) + \text{SE}_M(f, d)$
#    and explain why conductors and percolated CNT rubbers shield primarily by
#    front-face reflection ($\text{SE}_R$), why magnetic elastomers attenuate
#    waves inside the slab ($\text{SE}_A$), and why an $\varepsilon_r = 100$
#    low-loss ceramic reflects strongly without absorbing.
# 4. **Design metal-backed quarter-wave absorbers and evaluate Rozanov's bound:**
#    simulate PEC-backed sheets (`bc_high = 'pec'`) of magnetic-particle silicone and
#    *Sendust* flake composite, trace the quarter-wave matching dip
#    $d_{\lambda/4}(f) = \lambda_0 / (4\operatorname{Re}\sqrt{\varepsilon\mu})$,
#    and verify the **Rozanov (2000)** thickness–bandwidth causality bound
#    $\left|\int_0^\infty \ln|\Gamma(\lambda)|\,d\lambda\right| \le 2\pi^2 \mu_s d$ [12].
#
# ---
#
# ## 1. Physics of EMI Shielding and Absorbing Materials
#
# ### 1.1 Six commercial EMI material classes in `wakis.material_library`
#
# In a general linear isotropic medium with dispersive permittivity
# $\varepsilon(\omega) = \varepsilon_\infty + \sum_p \chi_{e,p}(\omega) + \frac{\sigma}{i\omega\varepsilon_0}$
# and dispersive permeability
# $\mu(\omega) = \mu_\infty + \sum_q \chi_{m,q}(\omega)$, a plane wave
# $\hat{E}_x(z, \omega) = E_0 e^{-i k_0 n(\omega) z}$ propagates with complex
# refractive index $n(\omega)$ and normalized wave impedance $z_{\text{rel}}(\omega)$:
#
# $$n(\omega) = \sqrt{\varepsilon(\omega)\mu(\omega)} = n'(\omega) - i\kappa(\omega), \qquad
# z_{\text{rel}}(\omega) = \frac{Z(\omega)}{Z_0} = \sqrt{\frac{\mu(\omega)}{\varepsilon(\omega)}},$$
#
# with $n'(\omega) \ge 0$, $\kappa(\omega) = -\operatorname{Im}n(\omega) \ge 0$,
# and $\operatorname{Re}z_{\text{rel}}(\omega) \ge 0$. The power/amplitude
# attenuation rate per unit length in $\text{dB/cm}$ is
#
# $$\alpha(f)\;[\text{dB/cm}] = \frac{20}{\ln 10}\,\alpha_{\text{Np}}(f)\times 10^{-2}
# = 8.6859\,\frac{2\pi f}{c_0}\left(-\operatorname{Im}\sqrt{\varepsilon(f)\mu(f)}\right)\times 10^{-2}.$$
#
# | Class | Library key | Representative commercial / literature material | Constitutive parameters ($10\;\text{MHz}\text{–}6\;\text{GHz}$) | Primary EMI mechanism | Verified source |
# | :---: | :--- | :--- | :--- | :--- | :--- |
# | **1** | `'magnetic particle silicone'` | Commercially available magnetic-particle-loaded silicone elastomer ($\rho_v = 7.45\times 10^{11}\;\Omega\cdot\text{cm}$) | $\mu_s = 36.8$ (1 Debye + 1 Lorentz fit), inferred $\varepsilon' \approx 81.1$ ($\tan\delta_e \approx 0.01$), $\sigma = 0$ | Broadband magnetic absorption ($\alpha = 27.8\text{–}81.8\;\text{dB/cm}$) | Manufacturer datasheet of a commercially available magnetic-particle-loaded silicone absorber (digitized curves in `data/`) [1] |
# | **2** | `'sendust flake sheet'` | Aligned $\text{Fe}_{85}\text{Si}_{9.6}\text{Al}_{5.4}$ (Sendust) flake-filled polymer sheet | $\mu_s = 70.0$ (Debye DW + Lorentz FMR), $\varepsilon' \approx 116$ (inter-flake capacitance), $\sigma = 0$ | Sub-GHz to GHz magnetic noise suppression + $\lambda/4$ PEC absorber | Yoshida *et al.* (1999) [2]; Whalen *et al.* (2025) [3] |
# | **3** | `'tio2 ceramic'` | Sintered polycrystalline rutile $\text{TiO}_2$ resonator ceramic | $\varepsilon' = 100.0$, $\tan\delta = 2.1\times 10^{-4}$ at $3\;\text{GHz}$ ($Q\times f \approx 14\,000\;\text{GHz}$), $\mu = 1$ | Pure reactive dielectric reflection (Fabry–Pérot etalon, $A \approx 0$) | Templeton *et al.* (2000) [5]; Tobar *et al.* (1998) [6] |
# | **4** | `'cnt silicone'` | $3\;\text{wt}\%$ MWCNT/PDMS silicone rubber (above percolation) | $\sigma = 25\;\text{S/m}$, $\varepsilon_\infty = 14.0$, interfacial Debye $\Delta\varepsilon = 8.0$, $\mu = 1$ | Mixed conductive reflection ($\text{SE}_R$) + skin-depth absorption ($\text{SE}_A$) | Al-Saleh & Sundararaj (2009) [7]; Ananthasubramanian *et al.* (2025) [8] |
# | **5** | `'ni-graphite silicone'` | Nickel-graphite filled silicone EMI gasket ($\rho_v = 0.50\;\Omega\cdot\text{cm}$) | $\sigma = 200\;\text{S/m}$, $\varepsilon_\infty = 12.0$, $\mu = 1$ ($\delta(4\;\text{GHz}) = 0.56\;\text{mm}$) | Reflection-dominated enclosure gasket + multi-skin-depth $\text{SE}_A$ | Ott (2009) [10]; Paul (2006) [13] |
# | **6** | `'copper foil'` | Subcell **copper film, 50 nm** (`add_sheet`, $d < \Delta z$) & PCB copper foils ($18\;\mu\text{m}$, $35\;\mu\text{m}$) | $\sigma_{\text{Cu}} = 5.8\times 10^7\;\text{S/m}$, $\mu = 1$ ($\delta(1\;\text{GHz}) = 2.09\;\mu\text{m}$) | Classical Schelkunoff reflective + skin-depth shield ($\text{SE}_A = 8.686\,t/\delta$ for $t \gg \delta$) | Schelkunoff (1943) [11]; Ott (2009) [10] |
#
# ### 1.2 Magnetic-particle silicone: measured $\mu(f)$, inferred $\varepsilon'$, and $1\text{–}6\;\text{GHz}$ extrapolation
#
# The manufacturer datasheet of a commercially available magnetic-particle-loaded
# silicone absorber (digitized curves in `data/`) [1]
# provides two plots and a physical-property table:
#
# * **Measured on the datasheet:**
#   1. Complex permeability $\mu'(f), \mu''(f)$ from $10\;\text{MHz}$ to $1\;\text{GHz}$
#      (digitized in `data/magnetic_silicone_permeability.csv`: $\mu' / \mu'' \approx 36.5 / 2.5$
#      at $10\;\text{MHz}$, $25.7 / 9.9$ at $100\;\text{MHz}$, and $7.4 / 12.9$ at $1\;\text{GHz}$).
#   2. Plane-wave attenuation $\alpha(f)$ in $\text{dB/cm}$ from $0.1\;\text{GHz}$ to $6.0\;\text{GHz}$
#      (digitized in `data/magnetic_silicone_attenuation.csv`: $\alpha \approx 2.8\;\text{dB/cm}$
#      at $0.1\;\text{GHz}$, $27.8\;\text{dB/cm}$ at $1\;\text{GHz}$, and $81.8\;\text{dB/cm}$ at $6\;\text{GHz}$).
#   3. Volume resistivity $\rho_v = 7.45 \times 10^{11}\;\Omega\cdot\text{cm}$
#      ($\sigma_{\text{dc}} = 1.34 \times 10^{-10}\;\text{S/m}$, so ohmic conduction loss is negligible).
# * **Inferred over the $0.1\text{–}1.0\;\text{GHz}$ overlap band:**
#   The datasheet does not state the relative permittivity $\varepsilon(f)$.
#   Over $0.1\text{–}1.0\;\text{GHz}$, where *both* $\mu(f)$ and $\alpha(f)$ are
#   measured, we invert $\alpha(f) = 8.6859\,(\omega/c_0)(-\operatorname{Im}\sqrt{\varepsilon\mu})\times 10^{-2}$
#   for $\varepsilon'$ under a small assumed silicone-matrix dielectric loss tangent
#   $\tan\delta_e = \varepsilon''/\varepsilon' \in [0.00, 0.05]$. This yields
#   $\varepsilon' = 82.7$ (for $\tan\delta_e = 0$), $\varepsilon' = 81.1$
#   (for $\tan\delta_e = 0.01$), and $\varepsilon' = 74.9$ (for $\tan\delta_e = 0.05$).
#   No independent measurement of $\varepsilon'$ for this magnetic-particle-loaded
#   silicone absorber is available in the cited manufacturer datasheet [1], though
#   the inferred $\varepsilon' \approx 75\text{–}83$ is of the same order of
#   magnitude as the $\varepsilon' \approx 116$ reported for the cited Sendust
#   magnetic-flake polymer composite [2, 3].
# * **Extrapolated from $1\;\text{GHz}$ to $6\;\text{GHz}$:**
#   Fitting $\mu(f)$ over $10\;\text{MHz}\text{–}1\;\text{GHz}$ with **1 Debye
#   (domain-wall relaxation) + 1 Lorentz (gyromagnetic spin resonance)** pole
#   achieves a $0.36\%$ relative RMS error on $\mu(f)$ and reproduces the
#   $6\;\text{GHz}$ endpoint ($\alpha_{\text{pred}}(6\;\text{GHz}) = 84.7\;\text{dB/cm}$
#   vs. datasheet $81.8\;\text{dB/cm}$), with an out-of-band Lorentz resonance
#   bulge around $1.8\text{–}4.5\;\text{GHz}$ ($\text{RMS} = 13.16\;\text{dB/cm}$)
#   because $f_0 = 1.84\;\text{GHz}$ lies above the $1\;\text{GHz}$ permeability
#   measurement window. Conversely, the overdamped **2-Debye** relaxation fit
#   ($4.53\%$ error on $\mu$) has no out-of-band resonance peak and tracks the
#   $1\text{–}6\;\text{GHz}$ datasheet attenuation curve to $\text{RMS} = 3.42\;\text{dB/cm}$.
#
# ### 1.3 Normal-incidence transfer matrix and Schelkunoff shielding decomposition
#
# For a planar slab of thickness $d$, refractive index $n(\omega)$, and
# normalized wave impedance $z_{\text{rel}}(\omega) = \sqrt{\mu(\omega)/\varepsilon(\omega)}$
# immersed in free space ($z_0 = 1$), the single-interface Fresnel reflection
# coefficient from vacuum into the slab is $r_{12} = (z_{\text{rel}} - 1)/(z_{\text{rel}} + 1)$
# and the single-pass propagation factor is $P = e^{-i(\omega/c_0) n d}$. Summing
# the Airy multiple-reflection series gives the exact complex reflection and
# transmission amplitudes:
#
# $$r(\omega, d) = \frac{r_{12}\left(1 - P^2\right)}{1 - r_{12}^2 P^2}, \qquad
# t(\omega, d) = \frac{\left(1 - r_{12}^2\right) P}{1 - r_{12}^2 P^2}
# = \frac{4 z_{\text{rel}}}{(1 + z_{\text{rel}})^2}\,\frac{P}{1 - r_{12}^2 P^2},$$
#
# with power reflectance $R = |r|^2$, transmittance $T = |t|^2$, and absorbed
# power fraction $A = 1 - R - T$. Taking $-10\log_{10}|T(\omega, d)|^2 = -20\log_{10}|t(\omega, d)|$
# yields **Schelkunoff's exact three-term shielding effectiveness decomposition** [10, 11]:
#
# $$\text{SE}(\omega, d) = -10\log_{10}|T(\omega, d)|^2 = \text{SE}_R(\omega) + \text{SE}_A(\omega, d) + \text{SE}_M(\omega, d),$$
#
# where:
#
# $$\text{SE}_R(\omega) = 20\log_{10}\frac{|1 + z_{\text{rel}}(\omega)|^2}{4|z_{\text{rel}}(\omega)|}, \qquad
# \text{SE}_A(\omega, d) = -20\log_{10}|P| = 8.6859\,\frac{\omega}{c_0}\left(-\operatorname{Im}n(\omega)\right) d,$$
#
# $$\text{SE}_M(\omega, d) = 20\log_{10}\left|1 - r_{12}(\omega)^2 P(\omega, d)^2\right|.$$
#
# This decomposition immediately explains the contrasting behavior of the six classes:
#
# * **Good conductors (`copper film, 50 nm`, PCB copper foils $18\;\mu\text{m}$ / $35\;\mu\text{m}$, `ni-graphite silicone`) and percolated CNT rubber (`cnt silicone`):**
#   For $\sigma \gg \omega\varepsilon_0\varepsilon'$, $\varepsilon(\omega) \approx -i\sigma/(\omega\varepsilon_0)$
#   so $|z_{\text{rel}}| = \sqrt{\omega\mu_0/\sigma} \ll 1$ and the internal
#   wavenumber is $k = (1 - i)/\delta$ with skin depth $\delta = \sqrt{2/(\omega\mu_0\sigma)}$.
#   Notice that the internal wavelength is $\lambda_{\text{mat}} = 2\pi/\operatorname{Re}k = 2\pi\delta \approx 6.283\,\delta$,
#   so resolving the skin depth with $\ge 4\;\text{cells}/\delta$ automatically
#   resolves the internal wavelength with $2\pi \times 4 \approx 25.1 \ge 20\;\text{cells}/\lambda_{\text{mat}}$!
#   The large impedance mismatch $|z_{\text{rel}}| \ll 1$ makes front-face
#   reflection $\text{SE}_R$ dominant for electrically thin films ($t \lesssim \delta$), while
#   for thick sheets ($t \gg \delta$) the Schelkunoff absorption term
#   $\text{SE}_A = 8.686\,(t/\delta)$ grows linearly with thickness and multiple-reflection
#   correction vanishes ($\text{SE}_M \to 0$).
#   - **Thin copper film (`copper film, 50 nm`, $t = 50\;\text{nm} \ll \delta$):** At $1\text{–}2\;\text{GHz}$,
#     $\delta \approx 2.09\text{–}1.48\;\mu\text{m}$, so $t/\delta \approx 0.024\text{–}0.034 \ll 1$
#     and $\text{SE}_A = 8.686\,t/\delta \approx 0.21\text{–}0.30\;\text{dB}$. Because the film is
#     much thinner than both the skin depth and a grid cell ($t \ll \delta \ll \Delta z$),
#     it reduces to a zero-thickness conductive sheet of sheet conductance
#     $\sigma_s = \sigma t$ (`solver.add_sheet`), with $t_{\text{sheet}} = 1/(1 + Z_0 \sigma_s / 2)$
#     and $\text{SE}_{\text{thin}} = 20\log_{10}(1 + Z_0\sigma t / 2) = 54.76\;\text{dB}$,
#     matching the full Schelkunoff sum $\text{SE}_R + \text{SE}_A + \text{SE}_M = 54.76\;\text{dB}$.
#   - **Real PCB copper foils ($t = 18\;\mu\text{m}$ and $t = 35\;\mu\text{m}$, i.e., $0.5\;\text{oz/ft}^2$ and $1\;\text{oz/ft}^2$ copper weights):**
#     Here $t \gg \delta$ ($t/\delta = 8.6\text{–}12.3$ for $18\;\mu\text{m}$ and
#     $16.8\text{–}23.9$ for $35\;\mu\text{m}$ across $1.0\text{–}2.03\;\text{GHz}$).
#     The unattenuated thin-sheet formula predicts a frequency-independent
#     $\text{SE}_{\text{thin}} = 105.87\;\text{dB}$ ($18\;\mu\text{m}$) and
#     $111.65\;\text{dB}$ ($35\;\mu\text{m}$), whereas including the full Schelkunoff
#     absorption term $\text{SE}_A = 8.686\,t/\delta$ adds $74.8\text{–}106.7\;\text{dB}$
#     ($18\;\mu\text{m}$) and $145.5\text{–}207.5\;\text{dB}$ ($35\;\mu\text{m}$) to
#     $\text{SE}_R \approx 75.1\text{–}78.1\;\text{dB}$, giving total
#     $\text{SE} = 153.0\text{–}181.8\;\text{dB}$ ($18\;\mu\text{m}$) and
#     $223.6\text{–}282.6\;\text{dB}$ ($35\;\mu\text{m}$). Because a foil
#     $\text{SE} > 100\;\text{dB}$ ($|T|^2 < 10^{-10}$) exceeds what the time-domain
#     3D FIT simulation dynamic range resolves, and because real metallic shields are
#     limited in practice to $60\text{–}90\;\text{dB}$ by seams, joints, and apertures
#     rather than bulk foil penetration (**Ott, 2009** [10], ch. 6), we evaluate the
#     $18\;\mu\text{m}$ and $35\;\mu\text{m}$ foils analytically via
#     `evaluate_copper_film_and_foil_se` while running the subcell 3D FIT benchmark
#     on the **copper film, 50 nm**.
# * **Magnetic absorbers (`magnetic particle silicone`, `sendust flake sheet`):**
#   Because $\mu(\omega)$ and $\varepsilon(\omega)$ are both large, the wave
#   impedance $|z_{\text{rel}}| = |\sqrt{\mu/\varepsilon}|$ stays much closer to
#   unity than in a conductor, reducing front-face reflection $\text{SE}_R$ while
#   the large refractive index $\operatorname{Im}\sqrt{\varepsilon\mu}$ produces
#   rapid internal attenuation $\text{SE}_A$ and high absorbed fraction $A(f)$.
# * **High-permittivity low-loss ceramic (`tio2 ceramic`, $\varepsilon_r \approx 100$):**
#   Here $z_{\text{rel}} \approx 1/\sqrt{100} = 0.10$ is purely real and
#   $\operatorname{Im}n \approx 10^{-3} \approx 0$. Thus $\text{SE}_A \approx 0$
#   and $A(f) \approx 0$: the slab acts as a lossless Fabry–Pérot etalon that
#   reflects up to $|R|^2 = ((n^2 - 1)/(n^2 + 1))^2 = (99/101)^2 = 96.1\%$ at
#   quarter-wave frequencies ($d = (2m+1)\lambda_0/(4n)$) but becomes completely
#   transparent ($|T|^2 \to 1$, $\text{SE}_M \to -\text{SE}_R$) at half-wave
#   resonances ($d = m\lambda_0/(2n)$)!
#
# ### 1.4 Metal-backed absorbers, quarter-wave matching, and the Rozanov bound
#
# In cavity-resonance damping and radar absorbing structures, an absorber sheet
# of thickness $d$ is bonded directly to a metallic enclosure wall (modeled as a
# PEC boundary at $z = d$, where $T = 0$ identically). A short-circuited
# transmission line of length $d$ has normalized input impedance and reflection
# coefficient:
#
# $$z_{\text{in}}(f, d) = \frac{Z_{\text{in}}(f, d)}{Z_0}
# = i\sqrt{\frac{\mu(f)}{\varepsilon(f)}}\tan\!\left(\frac{2\pi f}{c_0}\sqrt{\varepsilon(f)\mu(f)}\,d\right), \qquad
# \Gamma(f, d) = \frac{z_{\text{in}}(f, d) - 1}{z_{\text{in}}(f, d) + 1},$$
#
# with **reflection loss** $\text{RL}(f, d) = -20\log_{10}|\Gamma(f, d)| = -10\log_{10}R(f, d)\;[\text{dB}]$
# and absorbed fraction $A(f, d) = 1 - |\Gamma(f, d)|^2$. Near the quarter-wave
# thickness $d_{\lambda/4}(f) = \lambda_0 / \left(4\operatorname{Re}\sqrt{\varepsilon(f)\mu(f)}\right)$,
# the tangent in $z_{\text{in}}$ converts the short circuit at $z = d$ into a
# finite real resistance near the front face, creating a deep matching dip in
# $|\Gamma(f, d)|$.
#
# **Rozanov (2000)** [12] proved from Kramers–Kronig causality of $\mu(\omega)$
# and $\varepsilon(\omega)$ that for any passive metal-backed slab of thickness
# $d$ and static relative permeability $\mu_s = \lim_{f \to 0}\operatorname{Re}\mu(f)$,
# the wavelength-integrated reflection loss is bounded by:
#
# $$\left|\int_0^\infty \ln|\Gamma(\lambda)|\,d\lambda\right| \le 2\pi^2 \mu_s d.$$
#
# Because the right-hand side scales linearly with $\mu_s d$, a magnetic absorber
# with $\mu_s \approx 37\text{–}70$ (`magnetic particle silicone` or `sendust flake sheet`)
# possesses a **$37\times\text{–}70\times$ larger causal bandwidth–thickness
# budget** than any non-magnetic dielectric/conductive absorber ($\mu_s = 1$) of
# the same physical thickness $d$.

# %%
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    EMI_CLASSES,
    attenuation_db_per_cm,
    evaluate_copper_film_and_foil_se,
    evaluate_rozanov_bound,
    exact_pec_backed_absorber,
    exact_slab_rt,
    infer_eps_real_from_attenuation,
    load_magnetic_silicone_csv,
    render_all_figures,
    run_pec_backed_absorber_fit,
    run_slab_rt_fit,
    run_task_0,
    run_task_1,
    run_task_2,
    run_task_3,
    schelkunoff_decomposition,
)
from wakis.dispersive import dispersive_lib  # noqa: E402
from wakis.material_library import info, list_materials  # noqa: E402

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# Run the interactive walkthrough cells when invoked inside Jupyter / ipython
# or directly as a script with no CLI flags.
_RUN_WALKTHROUGH = "ipykernel" in sys.modules or (
    __name__ == "__main__" and len(sys.argv) == 1
)

# %% [markdown]
# ## 2. Simulation Setups, Schematics & Interactive Walkthrough
#
# Every 3D FIT simulation in this tutorial uses one of two distinct physical
# setups, each documented by an uncluttered geometry/boundary schematic (`00a`, `00c`),
# a locally refined mesh and source-spectrum view (`00b`, `00d`), and a setup
# specification table:
#
# ### Setup 1 · Free-space EMI slab reflection, transmission, and absorption (`task_1`, `task_3`)
#
# ![Setup 1 geometry, materials, sources, monitors, and boundaries](figures/00a_free_space_slab_setup.png)
#
# ![Setup 1 locally refined z-mesh and broadband Gaussian pulse spectrum](figures/00b_free_space_slab_mesh_pulse.png)
#
# | Parameter | Value / Specification | Physical & numerical role |
# | :--- | :--- | :--- |
# | **Domain extents** | $x, y \in [-d_{xy}, +d_{xy}]$ ($N_x = N_y = 2$), $z \in [-45\;\text{mm},\, d + 45\;\text{mm}]$ | Quasi-1D normal-incidence plane-wave channel in 3D `SolverFIT3D` |
# | **Boundary conditions** | `periodic` on $\pm x, \pm y$; 10-cell `cpml` on $-z$ and $+z$ | Exact TEM plane wave ($k_x = k_y = 0$) with reflectionless longitudinal termination |
# | **Slab region** | $z \in [0, d]$ ($d \in \{0.5, 1.0, 1.5, 2.0, 3.0\}\;\text{mm}$ for Classes 1–5; subcell $z = 0$ for Class 6) | `solver.add_dispersive_material` for bulk slabs; `solver.add_sheet(..., mode='thin')` for $10\text{–}200\;\text{nm}$ Cu films |
# | **Local mesh refinement** | Uniform fine $\Delta z_{\text{slab}}$ in $[-4\Delta z_{\text{slab}},\, d + 4\Delta z_{\text{slab}}]$, graded by $1.16\times$ to $\Delta z_{\text{vac}}$ | Enforces $\ge 20\;\text{cells}/\lambda_{\text{mat}}$, $\ge 4\;\text{cells}$ across $d$, and $\ge 4\;\text{cells}/\delta$ at $6\times\text{–}10\times$ fewer cells than a uniform grid |
# | **Excitation** | `CurrentSource` sheet ($J_x$) at $z_{\text{src}} \approx -15\;\text{mm}$ with `GaussianSource` ($f_0 = 1.4\;\text{GHz}$, $\Delta f = 1.0\;\text{GHz}$) | Launches a broadband $x$-polarized TEM pulse covering $0.3\text{–}2.5\;\text{GHz}$ |
# | **Monitors & normalization** | Two-run `FluxMonitor` planes at $z_r \approx -6\;\text{mm}$ and $z_t \approx d + 6\;\text{mm}$ | $R(f) = 1 - \Phi_r/\Phi_{r,\text{ref}}$, $T(f) = \Phi_t/\Phi_{t,\text{ref}}$, $A(f) = 1 - R(f) - T(f)$ |
#
# ### Setup 2 · Metal-backed EMI absorber reflection loss (`task_2`)
#
# ![Setup 2 PEC-backed absorber geometry and boundary conditions](figures/00c_metal_backed_absorber_setup.png)
#
# ![Setup 2 locally refined z-mesh at PEC ground plane and pulse spectrum](figures/00d_metal_backed_absorber_mesh_pulse.png)
#
# | Parameter | Value / Specification | Physical & numerical role |
# | :--- | :--- | :--- |
# | **Domain extents** | $x, y \in [-d_{xy}, +d_{xy}]$ ($N_x = N_y = 2$), $z \in [-45\;\text{mm},\, d]$ | Absorber bonded directly to the $+z$ ground plane ($T(f) \equiv 0$) |
# | **Boundary conditions** | `periodic` on $\pm x, \pm y$; 10-cell `cpml` on $-z$; `pec` at $z = d$ (`bc_high = ['periodic', 'periodic', 'pec']`) | Enforces tangential electric node $E_x(z = d) = 0$ on the metal backing |
# | **Absorber region** | $z \in [0, d]$ (`magnetic particle silicone` at $d \in \{1, 2, 3\}\;\text{mm}$; `sendust flake sheet` at $d \in \{0.5, 1.0, 1.5\}\;\text{mm}$) | Quarter-wave impedance matching over $0.3\text{–}4.5\;\text{GHz}$ |
# | **Monitor & observable** | `FluxMonitor` at $z_r \approx -6\;\text{mm}$ normalized by incident vacuum flux $\Phi_{r,\text{ref}}$ | Reflection loss $\text{RL}(f, d) = -10\log_{10}(1 - \Phi_r/\Phi_{r,\text{ref}})\;[\text{dB}]$ |
#
# ---
#
# ### Step 1 — Inspect the 6 EMI materials in `wakis.material_library` and fit the magnetic-particle silicone absorber

# %%
if _RUN_WALKTHROUGH:
    emi_names = list_materials(category="emi")
    print(f"Registered EMI materials ({len(emi_names)}): {emi_names}")
    for cinfo in EMI_CLASSES:
        key = cinfo["key"]
        mat = dispersive_lib[key]
        meta = info(key)
        e_1g = complex(mat.epsilon(1.0e9))
        m_1g = complex(mat.mu(1.0e9))
        print(
            f"  {key:26s}: eps(1 GHz) = {e_1g.real:7.2f} - {abs(e_1g.imag):8.2e}j, "
            f"mu(1 GHz) = {m_1g.real:5.2f} - {abs(m_1g.imag):5.2f}j | {meta['citation']['reference']}"
        )

    ds_rf = load_magnetic_silicone_csv(DATA_DIR)
    mat_rf = dispersive_lib["magnetic particle silicone"]
    mask_ov = ds_rf["f_att_hz"] <= 1.001e9
    inf_001 = infer_eps_real_from_attenuation(
        ds_rf["f_att_hz"][mask_ov],
        mat_rf.mu(ds_rf["f_att_hz"][mask_ov]),
        ds_rf["atten_db_cm"][mask_ov],
        tan_delta_e=0.01,
    )
    att_6g_pred = float(attenuation_db_per_cm(6.0e9, mat_rf.epsilon(6.0e9), mat_rf.mu(6.0e9)))
    print(
        f"Magnetic-particle silicone: inferred eps' = {inf_001['eps_real']:.2f} (tan_delta_e = 0.01, "
        f"overlap RMS = {inf_001['rms_overlap_db_cm']:.2f} dB/cm), "
        f"extrapolated alpha(6 GHz) = {att_6g_pred:.1f} dB/cm (datasheet {ds_rf['atten_db_cm'][-1]:.1f} dB/cm)"
    )

# %% [markdown]
# ### Step 2 — Schelkunoff shielding decomposition across the 6 material classes
#
# Evaluating `schelkunoff_decomposition` at $f = 2.0\;\text{GHz}$ verifies the
# exact identity $\text{SE}_{\text{total}} = \text{SE}_R + \text{SE}_A + \text{SE}_M$
# and highlights why magnetic absorbers, high-$\varepsilon$ ceramics, percolated
# CNT elastomers, and metals shield by completely different physical mechanisms:

# %%
if _RUN_WALKTHROUGH:
    f_check = np.array([2.0e9])
    for cinfo in EMI_CLASSES:
        key = cinfo["key"]
        mat = dispersive_lib[key]
        d_m = cinfo["thicknesses_mm"][1] * 1e-3
        sch = schelkunoff_decomposition(f_check, mat.epsilon(f_check), mat.mu(f_check), d_m)
        sum_check = float(sch["SE_R_db"][0] + sch["SE_A_db"][0] + sch["SE_M_db"][0])
        print(
            f"  {cinfo['short']:24s} (d={d_m*1e3:7.4f} mm): "
            f"SE = {float(sch['SE_total_db'][0]):6.2f} dB "
            f"[SE_R = {float(sch['SE_R_db'][0]):6.2f}, SE_A = {float(sch['SE_A_db'][0]):6.2f}, "
            f"SE_M = {float(sch['SE_M_db'][0]):+6.2f} dB] | A = {float(sch['A'][0]):.3f}"
        )
        assert abs(sum_check - float(sch["SE_total_db"][0])) < 1e-12

    cu_comp = evaluate_copper_film_and_foil_se(
        np.array([1.0e9, 2.0e9]),
        thicknesses_m=(50.0e-9, 18.0e-6, 35.0e-6),
    )
    cu_labels = ("copper film, 50 nm", "copper foil, 18 µm", "copper foil, 35 µm")
    for idx_cu, lbl_cu in enumerate(cu_labels):
        print(
            f"  {lbl_cu:20s} @ 2.0 GHz (t/delta={cu_comp['t_over_delta'][idx_cu, 1]:5.2f}): "
            f"SE_thin={cu_comp['SE_thin_sheet_db'][idx_cu, 1]:6.2f} dB | "
            f"Schelkunoff SE={cu_comp['SE_total_db'][idx_cu, 1]:6.2f} dB "
            f"(SE_R={cu_comp['SE_R_db'][idx_cu, 1]:5.2f}, "
            f"SE_A=8.686*t/delta={cu_comp['SE_A_asymptotic_db'][idx_cu, 1]:6.2f}, "
            f"SE_M={cu_comp['SE_M_db'][idx_cu, 1]:+6.2f} dB)"
        )

# %% [markdown]
# ### Step 3 — Live 3D FIT slab verification (bulk graded $z$-mesh & subcell `add_sheet`)
#
# Next we run two fast 3D FIT (`SolverFIT3D`) slab benchmarks: a $2.0\;\text{mm}$
# magnetic-particle silicone slab on the locally refined $z$-mesh and a **copper film, 50 nm**
# modeled via Wakis's F3 subcell zero-thickness sheet update
# (`solver.add_sheet(..., mode="thin")`):

# %%
if _RUN_WALKTHROUGH:
    freqs_w = np.linspace(0.40e9, 2.20e9, 19)
    res_rf_w = run_slab_rt_fit("magnetic particle silicone", 2.0e-3, freqs_w, cpw_target=20.0, graded=True)
    res_cu_w = run_slab_rt_fit(
        "copper foil", 50.0e-9, freqs_w, cpw_target=20.0, graded=False, use_thin_sheet=True
    )
    print(
        f"3D FIT magnetic-particle silicone (2.0 mm): Nz={res_rf_w['Nz']}, cpw={res_rf_w['cpw_slab']:.1f}, "
        f"RMS(|R|^2) = {res_rf_w['rms_R']:.4f}, RMS(|T|^2) = {res_rf_w['rms_T']:.4f}"
    )
    print(
        f"3D FIT copper film, 50 nm (add_sheet): Nz={res_cu_w['Nz']}, "
        f"RMS(|R|^2) = {res_cu_w['rms_R']:.5f}, mean SE = {np.mean(res_cu_w['SE_fit_db']):.2f} dB "
        f"(exact {np.mean(res_cu_w['SE_ex_db']):.2f} dB)"
    )

# %% [markdown]
# ### Step 4 — Metal-backed quarter-wave absorber and the Rozanov causality bound

# %%
if _RUN_WALKTHROUGH:
    f_roz_w = np.geomspace(1.0e7, 30.0e9, 400)
    mat_rf_w = dispersive_lib["magnetic particle silicone"]
    pec_rf_2mm = exact_pec_backed_absorber(
        f_roz_w, mat_rf_w.epsilon(f_roz_w), mat_rf_w.mu(f_roz_w), 2.0e-3
    )
    mu_s_rf_w = float(np.real(mat_rf_w.mu(1.0e5)))
    roz_2mm = evaluate_rozanov_bound(f_roz_w, pec_rf_2mm["gamma"], mu_s_rf_w, 2.0e-3)
    print(
        f"PEC-backed Mag. silicone (2.0 mm): peak RL = {np.max(pec_rf_2mm['RL_db']):.2f} dB, "
        f"Rozanov integral = {roz_2mm['integral_m']*1e3:.1f} mm <= 2*pi^2*mu_s*d = "
        f"{roz_2mm['rozanov_bound_m']*1e3:.1f} mm (exceeds non-magnetic bound "
        f"2*pi^2*d = {roz_2mm['nonmag_bound_m']*1e3:.1f} mm: {roz_2mm['exceeds_nonmag_bound']})"
    )


# %% [markdown]
# ## 3. Multi-Task Benchmark and Resolution Study Runner
#
# The script defines four deterministic tasks covering every figure and table in
# the tutorial:
#
# * `task_0` (`magnetic_silicone_characterization`): digitized datasheet curve
#   loading, passive magnetic pole fitting (`2 Debye`, `1 Debye + 1 Lorentz`,
#   `2 Debye + 1 Lorentz`), overlap-band $\varepsilon'$ inference and sensitivity
#   over $0.1\text{–}1.0\;\text{GHz}$, $1\text{–}6\;\text{GHz}$ attenuation
#   extrapolation, and broadband constitutive spectra of all 6 EMI material classes.
# * `task_1` (`free_space_slab_6classes`): 3D FIT (`SolverFIT3D`) normal-incidence
#   slab simulations across all 6 material classes and 3 thicknesses per class
#   (`--full`, 2 in `--quick`), Schelkunoff $\text{SE} = \text{SE}_R + \text{SE}_A + \text{SE}_M$
#   decomposition, and space-time pulse snapshots/GIF frames.
# * `task_2` (`metal_backed_absorber_and_rozanov`): 3D FIT PEC-backed absorber
#   simulations (`bc_high = ['periodic', 'periodic', 'pec']`) for `magnetic particle silicone`
#   ($d \in \{1.0, 2.0, 3.0\}\;\text{mm}$) and `sendust flake sheet`
#   ($d \in \{0.5, 1.0, 1.5\}\;\text{mm}$), 2D quarter-wave matching locus, and
#   Rozanov causality bound evaluation.
# * `task_3` (`resolution_and_mesh_study`): grid convergence across
#   $\text{cpw}_{\text{slab}} \in \{10, 14, 20, 28, 40\}$ and quantitative
#   comparison of the locally refined non-uniform $z$-mesh against a uniform
#   $z$-mesh baseline at equal slab resolution ($\Delta z_{\text{slab}}$).

# %%
def get_task_table(quick: bool = False) -> list[dict]:
    """Return the 4-task benchmark table for ``--quick`` or ``--full``."""
    cpw_main = 20 if quick else 24
    return [
        {
            "index": 0,
            "name": "magnetic_silicone_characterization",
            "kind": "characterization",
            "cpw": 24,
            "quick": quick,
        },
        {
            "index": 1,
            "name": "free_space_slab_6classes",
            "kind": "free_space_slabs",
            "cpw": cpw_main,
            "quick": quick,
        },
        {
            "index": 2,
            "name": "metal_backed_absorber_and_rozanov",
            "kind": "metal_backed",
            "cpw": cpw_main,
            "quick": quick,
        },
        {
            "index": 3,
            "name": "resolution_and_mesh_study",
            "kind": "resolution_study",
            "cpw": 10,
            "quick": quick,
        },
    ]


def run_single_task(cfg: dict, data_dir: Path = DATA_DIR) -> dict:
    """Execute one task from ``get_task_table`` and write ``data/task_<idx>.{json,npz}``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    kind = cfg["kind"]
    if kind == "characterization":
        res = run_task_0(cfg, data_dir)
    elif kind == "free_space_slabs":
        res = run_task_1(cfg, data_dir)
    elif kind == "metal_backed":
        res = run_task_2(cfg, data_dir)
    elif kind == "resolution_study":
        res = run_task_3(cfg, data_dir)
    else:
        raise ValueError(f"Unknown task kind {kind!r}.")
    wall = time.perf_counter() - t0

    idx = int(cfg["index"])
    npz_payload = {
        k: (np.asarray(v) if not isinstance(v, np.ndarray) else v)
        for k, v in res.items()
    }
    npz_payload["wall_time_s"] = np.asarray(wall, dtype=float)
    np.savez_compressed(data_dir / f"task_{idx}.npz", **npz_payload)

    meta = {
        "index": idx,
        "name": str(res["name"]),
        "kind": str(kind),
        "cpw": int(res["cpw"]),
        "cells_across_feature": float(res["cells_across_feature"]),
        "dx_min_mm": float(res["dx_min_mm"]),
        "min_feature_mm": float(res["min_feature_mm"]),
        "n_cells": int(res["n_cells"]),
        "n_steps": int(res["n_steps"]),
        "wall_time_s": float(wall),
    }
    if kind == "characterization":
        meta["rel_rms_1d1l"] = float(res["rel_rms_1d1l"])
        meta["rel_rms_2deb"] = float(res["rel_rms_2deb"])
        meta["rel_rms_2d1l"] = float(res["rel_rms_2d1l"])
        meta["max_abs_dmu_real"] = float(res["max_abs_dmu_real"])
        meta["max_abs_dmu_imag"] = float(res["max_abs_dmu_imag"])
        meta["eps_inferred_td001"] = float(res["eps_inferred_grid"][1])
        meta["rms_att_full_1d1l"] = float(res["rms_att_full_1d1l"])
        meta["rms_att_extrap_1d1l"] = float(res["rms_att_extrap_1d1l"])
        meta["rms_att_extrap_2deb"] = float(res["rms_att_extrap_2deb"])
    elif kind == "free_space_slabs":
        meta["max_rms_R"] = float(res["max_rms_R"])
        meta["max_rms_T"] = float(res["max_rms_T"])
        meta["rms_R_by_class"] = {
            EMI_CLASSES[i]["key"]: [float(x) for x in res["rms_R_table"][i]]
            for i in range(len(EMI_CLASSES))
        }
        meta["rms_T_by_class"] = {
            EMI_CLASSES[i]["key"]: [float(x) for x in res["rms_T_table"][i]]
            for i in range(len(EMI_CLASSES))
        }
    elif kind == "metal_backed":
        meta["max_rms_RL_db"] = float(res["max_rms_RL_db"])
        meta["rms_RL_rf_db"] = [float(x) for x in res["rms_RL_rf"]]
        meta["rms_RL_sd_db"] = [float(x) for x in res["rms_RL_sd"]]
    elif kind == "resolution_study":
        meta["cpw_levels"] = [float(x) for x in res["cpw_levels"]]
        meta["rms_R_mag_silicone"] = [float(x) for x in res["rms_R_conv"][0]]
        meta["nz_graded_mag_silicone"] = [int(x) for x in res["nz_graded_conv"][0]]
        meta["nz_uniform_mag_silicone"] = [int(x) for x in res["nz_uniform"]]
        meta["wall_graded_s"] = [float(x) for x in res["wall_graded_conv"][0]]
        meta["wall_uniform_s"] = [float(x) for x in res["wall_uniform"]]

    with open(data_dir / f"task_{idx}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Render all tutorial setup figures, data plots, and GIFs into ``fig_dir``."""
    return render_all_figures(data_dir=data_dir, fig_dir=fig_dir)


# %% [markdown]
# ## 4. Results & Validation against Exact Transfer-Matrix and Input-Impedance Theory
#
# Running `emi_shielding_materials.py --full` executes all 4 benchmark tasks and
# renders the setup schematics (`00a`–`00d`), eight multi-panel diagnostic
# figures (`01`–`07`, including `04b`), and the wave-propagation animation (`08`):
#
# ### 4.1 Magnetic-particle silicone pole fit, $\varepsilon'$ inference, and $1\text{–}6\;\text{GHz}$ attenuation check
#
# ![Magnetic-particle silicone permeability fit, permittivity inference, and 1-6 GHz attenuation prediction](figures/01_magnetic_silicone_fit_and_attenuation.png)
#
# ![Constitutive dispersion spectra of the 6 EMI material classes](figures/02_six_material_classes_dispersion.png)
#
# | Model / Observable | Frequency band | Parameter / Result | Datasheet / Reference target | Residual / Physical interpretation |
# | :--- | :---: | :---: | :---: | :--- |
# | **Mag. silicone `1 Debye + 1 Lorentz` $\mu(f)$ fit** | $10\;\text{MHz}\text{–}1.0\;\text{GHz}$ (measured) | $\Delta\mu_D = 15.64, \tau_D = 2.37\;\text{ns}$; $\Delta\mu_L = 20.14, f_0 = 1.84\;\text{GHz}$ | Digitized $\mu'(f), \mu''(f)$ (21 pts) | $\text{rel\_rms} = 0.36\%$ ($\max|\Delta\mu'| = 0.14$) |
# | **Mag. silicone `2 Debye` $\mu(f)$ fit** | $10\;\text{MHz}\text{–}1.0\;\text{GHz}$ (measured) | Two relaxation poles ($\mu_\infty = 1.0$) | Digitized $\mu'(f), \mu''(f)$ (21 pts) | $\text{rel\_rms} = 4.53\%$ (overdamped relaxation only) |
# | **Inferred $\varepsilon'$ ($\tan\delta_e = 0.00 / 0.01 / 0.05$)** | $0.1\text{–}1.0\;\text{GHz}$ (overlap) | $\varepsilon' = \{82.67,\, 81.06,\, 74.99\}$ | $\alpha(f) = 2.81\text{–}27.75\;\text{dB/cm}$ | Overlap RMS $= 2.86\;\text{dB/cm}$ ($\tan\delta_e = 0.01$) |
# | **Extrapolated attenuation (`1 Debye + 1 Lorentz`)** | $1.0\text{–}6.0\;\text{GHz}$ (extrapolated) | $\alpha(6\;\text{GHz}) = 84.7\;\text{dB/cm}$ | Datasheet $81.77\;\text{dB/cm}$ at $6\;\text{GHz}$ | Extrapolation RMS $= 13.16\;\text{dB/cm}$ (unconstrained $f_0 = 1.84\;\text{GHz}$ peak) |
# | **Extrapolated attenuation (`2 Debye`)** | $1.0\text{–}6.0\;\text{GHz}$ (extrapolated) | $\alpha(6\;\text{GHz}) = 89.1\;\text{dB/cm}$ | Datasheet $81.77\;\text{dB/cm}$ at $6\;\text{GHz}$ | Extrapolation RMS $= 3.42\;\text{dB/cm}$ (smooth monotonic rise) |
#
# ### 4.2 Free-space slab $|R|^2, |T|^2, A(f)$ and Schelkunoff shielding decomposition
#
# ![Free-space slab reflectance, transmittance, and absorption spectra across the 6 classes](figures/03_slab_rt_and_absorption_spectra.png)
#
# ![Schelkunoff shielding effectiveness decomposition across classes and thicknesses](figures/04_schelkunoff_se_decomposition.png)
#
# ![Copper film (50 nm) vs. real PCB copper foil (18 um and 35 um) shielding effectiveness](figures/04b_copper_film_vs_foil_se.png)
#
# ![Pulse-slab interaction snapshots with geometry overlays](figures/07_pulse_slab_interaction_snapshots.png)
#
# ![Animated pulse propagation across magnetic absorber, high-eps ceramic, and CNT rubber](figures/08_emi_slab_wave_propagation.gif)
#
# | EMI Material Class ($d$ shown) | $\text{SE}_{\text{total}}$ at $2.03\;\text{GHz}$ [dB] | $\text{SE}_R$ [dB] | $\text{SE}_A$ [dB] | $\text{SE}_M$ [dB] | Absorbed fraction $A(2.03\;\text{GHz})$ | 3D FIT vs. Transfer Matrix $\text{RMS}(|R|^2)$ / $\text{RMS}(|T|^2)$ |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Class 1 · `magnetic particle silicone` ($2.0\;\text{mm}$)** | $15.04\;\text{dB}$ | $1.93\;\text{dB}$ | $12.99\;\text{dB}$ | $+0.12\;\text{dB}$ | $0.518$ ($51.8\%$) | $1.72 \times 10^{-3}$ / $9.01 \times 10^{-4}$ |
# | **Class 2 · `sendust flake sheet` ($1.0\;\text{mm}$)** | $11.76\;\text{dB}$ | $1.54\;\text{dB}$ | $10.57\;\text{dB}$ | $-0.34\;\text{dB}$ | $0.357$ ($35.7\%$) | $1.97 \times 10^{-3}$ / $7.75 \times 10^{-4}$ |
# | **Class 3 · `tio2 ceramic` ($2.0\;\text{mm}$, $\varepsilon_r = 100$)** | $11.72\;\text{dB}$ | $9.62\;\text{dB}$ | $0.00\;\text{dB}$ | $+2.10\;\text{dB}$ | $0.000$ ($0.0\%$) | $6.82 \times 10^{-4}$ / $7.22 \times 10^{-4}$ |
# | **Class 4 · `cnt silicone` ($1.0\;\text{mm}$, $\sigma = 25\;\text{S/m}$)** | $15.24\;\text{dB}$ | $12.34\;\text{dB}$ | $3.78\;\text{dB}$ | $-0.88\;\text{dB}$ | $0.287$ ($28.7\%$) | $9.34 \times 10^{-4}$ / $6.47 \times 10^{-5}$ |
# | **Class 5 · `ni-graphite silicone` ($1.0\;\text{mm}$, $\sigma = 200\;\text{S/m}$)** | $32.24\;\text{dB}$ | $20.74\;\text{dB}$ | $10.96\;\text{dB}$ | $+0.55\;\text{dB}$ | $0.061$ ($6.1\%$) | $2.11 \times 10^{-4}$ / $1.32 \times 10^{-6}$ |
# | **Class 6 · `copper film, 50 nm` (subcell `add_sheet`)** | $54.76\;\text{dB}$ | $75.07\;\text{dB}$ | $0.30\;\text{dB}$ | $-20.60\;\text{dB}$ | $0.004$ ($0.4\%$) | $3.70 \times 10^{-5}$ / $1.32 \times 10^{-9}$ |
#
# #### Thin copper film ($50\;\text{nm}$) vs. real PCB copper foils ($18\;\mu\text{m}$ and $35\;\mu\text{m}$)
#
# Figure `04b` and the table below compare the sub-skin-depth **copper film, 50 nm**
# ($t \ll \delta$) against standard PCB copper foils ($18\;\mu\text{m} \approx 0.5\;\text{oz/ft}^2$
# and $35\;\mu\text{m} \approx 1\;\text{oz/ft}^2$, where $t \gg \delta$) at
# $1.00\;\text{GHz}$ ($\delta = 2.09\;\mu\text{m}$) and $2.03\;\text{GHz}$
# ($\delta = 1.46\;\mu\text{m}$):
#
# | Copper layer ($\sigma = 5.8\times 10^7\;\text{S/m}$) | $f$ [GHz] | $t / \delta$ | Thin-sheet $\text{SE}_{\text{thin}}$ [dB] | $\text{SE}_R$ [dB] | $\text{SE}_A = 8.686\,t/\delta$ [dB] | $\text{SE}_M$ [dB] | Full Schelkunoff $\text{SE}_{\text{total}}$ [dB] |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **copper film, 50 nm** ($t = 0.05\;\mu\text{m}$) | $1.00$ | $0.024$ | $54.76$ | $78.14$ | $0.21$ | $-23.58$ | $54.76$ |
# | **copper film, 50 nm** ($t = 0.05\;\mu\text{m}$) | $2.03$ | $0.034$ | $54.76$ | $75.06$ | $0.30$ | $-20.59$ | $54.76$ |
# | **copper foil, 18 µm** ($0.5\;\text{oz/ft}^2$ PCB) | $1.00$ | $8.61$ | $105.87$ | $78.14$ | $74.81$ | $0.00$ | $152.95$ |
# | **copper foil, 18 µm** ($0.5\;\text{oz/ft}^2$ PCB) | $2.03$ | $12.29$ | $105.87$ | $75.06$ | $106.71$ | $0.00$ | $181.77$ |
# | **copper foil, 35 µm** ($1.0\;\text{oz/ft}^2$ PCB) | $1.00$ | $16.75$ | $111.65$ | $78.14$ | $145.47$ | $0.00$ | $223.61$ |
# | **copper foil, 35 µm** ($1.0\;\text{oz/ft}^2$ PCB) | $2.03$ | $23.89$ | $111.65$ | $75.06$ | $207.49$ | $0.00$ | $282.55$ |
#
# Both the $18\;\mu\text{m}$ and $35\;\mu\text{m}$ PCB copper foils exhibit
# $\text{SE} > 100\;\text{dB}$ in both models ($105.9\text{–}111.7\;\text{dB}$
# in the thin-sheet approximation and $153.0\text{–}282.6\;\text{dB}$ when the
# full Schelkunoff skin-depth absorption term $\text{SE}_A = 8.686\,t/\delta$ is
# included), which exceeds what the 3D FIT time-domain simulation dynamic range
# resolves; in engineering practice, real metallic shields are limited to
# $60\text{–}90\;\text{dB}$ by seams, joints, and apertures rather than bulk foil
# penetration (**Ott, 2009** [10], ch. 6).
#
# ### 4.3 Metal-backed quarter-wave absorber and Rozanov bound
#
# ![Metal-backed absorber reflection loss, quarter-wave locus, and Rozanov bound](figures/05_metal_backed_absorber_and_rozanov.png)
#
# When backed by a PEC ground plane (`bc_high = ['periodic', 'periodic', 'pec']`),
# both magnetic composites exhibit clear quarter-wave matching peaks where
# $d \approx \lambda_0 / (4\operatorname{Re}\sqrt{\varepsilon\mu})$:
#
# * **Magnetic-particle silicone:** the matching peak shifts from $f_{\text{peak}} = 1.35\;\text{GHz}$
#   ($\text{RL} = 6.28\;\text{dB}$ at $d = 1.0\;\text{mm}$) to $0.88\;\text{GHz}$
#   ($\text{RL} = 13.35\;\text{dB}$ at $d = 2.0\;\text{mm}$) and $0.53\;\text{GHz}$
#   ($\text{RL} = 20.73\;\text{dB}$ at $d = 3.0\;\text{mm}$), with 3D FIT matching
#   exact input-impedance theory to $\text{RMS}(\text{RL}) \le 0.026\;\text{dB}$.
# * **Rozanov causality bound:** for a $2.0\;\text{mm}$ magnetic-particle silicone sheet, the
#   wavelength-integrated reflection loss $\int |\ln|\Gamma(\lambda)||\,d\lambda = 1385.0\;\text{mm}$
#   utilizes **$95.4\%$** of Rozanov's magnetic causal bound $2\pi^2 \mu_s d = 1452.1\;\text{mm}$
#   ($\mu_s = 36.78$) and exceeds the maximum possible non-magnetic dielectric
#   bound $2\pi^2 d = 39.5\;\text{mm}$ ($\mu_s = 1$) by **$35.1\times$**.
#
# ---
#
# ## 5. Resolution Study & Local Mesh Refinement vs. Uniform Baseline
#
# ![Resolution convergence study and locally refined z-mesh vs. uniform baseline](figures/06_resolution_convergence_study.png)
#
# Sweeping $\text{cpw}_{\text{slab}} \in \{10, 14, 20, 28, 40\}$ across the bulk
# EMI classes confirms second-order $\mathcal{O}(\Delta z^2)$ convergence of the
# 3D FIT dispersive and conductive updates. Comparing the **locally refined
# graded $z$-mesh** (`build_graded_z_grid`) against the **uniform $z$-mesh
# baseline** (`graded=False`) on the $2.0\;\text{mm}$ magnetic-particle silicone slab at
# identical slab cell sizes $\Delta z_{\text{slab}}$ demonstrates why local
# refinement is essential when a sub-millimeter high-index slab ($|n| \approx 30$)
# sits inside a centimeter-scale vacuum domain:
#
# | Resolution level | $\text{cpw}_{\text{slab}}$ | $\Delta z_{\text{slab}}$ [mm] | Graded $N_z$ | Uniform baseline $N_z$ | Cell reduction | Graded RMS($|R|^2$) | Uniform baseline RMS($|R|^2$) | Graded wall time [s] | Uniform wall time [s] |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | Coarse | $10$ | $0.250$ | $63$ | $369$ | **$5.9\times$** | $3.62 \times 10^{-3}$ | $2.50 \times 10^{-3}$ | $1.67\;\text{s}$ | $2.13\;\text{s}$ |
# | Medium-coarse | $14$ | $0.200$ | $67$ | $461$ | **$6.9\times$** | $3.41 \times 10^{-3}$ | $1.59 \times 10^{-3}$ | $2.08\;\text{s}$ | $2.80\;\text{s}$ |
# | **Production (`20 cpw` rule)** | **$20$** | **$0.133$** | **$78$** | **$692$** | **$8.9\times$** | **$2.32 \times 10^{-3}$** | **$7.10 \times 10^{-4}$** | **$3.17\;\text{s}$** | **$4.78\;\text{s}$** |
# | Fine | $28$ | $0.100$ | $89$ | $921$ | **$10.3\times$** | $9.47 \times 10^{-4}$ | $4.09 \times 10^{-4}$ | $4.23\;\text{s}$ | $7.16\;\text{s}$ |
# | Reference fine | $40$ | $0.069$ | $110$ | $1333$ (est.) | **$12.1\times$** | $4.80 \times 10^{-4}$ | — | $6.30\;\text{s}$ | — |
#
# At the production $20\text{–}28\;\text{cpw}$ resolution, the locally refined
# $z$-mesh achieves sub-$0.25\%$ reflectance accuracy ($\text{RMS}(|R|^2) \approx 0.95\text{–}2.32 \times 10^{-3}$)
# with **$8.9\times\text{–}10.3\times$ fewer grid cells** than the uniform $z$-mesh baseline.
#
# ---
#
# ## 6. Limits & Failure Modes
#
# 1. **Extrapolating purely relaxational (Debye) magnetic fits above ferromagnetic resonance:**
#    In a magnetic composite, spin precession has inertia ($\partial^2\mathbf{M}/\partial t^2$),
#    causing $\mu''(\omega)$ to decay as $\omega^{-3}$ and $\mu'(\omega) - 1$ as
#    $\omega^{-2}$ well above the natural resonance $f_0$. A first-order Debye
#    pole ($\chi_m \propto (1 + i\omega\tau)^{-1}$) decays only as $\omega^{-1}$
#    in $\mu''$, systematically overpredicting microwave attenuation when
#    extrapolated above $1\;\text{GHz}$ (Figure 01c: $98.0\;\text{dB/cm}$ vs.
#    datasheet $81.8\;\text{dB/cm}$ at $6\;\text{GHz}$). Including a second-order
#    Lorentz pole captures the gyromagnetic roll-off.
# 2. **Under-resolving the skin depth $\delta(f)$ in bulk conductive elastomers:**
#    In a conductor with $\sigma \gg \omega\varepsilon_0\varepsilon'$, both the
#    decay length $\delta$ and internal wavelength $\lambda_{\text{mat}} = 2\pi\delta$
#    shrink as $\sigma^{-1/2}$. If $\Delta z > \delta / 2$, the discrete Yee
#    stencil cannot resolve the exponential boundary layer at the front face,
#    corrupting the surface impedance and transmission phase. Always enforce
#    $\Delta z \le \delta_{\min} / 4$ for bulk conductors (`cnt silicone`,
#    `ni-graphite silicone`), or switch to Wakis's subcell sheet/SIBC model
#    (`solver.add_sheet`) when $d < \Delta z$ (`copper foil`).
# 3. **Using `Nx = 1` or `Ny = 1` with periodic transverse boundaries in 3D dispersive runs:**
#    When `bc_low = ['periodic', 'periodic', ...]` is active on `GridFIT3D`,
#    `SolverFIT3D` removes the upper periodic ghost cell along $x$ and $y$.
#    Using `Nx = 2, Ny = 2` (three node coordinates along $x$ and $y$) ensures
#    interior primary/dual transverse edges remain non-empty for
#    `solver.add_dispersive_material`.
# 4. **Abrupt mesh jumps right at a high-contrast material interface:**
#    In FIT, the tangential electric update at a material interface weights the
#    two adjacent cells by $1/2$ assuming symmetric dual-edge halves. Keeping a
#    small uniform collar ($3\text{–}4$ fine cells of size $\Delta z_{\text{slab}}$)
#    outside the slab faces before starting geometric mesh grading preserves
#    second-order interface accuracy.
#
# ---
#
# ## 7. Exercises
#
# 1. **Two-layer Jaumann / graded-impedance absorber:** Stack a $1.0\;\text{mm}$
#    front matching layer of `'cnt silicone'` or `'fr4'` in front of a
#    $2.0\;\text{mm}$ PEC-backed `'magnetic particle silicone'` layer and compute the
#    broadband reflection loss $\text{RL}(f)$ from $0.5\text{ to }6.0\;\text{GHz}$.
#    Compare the $-10\;\text{dB}$ bandwidth and the Rozanov integral
#    $\int |\ln|\Gamma(\lambda)||\,d\lambda$ against the single-layer magnetic-particle silicone absorber.
# 2. **Percolation threshold sensitivity in CNT rubber:** Vary `sigma` in
#    `dispersive_lib['cnt silicone']` across $\{0.1, 1.0, 10.0, 25.0, 100.0\}\;\text{S/m}$
#    (spanning the insulator-to-conductor percolation transition) for a
#    $1.5\;\text{mm}$ slab at $2\;\text{GHz}$, and plot the crossover between
#    absorption-dominated ($\text{SE}_A > \text{SE}_R$) and reflection-dominated
#    ($\text{SE}_R > \text{SE}_A$) shielding.
# 3. **Subcell skin-effect SIBC sheet (`mode='sibc'`):** Replace `mode='thin'`
#    in `solver.add_sheet` with `metal_sheet(sigma=5.8e7, f_range=(0.3e9, 3.0e9), mode='sibc', sides=2)`
#    to model the frequency-dependent surface impedance $Z_s(\omega) = (1+i)/(\sigma\delta(\omega))$
#    in the thick-skin regime.
#
# ---
#
# ## Verified References
#
# 1. Manufacturer datasheet of a commercially available magnetic-particle-loaded
#    silicone absorber (digitized curves in `data/`).
# 2. S. Yoshida, M. Sato, E. Sugawara, and Y. Shimada, "Permeability and
#    electromagnetic-interference characteristics of Fe–Si–Al alloy flakes–polymer
#    composite," *J. Appl. Phys.* **85**(8), 4636–4638 (1999),
#    doi:[10.1063/1.370432](https://doi.org/10.1063/1.370432).
# 3. S. Whalen, S. K. Chaturvedi, C. Overman, Z. Li, L. T. Nguyen,
#    R. K. Kukkadapu, D. A. Cullen, T. J. Roosendaal, and K. S. Kappagantula,
#    "Microstructure and Magnetic Properties of Fe-Si-Al (Sendust) Core-Shell
#    Flakes Consolidated via Shear Assisted Processing and Extrusion," *IEEE
#    Trans. Magn.* **61**(7), 2000409 (2025),
#    doi:[10.1109/TMAG.2025.3561655](https://doi.org/10.1109/TMAG.2025.3561655).
# 4. J. L. Snoek, "Dispersion and absorption in magnetic ferrites at frequencies
#    above one Mc/s," *Physica* **14**(4), 207–217 (1948),
#    doi:[10.1016/0031-8914(48)90038-X](https://doi.org/10.1016/0031-8914(48)90038-X).
# 5. A. Templeton, X. Wang, S. J. Penn, S. J. Webb, L. F. Cohen, and
#    N. McN. Alford, "Microwave Dielectric Loss of Titanium Oxide," *J. Am.
#    Ceram. Soc.* **83**(1), 95–100 (2000),
#    doi:[10.1111/j.1151-2916.2000.tb01154.x](https://doi.org/10.1111/j.1151-2916.2000.tb01154.x).
# 6. M. E. Tobar, J. Krupka, E. N. Ivanov, and R. A. Woode, "Anisotropic complex
#    permittivity measurements of mono-crystalline rutile between 10 and 300 K,"
#    *J. Appl. Phys.* **83**(3), 1604–1609 (1998),
#    doi:[10.1063/1.366871](https://doi.org/10.1063/1.366871).
# 7. M. H. Al-Saleh and U. Sundararaj, "Electromagnetic interference shielding
#    mechanisms of CNT/polymer composites," *Carbon* **47**(7), 1738–1746 (2009),
#    doi:[10.1016/j.carbon.2009.02.030](https://doi.org/10.1016/j.carbon.2009.02.030).
# 8. S. Ananthasubramanian, T. G. R. N. G. R. P. K. Reddy, *et al.*, "Unlocking
#    superior EMI shielding at low thickness: the role of dispersion in CNT-PDMS
#    composites," *Mater. Adv.* **6**, 2345 (2025),
#    doi:[10.1039/d5ma00120j](https://doi.org/10.1039/d5ma00120j).
# 9. S. Galyaltdinov, *et al.*, "Electromagnetic Interference Shielding Materials
#    from Silicone-Coated Carbon Nanotube Fabrics," *Nanomaterials* **12**(19),
#    3538 (2022), doi:[10.3390/nano12193538](https://doi.org/10.3390/nano12193538).
# 10. H. W. Ott, *Electromagnetic Compatibility Engineering*, John Wiley & Sons,
#     Hoboken, NJ (2009), ch. 6 (Shielding),
#     doi:[10.1002/9780470508510](https://doi.org/10.1002/9780470508510).
# 11. S. A. Schelkunoff, *Electromagnetic Waves*, D. Van Nostrand, New York
#     (1943), ch. 8 ( Wave transmission and shielding).
# 12. K. N. Rozanov, "Ultimate thickness to bandwidth ratio of radar absorbers,"
#     *IEEE Trans. Antennas Propag.* **48**(8), 1230–1234 (2000),
#     doi:[10.1109/8.884491](https://doi.org/10.1109/8.884491).
# 13. C. R. Paul, *Introduction to Electromagnetic Compatibility*, 2nd ed.,
#     John Wiley & Sons, Hoboken, NJ (2006), ch. 10 (Shielding).

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial 2.4: EMI shielding and absorbing materials (6 classes, Schelkunoff SE, PEC absorber, Rozanov bound)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run the fast 4-task benchmark (2 thicknesses per class) and render all figures.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full 4-task benchmark (3 thicknesses per class, up to 40 cpw) and render all figures.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run only a single task index (0..3) and write data/task_<idx>.{json,npz}.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the task table as JSON and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render figures/ from existing data/task_*.npz without re-running simulations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quick = bool(args.quick)
    if (
        not args.quick
        and not args.full
        and not args.plot
        and not args.list_tasks
        and args.task is None
    ):
        quick = True

    tasks = get_task_table(quick=quick)
    if args.list_tasks:
        print(json.dumps(tasks, indent=2))
        return 0

    if args.plot and args.task is None and not args.quick and not args.full:
        written = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"Rendered {len(written)} figures in {FIG_DIR}")
        return 0

    if args.task is not None:
        if args.task < 0 or args.task >= len(tasks):
            raise ValueError(f"--task must be in [0, {len(tasks) - 1}], got {args.task}.")
        meta = run_single_task(tasks[args.task], data_dir=DATA_DIR)
        print(json.dumps(meta, indent=2))
        return 0

    for cfg in tasks:
        meta = run_single_task(cfg, data_dir=DATA_DIR)
        print(
            f"[Task {meta['index']:02d}/{len(tasks)-1:02d}] {meta['name']:34s} "
            f"cpw={meta['cpw']:2d} wall={meta['wall_time_s']:.2f}s"
        )

    written = generate_all_figures(DATA_DIR, FIG_DIR)
    print(f"Rendered {len(written)} figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
