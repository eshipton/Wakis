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
# # Tutorial 8.2 · Cherenkov Radiation from a Moving Charge
#
# When a charged particle or relativistic electron bunch travels at velocity
# $v = \beta c$ through a dielectric medium whose refractive index $n(\omega)$
# reduces the phase velocity of light below $v$ ($\beta n(\omega) > 1$), the
# spherical electromagnetic wavelets emitted along its trajectory interfere
# constructively along a conical shock front—**Cherenkov radiation**
# (Cherenkov, 1934; Frank & Tamm, 1937). The same phase-matching mechanism
# underpins slow-wave vacuum-electronics amplifiers (Tutorial 6.2 and the
# upcoming bunch-over-slow-wave-structure tutorial) and, on a discrete mesh,
# explains why standard Yee/FIT schemes suffer from spurious **numerical
# Cherenkov radiation** in vacuum unless a longitudinal dispersion-free time
# integrator (`scheme="tete"`, `wakis/tete.py`) is used. This tutorial was
# inspired by the moving-source FDTD demonstration in Oskooi et al., *Comput.
# Phys. Commun.* **181**, 687 (2010), `doi:10.1016/j.cpc.2009.11.008`, and is
# implemented here in 3D using Wakis's finite-integration `Beam` current
# source, coaxial `FluxMonitor`, dispersive ADE materials, and explicit TE/TM
# (`tete`) solver.
#
# ### Learning goals
#
# 1. **Cherenkov threshold and Mach cone geometry** — Derive and measure both
#    the wavevector angle $\cos\theta_c = 1/(\beta n)$ and the shock-front
#    half-angle $\alpha_c = 90^\circ - \theta_c$ in a 3D dielectric ($n = 1.5$)
#    above threshold ($\beta = 0.90$), and contrast them with the non-radiating
#    Lorentz-contracted Coulomb field below threshold ($\beta = 0.60$).
# 2. **Frank–Tamm spectrum and finite-bunch form factor** — Extract the
#    radiated spectral energy density per unit path length
#    $d^2 W / (d\omega\,dz)$ through a coaxial `FluxMonitor` surface and
#    compare it with the analytical Frank–Tamm law weighted by the coherent
#    Gaussian bunch form factor $F(\omega) = \exp(-\omega^2 \sigma_z^2 / v^2)$.
# 3. **Material dispersion and Cherenkov band edges** — Couple the moving
#    bunch to a dispersive Lorentz medium (`wakis.dispersive.DispersiveMaterial`)
#    from Chapter 2 and verify that radiation is emitted only in frequency
#    bands where $\beta \operatorname{Re} n(\omega) > 1$, cutting off at the
#    resonance band edge.
# 4. **Numerical Cherenkov radiation in vacuum ($\beta = 0.999$): Yee vs
#    TETE** — Show that sub-$c$ numerical phase velocity along $z$ on a
#    standard Yee grid causes an ultrarelativistic bunch in **vacuum** to
#    radiate spuriously, quantify its scaling with grid resolution, and show
#    that the dispersion-free explicit TE/TM scheme (`scheme="tete"`) removes
#    it by nearly five orders of magnitude.
# 5. **Grid convergence and failure modes** — Perform a 3-level 3D resolution
#    study on the cone angle and radiated spectrum, and document physical and
#    numerical failure modes.
#
# ---
#
# ## Physical Background & Governing Equations
#
# ### 1. Rigid Gaussian bunch current in Wakis (`wakis/sources.py`)
#
# Wakis drives a moving rigid bunch of total charge $q$ [C], RMS longitudinal
# length $\sigma_z$ [m], and velocity $\mathbf{v} = \beta c\,\hat{\mathbf{z}}$
# along the grid line $(x_s, y_s)$ via the axial current density
#
# $$
# J_z(x, y, z, t) = \frac{q v}{\tilde{\Delta x}\,\tilde{\Delta y}\sqrt{2\pi}\,\sigma_z}
# \exp\!\left(-\frac{(z - z_0 - v t)^2}{2\sigma_z^2}\right)
# \delta_{i, i_s}\delta_{j, j_s}
# \quad [\text{A/m}^2],
# $$
#
# whose longitudinal Fourier transform at frequency $\omega = k_z v$ is
#
# $$
# \hat{I}_z(z, \omega) = \int_{-\infty}^{\infty} I_z(z, t)\,e^{-i\omega t}\,dt
# = q\,\exp\!\left(-\frac{\omega^2 \sigma_z^2}{2 v^2}\right)
# e^{-i \omega (z - z_0)/v} \quad [\text{C} = \text{A}\cdot\text{s}].
# $$
#
# ### 2. Wavevector cone angle $\theta_c$, shock half-angle, and threshold
#
# In an isotropic medium with phase velocity $v_p(\omega) = c / n(\omega)$, each
# frequency component $\omega$ has wavevector magnitude
# $k(\omega) = n(\omega)\omega / c$ and axial wavenumber forced by the moving
# source, $k_z = \omega / v = \omega / (\beta c)$. The transverse radial
# wavenumber is therefore
#
# $$
# k_\perp(\omega) = \sqrt{k(\omega)^2 - k_z^2}
# = \frac{\omega}{v}\sqrt{\beta^2 n(\omega)^2 - 1} \quad [\text{rad/m}].
# $$
#
# - **Above threshold ($\beta n(\omega) > 1$)**: $k_\perp(\omega)$ is real, so
#   conical cylindrical waves propagate outward to infinity. The **wavevector
#   cone angle** $\theta_c$ between $\mathbf{k}$ and the particle velocity
#   $\hat{\mathbf{z}}$ satisfies
#
#   $$
#   \cos\theta_c(\omega) = \frac{k_z}{k(\omega)} = \frac{1}{\beta\,n(\omega)},
#   \qquad
#   \tan\theta_c(\omega) = \frac{k_\perp}{k_z} = \sqrt{\beta^2 n(\omega)^2 - 1},
#   $$
#
#   while the **shock-front (Mach cone) half-angle** between the wavefront
#   surface (normal to $\mathbf{k}$) and the particle axis is the complement
#
#   $$
#   \alpha_c = 90^\circ - \theta_c, \qquad
#   \sin\alpha_c = \cos\theta_c = \frac{1}{\beta\,n(\omega)}.
#   $$
#
#   On a meridian slice $(x, z)$ at snapshot time $t$, the shock ridge lags
#   behind the bunch centroid $z_b(t) = z_0 + v t$ linearly with transverse
#   radius $r = |x|$:
#
#   $$
#   z_b(t) - z_{\text{shock}}(r) = r \cot\alpha_c = r \tan\theta_c.
#   $$
#
# - **Below threshold ($\beta n(\omega) < 1$)**: $k_\perp = i\kappa_\perp$ with
#   $\kappa_\perp = \omega / (\gamma_n v)$ and effective contraction factor
#   $\gamma_n = (1 - \beta^2 n^2)^{-1/2}$ is purely imaginary. In the Lorenz
#   gauge, the comoving scalar potential $\Phi(r, \zeta) > 0$ is even in
#   $\zeta = z - v t$, and since $A_z = (\beta^2 n^2 / v)\Phi$ and
#   $\partial_t = -v\,\partial_\zeta$, the longitudinal electric field is
#
#   $$
#   E_z(r, \zeta) = -\frac{\partial\Phi}{\partial z} - \frac{\partial A_z}{\partial t}
#   = -\frac{1}{\gamma_n^2}\frac{\partial\Phi}{\partial\zeta}
#   = +\frac{q}{2\pi^2 \varepsilon\,\gamma_n^2}\int_0^\infty
#   k_z K_0\!\left(\frac{k_z r}{\gamma_n}\right)
#   e^{-\frac{1}{2}k_z^2\sigma_z^2}\sin(k_z\zeta)\,dk_z,
#   $$
#
#   where differentiating $\cos(k_z\zeta)$ with respect to $\zeta$ cancels the
#   leading minus sign. For $q > 0$, the bound Lorentz-contracted Coulomb field
#   is strictly odd in $\zeta = z - v t$—pointing forward ($E_z > 0$) ahead of
#   the bunch ($\zeta > 0$) and backward ($E_z < 0$) behind the bunch
#   ($\zeta < 0$)—decays exponentially with $r$, and carries **zero outward
#   radial Poynting flux**.
#
# ### 3. Frank–Tamm spectrum and Gaussian bunch form factor (SI units)
#
# Solving the cylindrical Helmholtz equation for the magnetic potential vector
# $\hat{A}_z(r, \omega) = \frac{\mu}{4 i}\hat{I}_z(\omega) H_0^{(2)}(k_\perp r)$
# in a medium with permeability $\mu = \mu_r \mu_0$ and permittivity
# $\varepsilon(\omega) = \varepsilon_0 n(\omega)^2 / \mu_r$, and integrating
# the outward radial Poynting flux across a coaxial cylinder of length $L$
# yields the **Frank–Tamm spectral energy density per unit angular frequency
# and unit path length** (Frank & Tamm, 1937; Jackson, 1999, Eq. 13.48):
#
# $$
# \frac{d^2 W}{d\omega\,dz}
# = \frac{q^2 \mu(\omega)\,\omega}{4\pi}
# \left(1 - \frac{1}{\beta^2 n(\omega)^2}\right)
# F(\omega)
# \quad \left[\frac{\text{J}\cdot\text{s}}{\text{m}}
# = \frac{\text{J}}{\text{m}\cdot(\text{rad/s})}\right],
# $$
#
# where the coherent **bunch form factor** for a Gaussian line bunch of RMS
# length $\sigma_z$ is
#
# $$
# F(\omega) = \left|\hat{\lambda}(k_z = \omega/v)\right|^2
# = \exp\!\left(-\frac{\omega^2 \sigma_z^2}{v^2}\right) \quad [-].
# $$
#
# Dimensional check in SI:
# $[q^2 \mu_0 \omega] = \text{C}^2 \cdot (\text{N}/\text{A}^2) \cdot \text{s}^{-1}
# = (\text{A}\cdot\text{s})^2 \cdot (\text{J}\cdot\text{m}^{-1}\cdot\text{A}^{-2}) \cdot \text{s}^{-1}
# = \text{J}\cdot\text{s}/\text{m}$.
# For a point charge ($\sigma_z \to 0$, $F(\omega) \equiv 1$) in a
# non-dispersive medium, $d^2 W / (d\omega\,dz) \propto \omega$ grows without
# bound at high frequencies; for a finite bunch of length $\sigma_z$, coherent
# Cherenkov emission rolls off exponentially above the characteristic cutoff
# frequency $\omega_c \sim v / \sigma_z$ ($f_c = \beta c / (2\pi\sigma_z)$).
#
# By Parseval's theorem (`wakis/field_monitors.py`), a coaxial `FluxMonitor` of
# axial length $L_{\text{box}}$ accumulates the one-sided Fourier flux
# $\Phi(f) = \operatorname{Re}\oint (\hat{\mathbf{E}}(f) \times \hat{\mathbf{H}}^*(f))\cdot d\mathbf{A}$
# $[\text{J}/\text{Hz}^2]$ such that $W_{\text{rad}} = 2\int_0^\infty \Phi(f)\,df$.
# Converting from $df$ to $d\omega = 2\pi\,df$ and normalizing by $L_{\text{box}}$
# gives the direct experimental estimator
#
# $$
# \left.\frac{d^2 W}{d\omega\,dz}\right|_{\text{sim}}
# = \frac{\Phi_{\text{lateral}}(f)}{\pi\,L_{\text{box}}}.
# $$
#
# ### 4. Dispersive media and Numerical Cherenkov Radiation (Yee vs TETE)
#
# - **Physical dispersion**: In a Lorentz oscillator medium from Chapter 2,
#
#   $$
#   \varepsilon_r(\omega) = \varepsilon_\infty
#   + \frac{\Delta\varepsilon\,\omega_0^2}{\omega_0^2 - \omega^2 + i\gamma\omega},
#   $$
#
#   Cherenkov radiation is emitted **only** in frequency bands where
#   $\beta \operatorname{Re} n(\omega) > 1$. Above the resonance frequency
#   $\omega_0$, the medium enters the anomalous/Reststrahlen stopband where
#   $\operatorname{Re}\varepsilon_r(\omega) < \varepsilon_\infty = 1$, so
#   $\beta \operatorname{Re} n(\omega) < 1$ and Cherenkov radiation is quenched.
#
# - **Numerical Cherenkov radiation (NCR)**: On a standard 3D Yee/FIT grid, the
#   vacuum dispersion relation along $z$ ($k_\perp = 0$) is
#
#   $$
#   \frac{v_{p,z}^{\text{Yee}}(k_z)}{c}
#   = \frac{2}{k_z c \Delta t}\arcsin\!\left(\frac{c\Delta t}{\Delta z}\sin\frac{k_z\Delta z}{2}\right)
#   \approx 1 - \frac{1 - (c\Delta t / \Delta z)^2}{24}(k_z\Delta z)^2 < 1,
#   $$
#
#   because the 3D Courant stability limit forces $c\Delta t / \Delta z < 1$.
#   Consequently, an ultrarelativistic bunch ($\beta = 0.999 \to 1$) in
#   **vacuum** moves faster than the grid phase velocity at short wavelengths
#   ($v_{p,z}^{\text{Yee}}(k_z) < \beta c$) and emits a spurious numerical
#   Cherenkov wake. Wakis's explicit TE/TM scheme (`scheme="tete"`,
#   `wakis/tete.py`, Dohlus & Zagorodnov, 2009) splits the transverse update so
#   that the magic longitudinal step $c\Delta t = \Delta z$ is stable whenever
#   $\Delta z \le (\Delta x^{-2} + \Delta y^{-2})^{-1/2}$, yielding the exact
#   dispersion relation
#
#   $$
#   \sin^2\!\left(\frac{\omega\Delta t}{2}\right)
#   = \sin^2\!\left(\frac{k_z\Delta z}{2}\right)
#   + (c\Delta t)^2\left(\frac{\sin^2(k_x\Delta x/2)}{\Delta x^2}
#   + \frac{\sin^2(k_y\Delta y/2)}{\Delta y^2}\right)\cos^2\!\left(\frac{k_z\Delta z}{2}\right).
#   $$
#
#   At $c\Delta t = \Delta z$, $v_{p,z}^{\text{TETE}}(k_z) \equiv c$ for all
#   $k_z$ and $\omega(k_\perp, k_z) > c k_z > \beta c k_z$ for every pipe mode,
#   eliminating numerical Cherenkov radiation completely.

# %%
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _helpers as h  # noqa: E402

TASKS = ("dielectric", "dispersive", "numerical_cherenkov", "convergence")


# %% [markdown]
# ## Task 1 · Dielectric ($n = 1.5$): Cherenkov Cone, Threshold, and Frank–Tamm Spectrum
#
# We simulate a Gaussian bunch ($q = 1\;\text{nC}$, $\sigma_z = 1.5\;\text{mm}$)
# crossing a non-dispersive dielectric with $n = 1.5$ ($\varepsilon_r = 2.25$,
# threshold $\beta_{\text{th}} = 1/n = 2/3 \approx 0.6667$) surrounded by CPML
# absorbing boundaries on all six faces:
#
# - **Above threshold ($\beta = 0.90$, $\beta n = 1.35 > 1$)**: The bunch
#   radiates a sharp conical shock front with theoretical wavevector angle
#   $\theta_c = \arccos(1/1.35) = 42.20^\circ$ ($\cos\theta_c = 0.7407$) and
#   shock half-angle $\alpha_c = 90^\circ - \theta_c = 47.80^\circ$.
# - **Below threshold ($\beta = 0.60$, $\beta n = 0.90 < 1$)**: The field is the
#   Lorentz-contracted, fore-aft antisymmetric Coulomb field ($E_z < 0$ behind
#   the bunch at $\zeta_{\min} \approx -2.1\;\text{mm}$, $E_z > 0$ ahead at
#   $\zeta_{\max} \approx +2.4\;\text{mm}$ at $r = 3.6\;\text{mm}$) with zero
#   radiated Cherenkov cone.

# %%
def run_dielectric_section(quick: bool = False) -> dict:
    """Execute or load Task 1 (non-dispersive dielectric n = 1.5)."""
    json_path = h.DATA_DIR / "task_dielectric.json"
    if not json_path.exists():
        h.run_task_dielectric(quick=quick)
    summary = json.loads(json_path.read_text())
    print(
        f"[Task 1 · Dielectric n={summary['n_refr']:.2f}] "
        f"beta=0.90: theta_c={summary['theta_c_sim_deg']:.2f} deg "
        f"(theory {summary['theta_c_theory_deg']:.2f} deg, "
        f"rel err on cos(theta_c) = {100*summary['cos_theta_c_rel_err']:.2f}%), "
        f"dW/dz={summary['dW_dz_super_sim_J_m']:.3e} J/m "
        f"(theory {summary['dW_dz_super_theory_J_m']:.3e} J/m); "
        f"beta=0.60 sub/super power ratio = {summary['sub_to_super_power_ratio']:.2e}"
    )
    return summary


# %% [markdown]
# ## Task 2 · Dispersive Lorentz Medium: Cherenkov Passband and Band Edge
#
# Next we replace the non-dispersive background with a single-pole Lorentz
# medium (`DispersiveMaterial` + `Lorentz` from Chapter 2) having
# $\varepsilon_\infty = 1.0$, $\Delta\varepsilon = 1.25$ ($\varepsilon_s = 2.25$,
# $n(0) = 1.5$), resonance frequency $f_0 = 38\;\text{GHz}$, and damping
# $\gamma/(2\pi) = 1.5\;\text{GHz}$, traversed by a bunch at $\beta = 0.85$
# ($n_{\text{th}} = 1/\beta = 1.1765$). Below $f_0$, normal dispersion keeps
# $\beta \operatorname{Re} n(\omega) > 1$ (passband); above $f_0$,
# $\operatorname{Re}\varepsilon_r(\omega) < 1$ so $\beta \operatorname{Re} n(\omega) < 1$
# (Reststrahlen stopband), cutting off the Cherenkov spectrum at the band edge.

# %%
def run_dispersive_section(quick: bool = False) -> dict:
    """Execute or load Task 2 (dispersive Lorentz medium)."""
    json_path = h.DATA_DIR / "task_dispersive.json"
    if not json_path.exists():
        h.run_task_dispersive(quick=quick)
    summary = json.loads(json_path.read_text())
    print(
        f"[Task 2 · Dispersive Lorentz f0={summary['f0_ghz']:.1f} GHz, beta={summary['beta']:.2f}] "
        f"Passband energy (<32 GHz) = {summary['passband_energy_J_m']:.3e} J/m, "
        f"Stopband energy (>44 GHz) = {summary['stopband_energy_J_m']:.3e} J/m "
        f"(stop/pass ratio = {summary['stop_to_pass_ratio']:.3e})"
    )
    return summary


# %% [markdown]
# ## Task 3 · Vacuum ($\beta = 0.999$): Spurious Numerical Cherenkov in Yee vs Dispersion-Free TETE
#
# In a smooth vacuum drift pipe ($\varepsilon_r = 1, \mu_r = 1$), a bunch moving
# at $\beta = 0.999 < 1$ cannot physically emit Cherenkov radiation. On a
# standard Yee mesh (`scheme="yee"`), however, the numerical phase velocity
# $v_{p,z}^{\text{Yee}}(k_z) < c$ drops below $\beta c = 0.999 c$ at high
# wavenumbers, exciting a strong spurious wake cone behind the bunch. Switching
# to `SolverFIT3D(..., scheme="tete")` (`wakis/tete.py`, with $\Delta x = \Delta y = 2\Delta z$
# so the magic step $c\Delta t = \Delta z$ is stable) makes longitudinal
# propagation exact ($v_{p,z}^{\text{TETE}} \equiv c$) and suppresses the
# spurious wake energy by nearly five orders of magnitude. This directly impacts
# beam-driven wakefield and traveling-wave-tube (TWT) simulations (see
# Tutorial 6.2 `06-02_sws_cold_test` and the planned bunch-over-slow-wave-structure
# tutorial), where numerical Cherenkov noise on a Yee grid can mask the physical
# synchronous mode interaction.

# %%
def run_numerical_cherenkov_section(quick: bool = False) -> dict:
    """Execute or load Task 3 (vacuum beta=0.999 Yee vs TETE)."""
    json_path = h.DATA_DIR / "task_numerical_cherenkov.json"
    if not json_path.exists():
        h.run_task_numerical_cherenkov(quick=quick)
    summary = json.loads(json_path.read_text())
    print(
        f"[Task 3 · Vacuum beta={summary['beta']:.3f} Yee vs TETE] "
        f"dz={summary['dz_mm'][0]:.2f} mm: W_wake(Yee)={summary['W_wake_yee_J'][0]:.3e} J vs "
        f"W_wake(TETE)={summary['W_wake_tete_J'][0]:.3e} J "
        f"(ratio = {summary['yee_to_tete_energy_ratio'][0]:.1e}x)"
    )
    return summary


# %% [markdown]
# ## Task 4 · Resolution Convergence Study
#
# To satisfy the FDTD/FIT grid rule of **$\ge 20$ cells per wavelength in the
# highest-index medium ($n = 1.5$)** within the 3D memory and runtime budget, we
# **restrict the analysed frequency band (for both the Frank–Tamm spectrum and
# the Cherenkov cone extraction) to $f \in [12, 30]\;\text{GHz}$**, where the
# shortest wavelength in $n = 1.5$ is
# $\lambda_{30\;\text{GHz}} = c / (n f_{\max}) = 6.662\;\text{mm}$ (covering the
# coherent Gaussian bunch emission peak at $20.2\;\text{GHz}$ up to the
# $f_c = v/(2\pi\sigma_z) = 28.6\;\text{GHz}$ roll-off). Across the three 3D
# refinement levels (Coarse $\Delta x = 0.56\;\text{mm}$, Medium
# $\Delta x = 0.42\;\text{mm}$, Fine $\Delta x = 0.32\;\text{mm}$), the
# resolution at $30\;\text{GHz}$ in $n = 1.5$ is $11.9$, $15.9$, and
# **$20.8\;\text{cells}/\lambda$**, so the finest level meets the
# $\ge 20\;\text{cells}/\lambda$ criterion (whereas at $50\;\text{GHz}$, where
# $\lambda_{50\;\text{GHz}} = 4.00\;\text{mm}$, those same grids reach only
# $7.1\text{–}12.5\;\text{cells}/\lambda$):
#
# | Level | $\Delta x = \Delta z$ [mm] | $\lambda_{30\;\text{GHz}}/\Delta x$ ($n=1.5$) | $\lambda_{50\;\text{GHz}}/\Delta x$ (unrestricted) | $\theta_c$ [deg] (theory $42.21^\circ$) | $|\cos\theta_c - 1/(\beta n)| / (1/(\beta n))$ | Spectrum $L_2$ error ($12\text{–}30\;\text{GHz}$) | Integrated $dW/dz$ error ($12\text{–}30\;\text{GHz}$) |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Coarse** | $0.56$ | $11.9$ | $7.1$ | $43.30^\circ$ | $1.75\%$ | $3.22\%$ | $1.28\%$ |
# | **Medium** | $0.42$ | $15.9$ | $9.5$ | $43.10^\circ$ | $1.42\%$ | $3.00\%$ | $1.32\%$ |
# | **Fine** | $0.32$ | **$20.8$** ($\ge 20$) | $12.5$ | $42.87^\circ$ | $1.05\%$ | $2.81\%$ | $1.44\%$ |

# %%
def run_convergence_section(quick: bool = False) -> dict:
    """Execute or load Task 4 (3-level 3D resolution study)."""
    json_path = h.DATA_DIR / "task_convergence.json"
    if not json_path.exists():
        h.run_task_convergence(quick=quick)
    summary = json.loads(json_path.read_text())
    ppw_30 = summary.get(
        "ppw_30ghz", [p * (50.0 / 30.0) for p in summary["ppw_50ghz"]]
    )
    for idx, lvl in enumerate(summary["levels"]):
        print(
            f"[Task 4 · {lvl:6s}] dx={summary['dx_mm'][idx]:.2f} mm, "
            f"ppw@30GHz={ppw_30[idx]:.1f} (ppw@50GHz={summary['ppw_50ghz'][idx]:.1f}): "
            f"theta_c={summary['theta_c_deg'][idx]:.2f} deg "
            f"(cos rel err={100*summary['cos_rel_err'][idx]:.2f}%), "
            f"spec L2 err (12-30 GHz)={100*summary['spec_l2_rel_err'][idx]:.2f}%"
        )
    return summary


# %% [markdown]
# ## Results, Figures, and Animations
#
# All seven diagnostic PNG figures and three animated GIFs (`< 5 MB` each) are
# rendered into `figures/`:
#
# - `figures/01_geometry_and_mesh.png`: 3D FIT mesh, CPML shell, coaxial
#   `FluxMonitor`, and Huygens cone geometry ($\theta_c$ and $90^\circ - \theta_c$).
# - `figures/02_field_maps_cone_vs_coulomb.png`: Meridian $E_z(x, 0, z)$
#   snapshots comparing $\beta = 0.90$ (Cherenkov cone) vs $\beta = 0.60$
#   (contracted Coulomb field), wavefront ridge fit, and sub-threshold profile.
# - `figures/03_cone_angle_vs_beta.png`: Measured $\theta_c(\beta)$,
#   $90^\circ - \theta_c(\beta)$, and $\cos\theta_c(\beta)$ vs $1/(\beta n)$
#   across the threshold $\beta_{\text{th}} = 1/n$.
# - `figures/04_frank_tamm_spectrum.png`: Coaxial `FluxMonitor` spectrum vs the
#   analytical Frank–Tamm law and Gaussian bunch form factor $F(\omega)$.
# - `figures/05_dispersive_cherenkov_band.png`: Dispersive Lorentz refractive
#   index $n(\omega)$ and Cherenkov spectrum cut off at the resonance band edge.
# - `figures/06_numerical_cherenkov_yee_vs_tete.png`: Vacuum $\beta = 0.999$
#   spurious numerical Cherenkov radiation in Yee vs dispersion-free TETE and
#   scaling with resolution.
# - `figures/07_resolution_convergence.png`: 3-level convergence of the cone
#   angle and Frank–Tamm spectrum.
# - `figures/cherenkov_cone_beta09.gif`: Animated meridian $E_z$ field as the
#   $\beta = 0.90$ bunch crosses the $n = 1.5$ dielectric with the analytic cone
#   overlaid.
# - `figures/cherenkov_sub_vs_super_threshold.gif`: Side-by-side animation of
#   $\beta = 0.60$ (no cone) vs $\beta = 0.90$ (Cherenkov cone).
# - `figures/numerical_cherenkov_yee_vs_tete.gif`: Side-by-side vacuum
#   $\beta = 0.999$ animation comparing Yee (spurious numerical Cherenkov) vs
#   TETE (dispersion-free).
#
# ---
#
# ## Limits & Failure Modes
#
# 1. **Point-charge ultraviolet catastrophe vs finite bunch length $\sigma_z$**:
#    In a non-dispersive dielectric, the point-charge Frank–Tamm spectrum grows
#    linearly with frequency ($d^2 W/(d\omega\,dz) \propto \omega$) up to the
#    grid Nyquist limit, where under-resolved wavelengths ($\lambda < 6\Delta x$)
#    suffer severe numerical dispersion. A physical Gaussian bunch of RMS length
#    $\sigma_z \gtrsim 2\Delta z$ provides an exponential form-factor roll-off
#    $F(\omega) = \exp(-\omega^2 \sigma_z^2 / v^2)$ that regularizes the spectrum
#    before the Nyquist cutoff.
# 2. **Entrance/exit transition radiation on finite domains**:
#    Turning on a bare line current $J_z$ at the domain boundary without its
#    self-consistent transverse field launches a transition-radiation flash from
#    the entrance plane. Sampling the coaxial `FluxMonitor` over the central
#    steady-state section (away from the entrance/exit CPML) or initializing the
#    discrete Lorentz-contracted comoving self-field isolates the true
#    Cherenkov emission.
# 3. **Near-threshold evanescent tail leakage ($\beta n \lesssim 1$)**:
#    Just below threshold ($\beta n \to 1^-$), the radial decay length of the
#    bound Coulomb field $\gamma_n v / \omega = (1 - \beta^2 n^2)^{-1/2} v/\omega$
#    diverges. A flux surface placed too close to the beam axis ($r_{\text{box}} \lesssim \sigma_z$)
#    can pick up small transient end-cap leakage; keeping $r_{\text{box}} \ge 3\sigma_z$
#    and integrating only the lateral cylindrical/box faces ensures the
#    sub-threshold flux vanishes to numerical precision.
# 4. **Transverse stability constraint of `scheme="tete"`**:
#    The explicit TE/TM scheme (`wakis/tete.py`) is dispersion-free along $z$ at
#    the magic step $c\Delta t = \Delta z$, which requires
#    $\Delta z \le (\Delta x^{-2} + \Delta y^{-2})^{-1/2}$ (for example
#    $\Delta x = \Delta y = 2\Delta z$). Using an isotropic cubic grid
#    ($\Delta x = \Delta y = \Delta z$) at the magic step violates the transverse
#    stability bound and raises a `ValueError` in `SolverFIT3D`. Furthermore,
#    `scheme="tete"` currently supports CPML only on the $z$ boundaries (with
#    PEC, PMC, or periodic transverse walls) and does not couple to ADE
#    dispersive poles.
#
# ---
#
# ## References
#
# 1. P. A. Cherenkov, "Visible emission of clean liquids by action of $\gamma$
#    radiation," *Dokl. Akad. Nauk SSSR* **2**, 451 (1934).
# 2. I. M. Frank and I. E. Tamm, "Coherent visible radiation of fast electrons
#    passing through matter," *C. R. (Dokl.) Acad. Sci. URSS* **14**, 109–114
#    (1937).
# 3. J. D. Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, New York,
#    1999), Chapter 13, Section 13.4 ("Cherenkov Radiation"), Eqs. (13.44)–(13.48).
# 4. M. Dohlus and I. Zagorodnov, "Explicit TE/TM scheme for particle beam
#    simulations," *J. Comput. Phys.* **228**(8), 2822–2833 (2009),
#    `doi:10.1016/j.jcp.2008.12.023` (explicit dispersion-free TE/TM
#    scheme implemented in `wakis/tete.py`).
# 5. B. B. Godfrey, "Numerical Cherenkov instabilities in electromagnetic
#    particle codes," *J. Comput. Phys.* **15**, 504–521 (1974),
#    `doi:10.1016/0021-9991(74)90076-X`.
# 6. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and
#    S. G. Johnson, "Meep: A flexible free-software package for electromagnetic
#    simulations by the FDTD method," *Comput. Phys. Commun.* **181**, 687–702
#    (2010), `doi:10.1016/j.cpc.2009.11.008` (moving-point-source FDTD example
#    that inspired this tutorial).

# %%
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial 8.2: Cherenkov radiation from a moving charge."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run fast coarse validation (< 1 min).")
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
        runners = {
            "dielectric": h.run_task_dielectric,
            "dispersive": h.run_task_dispersive,
            "numerical_cherenkov": h.run_task_numerical_cherenkov,
            "convergence": h.run_task_convergence,
        }
        selected = [args.task] if args.task is not None else list(TASKS)
        for name in selected:
            res = runners[name](quick=quick)
            print(f"[done] task={name}: {json.dumps(res)}")

    if args.plot or args.task is None:
        needed = [
            h.DATA_DIR / f"task_{name}.npz" for name in TASKS
        ]
        if all(p.exists() for p in needed):
            paths = h.render_all_figures()
            for p in paths:
                size_kb = p.stat().st_size / 1024.0
                print(f"[figure] {p.name} ({size_kb:.1f} KB)")

    return 0


if __name__ == "__main__":
    # When executed inside Jupyter / nbconvert, ignore kernel CLI flags
    if any("ipykernel" in arg for arg in sys.argv):
        run_dielectric_section(quick=True)
        run_dispersive_section(quick=True)
        run_numerical_cherenkov_section(quick=True)
        run_convergence_section(quick=True)
        h.render_all_figures()
    else:
        raise SystemExit(main())
