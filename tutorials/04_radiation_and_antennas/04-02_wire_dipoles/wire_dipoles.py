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
# # Tutorial 4.2 · Half-Wave Dipole and Quarter-Wave Monopole Wire Antennas
#
# Whereas Tutorial 4.1 analyzed an electrically short Hertzian dipole ($\ell \ll \lambda_0$) driven
# by an impressed uniform current source (`wakis.sources.CurrentSource`), practical wire antennas
# operate near a **half-wavelength resonance** ($L \approx \lambda_0 / 2$ for a center-fed dipole
# in free space, or $h \approx \lambda_0 / 4$ for a base-fed monopole above a conducting ground
# plane) where the current distribution $I(z)$ is determined self-consistently by the boundary
# condition $E_z = 0$ along the conducting wire surface and the voltage across a discrete feed port
# (`wakis.lumped.LumpedPort`).
#
# Modeling a resonant wire antenna in a 3D time-domain Finite Integration Technique (FIT) solver
# presents a fundamental multi-scale challenge: a physical wire radius $a \sim 0.15\text{–}1.8\;\text{mm}$
# at $f_0 = 1.0\;\text{GHz}$ ($\lambda_0 \approx 300\;\text{mm}$) is two to three orders of magnitude
# smaller than the wavelength. This tutorial demonstrates two complementary ways to bridge that
# scale separation in Wakis:
#
# 1. **Contour-path / in-cell inductance thin-wire subcell model (`wakis.subcell.add_thin_wire`)**:
#    Embeds a wire of arbitrary radius $a < \Delta_\perp / 2$ inside a single line of primary grid
#    edges by scaling the inverse permeability Hodge entries on the four circulating magnetic facets
#    by $1 / m$ with $m = \frac{1}{2}\ln(\Delta_\perp / a)$ and applying degree-weighted radial
#    capacitance/inductance pairing at wire interior and tip nodes ([Umashankar, Beker & Taflove 1987](https://doi.org/10.1109/TAP.1987.1144000);
#    [Noda & Yokoyama 2002](https://doi.org/10.1109/TPWRD.2002.1022813)).
# 2. **Explicitly resolved 3D PEC wire on a locally graded non-uniform mesh**:
#    Places $4 \times 4$ primary cells across the conductor cross-section ($\Delta_{\perp,\text{core}} = 0.76\;\text{mm}$,
#    matched to the circular wire radius $a$ via Lo's conformal equivalence $a_{\text{eq}} = 1.18\,w$)
#    and grades the mesh geometrically outward to $\Delta_{\max} \approx 14\;\text{mm}$ in the outer domain.
#
# ---
#
# ## Learning Goals
#
# 1. **Setup schematic and non-uniform mesh design**: Construct labeled setup schematics (`wakis.schematic.draw_setup`)
#    and mesh views (`wakis.schematic.draw_mesh`) for three canonical wire-antenna geometries:
#    (a) a center-fed $\lambda_0 / 2$ dipole in 6-face CPML free space,
#    (b) a base-fed $\lambda_0 / 4$ monopole over an infinite PEC ground plane (`bc_low=['cpml','cpml','pec']`,
#    `Near2FarMonitor(..., ground_plane='-z')`), and
#    (c) a $\lambda_0 / 4$ monopole mounted at the center of a finite PEC ground sheet (`solver.add_sheet(..., sigma_s='pec')`).
# 2. **Current distribution along the wire**: Extract the axial conduction current $I(z, f_0)$ directly
#    from the 3D FIT magnetic circulation $\oint_{\partial \tilde{A}_z} \mathbf{H}\cdot\mathrm{d}\tilde{\mathbf{s}}$
#    around each wire segment and compare both its amplitude $|I(z)/I(0)|$ and phase $\angle[I(z)/I(0)]$
#    against the classical sinusoidal standing-wave approximation $I(0)\sin[k(L/2 - |z|)]/\sin(kL/2)$
#    and the Hallén moment-method solution (`wakis.analytic.thin_wire.hallen_dipole`).
# 3. **Input impedance $Z_{\text{in}}(f)$, $|S_{11}(f)|$, and resonant length shortening**: Compare the
#    3D FIT `LumpedPort` impedance against [Carter's (1932)](https://doi.org/10.1109/JRPROC.1932.227723)
#    induced-EMF value $Z_{\text{in}}(L = \lambda_0/2, a \to 0) \approx 73.08 + j 42.51\;\Omega$
#    (`wakis.analytic.thin_wire.induced_emf_impedance`) and the [King & Middleton (1946)](https://doi.org/10.1090/qam/15323)
#    / Hallén integral-equation prediction for how finite wire radius $a$ (thickness parameter
#    $\Omega = 2\ln(L/a)$) lowers the series-resonant length $L_{\text{res}} / \lambda_0 < 0.50$.
# 4. **Subcell thin-wire vs. resolved 3D PEC wire and uniform-mesh baseline**: Benchmark the F3
#    subcell thin-wire model against a $4\times 4$-cell resolved PEC wire on a locally graded mesh
#    and a uniform-mesh baseline, preflighted with `wakis.estimate`.
# 5. **Quarter-wave monopole image theory and finite ground plane edge diffraction**: Verify
#    $Z_{\text{in,mono}}(f) = \frac{1}{2} Z_{\text{in,dip}}(f) \approx 36.5 + j 21.3\;\Omega$ and
#    the $+3.01\;\text{dB}$ upper-hemisphere directivity doubling ($5.17\;\text{dBi}$ vs. $2.15\;\text{dBi}$)
#    over an infinite PEC ground plane, and quantify how a finite PEC ground plane of width
#    $W_{\text{gnd}} / \lambda_0 \in \{0.40, 0.80, 1.20\}$ tilts the main lobe upward ($\theta_{\max} < 90^\circ$)
#    and leaks radiation into the lower shadow hemisphere ([Weiner 1987](https://doi.org/10.1109/TAP.1987.1144136)).
#
# ---
#
# ## Physical Theory and Governing Equations
#
# ### 1. Hallén's Integral Equation, Induced-EMF Impedance, and Resonant Shortening
#
# Consider a straight, perfectly conducting cylindrical wire of total length $L$ and radius $a \ll L$
# aligned along the $z$-axis ($z \in [-L/2, +L/2]$) in free space and driven across a narrow feed gap
# at $z = 0$ by a voltage $V_0$ in the $\exp(+j\omega t)$ convention. Enforcing $E_z = 0$ on the wire
# surface $r = a$ yields **Hallén's integral equation** for the axial current $I(z)$
# ([King & Middleton 1946](https://doi.org/10.1090/qam/15323); Balanis 2016, Sec. 8.3):
#
# $$\int_{-L/2}^{+L/2} I(z')\,\frac{e^{-jk\sqrt{(z - z')^2 + a^2}}}{\sqrt{(z - z')^2 + a^2}}\,\mathrm{d}z'
# = C \cos(kz) - j\,\frac{2\pi}{\eta_0}\,V_0 \sin(k|z|), \qquad I(\pm L/2) = 0,$$
#
# where $k = 2\pi / \lambda_0$, $\eta_0 = \sqrt{\mu_0/\varepsilon_0} \approx 376.73\;\Omega$, and the
# constant $C$ is determined by the vanish-at-tip condition $I(\pm L/2) = 0$.
#
# In the **infinitely thin limit** ($a / L \to 0$, so the Hallén–King thickness parameter
# $\Omega = 2\ln(L/a) \to \infty$), the current approaches the standing-wave sinusoidal profile
#
# $$I_{\text{sin}}(z) = I(0)\,\frac{\sin\!\bigl[k(L/2 - |z|)\bigr]}{\sin(kL/2)}, \qquad |z| \le \frac{L}{2}.$$
#
# Integrating the tangential electric field produced by $I_{\text{sin}}(z)$ along the wire surface
# gives [Carter's (1932)](https://doi.org/10.1109/JRPROC.1932.227723) **induced-EMF input impedance**
# (`wakis.analytic.thin_wire.induced_emf_impedance`):
#
# $$Z_{\text{in,EMF}}(L = \lambda_0 / 2,\,a \to 0)
# = \frac{\eta_0}{4\pi}\Bigl[\bigl(\gamma_E + \ln(2\pi) - \operatorname{Ci}(2\pi)\bigr) + j\,\operatorname{Si}(2\pi)\Bigr]
# \approx 73.08 + j\,42.51\;\Omega,$$
#
# and the far-field radiation pattern and directivity:
#
# $$D_{\text{dip}}(\theta) = \frac{4}{\operatorname{Cin}(2\pi)}\left[\frac{\cos\!\bigl(\frac{\pi}{2}\cos\theta\bigr)}{\sin\theta}\right]^2,
# \qquad D_{0,\text{dip}} = \frac{4}{\operatorname{Cin}(2\pi)} \approx 1.641\;\;(2.15\;\text{dBi}),
# \qquad \text{HPBW} \approx 78.0^\circ.$$
#
# Because a dipole of exact physical length $L = \lambda_0 / 2$ has an **inductive reactance**
# $X_{\text{in}} \approx +42.5\;\Omega > 0$ (increased further in the fringing fields of a finite-radius
# wire), series resonance $X_{\text{in}}(f_{\text{res}}) = 0$ occurs at a slightly **shortened electrical
# length** $L_{\text{res}} / \lambda_0 \approx 0.46\text{–}0.48 < 0.50$. Asymptotic expansion of Hallén's
# equation in $\Omega^{-1} = [2\ln(L/a)]^{-1}$ ([King & Middleton 1946](https://doi.org/10.1090/qam/15323))
# shows that $0.50 - L_{\text{res}}/\lambda_0 \propto \Omega^{-1}$: thicker wires (smaller $\Omega$)
# experience stronger capacitive fringing near the tips and resonate at a shorter physical length.
#
# ### 2. Subcell Thin-Wire Hodge Model (`wakis.subcell`) vs. Resolved Wire
#
# Near a wire of subcell radius $a < \Delta_\perp / 2$ aligned with a primary $z$-edge, the azimuthal
# magnetic field and radial electric field vary quasi-statically as $H_\phi(r) \propto 1/r$ and
# $E_r(r) \propto 1/r$ for $r \in [a, \Delta_\perp]$. Integrating the magnetic flux across a primary
# facet of width $\Delta_\perp$ yields the contour-path inductance factor
# ([Umashankar, Beker & Taflove 1987](https://doi.org/10.1109/TAP.1987.1144000);
# [Noda & Yokoyama 2002](https://doi.org/10.1109/TPWRD.2002.1022813)):
#
# $$\Phi_B = \mu_0 \int_0^{\Delta_\perp} H_\phi(r)\,\mathrm{d}r\,\Delta z
# \approx \mu_0\,\frac{\ln(\Delta_\perp / a)}{2}\,H_\phi(\Delta_\perp / 2)\,\Delta_\perp \Delta z
# \implies \mu_{\text{eff}} = m\,\mu_0, \qquad m = \frac{1}{2}\ln\!\left(\frac{\Delta_\perp}{a}\right).$$
#
# `solver.add_thin_wire(..., radius=a, gap=port)` divides the inverse permeability Hodge entries `imu`
# on the four circulating magnetic facets around each wire segment by $m$ and scales the radial
# electric edges at wire nodes of degree $d$ ($d=2$ at interior nodes, $d=1$ at wire tips) by
# $w_{\text{node}} = (2/d)\,m$ with paired permeability scaling to preserve the 3D Courant stability limit.
#
# When we instead resolve the conductor explicitly on a locally graded mesh using $4 \times 4$ primary
# cells across a square PEC cross-section of full width $2w = 4\Delta_{\perp,\text{core}}$, conformal
# mapping of the exterior Laplace problem ([Lo 1953](https://doi.org/10.1063/1.1721192); Balanis 2016,
# Table 9.3) gives the **equivalent circular radius**
#
# $$a_{\text{eq}} = 0.59\,(2w) = 1.18\,w.$$
#
# Matching $w = a / 1.18$ allows a direct, parameter-free comparison between the $4\times 4$-cell
# resolved PEC wire and the single-edge subcell thin-wire model `add_thin_wire(radius=a)`.
#
# ### 3. Quarter-Wave Monopole Image Theory and Finite Ground Plane Diffraction
#
# A vertical monopole of height $h = L / 2 = \lambda_0 / 4$ mounted at $z = 0$ over an **infinite PEC
# ground plane** ($E_x = E_y = 0$ at $z = 0$) creates a virtual mirror image extending from $z = 0$
# to $z = -h$ with the same vertical current $I_{\text{image}}(-z) = +I_{\text{mono}}(+z)$
# ([Maloney, Smith & Scott 1990](https://doi.org/10.1109/8.55618); Balanis 2016, Sec. 4.7). For the same
# base current $I(0)$, the monopole terminal voltage between $z = 0$ and the base gap is half the
# dipole terminal voltage, and electromagnetic energy is radiated solely into the upper hemisphere
# ($0 \le \theta \le \pi/2$):
#
# $$Z_{\text{in,mono}}(f) = \frac{1}{2}\,Z_{\text{in,dip}}(f) \approx 36.54 + j\,21.26\;\Omega\;\;(a \to 0),
# \qquad D_{0,\text{mono}} = 2\,D_{0,\text{dip}} \approx 3.282\;\;(5.16\;\text{dBi},\;\;+3.01\;\text{dB}).$$
#
# When the ground plane has a **finite width** $W_{\text{gnd}} \sim 0.4\text{–}1.2\,\lambda_0$ in free
# space (`solver.add_sheet(..., sigma_s='pec')`), the radial surface current $\mathbf{J}_s$ on the
# ground plane diffracts at the sharp perimeter $x, y = \pm W_{\text{gnd}}/2$
# ([Weiner 1987](https://doi.org/10.1109/TAP.1987.1144136)). Constructive interference between the
# direct monopole wave and the edge-diffracted wave tilts the main lobe upward away from the horizon
# ($\theta_{\max} < 90^\circ$), reduces the horizon directivity $D(90^\circ)$ by $3\text{–}5\;\text{dB}$,
# and radiates a significant fraction of power into the lower shadow hemisphere ($\theta > 90^\circ$).

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    F0_HZ,
    H_QUARTER_M,
    L_HALF_M,
    LAM0_M,
    TASKS,
    generate_all_figures,
    preflight_estimates,
    print_reference_table,
    run_tasks,
)

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"


def _show_figure(path: Path) -> None:
    """Display a saved PNG in notebook mode while keeping headless runs quiet."""
    if "ipykernel" not in sys.modules or not path.exists() or path.suffix.lower() != ".png":
        return
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=120)
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %% [markdown]
# ## 1. Setup Schematics, Non-Uniform Meshing, and `wakis.estimate` Preflight
#
# Before running any time-domain steps, we inspect the analytical thin-wire and monopole benchmarks
# and call `wakis.estimate` across four candidate discretizations:
#
# - `graded_subcell`: $31 \times 31 \times 39 = 37,479$ cells ($\Delta_{\min} = 7.9\;\text{mm}$,
#   $\Delta_{\max} = 16.6\;\text{mm}$), resolving the wire axially with $N_{\text{seg}} = 19$ uniform
#   segments while embedding the wire radius $a < \Delta_\perp/2$ via `solver.add_thin_wire`;
# - `graded_resolved`: $31 \times 31 \times 35 = 33,635$ cells ($\Delta_{\perp,\text{core}} = 0.76\;\text{mm}$,
#   $4 \times 4$ cells across the conductor cross-section $2w = 3.05\;\text{mm}$, graded geometrically
#   to $\Delta_{\max} = 13.7\;\text{mm}$);
# - `uniform_subcell_cpw24`: $48 \times 48 \times 48 = 110,592$ uniform cells at $\Delta = 12.5\;\text{mm}$
#   ($24\;\text{cells}/\lambda_0$, $3\times$ more cells than `graded_subcell` with coarser axial wire resolution);
# - `uniform_resolved_4cell`: a uniform grid at $\Delta = 0.76\;\text{mm}$ across the $(0.49\;\text{m})^3$
#   domain would require $(646)^3 \approx 2.70 \times 10^8$ cells and $\sim 65\;\text{GB}$ of RAM.
#
# Figure `00_setup_schematic.png` shows the three simulation setups side by side:
# (a) the center-fed $\lambda_0/2$ dipole in 6-face CPML free space,
# (b) the base-fed $\lambda_0/4$ monopole over an infinite PEC ground plane (`bc_low=['cpml','cpml','pec']`,
# `Near2FarMonitor(..., ground_plane='-z')`), and
# (c) the $\lambda_0/4$ monopole on a finite PEC ground sheet (`solver.add_sheet(..., sigma_s='pec')`)
# in 6-face CPML. Figure `01_geometry_and_graded_mesh.png` shows the graded $xz$ mesh, the zoomed
# $xy$ wire core comparing the $4\times 4$-cell resolved PEC cross-section against the subcell thin-wire
# magnetic circulation loop, and the non-uniform cell-size profile $\Delta x(x)$.
#
# ![Setup schematics for free-space dipole, infinite-ground monopole, and finite-ground monopole](figures/00_setup_schematic.png)
#
# ![Graded xz mesh, zoomed xy wire core, and transverse cell grading profile](figures/01_geometry_and_graded_mesh.png)

# %%
if "ipykernel" in sys.modules:
    print_reference_table()
    preflight_estimates()
    if not (DATA_DIR / "task_half_wave_thin_wire_radii.npz").exists():
        run_tasks(list(TASKS), DATA_DIR, quick=True)
    if not (FIG_DIR / "00_setup_schematic.png").exists():
        generate_all_figures(DATA_DIR, FIG_DIR)
    _show_figure(FIG_DIR / "00_setup_schematic.png")
    _show_figure(FIG_DIR / "01_geometry_and_graded_mesh.png")


# %% [markdown]
# ## 2. Wire Current Distribution $I(z)$, Input Impedance $Z_{\text{in}}(f)$, and Resonant Shortening
#
# By integral Ampère's law on the dual grid, the total conduction current $I(z_{k+1/2}, t)$ on each
# PEC wire segment $k$ (where $E_z = 0$, so the axial displacement current vanishes identically) equals
# the discrete circulation of the four surrounding magnetic field components:
#
# $$I(z_{k+1/2}, t) = \oint_{\partial \tilde{A}_z(i_0, j_0, k)} \mathbf{H}\cdot\mathrm{d}\tilde{\mathbf{s}}
# = \tilde{L}_x\bigl[H_x(i_0, j_0-1, k) - H_x(i_0, j_0, k)\bigr]
# + \tilde{L}_y\bigl[H_y(i_0, j_0, k) - H_y(i_0-1, j_0, k)\bigr].$$
#
# Figure `02_current_distribution_wire.png` compares the 3D FIT current profile against both the
# classical sinusoidal standing wave $\sin[k(L/2 - |z|)]$ and the Hallén moment-method solution:
#
# - **Amplitude $|I(z)/I(0)|$ (panel a)**: Both 3D FIT and Hallén MoM follow the $\sin[k(L/2 - |z|)]$
#   envelope closely from the feed gap ($|I/I(0)| = 1$) to the wire tips ($I(\pm L/2) = 0$).
# - **Phase progression $\angle[I(z)/I(0)]$ (panel b)**: Whereas the zero-radius sinusoidal model
#   assumes a pure standing wave with zero phase shift along the arm, a wire of finite radius
#   ($a = 0.5\;\text{mm}$, $\Omega = 11.4$) carries a complex current whose phase lags toward the
#   center to supply the radiated power; 3D FIT reproduces the Hallén MoM phase curve across the arm.
# - **Monopole vs. dipole upper arm (panel c)**: The current distribution $|I_{\text{mono}}(z)|$ on
#   the $\lambda_0/4$ monopole ($z \in [0, \lambda_0/4]$) overlays the upper arm of the center-fed
#   $\lambda_0/2$ dipole.
#
# ![Current distribution amplitude and phase along the dipole and quarter-wave monopole](figures/02_current_distribution_wire.png)
#
# Figure `03_impedance_s11_and_shortening.png` shows the frequency-dependent input impedance
# $Z_{\text{in}}(f) = R_{\text{in}}(f) + j X_{\text{in}}(f)$ and reflection coefficient $|S_{11}(f)|$:
#
# - At the nominal half-wave length $L = 0.50\lambda_0$ ($f = f_0 = 1.0\;\text{GHz}$), the input
#   impedance is inductive ($X_{\text{in}} > 0$), approaching Carter's induced-EMF value
#   $73.08 + j 42.51\;\Omega$ as the wire radius $a$ decreases.
# - Thinner wires ($a = 0.15\;\text{mm}$, $\Omega = 13.8$ vs. $a = 1.5\;\text{mm}$, $\Omega = 9.2$)
#   exhibit a steeper reactance slope $\mathrm{d}X_{\text{in}}/\mathrm{d}f$ (higher antenna $Q$) and
#   a resonant frequency $f_{\text{res}}$ closer to $f_0$, matching the [King & Middleton (1946)](https://doi.org/10.1090/qam/15323)
#   shortening curve $L_{\text{res}} / \lambda_0$ in panel (c).
# - Trimming the physical wire length to $L_{\text{trim}} = 0.470\lambda_0$ cancels the inductive
#   reactance at $1.0\;\text{GHz}$ ($X_{\text{in}}(f_0) \approx 0\;\Omega$, $R_{\text{in}}(f_0) \approx 66\;\Omega$),
#   shifting the $|S_{11}|$ return-loss dip squarely onto $f_0 = 1.0\;\text{GHz}$ (panel b).
#
# ![Input impedance, return loss S11, and resonant length shortening vs thickness parameter Omega](figures/03_impedance_s11_and_shortening.png)

# %%
if "ipykernel" in sys.modules:
    _show_figure(FIG_DIR / "02_current_distribution_wire.png")
    _show_figure(FIG_DIR / "03_impedance_s11_and_shortening.png")


# %% [markdown]
# ## 3. Subcell Thin-Wire (`wakis.subcell`) vs. Resolved $4\times 4$-Cell Wire and Mesh Convergence
#
# Figure `04_subcell_vs_resolved_convergence.png` compares four 3D FIT models of the same dipole
# ($L = 0.48\lambda_0 = 143.9\;\text{mm}$, radius $a = 1.80\;\text{mm}$):
#
# 1. **Resolved $4\times 4$-cell PEC wire on a locally graded mesh** ($\Delta_{\perp,\text{core}} = 0.76\;\text{mm}$,
#    $4$ cells across $2w = 2a/1.18 = 3.05\;\text{mm}$);
# 2. **Subcell thin-wire model (`solver.add_thin_wire`, `correction=True`) on a graded mesh**
#    ($\Delta_{\perp,\text{core}} = 8.46\;\text{mm} > 2a$, $11\times$ larger transverse cell and timestep);
# 3. **Uniform-mesh subcell baseline** ($\Delta = 11.1\;\text{mm}$ everywhere); and
# 4. **Uncorrected single-edge wire (`correction=False`)**, which omits the $\frac{1}{2}\ln(\Delta_\perp / a)$
#    magnetic Hodge scaling and therefore reverts to the artificial numerical radius $a_0 \approx 0.23\,\Delta_\perp$
#    of the grid cell, shifting the reactance curve by $> 25\;\Omega$.
#
# Across the entire $0.80\text{–}1.15\;\text{GHz}$ band, the subcell thin-wire model on the graded mesh
# matches the $4\times 4$-cell resolved PEC wire to within $\sim 3\%$ while running $\sim 7\times$ faster
# (and using $7,800\times$ fewer cells than a uniform mesh that resolves $\Delta_{\perp,\text{core}} = 0.76\;\text{mm}$
# everywhere).
#
# ![Subcell thin-wire vs resolved 4x4-cell PEC wire impedance, axial convergence, and mesh cost](figures/04_subcell_vs_resolved_convergence.png)

# %%
if "ipykernel" in sys.modules:
    _show_figure(FIG_DIR / "04_subcell_vs_resolved_convergence.png")


# %% [markdown]
# ## 4. Quarter-Wave Monopole Over Infinite vs. Finite PEC Ground Planes
#
# Figure `05_monopole_vs_dipole_image_theory.png` validates image theory for a quarter-wave monopole
# ($h = \lambda_0 / 4 = 74.95\;\text{mm}$) over an infinite PEC ground plane at $z = 0$
# (`bc_low=['cpml','cpml','pec']`, `solver.add_sheet(plane=('z', 0.0), sigma_s='pec')`,
# `Near2FarMonitor(..., ground_plane='-z')`):
#
# - **Impedance halving (panel a)**: Across the entire $0.80\text{–}1.15\;\text{GHz}$ band, the
#   monopole input impedance satisfies $Z_{\text{in,mono}}(f) = \frac{1}{2} Z_{\text{in,dip}}(f)$
#   to within $0.07\%$ ($44.01 + j 20.05\;\Omega$ vs. $\frac{1}{2}(88.08 + j 40.09)\;\Omega$ at
#   $f_0$ for $a = 1.0\;\text{mm}$, and approaching $36.5 + j 21.3\;\Omega$ for $a = 0.2\;\text{mm}$).
# - **Directivity doubling (panel c)**: Because the PEC ground plane reflects all downward waves into
#   the upper hemisphere ($0^\circ \le \theta \le 90^\circ$), the radiated power for a given horizon
#   field is halved and the directivity doubles from $D_{0,\text{dip}} = 1.648$ ($2.17\;\text{dBi}$)
#   to $D_{0,\text{mono}} = 3.317$ ($5.21\;\text{dBi}$, a $+3.04\;\text{dB}$ increase vs. the exact
#   $+3.01\;\text{dB}$).
#
# ![Quarter-wave monopole vs half-wave dipole impedance, S11, and +3 dB upper-hemisphere directivity](figures/05_monopole_vs_dipole_image_theory.png)
#
# Figures `06_near_field_maps_dipole_monopole.png`, `07_far_field_3d_and_finite_ground.png`, and
# `08_wire_antenna_radiation.gif` compare the near- and far-field radiation when the infinite PEC
# boundary is replaced by a **finite square PEC ground sheet** of width $W_{\text{gnd}} / \lambda_0 \in \{0.40, 0.80, 1.20\}$
# in free space (`solver.add_sheet(center=(0,0,0), size=(W_gnd, W_gnd, 0), sigma_s='pec')`):
#
# - For a small ground plane ($W_{\text{gnd}} = 0.40\lambda_0$), $47.6\%$ of the radiated power leaks
#   into the lower shadow hemisphere ($\theta > 90^\circ$) and $R_{\text{in}}(f_0)$ drops to $27.8\;\Omega$.
# - For $W_{\text{gnd}} = 0.80\lambda_0$ and $1.20\lambda_0$, edge diffraction from the ground-plane
#   perimeter tilts the main lobe upward off the horizon to $\theta_{\max} \approx 53^\circ\text{–}54^\circ$
#   ($36^\circ\text{–}37^\circ$ elevation above the ground plane, in agreement with
#   [Weiner 1987](https://doi.org/10.1109/TAP.1987.1144136)), reduces the horizon directivity $D(90^\circ)$
#   from $3.32$ ($5.21\;\text{dBi}$) to $\sim 1.02\text{–}1.06$ ($0.1\text{–}0.3\;\text{dBi}$), and
#   leaves $26\text{–}31\%$ of the total power in the lower hemisphere.
#
# ![Near-field magnitude maps for the free-space half-wave dipole and finite-ground quarter-wave monopole](figures/06_near_field_maps_dipole_monopole.png)
#
# ![3D far-field directivity torus and finite PEC ground plane edge-diffraction study](figures/07_far_field_3d_and_finite_ground.png)
#
# ![Animated time-domain radiation of the center-fed half-wave dipole](figures/08_wire_antenna_radiation.gif)

# %%
if "ipykernel" in sys.modules:
    _show_figure(FIG_DIR / "05_monopole_vs_dipole_image_theory.png")
    _show_figure(FIG_DIR / "06_near_field_maps_dipole_monopole.png")
    _show_figure(FIG_DIR / "07_far_field_3d_and_finite_ground.png")


# %% [markdown]
# ## 5. Limits & Failure Modes
#
# 1. **Subcell thin-wire radius ceiling ($a < 0.5\,\Delta_\perp$)**:
#    The Umashankar–Taflove / Noda–Yokoyama contour-path factor $m = \frac{1}{2}\ln(\Delta_\perp / a)$
#    assumes that the azimuthal magnetic field at the dual-edge location $r = \Delta_\perp / 2$ obeys
#    the cylindrical $1/r$ law and that $m > 0$. As $a \to \Delta_\perp / 2$, $\ln(\Delta_\perp / a) \to \ln 2 \approx 0.693$
#    ($m < 0.35$), and for $a \ge 0.5\,\Delta_\perp$ the wire surface extends beyond the circulating
#    $H$ sample locations. `solver.add_thin_wire` raises a `ValueError` whenever $a \ge 0.5\,\Delta_\perp$;
#    thick conductors with $a \gtrsim 0.35\,\Delta_\perp$ should be meshed explicitly with $\ge 4$ cells
#    across the diameter on a locally graded grid (Section 3).
# 2. **Omitting the $\ln(\Delta_\perp / a)$ Hodge correction (`correction=False`)**:
#    Zeroing $E_z$ along a line of primary grid edges without scaling the surrounding `imu` entries
#    forces the wire to take on the intrinsic numerical radius of the grid cell,
#    $a_0 \approx 0.23\,\Delta_\perp$ ([Noda & Yokoyama 2002](https://doi.org/10.1109/TPWRD.2002.1022813)).
#    As shown in Figure `04_subcell_vs_resolved_convergence.png`(a), this eliminates any dependence on
#    the physical wire radius $a$ and makes the computed impedance grid-dependent.
# 3. **Thin-wire termination on a PEC boundary plane**:
#    When a thin wire starts on a PEC domain boundary (`bc_low[2] = 'pec'` at $z = 0$ for a monopole),
#    the tangential electric edges $E_x, E_y$ on $z = 0$ must have `ieps = 0` (`solver.add_sheet(plane=('z', 0.0), sigma_s='pec')`)
#    before `solver.add_thin_wire` is called. Otherwise `add_thin_wire` treats the ground-plane junction
#    as an open-space wire tip (`deg = 1`) and applies free-space tip capacitance/inductance scaling to
#    the base feed cell, doubling the base inductance and breaking the image-theory identity
#    $Z_{\text{in,mono}} = \frac{1}{2} Z_{\text{in,dip}}$.
# 4. **Feed-gap capacitance (`gap` dependence of $B_{\text{in}}$)**:
#    In both 3D FIT (`LumpedPort` across a gap of axial length $g = \Delta z_w$) and Hallén's integral
#    equation (`hallen_dipole(..., gap=g)`), a delta-gap generator ($g \to 0$) has a logarithmically
#    divergent local gap susceptance $B_{\text{gap}} \propto \varepsilon_0 a \ln(a / g)$. Meaningful
#    reactance comparisons require matching the physical feed-gap width $g = \Delta z_w$ between the
#    3D FIT port and the reference model.
#
# ---
#
# ## References
#
# 1. P. S. Carter, "Circuit Relations in Radiating Systems and Applications to Antenna Problems,"
#    *Proceedings of the Institute of Radio Engineers*, vol. 20, no. 6, pp. 1004–1041, 1932,
#    [doi:10.1109/JRPROC.1932.227723](https://doi.org/10.1109/JRPROC.1932.227723).
# 2. R. King and D. Middleton, "The cylindrical antenna; current and impedance,"
#    *Quarterly of Applied Mathematics*, vol. 3, no. 4, pp. 302–335, 1946,
#    [doi:10.1090/qam/15323](https://doi.org/10.1090/qam/15323).
# 3. Y. T. Lo, "A Note on the Cylindrical Antenna of Noncircular Cross Section,"
#    *Journal of Applied Physics*, vol. 24, no. 9, pp. 1222–1225, 1953,
#    [doi:10.1063/1.1721192](https://doi.org/10.1063/1.1721192).
# 4. K. R. Umashankar, B. Beker, and A. Taflove, "Calculation and experimental validation of induced
#    currents on coupled wires in an arbitrary shaped cavity," *IEEE Transactions on Antennas and
#    Propagation*, vol. 35, no. 11, pp. 1248–1257, 1987,
#    [doi:10.1109/TAP.1987.1144000](https://doi.org/10.1109/TAP.1987.1144000).
# 5. M. M. Weiner, "Monopole element at the center of a circular ground plane whose radius is small or
#    comparable to a wavelength," *IEEE Transactions on Antennas and Propagation*, vol. 35, no. 5,
#    pp. 488–495, 1987, [doi:10.1109/TAP.1987.1144136](https://doi.org/10.1109/TAP.1987.1144136).
# 6. J. G. Maloney, G. S. Smith, and W. R. Scott, Jr., "Accurate computation of the radiation from
#    simple antennas using the finite-difference time-domain method," *IEEE Transactions on Antennas
#    and Propagation*, vol. 38, no. 7, pp. 1059–1068, 1990,
#    [doi:10.1109/8.55618](https://doi.org/10.1109/8.55618).
# 7. T. Noda and S. Yokoyama, "Thin wire representation in finite difference time domain surge
#    simulation," *IEEE Transactions on Power Delivery*, vol. 17, no. 3, pp. 840–846, 2002,
#    [doi:10.1109/TPWRD.2002.1022813](https://doi.org/10.1109/TPWRD.2002.1022813).
# 8. C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley, 2016, Chs. 4, 8, and 9.

# %%
def _print_summary(data_dir: Path) -> None:
    """Print a concise quantitative validation summary from the saved task archives."""
    d0 = np.load(data_dir / "task_half_wave_thin_wire_radii.npz")
    d1 = np.load(data_dir / "task_resolved_wire_vs_subcell_mesh.npz")
    d2 = np.load(data_dir / "task_quarter_wave_monopole_infinite_gnd.npz")
    d3 = np.load(data_dir / "task_finite_ground_plane_sweep.npz")

    f_ghz = np.asarray(d0["freqs_ghz"], dtype=float)
    idx_f0 = int(np.argmin(np.abs(f_ghz - 1.0)))
    print("\n=== Tutorial 4.2 Validation Summary (3D FIT vs. Analytical & Hallén MoM) ===")
    for i, (r_mm, om) in enumerate(zip(d0["radii_mm"], d0["omegas"])):
        zf = complex(d0["z_fit_real"][i, idx_f0], d0["z_fit_imag"][i, idx_f0])
        zm = complex(d0["z_mom_real"][i, idx_f0], d0["z_mom_imag"][i, idx_f0])
        print(
            f"  Dipole a = {r_mm:.2f} mm (Omega = {om:.1f}) : Z_FIT(f0) = {zf.real:.1f}+j{zf.imag:.1f} Ohm "
            f"(MoM {zm.real:.1f}+j{zm.imag:.1f}), L_res/lam0 = {d0['l_res_fit_over_lam0'][i]:.4f} "
            f"(MoM {d0['l_res_mom_over_lam0'][i]:.4f})"
        )
    print(
        f"  Dipole far field (a=0.5 mm)     : D0 = {float(d0['d0_fit']):.4f} "
        f"({10*np.log10(float(d0['d0_fit'])):.2f} dBi vs 2.15 dBi), HPBW = {float(d0['hpbw_deg']):.1f} deg"
    )
    print(
        f"  Subcell vs. 4x4 Resolved wire   : rel L2 diff = {100*float(d1['rel_diff_res_vs_sub']):.2f}%, "
        f"wall = {float(d1['wall_sub_s']):.1f}s vs {float(d1['wall_res_s']):.1f}s"
    )
    print(
        f"  Monopole image theory (inf PEC) : max |2*Z_mono - Z_dip|/|Z_dip| = {100*float(d2['image_z_rel_err']):.3f}%, "
        f"D0_mono = {float(d2['d0_mono']):.4f} ({10*np.log10(float(d2['d0_mono'])):.2f} dBi, +{float(d2['d0_ratio_db']):.2f} dB)"
    )
    for w_lam, th_m, dh, bp in zip(
        d3["w_over_lam_list"], d3["theta_max_deg"], d3["d_horizon_lin"], d3["back_power_frac"]
    ):
        print(
            f"  Finite PEC ground W = {w_lam:.2f}*lam0 : peak at theta_max = {th_m:.1f} deg "
            f"(tilt {90.0 - th_m:.1f} deg), D(90 deg) = {dh:.2f}, shadow power = {100*bp:.1f}%"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standard Wakis tutorial CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Tutorial 4.2: Half-Wave Dipole and Quarter-Wave Monopole Wire Antennas"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run reduced-grid smoke tasks (~12 s)")
    mode.add_argument("--full", action="store_true", help="Run full validation tasks (~35 s)")
    parser.add_argument(
        "--task",
        choices=TASKS,
        default=None,
        help="Run only a single named task",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available task names and exit",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures from existing data/*.npz without re-running FDTD",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Tutorial 4.2."""
    args = parse_args(argv)
    if args.list_tasks:
        for name in TASKS:
            print(name)
        return

    print_reference_table()
    preflight_estimates()

    if args.plot:
        saved = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"\nRegenerated {len(saved)} figures in {FIG_DIR}:")
        for p in saved:
            print(f"  - {p.relative_to(TUTORIAL_DIR)} ({p.stat().st_size / 1024:.1f} KB)")
        _print_summary(DATA_DIR)
        return

    quick = True if not args.full else False
    if args.quick:
        quick = True

    selected = [args.task] if args.task else list(TASKS)
    print(f"\nRunning Tutorial 4.2 tasks ({'QUICK' if quick else 'FULL'} mode): {selected}")
    run_tasks(selected, DATA_DIR, quick=quick)

    if args.task is None:
        saved = generate_all_figures(DATA_DIR, FIG_DIR)
        print(f"\nGenerated {len(saved)} figures in {FIG_DIR}:")
        for p in saved:
            print(f"  - {p.relative_to(TUTORIAL_DIR)} ({p.stat().st_size / 1024:.1f} KB)")
        _print_summary(DATA_DIR)


if __name__ == "__main__":
    main()
