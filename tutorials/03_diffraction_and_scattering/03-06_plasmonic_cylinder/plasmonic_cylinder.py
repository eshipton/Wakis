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

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/03_diffraction_and_scattering/03-06_plasmonic_cylinder/plasmonic_cylinder.ipynb)
#
# # Tutorial 3.6 · Localized surface plasmons of a metal cylinder: pole count, accuracy and cost
# <h3 align="center">Dispersive pole models, error separation, and per-pole cost in the Finite Integration Technique</h3>
# <p align="center"><i>Erik Shipton &middot; Wakis tutorial series (fork of CERN's Wakis)</i></p>
#
# ---
#
# ### Abstract
#
# **How many Lorentz/Drude poles does a silver model need for an accurate
# localized surface plasmon resonance (LSPR) of a 2D cylinder, and what does
# each pole cost per time step? We illuminate a silver cylinder (diameter
# $2a = 40\text{ nm}$, plus $80\text{ nm}$ to show retardation, and a gold
# cylinder of $2a = 40\text{ nm}$ as a second material) in vacuum with a
# broadband plane-wave pulse ($300\text{--}900\text{ nm}$) using the in-plane
# electric-field polarization (`TE` in Wakis). For each metal we fit passive,
# stable $N = 1, 2, 3, 4, 6$ pole models to the public-domain (CC0) Johnson &
# Christy (1972) optical constants with `wakis.dispersive.fit_poles`, alongside
# the 6-pole Rakić et al. (1998) library model and a non-dispersive
# constant-$\varepsilon(\lambda_\mathrm{res})$ baseline. By comparing Wakis FIT
# against exact 2D Mie theory (`wakis.analytic.mie_cylinder.cylinder_efficiencies`)
# evaluated with both the tabulated Johnson & Christy data and each fitted model
# (continuous $\varepsilon(\omega)$ and discrete $\varepsilon_\mathrm{disc}(\omega, \Delta t)$),
# we separate *model error* from *discretization error* at two grid resolutions
# ($h = 4\text{ nm}$ and $h = 2\text{ nm}$, Kottke subpixel averaging via
# `wakis.tensor_dispersive`). Finally, we benchmark the single-thread CPU cost
# and memory per dispersive edge as a function of $N$ for both the cylinder and
# a worst-case bulk-filled domain.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to
#
# 1. **derive** the quasi-static localized surface plasmon resonance condition
#    $\mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) = -1$ of a 2D cylinder in
#    vacuum for the in-plane polarization (`polarization="TE"` in Wakis), and
#    explain its radiative broadening and retardation red-shift as the diameter
#    grows from $40\text{ nm}$ to $80\text{ nm}$;
# 2. **explain** why a time-domain code cannot step a static negative
#    permittivity ($\varepsilon_r < 0$ is unconditionally unstable in leapfrog
#    time stepping) and how auxiliary differential equations (ADEs) integrate
#    Drude and Lorentz poles stably and passively;
# 3. **contrast** the interband transitions of silver ($4d \to 5sp$ onset near
#    $3.9\text{ eV} \approx 320\text{ nm}$, above the $341\text{ nm}$ plasmon)
#    and gold ($5d \to 6sp$ onset near $2.4\text{ eV} \approx 517\text{ nm}$,
#    overlapping the $496\text{ nm}$ plasmon), and explain why silver converges
#    at $N = 2$ poles whereas gold requires $N \ge 3\text{--}6$ poles;
# 4. **fit** passive multi-pole Drude--Lorentz models to tabulated $(n, k)$
#    data with `wakis.dispersive.fit_poles` and verify discrete-time stability
#    with `is_stable` and `epsilon_discrete`;
# 5. **separate** *model error* ($\text{Mie}(\varepsilon_\mathrm{fit})$ vs
#    $\text{Mie}(\varepsilon_\mathrm{J\&C})$) from *discretization error*
#    ($\text{FIT}$ vs $\text{Mie}(\varepsilon_\mathrm{fit})$) and *total error*
#    ($\text{FIT}$ vs $\text{Mie}(\varepsilon_\mathrm{J\&C})$) across pole
#    count $N$ and mesh resolution $h$;
# 6. **quantify** the CPU time per step, slowdown relative to a named
#    non-dispersive baseline on the same grid, marginal cost per dispersive
#    edge per pole, and auxiliary-array memory per dispersive edge.
#
# ### Contents
# 0. Set-up (Colab install, CLI flags, run modes)
# 1. The physics: 2D localized surface plasmons, retardation, and time-domain ADEs
# 2. Pole fitting: $N = 1, 2, 3, 4, 6$ vs Johnson & Christy and Rakić (Ag and Au)
# 3. Geometry, skin-depth mesh rules, Kottke subpixel averaging, and preflight
# 4. Running the 2D plasmonic cylinder and inspecting the near field
# 5. Extinction spectra $Q_\mathrm{ext}(\lambda)$: FIT vs Mie, constant-$\varepsilon$ baseline, and $80\text{ nm}$ retardation
# 6. Separating model error, discretization error, and total error across $N$ and $h$
# 7. Cost per pole: time per step, per-edge cost, stability $\Delta t_\mathrm{max}$, and memory
# 8. Limits and failure modes
# 9. Conclusions
# 10. References

# %% [markdown]
# ## 0. Set-up
#
# On **Google Colab** the next cell installs the Wakis fork from GitHub and
# `xvfb` for off-screen rendering. Locally, run this script from the repository
# root (always under `bin/memguard` on shared/cloud machines):
#
# ```bash
# bin/memguard -m 6000 -- python tutorials/03_diffraction_and_scattering/03-06_plasmonic_cylinder/plasmonic_cylinder.py --quick
# bin/memguard -m 6000 -- python tutorials/03_diffraction_and_scattering/03-06_plasmonic_cylinder/plasmonic_cylinder.py --plot
# bin/memguard -m 6000 -- python tutorials/03_diffraction_and_scattering/03-06_plasmonic_cylinder/plasmonic_cylinder.py --full
# ```

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run("apt-get -qq install -y xvfb > /dev/null", shell=True, check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "wakis @ git+https://github.com/eshipton/Wakis.git@main"],
        check=True,
    )  # fmt: skip

# %%
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps_0

import wakis
import wakis.figstyle as figstyle
import wakis.schematic as schematic
from wakis import visualize as wv
from wakis.analytic.mie_cylinder import (
    cylinder_efficiencies,
    cylinder_rayleigh_efficiencies,
)
from wakis.benchmarks.meshing import cluster_mesh
from wakis.benchmarks.metrics import nrms, peak
from wakis.dispersive import (
    EV,
    DispersiveMaterial,
    Drude,
    Lorentz,
    dispersive_lib,
    eps_from_nk,
    fit_poles,
)
from wakis.estimate import estimate, preflight
from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D

print(wv.setup_rendering())
figstyle.apply_style()

IN_NOTEBOOK = "ipykernel" in sys.modules
parser = argparse.ArgumentParser(
    description="Wakis tutorial 02: localized surface plasmons of a 2D metal cylinder"
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument(
    "--quick",
    action="store_true",
    help="run quick verification (Ag 40nm N=2 at h=4nm) and generate all plots (~15 s)",
)
mode.add_argument(
    "--full",
    action="store_true",
    help="run full sweep (fits, h=4nm & h=2nm spectra, single-thread timing) and plots",
)
mode.add_argument(
    "--plot",
    action="store_true",
    help="generate all figures and tables from committed JSON data in data/",
)
parser.add_argument(
    "--task",
    choices=["all", "fit", "spectra", "timing", "plot"],
    default="all",
    help="sub-task to execute in --full mode",
)
parser.add_argument("--gpu", action="store_true", help="use the GPU")
parser.add_argument("--cpu", action="store_true", help="force the CPU")
parser.add_argument("--resolution", type=float, default=None, help="cell size h [nm]")
parser.add_argument("--conformal", default="all", help="staircase | all")
parser.add_argument("--metal", choices=["Ag", "Au"], default="Ag", help="metal for --arm")
parser.add_argument("--poles", default="N2", help="model key (N1,N2,N3,N4,N6,Rakic)")
parser.add_argument("--diameter", type=float, default=40.0, help="cylinder diameter [nm]")
parser.add_argument("--arm", action="store_true", help="study arm: one run, JSON out")
parser.add_argument("--out", default=None, help="JSON output path for --arm")
NB_ARGS = os.environ.get("WAKIS_TUTORIAL_ARGS", "").split()
ARGS, _ = parser.parse_known_args(NB_ARGS if IN_NOTEBOOK else None)

FULL = ARGS.full
PLOT_ONLY = ARGS.plot
QUICK = ARGS.quick or (not FULL and not PLOT_ONLY)
DX_NM = ARGS.resolution or (2.0 if FULL else 4.0)
CONFORMAL = ARGS.conformal or "all"

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIG_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
if ARGS.arm:
    import tempfile

    FIG_DIR = Path(tempfile.mkdtemp())
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print(
    f"mode={'full' if FULL else ('plot' if PLOT_ONLY else 'quick')}  "
    f"dx={DX_NM:g} nm  conformal={CONFORMAL}"
)


def savefig(fig, name, dpi=220):
    """Apply `wakis.figstyle.finish` and save `fig` in `figures/`."""
    path = FIG_DIR / name
    figstyle.finish(fig)
    figstyle.save(fig, path, dpi=dpi)
    return path


def _fix_unit_spaces(ax):
    """Ensure spacing before units in schematic text labels."""
    import re

    for txt in ax.texts:
        s = txt.get_text()
        s_new = s.replace(r"\mu m", r"\,\mu\mathrm{m}")
        s_new = re.sub(
            r"(\d)\\text\{(THz|GHz|MHz|nm|mm|ps|fs)\}", r"\1\\;\\text{\2}", s_new
        )
        s_new = re.sub(r"(\d)(THz|GHz|MHz)", r"\1\,\\mathrm{\2}", s_new)
        s_new = re.sub(r"(\d)(µm|nm|mm|ps|fs)\b", r"\1 \2", s_new)
        s_new = re.sub(r"\(9 cells,\s*[^)]+\)", "(9 cells)", s_new)
        if s_new != s:
            txt.set_text(s_new)


def _fix_cylinder_patch(ax, radius_nm=20.0, tol=6.0, fill=None):
    """Replace the STL bounding-box Rectangle at (-a, -a, 2a, 2a) with a Circle."""
    from matplotlib.patches import Circle, Rectangle

    for p in list(ax.patches):
        if isinstance(p, Rectangle):
            x0, y0 = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if (
                abs(x0 + radius_nm) < tol
                and abs(y0 + radius_nm) < tol
                and abs(w - 2 * radius_nm) < 2 * tol
                and abs(h - 2 * radius_nm) < 2 * tol
                and p.get_linestyle() in ("solid", "-")
            ):
                fc = p.get_facecolor() if fill is None else fill
                ec = p.get_edgecolor()
                lw = p.get_linewidth()
                zo = p.get_zorder()
                p.remove()
                ax.add_patch(
                    Circle(
                        (0.0, 0.0),
                        radius_nm,
                        facecolor=fc,
                        edgecolor=ec,
                        lw=lw,
                        zorder=zo,
                    )
                )
    for t in ax.texts:
        if "Ag" in t.get_text() or "Cylinder" in t.get_text():
            t.set_text("Ag ($N=2$)")
            t.set_position((0.0, 0.0))


# %% [markdown]
# ## 1. The physics: 2D localized surface plasmons, retardation, and time-domain ADEs
#
# ### 1.1 Which polarization excites the localized surface plasmon?
#
# Consider an infinitely long cylinder of radius $a$ and relative permittivity
# $\varepsilon(\omega)$ along the $z$-axis in vacuum ($\varepsilon_\mathrm{host} = 1$),
# illuminated at normal incidence ($\mathbf{k} \parallel +x$). The problem splits
# into two decoupled 2D polarizations:
#
# * **TM (`polarization="TM"` in Wakis, Bohren & Huffman Case I):** $\mathbf{E} = E_z \hat{\mathbf{z}}$
#   is parallel to the cylinder axis. Because $\mathbf{E}$ is everywhere tangent
#   to the boundary $\rho = a$, no surface charge $\sigma_s = \varepsilon_0 [\![E_\rho]\!]$
#   is induced. A sub-wavelength metal cylinder in TM acts like an inductive wire
#   without a localized surface plasmon resonance.
# * **TE (`polarization="TE"` in Wakis, Bohren & Huffman Case II):** $\mathbf{H} = H_z \hat{\mathbf{z}}$
#   is along the cylinder axis and $\mathbf{E} = (E_x, E_y)$ lies **in the
#   cross-section plane** (for propagation along $+x$, $\mathbf{E}_\mathrm{inc} = E_0 \hat{\mathbf{y}}$).
#   Here the normal field $E_\rho$ jumps across $\rho = a$, driving a dipolar
#   surface-charge oscillation around the circumference—the **localized surface
#   plasmon resonance (LSPR)**.
#
# **All simulations and Mie comparisons in this tutorial use Wakis's `"TE"`
# polarization** ($\mathbf{E}$ in the $xy$ cross-section plane).
#
# ### 1.2 Quasi-static condition ($\varepsilon = -1$) and retardation red-shift
#
# Solving Laplace's equation $\nabla^2 \Phi = 0$ in 2D cylindrical coordinates
# for a uniform in-plane field $\mathbf{E}_0 = E_0 \hat{\mathbf{y}}$ gives the
# potential outside the cylinder ($\rho > a$):
#
# $$\Phi_\mathrm{out}(\rho, \phi) = -E_0 \rho \cos\phi
#   + E_0 \frac{\varepsilon(\omega) - 1}{\varepsilon(\omega) + 1} \frac{a^2}{\rho} \cos\phi.$$
#
# The 2D dipolar polarizability per unit length is proportional to the
# Clausius--Mossotti factor
#
# $$\alpha_\mathrm{2D}(\omega) \propto a^2 \,\frac{\varepsilon(\omega) - 1}{\varepsilon(\omega) + 1},$$
#
# which has a pole when the denominator vanishes:
#
# $$\mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) = -1 \qquad \text{(2D cylinder in vacuum, vs. }\mathrm{Re}\,\varepsilon = -2\text{ for a 3D sphere)}.$$
#
# In the thin-cylinder ($x = 2\pi a / \lambda \ll 1$) limit, the $n = 1$ electric
# dipole term of the exact Bohren--Huffman TE series gives
#
# $$Q_\mathrm{abs}^\mathrm{qs} \approx \pi x \,\mathrm{Im}\!\left(\frac{\varepsilon^* - 1}{\varepsilon^* + 1}\right),
#   \qquad
#   Q_\mathrm{sca}^\mathrm{qs} = \frac{\pi^2}{4} x^3 \left|\frac{\varepsilon - 1}{\varepsilon + 1}\right|^2,
#   \qquad
#   Q_\mathrm{ext} = Q_\mathrm{abs} + Q_\mathrm{sca}.$$
#
# As the cylinder diameter $2a$ increases toward the optical wavelength
# ($2a = 40\text{ nm} \to 80\text{ nm}$), two electrodynamic effects appear in
# the full Bessel/Hankel series (`cylinder_efficiencies`):
#
# 1. **Dynamic depolarization (retardation red-shift):** the phase of the driving
#    wave varies across $2a$, shifting the dipole resonance denominator to
#    $\mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) \approx -(1 + \mathcal{O}(x^2)\ln x) < -1$.
#    Because $\mathrm{Re}\,\varepsilon(\lambda)$ becomes more negative with
#    increasing $\lambda$, the peak shifts to **longer wavelengths**
#    ($341.3\text{ nm}$ at $2a = 40\text{ nm}$ $\to$ $351.0\text{ nm}$ in $Q_\mathrm{ext}$
#    and $355.0\text{ nm}$ in $Q_\mathrm{sca}$ at $2a = 80\text{ nm}$), while an
#    $n = 2$ **quadrupole** peak emerges near $335.5\text{ nm}$ where
#    $\mathrm{Re}\,\varepsilon$ approaches $-1$.
# 2. **Radiation damping:** radiative energy loss scales as $x^3$ and broadens
#    the resonance from $\mathrm{FWHM} = 10.6\text{ nm}$ (quasi-static) to
#    $19.2\text{ nm}$ ($2a = 40\text{ nm}$) and $44.2\text{ nm}$ ($2a = 80\text{ nm}$).
#
# ### 1.3 Drude and Lorentz--Drude models, sign conventions, and ADEs
#
# Wakis defines frequency-domain phasors via the Fourier transform kernel
# $\int_{-\infty}^\infty x(t) e^{-i\omega t}\,\mathrm{d}t$ (`wakis.DFTMonitor`),
# corresponding to the harmonic time dependence $e^{+i\omega t}$. In this
# convention a passive (lossy) medium satisfies $\mathrm{Im}\,\varepsilon(\omega) \le 0$,
# and tabulated optical constants $(n, k)$ with $k \ge 0$ map to
#
# $$\varepsilon(\omega) = (n - i k)^2 = (n^2 - k^2) - 2 i n k \qquad (\text{`wakis.dispersive.eps_from_nk`}).$$
#
# A general $N$-pole Drude--Lorentz model writes $\varepsilon(\omega)$ as a
# high-frequency background $\varepsilon_\infty \ge 1$ plus one Drude free-electron
# pole and $N - 1$ Lorentz bound-electron (interband) oscillators:
#
# $$\varepsilon(\omega) = \varepsilon_\infty
#   - \frac{\omega_p^2}{\omega^2 - i\gamma_D\omega}
#   + \sum_{p=1}^{N-1} \frac{\Delta\varepsilon_p\,\omega_{0,p}^2}{\omega_{0,p}^2 - \omega^2 + i\gamma_p\omega}.$$
#
# **Why a time-domain code needs auxiliary differential equations (ADEs):**
#
# 1. **Convolution in time:** Multiplication $\mathbf{D}(\omega) = \varepsilon_0 \varepsilon(\omega)\mathbf{E}(\omega)$
#    in frequency becomes a causal history integral in time,
#    $\mathbf{D}(t) = \varepsilon_0 \varepsilon_\infty \mathbf{E}(t) + \varepsilon_0 \int_0^\infty \chi(\tau)\mathbf{E}(t-\tau)\,\mathrm{d}\tau$.
#    Storing the full time history of $\mathbf{E}$ at every edge would cost
#    $\mathcal{O}(N_t)$ memory and $\mathcal{O}(N_t^2)$ work.
# 2. **Static $\varepsilon_r < 0$ is unconditionally unstable:** In a standard
#    non-dispersive leapfrog step, the discrete dispersion relation has
#    $\sin^2(\omega\Delta t / 2) = (\Delta t^2 / (\varepsilon_0 \varepsilon_r \mu_0))\,k_\mathrm{grid}^2$.
#    If one plugs a constant negative permittivity $\varepsilon_r = \mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) \approx -1.17$
#    directly into Ampere's law ($\varepsilon_0 \varepsilon_r \partial_t \mathbf{E} = \nabla \times \mathbf{H}$),
#    $\sin^2(\omega\Delta t/2) < 0$: the wave equation becomes elliptic in space-time
#    and the highest grid spatial frequencies grow exponentially as $e^{+\alpha t}$
#    from round-off at **every** $\Delta t > 0$.
# 3. **ADE solution:** Because each rational pole susceptibility $\chi_p(\omega)$
#    satisfies a second-order ODE in time, Wakis introduces an auxiliary
#    polarization variable $\mathbf{P}_p(t)$ per pole on every dispersive edge:
#
# $$\frac{\mathrm{d}^2 \mathbf{P}_p}{\mathrm{d}t^2} + \gamma_p \frac{\mathrm{d}\mathbf{P}_p}{\mathrm{d}t} + \omega_{0,p}^2 \mathbf{P}_p
#   = \varepsilon_0 \Delta\varepsilon_p \omega_{0,p}^2 \mathbf{E}
#   \quad (\text{Lorentz}), \qquad
#   \frac{\mathrm{d}^2 \mathbf{P}_D}{\mathrm{d}t^2} + \gamma_D \frac{\mathrm{d}\mathbf{P}_D}{\mathrm{d}t}
#   = \varepsilon_0 \omega_p^2 \mathbf{E}
#   \quad (\text{Drude, }\omega_0 = 0),$$
#
# coupled to Ampere's law with $\varepsilon_\infty \ge 1$:
#
# $$\varepsilon_0 \varepsilon_\infty \frac{\partial \mathbf{E}}{\partial t}
#   = \nabla \times \mathbf{H} - \mathbf{J} - \sigma \mathbf{E} - \sum_{p=1}^N \frac{\partial \mathbf{P}_p}{\partial t}.$$
#
# Centred differences at time level $n$ yield a two-level recurrence
# $p_p^{n+1} = \alpha_p p_p^n + \xi_p p_p^{n-1} + \zeta_p E^n$, requiring
# **2 float64 arrays ($p_p^n, p_p^{n-1}$, 16 bytes) per dispersive edge per
# Lorentz or Drude pole** (and 1 array, 8 bytes, per first-order Debye pole).
# Replacing $\omega$ by the discrete leapfrog operator gives the exact discrete
# permittivity (`DispersiveMaterial.epsilon_discrete(f, dt)`):
#
# $$\varepsilon_\mathrm{disc}(\omega, \Delta t) = \varepsilon_\infty
#   + \sum_p \frac{s_p}{\omega_{0,p}^2 - \hat{\omega}^2 + i\gamma_p \tilde{\omega}},
#   \qquad
#   \hat{\omega} = \frac{2}{\Delta t}\sin\!\left(\frac{\omega\Delta t}{2}\right),
#   \quad
#   \tilde{\omega} = \hat{\omega}\cos\!\left(\frac{\omega\Delta t}{2}\right),$$
#
# which converges to $\varepsilon(\omega)$ as $\mathcal{O}(\omega^2 \Delta t^2)$.
#
# ### 1.4 Interband transitions: why gold needs more poles than silver
#
# In a noble metal, free $s$-band conduction electrons follow a Drude response,
# while bound $d$-band electrons undergo interband transitions ($d \to sp$) above
# a threshold photon energy $\hbar\omega_\mathrm{ib}$:
#
# * **Silver (Ag):** The $4d \to 5sp$ interband onset is in the ultraviolet at
#   $\hbar\omega_\mathrm{ib} \approx 3.9\text{ eV}$ ($\lambda \approx 318\text{ nm}$),
#   *above* the cylinder plasmon resonance ($\lambda_\mathrm{res} = 341.3\text{ nm}$,
#   $3.63\text{ eV}$). Near $\lambda_\mathrm{res}$, the interband transitions
#   primarily contribute a positive real background polarizability plus a steep
#   UV dispersion tail that a single UV Lorentz pole ($\omega_0 \approx 4.21\text{ eV}$,
#   $\lambda_0 \approx 294\text{ nm}$) captures accurately. Thus **$N = 2$
#   (Drude + 1 Lorentz) already captures the Ag cylinder LSPR**.
# * **Gold (Au):** Relativistic contraction of the $6s$ shell lowers the
#   $5d \to 6sp$ interband threshold to $\hbar\omega_\mathrm{ib} \approx 2.4\text{ eV}$
#   ($\lambda \approx 517\text{ nm}$), directly overlapping the localized surface
#   plasmon at $\lambda_\mathrm{res} = 496.0\text{ nm}$ ($2.50\text{ eV}$). A pure
#   Drude model ($N = 1$) misses the interband absorption plateau completely,
#   placing the resonance $97\text{ nm}$ too blue; reconstructing the broad
#   structured $5d \to 6sp$ band across $300\text{--}550\text{ nm}$ requires
#   **$N \ge 3\text{--}6$ poles**.
#
# ### 1.5 Simulation setups
#
# The two-figure setup layouts below show the Cartesian 2D `SolverFIT3D` TF/SF setup
# (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`)
# and the body-fitted locally refined 2D `UnstructuredFIT2D` triangular-mesh setup
# (`figures/00a_setup_geometry_unstructured.png`,
# `figures/00b_setup_mesh_and_excitation_unstructured.png`):
#
# ![Cartesian 2D plasmonic cylinder setup geometry](figures/00a_setup_geometry.png)
# ![Cartesian 2D plasmonic cylinder mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the Cartesian 2D `SolverFIT3D` plasmonic cylinder setup on the 4-cell periodic $z$ slab with Kottke subpixel averaging, a `TFSFSource` box ($\mathbf{E} \parallel y$, 2D TE), inner and outer `FluxMonitor` boxes, and 9-cell CPML boundaries.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 184 nm × 184 nm × 160 nm (z ∈ [-80 nm, 80 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 46 × 46 × 4 (8464 cells) |
# | Discretization | Δx & resolution | Δ_min = 4 nm, 35.6 cells/λ (in n_max = 2.11), 5.6 cells across feature (22.5 nm) |
# | Boundary | -x face | CPML (9 cells, 36 nm) |
# | Boundary | +x face | CPML (9 cells, 36 nm) |
# | Boundary | -y face | CPML (9 cells, 36 nm) |
# | Boundary | +y face | CPML (9 cells, 36 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Ag cylinder (N = 2 Drude–Lorentz) | ε_r = 4.44, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-20 nm, 20 nm] (d = 40 nm) |
# | Source | 1 · TFSFSource | pos: x ∈ [-36 nm, 36 nm], pol: E_y, pulse: f₀ = 666.2 THz, Δf = 166.6 THz |
# | Monitor | 2 · FluxMonitor | Outer FluxMonitor (P'_sca), pos: x, y ∈ [-44, 44] nm, freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
# | Monitor | 3 · FluxMonitor | Inner FluxMonitor (P'_abs), pos: x, y ∈ [-28, 28] nm, freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
# | Variation | Metals & diameters | Ag (2a = 40 nm and 80 nm) and Au (2a = 40 nm) |
# | Variation | Dispersive pole models | N = 1, 2, 3, 4, 6 fitted to CC0 Johnson & Christy (1972) + Rakić (1998) + constant-ε baseline |
# | Variation | Resolutions & subpixel | h = 4 nm (quick) and h = 2 nm (full), Kottke subpixel averaging (wakis.tensor_dispersive) |
#
# ![Body-fitted triangular 2D plasmonic cylinder setup geometry](figures/00a_setup_geometry_unstructured.png)
# ![Body-fitted triangular 2D plasmonic cylinder mesh and excitation](figures/00b_setup_mesh_and_excitation_unstructured.png)
#
# The table below summarizes the body-fitted locally refined 2D `UnstructuredFIT2D` triangular-mesh setup with Voronoi-dual Hodge operators, a scattered-field plane-wave drive, cylindrical harmonic rings, and a 10-layer radial CFS-PML.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly = 280 nm × 280 nm (x ∈ [-140 nm, 140 nm], y ∈ [-140 nm, 140 nm]) |
# | Domain | Grid cells | Nx × Ny = 4448 × 4448 (19784704 cells) |
# | Discretization | Δx & resolution | Δ_min = 1.61 nm, 88.7 cells/λ (in n_max = 2.11), 14.0 cells across feature (22.5 nm) |
# | Boundary | -x face | PEC |
# | Boundary | +x face | PEC |
# | Boundary | -y face | PEC |
# | Boundary | +y face | PEC |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Ag cylinder (N = 2 Drude–Lorentz) | ε_r = 4.44, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-20 nm, 20 nm] (d = 40 nm) |
# | Source | 1 · PlaneWave | pos: r <= 20 nm (Ag cylinder interior), pol: TE (E_x, E_y in-plane, H_z axial), pulse: f₀ = 666.2 THz, Δf = 166.6 THz |
# | Monitor | 2 · HarmonicRings | Cylindrical harmonic rings (r = 45, 55, 65 nm), pos: r = 45, 55, 65 nm (inside radial CFS-PML r ∈ [80, 140] nm), freqs: 241 freqs in [333.1, 999.3] THz (300–900 nm) |
# | Variation | Radial CFS-PML | 10 annular layers in r ∈ [80, 140] nm with PEC outer termination |
# | Variation | Local mesh refinement | h_s = 2.0 nm (4448 tris) and h_s = 1.25 nm (7097 tris) on r = 20 nm boundary |

# %% [markdown]
# ## 2. Pole fitting: $N = 1, 2, 3, 4, 6$ vs Johnson & Christy and Rakić
#
# We fit passive multi-pole models ($1\text{ Drude} + (N-1)\text{ Lorentz}$ for
# $N \in \{1, 2, 3, 4, 6\}$) to the public-domain (CC0) Johnson & Christy (1972)
# tabulated data over $300\text{--}900\text{ nm}$ using `wakis.dispersive.fit_poles`.
# Because the raw J&C table has only 23 points in $300\text{--}900\text{ nm}$
# (spaced by $10\text{--}70\text{ nm}$), unconstrained multi-pole least squares
# on the raw nodes alone can drop narrow resonant spikes *between* sample points.
# Evaluating the linear $(n, k)$ interpolant on a uniform 241-point grid
# ($\Delta\lambda = 2.5\text{ nm}$) prevents inter-sample spikes while keeping
# all fits strictly passive ($\mathrm{Im}\,\varepsilon(\omega) \le 0$) and
# discrete-time stable (`mat.is_stable(dt, courant=0.85)`).

# %%
# Embedded CC0 Johnson & Christy (1972) data (Phys. Rev. B 6, 4370; 0.1879-1.9370 um)
# so the script and Colab notebook run standalone even without tests/data/.
JC_EMBEDDED = {
    "Ag": [
        (187.9, 1.07, 1.212), (191.6, 1.10, 1.232), (195.3, 1.12, 1.255),
        (199.3, 1.14, 1.277), (203.3, 1.15, 1.296), (207.3, 1.18, 1.312),
        (211.9, 1.20, 1.325), (216.4, 1.22, 1.336), (221.4, 1.25, 1.342),
        (226.2, 1.26, 1.344), (231.3, 1.28, 1.357), (237.1, 1.28, 1.367),
        (242.6, 1.30, 1.378), (249.0, 1.31, 1.389), (255.1, 1.33, 1.393),
        (261.6, 1.35, 1.387), (268.9, 1.38, 1.372), (276.1, 1.41, 1.331),
        (284.4, 1.41, 1.264), (292.4, 1.39, 1.161), (300.9, 1.34, 0.964),
        (310.7, 1.13, 0.616), (320.4, 0.81, 0.392), (331.5, 0.17, 0.829),
        (342.5, 0.14, 1.142), (354.2, 0.10, 1.419), (367.9, 0.07, 1.657),
        (381.5, 0.05, 1.864), (397.4, 0.05, 2.070), (413.3, 0.05, 2.275),
        (430.5, 0.04, 2.462), (450.9, 0.04, 2.657), (471.4, 0.05, 2.869),
        (495.9, 0.05, 3.093), (520.9, 0.05, 3.324), (548.6, 0.06, 3.586),
        (582.1, 0.05, 3.858), (616.8, 0.06, 4.152), (659.5, 0.05, 4.483),
        (704.5, 0.04, 4.838), (756.0, 0.03, 5.242), (821.1, 0.04, 5.727),
        (892.0, 0.04, 6.312), (984.0, 0.04, 6.992),
    ],
    "Au": [
        (187.9, 1.28, 1.188), (191.6, 1.32, 1.203), (195.3, 1.34, 1.226),
        (199.3, 1.33, 1.251), (203.3, 1.33, 1.277), (207.3, 1.30, 1.304),
        (211.9, 1.30, 1.350), (216.4, 1.30, 1.387), (221.4, 1.30, 1.427),
        (226.2, 1.31, 1.460), (231.3, 1.30, 1.497), (237.1, 1.32, 1.536),
        (242.6, 1.32, 1.577), (249.0, 1.33, 1.631), (255.1, 1.33, 1.688),
        (261.6, 1.35, 1.749), (268.9, 1.38, 1.803), (276.1, 1.43, 1.847),
        (284.4, 1.47, 1.869), (292.4, 1.49, 1.878), (300.9, 1.53, 1.889),
        (310.7, 1.53, 1.893), (320.4, 1.54, 1.898), (331.5, 1.48, 1.883),
        (342.5, 1.48, 1.871), (354.2, 1.50, 1.866), (367.9, 1.48, 1.895),
        (381.5, 1.46, 1.933), (397.4, 1.47, 1.952), (413.3, 1.46, 1.958),
        (430.5, 1.45, 1.948), (450.9, 1.38, 1.914), (471.4, 1.31, 1.849),
        (495.9, 1.04, 1.833), (520.9, 0.62, 2.081), (548.6, 0.43, 2.455),
        (582.1, 0.29, 2.863), (616.8, 0.21, 3.272), (659.5, 0.14, 3.697),
        (704.5, 0.13, 4.103), (756.0, 0.14, 4.542), (821.1, 0.16, 5.083),
        (892.0, 0.17, 5.663), (984.0, 0.22, 6.350),
    ],
}


def reconstruct_material(entry):
    """Reconstruct a `DispersiveMaterial` from a serialized JSON model dict."""
    poles = []
    for d in entry.get("drude", []):
        poles.append(Drude(omega_p=d["omega_p_rad_s"], gamma=d["gamma_rad_s"]))
    for lo in entry.get("lorentz", []):
        poles.append(
            Lorentz(
                delta_eps=lo["delta_eps"],
                omega_0=lo["omega_0_rad_s"],
                gamma=lo["gamma_rad_s"],
            )
        )
    return DispersiveMaterial(
        eps_inf=entry["eps_inf"], poles=poles, name=entry.get("name", "")
    )


def fit_jc_poles(metal="Ag", n_poles=2, n_starts=6):
    """Fit a passive 1-Drude + (n_poles - 1)-Lorentz model to CC0 J&C data (300-900 nm)."""
    arr = np.array(JC_EMBEDDED[metal], dtype=float)
    lam_fit_nm = np.linspace(300.0, 900.0, 241)
    f_fit = c0 / (lam_fit_nm * 1e-9)
    n_fit = np.interp(lam_fit_nm, arr[:, 0], arr[:, 1])
    k_fit = np.interp(lam_fit_nm, arr[:, 0], arr[:, 2])
    eps_fit = eps_from_nk(n_fit, k_fit)
    mat = fit_poles(
        f_fit,
        eps_fit,
        n_lorentz=n_poles - 1,
        n_drude=1,
        n_starts=n_starts,
        name=f"{metal}_N{n_poles}",
    )
    rel_rms = float(
        np.sqrt(np.mean(np.abs((mat.epsilon(f_fit) - eps_fit) / np.abs(eps_fit)) ** 2))
    )
    return mat, rel_rms


def load_committed_json(filename):
    """Load `data/<filename>` locally or fall back to GitHub when running on Colab."""
    path = DATA_DIR / filename
    if path.exists():
        return json.loads(path.read_text())
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/eshipton/Wakis/main/tutorials/"
        f"03_diffraction_and_scattering/03-06_plasmonic_cylinder/data/{filename}"
    )
    with urllib.request.urlopen(url, timeout=20) as f:
        return json.loads(f.read())


MODELS_DATA = load_committed_json("models.json")
SPECTRA_DATA = load_committed_json("spectra_and_errors.json")
TIMING_DATA = load_committed_json("timing.json")
TRI_DATA = load_committed_json("triangle_lspr.json")
TRIANGLE_DATA = {"meshes": TRI_DATA} if "meshes" not in TRI_DATA else TRI_DATA
TRI_NPZ = np.load(DATA_DIR / "triangle_lspr_fields.npz")


def plot_setup_schematic():
    """Render setup geometry, mesh/excitation, and setup tables via save_setup_figures."""
    from types import SimpleNamespace

    dx_s = 4e-9
    r_m = 20e-9
    i_a = int(np.ceil(r_m / dx_s - 1e-9))
    i_in = i_a + 2
    i_tf = i_in + 2
    i_flux = i_tf + 2
    i_pml = i_flux + 3
    n_pml = 9
    n_half = i_pml + n_pml
    nz = 4
    dz_s = 10.0 * dx_s
    half_s = n_half * dx_s
    hz_s = 0.5 * nz * dz_s

    mat_ag = reconstruct_material(MODELS_DATA["metals"]["Ag"]["models"]["N2"])
    cyl_pv = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=r_m,
        height=10.0 * hz_s,
        resolution=360,
        capping=True,
    ).triangulate()
    grid_c = wakis.GridFIT3D(
        -half_s,
        half_s,
        -half_s,
        half_s,
        -hz_s,
        hz_s,
        2 * n_half,
        2 * n_half,
        nz,
        stl_solids={"cylinder": cyl_pv},
        stl_materials={"cylinder": mat_ag},
        conformal="all",
        verbose=0,
    )
    solver_c = wakis.SolverFIT3D(
        grid_c,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        use_stl=True,
        cfln=0.85,
        verbose=0,
    )
    grid_c.stl_solids = {}
    cyl_struct = SimpleNamespace(
        name="Ag cylinder (N = 2 Drude–Lorentz)",
        center=(0.0, 0.0, 0.0),
        radius=r_m,
        height=10.0 * hz_s,
        axis=2,
        material=mat_ag,
    )
    solver_c._assigned_structures = [cyl_struct]

    f_min, f_max = c0 / 900e-9, c0 / 300e-9
    f0 = 0.5 * (f_min + f_max)
    df = 0.25 * (f_max - f_min)
    freqs_s = np.linspace(f_min, f_max, 241)
    freq_str_s = f"241 freqs in [{f_min / 1e12:.1f}, {f_max / 1e12:.1f}] THz (300–900 nm)"
    pulse = GaussianSource(frequency=f0, fwidth=df)
    src_c = wakis.TFSFSource(
        pulse,
        center=(0.0, 0.0, 0.0),
        size=(2.0 * i_tf * dx_s, 2.0 * i_tf * dx_s, None),
        direction="+x",
        polarization="y",
        amplitude=1.0,
    )
    src_c.label = "TFSFSource (k || +x, E || y, 2D TE)"
    mon_c = [
        {
            "kind": "box",
            "type_name": "FluxMonitor",
            "short_word": "flux",
            "desc": f"outer flux box (FluxMonitor, x, y ∈ [-{i_flux*dx_s*1e9:.0f}, {i_flux*dx_s*1e9:.0f}] nm)",
            "label": "Outer FluxMonitor (P'_sca)",
            "pos_str": f"x, y ∈ [-{i_flux*dx_s*1e9:.0f}, {i_flux*dx_s*1e9:.0f}] nm",
            "freq_str": freq_str_s,
            "frequencies": freqs_s,
            "h_bounds": (-i_flux * dx_s, i_flux * dx_s),
            "v_bounds": (-i_flux * dx_s, i_flux * dx_s),
        },
        {
            "kind": "box",
            "type_name": "FluxMonitor",
            "short_word": "flux",
            "desc": f"inner flux box (FluxMonitor, x, y ∈ [-{i_in*dx_s*1e9:.0f}, {i_in*dx_s*1e9:.0f}] nm)",
            "label": "Inner FluxMonitor (P'_abs)",
            "pos_str": f"x, y ∈ [-{i_in*dx_s*1e9:.0f}, {i_in*dx_s*1e9:.0f}] nm",
            "freq_str": freq_str_s,
            "frequencies": freqs_s,
            "h_bounds": (-i_in * dx_s, i_in * dx_s),
            "v_bounds": (-i_in * dx_s, i_in * dx_s),
        },
    ]

    schematic.save_setup_figures(
        solver_c,
        FIG_DIR,
        plane="xy",
        structures=[cyl_struct],
        material_names={"cylinder": "Ag cylinder (N = 2 Drude–Lorentz)"},
        sources=[src_c],
        monitors=mon_c,
        unit="nm",
        wavelength=300e-9,
        freq=f0,
        min_feature=22.5e-9,
        title_geometry="Cartesian 2D plasmonic cylinder setup (TF/SF + FluxMonitor, XY plane)",
        title_mesh="Cartesian 2D mesh & broadband pulse (300–900 nm, skin depth δ_min = 22.5 nm)",
        variations=[
            {
                "name": "Metals & diameters",
                "details": "Ag (2a = 40 nm and 80 nm) and Au (2a = 40 nm)",
            },
            {
                "name": "Dispersive pole models",
                "details": "N = 1, 2, 3, 4, 6 fitted to CC0 Johnson & Christy (1972) + Rakić (1998) + constant-ε baseline",
            },
            {
                "name": "Resolutions & subpixel",
                "details": "h = 4 nm (quick) and h = 2 nm (full), Kottke subpixel averaging (wakis.tensor_dispersive)",
            },
        ],
    )

    h_s_b, h_out_b, r_pml_b = 2.0e-9, 6.0e-9, 80.0e-9
    mesh_b = cluster_mesh(
        [dict(center=(0.0, 0.0), radius=r_m)],
        h_s_b,
        h_out_b,
        r_pml_b,
        growth=1.18,
        pml_layers=10,
        pml_dr=h_out_b,
    )
    solver_b = UnstructuredFIT2D(
        mesh_b,
        {0: [1.0, 1.0], 1: mat_ag, 2: [1.0, 1.0]},
        polarization="TE",
        bc="pec",
        hodge="voronoi",
        cfl=0.90,
        pml=dict(r0=r_pml_b, r1=r_pml_b + 10 * h_out_b, kappa_max=1.0, alpha=0.0),
    )
    solver_b._assigned_structures = [cyl_struct]
    tri_src = {
        "kind": "sheet",
        "type_name": "PlaneWave",
        "short_word": "PW",
        "desc": "PlaneWave drive (k || +x, E || y, scattered-field)",
        "label": "Scattered-field plane-wave drive (k || +x, E || y)",
        "pos_str": "r <= 20 nm (Ag cylinder interior)",
        "polarization": "TE (E_x, E_y in-plane, H_z axial)",
        "h_bounds": (-r_m, -r_m),
        "v_bounds": (-0.7 * r_m, 0.7 * r_m),
        "k_vec": (1.0, 0.0, 0.0),
        "e_vec": (0.0, 1.0, 0.0),
        "h_vec": (0.0, 0.0, 1.0),
        "pulse": schematic._extract_pulse_metadata(pulse),
    }
    tri_mon = {
        "kind": "box",
        "type_name": "HarmonicRings",
        "short_word": "rings",
        "desc": "Harmonic rings (r = 45, 55, 65 nm)",
        "label": "Cylindrical harmonic rings (r = 45, 55, 65 nm)",
        "pos_str": "r = 45, 55, 65 nm (inside radial CFS-PML r ∈ [80, 140] nm)",
        "freq_str": freq_str_s,
        "frequencies": freqs_s,
        "h_bounds": (-55e-9, 55e-9),
        "v_bounds": (-55e-9, 55e-9),
    }
    schematic.save_setup_figures(
        solver_b,
        FIG_DIR,
        suffix="_unstructured",
        plane="xy",
        structures=[cyl_struct],
        sources=[tri_src],
        monitors=[tri_mon],
        unit="nm",
        wavelength=300e-9,
        freq=f0,
        min_feature=22.5e-9,
        title_geometry="Body-fitted triangular 2D plasmonic cylinder setup (radial CFS-PML, XY plane)",
        title_mesh="Body-fitted locally refined 2D triangular mesh & pulse (h_s = 2 nm)",
        variations=[
            {
                "name": "Radial CFS-PML",
                "details": "10 annular layers in r ∈ [80, 140] nm with PEC outer termination",
            },
            {
                "name": "Local mesh refinement",
                "details": "h_s = 2.0 nm (4448 tris) and h_s = 1.25 nm (7097 tris) on r = 20 nm boundary",
            },
        ],
    )


plot_setup_schematic()
for _fname in ("00a_setup_geometry.png", "00b_setup_mesh_and_excitation.png"):
    fig_s, ax_s = plt.subplots(figsize=(8.0, 5.0))
    ax_s.imshow(plt.imread(FIG_DIR / _fname))
    ax_s.axis("off")
    plt.show()

if not PLOT_ONLY and ARGS.task in ("all", "fit"):
    demo_mat, demo_rms = fit_jc_poles("Ag", n_poles=2, n_starts=4)
    print(
        f"Live fit_poles('Ag', N=2): eps_inf={demo_mat.eps_inf:.3f}, "
        f"rel_rms={demo_rms:.4f}, passive={bool(np.all(demo_mat.epsilon(c0 / ( np.linspace(300, 900, 100) * 1e-9 )).imag <= 1e-12))}"
    )

LAM_EVAL_NM = np.array(MODELS_DATA["wavelengths_eval_nm"], dtype=float)
FREQS_EVAL = c0 / (LAM_EVAL_NM * 1e-9)

for metal in ("Ag", "Au"):
    m_dict = MODELS_DATA["metals"][metal]["models"]
    print(f"\n=== {metal} dispersive models (300-900 nm, fitted to CC0 J&C 1972) ===")
    print(
        f"{'Model':<7} {'N':>2} {'eps_inf':>8} {'rms(dense)':>11} {'rms(raw)':>9} "
        f"{'passive':>8} {'dt_max(0.85)[as]':>17}  Poles (eV)"
    )
    for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
        info = m_dict[k]
        d_str = ", ".join(
            f"D(wp={d['omega_p_eV']:.2f}, g={d['gamma_eV']:.3f})"
            for d in info["drude"]
        )
        l_str = ", ".join(
            f"L(de={lo['delta_eps']:.2f}, w0={lo['omega_0_eV']:.2f}, g={lo['gamma_eV']:.2f})"
            for lo in info["lorentz"]
        )
        poles_str = "; ".join(s for s in (d_str, l_str) if s)
        print(
            f"{k:<7} {info['n_poles']:>2} {info['eps_inf']:>8.3f} "
            f"{info['fit_rms_rel_dense']:>11.4f} {info['fit_rms_rel_raw_jc']:>9.4f} "
            f"{str(info['is_passive']):>8} {info['dt_max_cfl085_as']:>17.1f}  {poles_str}"
        )

# %%
# Figure 1 (3 panels): (a) Ag Re eps(lambda), (b) Au Re eps(lambda), (c) Ag & Au loss -Im eps(lambda) >= 0
MODEL_COLORS = {
    "N1": "#E69F00",
    "N2": "#0072B2",
    "N3": "#009E73",
    "N4": "#56B4E9",
    "N6": "#CC79A7",
    "Rakic": "#D55E00",
}
MODEL_LABELS = {
    "N1": "N=1 (Drude)",
    "N2": "N=2 (Drude + 1L)",
    "N3": "N=3 (Drude + 2L)",
    "N4": "N=4 (Drude + 3L)",
    "N6": "N=6 (Drude + 5L)",
    "Rakic": "Rakić 1998 (1D + 5L)",
}

fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=14.4, height_in=4.3)
for col_idx, (metal, panel_lbl, title_metal) in enumerate(
    (("Ag", "(a)", "Silver (Ag)"), ("Au", "(b)", "Gold (Au)"))
):
    mdata = MODELS_DATA["metals"][metal]
    jc_raw = mdata["jc_raw"]
    ax_re = axes[col_idx]

    ax_re.plot(
        jc_raw["lam_nm"],
        jc_raw["eps_real"],
        "ko",
        ms=3.8,
        label="J&C 1972 (CC0 data)",
        zorder=5,
    )
    for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
        minfo = mdata["models"][k]
        ls = "--" if k in ("N1", "Rakic") else "-"
        lw = 1.8 if k in ("N2", "N6") else 1.2
        rms_pct = minfo["fit_rms_rel_dense"] * 100.0
        lbl = f"{MODEL_LABELS[k]} ({rms_pct:.1f}% RMS)"
        ax_re.plot(
            LAM_EVAL_NM, minfo["eps_real"], color=MODEL_COLORS[k], ls=ls, lw=lw, label=lbl
        )

    ax_re.axhline(
        -1.0, color="0.35", ls=":", lw=1.0, label=r"2D quasi-static $\varepsilon' = -1$"
    )
    lam_res_jc = SPECTRA_DATA[f"{metal}_40nm"]["mie_jc"]["lam_peak_nm"]
    ax_re.axvline(lam_res_jc, color="0.5", ls="-.", lw=0.9)
    ax_re.set_title(
        f"{panel_lbl} {title_metal}: $\\mathrm{{Re}}\\,\\varepsilon(\\lambda)$"
    )
    ax_re.set_ylabel(r"$\mathrm{Re}\,\varepsilon(\lambda)$")
    ax_re.set_xlabel(r"wavelength $\lambda$ [nm]")
    ax_re.set_ylim(-42, 6)
    ax_re.legend(fontsize=6.8, loc="lower left")

# Panel (c): Loss -Im eps(lambda) >= 0 (passivity) for Ag and Au
ax_im = axes[2]
for metal, m_col, m_mk in (("Ag", "#0072B2", "o"), ("Au", "#D55E00", "s")):
    mdata = MODELS_DATA["metals"][metal]
    jc_raw = mdata["jc_raw"]
    ax_im.plot(
        jc_raw["lam_nm"],
        [-v for v in jc_raw["eps_imag"]],
        marker=m_mk,
        ls="none",
        ms=3.5,
        color=m_col,
        alpha=0.75,
        label=f"{metal} J&C 1972 data",
    )
ax_im.plot(
    LAM_EVAL_NM,
    [-v for v in MODELS_DATA["metals"]["Ag"]["models"]["N2"]["eps_imag"]],
    "-",
    color="#0072B2",
    lw=1.9,
    label="Ag $N=2$ (Drude + 1L)",
)
ax_im.plot(
    LAM_EVAL_NM,
    [-v for v in MODELS_DATA["metals"]["Ag"]["models"]["Rakic"]["eps_imag"]],
    "--",
    color="#0072B2",
    lw=1.2,
    alpha=0.8,
    label="Ag Rakić 1998",
)
ax_im.plot(
    LAM_EVAL_NM,
    [-v for v in MODELS_DATA["metals"]["Au"]["models"]["N1"]["eps_imag"]],
    ":",
    color="#E69F00",
    lw=1.4,
    label="Au $N=1$ (Drude only)",
)
ax_im.plot(
    LAM_EVAL_NM,
    [-v for v in MODELS_DATA["metals"]["Au"]["models"]["N3"]["eps_imag"]],
    "-.",
    color="#009E73",
    lw=1.5,
    label="Au $N=3$ (Drude + 2L)",
)
ax_im.plot(
    LAM_EVAL_NM,
    [-v for v in MODELS_DATA["metals"]["Au"]["models"]["N6"]["eps_imag"]],
    "-",
    color="#D55E00",
    lw=1.9,
    label="Au $N=6$ (Drude + 5L)",
)
ax_im.set_title(r"(c) Loss $-\mathrm{Im}\,\varepsilon(\lambda) \geq 0$ (Ag vs Au)")
ax_im.set_ylabel(r"$-\mathrm{Im}\,\varepsilon(\lambda)$")
ax_im.set_xlabel(r"wavelength $\lambda$ [nm]")
ax_im.set_ylim(0, 7.2)
ax_im.legend(fontsize=6.8, loc="upper right")

savefig(fig, "eps_fits_ag_au.png")
plt.show()

# %% [markdown]
# ### Fitted pole parameters (300--900 nm, passive, $\mathrm{Im}\,\varepsilon \le 0$)
#
# All fits use the 241-point interpolated Johnson & Christy (1972) data on
# $300\text{--}900\text{ nm}$; relative RMS error is
# $\bigl(\frac{1}{M}\sum_j |\varepsilon_\mathrm{fit}(\omega_j) - \varepsilon_\mathrm{J\&C}(\omega_j)|^2 / |\varepsilon_\mathrm{J\&C}(\omega_j)|^2\bigr)^{1/2}$.
#
# **Table 1a — Silver (Ag) dispersive pole models:**
#
# | Model | $N$ | $\varepsilon_\infty$ | Drude $(\omega_p, \gamma_D)$ [eV] | Lorentz poles $(\Delta\varepsilon_p, \omega_{0,p}\text{ [eV]}, \gamma_p\text{ [eV]})$ | Rel. RMS $\varepsilon$ (dense / raw J&C) | $\Delta t_\mathrm{max}$ (CFL 0.85) [as] |
# |---|---:|---:|---|---|---:|---:|
# | `N1` (Drude) | 1 | 6.462 | $(9.873, 0.136)$ | — | 21.68% / 33.46% | 319.4 |
# | `N2` (Drude + 1L) | 2 | 4.435 | $(9.442, 0.020)$ | $(0.317, 4.234, 0.307)$ | 6.24% / 12.09% | 247.9 |
# | `N3` (Drude + 2L) | 3 | 4.740 | $(9.487, 0.000)$ | $(0.340, 2.022, 0.774)$; $(0.292, 4.213, 0.314)$ | 5.45% / 11.19% | 254.8 |
# | `N4` (Drude + 3L) | 4 | 4.772 | $(9.498, 0.000)$ | $(0.258, 1.923, 0.562)$; $(0.060, 2.559, 0.456)$; $(0.295, 4.211, 0.318)$ | 5.33% / 10.99% | 255.4 |
# | `N6` (Drude + 5L) | 6 | 4.783 | $(9.500, 0.000)$ | $(0.159, 1.747, 0.362)$; $(0.111, 2.111, 0.368)$; $(0.036, 2.522, 0.313)$; $(0.015, 2.880, 0.256)$; $(0.297, 4.210, 0.319)$ | 5.23% / 10.84% | 255.5 |
# | `Rakic` (library) | 6 | 1.000 | $(8.285, 0.048)$ | $(7.925, 0.816, 3.886)$; $(0.501, 4.481, 0.452)$; $(0.013, 8.185, 0.065)$; $(0.826, 9.083, 0.916)$; $(1.113, 20.29, 2.419)$ | 24.39% / 31.07% | 26.2 |
#
# **Table 1b — Gold (Au) dispersive pole models:**
#
# | Model | $N$ | $\varepsilon_\infty$ | Drude $(\omega_p, \gamma_D)$ [eV] | Lorentz poles $(\Delta\varepsilon_p, \omega_{0,p}\text{ [eV]}, \gamma_p\text{ [eV]})$ | Rel. RMS $\varepsilon$ (dense / raw J&C) | $\Delta t_\mathrm{max}$ (CFL 0.85) [as] |
# |---|---:|---:|---|---|---:|---:|
# | `N1` (Drude) | 1 | 5.660 | $(8.146, 0.488)$ | — | 53.54% / 69.14% | 358.8 |
# | `N2` (Drude + 1L) | 2 | 5.412 | $(8.952, 0.000)$ | $(3.619, 3.471, 2.067)$ | 11.24% / 14.82% | 232.6 |
# | `N3` (Drude + 2L) | 3 | 4.462 | $(8.864, 0.000)$ | $(1.050, 2.880, 0.822)$; $(2.361, 3.968, 1.816)$ | 4.65% / 6.17% | 207.5 |
# | `N4` (Drude + 3L) | 4 | 4.425 | $(8.807, 0.000)$ | $(0.159, 1.424, 0.214)$; $(1.080, 2.891, 0.835)$; $(2.301, 3.972, 1.784)$ | 4.43% / 6.13% | 207.4 |
# | `N6` (Drude + 5L) | 6 | 4.046 | $(8.729, 0.000)$ | $(0.219, 1.341, 0.168)$; $(0.090, 1.568, 0.252)$; $(0.409, 2.702, 0.502)$; $(0.958, 3.141, 0.952)$; $(1.966, 4.139, 1.662)$ | 2.43% / 3.58% | 197.6 |
# | `Rakic` (library) | 6 | 1.000 | $(7.872, 0.053)$ | $(11.36, 0.415, 0.241)$; $(1.184, 0.830, 0.345)$; $(0.656, 2.969, 0.870)$; $(2.645, 4.304, 2.494)$; $(2.015, 13.32, 2.214)$ | 17.20% / 18.44% | 30.5 |
#
# Notice why `Rakic` (Rakić et al. 1998, fitted to Palik / Lynch & Hunter handbook
# data from $0.124\text{ to }6.2\text{ eV}$) differs noticeably from Johnson &
# Christy (1972): the Palik Ag data have nearly twice the damping in the near-UV,
# and Rakić includes a deep-UV oscillator at $20.29\text{ eV}$ (Ag) and
# $13.32\text{ eV}$ (Au) that lowers $\Delta t_\mathrm{max}$ from $\sim 250\text{ as}$
# to $26\text{--}30\text{ as}$.

# %% [markdown]
# ## 3. Geometry, skin-depth mesh rules, Kottke subpixel averaging, and preflight
#
# ### 3.1 Skin depth and resolution requirements
#
# Inside a metal with complex refractive index $m(\lambda) = n(\lambda) - i k(\lambda)$,
# the evanescent electric field decays with **field skin depth**
#
# $$\delta(\lambda) = \frac{\lambda}{2\pi k(\lambda)}.$$
#
# Evaluating $\delta(\lambda)$ from the Johnson & Christy data across
# $\lambda \in [300, 900]\text{ nm}$:
#
# * **Silver (Ag):** $\delta_\mathrm{min} = 22.50\text{ nm}$ at $\lambda = 900\text{ nm}$
#   ($k = 6.366$), and $\delta(\lambda_\mathrm{res}) = 28.59\text{ nm}$ at the
#   cylinder resonance $\lambda_\mathrm{res} = 341.33\text{ nm}$ ($k = 1.900$).
# * **Gold (Au):** $\delta_\mathrm{min} = 24.47\text{ nm}$ at $\lambda = 900\text{ nm}$,
#   and $\delta(\lambda_\mathrm{res}) = 42.31\text{ nm}$ at
#   $\lambda_\mathrm{res} = 496.01\text{ nm}$.
#
# We compare two Cartesian cross-section resolutions ($h = \Delta x = \Delta y = 4\;\mathrm{nm}$
# and $2\;\mathrm{nm}$ in `SolverFIT3D` with Kottke subpixel averaging) against two
# **body-fitted locally refined triangular meshes** (`UnstructuredFIT2D` with
# `wakis.benchmarks.meshing.cluster_mesh` and a 10-layer radial CFS-PML, where the
# triangle edge length is refined to $h_s = 2.0\;\mathrm{nm}$ or $1.25\;\mathrm{nm}$
# on the $r = a = 20\;\mathrm{nm}$ cylinder surface and graded outward with ratio
# $1.18$ to $h_\mathrm{out} = 6.0\;\mathrm{nm}$ or $5.0\;\mathrm{nm}$). All four
# meshes satisfy $\ge 4$ elements per minimum skin depth ($\delta_\mathrm{min} = 22.50\;\mathrm{nm}$)
# and $\ge 20$ elements per shortest vacuum wavelength ($\lambda_\mathrm{min} = 300\;\mathrm{nm}$):
#
# | Mesh / Solver | $h_\mathrm{cyl}$ ($h_\mathrm{out}$) [nm] | Elements / $\delta_\mathrm{min}$ ($22.5\text{ nm}$) | Elements / $\lambda_\mathrm{min}$ ($300\text{ nm}$) | Cells / Unknowns ($2a = 40\text{ nm}$) | $\Delta t$ [as] | Steps ($T$) | Wall time (`N2`) [s] | Ag `N2` $\lambda_\mathrm{res}$ [nm] (Disc. NRMS vs Mie) |
# |---|---|---|---|---|---:|---:|---:|---:|
# | Cartesian coarse (`--quick`, `SolverFIT3D`) | $4.0$ ($4.0$) | **5.63** | **75.0** | $8,464\text{ cells} / 26,132\text{ E-edges}$ | 7.97 | 4,390 ($35\text{ fs}$) | 5.4 | $343.32\text{ nm}$ ($5.46\%$) |
# | Cartesian fine (`--full`, `SolverFIT3D`) | $2.0$ ($2.0$) | **11.25** | **150.0** | $12,544\text{ cells} / 37,632\text{ E-edges}$ | 3.99 | 8,780 ($35\text{ fs}$) | 14.9 | $343.59\text{ nm}$ ($2.47\%$) |
# | **Body-fitted triangles (`UnstructuredFIT2D`, `tri_h2nm`)** | **$2.0$ ($6.0$)** | **11.25** | **50.0--150.0** | **$4,448\text{ tris} / 8,981\text{ dofs}$** | **4.28** | **14,021 ($60\text{ fs}$)** | **11.9** | **$343.09\text{ nm}$ ($0.28\%$)** |
# | **Body-fitted triangles fine (`UnstructuredFIT2D`, `tri_h1p25nm`)** | **$1.25$ ($5.0$)** | **18.00** | **60.0--240.0** | **$7,097\text{ tris} / 14,296\text{ dofs}$** | **2.65** | **22,600 ($60\text{ fs}$)** | **26.5** | **$343.09\text{ nm}$ ($0.35\%$)** |
#
# Because every triangle in `UnstructuredFIT2D` belongs to a single material
# ($322$ triangles inside the Ag cylinder at $h_s = 2.0\;\mathrm{nm}$, $824$ at
# $h_s = 1.25\;\mathrm{nm}$ with $101$ vertices on the $r = 20\;\mathrm{nm}$ circle),
# no edge mixes negative metal permittivity with positive vacuum permittivity. That
# eliminates the cut-cell zero-crossing artifacts of Cartesian subpixel averaging and
# lowers the Ag `N2` discretization NRMS error by **$8.8\times$** ($0.28\%$ vs $2.47\%$)
# with **$4.2\times$ fewer unknowns** ($8,981$ vs $37,632$) and lower wall time
# ($11.9\;\mathrm{s}$ vs $14.9\;\mathrm{s}$).
#
# Because fields are $z$-invariant in 2D, we set $\Delta z = 10 h$ across the
# $N_z = 4$ periodic slab so that $\Delta z$ adds negligible penalty ($< 0.25\%$)
# to the 3D Courant limit $\Delta t = 0.85 / \bigl(c\sqrt{2/h^2 + 1/\Delta z^2}\bigr)$.
#
# ### 3.2 Subpixel averaging for dispersive metals (`wakis.tensor_dispersive`)
#
# Yes—the Cartesian arm supports Kottke subpixel smoothing for dispersive solids!
# When `GridFIT3D(..., conformal="all")` is passed a `DispersiveMaterial` in
# `stl_materials`, `SolverFIT3D._setup_dispersive` builds a
# `wakis.tensor_dispersive.DispersiveNetwork`:
#
# * **Interior metal edges** ($f = 1$) carry a single dispersive capacitor.
# * **Cut boundary edges** ($0 < f < 1$) are decomposed into a **parallel
#   branch** (tangential component, volume-averaged permittivity
#   $f\varepsilon_\mathrm{metal}(\omega) + (1-f)\varepsilon_\mathrm{host}$) and a
#   **series branch** (normal component, harmonic-averaged inverse permittivity
#   $[f/\varepsilon_\mathrm{metal}(\omega) + (1-f)/\varepsilon_\mathrm{host}]^{-1}$)
#   weighted by the local STL surface normal $\hat{\mathbf{n}}$. Each sub-capacitor
#   maintains its own passive ADE pole states, updated together by solving a small
#   scalar linear equation per cut edge at each step.
#
# ### 3.3 Preflight with `wakis.estimate`

# %%
def layout_2d(dx_m, radius_m=20e-9, n_pml=9):
    """Grid and box positions for a 2D cylinder of radius `radius_m` at cell size `dx_m`."""
    i_a = int(np.ceil(radius_m / dx_m - 1e-9))
    i_in = i_a + 2
    i_tf = i_in + 2
    i_flux = i_tf + 2
    i_pml = i_flux + 3
    n_half = i_pml + n_pml
    nz = 4
    dz_m = 10.0 * dx_m
    return dict(
        dx=dx_m,
        dz=dz_m,
        nz=nz,
        n_pml=n_pml,
        n_half=n_half,
        n_cells=2 * n_half,
        half=n_half * dx_m,
        hz=0.5 * nz * dz_m,
        inner=i_in * dx_m,
        tf=i_tf * dx_m,
        flux=i_flux * dx_m,
        pml=i_pml * dx_m,
        radius_m=radius_m,
    )


def preflight_plan(dx_nm, radius_nm=20.0, t_run_fs=35.0, conformal="all", machine="laptop"):
    """Run `wakis.estimate.estimate` and `preflight` before allocating the solver."""
    lay = layout_2d(dx_nm * 1e-9, radius_m=radius_nm * 1e-9)
    n = lay["n_cells"]
    dt = 0.85 / (c0 * np.sqrt(2.0 / lay["dx"] ** 2 + 1.0 / lay["dz"] ** 2))
    steps = int(np.ceil(t_run_fs * 1e-15 / dt))
    n_flux = int(4 * (2 * lay["flux"] / lay["dx"]) * 3)
    n_freq = LAM_EVAL_NM.size
    est = estimate(
        (n, n, lay["nz"]),
        steps,
        backend="cpu",
        cpml=True,
        conformal=False if conformal == "staircase" else conformal,
        interface_cells=int(2 * 2 * np.pi * lay["radius_m"] / lay["dx"] * lay["nz"]),
        monitors=[(n_freq, n_flux, 4), (n_freq, n_flux, 4)],
    )
    problems = preflight(est, machine, raise_error=False)
    return lay, dt, steps, est, problems


MACHINE_TARGET = (
    dict(ram=12e9, gpu=0) if IN_COLAB else os.environ.get("WAKIS_MACHINE", "laptop")
)
for h_nm in (4.0, 2.0):
    lay_p, dt_p, steps_p, est_p, probs_p = preflight_plan(
        h_nm, radius_nm=0.5 * ARGS.diameter, conformal=CONFORMAL, machine=MACHINE_TARGET
    )
    status = "OK" if not probs_p else ("FAIL: " + "; ".join(probs_p))
    print(
        f"preflight h={h_nm:.1f} nm: {lay_p['n_cells']}x{lay_p['n_cells']}x{lay_p['nz']} "
        f"= {est_p.cells} cells, dt={dt_p*1e18:.2f} as, {steps_p} steps, "
        f"RAM={est_p.host_bytes/1e6:.1f} MB, est. time={est_p.seconds:.1f} s [{status}]"
    )

_, _, _, est_active, probs_active = preflight_plan(
    DX_NM, radius_nm=0.5 * ARGS.diameter, conformal=CONFORMAL, machine=MACHINE_TARGET
)
if probs_active:
    raise MemoryError("; ".join(probs_active))

# %% [markdown]
# ## 4. Running the 2D plasmonic cylinder and inspecting the near field
#
# As in Tutorial 01, we inject a broadband plane-wave pulse ($\lambda \in [300, 900]\text{ nm}$,
# $\mathbf{k} \parallel +x$, $\mathbf{E} \parallel \hat{\mathbf{y}}$, Wakis
# `polarization="y"` which corresponds to 2D `"TE"`) via `wakis.TFSFSource`. Two
# `FluxMonitor` boxes measure the scattered power per unit length ($P'_\mathrm{sca}$)
# outside the TF/SF boundary and the absorbed power per unit length
# ($P'_\mathrm{abs} = -P'_\mathrm{in}$) inside the total-field region, yielding
# $Q_\mathrm{sca}(\lambda)$, $Q_\mathrm{abs}(\lambda)$, and
# $Q_\mathrm{ext}(\lambda) = Q_\mathrm{sca}(\lambda) + Q_\mathrm{abs}(\lambda)$.

# %%
def run_plasmonic_cylinder(
    material,
    dx_nm=4.0,
    radius_nm=20.0,
    conformal="all",
    t_run_fs=35.0,
    record_dft_lam_nm=None,
):
    """Run one 2D FIT simulation of a dispersive cylinder in vacuum (TE polarization)."""
    lay = layout_2d(dx_nm * 1e-9, radius_m=radius_nm * 1e-9)
    t0 = time.time()
    cyl = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=lay["radius_m"],
        height=10.0 * lay["hz"],
        resolution=360,
        capping=True,
    ).triangulate()
    grid = wakis.GridFIT3D(
        -lay["half"],
        lay["half"],
        -lay["half"],
        lay["half"],
        -lay["hz"],
        lay["hz"],
        lay["n_cells"],
        lay["n_cells"],
        lay["nz"],
        stl_solids={"cylinder": cyl},
        stl_materials={"cylinder": material},
        conformal=False if conformal in (False, "staircase", None) else conformal,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=lay["n_pml"],
        use_stl=True,
        cfln=0.85,
        verbose=0,
    )
    setup_s = time.time() - t0

    f_min, f_max = c0 / 900e-9, c0 / 300e-9
    pulse = GaussianSource(
        frequency=0.5 * (f_min + f_max), fwidth=0.25 * (f_max - f_min)
    )
    src = wakis.TFSFSource(
        pulse,
        center=(0.0, 0.0, 0.0),
        size=(2.0 * lay["tf"], 2.0 * lay["tf"], None),
        direction="+x",
        polarization="y",
        amplitude=1.0,
    )
    lz = 2.0 * lay["dz"]
    scat = wakis.FluxMonitor(
        FREQS_EVAL, size=(2.0 * lay["flux"], 2.0 * lay["flux"], lz)
    )
    inner = wakis.FluxMonitor(
        FREQS_EVAL, size=(2.0 * lay["inner"], 2.0 * lay["inner"], lz)
    )
    monitors = [scat, inner]
    dft_mon = None
    if record_dft_lam_nm is not None:
        f_snap = np.atleast_1d(c0 / (np.asarray(record_dft_lam_nm, float) * 1e-9))
        span = 2.0 * lay["pml"]
        dft_mon = wakis.DFTMonitor(
            f_snap, components=("Ex", "Ey"), size=(span, span, 0.0)
        )
        monitors.append(dft_mon)

    n_steps = int(np.ceil(t_run_fs * 1e-15 / solver.dt))
    t1 = time.time()
    solver.emsolve(n_steps, source=[src], monitors=monitors, verbose=0)
    run_s = time.time() - t1

    phi_inc = src.incident_flux_density(FREQS_EVAL)
    width = 2.0 * lay["radius_m"]
    q_sca = scat.get_flux() / (lz * phi_inc) / width
    q_abs = -inner.get_flux() / (lz * phi_inc) / width
    q_ext = q_sca + q_abs

    n_disp_edges = 0
    n_caps = 0
    if getattr(solver, "dispersive_network", None) is not None:
        n_disp_edges = int(solver.dispersive_network.rows.size)
        n_caps = int(solver.dispersive_network.n_caps)
    elif getattr(solver, "_dispersive_groups", None):
        n_disp_edges = int(solver._dispersive_groups[0]["idx"].size)
        n_caps = n_disp_edges

    return dict(
        q_ext=q_ext,
        q_sca=q_sca,
        q_abs=q_abs,
        dt_s=float(solver.dt),
        steps=int(solver.Nt),
        cells=int(grid.Nx * grid.Ny * grid.Nz),
        n_disp_edges=n_disp_edges,
        n_caps=n_caps,
        setup_s=float(setup_s),
        run_s=float(run_s),
        dx_nm=float(dx_nm),
        conformal=conformal,
        dft_mon=dft_mon,
        src=src,
        lay=lay,
    )


if ARGS.arm:
    mat_arm = reconstruct_material(MODELS_DATA["metals"][ARGS.metal]["models"][ARGS.poles])
    res_arm = run_plasmonic_cylinder(
        mat_arm, dx_nm=DX_NM, radius_nm=0.5 * ARGS.diameter, conformal=CONFORMAL
    )
    p_arm = peak(LAM_EVAL_NM, res_arm["q_ext"])
    out_arm = dict(
        metal=ARGS.metal,
        poles=ARGS.poles,
        diameter_nm=ARGS.diameter,
        dx_nm=DX_NM,
        conformal=CONFORMAL,
        cells=res_arm["cells"],
        steps=res_arm["steps"],
        dt_s=res_arm["dt_s"],
        run_s=round(res_arm["run_s"], 3),
        lam_peak_nm=float(p_arm["lam_peak"]),
        q_peak=float(p_arm["height"]),
        fwhm_nm=float(p_arm["fwhm"]),
    )
    out_path = Path(
        ARGS.out or f"plasmon_{ARGS.metal}_{ARGS.poles}_{DX_NM:g}nm_{CONFORMAL}.json"
    )
    out_path.write_text(json.dumps(out_arm, indent=1))
    print(f"wrote {out_path}: lam_res={out_arm['lam_peak_nm']:.2f} nm, Q={out_arm['q_peak']:.3f}")
    sys.exit(0)

# In --quick or --full mode, execute a live Ag 40nm N=2 run with near-field DFT monitor
# at the resonance wavelength (343.2 nm) and off-resonance (550.0 nm)
if not PLOT_ONLY:
    mat_ag_n2 = reconstruct_material(MODELS_DATA["metals"]["Ag"]["models"]["N2"])
    snap_lams = [343.2, 550.0]
    live_run = run_plasmonic_cylinder(
        mat_ag_n2, dx_nm=4.0, radius_nm=20.0, conformal="all", record_dft_lam_nm=snap_lams
    )
    p_live = peak(LAM_EVAL_NM, live_run["q_ext"])
    mie_n2_ref = np.array(SPECTRA_DATA["Ag_40nm"]["models"]["N2"]["mie_cont_q_ext"])
    live_nrms = float(nrms(live_run["q_ext"], mie_n2_ref))
    print(
        f"Live quick run (Ag 40nm, N=2, h=4nm, conformal='all'): "
        f"{live_run['cells']} cells, {live_run['steps']} steps in {live_run['run_s']:.2f} s; "
        f"lam_res = {p_live['lam_peak']:.2f} nm, peak Q_ext = {p_live['height']:.3f}, "
        f"NRMS vs Mie(N=2) = {live_nrms:.2%}"
    )
    (DATA_DIR / "run_quick.json").write_text(
        json.dumps(
            dict(
                metal="Ag",
                poles="N2",
                diameter_nm=40.0,
                dx_nm=4.0,
                conformal="all",
                cells=live_run["cells"],
                steps=live_run["steps"],
                dt_s=live_run["dt_s"],
                run_s=round(live_run["run_s"], 3),
                lam_peak_nm=round(float(p_live["lam_peak"]), 4),
                q_peak=round(float(p_live["height"]), 4),
                fwhm_nm=round(float(p_live["fwhm"]), 4),
                nrms_vs_mie_n2=round(live_nrms, 5),
            ),
            indent=1,
        )
    )


def plot_near_field_comparison():
    """Render 3-panel near-field LSPR comparison: side-by-side meshes, Cartesian |E|/|E_inc|, and body-fitted triangular |E|/|E_inc|."""
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=14.6, height_in=4.5)

    # Panel (a): Side-by-side mesh comparison in [-38, 38] nm
    ax0 = axes[0]
    xs_c = np.arange(-38.0, 0.1, 2.0)
    ys_c = np.arange(-38.0, 38.1, 2.0)
    segs_c = [[(xv, -38.0), (xv, 38.0)] for xv in xs_c] + [
        [(-38.0, yv), (0.0, yv)] for yv in ys_c
    ]
    ax0.add_collection(
        LineCollection(segs_c, colors="#64748B", linewidths=0.6, alpha=0.65, zorder=2)
    )

    nodes_nm = TRI_NPZ["nodes_nm"]
    cells_t = TRI_NPZ["cells"]
    cent_nm = nodes_nm[cells_t].mean(axis=1)
    rmask = (cent_nm[:, 0] >= -1.0) & (np.abs(cent_nm[:, 0]) <= 39.5) & (np.abs(cent_nm[:, 1]) <= 39.5)
    segs_t = []
    for tri_c in cells_t[rmask]:
        p0, p1, p2 = nodes_nm[tri_c[0]], nodes_nm[tri_c[1]], nodes_nm[tri_c[2]]
        for pa, pb in ((p0, p1), (p1, p2), (p2, p0)):
            if pa[0] >= -0.5 and pb[0] >= -0.5:
                segs_t.append([pa, pb])
    ax0.add_collection(
        LineCollection(segs_t, colors="#0F766E", linewidths=0.6, alpha=0.7, zorder=2)
    )
    ax0.add_patch(
        Circle(
            (0.0, 0.0),
            20.0,
            facecolor="#CBD5E1",
            edgecolor="#0F172A",
            lw=1.6,
            alpha=0.38,
            zorder=4,
        )
    )
    ax0.axvline(0.0, color="#0F172A", ls="--", lw=1.1, zorder=5)
    ax0.text(
        -19.0,
        34.0,
        "Cartesian ($h = 2.0$ nm)\n$12,544$ cells, $14.9$ s\n$11.2\\;\\mathrm{cells}/\\delta_\\mathrm{min}$",
        ha="center",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.92},
        zorder=8,
    )
    ax0.text(
        19.0,
        34.0,
        "Triangles ($h_s = 1.25$ nm)\n$7,097$ tris, $26.5$ s\n$18.0\\;\\mathrm{elems}/\\delta_\\mathrm{min}$",
        ha="center",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        color="#0F766E",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#0F766E", "alpha": 0.92},
        zorder=8,
    )
    ax0.text(
        0.0,
        0.0,
        "Ag ($2a = 40$ nm)\n$\\delta_\\mathrm{min} = 22.5$ nm",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=figstyle.INK,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        zorder=8,
    )
    ax0.set_xlim(-38.0, 38.0)
    ax0.set_ylim(-38.0, 38.0)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("$x$ [nm]")
    ax0.set_ylabel("$y$ [nm]")
    ax0.set_title("(a) Meshes: Cartesian vs body-fitted triangles")

    # Panel (b): Cartesian (h = 2.0 nm) near-field |E|/|E_inc| at LSPR (343.2 nm)
    ax1 = axes[1]
    xc2 = TRI_NPZ["cart2_x_nm"]
    yc2 = TRI_NPZ["cart2_y_nm"]
    ec2 = TRI_NPZ["cart2_e_lspr"]
    tf_nm, in_nm, fl_nm = TRI_NPZ["cart2_boxes_nm"]
    im1 = ax1.pcolormesh(
        xc2, yc2, ec2.T, cmap="inferno", vmin=0.0, vmax=5.5, shading="auto"
    )
    ax1.add_patch(
        Circle((0.0, 0.0), 20.0, fill=False, edgecolor="white", lw=1.5, zorder=5)
    )
    ax1.add_patch(
        Rectangle(
            (-tf_nm, -tf_nm),
            2 * tf_nm,
            2 * tf_nm,
            fill=False,
            ec="#D55E00",
            ls="--",
            lw=1.1,
            zorder=5,
        )
    )
    ax1.add_patch(
        Rectangle(
            (-fl_nm, -fl_nm),
            2 * fl_nm,
            2 * fl_nm,
            fill=False,
            ec="#38BDF8",
            ls=":",
            lw=1.1,
            zorder=5,
        )
    )
    ax1.text(
        0.0,
        0.0,
        "Ag ($N=2$)",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=figstyle.INK,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        zorder=8,
    )
    ax1.text(
        -tf_nm + 1.5,
        tf_nm - 1.5,
        "TF/SF",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="#D55E00",
        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=8,
    )
    ax1.text(
        fl_nm - 1.5,
        -fl_nm + 1.5,
        "FluxMonitor",
        ha="right",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color="#0284C7",
        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=8,
    )
    ax1.set_xlim(-38.0, 38.0)
    ax1.set_ylim(-38.0, 38.0)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("$x$ [nm]")
    ax1.set_ylabel("$y$ [nm]")
    ax1.set_title("(b) Cartesian ($h=2$ nm) LSPR ($\\lambda=343.2$ nm)")
    fig.colorbar(im1, ax=ax1, label=r"$|\mathbf{E}| / |\mathbf{E}_\mathrm{inc}|$", shrink=0.84)

    # Panel (c): Body-fitted triangular mesh (h_s = 1.25 nm) near-field |E|/|E_inc| at LSPR (343.2 nm)
    ax2 = axes[2]
    triang = mtri.Triangulation(nodes_nm[:, 0], nodes_nm[:, 1], cells_t)
    im2 = ax2.tripcolor(
        triang,
        TRI_NPZ["e_lspr"],
        cmap="inferno",
        vmin=0.0,
        vmax=5.5,
        shading="flat",
    )
    ax2.triplot(triang, color="white", lw=0.22, alpha=0.28, zorder=3)
    ax2.add_patch(
        Circle((0.0, 0.0), 20.0, fill=False, edgecolor="white", lw=1.6, zorder=5)
    )
    ax2.text(
        0.0,
        0.0,
        "Ag ($N=2$)\n$|\\mathbf{E}|/|\\mathbf{E}_0| = 3.93$",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=figstyle.INK,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        zorder=8,
    )
    nf_stats = TRIANGLE_DATA["meshes"]["tri_h1p25nm_N2"]["near_field_stats"]
    ax2.text(
        0.03,
        0.03,
        f"Dipole lobes ($y = \\pm 20$ nm): max $|\\mathbf{{E}}|/|\\mathbf{{E}}_0| = {nf_stats['lspr_343p2nm_max_outside']:.2f}$\n"
        f"Off-res ($550$ nm): $|\\mathbf{{E}}|_\\mathrm{{in}}/|\\mathbf{{E}}_0| = {nf_stats['off_550nm_mean_in_cyl']:.2f}$ (shielded)",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=figstyle.INK,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#94A3B8", "alpha": 0.9},
        zorder=8,
    )
    ax2.set_xlim(-38.0, 38.0)
    ax2.set_ylim(-38.0, 38.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("$x$ [nm]")
    ax2.set_ylabel("$y$ [nm]")
    ax2.set_title("(c) Body-fitted triangles ($h_s=1.25$ nm) LSPR")
    fig.colorbar(im2, ax=ax2, label=r"$|\mathbf{E}| / |\mathbf{E}_\mathrm{inc}|$", shrink=0.84)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.14, top=0.88, wspace=0.28)
    savefig(fig, "near_field_lspr.png")
    plt.show()


plot_near_field_comparison()

# %% [markdown]
# ## 5. Extinction spectra $Q_\mathrm{ext}(\lambda)$: Body-fitted triangles vs Cartesian FIT, Mie, and $80\text{ nm}$ retardation
#
# Figure 3 (`qext_spectra_vs_mie.png`) compares the simulated Wakis FIT extinction
# efficiency $Q_\mathrm{ext}(\lambda)$ against exact 2D Mie theory
# (`cylinder_efficiencies(..., polarization="TE")`):
#
# * **Panel (a) — Ag $2a = 40\text{ nm}$ across pole count $N$ (Body-fitted Triangles vs Cartesian FIT vs Mie) + constant-$\varepsilon$ baseline:**
#   Pure Drude ($N=1$) places the resonance $+6.5\text{ nm}$ too red
#   ($\lambda_\mathrm{res} = 347.8\text{ nm}$ vs J&C's $341.3\text{ nm}$) and
#   broadens it by $+35\%$. Adding just **one UV Lorentz pole ($N=2$)** shifts
#   $\lambda_\mathrm{res}$ to $343.20\text{ nm}$ (Mie `N2`), and $N = 2, 3, 4, 6$
#   lie virtually on top of one another. On the **body-fitted triangular mesh**
#   (`tri_h2nm`, $4,448$ triangles, $h_s = 2\text{ nm}$), the $N=2$ curve
#   (solid stars, $\lambda_\mathrm{res} = 343.09\text{ nm}$, $Q_\mathrm{peak} = 4.986$)
#   tracks exact `Mie(N2)` to **$0.28\%$ NRMS** across $300\text{--}900\text{ nm}$
#   without the small cut-cell spikes seen on the $h = 2\text{ nm}$ Cartesian grid
#   ($2.47\%$ disc. NRMS). A constant-$\varepsilon(\lambda_\mathrm{res})$ model
#   (dashed orange) matches `Mie(J&C)` *only at that single wavelength* ($\mathrm{NRMS} = 99.6\%$).
# * **Panel (b) — Au $2a = 40\text{ nm}$ across pole count $N$ (Cartesian $h = 4\text{ nm}$):**
#   Because gold's $5d \to 6sp$ interband transition overlaps its plasmon peak
#   ($\lambda_\mathrm{res} = 496.0\text{ nm}$), $N=1$ (Drude) produces an
#   unphysical peak at $399\text{ nm}$ ($-97\text{ nm}$ error), $N=2$ is still
#   shifted by $-35\text{ nm}$, and $N \ge 3\text{--}6$ accurately reproduces
#   both the $496\text{ nm}$ plasmon peak and the $< 450\text{ nm}$ interband
#   absorption plateau.
# * **Panel (c) — Retardation and size effect ($2a = 40\text{ nm}$ vs $80\text{ nm}$ Ag):**
#   In the quasi-static limit ($\mathrm{Re}\,\varepsilon = -1$), the peak sits at
#   $338.0\text{ nm}$ ($\mathrm{FWHM} = 10.9\text{ nm}$). At $2a = 40\text{ nm}$
#   it red-shifts to $341.3\text{ nm}$ ($\mathrm{FWHM} = 19.2\text{ nm}$).
#   Doubling the diameter to $2a = 80\text{ nm}$ red-shifts the combined
#   $Q_\mathrm{ext}$ peak to $342.8\text{ nm}$ and the retarded dipole
#   $Q_\mathrm{sca}$ peak to $348.1\text{ nm}$, broadens the resonance to
#   $\mathrm{FWHM} = 44.2\text{ nm}$ via radiation damping, and excites a distinct
#   **higher-order $n = 2$ quadrupole plasmon** shoulder at $335.5\text{ nm}$.

# %%
fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=14.6, height_in=4.4)

# Panel (a): Ag 40 nm: Body-fitted Triangles (h_s=2nm) + Cartesian FIT (h=2nm) vs Mie + constant-eps
ax = axes[0]
ag40 = SPECTRA_DATA["Ag_40nm"]
c_eps = ag40["mie_constant_eps_at_res"]
tri_n2 = TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"]["N2"]
ax.plot(
    LAM_EVAL_NM,
    ag40["mie_jc"]["q_ext"],
    "k-",
    lw=2.1,
    label=f"Mie (J&C): $\\lambda_\\mathrm{{res}}={ag40['mie_jc']['lam_peak_nm']:.1f}$ nm",
    zorder=6,
)
ax.plot(
    LAM_EVAL_NM,
    c_eps["q_ext"],
    color="#E69F00",
    ls="--",
    lw=1.3,
    alpha=0.85,
    label=f"Const. $\\varepsilon(\\lambda_\\mathrm{{res}})$ (NRMS {c_eps['nrms_vs_mie_jc']*100:.0f}%)",
)
for k in ("N1", "N2", "N6", "Rakic"):
    m_entry = ag40["models"][k]
    fit_h2 = m_entry["resolutions"]["h_2nm"]
    lam_pk = fit_h2["total_error_vs_mie_jc"]["lam_peak_nm"]
    ax.plot(
        LAM_EVAL_NM,
        m_entry["mie_cont_q_ext"],
        color=MODEL_COLORS[k],
        ls="-",
        lw=1.1,
        alpha=0.75,
    )
    ax.plot(
        LAM_EVAL_NM[::4],
        fit_h2["q_ext"][::4],
        ls="none",
        marker="o",
        ms=2.8,
        color=MODEL_COLORS[k],
        alpha=0.8,
        label=f"Cart. $h=2$ nm {k} ($\\lambda_\\mathrm{{res}}={lam_pk:.1f}$ nm)",
    )
ax.plot(
    LAM_EVAL_NM[::3],
    np.array(tri_n2["q_ext"], dtype=float)[::3],
    ls="none",
    marker="*",
    ms=5.2,
    color="#004D40",
    zorder=7,
    label=(
        f"Tri. $h_s=2$ nm $N=2$ ($\\lambda_\\mathrm{{res}}={tri_n2['lam_peak_nm']:.1f}$ nm, "
        f"disc. {tri_n2['discretization_error_vs_mie_cont']['nrms']*100:.2f}%)"
    ),
)
ax.set_xlim(300, 560)
ax.set_ylim(0, 6.2)
ax.set_xlabel(r"wavelength $\lambda$ [nm]")
ax.set_ylabel(r"extinction efficiency $Q_\mathrm{ext} = C_\mathrm{ext}/(2a)$")
ax.set_title("(a) Ag ($2a = 40$ nm): Tri. vs Cart. & Mie")
ax.legend(fontsize=6.5, loc="upper right")

# Panel (b): Au 40 nm FIT (h=4nm) vs Mie for N=1,2,3,4,6,Rakic + Mie(J&C)
ax = axes[1]
au40 = SPECTRA_DATA["Au_40nm"]
ax.plot(
    LAM_EVAL_NM,
    au40["mie_jc"]["q_ext"],
    "k-",
    lw=2.1,
    label=f"Mie (J&C): $\\lambda_\\mathrm{{res}}={au40['mie_jc']['lam_peak_nm']:.1f}$ nm",
    zorder=6,
)
ax.axvline(au40["mie_jc"]["lam_peak_nm"], color="k", ls=":", lw=1.0)
for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
    m_entry = au40["models"][k]
    fit_h4 = m_entry["h_4nm"]
    lam_pk = fit_h4["total_error_vs_mie_jc"]["lam_peak_nm"]
    ax.plot(
        LAM_EVAL_NM,
        m_entry["mie_cont_q_ext"],
        color=MODEL_COLORS[k],
        ls="-",
        lw=1.1,
        alpha=0.75,
    )
    ax.plot(
        LAM_EVAL_NM[::4],
        fit_h4["q_ext"][::4],
        ls="none",
        marker="o",
        ms=2.8,
        color=MODEL_COLORS[k],
        label=f"FIT {MODEL_LABELS[k]}: $\\lambda_\\mathrm{{res}}={lam_pk:.1f}$ nm",
    )
ax.set_xlim(300, 750)
ax.set_ylim(0, 3.2)
ax.set_xlabel(r"wavelength $\lambda$ [nm]")
ax.set_ylabel(r"extinction efficiency $Q_\mathrm{ext}$")
ax.set_title("(b) Au ($2a = 40$ nm): Interband ($N \\geq 3$)")
ax.legend(fontsize=6.6, loc="upper right")

# Panel (c): Retardation red-shift & quadrupole emergence (Ag 40 nm vs 80 nm)
ax = axes[2]
ret = SPECTRA_DATA["Ag_80nm_retardation"]
ax.plot(
    LAM_EVAL_NM,
    ret["quasi_static_40nm"]["q_ext"],
    color="0.45",
    ls=":",
    lw=1.5,
    label=f"Quasi-static ($2a=40$ nm): $\\lambda_\\mathrm{{res}}={ret['quasi_static_40nm']['lam_peak_nm']:.1f}$ nm",
)
ax.plot(
    LAM_EVAL_NM,
    ag40["mie_jc"]["q_ext"],
    color="#0072B2",
    ls="-",
    lw=1.8,
    label=f"Mie $2a=40$ nm: $\\lambda_\\mathrm{{res}}={ag40['mie_jc']['lam_peak_nm']:.1f}$ nm, FWHM={ag40['mie_jc']['fwhm_nm']:.1f} nm",
)
ax.plot(
    LAM_EVAL_NM,
    ret["mie_jc_80nm"]["q_ext"],
    color="#D55E00",
    ls="-",
    lw=1.8,
    label=f"Mie $2a=80$ nm: $\\lambda_\\mathrm{{res}}={ret['mie_jc_80nm']['q_ext_lam_peak_nm']:.1f}$ nm, FWHM={ret['mie_jc_80nm']['q_ext_fwhm_nm']:.1f} nm",
)
ax.plot(
    LAM_EVAL_NM[::3],
    ret["fit_n2_80nm_h4nm"]["q_ext"][::3],
    "o",
    ms=3.0,
    color="#D55E00",
    label=f"FIT $2a=80$ nm ($N=2, h=4$ nm): NRMS={ret['fit_n2_80nm_h4nm']['nrms_vs_mie_n2']*100:.1f}%",
)
ax.annotate(
    "Quadrupole ($n=2$)\n335.5 nm",
    xy=(335.5, 4.95),
    xytext=(302.0, 5.55),
    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "k"},
    fontsize=7.3,
)
ax.annotate(
    "Retarded dipole ($n=1$)\n$Q_\\mathrm{sca}$ 348.1 nm ($Q_\\mathrm{ext}$ 342.8 nm)",
    xy=(348.1, 4.92),
    xytext=(382.0, 3.35),
    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#D55E00"},
    fontsize=7.2,
)
ax.set_xlim(300, 500)
ax.set_ylim(0, 6.6)
ax.set_xlabel(r"wavelength $\lambda$ [nm]")
ax.set_ylabel(r"extinction efficiency $Q_\mathrm{ext}$")
ax.set_title("(c) Retardation: Ag $2a=40$ vs $80$ nm")
ax.legend(fontsize=6.4, loc="upper right")

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.88, wspace=0.24)
savefig(fig, "qext_spectra_vs_mie.png")
plt.show()

# %% [markdown]
# ## 6. Separating model error, discretization error, and total error (Cartesian vs Body-Fitted Triangles)
#
# A time-domain plasmonic simulation contains two distinct sources of error
# that are often conflated:
#
# 1. **Model error ($\text{Mie}(\varepsilon_\mathrm{fit})$ vs $\text{Mie}(\varepsilon_\mathrm{J\&C})$):**
#    How well does the $N$-pole rational function approximate the experimental
#    permittivity? Both sides are evaluated with the exact cylinder series
#    (`cylinder_efficiencies`), so mesh discretization is completely absent.
#    The baseline denominator is `Mie(J&C)` ($\lambda_\mathrm{res} = 341.33\text{ nm}$,
#    $Q_\mathrm{ext} = 4.898$, $\mathrm{FWHM} = 19.23\text{ nm}$ for Ag $2a = 40\text{ nm}$;
#    $\lambda_\mathrm{res} = 496.01\text{ nm}$, $Q_\mathrm{ext} = 1.988$,
#    $\mathrm{FWHM} = 70.95\text{ nm}$ for Au $2a = 40\text{ nm}$).
# 2. **Discretization error ($\text{FIT}$ vs $\text{Mie}(\varepsilon_\mathrm{fit})$):**
#    How well does the spatial mesh—either the 3D Cartesian FIT grid ($h = 4\text{ nm}$
#    or $h = 2\text{ nm}$ with Kottke subpixel averaging) or the 2D body-fitted
#    triangular FIT mesh (`tri_h2nm` at $h_s = 2\text{ nm}$ and `tri_h1p25nm` at
#    $h_s = 1.25\text{ nm}$ with area-preserving polygon vertices)—solve Maxwell's
#    equations for *that exact same pole model*? Because $\Delta t \le 8\text{ as}$ is
#    $> 140\times$ smaller than the optical period at $341\text{ nm}$ ($1,138\text{ as}$),
#    $|\varepsilon_\mathrm{disc} - \varepsilon_\mathrm{cont}|$ shifts $\lambda_\mathrm{res}$
#    by $\le 0.03\text{ nm}$: spatial meshing of the curved metal--vacuum interface
#    dominates the discretization error. Eliminating mixed metal--vacuum cut cells
#    via the body-fitted triangular mesh drops the $N=2$ discretization NRMS from
#    **$5.46\%$ ($h = 4\text{ nm}$ Cart.)** and **$2.47\%$ ($h = 2\text{ nm}$ Cart.)**
#    to **$0.28\%$ (`tri_h2nm`)**, and reduces the peak $Q_\mathrm{ext}$ error from
#    $-5.92\%$ to **$-0.20\%$** ($Q_\mathrm{peak} = 4.986$ vs Mie `N2` $4.996$).
# 3. **Total error ($\text{FIT}$ vs $\text{Mie}(\varepsilon_\mathrm{J\&C})$):**
#    The end-to-end error against the experimental Johnson & Christy Mie reference
#    (baseline denominator: `Mie(J&C)`).
#
# **Table 2 — Error separation for the $2a = 40\text{ nm}$ Ag cylinder across Cartesian (`h = 4 nm`, `h = 2 nm`) and Body-Fitted Triangular (`tri_h2nm`) meshes:**
#
# | Model ($N$) | Error component (baseline denominator) | $\Delta\lambda_\mathrm{res}$ [nm] (%) | $\Delta Q_\mathrm{ext,peak}$ (%) | $\Delta\mathrm{FWHM}$ [nm] (%) | Spectrum NRMS (%) |
# |---|---|---:|---:|---:|---:|
# | **`N1` (1)** | Model: `Mie(N1)` vs `Mie(J&C)` | $+6.35$ ($+1.86\%$) | $+4.12\%$ | $+4.59$ ($+23.89\%$) | $14.54\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(N1)` | $-0.10$ / $+0.13$ | $-8.63\%$ / $-6.70\%$ | $+3.01$ / $+2.14$ | $3.44\%$ / $1.88\%$ |
# | | **Disc Tri ($h_s=2\text{ nm}$): `UnstructuredFIT2D` vs `Mie(N1)`** | **$-0.09$ ($-0.03\%$)** | **$+0.13\%$** | **$+0.10$ ($+0.42\%$)** | **$0.29\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+6.25$ / $+6.49$ / $+6.26$** | **$-4.87\%$ / $-2.86\%$ / $+4.25\%$** | **$+7.60$ / $+6.74$ / $+4.69$** | **$14.69\%$ / $14.36\%$ / $14.43\%$** |
# | **`N2` (2)** | Model: `Mie(N2)` vs `Mie(J&C)` | $+1.87$ ($+0.55\%$) | $+2.02\%$ | $+0.41$ ($+2.11\%$) | $3.91\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(N2)` | $+0.12$ / $+0.38$ | $-9.97\%$ / $-5.92\%$ | $+2.97$ / $+1.70$ | $5.46\%$ / $2.47\%$ |
# | | **Disc Tri ($h_s=2$ / $1.25\text{ nm}$): `UnstructuredFIT2D` vs `Mie(N2)`** | **$-0.11$ / $-0.11$ ($-0.03\%$)** | **$-0.20\%$ / $-0.62\%$** | **$+0.13$ / $+0.23$** | **$0.28\%$ / $0.35\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+1.99$ / $+2.26$ / $+1.77$** | **$-8.15\%$ / $-4.02\%$ / $+1.82\%$** | **$+3.37$ / $+2.11$ / $+0.54$** | **$7.21\%$ / $5.03\%$ / $3.65\%$** |
# | **`N3` (3)** | Model: `Mie(N3)` vs `Mie(J&C)` | $+1.96$ ($+0.57\%$) | $+3.25\%$ | $-0.04$ ($-0.21\%$) | $4.14\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(N3)` | $+0.36$ / $+0.30$ | $-9.75\%$ / $-6.40\%$ | $+2.98$ / $+1.73$ | $4.34\%$ / $2.26\%$ |
# | | **Disc Tri ($h_s=2\text{ nm}$): `UnstructuredFIT2D` vs `Mie(N3)`** | **$-0.11$ ($-0.03\%$)** | **$-0.26\%$** | **$+0.13$ ($+0.68\%$)** | **$0.29\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+2.32$ / $+2.26$ / $+1.85$** | **$-6.82\%$ / $-3.36\%$ / $+2.98\%$** | **$+2.94$ / $+1.69$ / $+0.09$** | **$6.51\%$ / $5.07\%$ / $3.86\%$** |
# | **`N4` (4)** | Model: `Mie(N4)` vs `Mie(J&C)` | $+1.94$ ($+0.57\%$) | $+4.12\%$ | $-0.33$ ($-1.71\%$) | $4.12\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(N4)` | $+0.30$ / $+0.32$ | $-9.70\%$ / $-6.64\%$ | $+3.00$ / $+1.75$ | $3.92\%$ / $2.20\%$ |
# | | **Disc Tri ($h_s=2\text{ nm}$): `UnstructuredFIT2D` vs `Mie(N4)`** | **$-0.11$ ($-0.03\%$)** | **$-0.30\%$** | **$+0.14$ ($+0.74\%$)** | **$0.29\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+2.24$ / $+2.26$ / $+1.83$** | **$-5.98\%$ / $-2.79\%$ / $+3.81\%$** | **$+2.67$ / $+1.42$ / $-0.19$** | **$6.18\%$ / $4.98\%$ / $3.83\%$** |
# | **`N6` (6)** | Model: `Mie(N6)` vs `Mie(J&C)` | $+1.92$ ($+0.56\%$) | $+5.14\%$ | $-0.64$ ($-3.32\%$) | $4.13\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(N6)` | $+0.31$ / $+0.37$ | $-10.00\%$ / $-6.79\%$ | $+3.02$ / $+1.76$ | $3.99\%$ / $2.11\%$ |
# | | **Disc Tri ($h_s=2\text{ nm}$): `UnstructuredFIT2D` vs `Mie(N6)`** | **$-0.11$ ($-0.03\%$)** | **$-0.35\%$** | **$+0.14$ ($+0.75\%$)** | **$0.30\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+2.23$ / $+2.29$ / $+1.81$** | **$-5.37\%$ / $-1.99\%$ / $+4.77\%$** | **$+2.38$ / $+1.12$ / $-0.50$** | **$6.17\%$ / $4.90\%$ / $3.83\%$** |
# | **`Rakic` (6)** | Model: `Mie(Rakic)` vs `Mie(J&C)` | $+9.38$ ($+2.75\%$) | $-36.09\%$ | $+20.31$ ($+105.6\%$) | $14.48\%$ |
# | | Disc Cart ($h=4$ / $h=2\text{ nm}$): `FIT` vs `Mie(Rakic)` | $+0.60$ / $+0.41$ | $-4.30\%$ / $-2.90\%$ | $+2.30$ / $+1.25$ | $4.63\%$ / $2.05\%$ |
# | | **Disc Tri ($h_s=2\text{ nm}$): `UnstructuredFIT2D` vs `Mie(Rakic)`** | **$-0.09$ ($-0.03\%$)** | **$+0.44\%$** | **$-0.02$ ($-0.05\%$)** | **$0.29\%$** |
# | | **Total ($h=4$ / $h=2$ / Tri $h_s=2\text{ nm}$) vs `Mie(J&C)`** | **$+9.98$ / $+9.79$ / $+9.29$** | **$-38.84\%$ / $-37.94\%$ / $-35.81\%$** | **$+22.61$ / $+21.56$ / $+20.29$** | **$15.34\%$ / $14.86\%$ / $14.45\%$** |

# %%
print("\n=== Ag 40nm Error Separation Summary (Cartesian vs Body-Fitted Triangles) ===")
print(
    f"{'Model':<7} {'Mod dlam':>9} {'Disc4 dlam':>10} {'Disc2 dlam':>10} {'Tri2 dlam':>10} "
    f"{'Mod NRMS%':>10} {'Disc4 NRMS%':>11} {'Disc2 NRMS%':>11} {'Tri2 NRMS%':>10} {'TotTri NRMS%':>12}"
)
for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
    m_ag = SPECTRA_DATA["Ag_40nm"]["models"][k]
    mod_e = m_ag["model_error_vs_mie_jc"]
    d4_e = m_ag["resolutions"]["h_4nm"]["discretization_error_vs_mie_cont"]
    d2_e = m_ag["resolutions"]["h_2nm"]["discretization_error_vs_mie_cont"]
    tri_m = TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"][k]
    dtri_e = tri_m["discretization_error_vs_mie_cont"]
    ttri_e = tri_m["total_error_vs_mie_jc"]
    print(
        f"{k:<7} {mod_e['dlam_nm']:>+9.2f} {d4_e['dlam_nm']:>+10.2f} {d2_e['dlam_nm']:>+10.2f} "
        f"{dtri_e['dlam_nm']:>+10.2f} {mod_e['nrms']*100:>10.2f} "
        f"{d4_e['nrms']*100:>11.2f} {d2_e['nrms']*100:>11.2f} "
        f"{dtri_e['nrms']*100:>10.2f} {ttri_e['nrms']*100:>12.2f}"
    )

print("\n=== Au 40nm Error Separation Summary (Cartesian h = 4 nm) ===")
print(
    f"{'Model':<7} {'Mod dlam[nm]':>13} {'Disc4 dlam':>11} {'Tot4 dlam':>10} "
    f"{'Mod NRMS%':>10} {'Disc4 NRMS%':>12} {'Tot4 NRMS%':>11}"
)
for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
    m_au = SPECTRA_DATA["Au_40nm"]["models"][k]
    mod_e = m_au["model_error_vs_mie_jc"]
    d4_e = m_au["h_4nm"]["discretization_error_vs_mie_cont"]
    t4_e = m_au["h_4nm"]["total_error_vs_mie_jc"]
    print(
        f"{k:<7} {mod_e['dlam_nm']:>+13.2f} {d4_e['dlam_nm']:>+11.2f} {t4_e['dlam_nm']:>+10.2f} "
        f"{mod_e['nrms']*100:>10.2f} {d4_e['nrms']*100:>12.2f} {t4_e['nrms']*100:>11.2f}"
    )

# %%
# Figure 4: Error separation (Model / Discretization / Total) vs pole count N (1x3 layout)
pole_keys = ["N1", "N2", "N3", "N4", "N6"]
ns_arr = np.array([1, 2, 3, 4, 6])

fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=14.6, height_in=4.4)

# (a) Ag |Delta lambda_res| [nm] vs N (Model, Cart h=4, Cart h=2, Tri h_s=2)
ax = axes[0]
ag_mod_dlam = [abs(SPECTRA_DATA["Ag_40nm"]["models"][k]["model_error_vs_mie_jc"]["dlam_nm"]) for k in pole_keys]
ag_d4_dlam = [abs(SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_4nm"]["discretization_error_vs_mie_cont"]["dlam_nm"]) for k in pole_keys]
ag_d2_dlam = [abs(SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_2nm"]["discretization_error_vs_mie_cont"]["dlam_nm"]) for k in pole_keys]
ag_dtri_dlam = [abs(TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"][k]["discretization_error_vs_mie_cont"]["dlam_nm"]) for k in pole_keys]
ag_t2_dlam = [abs(SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_2nm"]["total_error_vs_mie_jc"]["dlam_nm"]) for k in pole_keys]
ag_ttri_dlam = [abs(TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"][k]["total_error_vs_mie_jc"]["dlam_nm"]) for k in pole_keys]

ax.plot(ns_arr, ag_mod_dlam, "s-", color="#0072B2", lw=1.8, ms=5.5, label="Model: |Mie(fit) $-$ Mie(J&C)|")
ax.plot(ns_arr, ag_d4_dlam, "^--", color="#009E73", lw=1.3, ms=5.0, label="Disc. Cart ($h=4$ nm) vs Mie(fit)")
ax.plot(ns_arr, ag_d2_dlam, "v:", color="#009E73", lw=1.5, ms=5.0, label="Disc. Cart ($h=2$ nm) vs Mie(fit)")
ax.plot(ns_arr, ag_dtri_dlam, "*-", color="#004D40", lw=1.8, ms=6.5, label="Disc. Tri ($h_s=2$ nm) vs Mie(fit)")
ax.plot(ns_arr, ag_t2_dlam, "o--", color="#D55E00", lw=1.5, ms=5.0, label="Total Cart ($h=2$ nm) vs Mie(J&C)")
ax.plot(ns_arr, ag_ttri_dlam, "o-", color="#D55E00", lw=2.0, ms=5.8, label="Total Tri ($h_s=2$ nm) vs Mie(J&C)")
ax.set_xlabel("number of Drude + Lorentz poles $N$")
ax.set_ylabel(r"$|\Delta\lambda_\mathrm{res}|$ [nm]")
ax.set_title(r"(a) Ag $2a=40$ nm: $|\Delta\lambda_\mathrm{res}|$ vs $N$")
ax.set_xticks(ns_arr)
ax.legend(fontsize=6.6, loc="upper right")

# (b) Ag spectrum NRMS (%) vs N (Model, Cart h=4, Cart h=2, Tri h_s=2)
ax = axes[1]
ag_mod_nrms = [SPECTRA_DATA["Ag_40nm"]["models"][k]["model_error_vs_mie_jc"]["nrms"] * 100 for k in pole_keys]
ag_d4_nrms = [SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_4nm"]["discretization_error_vs_mie_cont"]["nrms"] * 100 for k in pole_keys]
ag_d2_nrms = [SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_2nm"]["discretization_error_vs_mie_cont"]["nrms"] * 100 for k in pole_keys]
ag_dtri_nrms = [TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"][k]["discretization_error_vs_mie_cont"]["nrms"] * 100 for k in pole_keys]
ag_t2_nrms = [SPECTRA_DATA["Ag_40nm"]["models"][k]["resolutions"]["h_2nm"]["total_error_vs_mie_jc"]["nrms"] * 100 for k in pole_keys]
ag_ttri_nrms = [TRIANGLE_DATA["meshes"]["tri_h2nm"]["models"][k]["total_error_vs_mie_jc"]["nrms"] * 100 for k in pole_keys]

ax.plot(ns_arr, ag_mod_nrms, "s-", color="#0072B2", lw=1.8, ms=5.5, label="Model NRMS: Mie(fit) vs Mie(J&C)")
ax.plot(ns_arr, ag_d4_nrms, "^--", color="#009E73", lw=1.3, ms=5.0, label="Disc. NRMS Cart ($h=4$ nm)")
ax.plot(ns_arr, ag_d2_nrms, "v:", color="#009E73", lw=1.5, ms=5.0, label="Disc. NRMS Cart ($h=2$ nm)")
ax.plot(ns_arr, ag_dtri_nrms, "*-", color="#004D40", lw=2.0, ms=6.8, label="Disc. NRMS Tri ($h_s=2$ nm, $\\approx 0.29\\%$)")
ax.plot(ns_arr, ag_t2_nrms, "o--", color="#D55E00", lw=1.5, ms=5.0, label="Total NRMS Cart ($h=2$ nm)")
ax.plot(ns_arr, ag_ttri_nrms, "o-", color="#D55E00", lw=2.0, ms=5.8, label="Total NRMS Tri ($h_s=2$ nm)")
ax.set_xlabel("number of Drude + Lorentz poles $N$")
ax.set_ylabel("relative error [%]")
ax.set_title(r"(b) Ag $2a=40$ nm: $Q_\mathrm{ext}$ NRMS error vs $N$")
ax.set_xticks(ns_arr)
ax.legend(fontsize=6.6, loc="upper right")

# (c) Au vs Ag: log-scale |Delta lambda_res| [nm] and Q_ext NRMS (%) vs N
ax = axes[2]
au_mod_dlam = [abs(SPECTRA_DATA["Au_40nm"]["models"][k]["model_error_vs_mie_jc"]["dlam_nm"]) for k in pole_keys]
au_t4_dlam = [abs(SPECTRA_DATA["Au_40nm"]["models"][k]["h_4nm"]["total_error_vs_mie_jc"]["dlam_nm"]) for k in pole_keys]
au_t4_nrms = [SPECTRA_DATA["Au_40nm"]["models"][k]["h_4nm"]["total_error_vs_mie_jc"]["nrms"] * 100 for k in pole_keys]

ax.semilogy(ns_arr, au_t4_dlam, "o-", color="#D55E00", lw=2.0, ms=5.8, label="Au Total $|\\Delta\\lambda_\\mathrm{res}|$ [nm] (Cart $h=4$ nm)")
ax.semilogy(ns_arr, au_t4_nrms, "s--", color="#E69F00", lw=1.7, ms=5.5, label="Au Total $Q_\\mathrm{ext}$ NRMS [%] (Cart $h=4$ nm)")
ax.semilogy(ns_arr, ag_ttri_dlam, "d-", color="#0072B2", lw=1.8, ms=5.8, label="Ag Total $|\\Delta\\lambda_\\mathrm{res}|$ [nm] (Tri $h_s=2$ nm)")
ax.semilogy(ns_arr, ag_ttri_nrms, "^--", color="#004D40", lw=1.7, ms=5.5, label="Ag Total $Q_\\mathrm{ext}$ NRMS [%] (Tri $h_s=2$ nm)")
ax.set_xlabel("number of Drude + Lorentz poles $N$")
ax.set_ylabel(r"$|\Delta\lambda_\mathrm{res}|$ [nm] or NRMS [%] (log scale)")
ax.set_title(r"(c) Why Au needs $N \geq 3$ while Ag converges at $N = 2$")
ax.set_xticks(ns_arr)
ax.legend(fontsize=6.5, loc="upper right")

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.88, wspace=0.24)
savefig(fig, "error_decomposition_vs_poles.png")
plt.show()

# %% [markdown]
# ## 7. Cost per pole: time per step, per-edge cost, stability $\Delta t_\mathrm{max}$, and memory
#
# ### 7.1 Benchmark protocol and CPU model (Cartesian $46 \times 46 \times 4$ grid vs Body-Fitted Triangular `tri_h2nm`)
#
# All per-step timing numbers in `data/timing.json` were measured on a single uncontended
# process on **Google Cloud `g2-standard-16` (`Intel Xeon @ 2.20 GHz`, 8 cores /
# 16 vCPU, 64 GB RAM)** with **single-thread execution**
# (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`), on the
# **same Cartesian $46 \times 46 \times 4 = 8,464$-cell ($26,132$ E-edge) grid ($h = 4\text{ nm}$)**
# with CPML boundaries in $x, y$ and periodic in $z$. For each case and pole
# count $N \in \{0, 1, 2, 3, 4, 6\}$, we run $200$ warm-up steps followed by
# **5 interleaved repeats of $2,500$ steps** with garbage collection disabled,
# and report the **median time per step**.
#
# On the **body-fitted triangular mesh `tri_h2nm`** ($4,448$ triangles, $6,714$ edges,
# $8,981$ unknowns, $14,021$ steps for $60\text{ fs}$), only the $910$ interior silver
# dual edges ($13.6\%$ of the $6,714$ mesh edges) carry ADE polarization poles, so the
# full $14,021$-step run time increases only from **$11.78\text{ s}$ (`N1`, $840\ \mu\text{s/step}$)**
# to **$11.94\text{ s}$ (`N2`, $852\ \mu\text{s/step}$)** and **$12.45\text{ s}$ (`N6`, $888\ \mu\text{s/step}$)**.
#
# In every relative number in Table 3 below, **the baseline denominator is `N0`**: the
# non-dispersive (`n_poles = 0`, static $\varepsilon_r$) solver step on the
# exact same $46 \times 46 \times 4$ Cartesian grid ($h = 4\text{ nm}$).
#
# We benchmark three Cartesian configurations to separate the sparse cylinder overhead
# from the worst-case full-domain ("bulk") overhead:
#
# 1. **`cylinder_conformal` ($1,196$ dispersive E-edges, $1,500$ Kottke sub-capacitors;
#    $4.58\%$ of the $26,132$ grid E-edges):** `conformal="all"`, updated via
#    `wakis.tensor_dispersive.DispersiveNetwork`.
# 2. **`cylinder_staircase` ($1,056$ dispersive E-edges; $4.04\%$ of grid E-edges):**
#    `conformal=False`, updated via `SolverFIT3D._step_dispersive`.
# 3. **`bulk` ($26,132$ dispersive E-edges; $100\%$ of grid E-edges):** the
#    dispersive material fills the entire domain (`solver.add_dispersive_material`),
#    giving the worst-case slowdown when every edge in the grid is dispersive.
#
# **Table 3 — Single-thread step time, slowdown vs `N0` (non-dispersive baseline on the same $46 \times 46 \times 4$ Cartesian $h = 4\text{ nm}$ grid), per-edge-per-pole cost, and auxiliary memory (`Intel Xeon @ 2.20 GHz`, 1 thread):**
#
# | Case ($N_\mathrm{edge}$ dispersive edges) | Model ($N$) | Median $\mu\text{s/step}$ | Slowdown vs `N0` baseline | $\Delta t_\mathrm{step}$ vs `N0` [$\mu\text{s}$] | Total $(\Delta t_\mathrm{step} / N_\mathrm{edge} / N)$ [ns/edge/pole] | Marginal slope ($N \ge 1$) [ns/edge/pole] | Pole aux. memory [B/disp. edge] ([B/cap]) |
# |---|---:|---:|---:|---:|---:|---:|---:|
# | **`cylinder_conformal`** | `N0` (0) | 952.04 | **1.0000$\times$** | +0.00 | — | | 0.0 (0.0) |
# | ($N_\mathrm{edge} = 1,196$, | `N1` (1) | 1059.78 | **1.1132$\times$** | +107.74 | 90.08 | | 20.1 (16.0) |
# | $N_\mathrm{cap} = 1,500$, | `N2` (2) | 1101.21 | **1.1567$\times$** | +149.17 | 62.36 | **22.74 ns/edge/pole** | 40.1 (32.0) |
# | $4.58\%$ of grid) | `N3` (3) | 1124.43 | **1.1811$\times$** | +172.39 | 48.05 | ($+27.19\ \mu\text{s/step/pole}$ | 60.2 (48.0) |
# | | `N4` (4) | 1151.63 | **1.2096$\times$** | +199.59 | 41.72 | $+88.28\ \mu\text{s}$ fixed) | 80.3 (64.0) |
# | | `N6` (6) | 1199.64 | **1.2601$\times$** | +247.60 | 34.50 | | 120.4 (96.0) |
# | **`cylinder_staircase`** | `N0` (0) | 947.87 | **1.0000$\times$** | +0.00 | — | | 0.0 (0.0) |
# | ($N_\mathrm{edge} = 1,056$, | `N1` (1) | 998.68 | **1.0536$\times$** | +50.81 | 48.11 | | 16.0 (16.0) |
# | $4.04\%$ of grid) | `N2` (2) | 1001.01 | **1.0561$\times$** | +53.14 | 25.16 | **9.06 ns/edge/pole** | 32.0 (32.0) |
# | | `N3` (3) | 1006.36 | **1.0617$\times$** | +58.48 | 18.46 | ($+9.56\ \mu\text{s/step/pole}$ | 48.0 (48.0) |
# | | `N4` (4) | 1023.40 | **1.0797$\times$** | +75.53 | 17.88 | $+38.80\ \mu\text{s}$ fixed) | 64.0 (64.0) |
# | | `N6` (6) | 1043.71 | **1.1011$\times$** | +95.84 | 15.13 | | 96.0 (96.0) |
# | **`bulk` (100% filled)** | `N0` (0) | 959.00 | **1.0000$\times$** | +0.00 | — | | 0.0 (0.0) |
# | ($N_\mathrm{edge} = 26,132$, | `N1` (1) | 1362.96 | **1.4212$\times$** | +403.96 | 15.46 | | 16.0 (16.0) |
# | $100\%$ of grid) | `N2` (2) | 1488.07 | **1.5517$\times$** | +529.07 | 10.12 | **4.39 ns/edge/pole** | 32.0 (32.0) |
# | | `N3` (3) | 1584.05 | **1.6518$\times$** | +625.05 | 7.97 | ($+114.78\ \mu\text{s/step/pole}$ | 48.0 (48.0) |
# | | `N4` (4) | 1699.29 | **1.7719$\times$** | +740.29 | 7.08 | $+289.18\ \mu\text{s}$ fixed) | 64.0 (64.0) |
# | | `N6` (6) | 1942.96 | **2.0260$\times$** | +983.96 | 6.28 | | 96.0 (96.0) |
#
# ### 7.2 Stability ($\Delta t_\mathrm{max}$ from `is_stable`) and when a pole forces extra steps
#
# Every pole in a leapfrog ADE scheme has its own discrete stability ceiling
# $\Delta t_\mathrm{max}$, checked by `DispersiveMaterial.is_stable(dt, courant=0.85)`:
#
# * On our $h = 4\text{ nm}$ grid ($\Delta t_\mathrm{CFL} = 7.97\text{ as}$),
#   $h = 2\text{ nm}$ grid ($\Delta t_\mathrm{CFL} = 3.99\text{ as}$), and
#   body-fitted triangular meshes ($\Delta t = 4.28\text{ as}$ and $2.65\text{ as}$),
#   all of our fitted $N = 1, 2, 3, 4, 6$ models have $\Delta t_\mathrm{max} = 238\text{--}429\text{ as}$,
#   and Rakić has $\Delta t_\mathrm{max} = 26.2\text{ as}$ (Ag) and $30.5\text{ as}$ (Au).
#   Because $\Delta t_\mathrm{CFL} \ll \Delta t_\mathrm{max}$, **none of the models
#   force a smaller $\Delta t$ than the vacuum Courant step on fine plasmonic grids**
#   ($h \le 11.3\text{ nm}$ for Rakić Ag; $h \le 107\text{ nm}$ for our fitted Ag `N2`),
#   so they require **0 extra time steps**.
# * **When does a model force $\Delta t < \Delta t_\mathrm{CFL}$?** On a coarser
#   grid—for example $h = 15\text{ nm}$ ($\lambda_\mathrm{min}/20$ in vacuum),
#   where $\Delta t_\mathrm{CFL} = 29.90\text{ as}$ ($1,171$ steps for $35\text{ fs}$)—our
#   fitted `N2` model ($\Delta t_\mathrm{max} = 247.9\text{ as}$) is still completely
#   Courant-limited ($0$ extra steps), whereas **Rakić Ag** ($\Delta t_\mathrm{max} = 26.20\text{ as}$,
#   limited by its $20.29\text{ eV}$ deep-UV pole) is **unstable at $\Delta t_\mathrm{CFL}$**
#   and forces $\Delta t \le 26.20\text{ as}$. That requires **$1,336$ steps instead
#   of $1,171$ ($+165$ extra steps, $+14.1\%$)**, multiplying Rakić's 6-pole
#   per-step overhead ($1.260\times$ in `cylinder_conformal` or $2.026\times$ in `bulk`)
#   by an additional $1.141\times$ in total wall time ($1.438\times$ total in `cylinder_conformal`,
#   $2.312\times$ in `bulk` relative to `N0` at $\Delta t_\mathrm{CFL}$).
#
# ### 7.3 Memory per dispersive edge vs $N$
#
# Each second-order Lorentz or Drude pole stores two `float64` state arrays
# ($p^n$ and $p^{n-1}$, $2 \times 8 = 16\text{ bytes}$) per dispersive element
# (and a first-order Debye pole stores one `float64` array, $8\text{ bytes}$):
#
# * In **`cylinder_staircase`**, **`bulk`**, and **`UnstructuredFIT2D`** (`tri_h2nm`),
#   there is 1 capacitor per dispersive E-edge, so the pole auxiliary memory is
#   **exact: $16 N\text{ bytes/edge}$** ($16, 32, 48, 64, 96\text{ B/edge}$ for $N = 1, 2, 3, 4, 6$).
# * In **`cylinder_conformal`** (`wakis.tensor_dispersive.DispersiveNetwork`),
#   interior edges have 1 capacitor and cut boundary edges have 2 sub-capacitors
#   (series + parallel branches, $1,500 / 1,196 = 1.2542$ capacitors per dispersive
#   edge), giving **$16 N\text{ bytes/capacitor} = 20.07 N\text{ bytes/dispersive edge}$**
#   ($20.1, 40.1, 60.2, 80.3, 120.4\text{ B/edge}$ for $N = 1, 2, 3, 4, 6$).

# %%
print(f"\nMachine: {TIMING_DATA['machine']}")
print(f"Baseline denominator: {TIMING_DATA['protocol']['baseline_denominator']}")
for case_name, cdata in TIMING_DATA["cases"].items():
    print(
        f"\n--- {case_name} (baseline N0 = {cdata['baseline_N0_us_per_step']:.2f} us/step, "
        f"marginal slope = +{cdata['marginal_us_per_step_per_pole']:.2f} us/step/pole = "
        f"{cdata['marginal_ns_per_disp_edge_per_pole']:.2f} ns/edge/pole) ---"
    )
    print(
        f"{'Key':<6} {'N':>2} {'edges':>7} {'caps':>7} {'us/step':>9} "
        f"{'slowdown':>9} {'ns/edge/pole':>13} {'B/edge':>8} {'B/cap':>7}"
    )
    for k in ("N0", "N1", "N2", "N3", "N4", "N6", "Rakic"):
        m = cdata["models"][k]
        print(
            f"{k:<6} {m['n_poles']:>2} {m['n_disp_edges']:>7} {m['n_caps']:>7} "
            f"{m['median_us_per_step']:>9.2f} {m['slowdown_vs_N0']:>8.4f}x "
            f"{m['cost_ns_per_disp_edge_per_pole']:>13.2f} "
            f"{m['pole_aux_bytes_per_disp_edge']:>8.1f} {m['pole_aux_bytes_per_cap']:>7.1f}"
        )

# %%
# Figure 5: Slowdown, per-edge cost, and auxiliary memory vs N (46x46x4 Cartesian grid, h = 4 nm)
fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=14.6, height_in=4.3)
ns_all = np.array([0, 1, 2, 3, 4, 6])
keys_all = ["N0", "N1", "N2", "N3", "N4", "N6"]
keys_pos = ["N1", "N2", "N3", "N4", "N6"]
ns_pos = np.array([1, 2, 3, 4, 6])

case_styles = {
    "cylinder_conformal": ("#0072B2", "o-", "Cylinder conformal (1,196 edges, 4.6%)"),
    "cylinder_staircase": ("#009E73", "s--", "Cylinder staircase (1,056 edges, 4.0%)"),
    "bulk": ("#D55E00", "^-", "Bulk domain (26,132 edges, 100%)"),
}

# (a) Slowdown vs N relative to N0 baseline on the same 46x46x4 grid
ax = axes[0]
for cname, (col, fmt, lbl) in case_styles.items():
    cdata = TIMING_DATA["cases"][cname]
    slow = [cdata["models"][k]["slowdown_vs_N0"] for k in keys_all]
    ax.plot(ns_all, slow, fmt, color=col, lw=1.8, ms=6, label=lbl)
ax.set_xlabel("number of Drude + Lorentz poles $N$")
ax.set_ylabel("slowdown relative to N0 baseline")
ax.set_title("(a) Step-time slowdown vs $N$ ($46\\times46\\times4$ Cart. $h=4$ nm)\n(baseline: non-dispersive N0 on same grid)")
ax.set_xticks(ns_all)
ax.legend(fontsize=6.8, loc="upper left")

# (b) Per-dispersive-edge cost (ns / edge / pole) vs N
ax = axes[1]
for cname, (col, fmt, lbl) in case_styles.items():
    cdata = TIMING_DATA["cases"][cname]
    cep = [cdata["models"][k]["cost_ns_per_disp_edge_per_pole"] for k in keys_pos]
    marg = cdata["marginal_ns_per_disp_edge_per_pole"]
    ax.plot(
        ns_pos,
        cep,
        fmt,
        color=col,
        lw=1.8,
        ms=6,
        label=f"{lbl.split(' (')[0]} (marginal: {marg:.1f} ns)",
    )
    ax.axhline(marg, color=col, ls=":", lw=1.0, alpha=0.75)
ax.set_xlabel("number of Drude + Lorentz poles $N$")
ax.set_ylabel(r"$(\Delta t_\mathrm{step}) / (N_\mathrm{edge}\,N)$ [ns / edge / pole]")
ax.set_title("(b) Amortized & marginal cost per\ndispersive edge per pole vs $N$")
ax.set_xticks(ns_pos)
ax.legend(fontsize=6.8, loc="upper right")

# (c) Auxiliary polarization memory per dispersive edge vs N
ax = axes[2]
for cname, (col, fmt, lbl) in case_styles.items():
    cdata = TIMING_DATA["cases"][cname]
    mem_e = [cdata["models"][k]["pole_aux_bytes_per_disp_edge"] for k in keys_all]
    ax.plot(ns_all, mem_e, fmt, color=col, lw=1.8, ms=6, label=lbl.split(" (")[0])
ax.plot(
    ns_all,
    8.0 * ns_all,
    "k:",
    lw=1.2,
    label="Debye reference (1 array = $8N$ B/edge)",
)
ax.set_xlabel("number of poles $N$")
ax.set_ylabel("auxiliary state memory [bytes / dispersive edge]")
ax.set_title("(c) Pole auxiliary memory per\ndispersive edge vs $N$")
ax.set_xticks(ns_all)
ax.legend(fontsize=6.8, loc="upper left")

fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.84, wspace=0.24)
savefig(fig, "cost_and_memory_vs_poles.png")
plt.show()

# %% [markdown]
# ## 8. Limits and failure modes
#
# * **2D geometry only (infinite cylinder):** A 2D cylinder has its quasi-static
#   plasmon condition at $\mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) = -1$,
#   whereas a 3D sphere resonates at $\mathrm{Re}\,\varepsilon(\lambda_\mathrm{res}) = -2$
#   (shifting the Ag dipole LSPR from $\sim 341\text{ nm}$ to $\sim 355\text{ nm}$)
#   and finite nanorods exhibit longitudinal modes that red-shift into the near-IR.
# * **Single isolated cylinder and single material:** Coupled dimers with
#   nanometre gaps (benchmark case `B3`) concentrate the field into high-order
#   gap plasmons that require $h \le 0.5\text{--}1\text{ nm}$ or adaptive local
#   refinement (`wakis.subgrid2d`, `wakis.subtri`).
# * **Subpixel averaging at negative-permittivity interfaces vs body-fitted triangles:**
#   While Kottke subpixel averaging (`wakis.tensor_dispersive.DispersiveNetwork`)
#   markedly improves upon staircasing, the harmonic (series) average of a negative
#   metal $\varepsilon(\omega) < 0$ and positive vacuum $\varepsilon_\mathrm{host} = 1$
#   can cross zero at intermediate fill fractions $f \in (0, 1)$, introducing weak
#   localized cut-cell resonances that limit Cartesian convergence on curved
#   plasmonic surfaces compared with boundary-conforming triangular meshes
#   (`wakis.unstructured.UnstructuredFIT2D`, which cuts the $N=2$ discretization
#   NRMS from $2.47\%$ to $0.28\%$ on the same $2a = 40\text{ nm}$ Ag cylinder).
# * **Tabulated dataset uncertainty exceeds $N \ge 2$ fit differences:** For
#   silver, the difference between $N = 2$ and $N = 6$ in $\lambda_\mathrm{res}$
#   is $< 0.05\text{ nm}$, whereas the discrepancy between two standard handbooks—Johnson
#   & Christy (1972) and Palik / Rakić (1998)—is **$9.4\text{ nm}$ in $\lambda_\mathrm{res}$
#   and $36\%$ in peak $Q_\mathrm{ext}$** due to thin-film morphology and surface
#   roughness in the original measurements.
# * **Fit bandwidth and inter-sample spikes:** Pole fits are guaranteed passive
#   and accurate only over their fitted spectral window ($300\text{--}900\text{ nm}$).
#   Fitting many Lorentz poles to sparse raw data without dense interpolation or
#   damping bounds can produce narrow inter-sample resonances; always inspect
#   $\varepsilon'(\lambda), \varepsilon''(\lambda)$ on a fine grid and check
#   `mat.is_stable(dt)` before time stepping.
# * **Single-machine CPU timing:** All timing figures were measured single-threaded
#   (`OMP/MKL/OpenBLAS = 1`) on one CPU model (**`Intel Xeon @ 2.20 GHz`** on a
#   `Google Cloud g2-standard-16` VM). Because the dispersive update uses indirect
#   gather/scatter indexing (`E[idx]`), its per-edge cost depends on cache
#   locality and memory bandwidth and will differ on other CPUs or GPUs.
#
# ## 9. Conclusions
#
# 1. **Body-fitted triangular mesh vs Cartesian FIT at the curved plasmonic boundary:**
#    Upgrading from the uniform $h = 2\text{ nm}$ Cartesian grid ($86 \times 86 \times 4 = 29,584$
#    cells, $37,632$ 2D unknowns, $10.97\text{ cells}/\delta_\mathrm{min}$, $8,782$ steps,
#    $14.87\text{ s}$) to a **body-fitted triangular mesh** (`tri_h2nm`: $h_s = 2\text{ nm}$
#    at $r = a = 20\text{ nm}$ graded to $6\text{ nm}$, $4,448$ triangles, $8,981$ unknowns,
#    $11.25\text{ elements}/\delta_\mathrm{min}$, $14,021$ steps, $11.94\text{ s}$) eliminates
#    mixed metal--vacuum cut cells, reducing the Ag `N2` discretization NRMS vs `Mie(N2)`
#    from **$2.47\%$ to $0.28\%$ ($8.8\times$ smaller)**, cutting the peak $Q_\mathrm{ext}$
#    discretization error from **$-5.92\%$ to $-0.20\%$**, and using **$4.2\times$ fewer unknowns**.
#    Refining the surface triangles to $h_s = 1.25\text{ nm}$ (`tri_h1p25nm`: $18.0\text{ elements}/\delta_\mathrm{min}$,
#    $7,097$ triangles, $14,296$ unknowns, $22,600$ steps, $26.47\text{ s}$) yields a clean,
#    artifact-free dipolar LSPR near-field map ($|\mathbf{E}|/|\mathbf{E}_\mathrm{inc}| = 3.93$
#    uniformly inside, $5.63$ at the exterior $\rho = a$ poles).
# 2. **How many poles does silver need?** **Two poles (`N2`: 1 Drude + 1 UV
#    Lorentz oscillator at $\omega_0 = 4.21\text{ eV} / 294\text{ nm}$)** are
#    sufficient for an Ag cylinder across $300\text{--}900\text{ nm}$. Going from
#    $N = 1$ (pure Drude) to $N = 2$ cuts the permittivity fit error from $21.7\%$
#    to $6.2\%$, reduces the model resonance wavelength error from $+6.35\text{ nm}$
#    to $+1.87\text{ nm}$ (and total error vs `Mie(J&C)` to $+1.77\text{ nm}$ on `tri_h2nm`
#    and $+2.26\text{ nm}$ on Cartesian $h = 2\text{ nm}$), and drops the total $Q_\mathrm{ext}$
#    NRMS error from $14.43\%$ to **$3.65\%$ on `tri_h2nm`** ($5.03\%$ on Cartesian $h = 2\text{ nm}$).
#    Adding more poles ($N = 3, 4, 6$) changes $\lambda_\mathrm{res}$ by $< 0.05\text{ nm}$—far
#    below the $9.4\text{ nm}$ discrepancy between experimental handbooks (J&C vs Rakić/Palik).
# 3. **Why does gold need more poles?** Gold's $5d \to 6sp$ interband transition
#    starts at $\sim 2.4\text{ eV}$ ($517\text{ nm}$), directly overlapping its
#    $\lambda_\mathrm{res} = 496.0\text{ nm}$ localized surface plasmon. Consequently
#    $N = 1$ fails completely ($\Delta\lambda_\mathrm{res} = -97.2\text{ nm}$,
#    $50.0\%$ NRMS), $N = 2$ still has $\Delta\lambda_\mathrm{res} = -35.0\text{ nm}$
#    ($6.7\%$ NRMS), and **$N \ge 3\text{--}6$ poles** are required to reach
#    $\Delta\lambda_\mathrm{res} = -9.7\text{ nm} \to -2.0\text{ nm}$ and
#    $2.5\% \to 1.5\%$ total NRMS.
# 4. **What does each pole cost?** On `Intel Xeon @ 2.20 GHz` (1 thread,
#    $46 \times 46 \times 4$ Cartesian grid at $h = 4\text{ nm}$, relative to the
#    non-dispersive `N0` baseline on the same grid):
#    * For the **conformal cylinder** ($1,196$ dispersive edges = $4.58\%$ of the
#      grid), enabling dispersion costs $+88.3\ \mu\text{s/step}$ fixed network
#      overhead plus **$+27.2\ \mu\text{s/step/pole}$ ($22.7\text{ ns/edge/pole}$
#      marginal; $90.1 \to 34.5\text{ ns/edge/pole}$ total amortized for
#      $N = 1 \to 6$)**, so `N2` is only **$1.157\times$** slower than `N0`
#      ($1,101\ \mu\text{s}$ vs $952\ \mu\text{s/step}$) and `N6` is **$1.260\times$**
#      slower. On the $4,448$-cell **`tri_h2nm` triangular mesh** ($910$ Ag dual edges),
#      going from `N1` to `N6` increases the full $14,021$-step run time by only
#      $5.7\%$ ($11.78\text{ s} \to 12.45\text{ s}$).
#    * For the **staircase cylinder** ($1,056$ dispersive edges = $4.04\%$ of the
#      grid), the marginal cost is **$+9.6\ \mu\text{s/step/pole}$
#      ($9.1\text{ ns/edge/pole}$)**, giving a **$1.056\times$** slowdown at `N2`
#      and **$1.101\times$** at `N6`.
#    * In the **worst-case bulk domain** ($26,132$ dispersive edges = $100\%$ of
#      the grid), the marginal cost is **$+114.8\ \mu\text{s/step/pole}$
#      ($4.39\text{ ns/edge/pole}$)** after a $+289.2\ \mu\text{s/step}$ fixed
#      ADE gather/scatter overhead, giving slowdowns of **$1.421\times$ (`N1`),
#      $1.552\times$ (`N2`), and $2.026\times$ (`N6`)** relative to `N0`.
#    * Each Lorentz/Drude pole adds **2 `float64` arrays ($16\text{ bytes/capacitor}$)**:
#      exactly **$16 N\text{ B/dispersive edge}$** in staircase, bulk, and
#      `UnstructuredFIT2D`, and **$20.1 N\text{ B/dispersive edge}$** in conformal
#      Cartesian mode ($1.254$ sub-capacitors per dispersive edge).
#
# ## 10. References
#
# 1. P. B. Johnson and R. W. Christy, "Optical constants of the noble metals,"
#    *Phys. Rev. B* **6**(12), 4370--4379 (1972).
#    DOI: [10.1103/PhysRevB.6.4370](https://doi.org/10.1103/PhysRevB.6.4370)
# 2. A. D. Rakić, A. B. Djurišić, J. M. Elazar, and M. L. Majewski, "Optical
#    properties of metallic films for vertical-cavity optoelectronic devices,"
#    *Appl. Opt.* **37**(22), 5271--5283 (1998).
#    DOI: [10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271)
# 3. C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small
#    Particles* (Wiley, New York, 1983), ch. 8 & ch. 12.
#    DOI: [10.1002/9783527618156](https://doi.org/10.1002/9783527618156)
# 4. S. A. Maier, *Plasmonics: Fundamentals and Applications* (Springer, New York,
#    2007). DOI: [10.1007/0-387-37825-1](https://doi.org/10.1007/0-387-37825-1)
# 5. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed. (Artech House, Norwood, MA,
#    2005), ch. 9 (dispersive materials and auxiliary differential equations).
# 6. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**(3), 036611 (2008).
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611)
