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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/03_diffraction_and_scattering/03-05_mie_sphere/mie_sphere.ipynb)
#
# # Tutorial 3.5 · Mie scattering by a sphere (3D) and the near-to-far-field transform
# <h3 align="center">Total-field/scattered-field plane waves in the Finite Integration Technique</h3>
# <p align="center"><i>Erik Shipton &middot; Wakis tutorial series (fork of CERN's Wakis)</i></p>
#
# ---
#
# ### Abstract
#
# **We shine a broadband plane-wave pulse on a glass-like sphere (refractive
# index $n = 2$, radius $r = 1$ µm) and compute its scattering efficiency
# $Q_\mathrm{sca}$ over a 5:1 band of size parameters $x = 2\pi r/\lambda$
# from a single time-domain run. The plane wave enters through a
# total-field/scattered-field (TF/SF) box, so that the field outside the box
# is purely the field scattered by the sphere; a flux box in that region
# measures the scattered power directly. We compare the result with an exact
# Lorenz-Mie series, look at the near fields and the far-field pattern, and
# measure how the error falls with resolution for a staircased sphere and
# for conformal (subcell-averaged) material models at equal cells per
# wavelength.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to
#
# 1. **explain** Mie scattering, the extinction/scattering/absorption cross
#    sections and efficiencies, and evaluate them with `wakis.analytic.mie`;
# 2. **derive** the TF/SF source from the surface equivalence principle
#    ($\mathbf J_s = \hat n \times \mathbf H_\mathrm{inc}$,
#    $\mathbf K_s = -\hat n \times \mathbf E_\mathrm{inc}$) and say why a
#    *discrete* incident field is needed for the incident wave to cancel
#    outside the box;
# 3. **build** a curved geometry with PyVista, mesh it with `GridFIT3D`, and
#    inspect what the solver actually sees on the grid (staircase versus
#    conformal permittivity on the edges);
# 4. **estimate** memory and run time with `wakis.estimate` and check the
#    target machine with `preflight` *before* launching a 3D run;
# 5. **measure** a cross section from a flux box, normalized by the incident
#    flux, and compare near fields, far-field patterns and spectra with
#    theory;
# 6. **quantify** discretization error against an analytic reference with a
#    convergence plot, naming the baseline (staircase) and the cost axis
#    (cells per wavelength, run time).
#
# ### Contents
# 0. Set-up (Colab install, GPU detection, run mode)
# 1. The physics: Mie scattering and cross sections
# 2. The TF/SF source: equivalence principle on a grid
# 3. Geometry: PyVista sphere &rarr; STL &rarr; `GridFIT3D`
# 4. The mesh the solver sees: staircase versus conformal
# 5. Preflight: estimate memory and run time
# 6. Run the simulation
# 7. Fields: total inside, scattered outside
# 8. $Q_\mathrm{sca}(x)$ versus Mie theory, and the far field
# 8b. Set-up check: is the PML far enough?
# 9. Convergence: staircase versus conformal at equal cells per wavelength
# 10. Limits and failure modes
# 11. Exercises
# 12. References

# %% [markdown]
# ## 0. Set-up
#
# On **Google Colab** the next cell installs the Wakis fork from GitHub and
# a virtual frame buffer (`xvfb`) so that PyVista can render without a
# screen. NumPy, SciPy, Matplotlib, h5py and tqdm are preinstalled on Colab;
# PyVista comes with Wakis. For the GPU, choose *Runtime &rarr; Change
# runtime type &rarr; GPU*: Colab images ship CuPy for their CUDA version,
# and the cell installs `cupy-cuda12x` only if it is missing.
#
# Locally, install the fork in a virtual environment instead
# (`uv pip install -e .` in the repository) and run this file as a script:
#
# ```bash
# python tutorials/03_diffraction_and_scattering/03-05_mie_sphere/mie_sphere.py --quick        # CPU, a few minutes
# python tutorials/03_diffraction_and_scattering/03-05_mie_sphere/mie_sphere.py --full --gpu   # GPU, accurate
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
    try:
        import cupy  # noqa: F401
    except ImportError:
        if subprocess.run("nvidia-smi", shell=True).returncode == 0:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"],
                check=False,
            )

# %%
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm in Jupyter

import matplotlib
import numpy as np

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # running as a script: headless
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.constants import c as c0

import wakis
from wakis import visualize as wv
from wakis.analytic import (
    mie_differential_cross_section,
    mie_efficiencies,
    rayleigh_efficiencies,
)
from wakis.estimate import estimate, interface_cells_from_area, preflight
from wakis.sources import GaussianSource

print(wv.setup_rendering())
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


def gpu_available():
    """True when CuPy is importable and sees a CUDA device."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_name():
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        return props["name"].decode()
    except Exception:
        return None


IN_NOTEBOOK = "ipykernel" in sys.modules
parser = argparse.ArgumentParser(
    description="Wakis Tutorial 3.5 · Mie scattering by a sphere (3D) and the near-to-far-field transform"
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--quick", action="store_true", help="CPU, ~0.5 Mcells, minutes")
mode.add_argument("--full", action="store_true", help="GPU, ~6 Mcells, accurate")
parser.add_argument("--gpu", action="store_true", help="force the GPU")
parser.add_argument("--cpu", action="store_true", help="force the CPU")
parser.add_argument("--resolution", type=float, default=None, help="cells per µm")
parser.add_argument(
    "--conformal", default=None, help="staircase | all | tensor (main run)"
)
parser.add_argument("--arm", action="store_true", help="study arm only: run + JSON")
parser.add_argument("--out", default=None, help="JSON output of --arm")
parser.add_argument("--no-movie", action="store_true", help="skip the GIF")
parser.add_argument(
    "--pml-gap",
    type=float,
    default=1.0,
    help="scale of the sphere-surface-to-CPML distance (study)",
)
parser.add_argument("--kappa-max", type=float, default=None, help="CPML kappa_max")
parser.add_argument(
    "--alpha-max",
    type=float,
    default=None,
    help="CPML alpha_max [S/m] (CFS shift; default: solver's)",
)
parser.add_argument("--n-pml", type=int, default=None, help="CPML layers")
parser.add_argument("--t-run", type=float, default=None, help="max time [fs]")
parser.add_argument(
    "--gap-um",
    type=float,
    default=None,
    help="sphere surface -> CPML distance [um] on a graded grid",
)
parser.add_argument(
    "--coarse-per-um",
    type=float,
    default=16.0,
    help="cells per um in the graded vacuum gap",
)
parser.add_argument(
    "--grade-ratio",
    type=float,
    default=1.15,
    help="max ratio of neighbouring cells on the graded grid",
)
# In a notebook, options come from the environment, e.g.
# WAKIS_TUTORIAL_ARGS="--quick --cpu" (used by tutorials/build.py --execute)
NB_ARGS = os.environ.get("WAKIS_TUTORIAL_ARGS", "").split()
ARGS, _ = parser.parse_known_args(NB_ARGS if IN_NOTEBOOK else None)

USE_GPU = (ARGS.gpu or gpu_available()) and not ARGS.cpu
FULL = ARGS.full or (not ARGS.quick and USE_GPU and IN_NOTEBOOK)
RESOLUTION = ARGS.resolution or (40.0 if FULL else 15.0)  # cells per µm
CONFORMAL = ARGS.conformal or "all"
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIG_DIR = HERE / "figures"
if ARGS.arm:  # study runs must not overwrite the tutorial figures
    import tempfile

    FIG_DIR = Path(tempfile.mkdtemp())
DATA_DIR = HERE / "data"
FIG_DIR.mkdir(exist_ok=True)
print(
    f"mode={'full' if FULL else 'quick'}  GPU={USE_GPU}"
    f"{' (' + str(gpu_name()) + ')' if USE_GPU else ''}  "
    f"resolution={RESOLUTION:g} cells/µm  conformal={CONFORMAL}"
)


def savefig(fig, name):
    """Save a figure in figures/ (small PNG) and return the path."""
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return path


# %% [markdown]
# ## 1. The physics: Mie scattering and cross sections
#
# **Why scattered fields?** Radar and lidar see objects only through the
# field they scatter; antennas, displays, solar-cell textures, structural
# colour, aerosols in climate models, biological imaging and every
# metasurface atom are scattering problems. The *scattered* field
# $\mathbf E_s = \mathbf E - \mathbf E_\mathrm{inc}$ is what carries the
# information about the object, and it is often orders of magnitude weaker
# than the incident wave, so a simulation should hand it to us *separately*.
#
# **Cross sections.** A plane wave of intensity
# $I_\mathrm{inc} = |\mathbf E_\mathrm{inc}|^2 / (2\eta)$ (time-averaged
# power per unit area, $\eta = \sqrt{\mu/\varepsilon}$ of the surrounding
# medium) hits a particle. The particle removes power
# $P_\mathrm{ext} = P_\mathrm{sca} + P_\mathrm{abs}$ from the beam: power
# re-radiated in all directions plus power absorbed. Dividing by the incident
# intensity gives areas,
#
# $$\sigma_\mathrm{sca} = \frac{P_\mathrm{sca}}{I_\mathrm{inc}},\qquad
#   \sigma_\mathrm{abs} = \frac{P_\mathrm{abs}}{I_\mathrm{inc}},\qquad
#   \sigma_\mathrm{ext} = \sigma_\mathrm{sca} + \sigma_\mathrm{abs},$$
#
# and dividing by the geometric cross section $\pi r^2$ gives dimensionless
# **efficiencies** $Q = \sigma / (\pi r^2)$. The scattered power is the
# outward Poynting flux of the scattered field through any closed surface
# around the particle:
# $P_\mathrm{sca} = \tfrac12 \oint \mathrm{Re}(\mathbf E_s \times
# \mathbf H_s^*)\cdot\hat n\, dA$. That is exactly what a flux box in the
# scattered-field region of a TF/SF simulation measures.
#
# **Mie theory.** For a homogeneous sphere, Maxwell's equations separate in
# spherical coordinates and the scattered field is a series of vector
# spherical harmonics with coefficients $a_n$ (electric multipoles) and $b_n$
# (magnetic multipoles) that depend only on the relative index
# $m = n_\mathrm{sphere}/n_\mathrm{medium}$ and the size parameter
# $x = k r = 2\pi n_\mathrm{medium} r / \lambda_0$
# (Bohren & Huffman 1983, ch. 4):
#
# $$Q_\mathrm{sca} = \frac{2}{x^2} \sum_{n=1}^{\infty} (2n+1)
#   \left(|a_n|^2 + |b_n|^2\right),\qquad
#   Q_\mathrm{ext} = \frac{2}{x^2} \sum_{n=1}^{\infty} (2n+1)\,
#   \mathrm{Re}(a_n + b_n).$$
#
# `wakis.analytic.mie` implements this series from scratch (stable downward
# recurrence for the logarithmic derivative $D_n(mx)$, complex $m$ for the
# metal spheres of the next tutorial) and is tested against the Rayleigh
# limit, direct Bessel-function evaluation, the optical theorem, Bohren &
# Huffman's BHMIE example and the independent `miepython` package
# (`tests/test_040_mie_theory.py`).
#
# Below: for $x \ll 1$ the sphere is a point dipole and
# $Q_\mathrm{sca} = \tfrac83 x^4 \left|\tfrac{m^2-1}{m^2+2}\right|^2$
# (Rayleigh: why the sky is blue). Around $x \sim 1$ to $3$ the first
# magnetic and electric dipole resonances appear; at larger $x$ ever-higher
# multipoles resonate, giving the ripple structure, and $Q$ oscillates around
# 2 (the *extinction paradox*: a large object removes twice its geometric
# cross section, half by diffraction).

# %%
R_SPHERE = 1.0e-6  # sphere radius [m]
N_SPHERE = 2.0  # refractive index (lossless)
X_MIN, X_MAX = 2.0, 10.0  # size-parameter band of the pulse
x_plot = np.geomspace(0.05, 12, 1500)
q_ext, q_sca, q_abs, q_back = mie_efficiencies(N_SPHERE, x_plot)
q_ray, _ = rayleigh_efficiencies(N_SPHERE, x_plot)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.loglog(x_plot, q_sca, color="k", lw=1.4, label=r"Mie $Q_\mathrm{sca}$")
ax.loglog(x_plot, q_ray, "--", color="#0072B2", label=r"Rayleigh $\frac{8}{3}x^4|L|^2$")
ax.axvspan(
    X_MIN,
    X_MAX,
    color="#E69F00",
    alpha=0.2,
    label="band covered by the simulation pulse",
)
ax.axhline(2, color="0.5", lw=0.9, ls=":", label=r"large-sphere limit, $Q \to 2$")
ax.set(
    xlabel=r"size parameter $x = 2\pi r/\lambda$",
    ylabel=r"$Q_\mathrm{sca} = \sigma_\mathrm{sca}/\pi r^2$",
    ylim=(1e-6, 10),
    title=f"Analytic theory only (no simulation): sphere, n = {N_SPHERE:g}",
)
ax.legend(loc="lower right")
savefig(fig, "mie_theory.png")
plt.show()

# %% [markdown]
# ## 2. The TF/SF source: equivalence principle on a grid
#
# **The problem.** We want a plane wave to hit the sphere, and we want to
# read off the scattered field alone. We cannot simply impose the plane-wave
# field on a plane of the grid: such a *hard* source would also reflect the
# waves scattered back onto it.
#
# **The equivalence principle** (Oskooi & Johnson 2013, sec. 4.2.1 and
# fig. 4.1). Take the incident wave $\psi_\mathrm{inc} = (\mathbf
# E_\mathrm{inc}, \mathbf H_\mathrm{inc})$, which solves the source-free
# Maxwell equations in the *background* medium everywhere. Draw an imaginary
# closed surface $\partial\Omega$ around the region of interest $\Omega$ and
# define a new field that equals $\psi_\mathrm{inc}$ inside and is *zero*
# outside. It satisfies Maxwell's equations everywhere except on
# $\partial\Omega$, where it jumps. A jump in a curl equation is a surface
# current, so the discontinuous field is the exact solution of Maxwell's
# equations driven by the surface currents (with $\hat n$ the unit normal
# pointing *into* $\Omega$; eq. 4.4 of Oskooi & Johnson)
#
# $$\mathbf J_s = \hat n \times \mathbf H_\mathrm{inc},\qquad
#   \mathbf K_s = -\hat n \times \mathbf E_\mathrm{inc}.$$
#
# These *equivalent currents* radiate the incident wave into $\Omega$ and
# exactly nothing outside. Because Maxwell's equations are linear, adding a
# scatterer inside $\Omega$ does not change what the currents radiate: the
# field inside is the **total field** (incident + scattered), the field
# outside is only the **scattered field**, and the currents are transparent
# to the scattered waves crossing them. This is the TF/SF method, first used
# in finite differences by Merewether, Fisher & Smith (1980) as a "numeric
# Huygens source", and described in Taflove & Hagness (2005, ch. 5).
#
# **Doing it by hand.** In the companion Meep notebook (*Total-Field /
# Scattered-Field Technique with Meep FDTD*) the currents are placed on the
# faces of a box with the analytic plane-wave phase $e^{-i\mathbf k\cdot
# \mathbf r}$, one electric and one magnetic source per face. This works,
# but the cancellation outside the box is only approximate: the analytic
# wave travels at $c$ while the grid wave travels at the slightly smaller,
# direction-dependent numerical phase velocity, and on a staggered grid
# $\mathbf E$ and $\mathbf H$ sit half a cell and half a time step apart, so
# currents evaluated from continuum fields do not match the discrete jump.
# The mismatch radiates a weak "leakage" wave into the scattered-field
# region, which grows with frequency and pollutes exactly the quantity we
# want to measure.
#
# **Doing it on the grid** (`wakis.TFSFSource`). Oskooi & Johnson point out
# that applying the same derivation to the *discretized* curl operators
# gives the TF/SF source terms, and that the incident field must then solve
# the *discrete* equations (sec. 4.2.2). For axis-aligned incidence the
# incident wave is uniform across the beam, so it is computed on a 1D
# auxiliary FIT line with the grid's own cell lengths, time step and time
# levels (its far end is absorbing). The 3D update is then corrected only on
# the edges that straddle the box surface. Since the auxiliary field obeys
# the very same discrete equations, the incident field cancels outside the
# box to **round-off**, at every frequency, independent of numerical
# dispersion.
#
# The sketch below shows the layout we will use (not to scale).


# %%
def draw_tfsf_diagram(ax):
    """Schematic x-z cut of the TF/SF set-up (own drawing)."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    ax.set_xlim(-3.2, 3.6)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    # CPML frame and scattered-field region
    ax.add_patch(Rectangle((-3, -3), 6, 6, fc="0.85", ec="0.5"))
    ax.add_patch(Rectangle((-2.4, -2.4), 4.8, 4.8, fc="white", ec="0.5"))
    ax.text(0, -2.75, "CPML (absorbing layers)", ha="center", va="center", fontsize=8)
    # Scattered waves outside the box
    for rad in (1.75, 2.05, 2.3):
        ax.add_patch(Circle((0, 0), rad, fill=False, ec="#0072B2", lw=0.6, alpha=0.6))
    # Total-field box with incident wavefronts
    ax.add_patch(Rectangle((-1.3, -1.3), 2.6, 2.6, fc="#E69F00", alpha=0.10, ec="none"))
    for zf in np.linspace(-1.2, 1.2, 9):
        ax.plot([-1.3, 1.3], [zf, zf], color="#E69F00", lw=0.8, alpha=0.7)
    ax.add_patch(Rectangle((-1.3, -1.3), 2.6, 2.6, fill=False, ec="#E69F00", lw=2))
    ax.add_patch(Circle((0, 0), 0.9, fc="#56B4E9", ec="k", lw=1))
    ax.text(0, 0, "sphere\n$n=2$", ha="center", va="center", fontsize=9)
    ax.text(-1.2, 1.12, "total field\n(inc. + scat.)", fontsize=7.5, va="top")
    ax.text(-2.3, 2.15, "scattered field only", fontsize=7.5, color="#0072B2")
    # Flux box
    ax.add_patch(
        Rectangle((-1.65, -1.65), 3.3, 3.3, fill=False, ec="#0072B2", lw=1.3, ls="--")
    )
    ax.text(1.68, -1.6, r"flux box $\to P_\mathrm{sca}$", fontsize=7.5,
            color="#0072B2", rotation=90, va="bottom")  # fmt: skip
    # Inward normal and equivalent currents on the bottom (entry) face
    ax.add_patch(FancyArrowPatch((0.55, -1.3), (0.55, -0.85), arrowstyle="-|>",
                                 mutation_scale=10, color="k"))  # fmt: skip
    ax.text(0.62, -1.15, r"$\hat n$", fontsize=9)
    ax.text(-1.25, -2.0, r"$\mathbf{J}_s=\hat n\times\mathbf{H}_\mathrm{inc}$, "
            r"$\mathbf{K}_s=-\hat n\times\mathbf{E}_\mathrm{inc}$",
            fontsize=8, color="#D55E00")  # fmt: skip
    # Incident direction and polarization
    ax.add_patch(FancyArrowPatch((-1.95, -2.35), (-1.95, -1.75), arrowstyle="-|>",
                                 mutation_scale=12, color="#D55E00", lw=1.5))  # fmt: skip
    ax.text(-1.85, -2.33, r"$\mathbf{k}\parallel+z$, $\mathbf{E}\parallel x$",
            fontsize=8, color="#D55E00")  # fmt: skip
    # 1D auxiliary line feeding the box
    ax.plot([3.25, 3.25], [-2.6, 2.6], color="#009E73", lw=3)
    ax.text(3.35, 0, "1D auxiliary FIT line\n(discrete $E_\\mathrm{inc}, H_\\mathrm{inc}$)",
            rotation=90, va="center", fontsize=7.5, color="#009E73")  # fmt: skip
    ax.add_patch(FancyArrowPatch((3.15, -1.3), (1.35, -1.3), arrowstyle="-|>",
                                 mutation_scale=8, color="#009E73", ls="--"))  # fmt: skip
    ax.add_patch(FancyArrowPatch((3.15, 1.3), (1.35, 1.3), arrowstyle="-|>",
                                 mutation_scale=8, color="#009E73", ls="--"))  # fmt: skip
    ax.text(-3.0, 2.85, "x-z cut (y = 0)", fontsize=7.5, color="0.3")


fig, ax = plt.subplots(figsize=(5.6, 5.0))
draw_tfsf_diagram(ax)
savefig(fig, "tfsf_diagram.png")
plt.show()

# %% [markdown]
# ## 3. Geometry: PyVista sphere &rarr; STL &rarr; `GridFIT3D`
#
# Wakis meshes solids given as closed triangulated surfaces (STL files or
# PyVista meshes). We make a finely faceted sphere (the facet sagitta,
# $r(1-\cos(\pi/n_\theta))\approx 0.15$ nm here, must be far below the cell
# size) and write it to an STL file, which is how CAD geometry would arrive.
#
# **Layout.** All boxes are placed on grid nodes, with a margin of a few
# cells between neighbours (the TF/SF faces must not cut the sphere, the
# flux box must lie in the scattered-field region, and the CPML must start
# well outside the flux box). From the inside out:
#
# | region | half-width | purpose |
# |---|---|---|
# | sphere | 1.0 µm | scatterer, $\varepsilon_r = 4$ |
# | inner flux box | $\ge r + \Delta$ | total field: $-$absorbed power (0 here) |
# | TF/SF box | $\approx$ 1.25 µm | plane wave inside, scattered field outside |
# | scattered flux box | TF + $\approx$ 0.2 µm | $P_\mathrm{sca}$ and near-to-far field |
# | CPML start | flux + $\approx$ 0.6 µm | absorbing layers ($n_\mathrm{pml}$ = 10 cells) |

# %%
N_PML = ARGS.n_pml or 10


def layout(resolution_per_um):
    """
    Grid and box positions for a given resolution (cells per µm).

    Every box face lies on a grid node: positions are integer multiples of
    the cell size dx, the grid is centred on the sphere.
    """
    dx = 1e-6 / resolution_per_um
    i_r = int(np.ceil(R_SPHERE / dx - 1e-9))
    i_in = i_r + 1  # inner (total-field) flux box: one cell outside the sphere
    i_tf = max(i_in + 2, int(round(1.25e-6 / dx)))
    i_flux = i_tf + max(2, int(round(0.2e-6 / dx)))
    i_pml = i_flux + max(6, int(round(0.6e-6 / dx)))
    # --pml-gap scales the distance sphere surface -> CPML (boxes stay put)
    i_pml += int(round((ARGS.pml_gap - 1.0) * (i_pml * dx - R_SPHERE) / dx))
    n_half = i_pml + N_PML
    lay = dict(
        dx=dx, n_cells=2 * n_half, half=n_half * dx,
        inner=i_in * dx, tf=i_tf * dx, flux=i_flux * dx, pml=i_pml * dx, nodes=None,
    )  # fmt: skip
    if ARGS.gap_um is not None:  # graded grid: fine region, grading, coarse gap
        nodes = graded_nodes(dx, (i_flux + 2) * dx, 1e-6 / ARGS.coarse_per_um,
                             ARGS.grade_ratio, R_SPHERE + ARGS.gap_um * 1e-6, N_PML)  # fmt: skip
        lay.update(nodes=nodes, n_cells=nodes.size - 1, half=nodes[-1],
                   pml=nodes[-1 - N_PML])  # fmt: skip
    return lay


def graded_nodes(dx, x_fine, dx_coarse, ratio, x_pml, n_pml):
    """
    Symmetric 1D node coordinates [m]: uniform dx up to x_fine (all boxes),
    then cells growing by `ratio` up to dx_coarse, uniform dx_coarse up to
    the CPML start (>= x_pml) and n_pml CPML cells of dx_coarse.
    """
    pos = list(np.arange(0, round(x_fine / dx) + 1) * dx)
    h = dx
    while h * ratio < dx_coarse:
        h *= ratio
        pos.append(pos[-1] + h)
    while pos[-1] < x_pml - 1e-12:
        pos.append(pos[-1] + dx_coarse)
    for _ in range(n_pml):
        pos.append(pos[-1] + dx_coarse)
    half = np.array(pos)
    return np.concatenate([-half[:0:-1], half])


STL_FILE = str(HERE / "sphere.stl")
sphere = pv.Sphere(radius=R_SPHERE, theta_resolution=180, phi_resolution=180)
sphere.save(STL_FILE)
print(f"STL: {sphere.n_cells} triangles, area {sphere.area * 1e12:.4f} µm² "
      f"(exact {4 * np.pi:.4f})")  # fmt: skip


def make_grid(resolution_per_um, conformal, half_cells=None):
    """
    GridFIT3D with the sphere; conformal in {'staircase', 'all', 'tensor'}.
    `half_cells` overrides the domain half-width (in cells) for small views.
    """
    lay = layout(resolution_per_um)
    h, n = lay["half"], lay["n_cells"]
    if half_cells is not None:
        h, n = half_cells * lay["dx"], 2 * half_cells
    elif lay["nodes"] is not None:  # graded grid
        q = lay["nodes"]
        return wakis.GridFIT3D(
            x=q, y=q, z=q,
            stl_solids={"sphere": STL_FILE},
            stl_materials={"sphere": [N_SPHERE**2, 1.0]},
            conformal=False if conformal == "staircase" else conformal,
            verbose=0,
        ), lay  # fmt: skip
    return wakis.GridFIT3D(
        -h, h, -h, h, -h, h, n, n, n,
        stl_solids={"sphere": STL_FILE},
        stl_materials={"sphere": [N_SPHERE**2, 1.0]},  # [eps_r, mu_r]
        conformal=False if conformal == "staircase" else conformal,
        verbose=0,
    ), lay  # fmt: skip


def make_solver(grid, use_gpu=False):
    cpml = {k: v for k, v in (("kappa_max", ARGS.kappa_max), ("alpha_max", ARGS.alpha_max))
            if v is not None}  # fmt: skip
    return wakis.SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=N_PML,
        use_stl=True,
        use_gpu=use_gpu,
        verbose=0,
        **cpml,
    )


# %% [markdown]
# A 3D view of the set-up, built with `wakis.visualize.plot_scene` (sphere
# clipped to show its inside; boxes as labelled translucent volumes). We use
# a coarse grid here: the boxes are the same at every resolution up to one
# cell.

# %%
grid_view, lay_view = make_grid(10, "staircase")
boxes_view = {
    "tfsf": wv.box_bounds((0, 0, 0), (2 * lay_view["tf"],) * 3),
    "flux": wv.box_bounds((0, 0, 0), (2 * lay_view["flux"],) * 3),
}
pml_inner = wv.box_bounds((0, 0, 0), (2 * lay_view["pml"],) * 3)
pl = wv.plot_scene(
    grid_view,
    boxes=boxes_view,
    planes={"source": ((0, 0, -lay_view["tf"]), "z", (2 * lay_view["tf"],) * 2)},
    cpml=(grid_view.grid.bounds, pml_inner),
)
wv.save_plotter(pl, str(FIG_DIR / "scene.png"))
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.imshow(plt.imread(FIG_DIR / "scene.png"))
ax.axis("off")
plt.show()

# %% [markdown]
# ## 4. The mesh the solver sees: staircase versus conformal
#
# FIT stores one permittivity per primary edge (the material in the edge's
# dual cell). The plot shows the in-plane edges of the $z = 0$ slice near
# the sphere surface, coloured by their $\varepsilon_r$, for three material
# models at 20 cells/µm:
#
# * **staircase** (`conformal=False`): each edge takes the value at its
#   location, so the surface is a staircase of whole cells; the effective
#   radius jumps by up to a cell as the resolution changes, which makes the
#   error first order and erratic;
# * **conformal** (`conformal='all'`): at an interface with unit normal
#   $\hat n$ cutting an edge's dual cell with fill fraction $f$, the
#   physically right inverse permittivity is the tensor
#   $T = \langle\varepsilon^{-1}\rangle\,\hat n\hat n^T +
#   (I - \hat n\hat n^T)/\langle\varepsilon\rangle$ (Kottke, Farjadpour &
#   Johnson 2008): field components tangential to the surface see the
#   average permittivity, the normal component sees the average inverse
#   permittivity. Mode `'all'` keeps the diagonal entry $T_{cc}$ of each
#   edge (fractions computed from the STL with subcell sampling), so the
#   interface position is resolved to a fraction of a cell;
# * **tensor** (`conformal='tensor'`): also applies the off-diagonal
#   couplings $T_{ij}$ between the x, y and z edges (a separate, energy-
#   conserving operator; the plot shows the diagonal only).

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
window = (0.45e-6, 1.25e-6, -0.4e-6, 0.4e-6)
for ax, model in zip(axes, ["staircase", "all", "tensor"]):
    # Only the materials are needed: a small box around the sphere, PEC walls
    g, _ = make_grid(20, model, half_cells=26)
    s = wakis.SolverFIT3D(g, bc_low=["pec"] * 3, bc_high=["pec"] * 3,
                          use_stl=True, verbose=0)  # fmt: skip
    lc = wv.plot_material_slice(
        s, axis="z", position=0.0, window=window, surface="sphere", ax=ax,
        vmin=1, vmax=N_SPHERE**2, colorbar=False,
        title={"staircase": "staircase", "all": "conformal ('all')",
               "tensor": "tensor averaging"}[model],
    )  # fmt: skip
    if ax is not axes[0]:
        ax.set_ylabel("")
fig.colorbar(lc, ax=axes, label=r"edge $\varepsilon_r$", shrink=0.85)
savefig(fig, "mesh_staircase_vs_conformal.png")
plt.show()
del g, s

# %% [markdown]
# ## 5. Preflight: estimate memory and run time *before* running
#
# A 3D electromagnetic problem grows as (cells per wavelength)$^3$ in memory
# and as (cells per wavelength)$^4$ in work (the time step shrinks with the
# cell size). Doubling the resolution of a run that took 5 minutes and 4 GB
# gives 80 minutes and 32 GB: enough to crash a laptop. `wakis.estimate`
# predicts peak memory and run time from coefficients measured on our
# benchmark machines, and `preflight` checks them against a target machine
# (raising `MemoryError` if the run does not fit). **Make this a habit.**
#
# Inputs: the number of cells, the number of time steps, the backend, CPML,
# the conformal mode with the number of interface cells (from the sphere
# area), and the DFT monitors (frequencies $\times$ samples $\times$
# components).

# %%
T_RUN = (ARGS.t_run or 300) * 1e-15  # max simulated time [s]
N_FREQ = 81
X_SIM = np.linspace(X_MIN, X_MAX, N_FREQ)
FREQS = X_SIM * c0 / (2 * np.pi * R_SPHERE)  # [Hz]


def plan(resolution_per_um, conformal, use_gpu, machine):
    lay = layout(resolution_per_um)
    n = lay["n_cells"]
    dt = 0.5 * lay["dx"] / (c0 * np.sqrt(3))  # CFL 0.5 on a cubic grid
    steps = int(T_RUN / dt)
    n_flux = 6 * (2 * lay["flux"] / lay["dx"]) ** 2
    est = estimate(
        (n, n, n), steps,
        backend="gpu" if use_gpu else "cpu",
        cpml=True,
        conformal=False if conformal == "staircase" else conformal,
        interface_cells=interface_cells_from_area(4 * np.pi * R_SPHERE**2, lay["dx"]),
        monitors=[(N_FREQ, n_flux, 4), (3, n * n, 3)],
    )  # fmt: skip
    problems = preflight(est, machine, raise_error=False)
    return lay, steps, est, problems


# Target machine: a name from wakis.estimate.MACHINES or a dict (bytes)
if IN_COLAB:
    MACHINE = dict(ram=12e9, gpu=15e9 if USE_GPU else 0)  # Colab T4 runtime
elif USE_GPU:
    MACHINE = "g2-standard-16"  # 64 GB RAM, NVIDIA L4 24 GB
else:
    MACHINE = os.environ.get("WAKIS_MACHINE", "laptop")  # 2 GB memguard cap
for res in (15, 20, 40, 50):
    lay, steps, est, problems = plan(res, CONFORMAL, USE_GPU, MACHINE)
    print(f"{res:>3} cells/µm: {lay['n_cells']}³ = {est.cells / 1e6:5.2f} Mcells, "
          f"<= {steps} steps, host {est.host_bytes / 1e9:5.2f} GB, "
          f"GPU {est.gpu_bytes / 1e9:5.2f} GB, ~{est.seconds / 60:5.1f} min on "
          f"{'GPU' if USE_GPU else 'one CPU core'}: "
          f"{'fits ' + str(MACHINE) if not problems else 'DOES NOT FIT: ' + '; '.join(problems)}")  # fmt: skip

lay, steps, est, problems = plan(RESOLUTION, CONFORMAL, USE_GPU, MACHINE)
print("\nThis run:\n", est)
if problems and not ARGS.arm:
    raise MemoryError("; ".join(problems))

# %% [markdown]
# ## 6. Run the simulation
#
# * **Time signal**: `GaussianSource` (a Gaussian-modulated sinusoid with no
#   DC content) centred at $x = 6$ with a spectral width covering
#   $x = 2 \dots 10$, i.e. 95 to 477 THz, or vacuum wavelengths from 3.1 µm
#   down to 0.63 µm.
# * **Source**: `TFSFSource`, $+z$ incidence, $E \parallel x$, amplitude
#   1 V/m.
# * **Monitors**: a `Near2FarMonitor` box in the scattered-field region
#   (it is a `FluxMonitor` too, so it gives $P_\mathrm{sca}$ and the
#   far-field pattern from the same DFT data); a `FluxMonitor` box hugging
#   the sphere inside the total-field region (its net inward flux is the
#   absorbed power: zero for a lossless sphere, a check that can fail);
#   a `DFTMonitor` plane at $y = 0$ at three frequencies; a 3D `DFTMonitor`
#   around the sphere at one frequency; and a `SliceRecorder` for a movie.
# * **Stopping**: run until the field energy has decayed by 60 dB after the
#   pulse, so the DFTs of the slowly ringing high-order resonances converge.

# %%
F_CENTER = 0.5 * (FREQS[0] + FREQS[-1])
F_WIDTH = 0.5 * (FREQS[-1] - FREQS[0]) / 2.0  # band edges at 2 sigma
X_SNAP = np.array([3.0, 6.0, 9.0])  # size parameters of the field maps
F_SNAP = X_SNAP * c0 / (2 * np.pi * R_SPHERE)


def run_mie(resolution_per_um, conformal, use_gpu=False, extras=True,
            movie=True, energy_decay=1e-6, verbose=0):  # fmt: skip
    """Build, run and post-process one simulation; returns a dict."""
    t0 = time.time()
    grid, lay = make_grid(resolution_per_um, conformal)
    solver = make_solver(grid, use_gpu)
    t_setup = time.time() - t0
    pulse = GaussianSource(frequency=F_CENTER, fwidth=F_WIDTH)
    src = wakis.TFSFSource(
        pulse, center=(0, 0, 0), size=(2 * lay["tf"],) * 3,
        direction="+z", polarization="x", amplitude=1.0,
    )  # fmt: skip
    scat = wakis.Near2FarMonitor(FREQS, size=(2 * lay["flux"],) * 3)
    inner = wakis.FluxMonitor(FREQS, size=(2 * lay["inner"],) * 3)
    monitors = [scat, inner]
    out = dict(grid=grid, solver=solver, src=src, scat=scat, inner=inner, lay=lay)
    if extras:
        span = 2 * lay["pml"]
        plane = wakis.DFTMonitor(F_SNAP, components=("Ex", "Ey", "Ez"),
                                 size=(span, 0.0, span))  # fmt: skip
        vol = wakis.DFTMonitor(F_SNAP[1:2], components=("Ex", "Ey", "Ez"),
                               size=(2 * lay["tf"],) * 3)  # fmt: skip
        monitors += [plane, vol]
        out.update(plane=plane, vol=vol)
        if movie:
            rec = wv.SliceRecorder("Ex", axis="y", position=0.0,
                                   every=max(1, int(T_RUN / solver.dt / 400)))  # fmt: skip
            monitors.append(rec)
            out["movie"] = rec
    n_max = int(T_RUN / solver.dt)
    solver.verbose = verbose
    t1 = time.time()
    solver.emsolve(n_max, source=[src], monitors=monitors,
                   stop_when_energy_decayed=(energy_decay, 50))  # fmt: skip
    t_run = time.time() - t1
    # Cross sections: flux spectrum / incident flux spectrum
    phi_inc = src.incident_flux_density(FREQS)
    q_sca = scat.get_flux() / phi_inc / (np.pi * R_SPHERE**2)
    q_abs = -inner.get_flux() / phi_inc / (np.pi * R_SPHERE**2)
    out.update(
        x=X_SIM, q_sca=q_sca, q_abs=q_abs, steps=solver.Nt, dt=solver.dt,
        cells=int(grid.Nx * grid.Ny * grid.Nz), setup_s=t_setup, run_s=t_run,
        resolution=resolution_per_um, conformal=conformal,
        gpu=gpu_name() if use_gpu else None, stopped_early=bool(solver.Nt < n_max),
    )  # fmt: skip
    return out


def far_field_errors(res, theta=np.linspace(0, np.pi, 181), floor_db=40.0):
    """
    Far-field pattern of the scattered-field box against Mie at X_SNAP:
    RMS difference in dB of dsigma/dOmega per plane (E: phi = 0, H: phi =
    90 deg), with both clipped at `floor_db` below the Mie maximum so that
    pattern nulls do not dominate. Also returns the patterns [dB re r^2].
    """
    out = {}
    # one far-field evaluation per plane (all frequencies at once)
    ffs = {
        p: res["scat"].get_farfield(theta, phi)
        for p, phi in (("E", 0.0), ("H", np.pi / 2))
    }
    for xs in X_SNAP:
        j = int(np.argmin(np.abs(X_SIM - xs)))
        E_j, _ = res["src"].incident_fields(FREQS[j : j + 1])
        for plane, phi in (("E", 0.0), ("H", np.pi / 2)):
            ff = ffs[plane]
            sim = (np.abs(ff["Etheta"][j]) ** 2 + np.abs(ff["Ephi"][j]) ** 2) / abs(
                E_j[0]
            ) ** 2
            mie = mie_differential_cross_section(
                N_SPHERE, R_SPHERE, c0 / FREQS[j], theta, phi
            )
            db_s = 10 * np.log10(sim / R_SPHERE**2)
            db_m = 10 * np.log10(mie / R_SPHERE**2)
            lo = db_m.max() - floor_db
            out[f"x{xs:g}_{plane}"] = dict(
                rms_db=float(np.sqrt(np.mean((np.maximum(db_s, lo) - np.maximum(db_m, lo)) ** 2))),
                rel_l2=float(np.linalg.norm(sim - mie) / np.linalg.norm(mie)),
                db_sim=[round(float(v), 4) for v in db_s],
            )  # fmt: skip
    return out


def summary(res):
    """JSON-serializable summary of a run (no field data)."""
    q_mie = mie_efficiencies(N_SPHERE, res["x"])[1]
    err = res["q_sca"] - q_mie
    lay = res["lay"] if "lay" in res else layout(res["resolution"])
    return dict(
        pml_gap_scale=ARGS.pml_gap, pml_start_m=lay["pml"], n_pml=N_PML,
        gap_um=ARGS.gap_um, graded=lay["nodes"] is not None,
        kappa_max=ARGS.kappa_max, alpha_max=ARGS.alpha_max,
        far_field=far_field_errors(res),
        resolution_per_um=res["resolution"],
        stopped_early=res["stopped_early"], conformal=res["conformal"],
        cells=res["cells"], steps=int(res["steps"]), dt_s=res["dt"],
        setup_s=res["setup_s"], run_s=res["run_s"], gpu=res["gpu"],
        x=list(res["x"]), q_sca=list(map(float, res["q_sca"])),
        q_abs=list(map(float, res["q_abs"])), q_mie=list(map(float, q_mie)),
        rel_rms_error=float(np.sqrt(np.mean(err**2) / np.mean(q_mie**2))),
        max_abs_error=float(np.max(np.abs(err))),
        cells_per_wavelength_in_sphere_at_xmax=float(
            2 * np.pi * R_SPHERE / X_MAX / N_SPHERE / layout(res["resolution"])["dx"]
        ),
    )  # fmt: skip


if ARGS.arm:  # study mode: one run, JSON out, no plots
    res = run_mie(RESOLUTION, CONFORMAL, USE_GPU, extras=False)
    info = summary(res)
    out_path = Path(ARGS.out or f"mie_{RESOLUTION:g}_{CONFORMAL}.json")
    out_path.write_text(json.dumps(info, indent=1))
    print(f"wrote {out_path}: rel. RMS error {info['rel_rms_error']:.4f}, "
          f"{info['run_s']:.0f} s run, {info['setup_s']:.0f} s setup")  # fmt: skip
    sys.exit(0)

# %%
res = run_mie(RESOLUTION, CONFORMAL, USE_GPU, extras=True, movie=not ARGS.no_movie,
              verbose=1)  # fmt: skip
info = summary(res)
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / f"run_{'full' if FULL else 'quick'}.json").write_text(
    json.dumps(info, indent=1)
)
print(f"{info['cells'] / 1e6:.2f} Mcells, {info['steps']} steps, setup "
      f"{info['setup_s']:.0f} s, run {info['run_s']:.0f} s "
      f"({info['gpu'] or 'CPU'}); predicted {est.seconds:.0f} s")  # fmt: skip

# %% [markdown]
# ## 7. Fields: total inside, scattered outside
#
# The DFT plane monitor gives the complex field on the $y = 0$ plane at
# three size parameters. Inside the dashed TF/SF box we see the total
# field: the incident wave focused by the sphere (a "photonic nanojet" forms
# on its shadow side at large $x$). Outside the box we see the scattered
# field alone, with no trace of the incident plane wave: the log-scale
# panels show the incident wave cut off at the box faces.
# The fields are normalized by the source spectrum, so they are in units of
# the incident amplitude.

# %%
boxes = {"tfsf": res["src"].box_bounds(),
         "flux": wv.box_bounds((0, 0, 0), (2 * res["lay"]["flux"],) * 3)}  # fmt: skip
E_inc, _ = res["src"].incident_fields(F_SNAP)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for j, xs in enumerate(X_SNAP):
    ex, co = wv.field_slice(res["plane"], "y", 0.0, "Ex")
    ex = ex[j] / E_inc[j]  # relative to the incident wave at the entry plane
    wv.plot_field_slice(ex, co, kind="real", ax=axes[0, j], boxes=boxes, axis="y",
                        surface=((0, 0, 0), R_SPHERE), vmax=2.0,
                        title=f"Re $E_x$ / $|E_\\mathrm{{inc}}|$, x = {xs:g}",
                        colorbar_label="")  # fmt: skip
    wv.plot_field_slice(ex, co, kind="log", ax=axes[1, j], boxes=boxes, axis="y",
                        surface=((0, 0, 0), R_SPHERE), log_floor=1e-4,
                        title=f"log$_{{10}}$ |$E_x$| / $|E_\\mathrm{{inc}}|$, x = {xs:g}",
                        colorbar_label="")  # fmt: skip
fig.tight_layout()
savefig(fig, "fields_xz.png")
plt.show()

# %% [markdown]
# In three dimensions: isosurfaces of $|\mathbf E|$ (total field) inside the
# TF/SF box at $x = 6$, normalized to the incident amplitude. The bright
# lobes on the $+z$ side are the focus of the sphere acting as a ball lens.
# The volume monitor covers only the TF/SF box, so the focus is cut flat
# at the box's upper face. The three field components live on different
# edges; they are averaged to the cell centres before $|\mathbf E|$ is formed.


# %%
def to_cell_centres(values, comp):
    """
    Average one E component from its own edge positions to the cell centres.

    E_x lives at (x_{i+1/2}, y_j, z_k): averaging the two neighbouring nodes
    along y and along z moves it to (x_{i+1/2}, y_{j+1/2}, z_{k+1/2}), where
    E_y and E_z are moved too. Combining components sampled at different
    points (e.g. by trimming the arrays to a common shape) misaligns them by
    half a cell and breaks the mirror symmetry of |E|.
    """
    a = "xyz".index(comp)
    for d in range(3):
        if d != a:
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[d], hi[d] = slice(0, -1), slice(1, None)
            values = 0.5 * (values[tuple(lo)] + values[tuple(hi)])
    return values


comps = {c: res["vol"].get_field("E" + c) for c in "xyz"}
mag = np.sqrt(sum(np.abs(to_cell_centres(comps[c][0][0], c)) ** 2 for c in "xyz"))
mag = mag / abs(E_inc[1])
# Cell-centre coordinates: E_x samples sit at x midpoints already; its y, z
# nodes are averaged
coords3 = [np.asarray(comps["x"][1][0])]
coords3 += [0.5 * (np.asarray(q)[:-1] + np.asarray(q)[1:]) for q in comps["x"][1][1:]]
levels = (0.35, 0.55, 0.75)  # fractions of the maximum
pl = wv.plot_isosurfaces(mag, coords3, levels=levels, scalar_label="|E|/max",
                         surfaces=[res["grid"].read_stl("sphere")])  # fmt: skip
wv.save_plotter(pl, str(FIG_DIR / "isosurfaces.png"))
fig, ax = plt.subplots(figsize=(6, 4.6))
ax.imshow(plt.imread(FIG_DIR / "isosurfaces.png"))
ax.axis("off")
ax.set_title(f"|E| isosurfaces at x = {X_SNAP[1]:g} (max {mag.max():.1f} |E_inc|)")
plt.show()

# %% [markdown]
# **A symmetry check (always worth doing).** The sphere and the incidence
# are symmetric under the mirrors $x \to -x$ and $y \to -y$, so the fields
# must be too, with a parity per component: the incident field
# $\hat x e^{-ikz}$ changes sign under the $x$ mirror and not under the $y$
# mirror, which makes $E_x$ even in $x$ and $y$, $E_y$ odd in both, and
# $E_z$ odd in $x$ and even in $y$. On a grid centred on the sphere every
# component's own samples form a mirror-symmetric set, so the check can be
# done on the raw data, $\|F(\mathbf r) - pF(M\mathbf r)\|/\|F\|$. Anything
# well above the run's truncation level points at a set-up error
# (off-centre grid or box, asymmetric mesh of the STL) or a post-processing
# error, such as combining staggered components without interpolating them,
# which here would give 10-20 % "asymmetry" of $|E|$.

# %%
parity = {
    ("x", 0): 1,
    ("y", 0): -1,
    ("z", 0): -1,
    ("x", 1): 1,
    ("y", 1): -1,
    ("z", 1): 1,
}
for c in "xyz":
    F, co = comps[c][0][0], comps[c][1]
    row = []
    for ax_ in (0, 1):
        assert np.allclose(co[ax_], -np.asarray(co[ax_])[::-1]), "samples not symmetric"
        G = np.flip(F, axis=ax_)
        row.append(np.linalg.norm(F - parity[(c, ax_)] * G) / np.linalg.norm(F))
    print(f"E{c}: mirror x {row[0]:.1e}, mirror y {row[1]:.1e}")
for ax_, name in ((0, "x"), (1, "y")):
    print(f"|E| at the cell centres: mirror {name} "
          f"{np.linalg.norm(mag - np.flip(mag, ax_)) / np.linalg.norm(mag):.1e}")  # fmt: skip

# %% [markdown]
# And the time domain: $E_x$ on the $y = 0$ plane as the pulse sweeps
# through the box. The incident pulse appears at the lower face, travels
# up and vanishes at the upper face; the sphere's scattered wavelets leave
# the box in all directions and are absorbed by the CPML.

# %%
if "movie" in res:
    gif = wv.save_animation(
        res["movie"], str(FIG_DIR / "mie_movie.gif"), boxes=boxes,
        surface=((0, 0, 0), R_SPHERE), max_frames=72, fps=12, dpi=65,
        title="Ex(y=0)",
    )  # fmt: skip
    print("movie:", gif, f"{os.path.getsize(gif) / 1e6:.2f} MB")
    if IN_NOTEBOOK:
        from IPython.display import Image, display

        display(Image(filename=gif))

# %% [markdown]
# ## 8. $Q_\mathrm{sca}(x)$ versus Mie theory, and the far field
#
# The scattered-field flux box gives $\Phi_\mathrm{sca}(f)$; the source
# reports the incident flux density spectrum $\Phi_\mathrm{inc}(f)$ with the
# same DFT convention, so their ratio is the cross section with no further
# normalization (the pulse spectrum cancels):
#
# $$Q_\mathrm{sca}(x) = \frac{\Phi_\mathrm{sca}(f)}{\Phi_\mathrm{inc}(f)\,
#   \pi r^2}.$$
#
# The inner flux box, inside the total-field region, measures
# $-P_\mathrm{abs}$; for this lossless sphere $Q_\mathrm{abs}$ must vanish to
# within the discretization error, a free consistency check.

# %%
q_mie = mie_efficiencies(N_SPHERE, X_SIM)[1]
x_fine = np.linspace(X_MIN, X_MAX, 800)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1.3]))  # fmt: skip
ax.plot(
    x_fine,
    mie_efficiencies(N_SPHERE, x_fine)[1],
    "k",
    lw=1.2,
    label="exact series (Mie)",
)
ax.plot(X_SIM, res["q_sca"], "o", ms=3.5, color="#D55E00",
        label=f"Wakis (FIT) ({CONFORMAL}, {RESOLUTION:g} cells/µm)")  # fmt: skip
ax.plot(
    X_SIM,
    res["q_abs"],
    "s",
    ms=2.5,
    color="#0072B2",
    label=r"$Q_\mathrm{abs}$ (should be 0)",
)
ax.set(ylabel=r"$Q_\mathrm{sca}$", title=f"Lossless sphere n = {N_SPHERE:g}, r = 1 µm")
ax.legend(fontsize=8)
ax2.plot(X_SIM, res["q_sca"] - q_mie, color="#D55E00")
ax2.axhline(0, color="k", lw=0.6)
ax2.set(
    xlabel=r"size parameter $x = 2\pi r/\lambda$", ylabel=r"$\Delta Q_\mathrm{sca}$"
)
sec = ax.secondary_xaxis("top", functions=(lambda x: 2 * np.pi / np.maximum(x, 1e-3),
                                            lambda w: 2 * np.pi / np.maximum(w, 1e-3)))  # fmt: skip
sec.set_xlabel(r"vacuum wavelength $\lambda$ [µm]")
fig.tight_layout()
savefig(fig, "qsca_vs_mie.png")
plt.show()
print(f"relative RMS error {info['rel_rms_error']:.2%}, max |ΔQ| "
      f"{info['max_abs_error']:.3f}, max |Q_abs| {np.max(np.abs(res['q_abs'])):.2e}")  # fmt: skip

# %% [markdown]
# **Far field.** The same box, used as a `Near2FarMonitor`, propagates the
# scattered near field to infinity with the surface equivalence principle
# (the *other* use of the theorem of section 2). The differential cross
# section is
#
# $$\frac{d\sigma}{d\Omega} = \frac{|\mathbf F(\theta,\phi)|^2}
#   {|E_\mathrm{inc}|^2},\qquad
#   \text{Mie: } \frac{|S_2|^2\cos^2\phi + |S_1|^2\sin^2\phi}{k^2},$$
#
# with $\mathbf F = r e^{ikr}\mathbf E_\mathrm{far}$ the pattern. We compare
# the E-plane ($\phi = 0$, the $xz$ plane, $|S_2|^2$) and the H-plane
# ($\phi = 90°$, the $yz$ plane, $|S_1|^2$). **How to read the polar plots:**
# each panel is split, the left half shows the E-plane and the right half
# the H-plane, both for $\theta = 0 \dots 180°$ from the forward direction
# (top). The two halves differ because the two planes scatter differently
# ($S_2 \ne S_1$ away from $\theta = 0, 180°$); each plane on its own is
# symmetric in $\pm\theta$, so plotting it twice would add nothing.

# %%
theta = np.linspace(0, np.pi, 181)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), subplot_kw=dict(projection="polar"))
for ax, xs in zip(axes, X_SNAP):
    j = int(np.argmin(np.abs(X_SIM - xs)))
    E_j, _ = res["src"].incident_fields(FREQS[j : j + 1])
    top = -np.inf
    # Right half: E-plane (phi = 0); left half: H-plane (phi = 90 deg)
    for phi, sgn, col in ((0.0, 1, "#D55E00"), (np.pi / 2, -1, "#0072B2")):
        ff = res["scat"].get_farfield(theta, phi)
        dsig = np.abs(ff["Etheta"][j]) ** 2 + np.abs(ff["Ephi"][j]) ** 2
        dsig = dsig / abs(E_j[0]) ** 2
        mie = mie_differential_cross_section(
            N_SPHERE, R_SPHERE, c0 / FREQS[j], theta, phi
        )
        db_mie = 10 * np.log10(mie / R_SPHERE**2)
        db_sim = 10 * np.log10(dsig / R_SPHERE**2)
        top = max(top, db_mie.max())
        ax.plot(sgn * theta, db_mie, color=col, lw=2.2, alpha=0.55,
                label=("E-plane" if sgn > 0 else "H-plane") + " exact series (Mie)")  # fmt: skip
        ax.plot(sgn * theta, db_sim, "--", color="k", lw=0.9,
                label="Wakis (FIT)" if sgn > 0 else None)  # fmt: skip
    ax.set_rlim(top - 40, top + 3)
    ax.set_rgrids(np.round(np.arange(top - 30, top + 1, 10)), fontsize=6, angle=200)
    ax.set_theta_zero_location("N")
    ax.set_thetagrids(range(0, 360, 45))
    ax.set_title(f"x = {X_SIM[j]:g}:  dσ/dΩ / r²  [dB]", fontsize=9)
    ax.text(np.radians(60), top + 12, "E-plane", color="#D55E00", fontsize=8)
    ax.text(np.radians(-80), top + 14, "H-plane", color="#0072B2", fontsize=8)
axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.3, -0.2))
fig.suptitle("left half: E-plane (xz, $|S_2|^2$)   ·   right half: H-plane (yz, "
             "$|S_1|^2$)   ·   each plane is symmetric in $\\pm\\theta$",
             fontsize=10, y=1.02)  # fmt: skip
fig.tight_layout()
savefig(fig, "far_field_vs_mie.png")
plt.show()

# %% [markdown]
# ## 8b. Set-up check: is the PML far enough?
#
# In the 40 cells/µm run above, the CPML starts at $2.05$ µm ($1.05$ µm from
# the sphere surface), which is only $0.33\,\lambda$ at $x = 2$, and the
# far field still shows $\sim 1$ dB deviations from Mie at $x = 9$ E-plane
# ($100\text{–}150°$) and $x = 6$ H-plane ($\sim 150°$). Could the CPML be
# absorbing the sphere's reactive near field or reflecting grazing waves?
# Rather than guessing, we test the set-up directly
# (`studies/sphere-pml-gap/`): we switch on a complex-frequency-shifted
# (CFS) CPML (`--alpha-max 1000`, $\alpha/\varepsilon_0 \approx
# 0.2\,\omega_\mathrm{min}$), push the CPML from $1.05$ to $2.1$ and
# $4.2$ µm on a graded grid (`--gap-um`, uniform 40 cells/µm out to the flux
# box, cells growing $\le 1.15\times$ to 16 cells/µm in the vacuum gap), and
# refine the uniform grid from 40 to 50 cells/µm.
#
# | run | cells | Q_sca RMS err x=2–10 | far field RMS dB x=3 E/H, 6 E/H, 9 E/H |
# |---|---|---|---|
# | uniform, gap 1.05 µm (tutorial, 40 cells/µm) | 6.3 M | 3.40 % | 0.07/0.19, 0.23/0.62, 1.17/1.36 |
# | same, CFS CPML (alpha_max = 1000 S/m) | 6.3 M | 3.40 % | identical |
# | graded, gap 1.05 µm | 4.3 M | 3.37 % | 0.07/0.19, 0.19/0.65, 1.18/1.37 |
# | graded, gap 2.1 µm (≈1 λ at x=3) | 7.6 M | 3.37 % | same |
# | graded, gap 4.2 µm (≈2 λ at x=3) | 18.6 M | 3.37 % | same |
# | uniform, gap 1.05 µm, 50 cells/µm | 11.7 M | 2.92 % | 0.06/0.20, 0.21/0.47, 0.78/0.85 |
#
# * **Gap distance and CFS shift do not matter here.** Between the graded
#   $1.05$ µm and $4.2$ µm gaps, $\max|\Delta Q_\mathrm{sca}| = 3.7\times
#   10^{-4}$ (3.7e-4, absolute) and the max far-field difference is
#   $0.010$ dB ($2.1$ vs $4.2$ µm: 3.2e-4 and $0.007$ dB). Adding CFS
#   $\alpha_\mathrm{max} = 1000$ S/m produces no change at the printed
#   digits. The small difference between the uniform and graded $1.05$ µm
#   runs ($\max|\Delta Q| = 1.0\times 10^{-2}$, $0.43$ dB) comes from the
#   coarser grid outside the flux box, not from the CPML position.
# * **The remaining error is bulk discretization.** The hypothesis "PML
#   absorbs the near field" is rejected: the CPML $1.05$ µm from the sphere
#   does not affect $Q_\mathrm{sca}$ or the far field. Refining the uniform
#   grid from 40 to 50 cells/µm shrinks the deviations from Mie ($1.17 \to
#   0.78$ dB at $x = 9$ E-plane, $1.36 \to 0.85$ dB at $x = 9$ H-plane,
#   $0.62 \to 0.47$ dB at $x = 6$ H-plane; $Q_\mathrm{sca}$ RMS error
#   $3.40 \to 2.92\,\%$).
# * **Why $1.05$ µm is enough, and what to take away.** The reactive
#   near-field decay length is $\lambda / 2\pi = r / x$, which is at most
#   $0.5$ µm across our band ($x \ge 2$), so at $1.05$ µm the CPML sees
#   almost purely propagating scattered waves that it absorbs well.
#   **Rule of thumb:** always test numerical set-up choices instead of
#   assuming; keep the CPML at least $\sim\lambda / 2\pi$ beyond the near
#   field of the scatterer (and further, or with CFS $\alpha_\mathrm{max}$,
#   for strongly resonant or lossy/plasmonic near fields); and use a graded
#   grid outside the monitor box to make a large vacuum buffer cheap.

# %%
pml_fig = (
    HERE.parents[0].parent
    / "studies"
    / "sphere-pml-gap"
    / "figures"
    / "pml_gap_far_field.png"
    if HERE.parents
    else Path("pml_gap_far_field.png")
)
if not pml_fig.exists():
    pml_fig = FIG_DIR / "pml_gap_far_field.png"
if pml_fig.exists():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(plt.imread(pml_fig))
    ax.axis("off")
    plt.show()

# %% [markdown]
# ## 9. Convergence: staircase versus conformal at equal cells per wavelength
#
# How fast does the error fall as the grid is refined, and what does
# conformal averaging buy? The study below runs the same simulation at
# 10 to 50 cells/µm for the three material models. We plot the relative RMS
# error of $Q_\mathrm{sca}$ over the band against the number of cells per
# wavelength *inside the sphere at the shortest wavelength* ($x = 10$:
# $\lambda/n = 0.314$ µm), the hardest-resolved length in the problem, and
# against the measured run time (an equal-cost comparison). The runs are
# independent, so they are launched in parallel
# (`tutorials/study.py`); the notebook loads
# their committed summary from `data/convergence.json`.


# %%
def load_convergence():
    path = DATA_DIR / "convergence.json"
    if path.exists():
        return json.loads(path.read_text())
    try:  # e.g. on Colab, where only the notebook was downloaded
        import urllib.request

        url = ("https://raw.githubusercontent.com/eshipton/Wakis/main/tutorials/"
               "03_diffraction_and_scattering/03-05_mie_sphere/data/convergence.json")  # fmt: skip
        with urllib.request.urlopen(url, timeout=20) as f:
            return json.loads(f.read())
    except Exception as err:
        print("convergence data not available:", err)
        return None


study = load_convergence()
if study:
    styles = {"staircase": ("o-", "#D55E00"), "all": ("s-", "#0072B2"),
              "tensor": ("^-", "#009E73")}  # fmt: skip
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, (mk, col) in styles.items():
        runs = sorted((r for r in study["runs"] if r["conformal"] == model),
                      key=lambda r: r["resolution_per_um"])  # fmt: skip
        if not runs:
            continue
        cpw = [r["cells_per_wavelength_in_sphere_at_xmax"] for r in runs]
        err = [r["rel_rms_error"] for r in runs]
        cost = [r["run_s"] + r["setup_s"] for r in runs]
        label = {"staircase": "staircase", "all": "conformal ('all')",
                 "tensor": "tensor averaging"}[model]  # fmt: skip
        ax.loglog(cpw, err, mk, color=col, label=label)
        ax2.loglog(cost, err, mk, color=col, label=label)
    ref = np.array([3, 16])
    for order, ls in ((1, ":"), (2, "--")):
        e0 = study["runs"][0]["rel_rms_error"]
        ax.loglog(ref, 0.3 * (ref / ref[0]) ** (-order), "k" + ls, lw=0.8,
                  label=f"order {order}")  # fmt: skip
    ax.set(xlabel=r"cells per wavelength in the sphere at $x = 10$",
           ylabel=r"relative RMS error of $Q_\mathrm{sca}$")  # fmt: skip
    ax2.set(xlabel=f"wall time [s] ({study.get('machine', '')})",
            ylabel=r"relative RMS error of $Q_\mathrm{sca}$")  # fmt: skip
    ax.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "convergence.png")
    plt.show()

# %% [markdown]
# ## 10. Limits and failure modes
#
# * **Resolution inside the sphere.** At $x = 10$ the wavelength inside the
#   $n = 2$ sphere is $0.314$ µm. Below ~10 cells per internal wavelength the
#   numerical phase velocity error shifts the narrow high-order resonances
#   (visible as horizontally shifted peaks in $\Delta Q$), and the staircase
#   model adds an erratic effective-radius error of up to half a cell.
# * **Truncated ring-down.** High-order resonances ring for many periods.
#   Stopping the run before the field energy has decayed turns the DFT into
#   a windowed transform and adds spurious ripples to $Q(x)$; the energy
#   criterion (`stop_when_energy_decayed`) guards against this.
# * **Box placement.** The TF/SF faces must not touch the scatterer or the
#   CPML, and the flux box must lie entirely in the scattered-field region
#   (otherwise it integrates the incident wave too). A flux box very close
#   to the CPML picks up the small reflections of evanescent near fields.
# * **TF/SF limitations** of the current implementation: axis-aligned
#   incidence, homogeneous lossless background around the box surface, no
#   MPI; oblique incidence and layered backgrounds (a particle on a
#   substrate) need a different incident-field solver.
# * **Conformal models** assume a smooth surface on the scale of a cell;
#   sharp edges and corners smaller than a cell are averaged away.
# * **What agreement does not prove.** Matching $Q_\mathrm{sca}$ checks the
#   power balance, not the phase of the near field; the far-field pattern
#   comparison and the $Q_\mathrm{abs} = 0$ check test different parts of
#   the pipeline.

# %% [markdown]
# ## 11. Exercises
#
# 1. **Resonances.** Change the index to $n = 1.5$ and $n = 3.5$ (silicon in
#    the near infrared). How do the resonance widths change, and how many
#    cells per internal wavelength do you need for 1 % RMS error?
#    *Hint:* run `preflight` first.
# 2. **Where is the error?** Plot $\Delta Q(x)$ for two resolutions. Is the
#    error concentrated at the resonances? Shift the curves in $x$ to
#    separate a phase-velocity (dispersion) error from an amplitude error.
# 3. **Equivalent currents by hand.** Rebuild the TF/SF box from
#    `CurrentSource` and `MagneticCurrentSource` planes on the six faces
#    with the analytic plane-wave phase, as done in the companion Meep
#    notebook (*Total-Field/Scattered-Field Technique with Meep FDTD*). Run
#    without the sphere and measure the leakage flux out of the scattered
#    flux box versus frequency; compare with `TFSFSource`. Which part of the
#    leakage comes from the half-cell offset of E and H, and which from
#    numerical dispersion?
# 4. **Two-run subtraction** (Oskooi & Johnson, sec. 4.3). Replace the TF/SF
#    box by a plane source below the sphere, run once with and once without
#    the sphere, and subtract the DFT fields on the flux box. Compare the
#    accuracy and the cost with TF/SF.
# 5. **Symmetry test.** Rotate the incidence to `direction="-x",
#    polarization="z"`. $Q_\mathrm{sca}$ must not change for a sphere: how
#    large is the difference on the grid, and how does it scale with
#    resolution? (A test that can fail: the grid is not isotropic.)
# 6. **Optical theorem.** Compute $Q_\mathrm{ext} = \frac{4}{x^2}
#    \mathrm{Re}\,S(0)$ from the forward far-field amplitude of the
#    `Near2FarMonitor` (mind the phase reference and the time convention)
#    and compare it with $Q_\mathrm{sca} + Q_\mathrm{abs}$.
# 7. **Towards plasmonics** (the next tutorial). Make the sphere silver or
#    gold with a Drude-Lorentz model from `wakis.dispersive`, sweep the radius
#    from 10 nm to 1 µm, and use the inner flux box for $Q_\mathrm{abs}$.
# 8. **CPML distance and CFS for plasmonics.** Move the CPML (`--gap-um`)
#    and switch on CFS (`--alpha-max`) for a lossy (Drude) sphere near its
#    plasmon resonance: does the conclusion still hold?

# %% [markdown]
# ## 12. References
#
# 1. D. E. Merewether, R. Fisher and F. W. Smith, "On implementing a numeric
#    Huygen's source scheme in a finite difference program to illuminate
#    scattering bodies," *IEEE Trans. Nucl. Sci.* **27**(6), 1829-1833
#    (1980).
# 2. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed. (Artech House, 2005).
# 3. A. Oskooi and S. G. Johnson, "Electromagnetic wave source conditions,"
#    ch. 4 in A. Taflove, A. Oskooi and S. G. Johnson (eds.), *Advances in
#    FDTD Computational Electrodynamics: Photonics and Nanotechnology*
#    (Artech House, 2013); arXiv:1301.5366.
# 4. C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by
#    Small Particles* (Wiley, 1983).
# 5. W. J. Wiscombe, "Improved Mie scattering algorithms," *Appl. Opt.*
#    **19**, 1505-1509 (1980).
# 6. C. Kottke, A. Farjadpour and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel
#    smoothing of discretized numerical methods," *Phys. Rev. E* **77**,
#    036611 (2008).
# 7. Meep tutorial "Mie scattering of a lossless dielectric sphere"
#    (https://meep.readthedocs.io), whose physical set-up (n = 2, r = 1 µm,
#    x = 2 to 10) this tutorial reproduces with an independent code.
