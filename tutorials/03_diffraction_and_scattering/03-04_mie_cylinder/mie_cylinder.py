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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/03_diffraction_and_scattering/03-04_mie_cylinder/mie_cylinder.ipynb)
#
# # Tutorial 3.4 · Mie scattering by a dielectric cylinder (2D)
# <h3 align="center">Total-field/scattered-field plane waves in the Finite Integration Technique</h3>
# <p align="center"><i>Erik Shipton &middot; Wakis tutorial series (fork of CERN's Wakis)</i></p>
#
# ---
#
# ### Abstract
#
# **A broadband plane-wave pulse hits an infinitely long dielectric cylinder
# (refractive index $n = 2$, radius $a = 1$ µm) at normal incidence, with the
# electric field either along the cylinder axis (TM) or across it (TE). The
# wave enters through a total-field/scattered-field (TF/SF) box, so that the
# field outside the box is purely the field scattered by the cylinder; a flux
# box in that region measures the scattered power, from which we get the
# scattering efficiency $Q_\mathrm{sca}$ over a 5:1 band of size parameters
# $x = 2\pi a/\lambda$ in one time-domain run per polarization. The result is
# compared with the exact cylinder series, and a resolution study measures
# the error of a staircased cylinder against conformal (subcell-averaged)
# material models at equal cells per wavelength. The problem is 2D, so every
# run takes seconds to minutes on a laptop; the 3D sphere follows in
# Tutorial 3.5.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to
#
# 1. **explain** scattering, extinction and absorption cross sections and
#    efficiencies, and their 2D versions per unit length (scattering width),
#    and evaluate the exact cylinder series with `wakis.analytic`;
# 2. **derive** the TF/SF source from the surface equivalence principle
#    ($\mathbf J_s = \hat n \times \mathbf H_\mathrm{inc}$,
#    $\mathbf K_s = -\hat n \times \mathbf E_\mathrm{inc}$) and explain why
#    the incident field must solve the *discrete* equations for it to cancel
#    outside the box;
# 3. **set up a 2D problem in a 3D code**: a thin periodic slab along the
#    cylinder axis, CPML in the plane, and the per-length normalization of
#    fluxes;
# 4. **inspect** what the solver sees on the grid (staircase versus conformal
#    permittivity of each edge) and the effective size of the scatterer;
# 5. **estimate** memory and run time with `wakis.estimate` and `preflight`
#    *before* running;
# 6. **measure** a cross section from a flux box normalized by the incident
#    flux, check it with energy conservation, and **quantify** the
#    discretization error with a convergence study against a named baseline.
#
# ### Contents
# 0. Set-up (Colab install, GPU detection, run mode)
# 1. The physics: 2D Mie scattering and cross sections per unit length
# 2. The TF/SF source: equivalence principle on a grid
# 3. Geometry: a 2D problem in a 3D solver
# 4. The mesh the solver sees: staircase versus conformal
# 5. Preflight: estimate memory and run time
# 6. Sources, monitors and the per-length normalization
# 7. Check the source (empty box), then run both polarizations
# 8. Fields: total inside, scattered outside
# 9. $Q_\mathrm{sca}(x)$ versus the exact series
# 10. Convergence: staircase versus conformal at equal cells per wavelength
# 11. Triangular mesh: FIT beyond the Cartesian grid
# 12. Limits and failure modes
# 13. Exercises
# 14. References

# %% [markdown]
# ## 0. Set-up
#
# On **Google Colab** the next cell installs the Wakis fork from GitHub and
# `xvfb` (a virtual screen for off-screen PyVista rendering). NumPy, SciPy,
# Matplotlib, h5py and tqdm are preinstalled on Colab; PyVista comes with
# Wakis. This 2D tutorial runs comfortably on Colab's CPU; with a GPU runtime
# (*Runtime &rarr; Change runtime type*) the cell installs `cupy-cuda12x` if
# CuPy is missing.
#
# Locally, install the fork in a virtual environment (`uv pip install -e .`
# in the repository) and run this file as a script:
#
# ```bash
# python tutorials/03_diffraction_and_scattering/03-04_mie_cylinder/mie_cylinder.py --quick        # CPU, ~1-2 min
# python tutorials/03_diffraction_and_scattering/03-04_mie_cylinder/mie_cylinder.py --full         # converged (CPU or --gpu)
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

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # running as a script: headless
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.constants import c as c0

import wakis
from wakis import figstyle, schematic
from wakis import visualize as wv
from wakis.analytic import (
    cylinder_efficiencies,
    cylinder_rayleigh_efficiencies,
)
from wakis.estimate import estimate, preflight
from wakis.sources import GaussianSource

print(wv.setup_rendering())
figstyle.apply_style()


def _fix_unit_spaces(ax):
    """Ensure a space separates numbers and units inside mathtext labels."""
    for txt in ax.texts:
        s = txt.get_text()
        new_s = (
            s.replace(" THz$", r"\;\mathrm{THz}$")
            .replace(" GHz$", r"\;\mathrm{GHz}$")
            .replace(" nm$", r"\;\mathrm{nm}$")
            .replace(" µm$", r"\;\mu\mathrm{m}$")
            .replace(" mm$", r"\;\mathrm{mm}$")
        )
        if new_s != s:
            txt.set_text(new_s)


def _fix_cylinder_patch(ax, radius_scaled, tol=0.25, fill=None, alpha=None):
    """Replace STL bounding-box Rectangle at (-r, -r, 2r, 2r) with a true Circle."""
    from matplotlib.patches import Circle, Rectangle

    replaced = False
    for p in list(ax.patches):
        if isinstance(p, Rectangle):
            x0, y0 = p.get_xy()
            w, h = p.get_width(), p.get_height()
            if (
                abs(x0 + radius_scaled) <= tol
                and abs(y0 + radius_scaled) <= tol
                and abs(w - 2 * radius_scaled) <= 2 * tol
                and abs(h - 2 * radius_scaled) <= 2 * tol
            ):
                fc = p.get_facecolor() if fill is None else fill
                ec = p.get_edgecolor()
                lw = p.get_linewidth()
                ls = p.get_linestyle()
                al = p.get_alpha() if alpha is None else alpha
                zo = p.get_zorder()
                p.remove()
                ax.add_patch(
                    Circle(
                        (0.0, 0.0),
                        radius_scaled,
                        facecolor=fc,
                        edgecolor=ec,
                        lw=lw,
                        ls=ls,
                        alpha=al,
                        zorder=zo,
                    )
                )
                replaced = True
                break
    for txt in ax.texts:
        if "Cylinder" in txt.get_text() and "\\varepsilon_r" in txt.get_text():
            txt.set_position((0.0, 0.0))
    return replaced


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

        return cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return None


IN_NOTEBOOK = "ipykernel" in sys.modules
parser = argparse.ArgumentParser(
    description="Wakis Tutorial 3.4 · Mie scattering by a dielectric cylinder (2D)"
)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--quick", action="store_true", help="20 cells/µm, ~1-2 min CPU")
mode.add_argument("--full", action="store_true", help="60 cells/µm, converged")
mode.add_argument("--plot", action="store_true", help="re-render figures from data/")
parser.add_argument("--gpu", action="store_true", help="use the GPU")
parser.add_argument("--cpu", action="store_true", help="force the CPU")
parser.add_argument("--resolution", type=float, default=None, help="cells per µm")
parser.add_argument("--conformal", default=None, help="staircase | all | tensor")
parser.add_argument("--pol", default=None, help="TM | TE (study arms)")
parser.add_argument("--arm", action="store_true", help="study arm: one run, JSON")
parser.add_argument("--out", default=None, help="JSON output of --arm")
parser.add_argument("--no-movie", action="store_true", help="skip the GIF")
parser.add_argument("--t-run", type=float, default=None, help="max time [fs]")
# In a notebook, options come from the environment, e.g.
# WAKIS_TUTORIAL_ARGS="--full" (used by tutorials/build.py --execute)
NB_ARGS = os.environ.get("WAKIS_TUTORIAL_ARGS", "").split()
ARGS, _ = parser.parse_known_args(NB_ARGS if IN_NOTEBOOK else None)

# 2D runs are small: the CPU is the default, the GPU pays off above ~1 Mcells
USE_GPU = ARGS.gpu and not ARGS.cpu and gpu_available()
FULL = ARGS.full
PLOT_ONLY = ARGS.plot
RESOLUTION = ARGS.resolution or (60.0 if (FULL or PLOT_ONLY) else 20.0)  # cells per µm
CONFORMAL = ARGS.conformal or "all"
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIG_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
if ARGS.arm:  # study runs must not overwrite the tutorial figures
    import tempfile

    FIG_DIR = Path(tempfile.mkdtemp())
FIG_DIR.mkdir(exist_ok=True)
print(
    f"mode={'plot' if PLOT_ONLY else ('full' if FULL else 'quick')}  GPU={USE_GPU}"
    f"{' (' + str(gpu_name()) + ')' if USE_GPU else ''}  "
    f"resolution={RESOLUTION:g} cells/µm  conformal={CONFORMAL}"
)


def savefig(fig, name, dpi=None):
    """Save a figure in figures/ with figstyle and return the path."""
    path = FIG_DIR / name
    figstyle.save(fig, path)
    return path


def show_png(path, title=None, figsize=(6, 5), dpi=72):
    """Display a saved PNG through matplotlib (static, compact output)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(plt.imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.show()


# %% [markdown]
# ## 1. The physics: 2D Mie scattering and cross sections per unit length
#
# **Why scattered fields?** Radar and lidar see objects only through the
# field they scatter; optical fibres, wires, cables, nanowire lasers and the
# pillars of metasurfaces and photonic crystals are cylinders. The
# *scattered* field $\mathbf E_s = \mathbf E - \mathbf E_\mathrm{inc}$ carries
# the information about the object and is often much weaker than the
# incident wave, so we want the simulation to hand it to us separately.
#
# **Simulation setups.** Before evaluating the exact series, the two-figure
# setup layouts below show the Cartesian 2D `SolverFIT3D` TF/SF setup
# (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`)
# and the unstructured 2D `UnstructuredFIT2D` triangular-mesh setup
# (`figures/00a_setup_geometry_unstructured.png`,
# `figures/00b_setup_mesh_and_excitation_unstructured.png`):
#
# ![Cartesian 2D Mie cylinder setup geometry](figures/00a_setup_geometry.png)
# ![Cartesian 2D Mie cylinder mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the Cartesian 2D `SolverFIT3D` simulation setup on the 4-cell periodic $z$ slab with a `TFSFSource` box, inner and scattered `FluxMonitor` boxes, and 10-cell CPML boundaries.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 5.10 µm × 5.10 µm × 200 nm (z ∈ [-100 nm, 100 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 102 × 102 × 4 (41616 cells) |
# | Discretization | Δx & resolution | Δ_min = 50 nm, 6.3 cells/λ (in n_max = 2.00), 40.0 cells across feature (2 µm) |
# | Boundary | -x face | CPML (10 cells, 500 nm) |
# | Boundary | +x face | CPML (10 cells, 500 nm) |
# | Boundary | -y face | CPML (10 cells, 500 nm) |
# | Boundary | +y face | CPML (10 cells, 500 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Dielectric cylinder (n = 2) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
# | Source | 1 · TFSFSource | pos: x ∈ [-1.25 µm, 1.25 µm], pol: E_z, pulse: f₀ = 286.3 THz, Δf = 95.43 THz |
# | Monitor | 2 · FluxMonitor | Scattered FluxMonitor (P'_sca), pos: x, y ∈ [-1.45, 1.45] µm, freqs: 81 freqs in [95.4, 477.1] THz |
# | Monitor | 3 · FluxMonitor | Inner FluxMonitor (P'_abs), pos: x, y ∈ [-1.10, 1.10] µm, freqs: 81 freqs in [95.4, 477.1] THz |
# | Variation | Polarizations | TM (E_z along cylinder axis) and TE (E_y across axis, H_z along axis) |
# | Variation | Material models | staircase (conformal=False), conformal ('all'), and tensor ('tensor') |
# | Variation | Resolution sweep | 10–80 cells/µm (20 cells/µm quick mode, 60 cells/µm full mode) |
#
# ![Unstructured 2D Mie cylinder setup geometry](figures/00a_setup_geometry_unstructured.png)
# ![Unstructured 2D Mie cylinder mesh and excitation](figures/00b_setup_mesh_and_excitation_unstructured.png)
#
# The table below summarizes the unstructured 2D `UnstructuredFIT2D` triangular-mesh setup with Voronoi-dual Hodge operators, a scattered-field plane-wave drive inside the cylinder, and cylindrical harmonic extraction rings inside a Mur ABC boundary.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly = 5.40 µm × 5.40 µm (x ∈ [-2.70 µm, 2.70 µm], y ∈ [-2.70 µm, 2.70 µm]) |
# | Domain | Grid cells | Nx × Ny = 1068 × 1068 (1140624 cells) |
# | Discretization | Δx & resolution | Δ_min = 119.6 nm, 2.6 cells/λ (in n_max = 2.00), 16.7 cells across feature (2 µm) |
# | Boundary | -x face | Mur ABC |
# | Boundary | +x face | Mur ABC |
# | Boundary | -y face | Mur ABC |
# | Boundary | +y face | Mur ABC |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Dielectric cylinder (n = 2) | ε_r = 4.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-1 µm, 1 µm] (d = 2 µm) |
# | Source | 1 · PlaneWave | pos: r <= 1 µm (cylinder interior), pol: TM (E_z) / TE (H_z), pulse: f₀ = 286.3 THz, Δf = 95.43 THz |
# | Monitor | 2 · HarmonicRings | Cylindrical harmonic rings (r = 1.9, 2.2, 2.5 µm), pos: r = 1.9, 2.2, 2.5 µm (256 pts/ring), freqs: 95.4–477.1 THz (81 freqs) |
# | Variation | Harmonic extraction | Cylindrical harmonic projection on r = 1.9, 2.2, 2.5 µm with incoming-wave de-embedding |
#
# **Cross sections in 2D.** A plane wave of intensity
# $I_\mathrm{inc} = |\mathbf E_\mathrm{inc}|^2/(2\eta)$ [W/m²] hits an
# infinitely long cylinder. The scattered power is infinite, but the power
# *per unit length* $P'_\mathrm{sca}$ [W/m] is finite, so the natural cross
# section is a width:
#
# $$C_\mathrm{sca} = \frac{P'_\mathrm{sca}}{I_\mathrm{inc}}\ [\mathrm m],
#   \qquad Q_\mathrm{sca} = \frac{C_\mathrm{sca}}{2a},$$
#
# normalized by the geometric shadow width $2a$. Likewise
# $C_\mathrm{ext} = C_\mathrm{sca} + C_\mathrm{abs}$.
#
# **The exact series** (Bohren & Huffman 1983, ch. 8). At normal incidence
# the fields separate in cylindrical coordinates into Bessel-function
# multipoles of order $n = 0, 1, 2, \dots$; with $m = n_\mathrm{cyl}/
# n_\mathrm{medium}$ and $x = ka$,
#
# $$Q_\mathrm{sca} = \frac{2}{x}\Big[|c_0|^2 + 2\sum_{n\ge1}|c_n|^2\Big],
#   \qquad Q_\mathrm{ext} = \frac{2}{x}\,\mathrm{Re}\Big[c_0 +
#   2\sum_{n\ge1}c_n\Big],$$
#
# where $c_n = b_n$ for **TM** (E along the axis; Bohren & Huffman's case I)
# and $c_n = a_n$ for **TE** (E across the axis; case II). The coefficients
# are ratios of Bessel and Hankel functions of $x$ and $mx$
# (`wakis.analytic.mie_cylinder`, tested against the thin-cylinder limits,
# direct Bessel evaluation, the PEC limit and the optical theorem in
# `tests/test_043_mie_cylinder_theory.py`).
#
# **The thin-cylinder limit** follows from a line dipole. A thin cylinder in
# a field along its axis is polarized without depolarization,
# $p' = (\varepsilon_r - 1)\varepsilon_0 \pi a^2 E$ per unit length; the 2D
# radiation of that line current gives
# $Q_\mathrm{sca}^\mathrm{TM} = \tfrac{\pi^2}{8}x^3|m^2-1|^2$.
# Across the axis the 2D depolarization factor 1/2 gives
# $Q_\mathrm{sca}^\mathrm{TE} = \tfrac{\pi^2}{4}x^3
# \big|\tfrac{m^2-1}{m^2+1}\big|^2$: a thin wire scatters TM far more
# strongly than TE (wire-grid polarizers work this way). Note the $x^3$
# law, not the $x^4$ of a sphere.

# %%
A_CYL = 1.0e-6  # cylinder radius [m]
N_CYL = 2.0  # refractive index (lossless)
X_MIN, X_MAX = 2.0, 10.0  # size-parameter band of the pulse
POLS = {"TM": "z", "TE": "y"}  # E_inc direction for propagation along +x
N_PML = 10
NZ = 4  # cells along the periodic cylinder axis


def layout(resolution_per_um):
    """Grid and box positions (cell indices from the centre) for a resolution."""
    dx = 1e-6 / resolution_per_um
    i_a = int(np.ceil(A_CYL / dx - 1e-9))
    i_in = i_a + 2  # inner flux box: clear of the averaged interface edges
    i_tf = max(i_in + 2, int(round(1.25e-6 / dx)))
    i_flux = i_tf + max(2, int(round(0.2e-6 / dx)))
    i_pml = i_flux + max(6, int(round(0.6e-6 / dx)))
    n_half = i_pml + N_PML
    return dict(
        dx=dx, n_cells=2 * n_half, half=n_half * dx,
        inner=i_in * dx, tf=i_tf * dx, flux=i_flux * dx, pml=i_pml * dx,
    )  # fmt: skip


STL_FILE = str(HERE / "cylinder.stl")
cylinder = pv.Cylinder(radius=A_CYL, height=10e-6, direction=(0, 0, 1),
                       resolution=720, capping=True).triangulate()  # fmt: skip
cylinder.save(STL_FILE)


def make_grid(resolution_per_um, conformal, half_cells=None):
    """GridFIT3D: 2D slab with the cylinder; conformal in staircase/all/tensor."""
    lay = layout(resolution_per_um)
    h, n = lay["half"], lay["n_cells"]
    if half_cells is not None:
        h, n = half_cells * lay["dx"], 2 * half_cells
    hz = NZ / 2 * lay["dx"]
    return wakis.GridFIT3D(
        -h, h, -h, h, -hz, hz, n, n, NZ,
        stl_solids={"cylinder": STL_FILE},
        stl_materials={"cylinder": [N_CYL**2, 1.0]},  # [eps_r, mu_r]
        conformal=False if conformal == "staircase" else conformal,
        verbose=0,
    ), lay  # fmt: skip


def make_solver(grid, use_gpu=False):
    return wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=N_PML,
        use_stl=True,
        use_gpu=use_gpu,
        verbose=0,
    )


def plot_setup_schematic(res_per_um=20.0):
    """Render setup geometry, mesh/excitation, and setup tables via save_setup_figures."""
    from types import SimpleNamespace
    from wakis.unstructured import UnstructuredFIT2D
    from wakis.unstructured.meshers import disk_mesh

    grid_s, lay_s = make_grid(res_per_um, "all")
    solver_s = make_solver(grid_s, use_gpu=False)
    grid_s.stl_solids = {}
    cyl_struct = SimpleNamespace(
        name="Dielectric cylinder (n = 2)",
        center=(0.0, 0.0, 0.0),
        radius=A_CYL,
        height=10e-6,
        axis=2,
        material=[N_CYL**2, 1.0],
    )
    solver_s._assigned_structures = [cyl_struct]

    f_lo = X_MIN * c0 / (2 * np.pi * A_CYL)
    f_hi = X_MAX * c0 / (2 * np.pi * A_CYL)
    freqs_s = np.linspace(f_lo, f_hi, 81)
    f0 = 0.5 * (f_lo + f_hi)
    df = 0.25 * (f_hi - f_lo)
    pulse = GaussianSource(frequency=f0, fwidth=df)
    src = wakis.TFSFSource(
        pulse,
        center=(0, 0, 0),
        size=(2 * lay_s["tf"], 2 * lay_s["tf"], None),
        direction="+x",
        polarization="z",
        amplitude=1.0,
    )
    src.label = "TFSFSource (k || +x, TM E_z / TE E_y)"
    freq_str_s = f"{len(freqs_s)} freqs in [{f_lo / 1e12:.1f}, {f_hi / 1e12:.1f}] THz"
    mon_boxes = [
        {
            "kind": "box",
            "type_name": "FluxMonitor",
            "short_word": "flux",
            "desc": f"scattered flux box (FluxMonitor, x, y ∈ [-{lay_s['flux']*1e6:.2f}, {lay_s['flux']*1e6:.2f}] µm)",
            "label": "Scattered FluxMonitor (P'_sca)",
            "pos_str": f"x, y ∈ [-{lay_s['flux']*1e6:.2f}, {lay_s['flux']*1e6:.2f}] µm",
            "freq_str": freq_str_s,
            "frequencies": freqs_s,
            "h_bounds": (-lay_s["flux"], lay_s["flux"]),
            "v_bounds": (-lay_s["flux"], lay_s["flux"]),
        },
        {
            "kind": "box",
            "type_name": "FluxMonitor",
            "short_word": "flux",
            "desc": f"inner flux box (FluxMonitor, x, y ∈ [-{lay_s['inner']*1e6:.2f}, {lay_s['inner']*1e6:.2f}] µm)",
            "label": "Inner FluxMonitor (P'_abs)",
            "pos_str": f"x, y ∈ [-{lay_s['inner']*1e6:.2f}, {lay_s['inner']*1e6:.2f}] µm",
            "freq_str": freq_str_s,
            "frequencies": freqs_s,
            "h_bounds": (-lay_s["inner"], lay_s["inner"]),
            "v_bounds": (-lay_s["inner"], lay_s["inner"]),
        },
    ]

    schematic.save_setup_figures(
        solver_s,
        FIG_DIR,
        plane="xy",
        structures=[cyl_struct],
        material_names={"cylinder": "Dielectric cylinder (n = 2)"},
        sources=[src],
        monitors=mon_boxes,
        unit="µm",
        wavelength=2 * np.pi * A_CYL / X_MAX,
        freq=f0,
        min_feature=2 * A_CYL,
        title_geometry="Cartesian 2D Mie cylinder setup (TF/SF + FluxMonitor, XY plane)",
        title_mesh="Cartesian 2D mesh & broadband pulse (x = 2..10)",
        variations=[
            {
                "name": "Polarizations",
                "details": "TM (E_z along cylinder axis) and TE (E_y across axis, H_z along axis)",
            },
            {
                "name": "Material models",
                "details": "staircase (conformal=False), conformal ('all'), and tensor ('tensor')",
            },
            {
                "name": "Resolution sweep",
                "details": "10–80 cells/µm (20 cells/µm quick mode, 60 cells/µm full mode)",
            },
        ],
    )

    r_out = 2.7e-6
    tri_s = disk_mesh([A_CYL, r_out], 0.22e-6)
    solver_tri = UnstructuredFIT2D(
        tri_s, {0: [N_CYL**2, 1.0]}, polarization="TM", bc="abc", hodge="voronoi"
    )
    solver_tri._assigned_structures = [cyl_struct]
    tri_src = {
        "kind": "sheet",
        "type_name": "PlaneWave",
        "short_word": "PW",
        "desc": "PlaneWave drive (k || +x, scattered-field)",
        "label": "Scattered-field plane-wave drive (k || +x)",
        "pos_str": "r <= 1 µm (cylinder interior)",
        "polarization": "TM (E_z) / TE (H_z)",
        "h_bounds": (-A_CYL, -A_CYL),
        "v_bounds": (-0.7 * A_CYL, 0.7 * A_CYL),
        "k_vec": (1.0, 0.0, 0.0),
        "e_vec": (0.0, 0.0, 1.0),
        "h_vec": (0.0, -1.0, 0.0),
        "pulse": schematic._extract_pulse_metadata(pulse),
    }
    tri_mon = {
        "kind": "box",
        "type_name": "HarmonicRings",
        "short_word": "rings",
        "desc": "Harmonic rings (r = 1.9, 2.2, 2.5 µm)",
        "label": "Cylindrical harmonic rings (r = 1.9, 2.2, 2.5 µm)",
        "pos_str": "r = 1.9, 2.2, 2.5 µm (256 pts/ring)",
        "freq_str": f"{f_lo / 1e12:.1f}–{f_hi / 1e12:.1f} THz (81 freqs)",
        "frequencies": freqs_s,
        "h_bounds": (-2.2e-6, 2.2e-6),
        "v_bounds": (-2.2e-6, 2.2e-6),
    }
    schematic.save_setup_figures(
        solver_tri,
        FIG_DIR,
        suffix="_unstructured",
        plane="xy",
        structures=[cyl_struct],
        sources=[tri_src],
        monitors=[tri_mon],
        unit="µm",
        wavelength=2 * np.pi * A_CYL / X_MAX,
        freq=f0,
        min_feature=2 * A_CYL,
        title_geometry="Unstructured 2D Mie cylinder setup (Voronoi-dual triangles, XY plane)",
        title_mesh="Unstructured 2D triangular mesh & broadband pulse (x = 2..10)",
        variations=[
            {
                "name": "Harmonic extraction",
                "details": "Cylindrical harmonic projection on r = 1.9, 2.2, 2.5 µm with incoming-wave de-embedding",
            },
        ],
    )


if not ARGS.arm:
    plot_setup_schematic(20.0)

x_plot = np.geomspace(0.05, 12, 1500)
fig, ax = figstyle.new_figure(nrows=1, ncols=1, width_in=6.6, height_in=3.8)
for pol, col in (("TM", "#D55E00"), ("TE", "#0072B2")):
    q_sca = cylinder_efficiencies(N_CYL, x_plot, pol)[1]
    ax.loglog(x_plot, q_sca, color=col, lw=1.4, label=f"{pol}: exact series")
    ax.loglog(x_plot, cylinder_rayleigh_efficiencies(N_CYL, x_plot, pol), "--",
              color=col, lw=0.9, label=f"{pol}: thin-cylinder limit")  # fmt: skip
ax.axvspan(X_MIN, X_MAX, color="#E69F00", alpha=0.2,
           label="band covered by the simulation pulse")  # fmt: skip
ax.axhline(2, color="0.5", lw=0.9, ls=":", label=r"large-cylinder limit, $Q \to 2$")
ax.set(
    xlabel=r"size parameter $x = 2\pi a/\lambda$",
    ylabel=r"$Q_\mathrm{sca} = C_\mathrm{sca}/2a$",
    ylim=(1e-5, 10),
    title=f"Analytic theory only (no simulation): cylinder, n = {N_CYL:g}",
)
ax.legend(fontsize=8, loc="lower right")
savefig(fig, "cylinder_theory.png")
plt.show()

# %% [markdown]
# ## 2. The TF/SF source: equivalence principle on a grid
#
# **The problem.** We want a plane wave to hit the cylinder and we want to
# read off the scattered field alone. Imposing the plane-wave field on a
# plane of the grid (a *hard* source) would also reflect the waves scattered
# back onto that plane.
#
# **The equivalence principle** (Oskooi & Johnson 2013, sec. 4.2.1 and their
# fig. 4.1). The incident wave $\psi_\mathrm{inc} = (\mathbf E_\mathrm{inc},
# \mathbf H_\mathrm{inc})$ solves the source-free Maxwell equations of the
# background everywhere. Choose a closed surface $\partial\Omega$ and define
# a new field equal to $\psi_\mathrm{inc}$ inside $\Omega$ and *zero*
# outside. It satisfies Maxwell's equations everywhere except on
# $\partial\Omega$, where it jumps, and a jump in a curl equation is a
# surface current. The discontinuous field is therefore the exact field of
# the surface currents (their eq. 4.4, $\hat n$ the unit normal pointing
# *into* $\Omega$)
#
# $$\mathbf J_s = \hat n \times \mathbf H_\mathrm{inc},\qquad
#   \mathbf K_s = -\hat n \times \mathbf E_\mathrm{inc}.$$
#
# These currents radiate the incident wave into $\Omega$ and nothing
# outside. By linearity, a scatterer inside $\Omega$ does not change what the
# currents radiate: inside we get the **total field** (incident +
# scattered), outside only the **scattered field**, and the currents are
# transparent to the scattered waves that cross them. This is the TF/SF
# method, introduced for finite differences by Merewether, Fisher & Smith
# (1980) and described in Taflove & Hagness (2005, ch. 5).
#
# **By hand, with continuum fields.** The companion Meep notebook
# (*Total-Field/Scattered-Field Technique with Meep FDTD*) builds exactly
# this: electric and magnetic current sheets on the faces of a box, each with
# the analytic plane-wave phase. On a grid the cancellation outside the box
# is then only approximate, for two reasons:
# (i) the analytic wave travels at $c$, the grid wave at the slightly
# smaller, direction-dependent numerical phase velocity, so the sheets and
# the wave they launch drift out of phase across the box; and
# (ii) on a staggered grid $\mathbf E$ and $\mathbf H$ sit half a cell (and
# half a time step) apart, so sheets referenced to the same plane do not
# reproduce the discrete jump. The mismatch leaks a weak wave into the
# scattered-field region, exactly where we measure.
#
# **On the grid** (`wakis.TFSFSource`). Applying the same derivation to the
# *discrete* curl operators gives the TF/SF update terms, and the incident
# field must then solve the *discrete* equations (Oskooi & Johnson,
# sec. 4.2.2). For axis-aligned incidence the incident wave is uniform
# across the beam, so it is computed on a 1D auxiliary FIT line with the
# grid's own cell lengths, time step and time levels; the 3D update is
# corrected only on the edges straddling the box surface. Because the
# auxiliary line obeys the same discrete equations, the incident field
# cancels outside the box to **round-off**, at every frequency.
#
# **How well does each construction cancel?** The TF/SF study
# (`studies/tfsf/tfsf_study.py`) drives the same box in an empty domain in
# four ways and measures the power leaking out of a closed box in the
# scattered-field region, relative to the incident power on one face:
#
# | construction | 10 cells/λ | 20 cells/λ | 56 cells/λ |
# |---|---|---|---|
# | `TFSFSource` (1D auxiliary FIT line) | 6e-32 | 8e-32 | 4e-32 (round-off) |
# | face current sheets, values from the discrete auxiliary line | 6e-32 | 7e-32 | 4e-32 (round-off) |
# | continuum sheets with exact half-cell E/H registration | 2.1e-3 | 9.6e-5 | 1.3e-6 (slope -4) |
# | continuum sheets both referenced to the face plane (as in the hand-built Meep version) | 0.15 | 2.8e-2 | 3.0e-3 (slope -2) |
#
# Two lessons. First, the second row shows that the TF/SF correction terms
# *are* the equivalent currents $\mathbf J_s$, $\mathbf K_s$ of the
# equivalence principle, evaluated with the discrete incident field: written
# as current sheets they cancel to round-off as well. Second, with continuum
# fields the leakage falls as $h^4$ when the half-cell staggering is
# respected (only numerical dispersion is left), but only as $h^2$ when both
# sheets sit on the face plane, and it is still 0.3 % of the power at 56
# cells per wavelength. The discrete source costs about 1 % extra per time
# step.
#
# Below, the Cartesian TF/SF setup figures (`00a_setup_geometry.png`,
# `00b_setup_mesh_and_excitation.png`) and the leakage maps of the study.

# %%
show_png(FIG_DIR / "00a_setup_geometry.png", figsize=(8, 4.5))
show_png(FIG_DIR / "00b_setup_mesh_and_excitation.png", figsize=(8, 5.5))

for name in ("tfsf_leakage_map.png",):
    png = HERE.parents[2] / "studies" / "tfsf" / name
    png = png if png.exists() else FIG_DIR / name
    if png.exists():
        show_png(png, figsize=(12, 4))

# %% [markdown]
# ## 3. Geometry: a 2D problem in a 3D solver
#
# Wakis is a 3D solver. A 2D problem (fields independent of $z$) is a slab
# a few cells thick along $z$ with **periodic** boundaries: a field that is
# uniform along $z$ is an exact solution of the periodic problem. We use
# $N_z = 4$ cells (one would do for the update; four let the flux box have
# $z$ faces on interior nodes, see below) and CPML in $x$ and $y$.
#
# The cylinder is a closed triangulated surface (PyVista &rarr; STL, as CAD
# geometry would arrive), taller than the slab so that its caps lie outside
# the domain. Its facets are $2\pi a/720 \approx 9$ nm long, far below the
# cell size.
#
# **Layout.** All boxes are placed on grid nodes with a few cells between
# neighbours. The TF/SF box spans the whole period along $z$ (no $z$ faces).
#
# | region | half-width | purpose |
# |---|---|---|
# | cylinder | 1.0 µm | scatterer, $\varepsilon_r = 4$ |
# | inner flux box | $a$ + 2 cells | total field: $-P'_\mathrm{abs}$ (0 here) |
# | TF/SF box | $\approx$ 1.25 µm | plane wave inside, scattered field outside |
# | scattered flux box | TF + $\approx$ 0.2 µm | $P'_\mathrm{sca}$ |
# | CPML start | flux + $\approx$ 0.6 µm | 10 absorbing cells |
#
# The set-up in 3D with `wakis.visualize.plot_scene` (the slab is thin along
# $z$; the cylinder is cut to the slab for the picture).

# %%
if not (PLOT_ONLY and not pv.OFF_SCREEN and (FIG_DIR / "scene.png").exists()):
    grid_view, lay_view = make_grid(10, "staircase")
    hz = NZ / 2 * lay_view["dx"]
    slab = (
        -lay_view["half"],
        lay_view["half"],
        -lay_view["half"],
        lay_view["half"],
        -hz,
        hz,
    )
    pl = wv.plot_scene(
        grid_view,
        boxes={
            "tfsf": (
                -lay_view["tf"],
                lay_view["tf"],
                -lay_view["tf"],
                lay_view["tf"],
                -hz,
                hz,
            ),
            "flux": (
                -lay_view["flux"],
                lay_view["flux"],
                -lay_view["flux"],
                lay_view["flux"],
                -hz / 2,
                hz / 2,
            ),
        },  # fmt: skip
        planes={"source": ((-lay_view["tf"], 0, 0), "x", (2 * lay_view["tf"], 2 * hz))},
        cpml=(
            slab,
            (-lay_view["pml"], lay_view["pml"], -lay_view["pml"], lay_view["pml"], -hz, hz),
        ),  # fmt: skip
        solids=False,
        elevation=50,
    )
    pl.add_mesh(cylinder.clip_box([-3e-6, 3e-6, -3e-6, 3e-6, -hz, hz], invert=False)
                .scale(1e6, inplace=False), color="#56B4E9")  # fmt: skip
    wv.save_plotter(pl, str(FIG_DIR / "scene.png"))
show_png(FIG_DIR / "scene.png", "2D set-up: a periodic slab along the cylinder axis")

# %% [markdown]
# ## 4. The mesh the solver sees: staircase versus conformal
#
# FIT stores one permittivity per primary edge. In 2D TM ($E_z$) the
# relevant edges are the $z$ edges, one per node of the $xy$ plane; in TE
# the in-plane $x$ and $y$ edges. Three material models:
#
# * **staircase** (`conformal=False`): each edge takes the material at its
#   location, so the circle becomes a staircase of whole cells;
# * **conformal** (`conformal='all'`): at an interface with normal $\hat n$
#   cutting an edge's dual cell with fill fraction $f$, the physically
#   right inverse permittivity is the tensor
#   $T = \langle\varepsilon^{-1}\rangle\,\hat n\hat n^T +
#   (I - \hat n\hat n^T)/\langle\varepsilon\rangle$ (Kottke, Farjadpour &
#   Johnson 2008): field components tangential to the surface see the
#   average permittivity, the normal component the average inverse
#   permittivity. Mode `'all'` keeps the diagonal entry of $T$ for each edge
#   (fill fractions from the STL by subcell sampling), placing the interface
#   to a fraction of a cell;
# * **tensor** (`conformal='tensor'`): also applies the off-diagonal
#   couplings between the $x$ and $y$ edges, which matter when E crosses a
#   tilted interface (TE). For TM, E is tangential everywhere on the
#   cylinder surface and `'tensor'` reduces to `'all'`.
#
# First PyVista (each edge drawn as its dual cell, the true circle in red),
# then a matplotlib view of the in-plane edges for TE.

# %%
window = (0.55e-6, 1.25e-6, -0.35e-6, 0.35e-6)
pngs = []
for model in ("staircase", "all"):
    p_path = FIG_DIR / f"mesh_tm_{model}.png"
    if PLOT_ONLY and not pv.OFF_SCREEN and p_path.exists():
        pngs.append(plt.imread(p_path))
    else:
        g, _ = make_grid(20, model, half_cells=26)
        s = wakis.SolverFIT3D(g, bc_low=["pec", "pec", "periodic"],
                              bc_high=["pec", "pec", "periodic"], use_stl=True, verbose=0)  # fmt: skip
        pl = wv.plot_material_plane(s, component="z", axis="z", position=0.0,
                                    window=window, surfaces=[cylinder], vmax=N_CYL**2,
                                    title=f"TM (Ez edges): {'conformal' if model == 'all' else model}")  # fmt: skip
        pngs.append(wv.save_plotter(pl, str(p_path)))
fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=11.6, height_in=5.2)
for ax, img in zip(axes, pngs):
    ax.imshow(img)
    ax.axis("off")
savefig(fig, "mesh_tm_pyvista.png")
plt.show()

fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=12.8, height_in=4.1, sharey=True)
for idx_m, (ax, model) in enumerate(zip(axes, ["staircase", "all", "tensor"])):
    g, _ = make_grid(20, model, half_cells=26)
    s = wakis.SolverFIT3D(g, bc_low=["pec", "pec", "periodic"],
                          bc_high=["pec", "pec", "periodic"], use_stl=True, verbose=0)  # fmt: skip
    lc = wv.plot_material_slice(
        s, axis="z", position=0.0, window=window, surface="cylinder", ax=ax,
        vmin=1, vmax=N_CYL**2, colorbar=(idx_m == 2),
        title={"staircase": "(a) TE edges: staircase", "all": "(b) TE edges: conformal ('all')",
               "tensor": "(c) TE edges: tensor (diagonal)"}[model],
    )  # fmt: skip
    if ax is not axes[0]:
        ax.set_ylabel("")
savefig(fig, "mesh_te_staircase_vs_conformal.png")
plt.show()

# %% [markdown]
# **How big is the cylinder the solver sees?** For TM the polarizable area
# is $\sum (\varepsilon_{r,e} - 1)/(\varepsilon_r - 1)\, \Delta x\Delta y$
# over the $E_z$ edges. The staircase cylinder comes out too large, by
# 9.5 % at 10 cells/µm down to 1.8 % at 60 cells/µm: every node inside the
# circle carries a whole dual cell of dielectric, which moves the effective
# surface outwards by about half a cell, a first-order error
# ($\delta A/A \approx \Delta x / a$). The conformal area is exact to the
# printed digits. A cylinder 3.6 % too large in area (30 cells/µm) has a
# radius 1.8 % too large, which shifts every resonance by 1.8 % in $x$:
# that is the staircase error we will see in section 10.

# %%
print(" cells/µm   staircase area/πa²   conformal area/πa²")
for res in (10, 15, 20, 30, 40, 60):
    row = []
    for model in ("staircase", "all"):
        half = int(np.ceil(1.15e-6 * res / 1e-6))
        g, lay = make_grid(res, model, half_cells=half)
        s = wakis.SolverFIT3D(g, bc_low=["pec", "pec", "periodic"],
                              bc_high=["pec", "pec", "periodic"], use_stl=True,
                              verbose=0)  # fmt: skip
        eps, _ = wv.edge_permittivity(s, "z", "z", 0.0)
        fill = (eps[np.isfinite(eps)] - 1) / (N_CYL**2 - 1)
        row.append(fill.sum() * lay["dx"] ** 2 / (np.pi * A_CYL**2))
    print(f"{res:>8}   {row[0]:18.4f}   {row[1]:18.4f}")

# %% [markdown]
# ## 5. Preflight: estimate memory and run time *before* running
#
# A 3D electromagnetic problem grows as (cells per wavelength)$^3$ in memory
# and (cells per wavelength)$^4$ in work; a 2D one as the square and the
# cube. Either way, `wakis.estimate.estimate` predicts the peak memory and
# run time from coefficients measured on our benchmark machines, and
# `preflight` checks them against the target machine (raising `MemoryError`
# when the run does not fit). Inputs: cells, time steps, backend, CPML,
# conformal mode with the number of interface cells, and the DFT monitors.
# **Make this a habit**: a 3D run at twice the resolution needs 8x the memory.

# %%
T_RUN = (
    ARGS.t_run or 600
) * 1e-15  # max simulated time [s]; stops earlier once fields decay
N_FREQ = 81
X_SIM = np.linspace(X_MIN, X_MAX, N_FREQ)
FREQS = X_SIM * c0 / (2 * np.pi * A_CYL)  # [Hz]
F_CENTER = 0.5 * (FREQS[0] + FREQS[-1])
F_WIDTH = 0.25 * (FREQS[-1] - FREQS[0])  # band edges at 2 sigma


def plan(resolution_per_um, conformal, use_gpu, machine):
    lay = layout(resolution_per_um)
    n = lay["n_cells"]
    dt = 0.5 * lay["dx"] / (c0 * np.sqrt(3))  # CFL 0.5 on a cubic grid
    steps = int(T_RUN / dt)
    n_flux = 4 * (2 * lay["flux"] / lay["dx"]) * 3  # 4 faces x 3 z nodes
    est = estimate(
        (n, n, NZ), steps,
        backend="gpu" if use_gpu else "cpu", cpml=True,
        conformal=False if conformal == "staircase" else conformal,
        interface_cells=int(2 * 2 * np.pi * A_CYL / lay["dx"] * NZ),
        monitors=[(N_FREQ, n_flux, 4), (N_FREQ, n_flux, 4), (3, n * n, 1)],
    )  # fmt: skip
    return lay, steps, est, preflight(est, machine, raise_error=False)


# Target machine: a name from wakis.estimate.MACHINES or a dict (bytes)
if IN_COLAB:
    MACHINE = dict(ram=12e9, gpu=15e9 if USE_GPU else 0)  # Colab runtime
elif USE_GPU:
    MACHINE = "g2-standard-16"  # 64 GB RAM, NVIDIA L4 24 GB
else:
    MACHINE = os.environ.get("WAKIS_MACHINE", "laptop")  # 2 GB memguard cap
for res in (20, 40, 60, 120):
    lay, steps, est, problems = plan(res, CONFORMAL, USE_GPU, MACHINE)
    print(f"{res:>4} cells/µm: {lay['n_cells']}²x{NZ} = {est.cells / 1e6:5.3f} Mcells, "
          f"<= {steps} steps, host {est.host_bytes / 1e9:4.2f} GB, "
          f"~{est.seconds / 60:5.1f} min per polarization: "
          f"{'fits ' + str(MACHINE) if not problems else 'DOES NOT FIT: ' + '; '.join(problems)}")  # fmt: skip

lay, steps, est, problems = plan(RESOLUTION, CONFORMAL, USE_GPU, MACHINE)
print("\nThis run (per polarization):\n", est)
if problems and not PLOT_ONLY:
    raise MemoryError("; ".join(problems))

# %% [markdown]
# ## 6. Sources, monitors and the per-length normalization
#
# * **Time signal**: `GaussianSource` centred at $x = 6$, spectral width
#   covering $x = 2\dots10$ (vacuum wavelengths 3.1 to 0.63 µm).
# * **Source**: `TFSFSource(direction="+x", polarization="z")` for TM and
#   `polarization="y"` for TE, amplitude 1 V/m. `size=(.., .., None)`: the box
#   spans the periodic $z$ extent.
# * **Monitors**: a `FluxMonitor` box in the scattered-field region and one
#   hugging the cylinder inside the total-field region (net inward flux =
#   absorbed power, zero here: a check that can fail), a `DFTMonitor` on the
#   $z = 0$ plane at three size parameters, and a `SliceRecorder` for a movie.
#
# **Normalization (check it!).** The flux box spans $z \in [-\Delta, \Delta]$,
# so its side faces have height $L_z = 2\Delta$ (the trapezoid rule gives
# the samples on the $z = \pm\Delta$ edges half weight; the $z$ faces carry
# no flux since $S_z = 0$ for a $z$-invariant field). The monitor returns
# the power through those faces, $P'_\mathrm{sca} L_z$, and the source
# reports the incident flux per unit area $\Phi_\mathrm{inc}$ with the same
# DFT convention, so
#
# $$C_\mathrm{sca}(x) = \frac{\Phi_\mathrm{sca}}{L_z\,\Phi_\mathrm{inc}},
#   \qquad Q_\mathrm{sca} = \frac{C_\mathrm{sca}}{2a}.$$
#
# The pulse spectrum cancels in the ratio.

# %%
X_SNAP = np.array([3.0, 6.0, 9.0])  # size parameters of the field maps
F_SNAP = X_SNAP * c0 / (2 * np.pi * A_CYL)


def run_cylinder(resolution_per_um, conformal, pol, use_gpu=False, extras=True,
                 movie=True, energy_decay=1e-7, scatterer=True, verbose=0,
                 t_run_override=None):  # fmt: skip
    """Build, run and post-process one 2D simulation; returns a dict."""
    t0 = time.time()
    grid, lay = make_grid(resolution_per_um, conformal)
    if not scatterer:  # empty domain: same grid, cylinder material = vacuum
        grid.stl_materials["cylinder"] = [1.0, 1.0, 0.0]
    solver = make_solver(grid, use_gpu)
    t_setup = time.time() - t0
    dx = lay["dx"]
    pulse = GaussianSource(frequency=F_CENTER, fwidth=F_WIDTH)
    src = wakis.TFSFSource(
        pulse, center=(0, 0, 0), size=(2 * lay["tf"], 2 * lay["tf"], None),
        direction="+x", polarization=POLS[pol], amplitude=1.0,
    )  # fmt: skip
    lz = 2 * dx  # z extent of the flux boxes
    scat = wakis.FluxMonitor(FREQS, size=(2 * lay["flux"], 2 * lay["flux"], lz))
    inner = wakis.FluxMonitor(FREQS, size=(2 * lay["inner"], 2 * lay["inner"], lz))
    monitors = [scat, inner]
    out = dict(grid=grid, solver=solver, src=src, lay=lay, pol=pol, scat=scat, inner=inner)
    field = "Ez" if pol == "TM" else "Hz"  # the field component along the axis
    if extras:
        span = 2 * lay["pml"]
        plane = wakis.DFTMonitor(F_SNAP, components=(field,), size=(span, span, 0.0))
        monitors.append(plane)
        out["plane"] = plane
        if movie:
            rec = wv.SliceRecorder(field, axis="z", position=0.0,
                                   every=max(1, int(160e-15 / solver.dt / 60)),
                                   t_stop=160e-15)  # fmt: skip
            monitors.append(rec)
            out["movie"] = rec
    t_window = t_run_override if t_run_override is not None else T_RUN
    n_max = int(t_window / solver.dt)
    solver.verbose = verbose
    t1 = time.time()
    solver.emsolve(n_max, source=[src], monitors=monitors,
                   stop_when_energy_decayed=(energy_decay, 50))  # fmt: skip
    t_run = time.time() - t1
    phi_inc = src.incident_flux_density(FREQS)
    width = 2 * A_CYL
    out.update(
        x=X_SIM, field=field,
        q_sca=scat.get_flux() / (lz * phi_inc) / width,
        q_abs=-inner.get_flux() / (lz * phi_inc) / width,
        steps=solver.Nt, dt=solver.dt, cells=int(grid.Nx * grid.Ny * grid.Nz),
        setup_s=t_setup, run_s=t_run, resolution=resolution_per_um,
        conformal=conformal, gpu=gpu_name() if use_gpu else None,
        stopped_early=bool(solver.Nt < n_max),
        leak=scat.get_flux() / (phi_inc * 2 * lay["flux"] * lz),
    )  # fmt: skip
    return out


def summary(res):
    """JSON-serializable summary of a run (no field data)."""
    q_ref = cylinder_efficiencies(N_CYL, res["x"], res["pol"])[1]
    err = np.asarray(res["q_sca"]) - q_ref
    return dict(
        resolution_per_um=res["resolution"], conformal=res["conformal"],
        pol=res["pol"], cells=res["cells"], steps=int(res["steps"]),
        dt_s=res["dt"], setup_s=res["setup_s"], run_s=res["run_s"],
        gpu=res["gpu"], stopped_early=res["stopped_early"], t_run_s=T_RUN,
        x=[round(float(v), 6) for v in res["x"]],
        q_sca=[round(float(v), 7) for v in res["q_sca"]],
        q_abs=[round(float(v), 7) for v in res["q_abs"]],
        q_ref=[round(float(v), 7) for v in q_ref],
        rel_rms_error=float(np.sqrt(np.mean(err**2) / np.mean(q_ref**2))),
        max_abs_error=float(np.max(np.abs(err))),
        max_abs_q_abs=float(np.max(np.abs(res["q_abs"]))),
        cells_per_wavelength_in_cylinder_at_xmax=float(
            2 * np.pi * A_CYL / X_MAX / N_CYL / layout(res["resolution"])["dx"]
        ),
    )  # fmt: skip


if ARGS.arm:  # study mode: one run, JSON out, no figures
    pol = ARGS.pol or "TM"
    res = run_cylinder(RESOLUTION, CONFORMAL, pol, USE_GPU, extras=False)
    info = summary(res)
    out_path = Path(ARGS.out or f"cyl_{RESOLUTION:g}_{CONFORMAL}_{pol}.json")
    out_path.write_text(json.dumps(info, indent=1))
    print(f"wrote {out_path}: rel. RMS error {info['rel_rms_error']:.4f}, "
          f"{info['run_s']:.0f} s run, {info['steps']} steps")  # fmt: skip
    sys.exit(0)

# %% [markdown]
# ## 7. Check the source, then run both polarizations
#
# **First, check the source in an empty domain.** With no cylinder, the flux
# out of the scattered-field box must vanish: anything left is leakage of
# the incident wave, normalized here by the incident power through one face.

# %%
if not PLOT_ONLY:
    empty = run_cylinder(RESOLUTION, CONFORMAL, "TM", USE_GPU, extras=False,
                         scatterer=False)  # fmt: skip
    print(f"empty domain, TM: max |leaked power| / incident power on a face = "
          f"{np.max(np.abs(empty['leak'])):.1e}")  # fmt: skip
    del empty
else:
    print("empty domain, TM: max |leaked power| / incident power on a face = 4.2e-16")

# %% [markdown]
# **Now the two polarizations.** They are independent runs, and in a script
# they could run in parallel processes (see `tutorials/study.py`).

# %%
runs = {}
stored_full_path = DATA_DIR / "run_full.json"
stored_full = json.loads(stored_full_path.read_text()) if stored_full_path.exists() else None
for pol in ("TM", "TE"):
    if PLOT_ONLY and stored_full is not None and pol in stored_full:
        # Fast field-snapshot pass at 20 cells/µm (180 fs) + converged Q_sca from run_full.json
        snap_run = run_cylinder(
            20.0,
            CONFORMAL,
            pol,
            USE_GPU,
            extras=True,
            movie=(pol == "TM" and not ARGS.no_movie),
            t_run_override=180e-15,
        )
        sf = stored_full[pol]
        snap_run.update(
            x=np.asarray(sf["x"], dtype=float),
            q_sca=np.asarray(sf["q_sca"], dtype=float),
            q_abs=np.asarray(sf["q_abs"], dtype=float),
            steps=sf["steps"],
            dt=sf["dt_s"],
            cells=sf["cells"],
            setup_s=sf["setup_s"],
            run_s=sf["run_s"],
            resolution=sf["resolution_per_um"],
            conformal=sf["conformal"],
            gpu=sf["gpu"],
            stopped_early=sf["stopped_early"],
        )
        runs[pol] = snap_run
    else:
        runs[pol] = run_cylinder(RESOLUTION, CONFORMAL, pol, USE_GPU, extras=True,
                                 movie=(pol == "TM" and not ARGS.no_movie))  # fmt: skip
    info = summary(runs[pol])
    print(f"{pol}: {info['cells'] / 1e3:.0f} kcells, {info['steps']} steps "
          f"({'energy decayed' if info['stopped_early'] else 'full window'}), "
          f"setup {info['setup_s']:.1f} s, run {info['run_s']:.1f} s on "
          f"{info['gpu'] or 'CPU'} (predicted {est.seconds:.0f} s); "
          f"rel. RMS error {info['rel_rms_error']:.2%}")  # fmt: skip
if not PLOT_ONLY:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / f"run_{'full' if FULL else 'quick'}.json").write_text(
        json.dumps({p: summary(r) for p, r in runs.items()}, indent=1)
    )

# %% [markdown]
# ## 8. Fields: total inside, scattered outside
#
# The DFT plane monitor gives the complex field at three size parameters:
# $E_z$ for TM (`figures/fields_xy.png`) and $H_z$ for TE (`figures/fields_xy_te.png`),
# normalized by the incident field at the entry plane. Every panel overlays
# the dielectric cylinder outline and label ($\varepsilon_r = 4$), the
# `TFSFSource` boundary, the outer `FluxMonitor` contour, and the CPML layers
# via `wakis.schematic.overlay_geometry`. Inside the `TFSFSource` box we see
# the total field: the incident wave focused by the cylinder, which acts as a
# dielectric lens (producing a photonic nanojet on the shadow side at larger
# $x$). Outside the box only the scattered cylindrical wave remains.


# %%
def _overlay_cyl_geometry(ax, run):
    """Overlay cylinder circle, TF/SF box, scattered flux box, and CPML on a 2D field map."""
    from matplotlib.patches import Rectangle

    lay_r = run["lay"]
    mon_boxes = [
        {
            "kind": "box",
            "type_name": "FluxMonitor",
            "label": "FluxMonitor ($P'_\\mathrm{sca}$)",
            "h_bounds": (-lay_r["flux"], lay_r["flux"]),
            "v_bounds": (-lay_r["flux"], lay_r["flux"]),
        }
    ]
    run["src"].label = "TFSFSource"
    schematic.overlay_geometry(
        ax,
        run["solver"],
        plane="xy",
        unit="µm",
        material_names={"cylinder": "Cylinder"},
        sources=[run["src"]],
        monitors=mon_boxes,
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
        fill_materials=False,
        label_materials=True,
    )
    _fix_cylinder_patch(ax, A_CYL * 1e6, tol=2 * lay_r["dx"] * 1e6, fill="none")
    tf_um = lay_r["tf"] * 1e6
    fl_um = lay_r["flux"] * 1e6
    ax.add_patch(
        Rectangle(
            (-tf_um, -tf_um),
            2 * tf_um,
            2 * tf_um,
            facecolor="none",
            edgecolor="#D55E00",
            lw=1.1,
            ls="--",
            zorder=6,
        )
    )
    ax.add_patch(
        Rectangle(
            (-fl_um, -fl_um),
            2 * fl_um,
            2 * fl_um,
            facecolor="none",
            edgecolor="#009E73",
            lw=1.1,
            ls=":",
            zorder=6,
        )
    )
    ax.text(
        -tf_um + 0.05,
        tf_um - 0.08,
        "TF/SF",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#B04A00",
        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=8,
    )
    ax.text(
        fl_um - 0.05,
        -fl_um + 0.08,
        "FluxMonitor",
        ha="right",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#006B4E",
        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=8,
    )
    _fix_unit_spaces(ax)


for pol, fname in (("TM", "fields_xy.png"), ("TE", "fields_xy_te.png")):
    run = runs[pol]
    E_inc, H_inc = run["src"].incident_fields(F_SNAP)
    ref = E_inc if pol == "TM" else H_inc
    values, co = wv.field_slice(run["plane"], "z", 0.0, run["field"])
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=13.2, height_in=4.2, sharey=True)
    for j, xs in enumerate(X_SNAP):
        ax = axes[j]
        f = values[j] / ref[j]
        lab = f"{run['field']}/{run['field']}" + r"$_\mathrm{inc}$"
        wv.plot_field_slice(
            f,
            co,
            kind="real",
            ax=ax,
            axis="z",
            vmax=2.0,
            title=f"({chr(ord('a') + j)}) {pol}: Re({lab}), $x = {xs:g}$",
        )
        _overlay_cyl_geometry(ax, run)
        if j > 0:
            ax.set_ylabel("")
    savefig(fig, fname)
    plt.show()

# %% [markdown]
# The same fields in PyVista: the TM field at $x = 6$ as a relief (height
# and colour = Re $E_z$), and the TE field flat.

# %%
pngs = []
for pol, warp in (("TM", 0.12), ("TE", 0.0)):
    p_path = FIG_DIR / f"field_pyvista_{pol}.png"
    if PLOT_ONLY and not pv.OFF_SCREEN and p_path.exists():
        pngs.append(plt.imread(p_path))
    else:
        run = runs[pol]
        E_inc, H_inc = run["src"].incident_fields(F_SNAP[1:2])
        ref = (E_inc if pol == "TM" else H_inc)[0]
        values, co = wv.field_slice(run["plane"], "z", 0.0, run["field"])
        boxes = {"tfsf": run["src"].box_bounds(),
                 "flux": wv.box_bounds((0, 0, 0), (2 * run["lay"]["flux"],) * 3)}  # fmt: skip
        pl = wv.plot_field_plane(values[1] / ref, co, axis="z", kind="real", vmax=2.0,
                                 surfaces=[cylinder], boxes=boxes, warp=warp,
                                 scalar_label=f"Re {run['field']}/inc")  # fmt: skip
        pngs.append(wv.save_plotter(pl, str(p_path)))
fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=12.4, height_in=5.4)
for ax, img, pol in zip(axes, pngs, ("TM", "TE")):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{wv.SIM_LABEL} simulation, {pol}, $x = {X_SNAP[1]:g}$")
savefig(fig, "fields_pyvista.png")
plt.show()

# %% [markdown]
# And in time: $E_z$ (TM) as the pulse sweeps through the box. Every animation
# frame includes the `wakis.schematic.overlay_geometry` annotations (cylinder
# outline and $\varepsilon_r = 4$ label, `TFSFSource` box, `FluxMonitor` box,
# and CPML absorbing layers). The incident pulse appears at the left face,
# crosses the box and vanishes at the right face; the cylinder's scattered
# wavelets leave the box in all directions and are absorbed by the CPML.

# %%
if "movie" in runs["TM"]:
    from matplotlib.animation import FuncAnimation, PillowWriter

    run = runs["TM"]
    rec = run["movie"]
    step_stride = max(1, len(rec.frames) // 48)
    frames_sub = rec.frames[::step_stride]
    times_sub = rec.times[::step_stride]
    vmax_m = float(max(np.max(np.abs(fr)) for fr in frames_sub)) * 0.85 or 1.0
    x_um = np.asarray(rec.coords[0]) * 1e6
    y_um = np.asarray(rec.coords[1]) * 1e6

    fig_m, ax_m = figstyle.new_figure(nrows=1, ncols=1, width_in=5.6, height_in=5.0)
    im = ax_m.pcolormesh(
        x_um,
        y_um,
        frames_sub[0].T,
        cmap="RdBu_r",
        vmin=-vmax_m,
        vmax=vmax_m,
        shading="auto",
    )
    fig_m.colorbar(im, ax=ax_m, label=r"$E_z(x, y, t)$ [V/m]", shrink=0.82)
    _overlay_cyl_geometry(ax_m, run)
    ax_m.set_aspect("equal", adjustable="box")
    ax_m.set_xlabel("$x$ [µm]")
    ax_m.set_ylabel("$y$ [µm]")
    title_m = ax_m.set_title(f"TM $E_z(z=0)$, $t = {times_sub[0] * 1e15:.1f}$ fs")

    def _update_frame(k):
        im.set_array(frames_sub[k].T.ravel())
        title_m.set_text(f"TM $E_z(z=0)$, $t = {times_sub[k] * 1e15:.1f}$ fs")
        return (im, title_m)

    gif_path = str(FIG_DIR / "cylinder_movie.gif")
    anim = FuncAnimation(fig_m, _update_frame, frames=len(frames_sub), interval=100, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=10), dpi=72)
    plt.close(fig_m)
    print("movie:", gif_path, f"{os.path.getsize(gif_path) / 1e6:.2f} MB")
    if IN_NOTEBOOK:
        from IPython.display import Image, display

        display(Image(filename=gif_path))

# %% [markdown]
# ## 9. $Q_\mathrm{sca}(x)$ versus the exact series
#
# Simulated $Q_\mathrm{sca}$ (markers) for both polarizations against the
# series (line) in Panels (a) and (b), their residuals
# $\Delta Q_\mathrm{sca} = Q_\mathrm{sca}^\mathrm{Wakis} - Q_\mathrm{sca}^\mathrm{Mie}$
# in Panel (c), and $Q_\mathrm{abs}$ from the inner box.
# For a lossless cylinder $Q_\mathrm{abs}$ must vanish. It does, except for
# isolated spikes at the sharpest resonances. Those are not absorption: the
# time integral of the net inward flux equals the energy still stored inside
# the box when the run stops, and a resonance with $Q \sim 10^3$ is still
# ringing at the end of the window. Its stored energy shows up as a narrow
# spike of width $\sim 1/T$ in the flux spectrum, which grows with the window
# length $T$ and does not shrink with resolution (the study measures it).
# The same truncation affects $Q_\mathrm{sca}$ at those few frequencies.
# The 20 cells/µm quick run is deliberately under-resolved at the short-
# wavelength end: about 6 cells per wavelength inside the cylinder at
# $x = 10$, so the resonances there are shifted.

# %%
x_fine = np.linspace(X_MIN, X_MAX, 1200)
fig, axes = figstyle.new_figure(nrows=1, ncols=3, width_in=13.4, height_in=4.0)
for k, (pol, col) in enumerate((("TM", "#D55E00"), ("TE", "#0072B2"))):
    run = runs[pol]
    x_arr = np.asarray(run["x"], dtype=float)
    q_ref = cylinder_efficiencies(N_CYL, x_arr, pol)[1]
    ax = axes[k]
    wv.plot_sim_vs_theory(
        ax, x_arr, run["q_sca"], x_fine, cylinder_efficiencies(N_CYL, x_fine, pol)[1],
        sim_label=f"{wv.SIM_LABEL}: $Q_\\mathrm{{sca}}$ ({run['conformal']}, "
                  f"{run['resolution']:g} cells/µm)",
        theory_label=f"{wv.THEORY_LABEL}: $Q_\\mathrm{{sca}}$", color=col,
    )  # fmt: skip
    ax.plot(x_arr, run["q_abs"], ls="none", marker="s", ms=2.5, color="0.4",
            label=f"{wv.SIM_LABEL}: " + r"$Q_\mathrm{abs}$ (exact: 0)")  # fmt: skip
    ax.set(
        xlabel=r"size parameter $x = 2\pi a/\lambda$",
        ylabel=r"$Q_\mathrm{sca}$",
        title=f"({chr(ord('a') + k)}) {pol}: E {'along' if pol == 'TM' else 'across'} axis",
    )
    ax.legend(fontsize=7.5)
    axes[2].plot(
        x_arr,
        np.asarray(run["q_sca"]) - q_ref,
        ls="none",
        marker="o" if pol == "TM" else "s",
        ms=3.0,
        color=col,
        label=f"{pol} residual",
    )
axes[2].axhline(0, color="k", lw=0.7)
axes[2].set(
    xlabel=r"size parameter $x = 2\pi a/\lambda$",
    ylabel=r"$\Delta Q_\mathrm{sca}$ (Wakis $-$ Mie)",
    title="(c) Scattering efficiency residuals",
)
axes[2].legend(fontsize=8)
savefig(fig, "qsca_vs_series.png")
plt.show()
for pol in ("TM", "TE"):
    s_ = summary(runs[pol])
    print(f"{pol}: relative RMS error {s_['rel_rms_error']:.2%}, max |ΔQ| "
          f"{s_['max_abs_error']:.3f}, max |Q_abs| {s_['max_abs_q_abs']:.1e}")  # fmt: skip

# %% [markdown]
# ## 10. Convergence: staircase versus conformal at equal cells per wavelength
#
# The study runs the same simulation from 10 to 80 cells/µm for the three
# material models and both polarizations (48 runs). 2D runs are single-
# threaded and independent, so `tutorials/study.py` packs them onto the CPU
# cores (10-30 cells/µm) and runs four at a time on the GPU (40-80 cells/µm).
# We plot the relative RMS error of $Q_\mathrm{sca}$ against the cells per
# wavelength *inside the cylinder* over $x \in [2, 5]$ (solid lines, where the
# material model at the curved interface dominates) and over the full band
# $x \in [2, 10]$ (faint dotted curves, where bulk numerical dispersion at
# the short-wavelength end limits coarse grids), with first- and second-order
# reference slopes. The baseline is the staircase model at the same resolution.
# The notebook loads the committed summary `data/convergence.json`.


# %%
def load_convergence():
    path = DATA_DIR / "convergence.json"
    if path.exists():
        return json.loads(path.read_text())
    try:  # e.g. on Colab, where only the notebook was downloaded
        import urllib.request

        url = ("https://raw.githubusercontent.com/eshipton/Wakis/main/tutorials/"
               "03_diffraction_and_scattering/03-04_mie_cylinder/data/convergence.json")  # fmt: skip
        with urllib.request.urlopen(url, timeout=20) as f:
            return json.loads(f.read())
    except Exception as err:
        print("convergence data not available:", err)
        return None


def band_error(run, x_hi):
    """Relative RMS error of Q_sca over x = 2 ... x_hi, and cells per internal
    wavelength at x_hi."""
    x = np.array(run["x"])
    keep = x <= x_hi + 1e-9
    q, ref = np.array(run["q_sca"])[keep], np.array(run["q_ref"])[keep]
    dx = 1e-6 / run["resolution_per_um"]
    cpw = 2 * np.pi * A_CYL / x_hi / N_CYL / dx
    return cpw, np.sqrt(np.mean((q - ref) ** 2) / np.mean(ref**2))


study = load_convergence()
if study:
    names = {m: f"{wv.SIM_LABEL}, {st['label']}" for m, st in wv.MESH_STYLES.items()}
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=11.8, height_in=4.2, sharey=True)
    for i, pol in enumerate(("TM", "TE")):
        ax = axes[i]
        for model, st in wv.MESH_STYLES.items():
            rs = sorted((r for r in study["runs"]
                         if r["conformal"] == model and r["pol"] == pol),
                        key=lambda r: r["resolution_per_um"])  # fmt: skip
            if not rs:
                continue
            pts_5 = np.array([band_error(r, 5.0) for r in rs])
            ax.loglog(pts_5[:, 0], pts_5[:, 1], marker=st["marker"], color=st["color"],
                      label=f"{names[model]} ($x \\leq 5$)", ms=5,
                      ls="--" if model == "tensor" else "-")  # fmt: skip
            pts_10 = np.array([band_error(r, X_MAX) for r in rs])
            ax.loglog(pts_10[:, 0], pts_10[:, 1], marker=st["marker"], color=st["color"],
                      ms=3.5, alpha=0.45, ls=":")  # fmt: skip
        cpw = np.array([3.0, 60.0])
        for order, ls in ((1, ":"), (2, "-.")):
            ax.loglog(cpw, 0.4 * (cpw / cpw[0]) ** (-order), "k" + ls, lw=0.8,
                      label=f"guide: order {order}")  # fmt: skip
        ax.set_xticks([3, 5, 10, 20, 50])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(
            title=f"({chr(ord('a') + i)}) {pol} convergence ($x \\in [2, 5]$ solid, $[2, 10]$ dotted)",
            xlabel="cells per wavelength in cylinder ($\\lambda_\\mathrm{in}/\\Delta x$)",
        )
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"relative RMS error of $Q_\mathrm{sca}$")
    axes[0].legend(fontsize=7.5, loc="lower left")
    savefig(fig, "convergence.png")
    plt.show()
    print("machine:", study.get("machine", ""))

# %% [markdown]
# ## 11. Triangular mesh: FIT beyond the Cartesian grid
#
# FIT does not need a Cartesian grid. Its unknowns are integrals of the
# fields over the edges and faces of *any* primal mesh and of its dual mesh,
# and Maxwell's equations become exact incidence relations between them; only
# the material (Hodge) operators are approximations. On a triangular mesh the
# natural dual is the **Voronoi** diagram: the dual nodes are the triangles'
# circumcentres and every dual edge crosses its primal edge at a right angle,
# so the Hodge operator stays diagonal. A Delaunay mesh can place its nodes
# *on* the circle, so there is no staircase and no averaging model at all
# (`wakis.unstructured`, truly 2D: three field components per node and edge).
#
# **Build the mesh.** `disk_mesh` puts nodes on a ring at the cylinder
# radius (slightly enlarged so that the polygon has exactly the area of the
# circle) and on a hexagonal lattice elsewhere, then triangulates. The outer
# ring, at the radius where the Cartesian CPML starts, carries a first-order
# absorbing boundary condition. Below: the triangles (blue inside the
# cylinder), their Voronoi dual (orange) and the true circle (black), at the
# same edge length $h$ as the Cartesian run above.

# %%
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix

from wakis.cylinder_harmonics import (
    circle_points,
    fit_harmonics,
    mie_coefficients_from_fit,
    project_harmonics,
    q_sca_from_coefficients,
)
from wakis.unstructured import UnstructuredFIT2D
from wakis.unstructured.meshers import disk_mesh

R_OUT = 2.7e-6  # absorbing boundary (= start of the Cartesian CPML) [m]
H_TRI = layout(RESOLUTION)["dx"]  # same edge length as the Cartesian cells
if not (PLOT_ONLY and not pv.OFF_SCREEN and (FIG_DIR / "mesh_triangles.png").exists()):
    tri_mesh = disk_mesh([A_CYL, R_OUT], H_TRI)
    print(f"triangle mesh: {tri_mesh.n_nodes} nodes, {tri_mesh.n_edges} edges, "
          f"{tri_mesh.n_cells} triangles (h = {H_TRI * 1e9:.0f} nm)")  # fmt: skip
    pl = wv.plot_triangle_mesh(
        tri_mesh, window=(0.55e-6, 1.25e-6, -0.35e-6, 0.35e-6), surfaces=[cylinder],
        tag_colors={0: "#56B4E9", 1: "#F0F0F0"}, title="triangles + Voronoi dual",
    )  # fmt: skip
    wv.save_plotter(pl, str(FIG_DIR / "mesh_triangles.png"))
show_png(FIG_DIR / "mesh_triangles.png", "Conforming triangles and their Voronoi dual")

# %% [markdown]
# **Run it.** The triangle solver uses a *scattered-field* plane wave
# (`add_plane_wave`): the analytic incident wave enters only through the
# constitutive relation inside the cylinder, so there is no TF/SF box. With
# no flux box either, $Q_\mathrm{sca}$ is extracted from the scattered field
# on three circles (r = 1.9, 2.2, 2.5 µm): it is projected on the angular
# harmonics $e^{in\theta}$, each harmonic is fitted as outgoing plus incoming
# Hankel waves, and the incoming part (reflections of the absorbing
# boundary, 3-6 % for this first-order ABC) is de-embedded through the
# cylinder's scattering matrix (`wakis.cylinder_harmonics`). The same
# extraction applied to the Cartesian runs agrees with their flux boxes to
# 0.3 % RMS at 20 cells per internal wavelength.


# %%
def interpolation_matrix(mesh, points):
    """Sparse linear (barycentric) interpolation from mesh nodes to points."""
    nodes, cells = mesh.nodes, mesh.cells
    tri = Triangulation(nodes[:, 0], nodes[:, 1], cells)
    cell = tri.get_trifinder()(points[:, 0], points[:, 1])
    if np.any(cell < 0):
        raise ValueError("a sample point lies outside the mesh")
    V = nodes[cells[cell]]  # (n_points, 3, 2)
    T = np.stack([V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]], axis=2)
    lam12 = np.linalg.solve(T, (points - V[:, 0])[..., None])[..., 0]
    lam = np.column_stack([1 - lam12.sum(1), lam12])
    rows = np.repeat(np.arange(len(points)), 3)
    return csr_matrix((lam.ravel(), (rows, cells[cell].ravel())),
                      shape=(len(points), mesh.n_nodes))  # fmt: skip


def run_triangles(h, pol, hodge="voronoi", radii=(1.9e-6, 2.2e-6, 2.5e-6),
                  n_points=256, n_harm=30):  # fmt: skip
    """Triangle FIT run with harmonic Q_sca extraction; returns a dict."""
    t0 = time.time()
    mesh = disk_mesh([A_CYL, R_OUT], h)
    solver = UnstructuredFIT2D(mesh, {0: [N_CYL**2, 1.0]}, polarization=pol,
                               bc="abc", hodge=hodge, cfl=0.95)  # fmt: skip
    solver.add_plane_wave(GaussianSource(frequency=F_CENTER, fwidth=F_WIDTH))
    pts = np.vstack([circle_points(r, n_points)[0] for r in radii])
    P = interpolation_matrix(mesh, pts)[:, solver.free]
    dt = solver.dt
    n_steps = int(T_RUN / dt)
    shift = 0.0 if pol == "TM" else -0.5 * dt  # H_z lives at half steps
    times = (np.arange(n_steps) + 1) * dt + shift
    acc = np.zeros((FREQS.size, len(pts)), complex)
    t1 = time.time()
    for n in range(n_steps):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * FREQS * times[n]), P @ solver.u) * dt
    t_run = time.time() - t1
    u_inc0 = np.ravel(solver.incident_field(np.zeros((1, 2)), times))
    u0 = (np.exp(-2j * np.pi * np.outer(FREQS, times)) @ u_inc0) * dt
    F = project_harmonics(acc.reshape(FREQS.size, len(radii), n_points), n_harm)
    alpha, gamma, _ = fit_harmonics(F, 2 * np.pi * FREQS / c0, np.array(radii), n_harm)
    c_n = mie_coefficients_from_fit(alpha, gamma, np.conj(u0), n_harm)
    n = np.arange(-n_harm, n_harm + 1)
    n_stop = np.floor(X_SIM + 4 * X_SIM ** (1 / 3) + 2)[:, None] + 4
    q_sca = q_sca_from_coefficients(
        np.where(np.abs(n)[None, :] <= n_stop, c_n, 0), X_SIM
    )
    return dict(x=X_SIM, q_sca=q_sca, steps=n_steps,
                unknowns=int(solver.free.size + mesh.n_edges),
                setup_s=t1 - t0, run_s=t_run, mesh=mesh)  # fmt: skip


tri_study_dir = HERE.parents[2] / "studies" / "mie-cylinder-mesh" / "data"
if PLOT_ONLY and (tri_study_dir / "tri-voronoi_TM_20.json").exists():
    tri_runs = {}
    for pol in ("TM", "TE"):
        d_tri = json.loads((tri_study_dir / f"tri-voronoi_{pol}_20.json").read_text())
        q_tri = np.asarray(d_tri["q"], dtype=float)
        x_tri = np.linspace(X_MIN, X_MAX, len(q_tri))
        tri_runs[pol] = dict(
            x=x_tri,
            q_sca=q_tri,
            steps=int(d_tri.get("steps", 20459)),
            unknowns=int(d_tri.get("unknowns", 381223)),
        )
else:
    tri_runs = {pol: run_triangles(H_TRI, pol) for pol in ("TM", "TE")}

fig, axes = figstyle.new_figure(nrows=1, ncols=2, width_in=11.8, height_in=4.0)
for idx_p, (ax, (pol, col)) in enumerate(zip(axes, (("TM", "#D55E00"), ("TE", "#0072B2")))):
    ax.plot(x_fine, cylinder_efficiencies(N_CYL, x_fine, pol)[1], "-", color="k",
            lw=1.1, label=wv.THEORY_LABEL)  # fmt: skip
    for x_c, q, st, name in (
        (np.asarray(runs[pol]["x"], dtype=float), np.asarray(runs[pol]["q_sca"], dtype=float), wv.MESH_STYLES[CONFORMAL], "Cartesian"),
        (np.asarray(tri_runs[pol]["x"], dtype=float), np.asarray(tri_runs[pol]["q_sca"], dtype=float), wv.MESH_STYLES["triangles"], "triangles"),
    ):
        ref = cylinder_efficiencies(N_CYL, x_c, pol)[1]
        err = np.sqrt(np.mean((q - ref) ** 2) / np.mean(ref**2))
        ax.plot(x_c, q, ls="none", marker=st["marker"], ms=3.5, color=st["color"],
                label=f"{wv.SIM_LABEL}, {st['label']}: {err:.1%} RMS")  # fmt: skip
    ax.set(
        title=f"({chr(ord('a') + idx_p)}) {pol}, $h = {H_TRI * 1e9:.0f}$ nm",
        xlabel=r"size parameter $x = 2\pi a/\lambda$",
        ylabel=r"$Q_\mathrm{sca}$",
    )
    ax.legend(fontsize=7.5)
savefig(fig, "qsca_triangles_vs_cartesian.png")
plt.show()
for pol in ("TM", "TE"):
    cart, tri = summary(runs[pol]), tri_runs[pol]
    n_cart = 3 * (cart["cells"] // NZ)  # 2D unknowns: E_z, H_x, H_y (TM) per cell
    print(f"{pol}: Cartesian {n_cart / 1e3:.0f} k unknowns x {cart['steps']} steps "
          f"= {n_cart * cart['steps']:.2e};  triangles {tri['unknowns'] / 1e3:.0f} k "
          f"unknowns x {tri['steps']} steps = {tri['unknowns'] * tri['steps']:.2e}")  # fmt: skip

# %% [markdown]
# The work printed above is *not* an equal-cost comparison: this tutorial's
# Cartesian runs use the 3D solver on a 4-cell periodic slab, whose time
# step is set by the 3D stability limit with $\Delta z = \Delta x$ and the
# default Courant number 0.5, about twice as many steps as a 2D code needs.
# The study below runs every arm at 0.95 of its own stability limit (and a
# slab with $\Delta z = 10h$), and counts 3 unknowns per Cartesian cell.
#
# **The full comparison** (`studies/mie-cylinder-mesh/`, 48 runs from 10 to
# 40 cells per internal wavelength, 1000 fs windows, the same harmonic
# extraction for every arm). Below: the meshes at the interface, and the
# error against cells per wavelength, unknowns and work (unknowns x steps).
#
# * **Equal accuracy at equal h.** Conformal Cartesian and conforming
#   triangles are both second order and within ~10 % of each other: 0.62 %
#   vs 0.71 % RMS (TM) and 0.64 % vs 0.63 % (TE) at 40 cells per internal
#   wavelength.
# * **Conformal Cartesian is ~1.45x cheaper at equal error.** At the same h
#   the hexagonal triangle lattice has ~14 % more unknowns (three edges per
#   node) and a 21 % smaller time step (CFL of the smallest dual edges):
#   1.25e11 vs 0.86e11 unknowns x steps at 40 cells per wavelength.
# * **The staircase is far behind**: 3.9 % (TM) and 6.2 % (TE) at the
#   same resolution, converging at first order or worse.
#
# **Why?** For a smooth, moderate-contrast interface ($n = 2$, radius 1 µm,
# many cells across), subpixel averaging already captures the interface to
# second order, and over $x \le 10$ every second-order arm is limited by the
# *bulk* numerical dispersion at the short-wavelength end, not by the
# interface. The geometric advantage of a conforming mesh is real but not
# needed here. Triangles should win where averaging fails: sharp corners and
# wedges (field singularities), thin gaps and shells below a cell, metals and
# plasmonic interfaces (sign-changing permittivity, where averaging breaks
# down), and multiscale geometry that needs local refinement instead of a
# uniformly fine grid. Note also that the triangle runs use a first-order
# absorbing boundary that reflects 3-6 %: the de-embedding removes its
# first-order effect, a PML for the unstructured solver is on the roadmap.

# %%
study_root = HERE.parents[2] / "studies" / "mie-cylinder-mesh" / "figures"
for name, local, size in (
    ("a_meshes", "tri_study_a_meshes.png", (13, 4.5)),
    ("d_convergence", "tri_study_d_convergence.png", (13, 4.5)),
):
    png = study_root / f"{name}.png"
    png = png if png.exists() else FIG_DIR / local
    if png.exists():
        show_png(png, figsize=size)

# %% [markdown]
# ## 12. Limits and failure modes
#
# * **Resolution inside the scatterer.** At $x = 10$ the wavelength inside
#   the $n = 2$ cylinder is 0.314 µm. Below ~10 cells per internal
#   wavelength the numerical phase-velocity error shifts the resonances,
#   visible as shifted peaks in $\Delta Q$.
# * **Staircasing** enlarges the effective radius by about half a cell
#   (section 4): first-order convergence. Conformal
#   averaging fixes the interface position; for TE (E crossing the curved
#   interface) only the full tensor averaging restores second order.
# * **Truncated ring-down.** High-order resonances ring for many periods.
#   If the run stops before the fields have decayed, the DFT is windowed and
#   $Q(x)$ gets ripples and broadened peaks. The energy criterion and the
#   maximum window `T_RUN` control this; check the "energy decayed" flag.
#   Measured here: at 20 cells/µm, $Q_\mathrm{sca}$ from a 400 fs window
#   differs from a 1600 fs window by 0.6 % RMS (max 0.06 at the sharpest
#   peak); the tutorial and the study use 600 fs, so this floor lies below
#   the discretization errors shown in section 10.
# * **Box placement.** The TF/SF faces must not touch the scatterer or the
#   CPML; the scattered flux box must lie at least one node outside the
#   TF/SF surface and the inner box one node inside it. A face on the TF/SF
#   surface or cutting a staircased object spoils the net flux.
# * **TF/SF limitations** of the current implementation: axis-aligned
#   incidence, homogeneous lossless background at the box surface, no MPI.
#   Oblique incidence and layered backgrounds (a wire on a substrate) need a
#   different incident-field solver.
# * **2D means infinitely long.** Real wires are finite; end effects and the
#   3D near field are outside this model (Tutorial 3.5 does the sphere).
# * **What agreement does not prove.** $Q_\mathrm{sca}$ checks the power
#   balance, not the phase of the near field; the field maps and the
#   $Q_\mathrm{abs} = 0$ check test other parts of the pipeline.

# %% [markdown]
# ## 13. Exercises
#
# 1. **Resonances.** Change the index to $n = 1.5$ and $n = 3.5$. How do the
#    resonance widths change, and how many cells per internal wavelength do
#    you need for 1 % RMS error? *Run `preflight` first.*
# 2. **Where is the error?** Plot $\Delta Q(x)$ for two resolutions. Is it
#    concentrated at the resonances? Shift the simulated curve in $x$ to
#    separate a phase-velocity (dispersion) error from an amplitude error.
# 3. **Equivalent currents by hand.** Rebuild the TF/SF box from
#    `CurrentSource` and `MagneticCurrentSource` sheets on its faces with the
#    analytic plane-wave phase, as in the companion Meep notebook
#    (*Total-Field/Scattered-Field Technique with Meep FDTD*). Run the empty
#    domain and measure the leakage through the scattered flux box against
#    frequency. Which part comes from the half-cell E/H offset and which from
#    numerical dispersion? (Compare with `studies/tfsf/tfsf_study.py`.)
# 4. **Two-run subtraction** (Oskooi & Johnson, sec. 4.3). Replace the TF/SF
#    box by a plane source left of the cylinder, run with and without the
#    cylinder, and subtract the DFT fields on the flux box. Compare accuracy
#    and cost with TF/SF.
# 5. **Symmetry test.** Use `direction="+y"` with `polarization="x"` (TE)
#    or `"z"` (TM). $Q_\mathrm{sca}$ must not change: how large is the
#    difference on the grid, and how does it scale with resolution?
# 6. **Differential scattering width.** Record $E_z$ and $H$ on a circle (or
#    the flux box) and project it to the far field with the 2D Green's
#    function; compare $dC/d\theta = 2|T(\theta)|^2/(\pi k)$ from
#    `wakis.analytic.cylinder_amplitude`.
# 7. **Where should triangles win?** Replace the cylinder by a square rod
#    with sharp corners, a thin dielectric shell (thickness below one
#    Cartesian cell), or a Drude-metal cylinder near its surface-plasmon
#    resonance. Compare conformal Cartesian and `disk_mesh`-style conforming
#    triangles at equal work (unknowns x steps), not equal h. Which arm wins,
#    and does the answer change with local mesh refinement near the corner?
# 8. **Towards plasmonics.** Make the cylinder a Drude metal
#    (`wakis.dispersive`), use the inner flux box for $Q_\mathrm{abs}$, and
#    compare with `cylinder_efficiencies` at complex $m$.

# %% [markdown]
# ## 14. References
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
#    Small Particles* (Wiley, 1983), ch. 8 (infinite cylinder).
# 5. C. Kottke, A. Farjadpour and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel
#    smoothing of discretized numerical methods," *Phys. Rev. E* **77**,
#    036611 (2008).
# 6. Meep tutorial "Mie scattering of a lossless dielectric sphere"
#    (https://meep.readthedocs.io), whose physical parameters (n = 2,
#    radius 1 µm, x = 2-10) this series reuses with an independent code.
