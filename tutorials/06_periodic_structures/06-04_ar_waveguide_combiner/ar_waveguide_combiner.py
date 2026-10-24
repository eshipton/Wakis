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
# # 6.4 AR Waveguide Combiner: $k$-Space Map and Grating Order Efficiencies
#
# **Level:** Advanced · **Time:** ~45 s (`--quick`, 2 cores) / ~3 min (`--full`)
#
# Diffractive augmented-reality (AR) waveguide combiners relay a micro-display's
# angular field of view (FOV) to the human eye-box via Total Internal Reflection
# (TIR) inside a high-index glass slab ($n_s \in [1.7, 2.0]$). An **in-coupler (IC)**
# surface-relief or metasurface grating diffracts incident display rays beyond the
# glass-air critical angle $\theta_c = \arcsin(1/n_s)$; an **eye-pupil expander (EPE)**
# (fold grating) replicates the beam laterally; and an **out-coupler (OC)** grating
# gradually extracts a small fraction ($\sim 2\text{--}15\%$) at each TIR bounce while
# maintaining high $0\text{th}$-order internal reflection ($R_0 \approx 75\text{--}85\%$)
# to illuminate the entire eye-box uniformly.
#
# In this tutorial we combine analytical $k$-space combiner design with full-wave 2D
# Finite Integration Technique (FIT) simulations using `wakis.floquet` (`FloquetCell`,
# `AngleSweep`, `FloquetPort`, `s_parameters_from_ports`, `grating_angles`, and the exact
# 1D Fourier Modal Method / RCWA reference `rcwa_1d_grating`), reproducing the physical
# mechanisms and design principles of:
# - **Tian et al. (2025)**, *"An achromatic metasurface waveguide for augmented reality displays"*,
#   *Light: Science & Applications* **14**, 94, DOI: [`10.1038/s41377-025-01761-w`](https://doi.org/10.1038/s41377-025-01761-w).
# - **Wang & Ma (2023)**, *"Surface relief grating near-eye display waveguide design"*,
#   `arXiv:2307.01600`, DOI: [`10.48550/arXiv.2307.01600`](https://doi.org/10.48550/arXiv.2307.01600).
# - **Kress & Chatterjee (2021)**, *"Waveguide combiners for mixed reality headsets: a nanophotonics design perspective"*,
#   *Nanophotonics* **10**(1), 41–74, DOI: [`10.1515/nanoph-2020-0410`](https://doi.org/10.1515/nanoph-2020-0410).
#
# ### Learning goals
# 1. Derive the **normalized $k$-space diagram** of a 3-grating AR combiner ($\mathbf{G}_{\text{IC}} + \mathbf{G}_{\text{EPE}} + \mathbf{G}_{\text{OC}} = \mathbf{0}$), the **TIR guidance annulus** $1 \le |\bar{\mathbf{k}}_\parallel| < n_s$, and the index-limited maximum symmetric FOV $\text{FOV}_{\max} = 2\arcsin((n_s - 1)/2)$.
# 2. Understand **chromatic dispersion** in single-layer waveguides and how **Tian et al. (2025)** eliminate angular colour separation across RGB ($\lambda_R = 663\;\text{nm}, \lambda_G = 530\;\text{nm}, \lambda_B = 442\;\text{nm}$) using a shared period $\Lambda = 1900\;\text{nm}$ and diffraction orders $(m_R, m_G, m_B) = (+4, +5, +6)$ satisfying $m_R\lambda_R = m_G\lambda_G = m_B\lambda_B$.
# 3. Run full-wave 2D oblique-incidence **Floquet/Bloch sweeps** (`AngleSweep`) for both the in-coupler (air $\to$ glass) and out-coupler (glass TIR $\to$ air), cross-checking every diffraction order against `rcwa_1d_grating` and verifying passivity ($\sum_m (R_m + T_m) = 1$) and Lorentz reciprocity.
# 4. Model multi-bounce **eye-box pupil expansion** and verify why graduated out-coupler groove depths (**Wang & Ma, 2023**, Table 1: $143\text{--}384\;\text{nm}$) dramatically improve spatial luminance uniformity over a constant-depth grating.
#
# ---
#
# ## Physics & Step-by-Step Derivations
#
# ### 1. Normalized $k$-space, the TIR annulus, and vector closure
# For a monochromatic plane wave at vacuum wavelength $\lambda_0$ ($k_0 = 2\pi/\lambda_0$),
# let $\bar{\mathbf{k}}_\parallel = (k_x/k_0, k_y/k_0)$ denote the normalized in-plane wavevector,
# which is conserved across any planar interface. In medium $j$ with refractive index $n_j$,
# the normal wavevector component is
# $$k_{z,j} = k_0 \sqrt{n_j^2 - |\bar{\mathbf{k}}_\parallel|^2}.$$
# Consequently, in the $(k_x/k_0, k_y/k_0)$ plane:
# - **Radiation into air ($|\bar{\mathbf{k}}_\parallel| < 1$):** Both $k_{z,\text{air}}$ and $k_{z,\text{glass}}$ are real; waves escape into free space.
# - **Guided TIR annulus ($1 \le |\bar{\mathbf{k}}_\parallel| < n_s$):** $k_{z,\text{glass}}$ is real ($\theta_{\text{wg}} \ge \theta_c = \arcsin(1/n_s)$) while $k_{z,\text{air}} = i k_0 \sqrt{|\bar{\mathbf{k}}_\parallel|^2 - 1}$ is purely imaginary (evanescent in air).
# - **Cut-off ($|\bar{\mathbf{k}}_\parallel| \ge n_s$):** Evanescent in both air and the glass slab.
#
# A periodic surface grating with period $\Lambda$ and orientation unit vector $\hat{\mathbf{u}}_g$ shifts $\bar{\mathbf{k}}_\parallel$ by integer multiples of its normalized grating vector $\bar{\mathbf{G}} = (\lambda_0/\Lambda)\hat{\mathbf{u}}_g$:
# $$\bar{\mathbf{k}}_{\parallel,m} = \bar{\mathbf{k}}_{\parallel,\text{in}} + m\,\bar{\mathbf{G}}, \qquad m \in \mathbb{Z}. \tag{1}$$
# For a 1D grating along $x$, $\bar{k}_{x,m} = n_{\text{inc}}\sin\theta_{\text{in}} + m\lambda_0/\Lambda = n_{\text{out}}\sin\theta_m$, which is Eq. (1) of **Tian et al. (2025)**.
# To relay an incident display ray $\bar{\mathbf{k}}_{\parallel,\text{in}}$ to the eye at the exact same angle ($\bar{\mathbf{k}}_{\parallel,\text{out}} = \bar{\mathbf{k}}_{\parallel,\text{in}}$) across an In-Coupler (IC), Eye-Pupil Expander (EPE), and Out-Coupler (OC), the grating vectors must form a closed polygon (**Kress & Chatterjee, 2021**, Fig. 28):
# $$\bar{\mathbf{G}}_{\text{IC}} + \bar{\mathbf{G}}_{\text{EPE}} + \bar{\mathbf{G}}_{\text{OC}} = \mathbf{0}. \tag{2}$$
#
# ### 2. Field-of-view limit of the glass refractive index
# Along the in-coupling axis $x$, every angle in the symmetric display field of view $\theta_{\text{in}} \in [-\frac{1}{2}\text{FOV}_x, +\frac{1}{2}\text{FOV}_x]$ must diffract into the guided TIR interval $[1, n_s\sin\theta_{\text{wg},\max}]$ (**Wang & Ma, 2023**, Eqs. 4–5):
# $$1 \le \sin\theta_{\text{in}} + \frac{|m|\lambda_0}{\Lambda} \le n_s \sin\theta_{\text{wg},\max}. \tag{3}$$
# Subtracting the lower bound at $-\frac{1}{2}\text{FOV}_x$ from the upper bound at $+\frac{1}{2}\text{FOV}_x$ eliminates $|m|\lambda_0/\Lambda$ and yields the maximum symmetric FOV:
# $$2\sin\!\left(\frac{\text{FOV}_{\max}}{2}\right) = n_s\sin\theta_{\text{wg},\max} - 1 \quad\Longrightarrow\quad \text{FOV}_{\max} = 2\arcsin\!\left(\frac{n_s\sin\theta_{\text{wg},\max} - 1}{2}\right), \tag{4}$$
# achieved when the grating period centers the FOV in the TIR band: $|m|\lambda_0/\Lambda = \frac{1}{2}(n_s\sin\theta_{\text{wg},\max} + 1)$ (**Wang & Ma, 2023**, Eq. 6). Evaluating our Eq. (4) formula in the grazing limit ($\theta_{\text{wg},\max}=90^\circ$) gives $\text{FOV}_{\max}(1.90) = 2\arcsin(0.45) = 53.5^\circ$ at $n_s = 1.90$ (the glass index used by **Tian et al., 2025**, who report a practical achromatic FOV of $> 45^\circ$ on p. 3, Fig. 2, matching Eq. 4 with $\theta_{\text{wg},\max} = 78^\circ \Rightarrow 46.1^\circ$), $\text{FOV}_{\max}(1.74) = 2\arcsin(0.37) = 43.4^\circ$ at $n_s = 1.74$ (the N-LAF2 glass index used by **Wang & Ma, 2023**, who report a $40^\circ$ horizontal design FOV on pp. 1, 6), and $\text{FOV}_{\max}(2.05) = 2\arcsin(0.525) = 63.3^\circ$ for representative $\text{SiC}/\text{TiO}_2$-class waveguides ($n_s = 2.05$).
#
# ### 3. Achromatic $(4, 5, 6)$ multi-order momentum matching (Tian et al., 2025)
# In a conventional $1\text{st}$-order grating ($\Lambda_0 = 380\;\text{nm}$, tuned to green $\lambda_G = 530\;\text{nm}$ so $\lambda_G/\Lambda_0 = 1.395$), the normalized momentum shift $\lambda/\Lambda_0$ varies from $1.163$ in the blue ($\lambda_B = 442\;\text{nm}$) to $1.745$ in the red ($\lambda_R = 663\;\text{nm}$), clipping red and blue rays at opposite edges of the TIR band and shrinking the common RGB FOV to $< 12^\circ$.
#
# **Tian et al. (2025)** (p. 3, Eq. 2) notice that the RGB wavelength ratios $\lambda_R : \lambda_G : \lambda_B = 663 : 530 : 442\;\text{nm}$ are almost exactly $1/4 : 1/5 : 1/6$. Choosing a single shared period $\Lambda = 1900\;\text{nm}$ and assigning diffraction orders $(m_R, m_G, m_B) = (+4, +5, +6)$ for the IMC and $(-4, -5, -6)$ for the OMC satisfies
# $$m_R\lambda_R = 4\times 663\;\text{nm} = 2652\;\text{nm}, \quad m_G\lambda_G = 5\times 530\;\text{nm} = 2650\;\text{nm}, \quad m_B\lambda_B = 6\times 442\;\text{nm} = 2652\;\text{nm}, \tag{5}$$
# so all three RGB channels experience the *same* normalized grating vector $|m_c|\lambda_c/\Lambda \approx 1.395$ ($\Delta\theta_{\text{wg}} < 0.08^\circ$ across RGB) and share the full $> 45^\circ$ achromatic FOV in a single waveguide layer!
#
# ### 4. Uniformity metrics and multi-bounce eye-box extraction
# **Tian et al. (2025)** (p. 5, Eq. 3) quantify angular diffraction uniformity across the FOV via
# $$U = 1 - \frac{\eta_{\max} - \eta_{\min}}{\eta_{\max} + \eta_{\min}} = \frac{2\eta_{\min}}{\eta_{\max} + \eta_{\min}} \in [0, 1]. \tag{6}$$
# **Note on supercell design comparison:** Because **Tian et al. (2025)** do not tabulate the sub-period ridge coordinates of their particle-swarm inverse-designed $\Lambda = 1900\;\text{nm}$ metasurface, our 6-ridge $\text{Si}_3\text{N}_4$ supercell (`ACHROMATIC_SUPERCELL_RIDGES`) is a forward reference design rather than a reproduction of their exact groove profile; comparing its single-channel angular uniformity ($U_c = 85.4\text{--}93.7\%$, or joint RGB+angle uniformity $U_{\text{RGB}} = 58.3\%$) against Tian *et al.*'s inverse-designed joint uniformity ($U = 72.8\%$, Fig. 3d, pp. 4–5) is therefore **not a like-for-like comparison**.
# Along the out-coupler, a guided beam with initial power $P_0 = \eta_{\text{IC}}$ undergoes $N$ TIR bounces. At bounce $k \in \{1,\dots,N\}$ with $0\text{th}$-order TIR reflectance $R_{0,k}$ and out-coupling efficiency $T_{\text{out},k}$, the power extracted toward the eye and the remaining guided power are
# $$I_{\text{eye},k} = P_{k-1}\,T_{\text{out},k}, \qquad P_k = P_{k-1}\,R_{0,k}. \tag{7}$$
# If $T_{\text{out}}$ is spatially uniform, $I_{\text{eye},k} = P_0 T_{\text{out}} R_0^{k-1}$ decays exponentially across the eye-box. Graduating the out-coupler groove depth from shallow to deep along the propagation path (**Wang & Ma, 2023**, Table 1: $d_1=143\;\text{nm}\to d_6=384\;\text{nm}$) increases $T_{\text{out},k}$ as $P_{k-1}$ depletes, flattening $I_{\text{eye},k}$.

# %%
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (
    ACHROMATIC_SUPERCELL_RIDGES,
    H_SIN_M,
    N_SIN_TIAN,
    N_WG_TIAN,
    PERIOD_ACHROMATIC_M,
    PERIOD_GREEN_1ST_M,
    RGB_ACHROMATIC_ORDERS,
    RGB_WAVELENGTHS_M,
    WANG_OUTCOUPLER_DEPTHS_NM,
    build_coupler_floquet_cell,
    kspace_combiner_map,
    max_symmetric_fov_deg,
    rgb_kspace_fov_comparison,
    run_task_0_rgb_incoupler,
    run_task_1_achromatic_supercell,
    run_task_2_outcoupler_and_eyebox,
    run_task_3_nearfield_and_resolution,
    supercell_uniformity_summary,
    tian_uniformity,
    tir_critical_angle_rad,
)

# High-contrast paper figure style (white background, sans-serif >= 8.5 pt, color-blind-safe Okabe-Ito palette)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    }
)

PALETTE = {
    "B": "#0072B2",  # Okabe-Ito Blue
    "G": "#009E73",  # Okabe-Ito Bluish Green
    "R": "#D55E00",  # Okabe-Ito Vermilion
    "gold": "#E69F00",  # Okabe-Ito Orange
    "purple": "#CC79A7",  # Okabe-Ito Reddish Purple
    "sky": "#56B4E9",  # Okabe-Ito Sky Blue
    "ink": "#1A1A1A",
    "grey": "#666666",
}

TASKS = {
    0: (
        "kspace_and_rgb_incoupler_sweep",
        "2D k-space combiner map + Wakis vs RCWA RGB in-coupler angular sweeps",
    ),
    1: (
        "achromatic_supercell_incoupler",
        "Tian et al. (2025) Lambda=1900 nm (4, 5, 6) achromatic supercell Wakis vs RCWA",
    ),
    2: (
        "outcoupler_tir_and_eyebox_uniformity",
        "Out-coupler TIR bounce sweep, Floquet reciprocity, and Wang & Ma (2023) eye-box uniformity",
    ),
    3: (
        "nearfield_and_resolution_study",
        "4-level FIT grid convergence study vs RCWA + steady-state near-field phasor maps",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial 6.4: AR Waveguide Combiner k-Space Map and Grating Order Efficiencies"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast, reduced-grid validation (< 1 min).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full-resolution benchmark sweeps.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=sorted(TASKS.keys()),
        default=None,
        help="Run a single task index (0..3) and save its data.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate figures from cached data/ without re-running FDTD.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TUTORIAL_DIR / "data",
        help="Output directory for cached .npz and .json data.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=TUTORIAL_DIR / "figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Number of worker processes for AngleSweep (default: 2).",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS", "").strip()
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    if not args.full:
        args.quick = True
    return args


# %% [markdown]
# ## 1. Combiner Architecture, Unit-Cell Setup, and FIT Mesh (Figure 1)
#
# Our first figure presents:
# - **Panel (a):** The complete diffractive AR waveguide combiner architecture showing the micro-display input cone ($\theta_{\text{in}} \in [-\frac{1}{2}\text{FOV}, +\frac{1}{2}\text{FOV}]$), the $\text{Si}_3\text{N}_4$ In-Coupler ($\mathbf{G}_{\text{IC}}$), the $1\;\text{mm}$ high-index glass slab ($n_s = 1.9$, $\theta_c = \arcsin(1/1.9) = 31.8^\circ$), the Eye-Pupil Expander ($\mathbf{G}_{\text{EPE}}$), and the graduated Out-Coupler ($\mathbf{G}_{\text{OC}}$) delivering replicated beams to the eye-box.
# - **Panel (b):** The 2D `FloquetCell` simulation domain ($x \in [-\Lambda/2, +\Lambda/2]$ with Bloch-periodic boundaries $E(x+\Lambda) = e^{-i k_x \Lambda}E(x)$, $z \in [z_{\min}, z_{\max}]$ terminated by 14-cell CPML absorbers, tilted plane-wave current sheet at $z_{\text{src}}$, reflection/transmission `DiffractionMonitor` planes at $z_{\text{refl}}$ and $z_{\text{tran}}$, and $\text{Si}_3\text{N}_4$ ridge height $h = 160\;\text{nm}$).
# - **Panel (c):** The actual `GridFIT3D` edge-level subpixel-smoothed permittivity map $\varepsilon_y(x, z)$ inside the grating tooth region, overlaid with every Yee grid line showing $\ge 22$ cells per minimum material wavelength and $\ge 8$ cells across the $160\;\text{nm}$ $\text{Si}_3\text{N}_4$ layer.


# %%
def plot_fig1_schematic_and_mesh(fig_dir: Path, quick: bool = True) -> Path:
    """Generate Figure 1: Full AR combiner schematic, FloquetCell setup, and FIT mesh view."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), constrained_layout=True)

    # --- Panel (a): Whole AR Waveguide Combiner Architecture ---
    ax = axes[0]
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 6.2)
    ax.axis("off")
    ax.set_title("(a) AR Waveguide Combiner Architecture (Side + Plan View)")

    # Waveguide glass slab
    slab = Rectangle(
        (0.6, 2.3),
        8.8,
        1.55,
        facecolor="#DCEEFB",
        edgecolor="#0072B2",
        lw=1.6,
        zorder=2,
    )
    ax.add_patch(slab)
    ax.text(
        5.0,
        3.62,
        r"High-index glass waveguide ($n_s = 1.90$, $\varepsilon_r = 3.61$, $\theta_c = 31.8^\circ$)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#084081",
    )
    ax.text(5.0, 4.18, "Air cladding ($n = 1.0$, TIR boundary)", ha="center", fontsize=8.3, color=PALETTE["grey"])
    ax.text(5.0, 1.98, "Air cladding ($n = 1.0$, see-through real-world scene)", ha="center", fontsize=8.3, color=PALETTE["grey"])

    # In-coupler (IC) ridges on bottom/top
    for k in range(5):
        rx = 1.15 + k * 0.28
        ax.add_patch(
            Rectangle((rx, 3.85), 0.16, 0.22, facecolor=PALETTE["R"], edgecolor="#802000", lw=0.9, zorder=4)
        )
    ax.text(
        1.8,
        4.35,
        r"In-Coupler (IC)" + "\n" + r"$\mathrm{Si_3N_4}$ ($n=2.0$, $h=160$ nm)" + "\n" + r"$\mathbf{G}_{\mathrm{IC}} = +\frac{2\pi}{\Lambda}\hat{\mathbf{x}}$",
        ha="center",
        va="bottom",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF5EB", ec=PALETTE["R"], lw=1.0),
    )

    # Graduated Out-coupler (OC) ridges (increasing depth from bounce 1 to 6)
    depths_vis = np.linspace(0.10, 0.26, 6)
    for k, dv in enumerate(depths_vis):
        rx = 4.55 + k * 0.68
        ax.add_patch(
            Rectangle((rx, 3.85), 0.24, dv, facecolor=PALETTE["G"], edgecolor="#005A40", lw=0.9, zorder=4)
        )
    ax.text(
        6.5,
        4.42,
        r"Graduated Out-Coupler (OC, $d_1=143\to d_6=384$ nm)"
        + "\n"
        + r"$\mathbf{G}_{\mathrm{OC}} = -\frac{2\pi}{\Lambda}\hat{\mathbf{y}}$ ($R_0 \approx 80\%$, $T_{-1} \approx 5\mathrm{-}18\%$)",
        ha="center",
        va="bottom",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.22", fc="#E8F8F2", ec=PALETTE["G"], lw=1.0),
    )

    # Micro-display + incident ray cone
    ax.add_patch(Rectangle((1.1, 0.45), 1.4, 0.45, facecolor="#333333", edgecolor="black", lw=1.1))
    ax.text(1.8, 0.67, "RGB Micro-Display", color="white", ha="center", va="center", fontsize=8.0, fontweight="bold")
    for dx_ray, ls_r in [(-0.25, "--"), (0.0, "-"), (0.25, "--")]:
        ax.annotate(
            "",
            xy=(1.8 + dx_ray * 0.4, 3.85),
            xytext=(1.8 + dx_ray, 0.92),
            arrowprops=dict(arrowstyle="->", color=PALETTE["R"], lw=1.4, ls=ls_r),
        )
    ax.text(0.82, 1.45, r"FOV $\pm\theta_{\mathrm{in}}$", fontsize=8.3, color=PALETTE["R"], fontweight="bold")

    # TIR zig-zag trajectory inside slab
    pts_tir = [
        (1.8, 3.85),
        (3.2, 2.30),
        (4.65, 3.85),
        (5.33, 2.30),
        (6.01, 3.85),
        (6.69, 2.30),
        (7.37, 3.85),
        (8.05, 2.30),
        (8.73, 3.85),
    ]
    xs_t, ys_t = zip(*pts_tir)
    ax.plot(xs_t, ys_t, color=PALETTE["B"], lw=2.0, zorder=3, label="Guided TIR mode")
    ax.text(
        3.35,
        2.95,
        r"TIR ($\theta_{\mathrm{wg}} > \theta_c$)" + "\n" + r"$T_0 = 0$ (evanescent)",
        fontsize=8.2,
        color="#084081",
        ha="center",
    )

    # Extracted rays downward to Eye-box
    for bx in [4.65, 6.01, 7.37, 8.73]:
        ax.annotate(
            "",
            xy=(bx, 0.95),
            xytext=(bx, 2.30),
            arrowprops=dict(arrowstyle="->", color=PALETTE["G"], lw=1.6),
        )
    ax.add_patch(
        Rectangle((4.35, 0.35), 4.65, 0.52, facecolor="#FFF8DC", edgecolor=PALETTE["gold"], lw=1.4)
    )
    ax.text(
        6.68,
        0.61,
        r"Expanded Eye-Box ($N$ TIR bounces, $\mathbf{k}_{\mathrm{out}} = \mathbf{k}_{\mathrm{in}}$)",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#5C3D00",
    )

    # --- Panel (b): 2D FloquetCell Unit-Cell Schematic ---
    ax = axes[1]
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.25, 1.35)
    ax.set_xlabel(r"Transverse coordinate $x / \Lambda$")
    ax.set_ylabel(r"Propagation coordinate $z$ [$\mu$m]")
    ax.set_title(r"(b) 2D $\mathrm{FloquetCell}$ Unit-Cell Setup")

    # Background regions: Air below (z < 0), Grating (0..0.16), Glass above (z > 0.16)
    ax.add_patch(Rectangle((-0.5, -0.95), 1.0, 0.95, facecolor="#F7FAFC", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((-0.5, 0.0), 1.0, 1.05, facecolor="#DCEEFB", edgecolor="none", zorder=1))
    # Top/bottom CPML shaded bands
    ax.add_patch(
        Rectangle((-0.5, -1.20), 1.0, 0.25, facecolor="#DDDDDD", edgecolor="#666666", hatch="///", lw=0.8, zorder=2)
    )
    ax.add_patch(
        Rectangle((-0.5, 1.05), 1.0, 0.25, facecolor="#DDDDDD", edgecolor="#666666", hatch="///", lw=0.8, zorder=2)
    )
    ax.text(0.0, -1.08, "Bottom CPML (14 cells, -z)", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.text(0.0, 1.17, "Top CPML (14 cells, +z)", ha="center", va="center", fontsize=8.0, fontweight="bold")

    # Grating tooth centered at x=0 (width f*Lambda)
    ax.add_patch(
        Rectangle((-0.25, 0.0), 0.50, 0.22, facecolor=PALETTE["R"], edgecolor="#802000", lw=1.3, zorder=3)
    )
    ax.text(
        0.0,
        0.11,
        r"$\mathrm{Si_3N_4}$ ($n=2.0$, $h=160$ nm)",
        ha="center",
        va="center",
        color="white",
        fontsize=8.0,
        fontweight="bold",
        zorder=4,
    )

    # Bloch boundaries at x = -0.5 and x = +0.5
    ax.plot([-0.5, -0.5], [-1.20, 1.30], color=PALETTE["purple"], lw=2.2, ls="-.")
    ax.plot([+0.5, +0.5], [-1.20, 1.30], color=PALETTE["purple"], lw=2.2, ls="-.")
    ax.text(
        -0.54,
        0.05,
        r"Bloch $-x$" + "\n" + r"$e^{+i k_x \Lambda/2}$",
        ha="right",
        va="center",
        fontsize=8.0,
        color=PALETTE["purple"],
        fontweight="bold",
    )
    ax.text(
        +0.54,
        0.05,
        r"Bloch $+x$" + "\n" + r"$e^{-i k_x \Lambda/2}$",
        ha="left",
        va="center",
        fontsize=8.0,
        color=PALETTE["purple"],
        fontweight="bold",
    )

    # Source & Monitor planes
    ax.axhline(-0.62, xmin=0.26, xmax=0.74, color=PALETTE["R"], lw=2.0, ls="-")
    ax.text(0.0, -0.73, r"Source $J_y(x,t) = s(t)e^{-ik_x x}$ ($z_{\mathrm{src}}$)", ha="center", fontsize=8.0, color=PALETTE["R"])
    ax.annotate("", xy=(0.12, -0.36), xytext=(-0.05, -0.60), arrowprops=dict(arrowstyle="->", color=PALETTE["R"], lw=1.6))

    ax.axhline(-0.32, xmin=0.26, xmax=0.74, color=PALETTE["B"], lw=1.8, ls="--")
    ax.text(0.0, -0.24, r"Port 1: Refl. $\mathrm{DiffractionMonitor}$ ($z_{\mathrm{refl}}$)", ha="center", fontsize=8.0, color=PALETTE["B"])

    ax.axhline(0.56, xmin=0.26, xmax=0.74, color=PALETTE["G"], lw=1.8, ls="--")
    ax.text(0.0, 0.63, r"Port 2: Tran. $\mathrm{DiffractionMonitor}$ ($z_{\mathrm{tran}}$)", ha="center", fontsize=8.0, color=PALETTE["G"])

    # Diffracted orders arrows in glass (m = -1, 0, +1)
    ax.annotate("", xy=(-0.34, 0.96), xytext=(0.0, 0.22), arrowprops=dict(arrowstyle="->", color=PALETTE["G"], lw=1.7))
    ax.text(-0.34, 0.97, r"$m=-1$ (TIR)", fontsize=8.0, color=PALETTE["G"], ha="center", fontweight="bold")
    ax.annotate("", xy=(0.14, 0.96), xytext=(0.0, 0.22), arrowprops=dict(arrowstyle="->", color="#084081", lw=1.3, ls=":"))
    ax.text(0.20, 0.97, r"$m=0$", fontsize=8.0, color="#084081", ha="center")
    ax.text(0.0, 0.82, r"Glass Waveguide ($\varepsilon_{\mathrm{above}} = n_s^2 = 3.61$)", ha="center", fontsize=8.2, color="#084081")
    ax.text(0.0, -0.90, r"Incident Medium ($\varepsilon_{\mathrm{below}} = 1.00$)", ha="center", fontsize=8.2, color=PALETTE["grey"])

    # --- Panel (c): Actual FIT Edge-Level Permittivity & Yee Mesh View ---
    ax = axes[2]
    wl_g = RGB_WAVELENGTHS_M["G"]
    cell_demo, minfo = build_coupler_floquet_cell(
        period=PERIOD_GREEN_1ST_M,
        height=H_SIN_M,
        wavelength_min=wl_g,
        fill_factor=0.50,
        eps_ridge=N_SIN_TIAN**2,
        eps_groove=1.0,
        eps_below=1.0,
        eps_above=N_WG_TIAN**2,
        cpw=24,
    )
    import wakis
    from scipy.constants import epsilon_0

    grid = wakis.GridFIT3D(
        -0.5 * cell_demo.Lx,
        0.5 * cell_demo.Lx,
        -0.5 * cell_demo.Ly,
        0.5 * cell_demo.Ly,
        cell_demo.zmin,
        cell_demo.zmax,
        cell_demo.nx,
        cell_demo.ny,
        cell_demo.nz,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        n_pml=cell_demo.n_pml,
        dtype=np.complex128,
        verbose=0,
    )
    cell_demo.material_callback(solver)
    eps_y = 1.0 / (np.asarray(solver.ieps[:, 0, :, "y"], dtype=float) * epsilon_0)

    x_nm = grid.x * 1e9
    z_nm = grid.z * 1e9
    z_mask = (z_nm[:-1] >= -120.0) & (z_nm[:-1] <= 280.0)
    iz_idx = np.where(z_mask)[0]
    z_sub_edges = z_nm[iz_idx[0] : iz_idx[-1] + 2]

    pcm = ax.pcolormesh(
        x_nm,
        z_sub_edges,
        eps_y[:, iz_idx].T,
        cmap="YlGnBu",
        vmin=1.0,
        vmax=4.1,
        shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label(r"Subpixel relative permittivity $\varepsilon_y(x, z)$")

    # Overlay actual Yee grid lines
    vlines = [((xv, z_sub_edges[0]), (xv, z_sub_edges[-1])) for xv in x_nm]
    hlines = [((x_nm[0], zv), (x_nm[-1], zv)) for zv in z_sub_edges]
    ax.add_collection(LineCollection(vlines, colors="black", linewidths=0.32, alpha=0.45))
    ax.add_collection(LineCollection(hlines, colors="black", linewidths=0.32, alpha=0.45))

    # Exact material contours
    ax.axhline(0.0, color=PALETTE["R"], lw=1.3, ls="--")
    ax.axhline(H_SIN_M * 1e9, color=PALETTE["R"], lw=1.3, ls="--")
    ax.text(
        0.0,
        80.0,
        f"Grid: {minfo['nx']}x{minfo['nz']} cells\n"
        f"dx={minfo['dx_nm']:.1f} nm, dz={minfo['dz_nm']:.1f} nm\n"
        f"({minfo['cpw_x']:.1f} cells / lam_mat)",
        ha="center",
        va="center",
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="black", alpha=0.88),
    )
    ax.text(0.0, -75.0, r"Air ($\varepsilon_r = 1.00$)", ha="center", va="center", fontsize=8.3, fontweight="bold")
    ax.text(
        0.0,
        225.0,
        r"Glass Waveguide ($\varepsilon_r = 3.61$)",
        ha="center",
        va="center",
        fontsize=8.3,
        color="white",
        fontweight="bold",
    )
    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$z$ [nm] (zoomed around grating tooth)")
    ax.set_title(r"(c) FIT Yee Mesh & Subpixel $\varepsilon_y(x,z)$")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "01_combiner_and_unit_cell_schematic.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


# %% [markdown]
# ## 2. Figure Generation Functions (Figures 2–6)
#
# Each figure adheres to the $\le 3$-panel layout and cross-checks Wakis time-domain
# Floquet simulations against the exact Fourier Modal Method (`rcwa_1d_grating`) and
# the analytical expressions from **Tian et al. (2025)** and **Wang & Ma (2023)**.


# %%
def plot_fig2_kspace_and_fov(data_t0: dict[str, np.ndarray], fig_dir: Path) -> Path:
    """Generate Figure 2: 2D k-space combiner diagram, RGB achromatic alignment, and FOV vs n_s."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)

    # --- Panel (a): 2D k-space Annulus and Grating Vector Closure ---
    ax = axes[0]
    phi = np.linspace(0.0, 2.0 * np.pi, 361)
    # Shade TIR annulus between r=1.0 and r=n_glass=1.9
    ax.fill(
        np.concatenate([1.9 * np.cos(phi), 1.0 * np.cos(phi[::-1])]),
        np.concatenate([1.9 * np.sin(phi), 1.0 * np.sin(phi[::-1])]),
        color="#DCEEFB",
        alpha=0.55,
        label=r"Guided TIR annulus ($1 \leq |\bar{\mathbf{k}}| < n_s=1.9$)",
    )
    ax.plot(np.cos(phi), np.sin(phi), color=PALETTE["ink"], lw=1.5, ls="--", label=r"Air escape circle ($|\bar{\mathbf{k}}|=1$)")
    ax.plot(1.9 * np.cos(phi), 1.9 * np.sin(phi), color=PALETTE["B"], lw=1.8, label=r"Waveguide cutoff ($|\bar{\mathbf{k}}|=n_s=1.9$)")

    k_in = data_t0["k_in"]
    k_ic = data_t0["k_ic"]
    k_epe = data_t0["k_epe"]
    k_oc = data_t0["k_oc"]

    ax.plot(k_in[:, 0], k_in[:, 1], color=PALETTE["ink"], lw=1.8, label=r"Input/Output FOV ($36^\circ\times 24^\circ$)")
    ax.plot(k_ic[:, 0], k_ic[:, 1], color=PALETTE["R"], lw=1.8, label=r"After IC ($+\bar{\mathbf{G}}_{\mathrm{IC}}$)")
    ax.plot(k_epe[:, 0], k_epe[:, 1], color=PALETTE["gold"], lw=1.8, label=r"After EPE ($+\bar{\mathbf{G}}_{\mathrm{EPE}}$)")
    ax.plot(k_oc[:, 0], k_oc[:, 1], color=PALETTE["G"], lw=1.4, ls=":", label=r"After OC ($\bar{\mathbf{k}}_{\mathrm{out}}=\bar{\mathbf{k}}_{\mathrm{in}}$)")

    # Grating vector arrows connecting FOV centers
    g_mag = float(RGB_WAVELENGTHS_M["G"] / PERIOD_GREEN_1ST_M)
    ax.annotate("", xy=(g_mag, 0.0), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color=PALETTE["R"], lw=2.0))
    ax.text(0.68, -0.22, r"$\bar{\mathbf{G}}_{\mathrm{IC}}$", color=PALETTE["R"], fontweight="bold", fontsize=9.0)
    ax.annotate("", xy=(0.0, g_mag), xytext=(g_mag, 0.0), arrowprops=dict(arrowstyle="->", color=PALETTE["gold"], lw=2.0))
    ax.text(0.82, 0.82, r"$\bar{\mathbf{G}}_{\mathrm{EPE}}$", color="#B27800", fontweight="bold", fontsize=9.0)
    ax.annotate("", xy=(0.0, 0.0), xytext=(0.0, g_mag), arrowprops=dict(arrowstyle="->", color=PALETTE["G"], lw=2.0))
    ax.text(-0.36, 0.72, r"$\bar{\mathbf{G}}_{\mathrm{OC}}$", color=PALETTE["G"], fontweight="bold", fontsize=9.0)

    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xlabel(r"Normalized wavevector $k_x / k_0$")
    ax.set_ylabel(r"Normalized wavevector $k_y / k_0$")
    ax.set_title(r"(a) 2D $k$-Space Map & Grating Vector Closure")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=6.9,
        framealpha=0.92,
    )

    # --- Panel (b): Conventional m=+1 vs Tian et al. (2025) (4,5,6) Achromatic k-Space ---
    ax = axes[1]
    th_scan = data_t0["theta_fov_scan_deg"]
    ax.axhspan(1.0, N_WG_TIAN, color="#DCEEFB", alpha=0.50, label="Guided TIR Window $[1.0, 1.9]$")
    ax.axhline(1.0, color=PALETTE["ink"], lw=1.2, ls="--")
    ax.axhline(N_WG_TIAN, color=PALETTE["B"], lw=1.4, ls="-")

    for col, ls_conv, lbl_m in [("B", "--", 6), ("G", "-.", 5), ("R", ":", 4)]:
        ax.plot(
            th_scan,
            data_t0[f"conv_kx_{col}"],
            color=PALETTE[col],
            ls=ls_conv,
            lw=1.5,
            alpha=0.75,
            label=f"{col} conv. $m=1$ ($\\Lambda=380$ nm)",
        )
    # Plot Tian et al. (4, 5, 6) achromatic curves (they overlap almost identically!)
    ax.plot(
        th_scan,
        data_t0["ach_kx_G"],
        color=PALETTE["ink"],
        lw=2.5,
        label=r"Tian et al. $(m_R,m_G,m_B)=(4,5,6)$" + "\n" + r"($\Lambda=1900$ nm, $m_c\lambda_c/\Lambda=1.395$)",
    )
    ax.set_xlim(-28.0, 28.0)
    ax.set_ylim(0.75, 2.15)
    ax.set_xlabel(r"Incident angle in air $\theta_{\mathrm{in}}$ [deg]")
    ax.set_ylabel(r"Guided momentum $\bar{k}_{x,m} = \sin\theta_{\mathrm{in}} + m\lambda/\Lambda$")
    ax.set_title("(b) Chromatic Dispersion vs (4, 5, 6) Achromatic Match")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # --- Panel (c): Maximum Symmetric Field of View vs Waveguide Refractive Index ---
    ax = axes[2]
    n_grid = np.linspace(1.45, 2.15, 141)
    fov_ideal = max_symmetric_fov_deg(n_grid, max_wg_angle_deg=90.0)
    fov_pract = max_symmetric_fov_deg(n_grid, max_wg_angle_deg=78.0)

    ax.plot(
        n_grid,
        fov_ideal,
        color=PALETTE["B"],
        lw=2.2,
        label=r"Our Eq. (4) formula ($\theta_{\mathrm{wg,max}}=90^\circ$)",
    )
    ax.plot(
        n_grid,
        fov_pract,
        color=PALETTE["G"],
        lw=2.0,
        ls="--",
        label=r"Our Eq. (4) formula ($\theta_{\mathrm{wg,max}}=78^\circ$)",
    )

    # Mark our Eq. (4) formula evaluated at benchmark glass indices
    formula_points = [
        ("our formula at $n=1.52$\n(N-BK7)", 1.52, PALETTE["grey"], "o", (1.46, 36.5)),
        ("our formula at $n=1.74$\n(Wang & Ma glass)", 1.74, PALETTE["gold"], "s", (1.53, 50.0)),
        ("our formula at $n=1.90$\n(Tian et al. glass)", 1.90, PALETTE["R"], "^", (1.68, 60.5)),
        ("our formula at $n=2.05$\n(SiC/TiO$_2$)", 2.05, PALETTE["purple"], "D", (1.86, 68.5)),
    ]
    for name_b, n_val, col_b, mk, xy_txt in formula_points:
        fv = float(max_symmetric_fov_deg(n_val, max_wg_angle_deg=90.0))
        ax.plot(n_val, fv, marker=mk, color=col_b, markersize=7.0, zorder=5)
        ax.annotate(
            f"{name_b}: {fv:.1f}$^\\circ$",
            xy=(n_val, fv),
            xytext=xy_txt,
            fontsize=7.0,
            color=col_b,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=col_b, lw=0.85),
        )

    # Also mark the actual design FOVs reported in the papers
    ax.plot(
        [1.74],
        [40.0],
        marker="s",
        mfc="white",
        mec=PALETTE["gold"],
        mew=1.8,
        ms=7.0,
        ls="none",
        zorder=6,
        label=r"Wang & Ma (2023, pp. 1, 6) reported: $40.0^\circ$",
    )
    ax.plot(
        [1.90],
        [45.0],
        marker="^",
        mfc="white",
        mec=PALETTE["R"],
        mew=1.8,
        ms=7.5,
        ls="none",
        zorder=6,
        label=r"Tian et al. (2025, p. 3, Fig. 2) reported: $>45^\circ$",
    )

    ax.set_xlim(1.45, 2.15)
    ax.set_ylim(10.0, 76.0)
    ax.set_xlabel(r"Waveguide refractive index $n_s$")
    ax.set_ylabel(r"Maximum symmetric 1D $\mathrm{FOV}_{\max}$ [deg]")
    ax.set_title(r"(c) Index-Limited FOV: $\mathrm{FOV}_{\max} = 2\arcsin((n_s-1)/2)$")
    ax.legend(loc="lower right", fontsize=6.8, framealpha=0.92)

    out_path = fig_dir / "02_kspace_diagram_and_fov_limits.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig3_incoupler_rgb(
    data_t0: dict[str, np.ndarray],
    data_t1: dict[str, np.ndarray],
    fig_dir: Path,
) -> Path:
    """Generate Figure 3: 2D efficiency map, RGB angular sweeps (Wakis vs RCWA), and Tian (4,5,6) supercell."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), constrained_layout=True)

    # --- Panel (a): 2D Guided Efficiency Map eta(theta_in, lambda) ---
    ax = axes[0]
    th_map = data_t0["th_map_deg"]
    wl_map = data_t0["wl_map_nm"]
    eff_map = data_t0["eff_map_conv"] * 100.0

    cmap_m = plt.cm.viridis.copy()
    cmap_m.set_bad(color="#EAEAEA")
    pcm = ax.pcolormesh(th_map, wl_map, eff_map, cmap=cmap_m, shading="auto", vmin=0.0, vmax=24.0)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"Guided $m=-1$ Efficiency $\eta_{-1}$ [%]")

    # Overlay analytic TIR cutoff boundaries: sin(theta) - lambda/Lambda = -1 and -n_s
    th_dense = np.linspace(-22.0, 22.0, 200)
    wl_tir_air = (1.0 + np.sin(np.deg2rad(th_dense))) * PERIOD_GREEN_1ST_M * 1e9
    wl_tir_sub = (N_WG_TIAN + np.sin(np.deg2rad(th_dense))) * PERIOD_GREEN_1ST_M * 1e9
    ax.plot(th_dense, wl_tir_air, color="white", lw=1.8, ls="--", label=r"Air TIR limit ($|\bar{k}_{x,-1}|=1$)")
    ax.plot(th_dense, wl_tir_sub, color=PALETTE["R"], lw=1.8, ls="-.", label=r"Substrate cutoff ($|\bar{k}_{x,-1}|=n_s$)")
    for col, wl_m in RGB_WAVELENGTHS_M.items():
        ax.axhline(wl_m * 1e9, color=PALETTE[col], lw=1.2, ls=":")

    ax.set_xlim(th_map[0], th_map[-1])
    ax.set_ylim(wl_map[0], wl_map[-1])
    ax.set_xlabel(r"Incident angle $\theta_{\mathrm{in}}$ [deg]")
    ax.set_ylabel(r"Wavelength $\lambda_0$ [nm]")
    ax.set_title(r"(a) Guided TIR Efficiency Map ($\Lambda=380$ nm)")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # --- Panel (b): Wakis vs RCWA RGB In-Coupler Efficiencies Across FOV ---
    ax = axes[1]
    ang_sym = data_t0["angles_sym_deg"]
    markers = {"B": "o", "G": "s", "R": "^"}
    lstyles = {"B": "-", "G": "--", "R": "-."}

    for col in ("B", "G", "R"):
        wl_nm = int(round(RGB_WAVELENGTHS_M[col] * 1e9))
        w_g = data_t0[f"wakis_guided_{col}"] * 100.0
        r_g = data_t0[f"rcwa_guided_{col}"] * 100.0
        u_c = tian_uniformity(w_g) * 100.0
        ax.plot(
            ang_sym,
            r_g,
            color=PALETTE[col],
            ls=lstyles[col],
            lw=1.8,
            label=f"{col} ({wl_nm} nm) RCWA",
        )
        ax.plot(
            ang_sym,
            w_g,
            color=PALETTE[col],
            marker=markers[col],
            ls="none",
            markersize=6.0,
            label=f"{col} Wakis FIT ($U={u_c:.1f}\\%$)",
        )

    # Also plot total energy sum (R_total + T_total) on secondary axis or scaled
    ang_pos = data_t0["angles_pos_deg"]
    e_tot_g = data_t0["energy_tot_G"] * 100.0
    ax.plot(
        ang_pos,
        e_tot_g * 0.25,
        color=PALETTE["ink"],
        ls=":",
        lw=1.4,
        marker="d",
        markersize=4.5,
        label=r"$0.25\times(R_{\mathrm{tot}}+T_{\mathrm{tot}})$ ($\approx 25\%$)",
    )
    ax.set_xlabel(r"Incident FOV angle $\theta_{\mathrm{in}}$ [deg]")
    ax.set_ylabel(r"Guided diffraction efficiency $\eta_{\mathrm{guided}}$ [%]")
    ax.set_title("(b) RGB In-Coupler FOV Sweep: Wakis vs RCWA")
    ax.set_ylim(0.0, 28.5)
    ax.legend(loc="lower center", ncol=2, fontsize=7.3, framealpha=0.92)

    # --- Panel (c): Forward Lambda=1900 nm Achromatic (4,5,6) Supercell (Different Design from Tian et al.) ---
    ax = axes[2]
    ang_t1 = data_t1["angles_deg"]
    u_sum_w = supercell_uniformity_summary(
        {col: data_t1[f"wakis_t_m_{col}"] for col in ("B", "G", "R")}
    )
    for col in ("B", "G", "R"):
        m_ord = RGB_ACHROMATIC_ORDERS[col]
        w_m = data_t1[f"wakis_t_m_{col}"] * 100.0
        r_m = data_t1[f"rcwa_t_m_{col}"] * 100.0
        u_m = tian_uniformity(w_m) * 100.0
        ax.plot(
            ang_t1,
            r_m,
            color=PALETTE[col],
            ls=lstyles[col],
            lw=1.9,
            label=f"{col} ($|m|={m_ord}$) RCWA",
        )
        ax.plot(
            ang_t1,
            w_m,
            color=PALETTE[col],
            marker=markers[col],
            ls="none",
            markersize=6.2,
            label=f"{col} ($|m|={m_ord}$) Wakis ($U_c={u_m:.0f}\\%$)",
        )
    ax.axhline(
        5.0,
        color=PALETTE["grey"],
        ls=":",
        lw=1.4,
        label=f"Tian et al. (2025, Fig. 3d, diff. design): ~5%, U=72.8% (our U_RGB={u_sum_w['U_joint_rgb']*100:.0f}%)",
    )
    ax.set_xlabel(r"Incident FOV angle $\theta_{\mathrm{in}}$ [deg]")
    ax.set_ylabel(r"Achromatic order efficiency $\eta_{m_c}$ [%]")
    ax.set_title(r"(c) Forward $\Lambda=1900$ nm $(4,5,6)$ Supercell (Diff. Design)")
    ax.set_ylim(0.0, 21.5)
    ax.legend(loc="lower right", fontsize=6.7, framealpha=0.92)

    out_path = fig_dir / "03_incoupler_rgb_efficiency_vs_angle.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig4_outcoupler_and_eyebox(data_t2: dict[str, np.ndarray], fig_dir: Path) -> Path:
    """Generate Figure 4: Out-coupler TIR bounce sweep, eye-box extraction, and uniformity trade-off."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), constrained_layout=True)

    # --- Panel (a): Out-Coupler TIR Bounce Sweep (Wakis vs RCWA) ---
    ax = axes[0]
    ang_wg = data_t2["angles_wg_deg"]
    ax.plot(ang_wg, data_t2["rcwa_r0"] * 100.0, color=PALETTE["B"], lw=2.0, ls="-", label=r"$R_0$ (0th-order TIR) RCWA")
    ax.plot(ang_wg, data_t2["wakis_r0"] * 100.0, color=PALETTE["B"], marker="o", ls="none", ms=6.2, label=r"$R_0$ Wakis FIT")

    ax.plot(
        ang_wg,
        data_t2["rcwa_tm1"] * 100.0,
        color=PALETTE["G"],
        lw=2.0,
        ls="--",
        label=r"$T_{-1}$ (Out-coupled to eye) RCWA",
    )
    ax.plot(ang_wg, data_t2["wakis_tm1"] * 100.0, color=PALETTE["G"], marker="s", ls="none", ms=6.2, label=r"$T_{-1}$ Wakis FIT")

    ax.plot(
        ang_wg,
        data_t2["rcwa_rm1"] * 100.0,
        color=PALETTE["R"],
        lw=1.8,
        ls="-.",
        label=r"$R_{-1}$ (Reflected -1 order) RCWA",
    )
    ax.plot(ang_wg, data_t2["wakis_rm1"] * 100.0, color=PALETTE["R"], marker="^", ls="none", ms=6.0, label=r"$R_{-1}$ Wakis FIT")

    ax.plot(
        ang_wg,
        data_t2["energy_oc"] * 100.0,
        color=PALETTE["ink"],
        lw=1.4,
        ls=":",
        marker="d",
        ms=4.5,
        label=r"Total energy $\sum(R_m+T_m) = 100\%$",
    )
    ax.axhline(80.0, color=PALETTE["B"], lw=1.0, ls=":", alpha=0.6)
    ax.text(35.5, 82.0, "Tian et al. (2025, Fig. 3b,e) $R_0 \\approx 80\\%$", fontsize=7.8, color=PALETTE["B"])

    ax.set_xlabel(r"Guided TIR bounce angle $\theta_{\mathrm{wg}}$ [deg] ($>\theta_c = 31.8^\circ$)")
    ax.set_ylabel("Diffraction efficiency [%]")
    ax.set_ylim(0.0, 106.0)
    ax.set_title("(a) Out-Coupler TIR Order Efficiencies: Wakis vs RCWA")
    ax.legend(loc="center right", fontsize=7.3, framealpha=0.92)

    # --- Panel (b): Eye-Box Pupil Expansion Across 6 Bounces (Constant vs Graduated Depths) ---
    ax = axes[1]
    bounces = np.arange(1, 7)
    ext_const = data_t2["eb_const_extracted"] * 100.0
    ext_grad = data_t2["eb_grad_extracted"] * 100.0
    u_const = tian_uniformity(ext_const) * 100.0
    u_grad = tian_uniformity(ext_grad) * 100.0

    width = 0.34
    ax.bar(
        bounces - width / 2,
        ext_const,
        width=width,
        color=PALETTE["sky"],
        edgecolor="#004C80",
        label=f"Uniform depth ($d=232$ nm, $U={u_const:.1f}\\%$)",
    )
    ax.bar(
        bounces + width / 2,
        ext_grad,
        width=width,
        color=PALETTE["G"],
        edgecolor="#004D38",
        label=f"Wang & Ma Table 1 graduated ($143\\to 384$ nm, $U={u_grad:.1f}\\%$)",
    )

    ax2 = ax.twinx()
    ax2.plot(
        bounces,
        data_t2["wang_depths_nm"],
        color=PALETTE["R"],
        marker="D",
        lw=1.8,
        ls="--",
        label="Groove depth $d_k$ [nm]",
    )
    ax2.set_ylabel("Out-coupler groove depth $d_k$ [nm]", color=PALETTE["R"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["R"])
    ax2.set_ylim(100.0, 440.0)
    ax2.grid(False)

    ax.set_xticks(bounces)
    ax.set_xlabel("Out-coupler TIR bounce index $k$ across eye-box")
    ax.set_ylabel("Extracted intensity per bounce $I_{\\mathrm{eye},k}$ [% of input]")
    ax.set_ylim(0.0, max(float(np.max(ext_const)), float(np.max(ext_grad))) * 1.38)
    ax.set_title("(b) Eye-Box Spatial Uniformity (Wang & Ma, 2023)")
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    # --- Panel (c): Analytic Trade-off Curve Between Single-Bounce Extraction & Uniformity ---
    ax = axes[2]
    t_out_sweep = np.linspace(0.01, 0.35, 140)
    n_b = 6
    # For constant extraction efficiency t_out and R_0 = 1 - 2*t_out:
    r0_sweep = np.clip(1.0 - 1.8 * t_out_sweep, 0.05, 0.99)
    # I_k = t_out * R_0^(k-1) -> eta_max = I_1, eta_min = I_N
    ratio_min_max = r0_sweep ** (n_b - 1)
    u_curve = (2.0 * ratio_min_max / (1.0 + ratio_min_max)) * 100.0
    tot_ext_curve = (1.0 - r0_sweep**n_b) * (t_out_sweep / (1.0 - r0_sweep)) * 100.0

    ax.plot(t_out_sweep * 100.0, u_curve, color=PALETTE["B"], lw=2.2, label="Eye-box uniformity $U$ (6 bounces) [%]")
    ax.plot(
        t_out_sweep * 100.0,
        tot_ext_curve,
        color=PALETTE["R"],
        lw=2.2,
        ls="--",
        label="Cumulative eye-box throughput [%]",
    )
    # Mark Tian et al. (2025) ~2% OMC operating point vs graduated Wang & Ma (2023)
    ax.axvline(2.0, color=PALETTE["gold"], lw=1.6, ls=":")
    ax.text(
        2.6,
        36.0,
        "Tian et al. (2025)\n$T_{\\mathrm{out}}\\approx 2\\%$, $R_0\\approx 80\\%$\n(High eye-box $U > 85\\%$)",
        fontsize=7.9,
        color="#8C5800",
        fontweight="bold",
    )
    ax.plot(11.5, u_grad, marker="*", markersize=11, color=PALETTE["G"], zorder=5)
    ax.annotate(
        f"Graduated depths\n(Wang & Ma Table 1)\n$U={u_grad:.1f}\\%$",
        xy=(11.5, u_grad),
        xytext=(15.5, u_grad - 12.0),
        fontsize=7.9,
        color=PALETTE["G"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["G"], lw=1.1),
    )

    ax.set_xlabel(r"Per-bounce out-coupling efficiency $T_{\mathrm{out}}$ [%]")
    ax.set_ylabel("Metric value [%]")
    ax.set_xlim(1.0, 35.0)
    ax.set_ylim(0.0, 105.0)
    ax.set_title("(c) Uniformity vs Throughput Trade-off")
    ax.legend(loc="center right", fontsize=7.6, framealpha=0.92)

    out_path = fig_dir / "04_outcoupler_tir_and_eyebox_uniformity.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig5_nearfield_and_convergence(
    data_t3: dict[str, np.ndarray],
    fig_dir: Path,
) -> Path:
    """Generate Figure 5: Labelled near-field maps (IC & OC) and 4-level convergence study."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), constrained_layout=True)

    # Helper to tile 3 unit cells laterally with Bloch phase exp(-i * kx * m * Lx)
    def _tile_3_periods(x_1d, phasor_2d, eps_2d, kx_val, period_m):
        xs_list, ph_list, ep_list = [], [], []
        for m_tile in (-1, 0, 1):
            xs_list.append((x_1d + m_tile * period_m) * 1e9)
            ph_list.append(phasor_2d * np.exp(-1j * kx_val * (m_tile * period_m)))
            ep_list.append(eps_2d)
        return (
            np.concatenate(xs_list),
            np.concatenate(ph_list, axis=0),
            np.concatenate(ep_list, axis=0),
        )

    # --- Panel (a): In-Coupler Steady-State Electric Field Re(E_y) ---
    ax = axes[0]
    x_ic_nm, ph_ic, ep_ic = _tile_3_periods(
        data_t3["ic_x"],
        data_t3["ic_phasor"],
        data_t3["ic_eps"],
        float(data_t3["ic_kx"]),
        PERIOD_GREEN_1ST_M,
    )
    z_ic_nm = data_t3["ic_z"] * 1e9
    re_ic = np.real(ph_ic)
    vmax_ic = float(np.percentile(np.abs(re_ic), 98.5))
    norm_ic = TwoSlopeNorm(vmin=-vmax_ic, vcenter=0.0, vmax=vmax_ic)

    pcm_a = ax.pcolormesh(x_ic_nm, z_ic_nm, re_ic.T, cmap="RdBu_r", norm=norm_ic, shading="auto")
    cbar_a = fig.colorbar(pcm_a, ax=ax, pad=0.02)
    cbar_a.set_label(r"$\mathrm{Re}(E_y)$ [arb. units]")

    # Overlay material boundaries and boundary labels
    ax.contour(x_ic_nm, z_ic_nm, ep_ic.T, levels=[2.0, 3.8], colors="black", linewidths=1.1)
    ax.axhline(z_ic_nm[14], color=PALETTE["grey"], ls=":", lw=1.1)
    ax.axhline(z_ic_nm[-14], color=PALETTE["grey"], ls=":", lw=1.1)
    ax.text(0.0, z_ic_nm[6], "Bottom CPML (-z)", ha="center", va="center", fontsize=7.6, bbox=dict(fc="white", alpha=0.8, pad=1.2))
    ax.text(0.0, z_ic_nm[-7], "Top CPML (+z)", ha="center", va="center", fontsize=7.6, bbox=dict(fc="white", alpha=0.8, pad=1.2))
    ax.text(
        0.0,
        360.0,
        r"Glass Waveguide ($n_s=1.9$, guided $m=-1$ wave)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(fc="white", ec=PALETTE["B"], alpha=0.88, pad=1.8),
    )
    ax.text(
        0.0,
        -450.0,
        r"Air ($n=1.0$, $\theta_{\mathrm{in}}=10^\circ$)",
        ha="center",
        fontsize=8.0,
        bbox=dict(fc="white", ec="black", alpha=0.85, pad=1.8),
    )
    ax.set_xlabel(r"$x$ [nm] ($3\Lambda$ Bloch-tiled view)")
    ax.set_ylabel(r"$z$ [nm]")
    ax.set_title(r"(a) In-Coupler Near Field $\mathrm{Re}(E_y)$ ($\lambda=530$ nm)")

    # --- Panel (b): Out-Coupler TIR Steady-State Electric Field Re(E_y) ---
    ax = axes[1]
    x_oc_nm, ph_oc, ep_oc = _tile_3_periods(
        data_t3["oc_x"],
        data_t3["oc_phasor"],
        data_t3["oc_eps"],
        float(data_t3["oc_kx"]),
        PERIOD_GREEN_1ST_M,
    )
    z_oc_nm = data_t3["oc_z"] * 1e9
    re_oc = np.real(ph_oc)
    vmax_oc = float(np.percentile(np.abs(re_oc), 98.5))
    norm_oc = TwoSlopeNorm(vmin=-vmax_oc, vcenter=0.0, vmax=vmax_oc)

    pcm_b = ax.pcolormesh(x_oc_nm, z_oc_nm, re_oc.T, cmap="RdBu_r", norm=norm_oc, shading="auto")
    cbar_b = fig.colorbar(pcm_b, ax=ax, pad=0.02)
    cbar_b.set_label(r"$\mathrm{Re}(E_y)$ [arb. units]")

    ax.contour(x_oc_nm, z_oc_nm, ep_oc.T, levels=[2.0, 3.8], colors="black", linewidths=1.1)
    ax.axhline(z_oc_nm[14], color=PALETTE["grey"], ls=":", lw=1.1)
    ax.axhline(z_oc_nm[-14], color=PALETTE["grey"], ls=":", lw=1.1)
    ax.text(0.0, z_oc_nm[6], "Bottom CPML (-z)", ha="center", va="center", fontsize=7.6, bbox=dict(fc="white", alpha=0.8, pad=1.2))
    ax.text(0.0, z_oc_nm[-7], "Top CPML (+z)", ha="center", va="center", fontsize=7.6, bbox=dict(fc="white", alpha=0.8, pad=1.2))
    ax.text(
        0.0,
        360.0,
        r"Air ($n=1.0$, extracted $m=-1$ wave to eye)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(fc="white", ec=PALETTE["G"], alpha=0.88, pad=1.8),
    )
    ax.text(
        0.0,
        -450.0,
        r"Glass Waveguide ($n_s=1.9$, TIR $\theta_{\mathrm{wg}}=45^\circ$)",
        ha="center",
        fontsize=8.0,
        bbox=dict(fc="white", ec=PALETTE["B"], alpha=0.88, pad=1.8),
    )
    ax.set_xlabel(r"$x$ [nm] ($3\Lambda$ Bloch-tiled view)")
    ax.set_ylabel(r"$z$ [nm]")
    ax.set_title(r"(b) Out-Coupler TIR Near Field $\mathrm{Re}(E_y)$ ($\theta_{\mathrm{wg}}=45^\circ$)")

    # --- Panel (c): 4-Level Grid Resolution Convergence Study vs RCWA ---
    ax = axes[2]
    cpw_lv = data_t3["cpw_levels"]
    err_m1 = data_t3["err_t_m1"]
    err_rms = data_t3["err_rms"]
    e_res = data_t3["energy_res"]

    ax.loglog(cpw_lv, err_m1, marker="o", color=PALETTE["R"], lw=2.0, label=r"$|T_{-1}^{\mathrm{Wakis}} - T_{-1}^{\mathrm{RCWA}}|$")
    ax.loglog(cpw_lv, err_rms, marker="s", color=PALETTE["B"], lw=2.0, ls="--", label=r"RMS error across $m\in\{-1,0,+1\}$")
    ax.loglog(cpw_lv, e_res, marker="^", color=PALETTE["G"], lw=1.8, ls="-.", label=r"Passivity $|\sum(R_m+T_m) - 1|$")

    # O(dx^2) second-order reference slope
    ref_O2 = err_rms[0] * (cpw_lv / float(cpw_lv[0])) ** (-2.0)
    ax.loglog(cpw_lv, ref_O2, color=PALETTE["ink"], ls=":", lw=1.5, label=r"$\mathcal{O}(\Delta x^2)$ reference slope")

    ax.set_xlabel(r"Grid resolution [cells per minimum material wavelength $\lambda_{\min}/n_{\max}$]")
    ax.set_ylabel("Absolute diffraction efficiency error")
    ax.set_title("(c) FIT Resolution Convergence vs Exact RCWA")
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    out_path = fig_dir / "05_nearfield_and_resolution_study.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_fig6_wavefront_gif(data_t3: dict[str, np.ndarray], fig_dir: Path) -> Path:
    """Generate Figure 6: Animated harmonic wavefront GIF (< 5 MB) with labelled materials & BCs."""
    x_1d = data_t3["ic_x"]
    z_nm = data_t3["ic_z"] * 1e9
    phasor = data_t3["ic_phasor"]
    eps_2d = data_t3["ic_eps"]
    kx_val = float(data_t3["ic_kx"])

    xs_list, ph_list, ep_list = [], [], []
    for m_tile in (-1, 0, 1):
        xs_list.append((x_1d + m_tile * PERIOD_GREEN_1ST_M) * 1e9)
        ph_list.append(phasor * np.exp(-1j * kx_val * (m_tile * PERIOD_GREEN_1ST_M)))
        ep_list.append(eps_2d)
    x_nm = np.concatenate(xs_list)
    ph_tiled = np.concatenate(ph_list, axis=0)
    ep_tiled = np.concatenate(ep_list, axis=0)

    vmax = float(np.percentile(np.abs(ph_tiled), 98.0))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    n_frames = 18
    phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)

    pcm = ax.pcolormesh(
        x_nm,
        z_nm,
        np.real(ph_tiled).T,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"Electric field $\mathrm{Re}\{E_y(x,z)e^{i\omega t}\}$ [arb. u.]")

    # Persistent material contours and boundary labels on every frame
    ax.contour(x_nm, z_nm, ep_tiled.T, levels=[2.0, 3.8], colors="black", linewidths=1.2)
    ax.axhline(z_nm[14], color=PALETTE["grey"], ls=":", lw=1.2)
    ax.axhline(z_nm[-14], color=PALETTE["grey"], ls=":", lw=1.2)
    ax.axvline(-0.5 * PERIOD_GREEN_1ST_M * 1e9, color=PALETTE["purple"], ls="-.", lw=1.1)
    ax.axvline(+0.5 * PERIOD_GREEN_1ST_M * 1e9, color=PALETTE["purple"], ls="-.", lw=1.1)

    ax.text(0.0, z_nm[6], "Bottom CPML Absorber (-z)", ha="center", va="center", fontsize=8.0, bbox=dict(fc="white", alpha=0.85, pad=1.5))
    ax.text(0.0, z_nm[-7], "Top CPML Absorber (+z)", ha="center", va="center", fontsize=8.0, bbox=dict(fc="white", alpha=0.85, pad=1.5))
    ax.text(
        0.0,
        360.0,
        r"Glass Waveguide ($n_s=1.90$) — Guided TIR Mode",
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        bbox=dict(fc="white", ec=PALETTE["B"], alpha=0.9, pad=2.0),
    )
    ax.text(
        0.0,
        80.0,
        r"$\mathrm{Si_3N_4}$ In-Coupler Ridges ($n=2.00$, $h=160$ nm)",
        ha="center",
        fontsize=8.0,
        bbox=dict(fc="white", ec=PALETTE["R"], alpha=0.88, pad=1.5),
    )
    ax.text(
        0.0,
        -450.0,
        r"Air Cladding ($n=1.00$) — Oblique Incident Beam ($\theta_{\mathrm{in}}=10^\circ$,Bloch $\pm x$)",
        ha="center",
        fontsize=8.0,
        bbox=dict(fc="white", ec="black", alpha=0.88, pad=1.8),
    )
    title_obj = ax.set_title(r"In-Coupler Harmonic Wavefront ($\omega t = 0.00\;\mathrm{rad}$)")
    ax.set_xlabel(r"$x$ [nm] ($3\Lambda$ Bloch-periodic super-window)")
    ax.set_ylabel(r"$z$ [nm]")

    def _update(frame_idx: int):
        phi_t = phases[frame_idx]
        frame_field = np.real(ph_tiled * np.exp(1j * phi_t)).T
        pcm.set_array(frame_field.ravel())
        title_obj.set_text(
            rf"In-Coupler Harmonic Wavefront ($\lambda=530$ nm, $\theta_{{\mathrm{{in}}}}=10^\circ$, $\omega t = {phi_t:.2f}$ rad)"
        )
        return pcm, title_obj

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=90, blit=False)
    out_path = fig_dir / "06_incoupler_wavefront_movie.gif"
    ani.save(out_path, writer=animation.PillowWriter(fps=11), dpi=110)
    plt.close(fig)
    return out_path


# %% [markdown]
# ## 3. Workflow Execution & Benchmark Comparison
#
# When executed from the CLI (`--quick`, `--full`, `--task`, `--plot`), we run or load
# the four simulation tasks, save `.npz` arrays and `.json` summaries in `data/`,
# and generate all six figures in `figures/`.


# %%
def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.list_tasks:
        for idx, (slug, desc) in TASKS.items():
            print(f"Task {idx} ({slug}): {desc}")
        return {}

    out_dir: Path = args.out
    fig_dir: Path = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        0: lambda: run_task_0_rgb_incoupler(quick=args.quick, processes=args.processes),
        1: lambda: run_task_1_achromatic_supercell(quick=args.quick, processes=args.processes),
        2: lambda: run_task_2_outcoupler_and_eyebox(quick=args.quick, processes=args.processes),
        3: lambda: run_task_3_nearfield_and_resolution(quick=args.quick),
    }

    if args.task is not None:
        t_idx = args.task
        slug, _ = TASKS[t_idx]
        summary, npz_dict = runners[t_idx]()
        np.savez_compressed(out_dir / f"task_{t_idx}_{slug}.npz", **npz_dict)
        (out_dir / f"task_{t_idx}_{slug}.json").write_text(json.dumps(summary, indent=2))
        print(f"[Tutorial 6.4] Saved Task {t_idx} ({slug}) to {out_dir}")
        return summary

    all_summaries: dict[str, object] = {}
    loaded_npz: dict[int, dict[str, np.ndarray]] = {}

    for t_idx, (slug, _) in TASKS.items():
        npz_path = out_dir / f"task_{t_idx}_{slug}.npz"
        json_path = out_dir / f"task_{t_idx}_{slug}.json"
        if args.plot and npz_path.exists() and json_path.exists():
            with np.load(npz_path) as d:
                loaded_npz[t_idx] = {k: d[k] for k in d.files}
            all_summaries[slug] = json.loads(json_path.read_text())
        else:
            summary, npz_dict = runners[t_idx]()
            np.savez_compressed(npz_path, **npz_dict)
            json_path.write_text(json.dumps(summary, indent=2))
            loaded_npz[t_idx] = npz_dict
            all_summaries[slug] = summary

    (out_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))

    # Generate all 6 tutorial figures
    f1 = plot_fig1_schematic_and_mesh(fig_dir, quick=args.quick)
    f2 = plot_fig2_kspace_and_fov(loaded_npz[0], fig_dir)
    f3 = plot_fig3_incoupler_rgb(loaded_npz[0], loaded_npz[1], fig_dir)
    f4 = plot_fig4_outcoupler_and_eyebox(loaded_npz[2], fig_dir)
    f5 = plot_fig5_nearfield_and_convergence(loaded_npz[3], fig_dir)
    f6 = plot_fig6_wavefront_gif(loaded_npz[3], fig_dir)

    print("=== Tutorial 6.4 Summary ===")
    print(json.dumps(all_summaries, indent=2))
    print("Generated figures:")
    for fp in (f1, f2, f3, f4, f5, f6):
        print(f"  - {fp.name} ({fp.stat().st_size / 1024:.1f} kB)")
    return all_summaries


if __name__ == "__main__":
    main()

# %% [markdown]
# ## 4. Results, Validation vs Literature, and Resolution Study
#
# ### Comparison against verified literature
#
# | Physical quantity / metric | Literature reference & value | Wakis 2D FIT / RCWA result | Agreement / Notes |
# | :--- | :--- | :--- | :--- |
# | Grating vector closure $|\mathbf{G}_{\text{IC}} + \mathbf{G}_{\text{EPE}} + \mathbf{G}_{\text{OC}}|$ | $\mathbf{0}$ (**Kress & Chatterjee, 2021**, Fig. 28) | $0.00$ ($\max|\bar{\mathbf{k}}_{\text{out}} - \bar{\mathbf{k}}_{\text{in}}| = 0$) | Exact analytical identity |
# | Achromatic RGB momentum product $(m_R\lambda_R, m_G\lambda_G, m_B\lambda_B)$ | $(4\times 663, 5\times 530, 6\times 442) = (2652, 2650, 2652)\;\text{nm}$ (**Tian et al., 2025**, Eq. 2, p. 3) | $\Delta\theta_{\text{wg}} = 0.047^\circ$ across RGB at $\Lambda = 1900\;\text{nm}$ | Exact match to Eq. (2) |
# | Common RGB 1D FOV in waveguide ($n_s = 1.90$ & $1.74$) | **Tian et al. (2025)**, p. 3, Fig. 2 ($n_s = 1.90$): $> 45^\circ$ achromatic vs $< 20^\circ$ conventional; **Wang & Ma (2023)**, pp. 1, 6 ($n_s = 1.74$): $40.0^\circ$ | Our Eq. (4) formula: $53.5^\circ$ ($\theta_{\text{wg},\max}=90^\circ$) / $46.1^\circ$ ($\theta_{\text{wg},\max}=78^\circ$) at $n_s=1.90$ vs $18.3^\circ$ conventional ($m=1$); $43.4^\circ$ at $n_s=1.74$; $63.3^\circ$ at $n_s=2.05$ | Formula upper bounds consistent with reported practical FOVs |
# | $\Lambda=1900\;\text{nm}$ $(4,5,6)$ IMC order efficiency & uniformity $U$ *(different designs)* | **Tian et al. (2025)** inverse-designed metasurface (Fig. 3a,d, pp. 4–5): $\bar{\eta} \approx 5\%$, joint RGB+angle $U = 72.8\%$ (ridge coordinates not tabulated) | Our 6-ridge forward supercell: $\bar{\eta} = 8.1\text{--}16.6\%$, per-channel $U_c = 85.4\text{--}93.7\%$, joint RGB+angle $U_{\text{RGB}} = 58.3\%$ (Wakis vs RCWA $|\Delta\eta| \le 1.9\times 10^{-3}$) | **Not a like-for-like comparison** (different supercell ridge designs) |
# | Out-coupler $0\text{th}$-order TIR reflectance $R_0$ & uniformity | $R_0 \approx 80\%$, $U = 85.1\%$ (read from **Tian et al., 2025**, Fig. 3b,e, p. 4–5) | $R_0 = 56.6\text{--}86.4\%$ (Wakis vs RCWA $|\Delta R_0| \le 2.9\times 10^{-3}$, $|\Delta T_{-1}| \le 1.0\times 10^{-3}$) | Evanescent $T_0 \equiv 0$ above $\theta_c = 31.8^\circ$ |
# | Eye-box 6-bounce extraction uniformity $U$ | Graduated depths $143\to 384\;\text{nm}$ (**Wang & Ma, 2023**, Table 1, p. 11) | Constant depth $U = 23.8\%$ $\to$ Graduated depths $U = 65.5\%$ | $+41.8\%$ uniformity improvement |
#
# ### Grid resolution convergence study (Task 3, $\lambda = 530\;\text{nm}$, $\theta_{\text{in}} = 10^\circ$)
#
# Refining the FIT mesh from $16$ to $34$ cells per minimum material wavelength ($\lambda_{\min}/n_{\max} = 265\;\text{nm}$ in $\text{Si}_3\text{N}_4$) with subpixel permittivity smoothing reduces the RMS order efficiency error against exact 51-harmonic RCWA monotonically while preserving passivity $|\sum_m (R_m + T_m) - 1| < 4\times 10^{-3}$.
#
# ---
#
# ## 5. Limits & Failure Modes
#
# 1. **TIR critical-angle leakage ($\theta_{\text{wg}} < \theta_c$) and Wood-Rayleigh anomalies:**
#    When an incident angle $\theta_{\text{in}}$ shifts $|\bar{k}_{x,m}|$ below $1.0$, the $m\text{th}$ order is no longer trapped by TIR and leaks directly into the air cladding. Conversely, at the exact cutoff $|\bar{k}_{x,m}| = n_s$ (or $1.0$), the longitudinal wavenumber $k_{z,m} \to 0$: the mode propagates grazingly along the grating plane, requiring longer time-domain ring-down (`run_time_cycles >= 40`) to resolve sharp Wood-Rayleigh cusps.
# 2. **Constant-$k_x$ vs constant-angle broadband Dispersion:**
#    A single Bloch time-domain run fixes $k_x = k_{x0}$ across all frequencies in the pulse, meaning the physical incidence angle $\theta(\omega) = \arcsin(c k_{x0} / (\omega n_{\text{inc}}))$ drifts with frequency. Calling `AngleSweep.run()` followed by `AngleSweepResult.resample()` is mandatory whenever broadband spectra at a fixed angle $\theta_{\text{in}}$ are required.
# 3. **Symmetric binary teeth split power equally at normal incidence:**
#    A symmetric binary ridge ($\text{fill}=0.50$) cannot distinguish $+x$ from $-x$ at $\theta_{\text{in}} = 0^\circ$, forcing $\eta_{+1} = \eta_{-1}$ and capping single-sided in-coupling efficiency below $50\%$. Breaking mirror symmetry via slanted or blazed grooves (**Tutorial 6.5**) is essential to achieve $> 75\%$ unidirectional coupling into the waveguide.
#
# ---
#
# ## 6. Exercises
# 1. **High-index SiC waveguide ($n_s = 2.6$):** Increase `N_WG_TIAN` from $1.9$ to $2.6$ in `kspace_combiner_map()` and verify that the theoretical symmetric 1D FOV expands from $53.5^\circ$ to $106.3^\circ$.
# 2. **TM polarization (`polarization="TM"`):** Re-run Task 0 with `polarization="TM"` and compare the angular roll-off of $\eta_{-1}(\theta_{\text{in}})$ near the Brewster angle against TE polarization.
# 3. **Out-coupler depth optimization:** Modify `WANG_OUTCOUPLER_DEPTHS_NM` in Task 2 to achieve $U > 95\%$ across all 6 eye-box bounces.
#
# ---
#
# ## 7. Verified References
# 1. **Z. Tian, X. Zhu, P. A. Surman, Z. Chen, and X. W. Sun**, *"An achromatic metasurface waveguide for augmented reality displays"*, **Light: Science & Applications** **14**, 94 (2025), DOI: [`10.1038/s41377-025-01761-w`](https://doi.org/10.1038/s41377-025-01761-w). (Eqs. 1–3, Figs. 2–3, $\Lambda = 1900\;\text{nm}$, $(m_R, m_G, m_B) = (4, 5, 6)$, $n_s = 1.9$, $h_{\text{SiN}} = 160\;\text{nm}$.)
# 2. **H. Wang and D. Ma**, *"Surface relief grating near-eye display waveguide design"*, `arXiv:2307.01600` (2023), DOI: [`10.48550/arXiv.2307.01600`](https://doi.org/10.48550/arXiv.2307.01600). (Eqs. 4–6 for TIR FOV limits; Table 1 & Fig. 7 for graduated 6-zone out-coupler depths $143\text{--}384\;\text{nm}$ at $\lambda = 525\;\text{nm}$, $n = 1.74$.)
# 3. **B. C. Kress and I. Chatterjee**, *"Waveguide combiners for mixed reality headsets: a nanophotonics design perspective"*, **Nanophotonics** **10**(1), 41–74 (2021), DOI: [`10.1515/nanoph-2020-0410`](https://doi.org/10.1515/nanoph-2020-0410). (Fig. 28 normalized $k$-space combiner diagrams and grating vector closure.)
