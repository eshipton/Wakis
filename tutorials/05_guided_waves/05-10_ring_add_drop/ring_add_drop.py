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
# # Tutorial 5.10 · Microring add-drop filter
#
# A **silicon microring add-drop filter** is one of the canonical building blocks
# of silicon photonics, serving as a wavelength-selective channel router in dense
# wavelength-division multiplexing (DWDM) transceivers, optical interconnects,
# modulators, and refractive-index biosensors. The device consists of a closed
# circular silicon strip waveguide of radius $R = 5.0\;\mu\text{m}$ side-coupled
# across symmetric $g = 100\;\text{nm}$ evanescent air/oxide gaps to two parallel
# straight bus waveguides of width $w_{\text{wg}} = 0.43\;\mu\text{m}$ and thickness
# $h_{\text{wg}} = 0.22\;\mu\text{m}$.
#
# When a broadband fundamental $\text{TE}_0$ mode is launched into the upper input
# port (`in`), off-resonance wavelengths bypass the ring and exit the `through`
# port with minimal insertion loss. Whenever the optical round-trip phase around
# the ring circumference $L = 2\pi R$ is an integer multiple of $2\pi$,
#
# $$m\,\lambda_m = n_{\text{eff}}(\lambda_m)\,L, \qquad m \in \mathbb{Z}^+,$$
#
# constructive interference builds up a circulating whispering-gallery wave inside
# the cavity. Destructive interference between the direct bus transmission and the
# wave coupled back from the ring extinguishes the `through` port and transfers
# the optical power into the backward-propagating `drop` port on the lower bus.
#
# Compared with non-resonant splitters or bends, a microring resonator is a
# **stringent benchmark for time-domain Maxwell solvers**:
#
# 1. **Phase-velocity accuracy on a curved path ($\text{cpw} \ge 10$)**: A $0.1\%$ error
#    in the numerical effective index $n_{\text{eff}}$ shifts every resonance by
#    $\delta\lambda \approx \lambda_m\,\delta n_{\text{eff}}/n_g \approx 1\;\text{nm}$
#    — comparable to the full-width at half-maximum ($\Delta\lambda_{\text{FWHM}} \approx 1.0\text{–}4.7\;\text{nm}$).
#    The background grid must never use fewer than $10\;\text{cells}/\lambda$ inside the
#    slab effective index.
# 2. **Coupler-gap feature resolution ($N \ge 4$ cells across the gap)**: At $10\;\text{cpw}$
#    ($\Delta x_{\text{bg}} = 52.5\;\text{nm}$), the $g = 100\;\text{nm}$ coupling gap spans
#    only $1.90$ cells ($1.14\text{–}2.86$ cells for $g \in \{60, 100, 150\}\;\text{nm}$).
#    An under-resolved gap ($1\text{–}2$ cells) makes the coupling coefficient $\kappa^2$
#    and loaded $Q_L$ depend on where the cell boundaries happen to fall relative to the
#    gap walls. We use local `mesh_overrides` boxes at the two coupling points to enforce
#    $N \ge 4$ cells across every gap ($L_1 = (10, 4)$, $L_2 = (14, 6)$, $L_3 = (20, 8)$,
#    and full reference $R = (40, 16)$).
# 3. **Long cavity ring-down**: Because loaded quality factors $Q_L \sim 400\text{–}2000$
#    store light for photon lifetimes $\tau_p = Q_L / \omega_0 \sim 0.3\text{–}1.6\;\text{ps}$,
#    the time-domain simulation must integrate for $3\text{–}12\;\text{ps}$ until
#    the circulating field decays below $10^{-3}$ of peak energy before evaluating the
#    discrete Fourier transform.
#
# ### What this tutorial covers
#
# 1. **Coupled-mode & transfer-matrix theory (Yariv 2000)** relating the directional
#    coupler coefficients $(t, \kappa)$ and round-trip amplitude transmission $a$
#    to the Lorentzian resonance linewidth $\Delta\lambda_{\text{FWHM}}$, loaded
#    quality factor $Q_L$, free spectral range $\Delta\lambda_{\text{FSR}}$, finesse
#    $\mathcal{F}$, and critical-coupling condition.
# 2. **Variational 2D Effective-Index Method (EIM)**: collapsing the $220\;\text{nm}$
#    SOI slab into an effective core index $n_{\text{eff,slab}}(\text{TE}_0) = 2.8478$
#    at $\lambda_0 = 1.55\;\mu\text{m}$, and comparing the 2D EIM group index
#    ($n_{g,\text{EIM}} \approx 3.30$, $\Delta\lambda_{\text{FSR}} \approx 23.2\;\text{nm}$)
#    with full 3D dispersive silicon ($n_{g,\text{3D}} \approx 4.15$,
#    $\Delta\lambda_{\text{FSR}} \approx 18.5\;\text{nm}$).
# 3. **Two-criterion resolution ladder $(\text{cpw}, N)$ with local `mesh_overrides`**
#    and **anisotropic Kottke subpixel smoothing** across the coupler gaps and curved
#    ring perimeter (`RingPICSimulation` + `FastRingPolySlab` + `FastModeMonitor` +
#    `attach_fast_2d_te_step`).
# 4. **14 publication-grade diagnostic figures** covering geometry, EIM stack,
#    coupler evanescent physics, zoomed coupler-gap mesh & permittivity tensor comparison
#    ($1.9$ cells vs $L_1$ $4$ cells vs $L_3$ $8$ cells), port eigenmodes,
#    waveguide dispersion, on- vs off-resonance steady-state fields, pulse
#    propagation GIF, broadband 4-port S-parameters, Lorentzian/CMT fits,
#    FSR & $Q_L$ spectra, a coupling-gap sweep ($g \in \{60, 100, 150\}\;\text{nm}$),
#    convergence of resonance shift (in pm) and loaded $Q_L$ against an independent
#    Kottke reference ($R = (40, 16)$ in `--full`, $L_3 = (20, 8)$ in `--quick`, with an
#    under-resolved $1.9$-cell warning pair), and dual-port ring-down time traces.

# %%
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle
from scipy.constants import c as c_light, epsilon_0

TUTORIAL_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    FastRingPolySlab,
    RingPICSimulation,
    _lorentzian_peak,
    cmt_add_drop_spectra,
    coupler_gap_mesh_overrides,
    find_ring_resonances,
    measure_cells_across_gap,
)

from wakis.templates.photonics import (  # noqa: E402
    PolySlab,
    Port,
    make_movie,
    n_to_eps,
    plot_eim_stack,
    plot_layout,
    ring,
    slab_neff,
    straight,
)

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

# Resolution ladders: each level is (level_name, cpw, n_cells_across_gap)
#   Rule 1: cpw >= 10 cells per wavelength in the slab effective index everywhere
#   Rule 2: N >= 4 cells across the smallest feature (coupler gap g) via mesh_overrides
#   --quick: L1=(10, 4), L2=(14, 6) + Kottke reference at L3=(20, 8); gap sweep at L2=(14, 6)
#   --full : L1=(10, 4), L2=(14, 6), L3=(20, 8) + Kottke reference at R=(40, 16); gap sweep at L3=(20, 8)
LADDER_L1 = ("L1", 10, 4)
LADDER_L2 = ("L2", 14, 6)
LADDER_L3 = ("L3", 20, 8)
LADDER_REF_FULL = ("R", 40, 16)
QUICK_LADDER = (LADDER_L1, LADDER_L2)
FULL_LADDER = (LADDER_L1, LADDER_L2, LADDER_L3)
QUICK_REF = LADDER_L3
FULL_REF = LADDER_REF_FULL
UNDERRESOLVED_DEMO = ("underresolved", 10, None)

# Device specification: 220 nm SOI microring add-drop filter (R = 5.0 um, w = 0.43 um, g = 100 nm)
N_SI = 3.476  # Palik low-loss c-Si at 1.55 um
N_SIO2 = 1.444  # Palik low-loss SiO2 at 1.55 um
THICKNESS_SI = 0.22e-6  # [m] 220 nm SOI slab thickness
W_WG = 0.43e-6  # [m] bus and ring waveguide width
R_RING = 5.0e-6  # [m] ring mean radius
GAP_DEFAULT = 0.10e-6  # [m] nominal 100 nm edge-to-edge coupling gap
GAP_SWEEP_NM = (60.0, 100.0, 150.0)  # [nm] coupling-gap study sweep
WL_MIN = 1.50e-6  # [m]
WL_MAX = 1.60e-6  # [m]
WL_REF = 1.55e-6  # [m]
N_FREQS = 1001  # 0.1 nm spectral grid across 1500-1600 nm

# Port & domain coordinates
X_PORT = 5.0e-6  # [m] ports placed at x = +-5.0 um
LX_DOMAIN = 12.6e-6  # [m] x in [-6.3, +6.3] um
PORT_WIDTH = 1.72e-6  # [m] 4 * w_wg modal cross-section window

# %% [markdown]
# ## 1. Coupled-Mode & Transfer-Matrix Theory of an Add-Drop Ring
#
# Following Yariv (2000) and Bogaerts *et al.* (2012), each bus-to-ring directional
# coupler is modeled as a lossless point scattering matrix with self-coupling
# amplitude $t = \sqrt{1 - \kappa^2}$ and cross-coupling amplitude $-j\kappa$,
# where $\kappa^2$ is the fraction of optical power transferred across the gap per
# pass. For a symmetric add-drop filter ($t_1 = t_2 = t$, $\kappa_1 = \kappa_2 = \kappa$)
# with single-pass ring amplitude transmission $a = \exp(-\alpha L / 2)$ and
# round-trip phase $\phi(\lambda) = \frac{2\pi}{\lambda}\,n_{\text{eff}}(\lambda)\,L$
# ($L = 2\pi R$), summing the geometric series of circulating waves yields the
# complex transmission amplitudes:
#
# $$S_{\text{through}}(\lambda) = \frac{t\,(1 - a\,e^{-j\phi})}{1 - t^2 a\,e^{-j\phi}},
# \qquad
# S_{\text{drop}}(\lambda) = \frac{-\kappa^2 \sqrt{a}\,e^{-j\phi/2}}{1 - t^2 a\,e^{-j\phi}}.$$
#
# Taking squared magnitudes gives the canonical Airy / Lorentzian power spectra:
#
# $$T_{\text{through}}(\lambda) =
# \frac{t^2 (1 - a)^2 + 4 t^2 a \sin^2(\phi/2)}{(1 - t^2 a)^2 + 4 t^2 a \sin^2(\phi/2)},
# \qquad
# T_{\text{drop}}(\lambda) =
# \frac{(1 - t^2)^2\,a}{(1 - t^2 a)^2 + 4 t^2 a \sin^2(\phi/2)}.$$
#
# ### Key figures of merit
#
# - **Resonance condition ($\phi_m = 2\pi m$)**:
#   $$\lambda_m = \frac{n_{\text{eff}}(\lambda_m)\,L}{m}.$$
# - **Free Spectral Range (FSR)** — spacing between adjacent azimuthal orders $m$ and $m\pm 1$,
#   governed to first order by the **group index**
#   $n_g(\lambda) = n_{\text{eff}}(\lambda) - \lambda\,\frac{\mathrm{d}n_{\text{eff}}}{\mathrm{d}\lambda}$:
#   $$\Delta\lambda_{\text{FSR}} = \frac{\lambda_m^2}{n_g(\lambda_m)\,L}.$$
# - **Full-Width at Half-Maximum (FWHM) & Loaded Quality Factor $Q_L$**:
#   $$\Delta\lambda_{\text{FWHM}} = \frac{\lambda_m^2}{\pi\,n_g\,L}\,\frac{1 - t^2 a}{\sqrt{t^2 a}},
#   \qquad
#   Q_L = \frac{\lambda_m}{\Delta\lambda_{\text{FWHM}}} = \frac{\pi\,n_g\,L}{\lambda_m}\,\frac{\sqrt{t^2 a}}{1 - t^2 a}.$$
# - **Finesse** (ratio of resonance spacing to linewidth, equal to the effective number of round trips):
#   $$\mathcal{F} = \frac{\Delta\lambda_{\text{FSR}}}{\Delta\lambda_{\text{FWHM}}} = \frac{\pi\sqrt{t^2 a}}{1 - t^2 a}.$$
# - **Critical coupling & extinction**: At resonance ($\sin(\phi/2)=0$),
#   $T_{\text{through}}(\lambda_m) = \frac{t^2(1-a)^2}{(1-t^2 a)^2}$ and
#   $T_{\text{drop}}(\lambda_m) = \frac{\kappa^4 a}{(1-t^2 a)^2}$. When intrinsic
#   cavity loss is small ($a \to 1$), $T_{\text{through}}(\lambda_m) \to 0$ and
#   $T_{\text{drop}}(\lambda_m) \to 1$ for any symmetric coupler ($t_1 = t_2 = t$).
#   Increasing the gap $g$ reduces $\kappa^2$, narrowing the linewidth and raising
#   $Q_L$; decreasing $g$ over-couples the cavity, broadening the resonance.

# %% [markdown]
# ## 2. Effective-Index Reduction & Waveguide Dispersion
#
# In the variational 2D **Effective-Index Method (EIM)**, the vertical $220\;\text{nm}$
# silicon slab bounded by $\text{SiO}_2$ cladding is solved first as a 1D slab
# waveguide at $\lambda_0 = 1.55\;\mu\text{m}$:
#
# $$n_{\text{eff,slab}}(\text{TE}_0) = 2.8478 \qquad (\text{vs. bulk } n_{\text{Si}} = 3.476).$$
#
# The 3D geometry is then projected onto the $(x, y)$ plane with core refractive
# index $n_{\text{core}} = n_{\text{eff,slab}}$ and cladding index $n_{\text{SiO}_2} = 1.444$,
# solved for the in-plane quasi-$\text{TE}$ polarization ($E_x, E_y, H_z$).
#
# Solving the horizontal channel eigenproblem across $\lambda \in [1.50, 1.60]\;\mu\text{m}$
# gives a fundamental mode effective index $n_{\text{eff}}(1.55\;\mu\text{m}) = 2.4029$
# and an in-plane geometric group index $n_{g,\text{EIM}}(1.55\;\mu\text{m}) = 3.3005$,
# corresponding to an analytical 2D EIM free spectral range of:
#
# $$\Delta\lambda_{\text{FSR,EIM}} = \frac{(1.55\;\mu\text{m})^2}{3.3005 \times 2\pi(5.0\;\mu\text{m})} = 23.17\;\text{nm}.$$
#
# In a full 3D dispersive SOI simulation, vertical slab confinement dispersion
# ($\mathrm{d}n_{\text{eff,slab}}/\mathrm{d}\lambda \approx -0.45\;\mu\text{m}^{-1}$)
# and crystalline silicon material dispersion ($\mathrm{d}n_{\text{Si}}/\mathrm{d}\lambda \approx -0.075\;\mu\text{m}^{-1}$)
# add to the horizontal waveguide dispersion, raising the total group index to
# $n_{g,\text{3D}} \approx 4.15$ and compressing the 3D FSR to $\Delta\lambda_{\text{FSR,3D}} \approx 18.5\;\text{nm}$
# (4 resonances in 2D EIM vs 5 resonances in 3D across $1.50\text{–}1.60\;\mu\text{m}$).
# We compute both curves below so the 2D EIM and 3D dispersive regimes can be
# compared directly.

# %%
def compute_waveguide_dispersion(n_pts=41):
    """
    Compute the TE0 channel effective index n_eff(lambda) and group index n_g(lambda)
    for both the 2D EIM waveguide (constant slab index at 1550 nm) and the 3D-equivalent
    dispersive SOI waveguide (wavelength-dependent slab + Palik c-Si dispersion).

    Parameters
    ----------
    n_pts : int, optional
        Number of wavelength samples across [1.50, 1.60] um.

    Returns
    -------
    dict
        Dictionary with wavelength array `wl_m`, `neff_eim`, `ng_eim`, `fsr_eim_nm`,
        `neff_3d_eq`, `ng_3d_eq`, `fsr_3d_eq_nm`, and `neff_slab_ref`.
    """
    wl_m = np.linspace(WL_MIN, WL_MAX, n_pts)
    neff_slab_ref = float(
        slab_neff(N_SI, N_SIO2, THICKNESS_SI, WL_REF, polarization="TE")
    )

    neff_eim = np.zeros_like(wl_m)
    neff_3d_eq = np.zeros_like(wl_m)

    for i, wl in enumerate(wl_m):
        # 1) 2D EIM with fixed slab index at 1550 nm (in-plane Ey perpendicular to y-walls -> TM in 1D y-slab)
        neff_eim[i] = float(
            slab_neff(neff_slab_ref, N_SIO2, W_WG, wl, polarization="TM")
        )

        # 2) 3D-equivalent including vertical slab dispersion + c-Si material dispersion
        n_si_disp = N_SI - 0.076 * ((wl - WL_REF) * 1e6)
        n_sio2_disp = N_SIO2 - 0.012 * ((wl - WL_REF) * 1e6)
        n_slab_wl = float(
            slab_neff(
                n_si_disp, n_sio2_disp, THICKNESS_SI, wl, polarization="TE"
            )
        )
        neff_3d_eq[i] = float(
            slab_neff(n_slab_wl, n_sio2_disp, W_WG, wl, polarization="TM")
        )

    dneff_dwl_eim = np.gradient(neff_eim, wl_m)
    ng_eim = neff_eim - wl_m * dneff_dwl_eim
    L_ring = 2.0 * np.pi * R_RING
    fsr_eim_nm = (wl_m**2 / (ng_eim * L_ring)) * 1e9

    dneff_dwl_3d = np.gradient(neff_3d_eq, wl_m)
    ng_3d_eq = neff_3d_eq - wl_m * dneff_dwl_3d
    fsr_3d_eq_nm = (wl_m**2 / (ng_3d_eq * L_ring)) * 1e9

    return {
        "wl_m": wl_m,
        "wl_nm": wl_m * 1e9,
        "neff_slab_ref": neff_slab_ref,
        "neff_eim": neff_eim,
        "ng_eim": ng_eim,
        "fsr_eim_nm": fsr_eim_nm,
        "neff_3d_eq": neff_3d_eq,
        "ng_3d_eq": ng_3d_eq,
        "fsr_3d_eq_nm": fsr_3d_eq_nm,
    }


# %% [markdown]
# ## 3. Geometry & 4-Port Simulation Builder
#
# We construct the microring add-drop filter using the `wakis.templates.photonics`
# primitives:
#
# - **Upper bus waveguide**: `PolySlab(straight((-6.3e-6, +y_bus), (+6.3e-6, +y_bus), width=0.43e-6), ...)`
#   where $y_{\text{bus}} = R + w_{\text{wg}} + g = 5.53\;\mu\text{m}$ for $g = 0.10\;\mu\text{m}$.
# - **Lower bus waveguide**: `PolySlab(straight((-6.3e-6, -y_bus), (+6.3e-6, -y_bus), width=0.43e-6), ...)`.
# - **Microring resonator**: `ring(center=(0, 0), radius=5.0e-6, width=0.43e-6, n=256)`,
#   generating a $256$-vertex `PolySlab` annulus with outer radius $5.215\;\mu\text{m}$
#   and inner radius $4.785\;\mu\text{m}$.
# - **Four modal ports** at $x = \pm 5.0\;\mu\text{m}$:
#   - `in` at $(-5.0\;\mu\text{m}, +y_{\text{bus}}, 0)$, $+x$ inward direction (driven port),
#   - `through` at $(+5.0\;\mu\text{m}, +y_{\text{bus}}, 0)$, $-x$ inward direction,
#   - `drop` at $(-5.0\;\mu\text{m}, -y_{\text{bus}}, 0)$, $+x$ inward direction (backward drop wave),
#   - `add` at $(+5.0\;\mu\text{m}, -y_{\text{bus}}, 0)$, $-x$ inward direction (isolated port).
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Accurate time-domain simulation of a microring add-drop filter requires obeying
# two distinct spatial discretization rules simultaneously:
#
# 1. **Propagating-wave resolution ($\text{cpw} \ge 10$)**: The background grid must
#    place at least $10\;\text{cells}$ per shortest wavelength inside the highest-index
#    guided medium ($n_{\text{eff,slab}} = 2.8478$), i.e.
#    $\Delta x_{\text{bg}} \le \lambda_{\min} / (n_{\text{eff,slab}}\,\text{cpw}) = 52.5\;\text{nm}$
#    at $10\;\text{cpw}$. Because the ring round-trip circumference $L = 2\pi R = 31.42\;\mu\text{m}$
#    spans $\sim 50$ guided wavelengths, coarser grids suffer severe numerical phase-velocity
#    dispersion ($\delta n_{\text{eff}} \propto \Delta x^2$) that shifts every resonance by several nanometres.
# 2. **Geometric-feature resolution ($N \ge 4$ cells across the smallest feature)**:
#    The smallest geometric feature of the add-drop filter is the pair of bus-to-ring
#    evanescent coupling gaps ($g = 100\;\text{nm}$ nominal; $g \in \{60, 100, 150\}\;\text{nm}$
#    in the gap sweep). On a uniform $10\;\text{cpw}$ grid ($\Delta y = 52.5\;\text{nm}$),
#    those gaps span only $1.14$, $1.90$, and $2.86$ cells — far too few to resolve the
#    steeply decaying evanescent tail between the curved ring wall and straight bus wall.
#
# To satisfy both rules without paying the cost of a uniform fine grid across the entire
# $12.6 \times 13.8\;\mu\text{m}^2$ domain, we pass two local refinement boxes via
# `mesh_overrides=[(lo, hi, max_cell), ...]` on `PICSimulation`:
# one box at the upper coupling point $(x \in [-0.8, +0.8]\;\mu\text{m},\; y \in [r_{\text{out}}, r_{\text{out}} + g])$
# and one at the lower coupling point $(x \in [-0.8, +0.8]\;\mu\text{m},\; y \in [-(r_{\text{out}} + g), -r_{\text{out}}])$,
# with per-axis cap `max_cell = (1.0, g / N, 1.0)`. Because the tensor-product mesh
# snaps nodes to the gap edges $y = \pm r_{\text{out}}$ and $y = \pm(r_{\text{out}} + g)$,
# places $N$ uniform cells of height $\Delta y_{\text{min}} = g / N$ across each gap, and
# grades back to $\Delta y_{\text{bg}}$ at ratio $\le 1.4$, the total cell count grows by
# only $3\text{–}5\%$ while guaranteeing $N \ge 4$ cells across every coupling gap.

# %%
def build_ring_add_drop(
    cells_per_wavelength=10,
    n_cells_across_gap=4,
    subpixel_smoothing=True,
    gap_m=GAP_DEFAULT,
    radius_m=R_RING,
    width_m=W_WG,
    n_freqs=N_FREQS,
    record_movie=False,
    movie_frames=50,
    cfln=0.95,
    build_solver=True,
):
    """
    Build a 2D effective-index `RingPICSimulation` for the microring add-drop filter.

    Parameters
    ----------
    cells_per_wavelength : float, optional
        Background grid resolution in cells per wavelength inside the slab effective index.
    n_cells_across_gap : int or None, optional
        Minimum number of cells `N` across each bus-ring coupling gap `gap_m`
        enforced via `mesh_overrides`. Pass `None` to disable mesh overrides
        (for the under-resolved demonstration pair).
    subpixel_smoothing : bool, optional
        Enable Kottke anisotropic subpixel permittivity smoothing.
    gap_m : float, optional
        Edge-to-edge coupling gap `g` [m] between the ring and each bus waveguide.
    radius_m : float, optional
        Mean radius `R` [m] of the microring resonator.
    width_m : float, optional
        Waveguide width `w_wg` [m] of the buses and ring.
    n_freqs : int, optional
        Number of frequency points across `[1.50, 1.60] um`.
    record_movie : bool, optional
        Record 2D midplane `Ey` snapshots for the field animation.
    movie_frames : int, optional
        Maximum number of movie frames to store.
    cfln : float, optional
        3D Courant factor (default `0.95`, equal to `0.775` of the 2D Courant limit).
    build_solver : bool, optional
        If `False`, build only the tensor-product mesh (`sim.grid`) without
        allocating `SolverFIT3D` or computing subpixel fills.

    Returns
    -------
    RingPICSimulation
        Configured and meshed simulation instance ready for `run_s_parameters()`.
    """
    y_bus = float(radius_m + width_m + gap_m)
    x_half = max(0.5 * LX_DOMAIN, radius_m + 1.3e-6)
    x_port = min(X_PORT, x_half - 1.2e-6)
    y_half = y_bus + 0.5 * width_m + 1.15e-6
    z0, z1 = -0.5 * THICKNESS_SI, 0.5 * THICKNESS_SI
    eps_si = n_to_eps(N_SI)

    top_bus = PolySlab(
        straight((-x_half, y_bus), (x_half, y_bus), width=width_m),
        z0,
        z1,
        material=eps_si,
    )
    bot_bus = PolySlab(
        straight((-x_half, -y_bus), (x_half, -y_bus), width=width_m),
        z0,
        z1,
        material=eps_si,
    )
    ring_outer, ring_holes = ring(
        center=(0.0, 0.0),
        radius=radius_m,
        width=width_m,
        n=256,
    )
    ring_slab = FastRingPolySlab(
        ring_outer,
        z0,
        z1,
        material=eps_si,
        holes=ring_holes,
        center=(0.0, 0.0),
        r_in=radius_m - 0.5 * width_m,
        r_out=radius_m + 0.5 * width_m,
    )

    port_size = (0.0, PORT_WIDTH, 6.0 * THICKNESS_SI)
    ports = [
        Port(
            name="in",
            center=(-x_port, y_bus, 0.0),
            size=port_size,
            direction=+1,
            mode_index=0,
        ),
        Port(
            name="through",
            center=(x_port, y_bus, 0.0),
            size=port_size,
            direction=-1,
            mode_index=0,
        ),
        Port(
            name="drop",
            center=(-x_port, -y_bus, 0.0),
            size=port_size,
            direction=+1,
            mode_index=0,
        ),
        Port(
            name="add",
            center=(x_port, -y_bus, 0.0),
            size=port_size,
            direction=-1,
            mode_index=0,
        ),
    ]

    overrides = coupler_gap_mesh_overrides(
        gap_m=gap_m,
        n_cells_across_gap=n_cells_across_gap,
        radius_m=radius_m,
        width_m=width_m,
        cells_per_wavelength=cells_per_wavelength,
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness_m=THICKNESS_SI,
        wavelength_min=WL_MIN,
        wavelength_ref=WL_REF,
    )

    return RingPICSimulation(
        structures=[top_bus, bot_bus, ring_slab],
        ports=ports,
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        wavelength_ref=WL_REF,
        n_core=N_SI,
        n_clad=N_SIO2,
        n_freqs=n_freqs,
        cells_per_wavelength=cells_per_wavelength,
        averaging="kottke" if subpixel_smoothing else "staircase",
        two_d=True,
        domain=(-x_half, x_half, -y_half, y_half),
        padding=0.0,
        n_pml=12,
        cfln=cfln,
        mesh_overrides=overrides,
        record_movie=record_movie,
        movie_frames=movie_frames,
        gap_m=gap_m,
        radius_m=radius_m,
        width_m=width_m,
        build_solver=build_solver,
    )


# %% [markdown]
# ## 4. Study Tasks: Convergence Ladder, Under-Resolved Demo & Coupling-Gap Sweep
#
# Each resolution level is a pair $(\text{cpw}, N)$ setting the background
# `cells_per_wavelength = cpw` and coupler-gap override `max_cell = gap / N`:
#
# - **Resolved ladder**: $L_1 = (10, 4)$, $L_2 = (14, 6)$, $L_3 = (20, 8)$, and
#   full reference $R = (40, 16)$.
# - **`--quick` schedule (9 tasks)**: Kottke and staircase at $L_1 = (10, 4)$ and
#   $L_2 = (14, 6)$; independent Kottke reference $T_{\text{ref}}$ at $L_3 = (20, 8)$;
#   coupling-gap sweep $g \in \{60, 150\}\;\text{nm}$ ($g = 100\;\text{nm}$ is `kottke_L2`)
#   at $L_2 = (14, 6)$ with each run's override scaled from its own gap ($\Delta y_{\text{gap}} = g / 6$);
#   plus one clearly labelled under-resolved demonstration pair (`underresolved_kottke`,
#   `underresolved_staircase`) at $10\;\text{cpw}$ with no mesh overrides ($1.90$ cells across
#   the $100\;\text{nm}$ gap).
# - **`--full` schedule (11 tasks)**: Kottke and staircase at $L_1 = (10, 4)$,
#   $L_2 = (14, 6)$, and $L_3 = (20, 8)$; independent Kottke reference at $R = (40, 16)$;
#   coupling-gap sweep at $L_3 = (20, 8)$ ($\Delta y_{\text{gap}} = g / 8$); plus the
#   under-resolved demonstration pair (`underresolved_kottke`, `underresolved_staircase`).

# %%
def study_tasks(quick=False):
    """
    Return the ordered list of study task dictionaries for `--quick` (9 tasks)
    or `--full` (11 tasks).

    Parameters
    ----------
    quick : bool, optional
        If True, use the `--quick` ladder (`L1, L2` + `L3` reference and `L2`
        gap sweep); if False, use the `--full` ladder (`L1, L2, L3` + `R`
        reference and `L3` gap sweep).

    Returns
    -------
    list of dict
        Task descriptors with `index`, `key`, `method`, `level`, `cpw`,
        `n_cells_across_gap`, `subpixel`, `gap_nm`, `is_demo`,
        `collect_fields`, and `record_movie`.
    """
    ladder = QUICK_LADDER if quick else FULL_LADDER
    ref_lvl, ref_cpw, ref_n = QUICK_REF if quick else FULL_REF
    sweep_lvl, sweep_cpw, sweep_n = LADDER_L2 if quick else LADDER_L3
    fine_lvl_name = ladder[-1][0]

    tasks = []
    for lvl_name, cpw, n_gap in ladder:
        for method, sub in (("kottke", True), ("staircase", False)):
            is_top = lvl_name == fine_lvl_name
            tasks.append(
                {
                    "index": len(tasks),
                    "key": f"{method}_{lvl_name}",
                    "method": method,
                    "level": lvl_name,
                    "cpw": int(cpw),
                    "n_cells_across_gap": int(n_gap),
                    "subpixel": bool(sub),
                    "gap_nm": 100.0,
                    "is_demo": False,
                    "collect_fields": bool(is_top and sub),
                    "record_movie": bool(is_top and sub),
                }
            )

    # Independent higher-resolution Kottke reference run (L3 in --quick, R in --full)
    tasks.append(
        {
            "index": len(tasks),
            "key": "kottke_reference",
            "method": "kottke",
            "level": ref_lvl,
            "cpw": int(ref_cpw),
            "n_cells_across_gap": int(ref_n),
            "subpixel": True,
            "gap_nm": 100.0,
            "is_demo": False,
            "collect_fields": False,
            "record_movie": False,
        }
    )

    # Coupling-gap sweep (g = 60 nm and g = 150 nm; g = 100 nm is kottke_L2 in --quick / kottke_L3 in --full)
    for gap_nm in (60.0, 150.0):
        tasks.append(
            {
                "index": len(tasks),
                "key": f"kottke_gap_{int(gap_nm)}nm",
                "method": "kottke",
                "level": sweep_lvl,
                "cpw": int(sweep_cpw),
                "n_cells_across_gap": int(sweep_n),
                "subpixel": True,
                "gap_nm": float(gap_nm),
                "is_demo": False,
                "collect_fields": False,
                "record_movie": False,
            }
        )

    # Under-resolved demonstration pair (10 cpw, no mesh overrides -> ~1.9 cells across 100 nm gap)
    _, demo_cpw, demo_n = UNDERRESOLVED_DEMO
    for method, sub in (("kottke", True), ("staircase", False)):
        tasks.append(
            {
                "index": len(tasks),
                "key": f"underresolved_{method}",
                "method": method,
                "level": "underresolved",
                "cpw": int(demo_cpw),
                "n_cells_across_gap": demo_n,
                "subpixel": bool(sub),
                "gap_nm": 100.0,
                "is_demo": True,
                "collect_fields": False,
                "record_movie": False,
            }
        )
    return tasks


def run_single_task(task_idx, out_dir, quick=False):
    """
    Execute a single study task and write `task_X.npz` and `task_X.json` to `out_dir`.
    """
    tasks = study_tasks(quick=quick)
    if task_idx < 0 or task_idx >= len(tasks):
        raise ValueError(f"Task index {task_idx} out of range 0..{len(tasks) - 1}")
    t = tasks[task_idx]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    sim = build_ring_add_drop(
        cells_per_wavelength=t["cpw"],
        n_cells_across_gap=t["n_cells_across_gap"],
        subpixel_smoothing=t["subpixel"],
        gap_m=t["gap_nm"] * 1e-9,
        record_movie=t["record_movie"],
        movie_frames=50,
    )
    S = sim.run_s_parameters(
        drive="in",
        energy_decay=1e-3,
        max_time=12e-12,
        collect_fields=t["collect_fields"],
    )
    elapsed = time.perf_counter() - t0

    res_info = find_ring_resonances(
        sim.wavelengths,
        S[("through", "in")],
        S[("drop", "in")],
        s_in=S[("in", "in")],
        s_add=S[("add", "in")],
    )

    n_target = -1 if t["n_cells_across_gap"] is None else int(t["n_cells_across_gap"])
    min_feature_nm = float(sim.min_feature_nm)
    dx_background_nm = float(sim.dx_background_nm)
    dx_min_nm = float(sim.dx_min_nm)
    cells_across_feature = float(sim.cells_across_feature)
    n_cells = int(sim.n_cells)
    n_steps = int(sim.solver.Nt)

    payload = {
        "key": np.array(t["key"]),
        "method": np.array(t["method"]),
        "level": np.array(t["level"]),
        "cpw": np.int64(t["cpw"]),
        "n_cells_across_gap": np.int64(n_target),
        "is_demo": np.bool_(t["is_demo"]),
        "subpixel": np.bool_(t["subpixel"]),
        "gap_nm": np.float64(t["gap_nm"]),
        "min_feature_nm": np.float64(min_feature_nm),
        "dx_background_nm": np.float64(dx_background_nm),
        "dx_min_nm": np.float64(dx_min_nm),
        "cells_across_feature": np.float64(cells_across_feature),
        "n_cells": np.int64(n_cells),
        "n_steps": np.int64(n_steps),
        "dx": np.float64(sim.dx),
        "Nx": np.int64(sim.grid.Nx),
        "Ny": np.int64(sim.grid.Ny),
        "Nz": np.int64(sim.grid.Nz),
        "Nt": np.int64(n_steps),
        "dt": np.float64(sim.solver.dt),
        "sim_time_ps": np.float64(sim.sim_time_s * 1e12),
        "final_energy_decay": np.float64(sim.final_energy_decay),
        "elapsed_s": np.float64(elapsed),
        "n_core_eff": np.float64(sim.n_core_eff),
        "wavelengths": sim.wavelengths,
        "frequencies": sim.frequencies,
        "S_through_in": S[("through", "in")],
        "S_drop_in": S[("drop", "in")],
        "S_in_in": S[("in", "in")],
        "S_add_in": S[("add", "in")],
    }
    if sim.flux_balance:
        for fk, fv in sim.flux_balance.items():
            payload[f"flux_{fk}"] = fv
    if sim.dft_on_resonance is not None:
        on_d = sim.dft_on_resonance
        payload["dft_x"] = on_d["x"]
        payload["dft_y"] = on_d["y"]
        payload["dft_on_wl_m"] = np.float64(on_d["wavelength"])
        payload["dft_on_Ex"] = on_d["Ex"]
        payload["dft_on_Ey"] = on_d["Ey"]
        payload["dft_on_Hz"] = on_d["Hz"]
        payload["dft_on_E2"] = on_d["E2"]
    if sim.dft_off_resonance is not None:
        off_d = sim.dft_off_resonance
        payload["dft_off_wl_m"] = np.float64(off_d["wavelength"])
        payload["dft_off_Ex"] = off_d["Ex"]
        payload["dft_off_Ey"] = off_d["Ey"]
        payload["dft_off_Hz"] = off_d["Hz"]
        payload["dft_off_E2"] = off_d["E2"]
    if sim.port_time_trace is not None:
        payload["trace_t"] = sim.port_time_trace["t"]
        payload["trace_Ey"] = sim.port_time_trace["Ey"]
        payload["trace_Ey_drop"] = sim.port_time_trace["Ey_drop"]
    if sim.movie_data is not None:
        payload["movie_x"] = sim.movie_data["x"]
        payload["movie_y"] = sim.movie_data["y"]
        payload["movie_times"] = sim.movie_data["times"]
        payload["movie_frames"] = sim.movie_data["frames"]
        payload["movie_peak"] = sim.movie_data["peak"]

    stem = f"task_{task_idx}"
    np.savez_compressed(out_dir / f"{stem}.npz", **payload)

    meta = {
        **t,
        "min_feature_nm": min_feature_nm,
        "dx_background_nm": dx_background_nm,
        "dx_min_nm": dx_min_nm,
        "cells_across_feature": cells_across_feature,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dx_nm": dx_background_nm,
        "Nx": int(sim.grid.Nx),
        "Ny": int(sim.grid.Ny),
        "Nt": n_steps,
        "sim_time_ps": float(sim.sim_time_s * 1e12),
        "final_energy_decay": float(sim.final_energy_decay),
        "elapsed_s": float(elapsed),
        "n_resonances": len(res_info["resonances"]),
        "mean_fsr_nm": float(res_info["mean_fsr_nm"]),
        "mean_Q_loaded": float(res_info["mean_Q_loaded"]),
        "resonances": res_info["resonances"],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task {task_idx:02d} {t['key']}] cpw={t['cpw']} N_target={t['n_cells_across_gap']} "
        f"min_feature={min_feature_nm:.0f}nm dx_bg={dx_background_nm:.2f}nm dx_min={dx_min_nm:.2f}nm "
        f"cells_across_feature={cells_across_feature:.2f} n_cells={n_cells} ({sim.grid.Nx}x{sim.grid.Ny}) "
        f"n_steps={n_steps} ({sim.sim_time_s * 1e12:.2f} ps, decay={sim.final_energy_decay:.2e}) "
        f"n_res={len(res_info['resonances'])} FSR={res_info['mean_fsr_nm']:.2f}nm "
        f"Q_mean={res_info['mean_Q_loaded']:.0f} in {elapsed:.1f}s",
        flush=True,
    )
    return out_dir / f"{stem}.npz"


def load_study_results(data_dir):
    """Load all `task_*.npz` files from `data_dir` into a dict keyed by task `key`."""
    data_dir = Path(data_dir)
    results = {}
    npz_paths = sorted(
        data_dir.glob("task_*.npz"),
        key=lambda p: (
            int(p.stem.split("_")[1])
            if len(p.stem.split("_")) > 1 and p.stem.split("_")[1].isdigit()
            else 9999
        ),
    )
    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=False) as d:
            entry = {k: d[k] for k in d.files}
        key = str(entry["key"])
        results[key] = entry
    return results


# %% [markdown]
# ## 5. Diagnostic Figures & Physical Analysis
#
# In the sections below we generate **14 diagnostic figures** examining the
# geometry, mode confinement, coupler-gap mesh refinement and subpixel tensor
# averaging across the $100\;\text{nm}$ gap, steady-state on/off-resonance field
# distributions, broadband 4-port spectra, Yariv CMT fits, FSR/dispersion analysis,
# coupling-gap tuning, convergence of resonance shift (in pm) and loaded $Q_L$,
# and cavity ring-down dynamics.

# %%
def _draw_ring_guides(ax, gap_m=GAP_DEFAULT, color="k", lw=0.8, alpha=0.65):
    """Draw analytical outlines of the two bus waveguides and microring on `ax` (in um)."""
    y_bus = (R_RING + W_WG + gap_m) * 1e6
    w_um = W_WG * 1e6
    r_in = (R_RING - 0.5 * W_WG) * 1e6
    r_out = (R_RING + 0.5 * W_WG) * 1e6
    for sgn in (+1.0, -1.0):
        ax.add_patch(
            Rectangle(
                (-6.3, sgn * y_bus - 0.5 * w_um),
                12.6,
                w_um,
                fill=False,
                edgecolor=color,
                lw=lw,
                alpha=alpha,
            )
        )
    for r_c in (r_in, r_out):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                r_c,
                fill=False,
                edgecolor=color,
                lw=lw,
                alpha=alpha,
            )
        )


def compute_resonance_shift_metrics(res_task, res_ref):
    """
    Compare resonance wavelengths and loaded quality factors of `res_task`
    against the independent reference `res_ref`.

    Parameters
    ----------
    res_task : dict
        Output of `find_ring_resonances(...)` for a study task.
    res_ref : dict
        Output of `find_ring_resonances(...)` for `kottke_reference`.

    Returns
    -------
    dict
        Dictionary with `shift_m49_pm`, `abs_shift_m49_pm`, `mean_abs_shift_pm`,
        `fsr_err_pm`, `q_m49`, `q_mean`, and `q_rel_err_pct`.
    """
    ref_wls = [r["wl0_nm"] for r in res_ref["resonances"]]
    task_res = res_task["resonances"]
    task_wls = [r["wl0_nm"] for r in task_res]

    shifts_pm = []
    for rw, tw in zip(ref_wls, task_wls):
        if -4.0 <= (tw - rw) <= 18.0:
            shifts_pm.append((tw - rw) * 1000.0)
    if not shifts_pm:
        for rw in ref_wls:
            if task_wls:
                closest = min(task_wls, key=lambda w: abs(w - rw))
                if abs(closest - rw) < 12.0:
                    shifts_pm.append((closest - rw) * 1000.0)

    # Match the m=49 resonance (3rd azimuthal mode in [1505, 1595] nm, around 1557-1572 nm)
    r_ref_m49 = (
        res_ref["resonances"][2]
        if len(res_ref["resonances"]) >= 3
        else (
            min(res_ref["resonances"], key=lambda r: abs(r["wl0_nm"] - 1560.0))
            if res_ref["resonances"]
            else None
        )
    )
    r_task_m49 = (
        task_res[2]
        if len(task_res) >= 3
        else (
            min(task_res, key=lambda r: abs(r["wl0_nm"] - 1564.0))
            if task_res
            else None
        )
    )
    if r_ref_m49 is not None and r_task_m49 is not None:
        shift_m49_pm = float((r_task_m49["wl0_nm"] - r_ref_m49["wl0_nm"]) * 1000.0)
        q_m49 = float(r_task_m49["Q_loaded"])
    else:
        shift_m49_pm = np.nan
        q_m49 = np.nan

    q_mean = float(res_task["mean_Q_loaded"])
    q_ref_mean = float(res_ref["mean_Q_loaded"])
    q_rel_err_pct = (
        100.0 * abs(q_mean - q_ref_mean) / max(q_ref_mean, 1e-12)
        if np.isfinite(q_mean) and q_ref_mean > 0
        else np.nan
    )
    fsr_err_pm = (
        float(abs(res_task["mean_fsr_nm"] - res_ref["mean_fsr_nm"]) * 1000.0)
        if len(task_res) >= 2 and len(res_ref["resonances"]) >= 2
        else np.nan
    )
    return {
        "shift_m49_pm": shift_m49_pm,
        "abs_shift_m49_pm": abs(shift_m49_pm) if np.isfinite(shift_m49_pm) else np.nan,
        "mean_abs_shift_pm": float(np.mean(np.abs(shifts_pm))) if shifts_pm else np.nan,
        "fsr_err_pm": fsr_err_pm,
        "q_m49": q_m49,
        "q_mean": q_mean,
        "q_rel_err_pct": q_rel_err_pct,
    }


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR, quick=False):
    """
    Render all 14 tutorial figures into `fig_dir` using the saved study data in `data_dir`.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = load_study_results(data_dir)
    fine_key = (
        "kottke_L3"
        if "kottke_L3" in results
        else ("kottke_L2" if "kottke_L2" in results else "kottke_L1")
    )
    fine_data = results.get(fine_key)

    # Build L1 (Kottke + Staircase), under-resolved demo, and L3=(20,8) level for mesh/mode plots
    sim_under = build_ring_add_drop(
        cells_per_wavelength=10,
        n_cells_across_gap=None,
        subpixel_smoothing=True,
    )
    sim_s_l1 = build_ring_add_drop(
        cells_per_wavelength=LADDER_L1[1],
        n_cells_across_gap=LADDER_L1[2],
        subpixel_smoothing=False,
    )
    sim_k_l1 = build_ring_add_drop(
        cells_per_wavelength=LADDER_L1[1],
        n_cells_across_gap=LADDER_L1[2],
        subpixel_smoothing=True,
    )
    ref_lvl_name, ref_cpw, ref_n = QUICK_REF if quick else LADDER_L3
    sim_k_ref = build_ring_add_drop(
        cells_per_wavelength=ref_cpw,
        n_cells_across_gap=ref_n,
        subpixel_smoothing=True,
    )
    sim_k = sim_k_ref
    disp = compute_waveguide_dispersion(n_pts=51)

    # ------------------------------------------------------------------
    # Fig 01: 2D Device Layout & Port Configuration
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    plot_layout(
        sim_k.structures_3d,
        ports=sim_k.ports,
        domain=(-6.3e-6, 6.3e-6, -6.9e-6, 6.9e-6),
        ax=ax,
        title=(
            f"Microring Resonator Add-Drop Filter Layout "
            f"(R = {R_RING * 1e6:.1f} um, w = {W_WG * 1e6:.2f} um, g = {GAP_DEFAULT * 1e9:.0f} nm)"
        ),
    )
    y_bus_um = (R_RING + W_WG + GAP_DEFAULT) * 1e6
    ax.annotate(
        f"Top coupler gap (mesh override)\ng = {GAP_DEFAULT * 1e9:.0f} nm (N >= 4 cells)",
        xy=(0.0, (R_RING + 0.5 * W_WG) * 1e6),
        xytext=(1.5, 3.7),
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.1},
        fontsize=8.8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#888888", "alpha": 0.9},
    )
    ax.annotate(
        f"Bottom coupler gap (mesh override)\ng = {GAP_DEFAULT * 1e9:.0f} nm (N >= 4 cells)",
        xy=(0.0, -(R_RING + 0.5 * W_WG) * 1e6),
        xytext=(1.5, -4.2),
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.1},
        fontsize=8.8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#888888", "alpha": 0.9},
    )
    ax.plot([0.0, R_RING * 1e6 / np.sqrt(2)], [0.0, R_RING * 1e6 / np.sqrt(2)], "k--", lw=1.1)
    ax.text(
        1.3,
        1.8,
        f"R = {R_RING * 1e6:.1f} um\nL = 2pi R = {2 * np.pi * R_RING * 1e6:.2f} um",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "01_layout.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 02: Vertical SOI Slab Stack (Effective-Index Reduction)
    # ------------------------------------------------------------------
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=THICKNESS_SI,
        wavelength_ref=WL_REF,
        out_path=fig_dir / "02_eim_stack.png",
    )

    # ------------------------------------------------------------------
    # Fig 03: Directional Coupler Geometry & Evanescent Separation Profile
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    x_c = np.linspace(-2.5, 2.5, 401)
    r_out_um = (R_RING + 0.5 * W_WG) * 1e6
    r_in_um = (R_RING - 0.5 * W_WG) * 1e6
    y_ring_top = np.sqrt(np.maximum(r_out_um**2 - x_c**2, 0.0))
    y_ring_bot = np.sqrt(np.maximum(r_in_um**2 - x_c**2, 0.0))
    y_bus_bot = np.full_like(x_c, y_bus_um - 0.5 * W_WG * 1e6)
    y_bus_top = np.full_like(x_c, y_bus_um + 0.5 * W_WG * 1e6)

    ax = axes[0]
    ax.fill_between(x_c, y_bus_bot, y_bus_top, color="#d95f02", alpha=0.55, label="Top bus waveguide (w = 0.43 um)")
    ax.fill_between(x_c, y_ring_bot, y_ring_top, color="#1b9e77", alpha=0.55, label="Ring waveguide (R = 5.0 um)")
    ax.fill_between(x_c, y_ring_top, y_bus_bot, color="#7570b3", alpha=0.18, label="Evanescent gap d(x)")
    ax.annotate(
        "",
        xy=(0.0, y_bus_bot[200]),
        xytext=(0.0, y_ring_top[200]),
        arrowprops={"arrowstyle": "<->", "color": "#b10026", "lw": 1.6},
    )
    ax.text(
        0.12,
        0.5 * (y_bus_bot[200] + y_ring_top[200]) - 0.03,
        "g = 100 nm",
        color="#b10026",
        fontweight="bold",
        fontsize=9.5,
    )
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(4.1, 5.95)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Upper Bus-to-Ring Directional Coupler Region")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    neff_0 = float(np.interp(WL_REF, disp["wl_m"], disp["neff_eim"]))
    gamma_cl = (2.0 * np.pi / WL_REF) * np.sqrt(neff_0**2 - N_SIO2**2)  # [1/m]
    for g_nm, col, ls in zip(GAP_SWEEP_NM, ("#d95f02", "#1f77b4", "#2ca02c"), ("--", "-", "-.")):
        d_x_nm = g_nm + (r_out_um - y_ring_top) * 1e3
        rel_cpl = np.exp(-gamma_cl * (d_x_nm * 1e-9))
        L_eff_um = float(np.trapezoid(rel_cpl / rel_cpl.max(), x_c))
        ax.plot(
            x_c,
            rel_cpl / np.exp(-gamma_cl * (100e-9)),
            color=col,
            ls=ls,
            lw=2.0,
            label=f"g = {g_nm:.0f} nm (L_eff = {L_eff_um:.2f} um)",
        )
    ax.set_xlabel("x along coupler [um]")
    ax.set_ylabel(r"Normalized evanescent overlap $\exp[-\gamma_{\mathrm{cl}}\,d(x)]$")
    ax.set_title(r"Local Evanescent Coupling Profile across Gap $d(x)$")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_coupler_geometry_zoom.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 04: Permittivity & Mesh Zoom on the 100 nm Coupler Gap
    #         (Under-resolved vs L1 Staircase/Kottke vs Reference Level)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 5.6), sharex=True, sharey=True)
    n_min, n_max = N_SIO2, float(sim_k_l1.n_core_eff)
    y_gap_lo_um = (R_RING + 0.5 * W_WG) * 1e6
    y_gap_hi_um = (R_RING + 0.5 * W_WG + GAP_DEFAULT) * 1e6

    panels = [
        (
            axes[0, 0],
            sim_under,
            "y",
            (
                f"(a) Under-resolved Demo (10 cpw, no override)\n"
                f"Kottke n_yy: {sim_under.cells_across_feature:.2f} cells across gap "
                f"(dy = {sim_under.dx_min_nm:.1f} nm)"
            ),
        ),
        (
            axes[0, 1],
            sim_s_l1,
            "y",
            (
                f"(b) L1 = (10 cpw, N = 4) Staircase\n"
                f"n_yy: {sim_s_l1.cells_across_feature:.2f} cells across gap "
                f"(dy_min = {sim_s_l1.dx_min_nm:.1f} nm)"
            ),
        ),
        (
            axes[1, 0],
            sim_k_l1,
            "y",
            (
                f"(c) L1 = (10 cpw, N = 4) Kottke Subpixel\n"
                f"n_yy: {sim_k_l1.cells_across_feature:.2f} cells across gap "
                f"(dy_min = {sim_k_l1.dx_min_nm:.1f} nm)"
            ),
        ),
        (
            axes[1, 1],
            sim_k_ref,
            "y",
            (
                f"(d) {'Reference ' if quick else ''}{ref_lvl_name} = ({ref_cpw} cpw, N = {ref_n}) Kottke Subpixel\n"
                f"n_yy: {sim_k_ref.cells_across_feature:.2f} cells across gap "
                f"(dy_min = {sim_k_ref.dx_min_nm:.1f} nm)"
            ),
        ),
    ]
    im = None
    for ax, sim_obj, comp, title in panels:
        g_obj = sim_obj.grid
        xc = 0.5 * (g_obj.x[:-1] + g_obj.x[1:]) * 1e6
        yc = 0.5 * (g_obj.y[:-1] + g_obj.y[1:]) * 1e6
        iz = sim_obj.solver.Nz // 2
        ieps_3d = sim_obj.solver.ieps.to_matrix(comp)[:, :, iz]
        eps_r = np.where(
            ieps_3d > 0,
            1.0 / (np.where(ieps_3d > 0, ieps_3d, 1.0) * epsilon_0),
            N_SIO2**2,
        )
        n_eff_2d = np.sqrt(np.maximum(eps_r, 1.0))
        im = ax.pcolormesh(
            xc,
            yc,
            n_eff_2d.T,
            cmap="viridis",
            vmin=n_min,
            vmax=n_max,
            shading="nearest",
        )
        # Overlay cell edges inside the coupler zoom window
        for xg in g_obj.x * 1e6:
            if -0.68 <= xg <= 0.68:
                ax.axvline(xg, color="w", lw=0.45, alpha=0.48)
        for yg in g_obj.y * 1e6:
            if 5.00 <= yg <= 5.54:
                in_gap = (y_gap_lo_um - 1e-3) <= yg <= (y_gap_hi_um + 1e-3)
                ax.axhline(
                    yg,
                    color="#ffd700" if in_gap else "w",
                    lw=0.75 if in_gap else 0.40,
                    alpha=0.85 if in_gap else 0.42,
                )
        _draw_ring_guides(ax, color="white", lw=1.3, alpha=0.92)
        # Annotate the 100 nm gap feature width at x = 0
        ax.annotate(
            "",
            xy=(0.0, y_gap_hi_um),
            xytext=(0.0, y_gap_lo_um),
            arrowprops={"arrowstyle": "<->", "color": "#ff2222", "lw": 1.8},
        )
        ax.text(
            0.05,
            0.5 * (y_gap_lo_um + y_gap_hi_um),
            (
                f"g = {sim_obj.min_feature_nm:.0f} nm\n"
                f"({sim_obj.cells_across_feature:.1f} cells, "
                f"dy = {sim_obj.dx_min_nm:.1f} nm)"
            ),
            color="white",
            fontsize=8.2,
            fontweight="bold",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#111111", "ec": "#ffd700", "alpha": 0.85},
        )
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(5.04, 5.49)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.8)

    for ax in axes[1, :]:
        ax.set_xlabel("x [um]")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [um]")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="Effective index n_yy = eps_yy^{1/2}",
        shrink=0.82,
        pad=0.02,
    )
    fig.suptitle(
        "Coupler-Gap Mesh Refinement & Subpixel Smoothing Zoom at g = 100 nm "
        "(Cell Edges & Permittivity Tensor)",
        fontsize=11.8,
        y=0.99,
    )
    fig.savefig(fig_dir / "04_mesh_permittivity_zoom.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 05: Port Eigenmode Profiles (Input & Drop Ports)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for ax, p_name, sgn in zip(axes, ("in", "drop"), (+1.0, -1.0)):
        m0 = sim_k.solve_port_modes(p_name)[0]
        y_um = m0.coords["Eb"][0] * 1e6
        ey_line = m0.e["b"][:, m0.e["b"].shape[1] // 2]
        hz_line = m0.h_mean["c"][:, m0.h_mean["c"].shape[1] // 2]
        ey_norm = np.abs(ey_line) / max(float(np.max(np.abs(ey_line))), 1e-30)
        hz_norm = np.abs(hz_line) / max(float(np.max(np.abs(hz_line))), 1e-30)
        yc_port = sgn * y_bus_um
        ax.axvspan(
            yc_port - 0.5 * W_WG * 1e6,
            yc_port + 0.5 * W_WG * 1e6,
            color="#d95f02",
            alpha=0.22,
            label="Bus waveguide core",
        )
        ax.plot(y_um, ey_norm, color="#1f77b4", lw=2.2, label=r"$|E_y| / \max|E_y|$")
        ax.plot(y_um, hz_norm, color="#2ca02c", lw=1.8, ls="--", label=r"$|H_z| / \max|H_z|$")
        ax.set_xlabel("Transverse y [um]")
        ax.set_ylabel("Normalized modal amplitude")
        ax.set_title(
            f"Port '{p_name}' Fundamental TE0 Mode (n_eff = {float(np.real(m0.neff)):.4f})"
        )
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"2D EIM Port Eigenmode Profiles at lambda_0 = {WL_REF * 1e9:.0f} nm "
        f"(w = {W_WG * 1e6:.2f} um)",
        fontsize=11.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "05_port_modes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 06: Waveguide Effective Index, Group Index, and Azimuthal Orders m
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    wl_nm_d = disp["wl_nm"]

    ax = axes[0]
    ax.plot(wl_nm_d, disp["neff_eim"], color="#1f77b4", lw=2.2, label="2D EIM (fixed slab @ 1550 nm)")
    ax.plot(wl_nm_d, disp["neff_3d_eq"], color="#d95f02", lw=2.2, ls="--", label="3D-equiv. (dispersive slab + c-Si)")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"Effective index $n_{\mathrm{eff}}(\lambda)$")
    ax.set_title(r"Channel Mode Effective Index $n_{\mathrm{eff}}(\lambda)$")
    ax.legend(fontsize=8.2)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(wl_nm_d, disp["ng_eim"], color="#1f77b4", lw=2.2, label=f"2D EIM (n_g = {np.mean(disp['ng_eim']):.2f})")
    ax.plot(wl_nm_d, disp["ng_3d_eq"], color="#d95f02", lw=2.2, ls="--", label=f"3D-equiv. (n_g = {np.mean(disp['ng_3d_eq']):.2f})")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"Group index $n_g(\lambda) = n_{\mathrm{eff}} - \lambda\,\mathrm{d}n_{\mathrm{eff}}/\mathrm{d}\lambda$")
    ax.set_title(r"Waveguide Group Index $n_g(\lambda)$")
    ax.legend(fontsize=8.2)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    L_ring = 2.0 * np.pi * R_RING
    m_curve = disp["neff_eim"] * L_ring / disp["wl_m"]
    ax.plot(wl_nm_d, m_curve, color="#1f77b4", lw=2.2, label=r"$n_{\mathrm{eff}}(\lambda)\,L / \lambda$")
    for m_int in range(int(np.ceil(m_curve.min())), int(np.floor(m_curve.max())) + 1):
        wl_m_res = float(np.interp(m_int, m_curve[::-1], wl_nm_d[::-1]))
        ax.axhline(m_int, color="#888888", ls=":", lw=0.9)
        ax.plot(wl_m_res, m_int, "o", color="#d62728", ms=6)
        ax.text(wl_m_res + 1.0, m_int + 0.08, f"m={m_int}\n({wl_m_res:.1f} nm)", fontsize=8)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"Azimuthal mode order $m = n_{\mathrm{eff}} L / \lambda$")
    ax.set_title("Unperturbed Ring Resonance Condition")
    ax.legend(fontsize=8.2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_waveguide_dispersion_ng.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 07: On-Resonance vs Off-Resonance Steady-State DFT Field Maps
    # ------------------------------------------------------------------
    if fine_data is not None and "dft_on_E2" in fine_data:
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.8), sharex=True, sharey=True)
        dx_um = fine_data["dft_x"] * 1e6
        dy_um = fine_data["dft_y"] * 1e6
        wl_on_nm = float(fine_data["dft_on_wl_m"]) * 1e9
        wl_off_nm = float(fine_data["dft_off_wl_m"]) * 1e9

        e2_on = np.asarray(fine_data["dft_on_E2"], dtype=np.float64)
        e2_off = np.asarray(fine_data["dft_off_E2"], dtype=np.float64)
        e2_peak = max(float(np.max(e2_on)), float(np.max(e2_off)), 1e-30)

        hz_raw_on = np.real(fine_data["dft_on_Hz"])
        hz_raw_off = np.real(fine_data["dft_off_Hz"])
        hz_peak = max(float(np.max(np.abs(hz_raw_on))), float(np.max(np.abs(hz_raw_off))), 1e-30)
        hz_on = hz_raw_on / hz_peak
        hz_off = hz_raw_off / hz_peak
        hz_lim = 0.75

        im00 = axes[0, 0].pcolormesh(
            dx_um,
            dy_um,
            np.clip(e2_on / e2_peak, 1e-4, 1.0).T,
            cmap="magma",
            norm=LogNorm(vmin=1e-3, vmax=1.0),
            shading="auto",
        )
        axes[0, 0].set_title(f"ON RESONANCE ({wl_on_nm:.2f} nm): |E|^2 / max|E|^2 (Drop Routing)")
        fig.colorbar(im00, ax=axes[0, 0], label="Normalized |E|^2")

        im01 = axes[0, 1].pcolormesh(
            dx_um,
            dy_um,
            np.clip(e2_off / e2_peak, 1e-4, 1.0).T,
            cmap="magma",
            norm=LogNorm(vmin=1e-3, vmax=1.0),
            shading="auto",
        )
        axes[0, 1].set_title(f"OFF RESONANCE ({wl_off_nm:.2f} nm): |E|^2 / max|E|^2 (Through Bypass)")
        fig.colorbar(im01, ax=axes[0, 1], label="Normalized |E|^2")

        im10 = axes[1, 0].pcolormesh(
            dx_um,
            dy_um,
            hz_on.T,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-hz_lim, vcenter=0.0, vmax=hz_lim),
            shading="auto",
        )
        axes[1, 0].set_title(f"ON RESONANCE ({wl_on_nm:.2f} nm): Re(H_z) Whispering-Gallery Mode")
        fig.colorbar(im10, ax=axes[1, 0], label="Normalized Re(H_z)")

        im11 = axes[1, 1].pcolormesh(
            dx_um,
            dy_um,
            hz_off.T,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-hz_lim, vcenter=0.0, vmax=hz_lim),
            shading="auto",
        )
        axes[1, 1].set_title(f"OFF RESONANCE ({wl_off_nm:.2f} nm): Re(H_z) Bus Propagation")
        fig.colorbar(im11, ax=axes[1, 1], label="Normalized Re(H_z)")

        for ax in axes.ravel():
            _draw_ring_guides(ax, color="cyan", lw=0.85, alpha=0.7)
            ax.set_aspect("equal")
            ax.set_xlim(-6.2, 6.2)
            ax.set_ylim(-6.5, 6.5)
        for ax in axes[1, :]:
            ax.set_xlabel("x [um]")
        for ax in axes[:, 0]:
            ax.set_ylabel("y [um]")

        fig.suptitle(
            "Steady-State Frequency-Domain Fields: On-Resonance Cavity Build-Up vs Off-Resonance Bypass",
            fontsize=12.5,
            y=0.99,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "07_fields_on_vs_off_resonance.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 08: Animated Time-Domain Pulse Propagation & Ring Storage (GIF)
    # ------------------------------------------------------------------
    if fine_data is not None and "movie_frames" in fine_data:
        frames_scaled = (
            np.asarray(fine_data["movie_frames"], dtype=np.float32)
            * float(fine_data["movie_peak"])
        )
        make_movie(
            fine_data["movie_x"],
            fine_data["movie_y"],
            frames_scaled,
            fine_data["movie_times"],
            structures=sim_k.structures_3d,
            out_path=fig_dir / "08_field_movie.gif",
            fps=10,
        )

    # ------------------------------------------------------------------
    # Fig 09: Broadband 4-Port S-Parameter Spectra (Linear & dB + Under-resolved Warning)
    # ------------------------------------------------------------------
    ref_or_fine = results.get("kottke_reference", fine_data)
    if fine_data is not None:
        res_f = find_ring_resonances(
            fine_data["wavelengths"],
            fine_data["S_through_in"],
            fine_data["S_drop_in"],
            s_in=fine_data["S_in_in"],
            s_add=fine_data["S_add_in"],
        )
        wl_nm = res_f["wl_nm"]
        fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), sharex=True)

        ax = axes[0]
        ax.plot(wl_nm, res_f["t_through"], color="#1f77b4", lw=1.8, label=rf"Through $|S_{{\mathrm{{through,in}}}}|^2$ ({fine_key})")
        ax.plot(wl_nm, res_f["t_drop"], color="#d95f02", lw=1.8, label=rf"Drop $|S_{{\mathrm{{drop,in}}}}|^2$ ({fine_key})")
        ax.plot(
            wl_nm,
            res_f["t_through"] + res_f["t_drop"],
            color="#2ca02c",
            lw=1.2,
            ls="--",
            alpha=0.8,
            label=r"Total Guided $|S_{\mathrm{through}}|^2 + |S_{\mathrm{drop}}|^2$",
        )
        if "underresolved_staircase" in results:
            du_s = results["underresolved_staircase"]
            ru_s = find_ring_resonances(du_s["wavelengths"], du_s["S_through_in"], du_s["S_drop_in"])
            ax.plot(
                ru_s["wl_nm"],
                ru_s["t_drop"],
                color="#b10026",
                lw=1.35,
                ls=":",
                alpha=0.85,
                label="WARNING: underresolved_staircase (10 cpw, 1.9 cells/gap — only 3 peaks!)",
            )
        for r in res_f["resonances"]:
            ax.plot(r["wl0_nm"], r["t_drop_peak"], "v", color="#b10026", ms=6)
            ax.annotate(
                f"{r['wl0_nm']:.2f} nm\n(Q={r['Q_loaded']:.0f})",
                xy=(r["wl0_nm"], r["t_drop_peak"]),
                xytext=(r["wl0_nm"] - 3.6, min(1.06, r["t_drop_peak"] + 0.06)),
                fontsize=8,
                fontweight="bold",
            )
        ax.axhline(1.0, color="#888888", ls=":", lw=0.8)
        ax.set_ylabel("Power transmission")
        ax.set_ylim(-0.03, 1.18)
        cells_f = float(fine_data.get("cells_across_feature", 6.0))
        ax.set_title(
            f"Broadband Add-Drop Filter Spectra ({fine_key}: cpw = {int(fine_data['cpw'])}, "
            f"{cells_f:.0f} cells/gap, g = {float(fine_data['gap_nm']):.0f} nm, FSR = {res_f['mean_fsr_nm']:.2f} nm)"
        )
        ax.legend(loc="center right", fontsize=8.3)
        ax.grid(True, alpha=0.25)

        ax = axes[1]
        eps_db = 1e-6
        ax.plot(
            wl_nm,
            10.0 * np.log10(np.maximum(res_f["t_through"], eps_db)),
            color="#1f77b4",
            lw=1.7,
            label=r"Through $|S_{\mathrm{through,in}}|^2$ [dB]",
        )
        ax.plot(
            wl_nm,
            10.0 * np.log10(np.maximum(res_f["t_drop"], eps_db)),
            color="#d95f02",
            lw=1.7,
            label=r"Drop $|S_{\mathrm{drop,in}}|^2$ [dB]",
        )
        ax.plot(
            wl_nm,
            10.0 * np.log10(np.maximum(res_f["t_add"], eps_db)),
            color="#7570b3",
            lw=1.1,
            alpha=0.8,
            label=r"Add isolation $|S_{\mathrm{add,in}}|^2$ [dB]",
        )
        ax.plot(
            wl_nm,
            10.0 * np.log10(np.maximum(res_f["t_in"], eps_db)),
            color="#555555",
            lw=1.1,
            ls=":",
            alpha=0.8,
            label=r"Return loss $|S_{\mathrm{in,in}}|^2$ [dB]",
        )
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_ylim(-48.0, 2.5)
        ax.set_xlim(1500.0, 1600.0)
        ax.legend(loc="lower right", ncol=2, fontsize=8.5)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(fig_dir / "09_sparameters_broadband.png", dpi=160)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Fig 10: High-Resolution Resonance Zoom + Lorentzian & Yariv CMT Fit
        # ------------------------------------------------------------------
        if res_f["resonances"]:
            r_target = min(res_f["resonances"], key=lambda r: abs(r["wl0_nm"] - 1564.0))
            wl0 = r_target["wl0_nm"]
            fwhm = r_target["fwhm_nm"]
            win = (wl_nm >= wl0 - 3.6) & (wl_nm <= wl0 + 3.6)
            wl_zoom = wl_nm[win]
            wl_dense = np.linspace(wl0 - 3.6, wl0 + 3.6, 601)

            neff_dense = np.interp(wl_dense, disp["wl_nm"], disp["neff_eim"])
            neff_at_wl0 = float(np.interp(wl0, disp["wl_nm"], disp["neff_eim"]))
            L_ring = 2.0 * np.pi * R_RING
            phi_0 = 2.0 * np.pi * neff_at_wl0 * L_ring / (wl0 * 1e-9)
            phase_off = 2.0 * np.pi * round(phi_0 / (2.0 * np.pi)) - phi_0
            t_thru_cmt, t_drop_cmt = cmt_add_drop_spectra(
                wl_dense * 1e-9,
                neff_dense,
                R_RING,
                kappa2=r_target["kappa2_cmt"],
                a_rt=r_target["a_rt_cmt"],
                phase_offset_rad=phase_off,
            )
            lor_dense = _lorentzian_peak(
                wl_dense,
                wl0,
                fwhm,
                r_target["fit_amp"],
                r_target["fit_bg"],
            )

            fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
            ax = axes[0]
            ax.plot(wl_zoom, res_f["t_through"][win], "o", color="#1f77b4", ms=4.5, alpha=0.8, label="Wakis FIT Through")
            ax.plot(wl_zoom, res_f["t_drop"][win], "s", color="#d95f02", ms=4.5, alpha=0.8, label="Wakis FIT Drop")
            ax.plot(wl_dense, t_thru_cmt, "-", color="#084594", lw=2.0, label="Yariv CMT Through")
            ax.plot(wl_dense, t_drop_cmt, "-", color="#b10026", lw=2.0, label="Yariv CMT Drop")
            ax.plot(wl_dense, lor_dense, "--", color="#111111", lw=1.4, label="Lorentzian Fit")

            half_h = 0.5 * r_target["t_drop_peak"]
            ax.annotate(
                "",
                xy=(wl0 - 0.5 * fwhm, half_h),
                xytext=(wl0 + 0.5 * fwhm, half_h),
                arrowprops={"arrowstyle": "<->", "color": "#111111", "lw": 1.5},
            )
            ax.text(
                wl0,
                half_h + 0.05,
                f"FWHM = {fwhm:.3f} nm\nQ_L = {r_target['Q_loaded']:.0f}",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#aaaaaa", "alpha": 0.9},
            )
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Power transmission")
            ax.set_ylim(-0.03, 1.08)
            ax.set_title(f"Linear Zoom Around lambda_res = {wl0:.2f} nm")
            ax.legend(loc="center right", fontsize=8.0)
            ax.grid(True, alpha=0.25)

            ax = axes[1]
            ax.plot(
                wl_zoom,
                10.0 * np.log10(np.maximum(res_f["t_through"][win], 1e-6)),
                "o-",
                color="#1f77b4",
                ms=4.0,
                lw=1.5,
                label="Wakis FIT Through [dB]",
            )
            ax.plot(
                wl_zoom,
                10.0 * np.log10(np.maximum(res_f["t_drop"][win], 1e-6)),
                "s-",
                color="#d95f02",
                ms=4.0,
                lw=1.5,
                label="Wakis FIT Drop [dB]",
            )
            ax.plot(
                wl_dense,
                10.0 * np.log10(np.maximum(t_thru_cmt, 1e-6)),
                "--",
                color="#084594",
                lw=1.8,
                label="Yariv CMT Through [dB]",
            )
            ax.plot(
                wl_dense,
                10.0 * np.log10(np.maximum(t_drop_cmt, 1e-6)),
                "--",
                color="#b10026",
                lw=1.8,
                label="Yariv CMT Drop [dB]",
            )
            info_txt = (
                f"lambda_0 = {wl0:.3f} nm\n"
                f"Delta lambda_FWHM = {fwhm:.3f} nm\n"
                f"Loaded Q_L = {r_target['Q_loaded']:.0f}\n"
                f"Finesse F = {r_target['finesse']:.1f}\n"
                f"Drop peak = {r_target['drop_db']:.2f} dB ({r_target['t_drop_peak'] * 100:.1f}%)\n"
                f"Through extinction = {r_target['extinction_db']:.1f} dB\n"
                f"CMT kappa^2 = {r_target['kappa2_cmt']:.3f}, a_rt = {r_target['a_rt_cmt']:.4f}"
            )
            ax.text(
                0.04,
                0.06,
                info_txt,
                transform=ax.transAxes,
                fontsize=8.4,
                bbox={"boxstyle": "round,pad=0.35", "fc": "#f8f9fa", "ec": "#888888", "alpha": 0.94},
            )
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Transmission [dB]")
            ax.set_ylim(-32.0, 2.0)
            ax.set_title("Logarithmic (dB) Extinction & Insertion-Loss View")
            ax.legend(loc="upper right", fontsize=8.0)
            ax.grid(True, alpha=0.25)

            fig.suptitle(
                "Single-Resonance Lineshape Analysis: Wakis 2D FIT vs Yariv (2000) Coupled-Mode Theory",
                fontsize=12.0,
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / "10_resonance_zoom_cmt_fit.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Fig 11: Free Spectral Range (FSR) vs lambda^2/(n_g L) & Quality Factor Q_L
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
        ax = axes[0]
        ax.plot(
            disp["wl_nm"],
            disp["fsr_eim_nm"],
            color="#1f77b4",
            lw=2.2,
            label=r"2D EIM theory $\lambda^2 / (n_{g,\mathrm{EIM}} L)$",
        )
        ax.plot(
            disp["wl_nm"],
            disp["fsr_3d_eq_nm"],
            color="#7570b3",
            lw=2.0,
            ls="--",
            label=r"3D dispersive SOI $\lambda^2 / (n_{g,\mathrm{3D}} L)$ (~18.5 nm)",
        )
        fsr_curve_specs = [("kottke_L1", "#ff7f0e", "^")]
        if "kottke_L2" in results and fine_key != "kottke_L2":
            fsr_curve_specs.append(("kottke_L2", "#9467bd", "D"))
        fsr_curve_specs.extend(
            [
                (fine_key, "#d95f02", "o"),
                ("kottke_reference", "#2ca02c", "s"),
            ]
        )
        for key_label, col, marker in fsr_curve_specs:
            if key_label in results:
                rk = find_ring_resonances(
                    results[key_label]["wavelengths"],
                    results[key_label]["S_through_in"],
                    results[key_label]["S_drop_in"],
                )
                if len(rk["resonances"]) >= 2:
                    wls_r = np.array([r["wl0_nm"] for r in rk["resonances"]])
                    mid_wls = 0.5 * (wls_r[:-1] + wls_r[1:])
                    fsrs = np.diff(wls_r)
                    cpw_lbl = int(results[key_label]["cpw"])
                    cells_lbl = float(results[key_label].get("cells_across_feature", 0.0))
                    ax.plot(
                        mid_wls,
                        fsrs,
                        marker=marker,
                        color=col,
                        ms=6.5,
                        lw=1.6,
                        label=f"Wakis FIT ({key_label}, {cpw_lbl} cpw, {cells_lbl:.0f} cells/gap)",
                    )
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"Free Spectral Range $\Delta\lambda_{\mathrm{FSR}}$ [nm]")
        ax.set_ylim(16.0, 27.0)
        ax.set_title("Free Spectral Range vs Group-Index Theory")
        ax.legend(fontsize=8.0)
        ax.grid(True, alpha=0.25)

        ax = axes[1]
        ax2 = ax.twinx()
        wls_f = [r["wl0_nm"] for r in res_f["resonances"]]
        qs_f = [r["Q_loaded"] for r in res_f["resonances"]]
        kappas_f = [r["kappa2_cmt"] for r in res_f["resonances"]]
        l1 = ax.plot(
            wls_f,
            qs_f,
            "o-",
            color="#1f77b4",
            ms=7,
            lw=2.0,
            label=rf"Loaded $Q_L$ ({fine_key}, {int(fine_data['cpw'])} cpw)",
        )
        if ref_or_fine is not None and ref_or_fine is not fine_data:
            res_ref = find_ring_resonances(
                ref_or_fine["wavelengths"],
                ref_or_fine["S_through_in"],
                ref_or_fine["S_drop_in"],
            )
            l_ref = ax.plot(
                [r["wl0_nm"] for r in res_ref["resonances"]],
                [r["Q_loaded"] for r in res_ref["resonances"]],
                "^--",
                color="#2ca02c",
                ms=6.5,
                lw=1.6,
                label=f"Reference Q_L ({int(ref_or_fine['cpw'])} cpw)",
            )
            l1 = l1 + l_ref
        l2 = ax2.plot(
            wls_f,
            kappas_f,
            "s:",
            color="#d95f02",
            ms=6.5,
            lw=1.8,
            label=r"Coupler power fraction $\kappa^2(\lambda)$",
        )
        ax.set_xlabel("Resonance wavelength [nm]")
        ax.set_ylabel(r"Loaded quality factor $Q_L$", color="#1f77b4")
        ax2.set_ylabel(r"Per-coupler power coupling $\kappa^2$", color="#d95f02")
        ax.set_title(r"Wavelength Dependence of $Q_L$ and Evanescent Coupling $\kappa^2$")
        lines = l1 + l2
        ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8.0)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(fig_dir / "11_fsr_and_q_analysis.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 12: Coupling-Gap Sweep (g = 60, 100, 150 nm) — Over- vs Under-Coupling
    # ------------------------------------------------------------------
    gap_keys = [
        (60.0, "kottke_gap_60nm", "#d95f02"),
        (100.0, fine_key, "#1f77b4"),
        (150.0, "kottke_gap_150nm", "#2ca02c"),
    ]
    avail_gaps = [(g, k, c) for g, k, c in gap_keys if k in results]
    if avail_gaps:
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
        gaps_list, q_list, fwhm_list, kappa_list, ext_list, drop_db_list = [], [], [], [], [], []

        for g_nm, k_name, col in avail_gaps:
            d = results[k_name]
            rk = find_ring_resonances(
                d["wavelengths"],
                d["S_through_in"],
                d["S_drop_in"],
            )
            if not rk["resonances"]:
                continue
            r_mid = min(rk["resonances"], key=lambda r: abs(r["wl0_nm"] - 1564.0))
            wl0 = r_mid["wl0_nm"]
            detuning = rk["wl_nm"] - wl0
            win = np.abs(detuning) <= 3.0
            dy_min_nm = float(d.get("dx_min_nm", g_nm / 6.0))
            cells_g = float(d.get("cells_across_feature", 6.0))

            axes[0].plot(
                detuning[win],
                rk["t_drop"][win],
                color=col,
                lw=2.1,
                label=f"g={g_nm:.0f} nm ({cells_g:.0f} cells, dy={dy_min_nm:.1f} nm, FWHM={r_mid['fwhm_nm']:.2f} nm)",
            )
            axes[0].plot(
                detuning[win],
                rk["t_through"][win],
                color=col,
                lw=1.3,
                ls="--",
                alpha=0.75,
            )

            gaps_list.append(g_nm)
            q_list.append(r_mid["Q_loaded"])
            fwhm_list.append(r_mid["fwhm_nm"])
            kappa_list.append(r_mid["kappa2_cmt"])
            ext_list.append(r_mid["extinction_db"])
            drop_db_list.append(r_mid["drop_db"])

        axes[0].set_xlabel(r"Detuning $\lambda - \lambda_{\mathrm{res}}$ [nm]")
        axes[0].set_ylabel("Power transmission (solid: Drop, dashed: Through)")
        axes[0].set_ylim(-0.03, 1.05)
        axes[0].set_title("Resonance Lineshape vs Coupling Gap g")
        axes[0].legend(fontsize=7.6)
        axes[0].grid(True, alpha=0.25)

        ax = axes[1]
        ax_r = ax.twinx()
        l1 = ax.plot(gaps_list, q_list, "o-", color="#1f77b4", ms=7, lw=2.1, label=r"Loaded $Q_L$")
        l2 = ax_r.plot(gaps_list, fwhm_list, "s--", color="#d95f02", ms=7, lw=2.0, label=r"Linewidth $\Delta\lambda_{\mathrm{FWHM}}$ [nm]")
        ax.set_xlabel("Edge-to-edge coupling gap g [nm]")
        ax.set_ylabel(r"Loaded quality factor $Q_L$", color="#1f77b4")
        ax_r.set_ylabel(r"Linewidth $\Delta\lambda_{\mathrm{FWHM}}$ [nm]", color="#d95f02")
        ax.set_title(r"$Q_L$ and Linewidth Scaling with Gap $g$")
        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="center right", fontsize=8.2)
        ax.grid(True, alpha=0.25)

        ax = axes[2]
        ax_r2 = ax.twinx()
        l1 = ax.plot(gaps_list, kappa_list, "o-", color="#2ca02c", ms=7, lw=2.1, label=r"Coupler $\kappa^2$")
        l2 = ax_r2.plot(gaps_list, ext_list, "^--", color="#7570b3", ms=7, lw=1.9, label="Through extinction [dB]")
        ax.set_xlabel("Edge-to-edge coupling gap g [nm]")
        ax.set_ylabel(r"Extracted power coupling $\kappa^2$", color="#2ca02c")
        ax_r2.set_ylabel("Through-port extinction ratio [dB]", color="#7570b3")
        ax.set_title("Evanescent Coupling & Extinction vs Gap g")
        lns = l1 + l2
        ax.legend(lns, [l.get_label() for l in lns], loc="center right", fontsize=8.2)
        ax.grid(True, alpha=0.25)

        n_gap_sweep = int(round(float(fine_data.get("cells_across_feature", 6.0)))) if fine_data is not None else 6
        cpw_sweep = int(fine_data.get("cpw", 14)) if fine_data is not None else 14
        fig.suptitle(
            f"Coupling-Gap Parametric Sweep ({cpw_sweep} cpw, Each Gap Resolved with N = {n_gap_sweep} Override Cells)",
            fontsize=12.0,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "12_coupling_gap_sweep.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 13: Grid Convergence vs Independent Kottke Reference (T_ref)
    #         Resonance Shift [pm], Loaded Q_L, and Under-Resolved Warning Pair
    # ------------------------------------------------------------------
    if "kottke_reference" in results:
        ref_d = results["kottke_reference"]
        ref_res = find_ring_resonances(
            ref_d["wavelengths"], ref_d["S_through_in"], ref_d["S_drop_in"]
        )
        ref_cpw_val = int(ref_d["cpw"])
        ref_cells_val = float(ref_d.get("cells_across_feature", 8.0))
        q_ref_mean = float(ref_res["mean_Q_loaded"])

        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0))

        # Panel (a): Drop-port resonance comb across 1554-1595 nm comparing Reference, fine level, L2, and Under-resolved
        ax = axes[0]
        comb_curves = [
            (
                "kottke_reference",
                f"Kottke Ref ({ref_cpw_val} cpw, {ref_cells_val:.0f} cells/gap)",
                "#111111",
                "-",
                2.0,
            ),
            (
                fine_key,
                f"Kottke {fine_key.split('_')[-1]} ({int(results[fine_key]['cpw'])} cpw, {float(results[fine_key].get('cells_across_feature', 6.0)):.0f} cells/gap)",
                "#1f77b4",
                "--",
                1.9,
            ),
            (
                f"staircase_{fine_key.split('_')[-1]}",
                f"Staircase {fine_key.split('_')[-1]} ({int(results[fine_key]['cpw'])} cpw, {float(results[fine_key].get('cells_across_feature', 6.0)):.0f} cells/gap)",
                "#d95f02",
                "-.",
                1.8,
            ),
            (
                "underresolved_kottke",
                "WARNING: underresolved_kottke (10 cpw, 1.9 cells)",
                "#2ca02c",
                ":",
                1.5,
            ),
            (
                "underresolved_staircase",
                "WARNING: underresolved_staircase (10 cpw, 1.9 cells)",
                "#b10026",
                ":",
                1.6,
            ),
        ]
        for key_n, lbl, col, ls, lw_c in comb_curves:
            if key_n in results:
                dk = results[key_n]
                rk = find_ring_resonances(dk["wavelengths"], dk["S_through_in"], dk["S_drop_in"])
                win = (rk["wl_nm"] >= 1554.0) & (rk["wl_nm"] <= 1595.0)
                ax.plot(rk["wl_nm"][win], rk["t_drop"][win], color=col, ls=ls, lw=lw_c, label=lbl)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"Drop transmission $|S_{\mathrm{drop,in}}|^2$")
        ax.set_title("Drop Spectrum: Resolved Ladder vs Under-Resolved Gap")
        ax.legend(fontsize=7.3, loc="upper right")
        ax.grid(True, alpha=0.25)

        # Collect resolved ladder (L1, L2, [L3]) and under-resolved demo points
        lvl_names = ["L1", "L2", "L3"]
        cpw_ticks, cells_ticks = [], []
        kottke_cpws, kottke_shifts_pm = [], []
        all_shifts_pm = []

        for method, col, marker, label in (
            ("kottke", "#1f77b4", "o", "Kottke subpixel (resolved N>=4)"),
            ("staircase", "#d95f02", "s", "Staircase (resolved N>=4)"),
        ):
            cpw_vals, cells_vals, shift_pm_vals, q_vals = [], [], [], []
            for lvl in lvl_names:
                k_name = f"{method}_{lvl}"
                if k_name not in results:
                    continue
                dk = results[k_name]
                rk = find_ring_resonances(
                    dk["wavelengths"],
                    dk["S_through_in"],
                    dk["S_drop_in"],
                    s_in=dk["S_in_in"],
                )
                m_metrics = compute_resonance_shift_metrics(rk, ref_res)
                cpw_v = float(dk["cpw"])
                cells_v = float(dk.get("cells_across_feature", dk.get("n_cells_across_gap", 4.0)))
                cpw_vals.append(cpw_v)
                cells_vals.append(cells_v)
                shift_pm_vals.append(m_metrics["mean_abs_shift_pm"])
                all_shifts_pm.append(m_metrics["mean_abs_shift_pm"])
                q_vals.append(m_metrics["q_mean"])
                if method == "kottke":
                    cpw_ticks.append(cpw_v)
                    cells_ticks.append(cells_v)

            if method == "kottke":
                kottke_cpws = cpw_vals
                kottke_shifts_pm = shift_pm_vals

            axes[1].plot(
                cpw_vals,
                shift_pm_vals,
                marker=marker,
                color=col,
                lw=2.1,
                ms=7.5,
                label=label,
            )
            axes[2].plot(
                cpw_vals,
                q_vals,
                marker=marker,
                color=col,
                lw=2.1,
                ms=7.5,
                label=label,
            )

            # Label every point with its cells across the smallest feature and metric value
            for c_v, n_v, sh_v, q_v in zip(cpw_vals, cells_vals, shift_pm_vals, q_vals):
                if method == "kottke" and c_v <= 10.5:
                    x_off_sh, y_off_sh = 0.20, 320.0
                    x_off_q, y_off_q = 0.22, -38.0
                elif method == "kottke" and c_v <= 15.0:
                    x_off_sh, y_off_sh = 0.20, -680.0
                    x_off_q, y_off_q = -0.90, -48.0
                elif method == "kottke":
                    x_off_sh, y_off_sh = -3.7, -420.0
                    x_off_q, y_off_q = -3.4, -46.0
                elif c_v <= 10.5:
                    x_off_sh, y_off_sh = 0.20, -1450.0
                    x_off_q, y_off_q = 0.22, -38.0
                elif c_v <= 15.0:
                    x_off_sh, y_off_sh = 0.20, 360.0
                    x_off_q, y_off_q = 0.22, -34.0
                else:
                    x_off_sh, y_off_sh = -3.9, 460.0
                    x_off_q, y_off_q = -3.8, 22.0
                axes[1].annotate(
                    f"{n_v:.0f} cells/gap ({sh_v:.0f} pm)",
                    xy=(c_v, sh_v),
                    xytext=(c_v + x_off_sh, sh_v + y_off_sh),
                    fontsize=7.3,
                    color=col,
                    fontweight="bold",
                )
                axes[2].annotate(
                    f"{n_v:.0f} cells/gap (Q={q_v:.0f})",
                    xy=(c_v, q_v),
                    xytext=(c_v + x_off_q, q_v + y_off_q),
                    fontsize=7.3,
                    color=col,
                    fontweight="bold",
                )

        # Plot under-resolved demonstration pair (excluded from converged curve, clearly labelled as warning)
        for u_key, u_col, u_marker, u_lbl, y_sh_off, y_q_off in (
            (
                "underresolved_kottke",
                "#2ca02c",
                "D",
                "WARNING: underresolved_kottke (1.9 cells/gap)",
                -1150.0,
                84.0,
            ),
            (
                "underresolved_staircase",
                "#b10026",
                "X",
                "WARNING: underresolved_staircase (1.9 cells/gap)",
                280.0,
                -42.0,
            ),
        ):
            if u_key in results:
                du = results[u_key]
                ru = find_ring_resonances(du["wavelengths"], du["S_through_in"], du["S_drop_in"])
                um = compute_resonance_shift_metrics(ru, ref_res)
                u_cpw = float(du["cpw"])
                u_cells = float(du.get("cells_across_feature", 1.90))
                all_shifts_pm.append(um["mean_abs_shift_pm"])
                axes[1].plot(
                    [u_cpw],
                    [um["mean_abs_shift_pm"]],
                    marker=u_marker,
                    color=u_col,
                    ms=9.0,
                    linestyle="None",
                    label=u_lbl,
                )
                axes[1].annotate(
                    f"Under-resolved ({u_cells:.1f} cells): {um['mean_abs_shift_pm']:.0f} pm",
                    xy=(u_cpw, um["mean_abs_shift_pm"]),
                    xytext=(u_cpw + 0.22, um["mean_abs_shift_pm"] + y_sh_off),
                    fontsize=7.2,
                    color=u_col,
                    fontweight="bold",
                )
                axes[2].plot(
                    [u_cpw],
                    [um["q_mean"]],
                    marker=u_marker,
                    color=u_col,
                    ms=9.0,
                    linestyle="None",
                    label=u_lbl,
                )
                axes[2].annotate(
                    f"Under-resolved ({u_cells:.1f} cells): Q={um['q_mean']:.0f}",
                    xy=(u_cpw, um["q_mean"]),
                    xytext=(u_cpw + 0.22, um["q_mean"] + y_q_off),
                    fontsize=7.2,
                    color=u_col,
                    fontweight="bold",
                )

        if len(kottke_cpws) >= 2 and np.isfinite(kottke_shifts_pm[0]):
            c_arr = np.linspace(min(kottke_cpws), max(kottke_cpws), 50)
            ref_slope = kottke_shifts_pm[0] * (kottke_cpws[0] / c_arr) ** 2
            axes[1].plot(c_arr, ref_slope, ":", color="#444444", lw=1.5, label=r"$\mathcal{O}(\mathrm{cpw}^{-2})$ slope")

        axes[2].axhline(
            q_ref_mean,
            color="#111111",
            ls="--",
            lw=1.5,
            label=f"Kottke Ref Q_L = {q_ref_mean:.0f} ({ref_cpw_val} cpw, {ref_cells_val:.0f} cells/gap)",
        )
        sh_min = min(all_shifts_pm) if all_shifts_pm else 1800.0
        sh_max = max(all_shifts_pm) if all_shifts_pm else 15000.0
        axes[1].set_ylim(max(500.0, sh_min - 900.0), sh_max + 4200.0)
        axes[2].set_ylim(350.0, 1180.0)

        # Add secondary top x-axis showing cells_across_feature next to cpw on panels (b) and (c)
        for ax_conv in (axes[1], axes[2]):
            ax_conv.set_xlim(9.3, max(cpw_ticks + [14.8]) + 1.8)
            ax_top = ax_conv.twiny()
            ax_top.set_xlim(ax_conv.get_xlim())
            ax_top.set_xticks(cpw_ticks)
            ax_top.set_xticklabels([f"{c_n:.0f} cells\n(L{i + 1})" for i, c_n in enumerate(cells_ticks)], fontsize=8.2)
            ax_top.set_xlabel("Cells across 100 nm coupler gap (mesh_overrides)", fontsize=8.8)

        axes[1].set_xlabel("Background resolution [cells / wavelength in slab]")
        axes[1].set_ylabel(r"Mean resonance shift $|\Delta\lambda_{\mathrm{res}}|$ vs $T_{\mathrm{ref}}$ [pm]")
        axes[1].set_title(f"Resonance Shift vs {ref_cpw_val} cpw Reference", pad=28)
        axes[1].legend(fontsize=7.0, loc="upper right")
        axes[1].grid(True, which="both", alpha=0.25)

        axes[2].set_xlabel("Background resolution [cells / wavelength in slab]")
        axes[2].set_ylabel(r"Mean loaded quality factor $Q_L$")
        axes[2].set_title(f"Loaded Quality Factor $Q_L$ vs {ref_cpw_val} cpw Reference", pad=28)
        axes[2].legend(fontsize=7.0, loc="upper right")
        axes[2].grid(True, which="both", alpha=0.25)

        fig.suptitle(
            f"Grid Convergence vs Independent Kottke Reference ({ref_cpw_val} cpw, {ref_cells_val:.0f} cells/gap): "
            "Resonance Shift [pm] & Loaded Q_L",
            fontsize=11.8,
            y=1.04,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "13_kottke_convergence.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 14: Dual-Port Time Traces & Cavity Ring-Down Decay Envelope
    # ------------------------------------------------------------------
    if fine_data is not None and "trace_t" in fine_data:
        t_ps = np.asarray(fine_data["trace_t"], dtype=np.float64) * 1e12
        ey_thru = np.asarray(fine_data["trace_Ey"], dtype=np.float64)
        ey_drop = np.asarray(fine_data["trace_Ey_drop"], dtype=np.float64)
        peak_e = max(float(np.max(np.abs(ey_thru))), float(np.max(np.abs(ey_drop))), 1e-30)
        ey_thru_n = ey_thru / peak_e
        ey_drop_n = ey_drop / peak_e

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
        ax = axes[0]
        t_zoom = t_ps <= min(4.5, float(t_ps[-1]))
        ax.plot(t_ps[t_zoom], ey_thru_n[t_zoom], color="#1f77b4", lw=1.1, label="Through port Ey(t)")
        ax.plot(t_ps[t_zoom], ey_drop_n[t_zoom], color="#d95f02", lw=1.1, alpha=0.9, label="Drop port Ey(t)")
        ax.set_xlabel("Time [ps]")
        ax.set_ylabel("Normalized field Ey(t) / max|Ey|")
        ax.set_title("Initial Ballistic Transit & Round-Trip Pulse Train")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.25)

        ax = axes[1]
        win_len = max(5, int(round(0.035 / max(float(t_ps[1] - t_ps[0]), 1e-4))))
        kernel = np.ones(win_len) / win_len
        env_thru = np.sqrt(np.convolve(ey_thru_n**2, kernel, mode="same") + 1e-16)
        env_drop = np.sqrt(np.convolve(ey_drop_n**2, kernel, mode="same") + 1e-16)
        ax.semilogy(t_ps, env_thru, color="#1f77b4", lw=1.5, label="Through envelope |Ey(t)|")
        ax.semilogy(t_ps, env_drop, color="#d95f02", lw=1.5, label="Drop envelope |Ey(t)|")

        res_f = find_ring_resonances(
            fine_data["wavelengths"], fine_data["S_through_in"], fine_data["S_drop_in"]
        )
        if res_f["mean_Q_loaded"] > 0:
            q_mean = res_f["mean_Q_loaded"]
            omega_0 = 2.0 * np.pi * c_light / WL_REF
            tau_p_ps = (q_mean / omega_0) * 1e12  # photon energy lifetime [ps]
            t_tail = np.linspace(1.0, float(t_ps[-1]), 200)
            win_1ps = (t_ps >= 0.80) & (t_ps <= 1.20)
            amp_0 = float(np.max(env_drop[win_1ps])) if np.any(win_1ps) else float(np.interp(1.0, t_ps, env_drop))
            decay_curve = amp_0 * np.exp(-(t_tail - 1.0) / (2.0 * tau_p_ps))
            ax.semilogy(
                t_tail,
                decay_curve,
                "k--",
                lw=1.8,
                label=f"Theory exp(-t / 2tau_p), tau_p = {tau_p_ps:.2f} ps (Q_L = {q_mean:.0f})",
            )

        ax.set_xlabel("Time [ps]")
        ax.set_ylabel("Normalized field envelope (log scale)")
        ax.set_ylim(1e-4, 2.0)
        ax.set_title("Cavity Ring-Down Decay & Photon Lifetime")
        ax.legend(loc="upper right", fontsize=8.2)
        ax.grid(True, which="both", alpha=0.25)

        fig.tight_layout()
        fig.savefig(fig_dir / "14_time_traces_ringdown.png", dpi=160)
        plt.close(fig)


# %% [markdown]
# ## 6. Resolution Table, Convergence Results & Limits/Failure Modes
#
# ### `--full` Study Resolution & Convergence Table (11 Tasks)
#
# Every run records `min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
# `cells_across_feature` (measured directly from the tensor-product mesh
# `sim.grid.y` inside the coupling gap), `n_cells`, `n_steps`, the mean
# resonance wavelength shift $|\Delta\lambda_{\text{res}}|$ in picometres [pm]
# across the azimuthal modes $m \in \{51, 50, 49, 48\}$ relative to the
# independent `kottke_reference` ($R = (40\;\text{cpw}, N = 16)$, with the
# earlier `--quick` shift vs. $L_3 = (20\;\text{cpw}, N = 8)$ shown in parentheses),
# the mean free spectral range $\Delta\lambda_{\text{FSR}}$, and the mean loaded
# quality factor $Q_L$:
#
# | Task | Key | $\text{cpw}$ | $N_{\text{target}}$ | $g$ [nm] | $\Delta x_{\text{bg}}$ [nm] | $\Delta x_{\min}$ [nm] | Cells across feature | $n_{\text{cells}}$ ($N_x \times N_y$) | $n_{\text{steps}}$ ($T_{\text{sim}}$) | $|\Delta\lambda_{\text{res}}|$ vs. $R$ (`--quick` vs. $L_3$) [pm] | $\Delta\lambda_{\text{FSR}}$ [nm] | Loaded $Q_L$ ($m=49$) | Wall time [s] |
# | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | 0 | `kottke_L1` | 10 | 4 | 100 | 52.38 | 25.00 | 4.00 | 78,470 ($266 \times 295$) | 74,751 ($4.90\;\text{ps}$) | 10,615 (8,822) | 23.41 | 659 (599) | 190.1 |
# | 1 | `staircase_L1` | 10 | 4 | 100 | 52.38 | 25.00 | 4.00 | 78,470 ($266 \times 295$) | 64,701 ($4.24\;\text{ps}$) | 13,466 (11,619) | 23.20 (3 peaks) | 585 (530) | 186.3 |
# | 2 | `kottke_L2` | 14 | 6 | 100 | 37.41 | 16.67 | 6.00 | 145,524 ($362 \times 402$) | 114,801 ($5.13\;\text{ps}$) | 4,663 (2,871) | 23.58 | 752 (698) | 389.5 |
# | 3 | `staircase_L2` | 14 | 6 | 100 | 37.41 | 16.67 | 6.00 | 145,524 ($362 \times 402$) | 124,801 ($5.58\;\text{ps}$) | 7,454 (5,661) | 23.56 | 450 (400) | 602.1 |
# | 4 | `kottke_L3` | 20 | 8 | 100 | 26.32 | 12.50 | 8.00 | 282,744 ($504 \times 561$) | 155,451 ($5.11\;\text{ps}$) | 1,793 (0, quick ref) | 23.71 | 760 (720) | 1,757.1 |
# | 5 | `staircase_L3` | 20 | 8 | 100 | 26.32 | 12.50 | 8.00 | 282,744 ($504 \times 561$) | 181,451 ($5.96\;\text{ps}$) | 3,006 (1,213) | 23.58 | 853 (816) | 1,692.2 |
# | 6 | `kottke_reference` ($R$) | 40 | 16 | 100 | 13.16 | 6.25 | 16.00 | 1,075,402 ($983 \times 1094$) | 310,051 ($5.10\;\text{ps}$) | 0 (ref) | 23.80 | 762 (734) | 16,397.4 |
# | 7 | `kottke_gap_60nm` ($L_3$) | 20 | 8 | 60 | 26.32 | 7.50 | 8.00 | 284,760 ($504 \times 565$) | 138,701 ($3.06\;\text{ps}$) | — | 23.70 | 359 (331) | 1,355.9 |
# | 8 | `kottke_gap_150nm` ($L_3$) | 20 | 8 | 150 | 26.32 | 18.75 | 8.00 | 282,240 ($504 \times 560$) | 218,851 ($9.15\;\text{ps}$) | — | 23.68 | 1,656 (1,657) | 2,093.4 |
# | 9 | `underresolved_kottke` (demo) | 10 | None | 100 | 52.50 | 52.50 | **1.90** | 76,055 ($265 \times 287$) | 52,701 ($5.07\;\text{ps}$) | 10,341 (8,548) | 23.38 | 700 (693) | 118.0 |
# | 10 | `underresolved_staircase` (demo) | 10 | None | 100 | 52.50 | 52.50 | **1.90** | 76,055 ($265 \times 287$) | 55,551 ($5.35\;\text{ps}$) | 14,835 (12,988) | 22.51 (3 peaks) | **996** (714) | 137.7 |
#
# *(In `--quick` mode, the 9-task schedule runs $L_1 = (10, 4)$ and $L_2 = (14, 6)$, uses `kottke_L3` at $(20\;\text{cpw}, N = 8)$ as `kottke_reference`, and evaluates the $g \in \{60, 150\}\;\text{nm}$ gap sweep at $L_2 = (14\;\text{cpw}, N = 6)$ where $\Delta x_{\min} = 10.00$ and $25.00\;\text{nm}$, $\Delta\lambda_{\text{FSR}} = 23.56$ and $23.57\;\text{nm}$, and mean $Q_L = 355$ and $1601$.)*
#
# ### Convergence Verdict & Resolution Audit ($L_1 \to L_2 \to L_3$ vs. Reference $R = (40, 16)$)
#
# 1. **Resolution audit across all non-demo tasks**: Every non-demo task (`task_0` through
#    `task_8`, `is_demo = False`) satisfies both spatial discretization rules with zero violations:
#    - Background resolution inside the slab effective index ($n_{\text{eff,slab}} = 2.8478$):
#      $\text{cpw} \in \{10, 14, 20, 40\} \ge 10\;\text{cells}/\lambda$.
#    - Smallest-feature resolution across the evanescent coupler gap ($g \in \{60, 100, 150\}\;\text{nm}$):
#      $\text{cells\_across\_feature} \in \{4.00, 6.00, 8.00, 16.00\} \ge 4.00$ cells.
#    - Only the two explicit failure-mode demonstration tasks (`task_9` `underresolved_kottke` and
#      `task_10` `underresolved_staircase`, `is_demo = True`) omit `mesh_overrides` and have
#      $\text{cells\_across\_feature} = 1.90 < 4$.
# 2. **Convergence at the accepted full level $L_3 = (20\;\text{cpw}, N = 8)$**:
#    - **Loaded quality factor $Q_L$**: Kottke converges monotonically across the ladder
#      ($Q_L = 658.8 \to 752.5 \to 760.2$ vs. reference $R = 762.0$). The relative change
#      between the last two ladder levels ($L_2 \to L_3$) is **$1.02\%$**, and the residual
#      discrepancy between $L_3$ and the $40\;\text{cpw}$ reference $R$ is **$0.24\%$**.
#    - **Free spectral range $\Delta\lambda_{\text{FSR}}$**: Kottke converges monotonically
#      ($23.412 \to 23.578 \to 23.706\;\text{nm}$ vs. $R = 23.803\;\text{nm}$). The relative
#      change from $L_2 \to L_3$ is **$0.54\%$** ($128.3\;\text{pm}$), and $L_3$ is within
#      **$0.41\%$** ($97.3\;\text{pm}$) of $R$.
#    - **Drop-port peak transmission $T_{\text{drop,peak}}$**: Mean peak transmission across
#      the 4 resonances converges from $0.962$ ($L_1$) $\to 0.976$ ($L_2$) $\to 0.980$ ($L_3$)
#      vs. $0.981$ ($R$), a **$0.38\%$** relative change from $L_2 \to L_3$ ($0.14\%$ vs. $R$).
#    - **Resonance wavelength $\lambda_m$ and shift $|\Delta\lambda_{\text{res}}|$**: Because
#      azimuthal order $m \sim 50$ accumulates phase over $50$ guided wavelengths per round trip,
#      absolute resonance wavelengths are the most grid-sensitive observable. Kottke reduces the
#      mean shift relative to $R = (40, 16)$ from $10,615\;\text{pm}$ ($L_1$) $\to 4,663\;\text{pm}$
#      ($L_2$, a $2.28\times$ reduction) $\to 1,793\;\text{pm}$ ($L_3$, a further $2.60\times$
#      reduction; $5.92\times$ total from $L_1 \to L_3$, faster than $\mathcal{O}(\text{cpw}^{-2})$).
#      Between $L_2$ and $L_3$, $\lambda_m$ shifts by $2,871\;\text{pm}$ (**$0.185\%$** of
#      $\lambda_0$), and at $L_3$ the remaining offset vs. $R$ is $1,793\;\text{pm}$ (**$0.115\%$**
#      of $\lambda_0$, $7.5\%$ of $\Delta\lambda_{\text{FSR}}$, and less than one resonance
#      linewidth $\Delta\lambda_{\text{FWHM}} = 2.08\;\text{nm}$).
#
# ### Coupling-Gap Sweep ($g \in \{60, 100, 150\}\;\text{nm}$) at Full Resolution ($L_3$) vs. Yariv CMT
#
# Evaluating each coupling gap with $N = 8$ override cells across its own gap at $L_3 = (20\;\text{cpw}, N = 8)$
# (and comparing against the earlier `--quick` $L_2 = (14\;\text{cpw}, N = 6)$ sweep and Yariv 2000 analytical CMT
# evaluated at the central $m = 49$ resonance $\lambda_{49} \approx 1559\;\text{nm}$) confirms exponential
# evanescent scaling of $\kappa^2(g)$ and quantitative agreement with transfer-matrix theory:
#
# | Gap $g$ [nm] | $\Delta y_{\min}$ ($L_3$) [nm] | Extracted $\kappa^2$ ($m=49$ / mean) | $\Delta\lambda_{\text{FSR}}$ ($L_3$ / `--quick` $L_2$ / EIM theory) [nm] | $\Delta\lambda_{\text{FWHM}}$ ($m=49$: $L_3$ / `--quick` $L_2$ / CMT) [nm] | Loaded $Q_L$ (mean: $L_3$ / `--quick` $L_2$) | Loaded $Q_L$ ($m=49$: $L_3$ / `--quick` $L_2$ / CMT) | Finesse $\mathcal{F}$ ($m=49$, $L_3$) | Drop Peak $T_{\text{drop}}$ ($m=49$ / 4-peak range) | Through Extinction ($m=49$ / 4-peak range) [dB] |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | **60** | 7.50 ($8.00$ cells) | $0.452$ / $0.433$ | $23.70$ / $23.56$ / $23.17$ | $4.720$ / $4.859$ / $4.715$ | $359$ / $355$ | $331$ / $322$ / $331$ | $5.1$ | $-0.02\;\text{dB}$ ($99.6\%$; $96.9\text{–}102.6\%$) | $43.4$ ($30.1\text{–}43.4$) |
# | **100** | 12.50 ($8.00$ cells) | $0.243$ / $0.235$ | $23.71$ / $23.58$ / $23.17$ | $2.167$ / $2.237$ / $2.153$ | $760$ / $752$ | $720$ / $698$ / $724$ | $11.2$ | $-0.01\;\text{dB}$ ($99.7\%$; $95.1\text{–}100.7\%$) | $27.9$ ($27.1\text{–}28.8$) |
# | **150** | 18.75 ($8.00$ cells) | $0.114$ / $0.114$ | $23.68$ / $23.57$ / $23.17$ | $0.941$ / $0.974$ / $0.934$ | $1,656$ / $1,601$ | $1,657$ / $1,603$ / $1,670$ | $25.9$ | $-0.05\;\text{dB}$ ($98.8\%$; $87.6\text{–}98.8\%$) | $26.7$ ($18.3\text{–}34.3$) |
#
# ### Comparison: Old (`--quick` $L_2$) vs. New (`--full` $L_3$) vs. Reference ($R$ & 2D EIM Theory)
#
# | Metric ($g = 100\;\text{nm}$, $R = 5.0\;\mu\text{m}$, $w_{\text{wg}} = 0.43\;\mu\text{m}$) | Old (`--quick` `kottke_L2`, $14\;\text{cpw}, N=6$) | New (`--full` `kottke_L3`, $20\;\text{cpw}, N=8$) | 2D FIT Reference (`kottke_reference`, $40\;\text{cpw}, N=16$) | Analytical 2D EIM / Yariv CMT |
# | :--- | :---: | :---: | :---: | :---: |
# | Grid $\Delta x_{\text{bg}}$ / $\Delta y_{\min}$ [nm] | $37.41$ / $16.67$ ($6.00$ cells/gap) | $26.32$ / $12.50$ ($8.00$ cells/gap) | $13.16$ / $6.25$ ($16.00$ cells/gap) | Continuum 1D/2D slab |
# | Effective / group index $n_{\text{eff}}(\lambda_0)$ / $n_g(\lambda_0)$ | $2.3906$ ($2.4008$ EIM) / $3.3005$ | $2.3886$ ($2.4029$ EIM) / $3.3005$ | $2.4029$ / $3.3005$ | $2.4029$ / $3.3005$ (2D EIM) |
# | Resonances in $1500\text{–}1600\;\text{nm}$ | 4 ($m = 51, 50, 49, 48$) | 4 ($m = 51, 50, 49, 48$) | 4 ($m = 51, 50, 49, 48$) | 4 ($m = 51, 50, 49, 48$) |
# | Mean Free Spectral Range $\Delta\lambda_{\text{FSR}}$ [nm] | $23.58$ | $23.71$ | $23.80$ | $23.17$ at $1550\;\text{nm}$ ($22.5\text{–}24.5$ across band) |
# | Central resonance $\lambda_{49}$ [nm] | $1562.22$ | $1559.39$ | $1557.63$ | $1558.5$ (unperturbed ring) |
# | Mean resonance shift $|\Delta\lambda_{\text{res}}|$ vs. $R$ [pm] | $4,663$ ($2,871$ vs. $L_3$) | $1,793$ | $0$ (ref) | — |
# | Mean loaded quality factor $Q_L$ ($m=49$ $Q_L$) | $752$ ($698$) | $760$ ($720$) | $762$ ($734$) | $724$ (CMT at $m=49$, $\kappa^2=0.243$) |
# | Linewidth $\Delta\lambda_{\text{FWHM}}$ at $m=49$ [nm] | $2.237$ | $2.167$ | $2.122$ | $2.153$ (CMT at $m=49$) |
# | Drop-port peak transmission $T_{\text{drop,peak}}$ | $-0.19\text{ to }+0.02\;\text{dB}$ ($95.8\text{–}100.4\%$) | $-0.22\text{ to }+0.03\;\text{dB}$ ($95.1\text{–}100.7\%$) | $-0.27\text{ to }+0.03\;\text{dB}$ ($94.0\text{–}100.6\%$) | $99.7\%$ ($-0.01\;\text{dB}$ for $a_{\text{rt}}=0.9996$) |
# | Through-port extinction ratio [dB] | $25.0\text{–}32.0$ ($30.2$ at $m=49$) | $27.1\text{–}28.8$ ($27.9$ at $m=49$) | $24.3\text{–}32.5$ ($32.5$ at $m=49$) | $> 27\;\text{dB}$ ($a_{\text{rt}} \to 1$) |
#
# ### Why 2D EIM has $\Delta\lambda_{\text{FSR}} \approx 23.7\;\text{nm}$ vs. $18.5\;\text{nm}$ in 3D
#
# Because $\Delta\lambda_{\text{FSR}} = \lambda^2 / (n_g L)$ is inversely proportional
# to the group index $n_g = n_{\text{eff}} - \lambda\,\mathrm{d}n_{\text{eff}}/\mathrm{d}\lambda$,
# freezing the vertical slab index at $n_{\text{eff,slab}}(1.55\;\mu\text{m}) = 2.8478$
# includes only the horizontal channel confinement dispersion ($n_{g,\text{EIM}} = 3.30$).
# In 3D SOI, the vertical slab confinement dispersion ($\mathrm{d}n_{\text{eff,slab}}/\mathrm{d}\lambda < 0$)
# and bulk silicon dispersion ($\mathrm{d}n_{\text{Si}}/\mathrm{d}\lambda < 0$) increase
# the group index by $4.15 / 3.30 = 1.258\times$, scaling the FSR by
# $23.71\;\text{nm} / 1.258 = 18.8\;\text{nm}$ — in quantitative agreement
# with analytical 3D waveguide dispersion theory ($18.5\;\text{nm}$).
# No geometry or material parameter was tuned to match the 3D FSR; the $1.26\times$
# ratio is an exact physical consequence of the 2D effective-index reduction with a
# fixed reference-wavelength slab index.
#
# ### Limits & Failure Modes: What Goes Wrong When a Gap Is Resolved by Only 1–2 Cells or Without Subpixel Smoothing
#
# Comparing the full ladder (`L1, L2, L3, R`) and the `underresolved_kottke` / `underresolved_staircase`
# demonstration pair ($10\;\text{cpw}$ with no `mesh_overrides`, giving only $1.90$ cells of size
# $\Delta y = 52.5\;\text{nm}$ across the $100\;\text{nm}$ coupling gap) highlights three key physical conclusions:
#
# 1. **Staircased gap-width quantization corrupts $\kappa^2$ and $Q_L$**: When a
#    $100\;\text{nm}$ gap spans $1.90$ cells whose boundaries do not coincide with the
#    bus or ring walls, `underresolved_staircase` snaps each interface cell to pure
#    $\text{Si}$ or pure $\text{SiO}_2$ based on cell centers. At $10\;\text{cpw}$ this
#    artificially widens the effective gap at $x = 0$, reducing the coupling $\kappa^2$
#    ($0.074\text{–}0.209$ vs. $0.208\text{–}0.275$ in `kottke_reference`) and inflating
#    the apparent loaded quality factor to $Q_L = 996$ ($+30.7\%$ above the converged
#    reference $Q_{L,\text{ref}} = 762$), while shifting resonances by $14.84\;\text{nm}$
#    ($14,835\;\text{pm}$ vs. $R$; $12,988\;\text{pm}$ vs. $L_3$), pushing the 4th resonance
#    ($m = 48$) out of the $1500\text{–}1600\;\text{nm}$ band, depressing drop-port peaks to
#    $21.1\text{–}64.7\%$ ($-6.75\text{ to }-1.89\;\text{dB}$), and degrading through-port
#    extinction to $4.4\text{–}15.1\;\text{dB}$.
# 2. **Subpixel smoothing helps, but cannot replace resolving the feature**:
#    `underresolved_kottke` uses Kottke's harmonic/arithmetic permittivity tensor to
#    preserve all 4 resonances, high drop transmission ($94.1\text{–}99.6\%$), and deep
#    extinction ($24.2\text{–}34.4\;\text{dB}$) even at $1.90$ cells across the gap.
#    However, with fewer than $2$ cells between the waveguides, the exponentially decaying
#    evanescent tail $\exp[-\gamma_{\text{cl}} d(x)]$ is sampled by a single interior node,
#    leaving $Q_L = 700$ under-converged ($-8.07\%$ vs. $R = 762$, compared to $-1.24\%$ for
#    `kottke_L2` at $Q_L = 752$ and $-0.24\%$ for `kottke_L3` at $Q_L = 760$). Adding local
#    `mesh_overrides` ($N \ge 4$ cells across the gap) resolves the evanescent coupling
#    region directly while increasing the total cell count by only $3.2\%$ at $L_1$.
# 3. **Why Kottke still outperforms staircase across all resolved levels ($L_1, L_2, L_3$)**:
#    Even when `mesh_overrides` enforces $N = 4, 6, 8$ cells across the gap at $x = 0$, the
#    circular ring perimeter away from $x = 0$ crosses the Cartesian grid at arbitrary angles.
#    Staircase facets along the curved ring and tapered coupler separation $d(x)$ produce both
#    a larger phase-velocity error ($|\Delta\lambda_{\text{res}}| = 13,466 \to 7,454 \to 3,006\;\text{pm}$
#    for staircase vs. $10,615 \to 4,663 \to 1,793\;\text{pm}$ for Kottke at $L_1 \to L_2 \to L_3$)
#    and non-monotonic quality-factor errors ($Q_L = 585 \to 450 \to 853$ for staircase, i.e.
#    $-23.3\%$, $-41.0\%$ from back-scattering doublet broadening at $L_2$, and $+12.0\%$
#    overshoot at $L_3$, vs. monotonic $659 \to 752 \to 760$ for Kottke with $13.5\% \to 1.24\% \to 0.24\%$
#    error relative to $R = 762$). Thus both rules are essential: $N \ge 4$ cells across the
#    smallest feature **and** Kottke subpixel averaging on curved dielectric boundaries.
#
# %%
def print_summary_table(data_dir=DATA_DIR):
    """Print a formatted summary table of all completed tasks in `data_dir`."""
    results = load_study_results(data_dir)
    if not results:
        print("No study results found in", data_dir)
        return
    ref_res = None
    if "kottke_reference" in results:
        ref_d = results["kottke_reference"]
        ref_res = find_ring_resonances(
            ref_d["wavelengths"], ref_d["S_through_in"], ref_d["S_drop_in"]
        )

    header = (
        f"{'Task Key':<24} {'CPW':>4} {'N':>4} {'Feat[nm]':>8} {'dx_bg':>6} "
        f"{'dx_min':>6} {'CellsGap':>8} {'n_cells':>8} {'n_steps':>8} "
        f"{'|dLam|[pm]':>10} {'FSR[nm]':>8} {'Q_mean':>7} {'Time[s]':>8}"
    )
    print(header)
    print("-" * len(header))
    for k, d in results.items():
        rk = find_ring_resonances(d["wavelengths"], d["S_through_in"], d["S_drop_in"])
        shift_pm = np.nan
        if ref_res is not None and float(d["gap_nm"]) == 100.0:
            shift_pm = compute_resonance_shift_metrics(rk, ref_res)["mean_abs_shift_pm"]
        n_tgt = int(d.get("n_cells_across_gap", -1))
        n_str = "None" if n_tgt < 0 else str(n_tgt)
        feat_nm = float(d.get("min_feature_nm", d["gap_nm"]))
        dx_bg = float(d.get("dx_background_nm", float(d["dx"]) * 1e9))
        dx_min = float(d.get("dx_min_nm", dx_bg))
        cells_gap = float(d.get("cells_across_feature", feat_nm / max(dx_min, 1e-9)))
        n_cells = int(d.get("n_cells", int(d["Nx"]) * int(d["Ny"])))
        n_steps = int(d.get("n_steps", int(d["Nt"])))
        sh_str = f"{shift_pm:>10.0f}" if np.isfinite(shift_pm) else f"{'—':>10}"
        print(
            f"{k:<24} {int(d['cpw']):>4d} {n_str:>4} {feat_nm:>8.0f} "
            f"{dx_bg:>6.2f} {dx_min:>6.2f} {cells_gap:>8.2f} "
            f"{n_cells:>8d} {n_steps:>8d} {sh_str} "
            f"{rk['mean_fsr_nm']:>8.2f} {rk['mean_Q_loaded']:>7.0f} "
            f"{float(d['elapsed_s']):>8.1f}"
        )


# %% [markdown]
# ## 7. Suggested Exercises
#
# 1. **Critical coupling with intrinsic ring loss**: Add weak material conductivity
#    or bending radiation loss ($a < 1$) and sweep the upper and lower gaps
#    asymmetrically ($g_1 \neq g_2$) to verify the critical-coupling condition
#    $t_1 = t_2\,a$ where through-port transmission vanishes identically on resonance.
# 2. **Vernier dual-ring filter**: Cascade two rings of slightly different radii
#    ($R_1 = 5.0\;\mu\text{m}$, $R_2 = 5.5\;\mu\text{m}$) between the buses and
#    confirm that the composite FSR expands to
#    $\Delta\lambda_{\text{Vernier}} = \frac{\text{FSR}_1\,\text{FSR}_2}{|\text{FSR}_1 - \text{FSR}_2|}$.
# 3. **Racetrack coupler length sweep**: Replace the point-coupled circle with a
#    straight coupling section of length $L_c \in [1, 6]\;\mu\text{m}$ using two
#    $180^\circ$ `arc` bends and two `straight` segments to observe sinusoidal
#    directional-coupler power oscillation $\kappa^2(L_c) = \sin^2(\pi L_c / 2 L_\pi)$.
#
# ## References
#
# 1. A. Yariv, "Universal relations for coupling of optical power between
#    microresonators and dielectric waveguides," *Electron. Lett.*, vol. 36,
#    no. 4, pp. 321–322, 2000, DOI: [10.1049/el:20000340](https://doi.org/10.1049/el:20000340).
# 2. W. Bogaerts *et al.*, "Silicon microring resonators," *Laser Photonics Rev.*,
#    vol. 6, no. 1, pp. 47–73, 2012, DOI: [10.1002/lpor.201100017](https://doi.org/10.1002/lpor.201100017).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discontinuous dielectrics," *Phys. Rev. E*, vol. 77, 036611, 2008,
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).

# %%
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tutorial 5.10 · Microring add-drop filter (2D Wakis FIT)."
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task by integer index (0..8 in --quick, 0..10 in --full).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for task .npz/.json files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick resolution ladder: L1=(10,4), L2=(14,6), Kottke ref L3=(20,8) + demo.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full resolution ladder: L1=(10,4), L2=(14,6), L3=(20,8), Kottke ref R=(40,16) + demo.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all 14 figures from existing data in --out.",
    )
    args = parser.parse_args(argv)

    if args.list_tasks:
        tasks = study_tasks(quick=bool(args.quick))
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            avg = "kottke" if t["subpixel"] else "staircase"
            desc = (
                f"level={t['level']} cpw={t['cpw']} N={t['n_cells_across_gap']} "
                f"gap={t['gap_nm']:.0f}nm averaging={avg} demo={t['is_demo']}"
            )
            print(f"{t['index']}\t{t['key']}\t{desc}")
        return

    if args.task is not None:
        run_single_task(args.task, args.out, quick=bool(args.quick))
        return

    if args.quick:
        quick = True
    elif args.full:
        quick = False
    else:
        quick = not (args.out / "task_10.npz").exists()
    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR, quick=quick)
        print_summary_table(data_dir=args.out)
        return

    # Default: run all study tasks sequentially, generate all figures, print summary
    tasks = study_tasks(quick=quick)
    for t in tasks:
        run_single_task(t["index"], args.out, quick=quick)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR, quick=quick)
    print_summary_table(data_dir=args.out)


if __name__ == "__main__":
    # When executed inside Jupyter/IPython without CLI flags, display precomputed summary
    if "ipykernel" in sys.modules:
        if DATA_DIR.exists() and list(DATA_DIR.glob("task_*.npz")):
            print_summary_table(DATA_DIR)
    else:
        main()
