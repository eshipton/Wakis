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

# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# %% [markdown]
# # Tutorial 5.5 · Waveguide crossing with cosine tapers
#
# **Abstract.** High-density silicon photonic routing fabrics—such as optical
# network-on-chip (ONoC) crossbar switches, programmable Mach–Zehnder meshes,
# and transponder multiplexers—require orthogonal strip waveguides to intersect
# in the same lithography plane with sub-$0.2\;\text{dB}$ insertion loss and
# $<-30\;\text{dB}$ cross-port crosstalk. When two single-mode silicon strip
# waveguides cross directly at $90^\circ$, lateral guidance vanishes inside the
# intersection box; the tightly confined fundamental $\text{TE}_0$ mode diffracts
# into the orthogonal waveguide arms and unguided cladding radiation, causing
# $\approx 1\;\text{dB}$ insertion loss per crossing. In this tutorial we model the
# ultra-compact O-band silicon-on-insulator (SOI) waveguide crossing of
# **Chandran *et al.* (2020)** [1] in Wakis using the **2D effective-index method
# (EIM, $n_{\text{eff,slab}} = 2.9274$)** and **Kottke subpixel permittivity
# smoothing**. Four symmetric convex cosine-profile multimode tapers excite a
# controlled $\text{TE}_0 + \text{TE}_2$ superposition that self-images to a
# collimated waist ($\text{FWHM} = 0.387\;\mu\text{m}$) at the crossing center,
# suppressing corner diffraction and cross-port scattering. In the full-resolution
# 2D EIM study (`L3`: $20\;\text{cpw}$, $16.27$ cells across the $350\;\text{nm}$
# access waveguide, verified against a $40\;\text{cpw}$ Kottke reference with
# $32.55$ cells across $w_{\text{in}}$), the cosine-tapered crossing achieves
# $T(1310\;\text{nm}) = 0.97958$ ($\text{IL} = 0.0896\;\text{dB}$ at `L3`,
# $0.0945\;\text{dB}$ at `R = 40 cpw`) and cross-port crosstalk
# $|S_{31}|^2 = |S_{41}|^2 = -43.47\;\text{dB}$ ($-43.66\;\text{dB}$ at `R`),
# outperforming a monotonic linear taper ($\text{IL} = 0.6495\;\text{dB}$,
# $\text{XT} = -37.35\;\text{dB}$) by $0.56\;\text{dB}$ and a plain untapered
# crossing ($\text{IL} = 0.9499\;\text{dB}$, $\text{XT} = -14.86\;\text{dB}$) by
# $0.86\;\text{dB}$ in insertion loss and $28.6\;\text{dB}$ in crosstalk
# suppression.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Reduce a $161\;\text{nm}$ O-band ($\lambda \in [1260, 1360]\;\text{nm}$) SOI
#    strip-waveguide crossing to a **2D effective-index (EIM)** finite-integration
#    (`Ex`, `Ey`, `Hz`) model and compute the local lateral $\text{TE}_0$ and
#    $\text{TE}_2$ slab eigenmodes along the taper.
# 2. Construct a $D_4$-symmetric 4-port waveguide crossing as a single seamless
#    counter-clockwise `PolySlab` with convex cosine tapers, linear tapers, or
#    untapered straight arms.
# 3. Extract the broadband $4 \times 4$ scattering matrix $S_{ij}(\lambda)$,
#    compare insertion loss and crosstalk across the three crossing geometries,
#    and audit $C_4$ rotational symmetry, mirror symmetry, Lorentz reciprocity,
#    and 2D Poynting flux conservation.
# 4. Quantify the resolution convergence of **Kottke (2008) subpixel smoothing**
#    against staircasing across `L1 = (10, 4)`, `L2 = (14, 6)`, and `L3 = (20, 8)`
#    relative to an independent `R = (40, 16)` fine-grid Kottke reference, and
#    explain why 2D EIM insertion loss ($\approx 0.09\;\text{dB}$) is lower than
#    3D FDTD ($\approx 0.20\;\text{dB}$).
#
# ## Contents
#
# 1. [Physical background & multimode self-imaging](#1.-Physical-background-&-multimode-self-imaging)
# 2. [Effective-Index Method (EIM) reduction](#2.-Effective-Index-Method-(EIM)-reduction)
# 3. [Parametric geometry construction](#3.-Parametric-geometry-construction)
# 4. [Subpixel permittivity smoothing vs staircasing](#4.-Subpixel-permittivity-smoothing-vs-staircasing)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete figure suite](#6.-Generating-the-complete-figure-suite)
# 7. [Quantitative results & when 2D EIM breaks down](#7.-Quantitative-results-&-when-2D-EIM-breaks-down)
# 8. [Exercises](#8.-Exercises)
# 9. [References](#9.-References)

# %% [markdown]
# ## 1. Physical background & multimode self-imaging
#
# Consider an orthogonal intersection of two single-mode silicon strip waveguides
# of width $w_{\text{in}} = 0.35\;\mu\text{m}$ and thickness
# $h_{\text{Si}} = 0.161\;\mu\text{m}$ operating in the O-band
# ($\lambda_0 = 1.31\;\mu\text{m}$, $\lambda \in [1.26, 1.36]\;\mu\text{m}$).
# In a **plain (untapered) crossing**, the lateral oxide sidewalls disappear over
# the central $w_{\text{in}} \times w_{\text{in}}$ square. A fundamental mode of
# effective half-waist $w_0 \approx w_{\text{in}}/2$ diverges with paraxial
# diffraction half-angle $\theta_{\text{div}} \approx \lambda_0 / (\pi n_{\text{eff,slab}} w_0)$,
# coupling substantial power into the orthogonal $+y$ (`north`) and $-y$ (`south`)
# cross ports ($|S_{31}|^2 = |S_{41}|^2 = -14.86\;\text{dB}$ at $1310\;\text{nm}$
# in 2D EIM) and radiating into the cladding ($10\log_{10}|S_{21}|^2 = -0.9499\;\text{dB}$
# in 2D EIM).
#
# **Chandran *et al.* (2020)** [1] replaced each straight access section near the
# intersection with a symmetric, **non-monotonic convex cosine taper** of length
# $l_t = 5.30\;\mu\text{m}$ connecting the access width $w_{\text{in}} = 0.35\;\mu\text{m}$
# at $|x| = w_{\text{out}}/2 + l_t = 5.85\;\mu\text{m}$ to a central crossing box
# of width $w_{\text{out}} = 1.10\;\mu\text{m}$ at $|x| = w_{\text{out}}/2 = 0.55\;\mu\text{m}$.
# For the $-x$ taper ($x \in [-5.85, -0.55]\;\mu\text{m}$), the upper and lower
# sidewall trajectories follow:
#
# $$y_{\text{edge}}(x) = \pm w_m \cos(a x + b), \qquad w_m = 0.75\;\mu\text{m},$$
#
# where the endpoint width conditions $w_m \cos(-a w_{\text{out}}/2 + b) = w_{\text{out}}/2$
# and $w_m \cos(-a(w_{\text{out}}/2 + l_t) + b) = w_{\text{in}}/2$ on the branch
# passing through the interior peak $w_{\max} = 2 w_m = 1.50\;\mu\text{m}$ at
# $x_{\text{peak}} = -b / a$ yield in closed form (matching `scipy.optimize.fsolve`
# with initial guess $(0.5, 2.0)$ to machine precision):
#
# $$\theta_{\text{out}} = \arccos\!\left(\frac{w_{\text{out}}}{2 w_m}\right), \qquad \theta_{\text{in}} = \arccos\!\left(\frac{w_{\text{in}}}{2 w_m}\right),$$
#
# $$a = \frac{\theta_{\text{in}} + \theta_{\text{out}}}{l_t} = 0.392996\;\mu\text{m}^{-1}, \qquad b = \theta_{\text{out}} + a\,\frac{w_{\text{out}}}{2} = 0.963732\;\text{rad}, \qquad x_{\text{peak}} = -\frac{b}{a} = -2.4523\;\mu\text{m}.$$
#
# Because the taper is symmetric ($y \mapsto -y$), an incident fundamental even
# $\text{TE}_0$ mode excites only even-parity lateral slab modes ($\text{TE}_0$
# and $\text{TE}_2$):
#
# $$E_y(x, y) \approx c_0(x)\,\phi_0(y; w(x))\,\mathrm{e}^{-\mathrm{i}\int \beta_0(x')\,\mathrm{d}x'} + c_2(x)\,\phi_2(y; w(x))\,\mathrm{e}^{-\mathrm{i}\int \beta_2(x')\,\mathrm{d}x'}.$$
#
# When the accumulated differential phase $\Delta\Phi = \int [\beta_0(x') - \beta_2(x')]\,\mathrm{d}x'$
# across the taper is close to an odd multiple of $\pi$ ($\Delta\Phi \approx 3\pi$
# in 3D SOI across this $5.30\;\mu\text{m}$ taper, corresponding to beat length
# $L_\pi = \pi / \langle \beta_0 - \beta_2 \rangle \approx 1.77\;\mu\text{m}$; in
# the 1D 2D-EIM lateral-TM dispersion model with bulk $n_{\text{SiO}_2} = 1.45$
# lateral cladding, the integral evaluates to $\Delta\Phi = 3.52\pi$, as shown in
# `03_cosine_taper_profile_and_mmi_phase.png`), the $\text{TE}_0$ and $\text{TE}_2$
# modes interfere constructively on axis ($y = 0$) and destructively at the
# sidewalls ($y = \pm w_{\text{out}}/2$), forming a self-imaged beam waist at
# $(x, y) = (0, 0)$ that traverses the $1.10\;\mu\text{m}$ intersection with
# minimal corner scattering or cross-waveguide coupling [1, 2].

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.optimize import brentq, fsolve

from wakis.templates.photonics import (
    PICSimulation,
    Port,
    make_movie,
    n_to_eps,
    plot_eim_stack,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_port_modes,
    plot_spectra_db,
    slab_neff,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    FastCrossingPolySlab,
    build_zoomed_2d_solver,
    crossing_mesh_overrides,
    inspect_task_mesh,
    measure_grid_resolution,
    plot_convergence_ladder,
    plot_feature_and_subpixel_mesh,
)

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ### Device parameters from Chandran *et al.* (2020)
#
# All geometric dimensions are specified in metres (SI units):
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index ($1310\;\text{nm}$) | $n_{\text{Si}}$ | $3.67$ ($\varepsilon_r = 13.4689$) |
# | $\text{SiO}_2$ cladding refractive index ($1310\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.45$ ($\varepsilon_r = 2.1025$) |
# | Silicon core thickness | $h_{\text{Si}}$ | $161\;\text{nm}$ ($z \in [-80.5, +80.5]\;\text{nm}$) |
# | Access strip waveguide width (smallest feature) | $w_{\text{in}}$ | $0.35\;\mu\text{m}$ ($350\;\text{nm}$) |
# | Central crossing box width | $w_{\text{out}}$ | $1.10\;\mu\text{m}$ |
# | Cosine taper length | $l_t$ | $5.30\;\mu\text{m}$ ($|x|, |y| \in [0.55, 5.85]\;\mu\text{m}$) |
# | Cosine taper peak half-width / full width | $w_m,\; 2w_m$ | $0.75\;\mu\text{m},\; 1.50\;\mu\text{m}$ at $|x| = 2.4523\;\mu\text{m}$ |
# | Cosine profile parameters | $a,\; b$ | $0.392996\;\mu\text{m}^{-1},\; 0.963732\;\text{rad}$ |
# | Port plane offset from crossing center | $L_{\text{port}}$ | $6.35\;\mu\text{m}$ |
# | Interior 2D simulation domain | $L_x \times L_y$ | $13.70 \times 13.70\;\mu\text{m}^2$ ($[-6.85, +6.85]\;\mu\text{m}$) |

# %%
# Non-dispersive O-band material refractive indices (Chandran et al. 2020)
N_SI = 3.67
N_SIO2 = 1.45
H_SI = 0.161e-6  # 161 nm SOI core thickness

# O-band wavelength range: 1260 nm to 1360 nm, centered at 1310 nm
WL_MIN = 1.26e-6
WL_MAX = 1.36e-6
WL_REF = 1.31e-6
N_FREQS = 51

# Crossing geometric parameters [m]
W_IN = 0.35e-6
MIN_FEATURE = W_IN  # 350 nm access-waveguide width is the smallest feature
W_OUT = 1.10e-6
W_M = 0.75e-6
L_TAPER = 5.30e-6
L_PORT = 6.35e-6
L_DOMAIN_HALF = 6.85e-6
L_WG_EXT = 8.50e-6  # extends straight waveguides completely through CPML

DOMAIN_2D = (-L_DOMAIN_HALF, L_DOMAIN_HALF, -L_DOMAIN_HALF, L_DOMAIN_HALF)
PORT_SPAN_XY = 1.60e-6
PORT_SPAN_Z = 0.644e-6

# Resolution ladder: each level is (cpw, N_min_across_feature)
RESOLUTION_LADDER = {
    "L1": (10, 4),
    "L2": (14, 6),
    "L3": (20, 8),
    "R": (40, 16),
}


def solve_cosine_taper_params(w_in=W_IN, w_out=W_OUT, w_m=W_M, l_t=L_TAPER):
    """
    Solve the cosine-taper boundary conditions ``w_m * cos(-a * w_out / 2 + b) = w_out / 2``
    and ``w_m * cos(-a * (w_out / 2 + l_t) + b) = w_in / 2`` for ``(a [1/m], b [rad])``.
    """
    theta_out = float(np.arccos(w_out / (2.0 * w_m)))
    theta_in = float(np.arccos(w_in / (2.0 * w_m)))
    a_analytic = (theta_in + theta_out) / float(l_t)
    b_analytic = theta_out + a_analytic * (0.5 * float(w_out))

    # Cross-check with fsolve in micrometres from initial guess (0.5, 2.0)
    w_in_um, w_out_um, w_m_um, l_t_um = (
        w_in * 1e6,
        w_out * 1e6,
        w_m * 1e6,
        l_t * 1e6,
    )

    def _residual(p):
        a_u, b_u = p
        return [
            w_m_um * np.cos(-a_u * (0.5 * w_out_um) + b_u) - 0.5 * w_out_um,
            w_m_um * np.cos(-a_u * (0.5 * w_out_um + l_t_um) + b_u) - 0.5 * w_in_um,
        ]

    guess = (
        (0.5, 2.0)
        if np.isclose(l_t, L_TAPER) and np.isclose(w_m, W_M)
        else (a_analytic * 1e-6 * 0.999, b_analytic * 0.999)
    )
    a_fs_um, b_fs = fsolve(_residual, guess)
    assert np.isclose(a_fs_um * 1e6, a_analytic, rtol=1e-9)
    assert np.isclose(b_fs, b_analytic, rtol=1e-9)
    return a_analytic, b_analytic


TAPER_A, TAPER_B = solve_cosine_taper_params()
X_PEAK_UM = -TAPER_B / (TAPER_A * 1e-6)  # -2.4523 um

# %% [markdown]
# ## 2. Effective-Index Method (EIM) reduction
#
# In the **2D effective-index method (EIM)**, the 3D quasi-TE field in the
# $161\;\text{nm}$ silicon core ($n_{\text{Si}} = 3.67$) clad by $\text{SiO}_2$
# ($n_{\text{SiO}_2} = 1.45$) is factored as
# $\mathbf{E}(x, y, z) \approx \psi(z)\,\mathbf{E}_\parallel(x, y)$. Solving the
# 1D vertical slab waveguide equation along $z$ at $\lambda_0 = 1.31\;\mu\text{m}$
# gives the fundamental vertical $\text{TE}_0$ slab effective index:
#
# $$n_{\text{eff,slab}}(1.31\;\mu\text{m}) = 2.9274.$$
#
# Replacing the 3D silicon core by a 2D cylinder of index $n_{\text{eff,slab}} = 2.9274$
# embedded in $n_{\text{clad}} = 1.45$ reduces the problem to a 2D in-plane FIT
# simulation (`Ex`, `Ey`, `Hz`, single periodic cell along $z$ with `ieps_z = 0`).
#
# To verify the multimode self-imaging condition analytically in 2D EIM, we also
# solve the local 1D lateral slab dispersion relation at width $w(x)$ for the
# even modes $m \in \{0, 2\}$:
#
# $$\kappa_m \frac{w(x)}{2} - \frac{m\pi}{2} = \arctan\!\left(\eta\,\frac{\gamma_m}{\kappa_m}\right), \qquad \kappa_m = k_0\sqrt{n_{\text{eff,slab}}^2 - n_{\text{eff},m}^2}, \quad \gamma_m = k_0\sqrt{n_{\text{eff},m}^2 - n_{\text{clad}}^2},$$
#
# where $\eta = (n_{\text{eff,slab}}/n_{\text{clad}})^2$ for in-plane $E_y$
# polarization (`Ex`, `Ey`, `Hz`) and $\eta = 1$ for out-of-plane polarization.

# %% [markdown]
# ## 3. Parametric geometry construction
#
# We construct the 4-port crossing as a **single closed counter-clockwise
# `PolySlab`** by generating the boundary chain of the $+x$ arm from the lower-right
# central corner $(+w_{\text{out}}/2, -w_{\text{out}}/2)$ to the upper-right
# central corner $(+w_{\text{out}}/2, +w_{\text{out}}/2)$ and rotating it by
# $0^\circ, 90^\circ, 180^\circ, 270^\circ$. Using a single seamless polygon
# eliminates internal polygon interfaces where consecutive material assignments
# would otherwise blend cut cells twice, while guaranteeing exact $D_4$ dihedral
# symmetry ($90^\circ$ rotations and axial/diagonal reflections) to machine
# precision.


# %%
def build_crossing(kind="cosine", n_taper=80, w_m=W_M, l_t=L_TAPER):
    """
    Build the 4-port orthogonal waveguide crossing as a single seamless
    counter-clockwise `PolySlab` and return ``([slab], ports)``.

    Parameters
    ----------
    kind : {'cosine', 'linear', 'plain'}
        Taper profile geometry:
        - ``'cosine'``: Chandran et al. (2020) convex cosine taper (w_m = 0.75 um).
        - ``'linear'``: Monotonic linear taper from w_in = 0.35 um to w_out = 1.10 um.
        - ``'plain'``: Direct untapered intersection of two w_in = 0.35 um waveguides.
    n_taper : int
        Number of discretization points along each taper edge.
    w_m : float
        Cosine taper peak half-width [m] (default 0.75e-6).
    l_t : float
        Taper length [m] (default 5.30e-6).
    """
    if kind == "plain":
        hw = 0.5 * W_IN
        arm_x = np.array([[hw, -hw], [L_WG_EXT, -hw], [L_WG_EXT, +hw]])
        u_nodes = np.array([0.0, hw, L_WG_EXT], dtype=float)
        h_nodes = np.array([hw, hw, hw], dtype=float)
    else:
        u = np.linspace(0.5 * W_OUT, 0.5 * W_OUT + l_t, int(n_taper))
        if kind == "cosine":
            a_par, b_par = solve_cosine_taper_params(
                w_in=W_IN, w_out=W_OUT, w_m=w_m, l_t=l_t
            )
            h = w_m * np.cos(-a_par * u + b_par)
        elif kind == "linear":
            s = (u - 0.5 * W_OUT) / l_t
            h = 0.5 * (W_OUT + (W_IN - W_OUT) * s)
        else:
            raise ValueError(f"Unknown crossing kind {kind!r}.")

        hw_in = 0.5 * W_IN
        arm_x = np.vstack(
            [
                np.column_stack([u, -h]),
                [[L_WG_EXT, -hw_in], [L_WG_EXT, +hw_in]],
                np.column_stack([u[::-1], +h[::-1]])[:-1],
            ]
        )
        u_nodes = np.concatenate([[0.0], u, [L_WG_EXT]])
        h_nodes = np.concatenate([[0.5 * W_OUT], h, [hw_in]])

    verts = np.vstack(
        [
            arm_x,
            np.column_stack([-arm_x[:, 1], arm_x[:, 0]]),
            np.column_stack([-arm_x[:, 0], -arm_x[:, 1]]),
            np.column_stack([arm_x[:, 1], -arm_x[:, 0]]),
        ]
    )
    slab = FastCrossingPolySlab(
        verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
        u_nodes=u_nodes,
        h_nodes=h_nodes,
    )
    ports = [
        Port(
            "west",
            center=(-L_PORT, 0.0, 0.0),
            size=(0.0, PORT_SPAN_XY, PORT_SPAN_Z),
            direction=+1,
        ),
        Port(
            "east",
            center=(+L_PORT, 0.0, 0.0),
            size=(0.0, PORT_SPAN_XY, PORT_SPAN_Z),
            direction=-1,
        ),
        Port(
            "north",
            center=(0.0, +L_PORT, 0.0),
            size=(PORT_SPAN_XY, 0.0, PORT_SPAN_Z),
            direction=-1,
        ),
        Port(
            "south",
            center=(0.0, -L_PORT, 0.0),
            size=(PORT_SPAN_XY, 0.0, PORT_SPAN_Z),
            direction=+1,
        ),
    ]
    return [slab], ports


def lateral_slab_mode_neff(
    width, m=0, n_core_2d=None, n_clad=N_SIO2, wavelength=WL_REF, pol="TM"
):
    """
    1D lateral slab effective index of even mode ``m`` (0, 2, ...) at local width ``width`` [m].
    """
    if n_core_2d is None:
        n_core_2d = slab_neff(N_SI, N_SIO2, H_SI, wavelength, polarization="TE")
    k0 = 2.0 * np.pi / float(wavelength)
    eta = (n_core_2d / n_clad) ** 2 if pol.upper() == "TM" else 1.0
    k_max = k0 * np.sqrt(max(n_core_2d**2 - n_clad**2, 0.0))
    if 0.5 * k_max * width <= 0.5 * m * np.pi:
        return np.nan
    neff_hi = np.sqrt(n_core_2d**2 - (m * np.pi / (k0 * width)) ** 2) - 1e-12

    def _disp(neff):
        kap = k0 * np.sqrt(max(n_core_2d**2 - neff**2, 0.0))
        gam = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
        return 0.5 * kap * width - 0.5 * m * np.pi - np.arctan2(eta * gam, kap)

    return float(brentq(_disp, n_clad + 1e-12, neff_hi))


# %% [markdown]
# ## 4. Subpixel permittivity smoothing vs staircasing
#
# Along the curved cosine-taper sidewalls ($y = \pm w_m \cos(a x + b)$ and
# $x = \pm w_m \cos(a y + b)$), rounding cell fill fractions $w \in [0, 1]$ to
# $\{0, 1\}$ (staircasing) introduces discrete grid steps that act as parasitic
# Bragg reflectors and perturb the local $\text{TE}_0\text{--}\text{TE}_2$ beat
# length. Wakis applies the diagonal **Kottke, Farjadpour & Johnson (2008)** [3]
# subpixel permittivity tensor on every cut cell:
#
# $$\varepsilon_{\text{eff}, c}^{-1} = n_c^2 \left(\frac{w}{\varepsilon_s} + \frac{1 - w}{\varepsilon_b}\right) + \left(1 - n_c^2\right)\left(w\varepsilon_s + (1 - w)\varepsilon_b\right)^{-1}, \qquad c \in \{x, y, z\},$$
#
# using harmonic averaging parallel to the outward boundary normal $\mathbf{n}$
# and arithmetic averaging tangential to the sidewall.
#
# ### Resolution: cells per wavelength and cells across the smallest feature
#
# Reliable time-domain photonic simulations must satisfy two simultaneous grid
# resolution criteria in every execution mode (`--quick`, `--full`, and
# parameter sweeps):
#
# 1. **Wavelength sampling ($\text{cpw} \ge 10$)**: At least $10$ cells per
#    shortest wavelength inside the highest-index material,
#    $\Delta x_{\text{bg}} \le \lambda_{\min} / (n_{\text{eff,slab}}\,\text{cpw})$,
#    to control numerical grid dispersion over the $13.7\;\mu\text{m}$ domain.
# 2. **Feature sampling ($N \ge 4$ cells across the smallest feature)**: At
#    least $N \ge 4$ cells across the narrowest geometric dimension
#    $d_{\min}$, i.e. $\Delta x_{\text{feat}} \le d_{\min} / N$, so that modal
#    confinement and sidewall boundary conditions are resolved rather than
#    collapsed into a single cut cell.
#
# Each resolution level in our ladder is defined by a pair $(\text{cpw}, N)$:
#
# - `L1 = (10, 4)`, `L2 = (14, 6)`, `L3 = (20, 8)`, and reference `R = (40, 16)`.
#
# When a device contains a sub-wavelength slot, coupler gap, or grating tooth
# where $d_{\min} / N < \Delta x_{\text{bg}}$, `PICSimulation(..., mesh_overrides=[(lo, hi, max_cell)])`
# refines only the strip spanning that feature and grades smoothly back to the
# background mesh at ratio $\le 1.4$. For this cosine-tapered crossing, the
# smallest geometric feature is the access-waveguide width
# $w_{\text{in}} = 0.35\;\mu\text{m}$ ($350\;\text{nm}$; the taper flares up to
# $1.50\;\mu\text{m}$ and the crossing box is $1.10\;\mu\text{m}$). At
# $\lambda_{\min} = 1260\;\text{nm}$ and $n_{\text{eff,slab}} = 2.9274$, the
# background slab cell size at `L1` ($10\;\text{cpw}$) is already
# $\Delta x_{\text{bg}} = 42.95\;\text{nm}$, giving $350 / 42.95 = 8.15$ cells
# across $w_{\text{in}}$—exceeding $N = 4$ without needing local override
# refinement (`crossing_mesh_overrides` activates an override box automatically
# whenever $d_{\min} / N < \Delta x_{\text{bg}}$).

# %% [markdown]
# ## 5. Study tasks & execution modes
#
# In `--full` mode (the 10-task study stored in `data/task_0.*` through
# `data/task_9.*`, executed on Cloud Batch with `n_pml = 10` and ring-down
# threshold `energy_decay = 1e-4`), Kottke and staircase permittivity averaging
# are evaluated across three resolution levels—`L1 = (10, 4)`, `L2 = (14, 6)`,
# and `L3 = (20, 8)`—against an independent fine-grid Kottke reference at
# `R = (40, 16)` ($1295 \times 1295 \times 1 = 1{,}677{,}025$ cells,
# $\Delta x = 10.754\;\text{nm}$, $32.547$ cells across $w_{\text{in}} = 350\;\text{nm}$,
# $33{,}451$ steps, $7{,}077.2\;\text{s} \approx 1.97\;\text{h}$ single-core wall
# time, $3{,}170.7\;\text{MB}$ peak RSS). The baseline geometry comparisons
# (`plain_crossing`, `linear_taper_crossing`) and the 4-port $D_4$ symmetry
# audit (`symmetry_4port` driving `west`, `east`, `north`, `south`) run at the
# accepted study level `L3 = (20, 8)`.
#
# For rapid local verification, `--quick` mode runs **8 tasks** (`task_0`
# through `task_7` with `n_pml = 8` and `energy_decay = 5e-3`): Kottke and
# staircase at `L1 = (10, 4)` and `L2 = (14, 6)`, an independent Kottke
# reference at `L3 = (20, 8)`, and the geometry/symmetry tasks at `L2 = (14, 6)`.
# Every task in both `--full` and `--quick` modes satisfies $\text{cpw} \ge 10$
# in the highest-index 2D EIM slab material ($n_{\text{eff,slab}} = 2.9274$) and
# $\text{cells\_across\_feature} \ge 4$ across $w_{\text{in}} = 350\;\text{nm}$
# (zero resolution violations):
#
# | Task (`--full` / `--quick`) | Name | Level (`--full` / `--quick`) | `(cpw, N)` (`--full` / `--quick`) | $\Delta x_{\text{bg}}$ [nm] (`--full`) | $\Delta x_{\min}$ [nm] (`--full`) | Cells across $w_{\text{in}}$ ($350\;\text{nm}$) | Grid (`--full`) / $N_{\text{cells}}$ (`--full` / `--quick`) | Steps (`--full` / `--quick`) | Time [s] (`--full` / `--quick`) | Peak RSS [MB] (`--full`) |
# |---|---|---|---|---|---|---|---|---|---|---|
# | `0` / `0` | `kottke_L1` | `L1` / `L1` | `(10, 4)` / `(10, 4)` | $42.947$ | $42.947$ | $8.150$ | $340\times340\times1$ ($115{,}600$ / $112{,}896$) | $8{,}501$ / $6{,}451$ | $108.4$ / $109.4$ | $778.5$ |
# | `1` / `1` | `staircase_L1` | `L1` / `L1` | `(10, 4)` / `(10, 4)` | $42.947$ | $42.947$ | $8.150$ | $340\times340\times1$ ($115{,}600$ / $112{,}896$) | $9{,}001$ / $6{,}501$ | $113.0$ / $110.2$ | $777.4$ |
# | `2` / `2` | `kottke_L2` | `L2` / `L2` | `(14, 6)` / `(14, 6)` | $30.717$ | $30.717$ | $11.394$ | $467\times467\times1$ ($218{,}089$ / $214{,}369$) | $11{,}801$ / $8{,}901$ | $307.6$ / $365.9$ | $967.5$ |
# | `3` / `3` | `staircase_L2` | `L2` / `L2` | `(14, 6)` / `(14, 6)` | $30.717$ | $30.717$ | $11.394$ | $467\times467\times1$ ($218{,}089$ / $214{,}369$) | $12{,}351$ / $8{,}951$ | $339.4$ / $291.6$ | $962.1$ |
# | `4` / `—` | `kottke_L3` | `L3` / `—` | `(20, 8)` / `—` | $21.507$ | $21.507$ | $16.274$ | $658\times658\times1$ ($432{,}964$ / `—`) | $16{,}801$ / `—` | $1{,}061.1$ / `—` | $1{,}315.7$ |
# | `5` / `—` | `staircase_L3` | `L3` / `—` | `(20, 8)` / `—` | $21.507$ | $21.507$ | $16.274$ | $658\times658\times1$ ($432{,}964$ / `—`) | $17{,}251$ / `—` | $865.7$ / `—` | $1{,}318.4$ |
# | `6` / `4` | `kottke_reference` | `R` / `L3` | `(40, 16)` / `(20, 8)` | $10.754$ ($21.507$ q) | $10.754$ ($21.507$ q) | $32.547$ ($16.274$ q) | $1295\times1295\times1$ ($1{,}677{,}025$ / $427{,}716$) | $33{,}451$ / $12{,}651$ | $7{,}077.2$ / $843.9$ | $3{,}170.7$ |
# | `7` / `5` | `plain_crossing` | `L3` / `L2` | `(20, 8)` / `(14, 6)` | $21.507$ ($30.717$ q) | $21.507$ ($30.717$ q) | $16.274$ ($11.394$ q) | $658\times658\times1$ ($432{,}964$ / $214{,}369$) | $13{,}851$ / $9{,}251$ | $832.5$ / $366.7$ | $995.4$ |
# | `8` / `6` | `linear_taper_crossing` | `L3` / `L2` | `(20, 8)` / `(14, 6)` | $21.507$ ($30.717$ q) | $21.507$ ($30.717$ q) | $16.274$ ($11.394$ q) | $658\times658\times1$ ($432{,}964$ / $214{,}369$) | $13{,}451$ / $8{,}951$ | $695.2$ / $291.9$ | $1{,}319.5$ |
# | `9` / `7` | `symmetry_4port` | `L3` / `L2` | `(20, 8)` / `(14, 6)` | $21.507$ ($30.717$ q) | $21.507$ ($30.717$ q) | $16.274$ ($11.394$ q) | $658\times658\times1$ ($432{,}964$ / $214{,}369$) | $67{,}204$ ($4\times16{,}801$) / $35{,}604$ | $3{,}257.5$ / $1{,}084.5$ | $1{,}514.0$ |


# %%
def get_task_table(quick=False):
    """
    Return the study task table for ``--quick`` (8 tasks: L1/L2 convergence + L3
    reference + L2 geometry/symmetry sweeps) or ``--full`` (10 tasks: L1/L2/L3
    convergence + R reference + L3 geometry/symmetry sweeps).
    """
    n_pml = 8 if quick else 10
    decay = 5e-3 if quick else 1e-4
    conv_levels = ("L1", "L2") if quick else ("L1", "L2", "L3")
    primary_lvl = "L2" if quick else "L3"
    ref_lvl = "L3" if quick else "R"
    sweep_lvl = "L2" if quick else "L3"

    specs = []
    for lvl in conv_levels:
        cpw, n_feat = RESOLUTION_LADDER[lvl]
        for avg in ("kottke", "staircase"):
            is_primary = (lvl == primary_lvl) and (avg == "kottke")
            extra = (
                ", DFT fields + movie + flux box" if is_primary else ""
            )
            specs.append(
                (
                    f"{avg}_{lvl}",
                    lvl,
                    "cosine",
                    avg,
                    cpw,
                    n_feat,
                    ["west"],
                    is_primary,
                    is_primary,
                    f"2D EIM cosine-taper crossing ({avg}, {lvl}: cpw={cpw}, N>={n_feat}{extra})",
                )
            )

    cpw_ref, n_ref = RESOLUTION_LADDER[ref_lvl]
    specs.append(
        (
            "kottke_reference",
            ref_lvl,
            "cosine",
            "kottke",
            cpw_ref,
            n_ref,
            ["west"],
            False,
            False,
            f"2D EIM cosine-taper crossing (kottke, reference {ref_lvl}: cpw={cpw_ref}, N>={n_ref})",
        )
    )

    cpw_sw, n_sw = RESOLUTION_LADDER[sweep_lvl]
    specs.extend(
        [
            (
                "plain_crossing",
                sweep_lvl,
                "plain",
                "kottke",
                cpw_sw,
                n_sw,
                ["west"],
                True,
                False,
                f"2D EIM plain untapered crossing (kottke, {sweep_lvl}: cpw={cpw_sw}, N>={n_sw}, DFT fields)",
            ),
            (
                "linear_taper_crossing",
                sweep_lvl,
                "linear",
                "kottke",
                cpw_sw,
                n_sw,
                ["west"],
                False,
                False,
                f"2D EIM linear-taper crossing (kottke, {sweep_lvl}: cpw={cpw_sw}, N>={n_sw})",
            ),
            (
                "symmetry_4port",
                sweep_lvl,
                "cosine",
                "kottke",
                cpw_sw,
                n_sw,
                ["west", "east", "north", "south"],
                False,
                False,
                f"2D EIM cosine-taper crossing 4-port symmetry audit (kottke, {sweep_lvl}: cpw={cpw_sw}, N>={n_sw})",
            ),
        ]
    )

    tasks = []
    for idx, (
        name,
        lvl,
        kind,
        avg,
        cpw,
        n_feat,
        drive,
        c_fields,
        rec_mov,
        desc,
    ) in enumerate(specs):
        overrides = crossing_mesh_overrides(
            cpw, n_feat, min_feature=MIN_FEATURE, wavelength_min=WL_MIN
        )
        tasks.append(
            {
                "index": idx,
                "name": name,
                "level": lvl,
                "kind": kind,
                "averaging": avg,
                "cpw": cpw,
                "n_feature_min": n_feat,
                "min_feature_nm": float(MIN_FEATURE * 1e9),
                "mesh_overrides": overrides,
                "drive": drive,
                "n_pml": n_pml,
                "energy_decay": decay,
                "collect_fields": c_fields,
                "record_movie": rec_mov,
                "movie_frames": (32 if quick else 48) if rec_mov else 0,
                "description": desc,
            }
        )
    return tasks


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM waveguide crossing study task and write ``task_<I>.npz``
    and ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    structures, ports = build_crossing(kind=task_cfg["kind"])
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    overrides = task_cfg.get(
        "mesh_overrides",
        crossing_mesh_overrides(
            task_cfg["cpw"],
            task_cfg.get("n_feature_min", 4),
            min_feature=MIN_FEATURE,
            wavelength_min=WL_MIN,
        ),
    )
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=task_cfg["averaging"],
        cells_per_wavelength=task_cfg["cpw"],
        domain=DOMAIN_2D,
        n_pml=task_cfg["n_pml"],
        mesh_overrides=overrides,
        field_wavelengths=(WL_REF,),
        record_movie=task_cfg["record_movie"],
        movie_frames=task_cfg["movie_frames"],
        use_gpu=use_gpu,
    )

    drive_ports = task_cfg["drive"]
    s_mat = pic.run_s_parameters(
        drive=drive_ports,
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )

    mesh_info = measure_grid_resolution(
        pic.grid.x,
        pic.grid.y,
        pic.grid.z,
        min_feature=MIN_FEATURE,
        n_cells_z=int(pic.grid.Nz),
        has_ghost=True,
    )
    n_steps_per_drive = int(getattr(pic.solver, "Nt", 0))
    n_steps = int(n_steps_per_drive * len(drive_ports))

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "level": np.array(task_cfg["level"]),
        "kind": np.array(task_cfg["kind"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_feature_min": np.int32(task_cfg.get("n_feature_min", 4)),
        "min_feature_nm": np.float64(mesh_info["min_feature_nm"]),
        "dx_background_nm": np.float64(mesh_info["dx_background_nm"]),
        "dx_min_nm": np.float64(mesh_info["dx_min_nm"]),
        "cells_across_feature": np.float64(mesh_info["cells_across_feature"]),
        "n_cells": np.int64(mesh_info["n_cells"]),
        "n_steps": np.int64(n_steps),
        "n_steps_per_drive": np.int64(n_steps_per_drive),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("west", "west")].astype(np.complex128),
        "S21": s_mat[("east", "west")].astype(np.complex128),
        "S31": s_mat[("north", "west")].astype(np.complex128),
        "S41": s_mat[("south", "west")].astype(np.complex128),
    }

    for (p_out, p_in), val in s_mat.items():
        payload[f"S_{p_out}_{p_in}"] = val.astype(np.complex128)

    if task_cfg["collect_fields"]:
        if task_cfg["kind"] == "cosine":
            modes_w = pic.solve_port_modes(
                "west", frequencies=c_light / WL_REF, n_modes=1
            )
            modes_n = pic.solve_port_modes(
                "north", frequencies=c_light / WL_REF, n_modes=1
            )
            y_w = modes_w[0].coords["Eb"][0]
            ey_w = modes_w[0].e["b"][:, 0]
            m_w = np.abs(y_w) <= 0.8e-6
            x_n = modes_n[0].coords["Ec"][1]
            ex_n = modes_n[0].e["c"][0, :]
            m_n = np.abs(x_n) <= 0.8e-6
            payload["port_y_in"] = y_w[m_w].astype(np.float32)
            payload["port_ey_in"] = ey_w[m_w].astype(np.complex64)
            payload["port_neff_in"] = np.float64(np.real(modes_w[0].neff))
            payload["port_x_cross"] = x_n[m_n].astype(np.float32)
            payload["port_ex_cross"] = ex_n[m_n].astype(np.complex64)
            payload["port_neff_cross"] = np.float64(np.real(modes_n[0].neff))

            wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
            f_disp = c_light / wl_disp
            disp_modes = pic.solve_port_modes("west", frequencies=f_disp, n_modes=1)
            neff_disp = np.array([float(np.real(m[0].neff)) for m in disp_modes])
            dneff_dl = np.gradient(neff_disp, wl_disp)
            ng_disp = neff_disp - wl_disp * dneff_dl
            payload["disp_wavelengths"] = wl_disp.astype(np.float64)
            payload["disp_neff"] = neff_disp.astype(np.float64)
            payload["disp_ng"] = ng_disp.astype(np.float64)

        if WL_REF in pic.dft_fields:
            fmap = pic.dft_fields[WL_REF]
            payload["dft_x"] = fmap["x"]
            payload["dft_y"] = fmap["y"]
            payload["dft_Ex"] = fmap["Ex"]
            payload["dft_Ey"] = fmap["Ey"]
            payload["dft_E2"] = fmap["E2"]

        if pic.movie_data is not None:
            payload["movie_x"] = pic.movie_data["x"]
            payload["movie_y"] = pic.movie_data["y"]
            payload["movie_times"] = pic.movie_data["times"]
            payload["movie_frames"] = pic.movie_data["frames"]
            payload["movie_peak"] = pic.movie_data["peak"]

        if pic.port_time_trace is not None:
            payload["trace_t"] = pic.port_time_trace["t"]
            payload["trace_Ey"] = pic.port_time_trace["Ey"]

        if pic.flux_balance is not None:
            payload["flux_east"] = pic.flux_balance["east"].astype(np.float64)
            payload["flux_west"] = pic.flux_balance["west"].astype(np.float64)
            payload["flux_lateral"] = pic.flux_balance["lateral"].astype(np.float64)
            payload["flux_total_box"] = pic.flux_balance["total_box"].astype(np.float64)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "level": task_cfg["level"],
        "kind": task_cfg["kind"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "n_feature_min": int(task_cfg.get("n_feature_min", 4)),
        "min_feature_nm": round(float(mesh_info["min_feature_nm"]), 3),
        "dx_background_nm": round(float(mesh_info["dx_background_nm"]), 3),
        "dx_min_nm": round(float(mesh_info["dx_min_nm"]), 3),
        "cells_across_feature": round(float(mesh_info["cells_across_feature"]), 3),
        "grid_shape": [int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)],
        "n_cells": int(mesh_info["n_cells"]),
        "n_steps": int(n_steps),
        "n_steps_per_drive": int(n_steps_per_drive),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] level={task_cfg['level']} "
        f"cpw={task_cfg['cpw']} N={meta['n_feature_min']} "
        f"min_feature={meta['min_feature_nm']:.1f}nm "
        f"dx_bg={meta['dx_background_nm']:.2f}nm dx_min={meta['dx_min_nm']:.2f}nm "
        f"cells_across={meta['cells_across_feature']:.2f} "
        f"n_cells={meta['n_cells']} n_steps={meta['n_steps']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete figure suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and writes 16 figures to
# `figures/`:
#
# 1. `01_crossing_layout.png` — Annotated top-view layout of the Chandran *et al.* (2020) cosine-tapered waveguide crossing.
# 2. `02_eim_vertical_stack.png` — Vertical $161\;\text{nm}$ O-band SOI stack, 1D $\text{TE}_0$ mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_cosine_taper_profile_and_mmi_phase.png` — Cosine taper width/slope profile and analytical $\text{TE}_0\text{--}\text{TE}_2$ self-imaging phase integral.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed permittivity map with Yee cell edges across the smallest feature ($w_{\text{in}} = 0.35\;\mu\text{m}$) at `L1` and at the reference level, plus staircase vs Kottke on the cosine taper.
# 5. `05_port_mode_profiles.png` — Orthogonal port eigenmode profiles (`west` $E_y(y)$ and `north` $E_x(x)$) at $\lambda_0 = 1310\;\text{nm}$.
# 6. `06_port_neff_dispersion.png` — Access strip waveguide effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$ across the O-band.
# 7. `07_field_intensity_1310.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda_0 = 1310\;\text{nm}$.
# 8. `08_field_real_ey_1310.png` — Steady-state wave-front phase map $\mathrm{Re}(E_y)$ at $\lambda_0 = 1310\;\text{nm}$.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the cosine-tapered crossing.
# 10. `10_sparameter_spectra_linear.png` — Linear through transmission, 2D Poynting flux-box budget, and zoomed insertion loss vs Chandran *et al.* (2020).
# 11. `11_sparameter_spectra_db.png` — Broadband 4-port S-parameter spectra ($S_{21}, S_{31}, S_{41}, S_{11}$) in dB.
# 12. `12_tapered_vs_plain_crossing.png` — Insertion loss, crosstalk, and return loss comparison: cosine taper vs linear taper vs plain untapered crossing.
# 13. `13_multimode_focusing_comparison.png` — Multimode self-imaging waist vs untapered diffraction: zoomed 2D $|E|^2$ and axial/transverse cross-sections.
# 14. `14_symmetry_4fold_audit.png` — 4-fold ($D_4$) rotational/mirror symmetry and Lorentz reciprocity audit across all 4 input ports.
# 15. `15_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke vs staircasing against the independent `kottke_reference` $T_{\text{ref}}$, labeled with both $\text{cpw}$ and cells across $w_{\text{in}}$.
# 16. `16_port_time_trace.png` — Through-port time-domain waveform $E_y(t)$ and exponential ring-down audit.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all 16 figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d

    primary_name = None
    for cand in ("kottke_L3", "kottke_L2", "kottke_fine"):
        if cand in tasks_data and "dft_E2" in tasks_data[cand]:
            primary_name = cand
            break
    if primary_name is None:
        raise FileNotFoundError(
            f"Missing primary task ('kottke_L2' or 'kottke_L3') in {data_dir}. Run --quick or --full first."
        )
    primary = tasks_data[primary_name]
    structures, ports = build_crossing(kind="cosine")

    # 1. Top-view layout with dimensional annotations
    annotations = [
        {
            "text": "Access WG\nw_in = 0.35 µm",
            "xy": (-6.0, -0.175),
            "xytext": (-4.8, -2.5),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": f"Cosine taper (l_t = 5.30 µm)\n2w_m = 1.50 µm @ x = {X_PEAK_UM:.2f} µm",
            "xy": (X_PEAK_UM, 0.75),
            "xytext": (-3.6, 3.2),
            "arrowprops": dict(arrowstyle="->", color="#2b5c8f", lw=1.1),
        },
        {
            "text": "Central crossing box\nw_out = 1.10 × 1.10 µm²",
            "xy": (0.55, -0.55),
            "xytext": (3.4, -2.8),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.1),
        },
        {
            "text": "Orthogonal cross arm\n(+y 'north' port)",
            "xy": (0.0, 4.2),
            "xytext": (3.6, 4.4),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=DOMAIN_2D,
        annotations=annotations,
        title="Chandran et al. (2020) cosine-tapered SOI waveguide crossing (2D EIM)",
        out_path=fig_dir / "01_crossing_layout.png",
    )

    # 2. Effective-Index Method (EIM) vertical stack and O-band slab dispersion
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=H_SI,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Cosine taper width/slope profile and analytical TE0-TE2 self-imaging phase
    x_taper_um = np.linspace(-(0.5 * W_OUT + L_TAPER) * 1e6, -0.5 * W_OUT * 1e6, 300)
    x_taper_m = x_taper_um * 1e-6
    w_cos_um = 2.0 * (W_M * 1e6) * np.cos(TAPER_A * x_taper_m + TAPER_B)
    dw_dx = -2.0 * W_M * TAPER_A * np.sin(TAPER_A * x_taper_m + TAPER_B)
    s_lin = (x_taper_um - x_taper_um[0]) / (x_taper_um[-1] - x_taper_um[0])
    w_lin_um = W_IN * 1e6 + (W_OUT - W_IN) * 1e6 * s_lin

    neff_slab_val = float(primary["n_eff_slab"])
    k0_ref = 2.0 * np.pi / WL_REF
    neff_te0 = np.array(
        [
            lateral_slab_mode_neff(w * 1e-6, m=0, n_core_2d=neff_slab_val)
            for w in w_cos_um
        ]
    )
    neff_te2 = np.array(
        [
            lateral_slab_mode_neff(w * 1e-6, m=2, n_core_2d=neff_slab_val)
            for w in w_cos_um
        ]
    )
    d_neff = np.where(np.isfinite(neff_te2), neff_te0 - neff_te2, 0.0)
    dx_m = x_taper_m[1] - x_taper_m[0]
    cum_dphi_pi = np.cumsum(k0_ref * d_neff * dx_m) / np.pi

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=150)
    ax0.plot(
        x_taper_um,
        w_cos_um,
        color="#2b5c8f",
        lw=2.2,
        label=r"Cosine taper $w(x) = 2w_m\cos(ax+b)$",
    )
    ax0.plot(
        x_taper_um,
        w_lin_um,
        "--",
        color="#d95f02",
        lw=1.7,
        label="Linear taper (0.35 → 1.10 µm)",
    )
    ax0.axhline(
        W_IN * 1e6, color="0.45", ls=":", lw=1.3, label="Untapered strip (0.35 µm)"
    )
    ax0.axvline(
        X_PEAK_UM,
        color="#1b9e77",
        ls="-.",
        lw=1.3,
        label=f"Peak 2w_m = 1.50 µm @ x = {X_PEAK_UM:.2f} µm",
    )
    ax0_r = ax0.twinx()
    ax0_r.plot(
        x_taper_um, dw_dx, color="#1b9e77", lw=1.4, alpha=0.65, label=r"$\mathrm{d}w/\mathrm{d}x$"
    )
    ax0_r.set_ylabel(r"Flare slope $\mathrm{d}w/\mathrm{d}x$ [rad]", color="#1b9e77")
    ax0.set(
        xlabel="Axial position x along input taper [µm]",
        ylabel="Full waveguide width w(x) [µm]",
        title="(a) Convex cosine vs linear taper profile",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=7.8)

    ax1.plot(
        x_taper_um, neff_te0, color="#2b5c8f", lw=2.1, label=r"Even mode $\mathrm{TE}_0$ $n_{\mathrm{eff}}(x)$"
    )
    ax1.plot(
        x_taper_um, neff_te2, color="#d95f02", lw=2.1, label=r"Even mode $\mathrm{TE}_2$ $n_{\mathrm{eff}}(x)$"
    )
    ax1.axhline(N_SIO2, color="0.5", ls=":", lw=1.2, label=f"Cladding n_SiO2 = {N_SIO2}")
    ax1_r = ax1.twinx()
    ax1_r.plot(
        x_taper_um,
        cum_dphi_pi,
        "--",
        color="#1b9e77",
        lw=1.9,
        label=f"Cumulative $\\Delta\\Phi/\\pi$ (end = {cum_dphi_pi[-1]:.2f}$\\pi$)",
    )
    ax1_r.set_ylabel(r"Accumulated phase $\Delta\Phi(x) / \pi$", color="#1b9e77")
    ax1.set(
        xlabel="Axial position x along input taper [µm]",
        ylabel="Local lateral mode effective index",
        title=r"(b) Local $\mathrm{TE}_0 / \mathrm{TE}_2$ dispersion & self-imaging phase",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center left", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "03_cosine_taper_profile_and_mmi_phase.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. Zoomed permittivity map with Yee cell edges across the smallest feature (L1 vs Ref) & cosine taper
    cpw_l1, n_l1 = RESOLUTION_LADDER["L1"]
    n_pml_plot = 10 if "kottke_L3" in tasks_data else 8
    if "kottke_reference" in tasks_data:
        cpw_ref = float(tasks_data["kottke_reference"]["cpw"])
        n_ref = int(tasks_data["kottke_reference"].get("n_feature_min", 8))
    else:
        cpw_ref, n_ref = RESOLUTION_LADDER["L3"]

    pic_l1_stair = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="staircase",
        cells_per_wavelength=cpw_l1,
        domain=DOMAIN_2D,
        n_pml=n_pml_plot,
        mesh_overrides=crossing_mesh_overrides(cpw_l1, n_l1, min_feature=MIN_FEATURE),
    )
    pic_l1_kottke = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_l1,
        domain=DOMAIN_2D,
        n_pml=n_pml_plot,
        mesh_overrides=crossing_mesh_overrides(cpw_l1, n_l1, min_feature=MIN_FEATURE),
    )
    solver_ref_kottke = build_zoomed_2d_solver(
        structures=structures,
        cpw=cpw_ref,
        n_feature_min=n_ref,
        averaging="kottke",
        xlim_um=(-6.45, -5.15),
        ylim_um=(-0.50, 0.50),
        n_pml=n_pml_plot,
        wl_min=WL_MIN,
        wl_max=WL_MAX,
        wl_ref=WL_REF,
        n_si=N_SI,
        n_sio2=N_SIO2,
        h_si=H_SI,
        domain_2d=DOMAIN_2D,
        min_feature=MIN_FEATURE,
    )
    plot_feature_and_subpixel_mesh(
        pic_l1_kottke.solver,
        solver_ref_kottke,
        pic_l1_stair.solver,
        cpw_l1=cpw_l1,
        n_l1=n_l1,
        cpw_ref=cpw_ref,
        n_ref=n_ref,
        min_feature_um=MIN_FEATURE * 1e6,
        component="y",
        out_path=fig_dir / "04_permittivity_mesh_zoom.png",
    )
    del pic_l1_stair, pic_l1_kottke, solver_ref_kottke

    # 5. Orthogonal port mode profiles (west Ey(y) and north Ex(x))
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_x_cross"] * 1e6,
        primary["port_ex_cross"],
        float(primary["port_neff_cross"]),
        structures_outline_y=[(-0.5 * W_IN * 1e6, 0.5 * W_IN * 1e6)],
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Access waveguide mode effective index and group index dispersion
    fig_disp, ax_disp = plot_neff_dispersion(
        primary["disp_wavelengths"] * 1e9,
        primary["disp_neff"],
        neff_slab=float(primary["n_eff_slab"]),
        ng_2d=primary["disp_ng"],
        out_path=None,
    )
    ax_disp.set_title("Access waveguide mode dispersion across the O-band (1260–1360 nm)")
    fig_disp.savefig(
        fig_dir / "06_port_neff_dispersion.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_disp)

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1310 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        structures=structures,
        kind="intensity_log",
        title=r"Steady-state modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda_0 = 1310\;\mathrm{nm}$",
        out_path=fig_dir / "07_field_intensity_1310.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1310 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        structures=structures,
        kind="real",
        title=r"Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $\lambda_0 = 1310\;\mathrm{nm}$",
        out_path=fig_dir / "08_field_real_ey_1310.png",
    )

    # 9. Animated pulse propagation GIF
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        make_movie(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            structures=structures,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Spectral arrays sorted by wavelength in the interior O-band [1265, 1355] nm
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1265.0) & (wl_nm <= 1355.0)
    wl_b = wl_nm[mask]

    s11 = primary["S11"][order][mask]
    s21 = primary["S21"][order][mask]
    s31 = primary["S31"][order][mask]
    s41 = primary["S41"][order][mask]
    t_thru = np.abs(s21) ** 2
    t_north = np.abs(s31) ** 2
    t_south = np.abs(s41) ** 2
    r_west = np.abs(s11) ** 2
    t_guided_tot = t_thru + t_north + t_south + r_west

    # 10. Linear S-parameter spectra, 2D flux-box budget, and zoomed insertion loss vs Chandran (2020)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.1), dpi=150)
    ax0.plot(
        wl_b,
        t_thru,
        color="#1b9e77",
        lw=2.2,
        label=r"Through $T_{\mathrm{through}} = |S_{21}|^2$",
    )
    ax0.plot(
        wl_b,
        t_guided_tot,
        "--",
        color="#2b5c8f",
        lw=1.8,
        label=r"Total guided $|S_{11}|^2 + |S_{21}|^2 + |S_{31}|^2 + |S_{41}|^2$",
    )
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask]
        ax0.plot(
            wl_b,
            f_box,
            ":",
            color="0.3",
            lw=1.7,
            label=f"2D Poynting flux box total (mean = {np.mean(f_box):.4f})",
        )
    ax0.plot(
        wl_b,
        1.0 - t_guided_tot,
        "-.",
        color="#d95f02",
        lw=1.6,
        label=r"Unguided radiative loss $1 - \sum_i |S_{i1}|^2$",
    )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Linear power fraction",
        ylim=(-0.02, 1.06),
        title="(a) Linear transmission & 2D energy conservation budget",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="center right", fontsize=8.0)

    il_db = 10.0 * np.log10(np.maximum(t_thru, 1e-12))
    prim_lvl = str(primary.get("level", "L3"))
    prim_cpw = int(float(primary.get("cpw", 20)))
    ax1.plot(
        wl_b,
        il_db,
        color="#1b9e77",
        lw=2.2,
        label=f"Wakis 2D EIM (Kottke {prim_lvl}, {prim_cpw} cpw)",
    )
    if "kottke_reference" in tasks_data:
        rd = tasks_data["kottke_reference"]
        rw = rd["wavelengths"] * 1e9
        ro = np.argsort(rw)
        rm = (rw[ro] >= 1265.0) & (rw[ro] <= 1355.0)
        r_il_db = 10.0 * np.log10(np.maximum(np.abs(rd["S21"][ro][rm]) ** 2, 1e-12))
        ax1.plot(
            rw[ro][rm],
            r_il_db,
            ":",
            color="#2b5c8f",
            lw=2.0,
            label=f"Wakis 2D EIM (Kottke ref {rd['level']}, {int(float(rd['cpw']))} cpw)",
        )
    ax1.axhline(
        -0.197,
        color="#d95f02",
        ls="--",
        lw=1.8,
        label="Chandran et al. (2020) [1] (~ −0.20 dB; −0.197 ± 0.017 dB @ 1310 nm)",
    )
    ax1.fill_between(
        wl_b,
        -0.197 - 0.035,
        -0.197 + 0.035,
        color="#d95f02",
        alpha=0.14,
        label="Chandran et al. (2020) [1] O-band variation (±0.035 dB, 1260–1360 nm)",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Through transmission $10\log_{10}|S_{21}|^2$ [dB]",
        ylim=(-0.45, 0.05),
        title="(b) Zoomed insertion loss vs Chandran et al. (2020)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "10_sparameter_spectra_linear.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 11. Full 4-port S-parameter spectra in dB
    plot_spectra_db(
        wl_b,
        {
            r"$T_{\mathrm{through}} = |S_{21}|^2$ (east)": il_db,
            r"$T_{\mathrm{north}} = |S_{31}|^2$ (+y cross)": 10.0
            * np.log10(np.maximum(t_north, 1e-12)),
            r"$T_{\mathrm{south}} = |S_{41}|^2$ (−y cross)": 10.0
            * np.log10(np.maximum(t_south, 1e-12)),
            r"$R = |S_{11}|^2$ (return loss)": 10.0
            * np.log10(np.maximum(r_west, 1e-12)),
        },
        title="Cosine-tapered crossing 4-port S-parameter spectra (1265–1355 nm, 2D EIM)",
        ylabel="Modal power [dB]",
        reference_lines=[
            {
                "y": -30.0,
                "color": "0.4",
                "ls": ":",
                "label": "−30 dB crosstalk benchmark",
            }
        ],
        out_path=fig_dir / "11_sparameter_spectra_db.png",
    )

    # 12. Cosine-tapered vs linear-tapered vs plain (untapered) crossing comparison
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=150)
    geom_specs = [
        (primary_name, "Cosine taper (2w_m = 1.50 µm)", "#1b9e77", "-"),
        ("linear_taper_crossing", "Linear taper (0.35 → 1.10 µm)", "#2b5c8f", "--"),
        ("plain_crossing", "Plain untapered (w = 0.35 µm)", "#d95f02", "-."),
    ]
    for g_name, g_label, g_col, g_ls in geom_specs:
        if g_name not in tasks_data:
            continue
        gd = tasks_data[g_name]
        gw = gd["wavelengths"] * 1e9
        go = np.argsort(gw)
        gm = (gw[go] >= 1265.0) & (gw[go] <= 1355.0)
        g_wl = gw[go][gm]
        g_t = np.abs(gd["S21"][go][gm]) ** 2
        g_x = np.abs(gd["S31"][go][gm]) ** 2
        g_r = np.abs(gd["S11"][go][gm]) ** 2
        ax0.plot(
            g_wl,
            10.0 * np.log10(np.maximum(g_t, 1e-12)),
            color=g_col,
            ls=g_ls,
            lw=2.1,
            label=g_label,
        )
        ax1.plot(
            g_wl,
            10.0 * np.log10(np.maximum(g_x / np.maximum(g_t, 1e-12), 1e-12)),
            color=g_col,
            ls=g_ls,
            lw=2.1,
            label=f"{g_label} (crosstalk)",
        )
        ax1.plot(
            g_wl,
            10.0 * np.log10(np.maximum(g_r, 1e-12)),
            color=g_col,
            ls=":",
            lw=1.4,
            alpha=0.8,
            label=f"{g_label} (R)",
        )

    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Through transmission $10\log_{10}|S_{21}|^2$ [dB]",
        title="(a) Insertion loss: cosine vs linear vs plain crossing",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.0)

    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Relative crosstalk / return loss [dB]",
        title=r"(b) Crosstalk $10\log_{10}(T_{\mathrm{cross}}/T_{\mathrm{through}})$ & reflection",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "12_tapered_vs_plain_crossing.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 13. Multimode taper focusing vs plain untapered diffraction across the intersection
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=150)
    xc_um = primary["dft_x"] * 1e6
    yc_um = primary["dft_y"] * 1e6
    e2_cos = primary["dft_E2"]
    e2_cos_norm = e2_cos / max(float(np.max(e2_cos)), 1e-30)

    if "plain_crossing" in tasks_data and "dft_E2" in tasks_data["plain_crossing"]:
        dp = tasks_data["plain_crossing"]
        xp_um = dp["dft_x"] * 1e6
        yp_um = dp["dft_y"] * 1e6
        e2_pln = dp["dft_E2"]
        e2_pln_norm = e2_pln / max(float(np.max(e2_pln)), 1e-30)
    else:
        xp_um, yp_um, e2_pln_norm = xc_um, yc_um, e2_cos_norm

    # Panel (a): Plain crossing zoomed intensity
    im0 = axes[0].pcolormesh(
        xp_um,
        yp_um,
        10.0 * np.log10(np.maximum(e2_pln_norm.T, 1e-4)),
        cmap="inferno",
        vmin=-35.0,
        vmax=0.0,
        shading="auto",
    )
    axes[0].set(
        aspect="equal",
        xlim=(-3.5, 3.5),
        ylim=(-3.5, 3.5),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Plain crossing: strong cross-port diffraction",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="|E|² [dB]")

    # Panel (b): Cosine-tapered crossing zoomed intensity
    im1 = axes[1].pcolormesh(
        xc_um,
        yc_um,
        10.0 * np.log10(np.maximum(e2_cos_norm.T, 1e-4)),
        cmap="inferno",
        vmin=-35.0,
        vmax=0.0,
        shading="auto",
    )
    axes[1].set(
        aspect="equal",
        xlim=(-3.5, 3.5),
        ylim=(-3.5, 3.5),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(b) Cosine taper: TE₀+TE₂ self-imaged waist",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, label="|E|² [dB]")

    # Panel (c): Transverse intensity slice |E(0, y)|^2 across the intersection center x = 0
    ix0_c = int(np.argmin(np.abs(xc_um)))
    prof_y_cos = e2_cos[ix0_c, :] / max(float(np.max(e2_cos[ix0_c, :])), 1e-30)
    ix_peak = int(np.argmin(np.abs(xc_um - X_PEAK_UM)))
    prof_y_bulge = e2_cos[ix_peak, :] / max(float(np.max(e2_cos[ix_peak, :])), 1e-30)
    axes[2].plot(
        yc_um,
        prof_y_cos,
        color="#1b9e77",
        lw=2.2,
        label="Cosine taper @ x = 0 (focused waist)",
    )
    axes[2].plot(
        yc_um,
        prof_y_bulge,
        "--",
        color="#2b5c8f",
        lw=1.8,
        label=f"Cosine taper @ x = {X_PEAK_UM:.2f} µm (bulge)",
    )
    if "plain_crossing" in tasks_data and "dft_E2" in tasks_data["plain_crossing"]:
        ix0_p = int(np.argmin(np.abs(xp_um)))
        prof_y_pln = e2_pln[ix0_p, :] / max(float(np.max(e2_pln[ix0_p, :])), 1e-30)
        axes[2].plot(
            yp_um,
            prof_y_pln,
            "-.",
            color="#d95f02",
            lw=1.7,
            label="Plain crossing @ x = 0",
        )
    axes[2].axvline(-0.5 * W_OUT * 1e6, color="0.4", ls=":", lw=1.2, label="±w_out/2 (±0.55 µm)")
    axes[2].axvline(+0.5 * W_OUT * 1e6, color="0.4", ls=":", lw=1.2)
    axes[2].set(
        xlim=(-1.6, 1.6),
        ylim=(-0.02, 1.08),
        xlabel="Transverse coordinate y [µm]",
        ylabel="Normalized transverse intensity |E(x, y)|²",
        title="(c) Transverse beam waist at crossing center",
    )
    axes[2].grid(True, ls=":", alpha=0.4)
    axes[2].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "13_multimode_focusing_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 14. 4-fold (D4) symmetry and Lorentz reciprocity audit
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.1), dpi=150)
    if "symmetry_4port" in tasks_data:
        sd = tasks_data["symmetry_4port"]
        sw = sd["wavelengths"] * 1e9
        so = np.argsort(sw)
        sm = (sw[so] >= 1265.0) & (sw[so] <= 1355.0)
        sw_b = sw[so][sm]
        thru_pairs = [
            ("east", "west", "West → East |S_ew|²", "#1b9e77", "-"),
            ("west", "east", "East → West |S_we|²", "#d95f02", "--"),
            ("south", "north", "North → South |S_sn|²", "#2b5c8f", "-."),
            ("north", "south", "South → North |S_ns|²", "#e7298a", ":"),
        ]
        thru_arrs = []
        for p_o, p_i, lbl, col, ls in thru_pairs:
            t_arr = np.abs(sd[f"S_{p_o}_{p_i}"][so][sm]) ** 2
            thru_arrs.append(t_arr)
            ax0.plot(
                sw_b,
                10.0 * np.log10(np.maximum(t_arr, 1e-12)),
                color=col,
                ls=ls,
                lw=2.0,
                label=lbl,
            )
        thru_stack = np.vstack(thru_arrs)
        c4_thru_spread = np.maximum(
            np.max(thru_stack, axis=0) - np.min(thru_stack, axis=0), 1e-16
        )

        # Reciprocity residual max_{i,j} |S_ij - S_ji|
        pnames = ("west", "east", "north", "south")
        recip_err = np.zeros_like(sw_b)
        for i, pi in enumerate(pnames):
            for pj in pnames[i + 1 :]:
                diff_ij = np.abs(
                    sd[f"S_{pi}_{pj}"][so][sm] - sd[f"S_{pj}_{pi}"][so][sm]
                )
                recip_err = np.maximum(recip_err, diff_ij)
        recip_err = np.maximum(recip_err, 1e-16)
        ns_mirror_err = np.maximum(
            np.abs(
                np.abs(sd["S_north_west"][so][sm]) ** 2
                - np.abs(sd["S_south_west"][so][sm]) ** 2
            ),
            1e-16,
        )

        ax1.semilogy(
            sw_b,
            c4_thru_spread,
            color="#1b9e77",
            lw=2.0,
            label=r"$C_4$ through-power spread $\max_{i,j}|T_i - T_j|$",
        )
        ax1.semilogy(
            sw_b,
            recip_err,
            "--",
            color="#2b5c8f",
            lw=1.9,
            label=r"Reciprocity residual $\max_{i,j}|S_{ij} - S_{ji}|$",
        )
        ax1.semilogy(
            sw_b,
            ns_mirror_err,
            "-.",
            color="#d95f02",
            lw=1.9,
            label=r"Mirror crosstalk imbalance $||S_{31}|^2 - |S_{41}|^2|$",
        )
    else:
        ax0.plot(wl_b, il_db, color="#1b9e77", lw=2.0, label="West → East |S_21|²")
        ax1.semilogy(
            wl_b,
            np.maximum(np.abs(t_north - t_south), 1e-16),
            color="#d95f02",
            lw=2.0,
            label=r"Mirror crosstalk imbalance $||S_{31}|^2 - |S_{41}|^2|$",
        )

    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Through transmission [dB]",
        title="(a) 4-fold through transmission from each input port",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.0)

    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Dimensionless residual (log scale)",
        title=r"(b) $D_4$ rotational/mirror symmetry & reciprocity residuals",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "14_symmetry_4fold_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 15. Multi-resolution convergence (Kottke vs Staircase against independent kottke_reference)
    levels = [
        lvl
        for lvl in ("L1", "L2", "L3")
        if f"kottke_{lvl}" in tasks_data and f"staircase_{lvl}" in tasks_data
    ]
    if len(levels) >= 2 and "kottke_reference" in tasks_data:
        ref_d = tasks_data["kottke_reference"]

        def _band_thru(d):
            w = d["wavelengths"] * 1e9
            o = np.argsort(w)
            m = (w[o] >= 1270.0) & (w[o] <= 1350.0)
            return np.abs(d["S21"][o][m]) ** 2

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m = (w >= 1270.0) & (w <= 1350.0)
            return float(np.mean(np.abs(d["S11"][m]) ** 2))

        def _cells_across(d):
            if "cells_across_feature" in d:
                return float(d["cells_across_feature"])
            return float(
                inspect_task_mesh(
                    {"kind": "cosine", "cpw": float(d["cpw"]), "n_feature_min": 4},
                    build_crossing,
                )["cells_across_feature"]
            )

        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]
        cells_across = [_cells_across(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        t_ref_band = _band_thru(ref_d)
        err_k = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (_band_thru(tasks_data[f"kottke_{lvl}"]) - t_ref_band) ** 2
                        )
                    )
                ),
                1e-5,
            )
            for lvl in levels
        ]
        err_s = [
            max(
                float(
                    np.sqrt(
                        np.mean(
                            (_band_thru(tasks_data[f"staircase_{lvl}"]) - t_ref_band)
                            ** 2
                        )
                    )
                ),
                1e-5,
            )
            for lvl in levels
        ]
        rk_vals = [_r_mean(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        rs_vals = [_r_mean(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        plot_convergence_ladder(
            cpws=cpws,
            cells_across=cells_across,
            err_s=err_s,
            err_k=err_k,
            rs_vals=rs_vals,
            rk_vals=rk_vals,
            ref_cpw=float(ref_d["cpw"]),
            ref_cells_across=_cells_across(ref_d),
            ref_r_mean=_r_mean(ref_d),
            level_labels=levels,
            out_path=fig_dir / "15_kottke_vs_staircase_convergence.png",
        )

    # 16. Port time trace and ring-down energy decay audit
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150)
        ax0.plot(t_fs, ey_norm, color="#2b5c8f", lw=1.4)
        ax0.set(
            xlabel="Time t [fs]",
            ylabel=r"Normalized $E_y(t) / \max|E_y|$",
            title="(a) Through-port ('east') time-domain waveform",
        )
        ax0.grid(True, ls=":", alpha=0.4)

        env = np.maximum(ey_norm**2, 1e-10)
        ax1.semilogy(
            t_fs, env, color="#d95f02", lw=1.4, label=r"$|E_y(t)|^2 / |E_y|_{\max}^2$"
        )
        ax1.axhline(
            1e-4, color="0.4", ls="--", lw=1.2, label="1e-4 energy decay threshold"
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-8, 2.0),
            title="(b) Single-pass transit & ring-down audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "16_port_time_trace.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Rendered 16 figures in {fig_dir}")


# %% [markdown]
# ## 7. Quantitative results & when 2D EIM breaks down
#
# ### Full-resolution (`--full`) S-parameter and convergence summary
#
# Evaluating the 10 `--full` study tasks (`data/task_0.*` through `data/task_9.*`,
# `n_pml = 10`, `energy_decay = 1e-4`) across the interior O-band
# ($1270\text{--}1350\;\text{nm}$) against the independent fine-grid
# `kottke_reference` (`R: cpw=40, N>=16`, $\Delta x = 10.754\;\text{nm}$,
# $32.547$ cells across $w_{\text{in}} = 350\;\text{nm}$) gives:
#
# | Task | Run (`averaging`, level) | `cpw` | Cells across $w_{\text{in}}$ | $T(1310\;\text{nm})$ | Insertion loss $\text{IL}(1310\;\text{nm})$ [dB] | Crosstalk $|S_{31}(1310\;\text{nm})|^2$ [dB] | Return loss $|S_{11}(1310\;\text{nm})|^2$ [dB] | Band mean $\langle |S_{11}|^2 \rangle$ | Band RMS $\|T - T_{\text{ref}}\|_{\text{RMS}}$ |
# |---|---|---|---|---|---|---|---|---|---|
# | `0` | `kottke_L1` (`kottke`, `L1`) | $10$ | $8.150$ | $0.98143$ | $0.0814$ | $-42.68$ | $-49.60$ | $5.40 \times 10^{-5}$ | $3.90 \times 10^{-3}$ |
# | `1` | `staircase_L1` (`staircase`, `L1`) | $10$ | $8.150$ | $0.97353$ | $0.1165$ | $-43.05$ | $-50.57$ | $1.81 \times 10^{-4}$ | $5.85 \times 10^{-3}$ |
# | `2` | `kottke_L2` (`kottke`, `L2`) | $14$ | $11.394$ | $0.98039$ | $0.0860$ | $-42.88$ | $-45.53$ | $3.53 \times 10^{-5}$ | $2.19 \times 10^{-3}$ |
# | `3` | `staircase_L2` (`staircase`, `L2`) | $14$ | $11.394$ | $0.97314$ | $0.1182$ | $-43.24$ | $-45.96$ | $4.61 \times 10^{-4}$ | $6.30 \times 10^{-3}$ |
# | `4` | `kottke_L3` (`kottke`, `L3`, primary) | $20$ | $16.274$ | $0.97958$ | $0.0896$ | $-43.47$ | $-41.82$ | $3.55 \times 10^{-5}$ | $1.13 \times 10^{-3}$ |
# | `5` | `staircase_L3` (`staircase`, `L3`) | $20$ | $16.274$ | $0.97554$ | $0.1075$ | $-44.27$ | $-44.59$ | $1.85 \times 10^{-4}$ | $3.33 \times 10^{-3}$ |
# | `6` | `kottke_reference` (`kottke`, `R`) | $40$ | $32.547$ | $0.97847$ | $0.0945$ | $-43.66$ | $-40.20$ | $3.78 \times 10^{-5}$ | $0$ (reference $T_{\text{ref}}$) |
# | `7` | `plain_crossing` (`kottke`, `L3`) | $20$ | $16.274$ | $0.80355$ | $0.9499$ | $-14.86$ | $-21.87$ | $6.43 \times 10^{-3}$ | — |
# | `8` | `linear_taper_crossing` (`kottke`, `L3`) | $20$ | $16.274$ | $0.86109$ | $0.6495$ | $-37.35$ | $-34.30$ | $3.66 \times 10^{-4}$ | — |
# | `9` | `symmetry_4port` (`kottke`, `L3`) | $20$ | $16.274$ | $0.97958$ | $0.0896$ | $-43.47$ | $-41.82$ | $3.55 \times 10^{-5}$ | $\max |T_i - T_j| = 5.60 \times 10^{-7}$ |
#
# ### Old (`--quick`) vs New (`--full`) vs Reference / Analytic Theory
#
# The table below compares the earlier `--quick` study (`L2: 14 cpw` primary,
# `L3: 20 cpw` reference, `n_pml = 8`, `energy_decay = 5e-3`) against the
# completed `--full` study (`L3: 20 cpw` primary, `R: 40 cpw` reference,
# `n_pml = 10`, `energy_decay = 1e-4`), the published 3D FDTD / experimental
# results of **Chandran *et al.* (2020)** [1], and analytical waveguide/MMI
# theory:
#
# | Quantity | Old (`--quick`: `L2` $14\;\text{cpw}$ / `L3` $20\;\text{cpw}$ ref) | New (`--full`: `L3` $20\;\text{cpw}$ / `R` $40\;\text{cpw}$ ref) | Published Reference [1] / Analytic Theory |
# |---|---|---|---|
# | **Cosine crossing $T(1310\;\text{nm}) = |S_{21}|^2$** | $0.95576$ (`L2`) / $0.95647$ (`L3`) | $0.97958$ (`L3`) / $0.97847$ (`R`) | $\approx 0.9556$ ($\text{IL} = 0.197 \pm 0.017\;\text{dB}$ at $1310\;\text{nm}$, Abstract of [1]) |
# | **Cosine crossing $\text{IL}(1310\;\text{nm})$ [dB]** | $0.1965\;\text{dB}$ (`L2`) / $0.1933\;\text{dB}$ (`L3`) | $0.0896\;\text{dB}$ (`L3`) / $0.0945\;\text{dB}$ (`R`) | $\sim 0.2\;\text{dB}$ (3D FDTD & exp.); $0.197 \pm 0.017\;\text{dB}$ at $1310\;\text{nm}$ (Abstract of [1]) |
# | **Cosine crossing O-band $\text{IL}$ ($1265\text{--}1355\;\text{nm}$) [dB]** | $[-0.0259, 0.1988]\;\text{dB}$ (`L2`, truncation ripple) | $[0.0877, 0.1005]\;\text{dB}$ (`L3`) / $[0.0929, 0.1030]\;\text{dB}$ (`R`) | $\pm 0.035\;\text{dB}$ variation across $1260\text{--}1360\;\text{nm}$ around $\sim 0.2\;\text{dB}$ (Abstract of [1]) |
# | **Cosine crosstalk $|S_{31}(1310\;\text{nm})|^2$ [dB]** | $-43.15\;\text{dB}$ (`L2`) / $-44.04\;\text{dB}$ (`L3`) | $-43.47\;\text{dB}$ (`L3`) / $-43.66\;\text{dB}$ (`R`) | $-35\;\text{dB}$ ($<-30\;\text{dB}$, Abstract of [1]) |
# | **Cosine O-band max crosstalk $\max |S_{31}|^2$ [dB]** | $-41.78\;\text{dB}$ (`L2`) / $-42.24\;\text{dB}$ (`L3`) | $-41.48\;\text{dB}$ (`L3`) / $-41.63\;\text{dB}$ (`R`) | $<-30\;\text{dB}$ across $1260\text{--}1360\;\text{nm}$ (Abstract of [1]) |
# | **Cosine return loss $|S_{11}(1310\;\text{nm})|^2$ [dB]** | $-47.95\;\text{dB}$ (`L2`) / $-45.73\;\text{dB}$ (`L3`) | $-41.82\;\text{dB}$ (`L3`) / $-40.20\;\text{dB}$ (`R`) | $<-30\;\text{dB}$ [1] |
# | **Linear taper $T(1310\;\text{nm})$ / $\text{IL}$ / $\text{XT}$** | $0.83659$ / $0.7749\;\text{dB}$ / $-38.26\;\text{dB}$ (`L2`) | $0.86109$ / $0.6495\;\text{dB}$ / $-37.35\;\text{dB}$ (`L3`) | Adiabatic flare lacks $\text{TE}_0 + \text{TE}_2$ self-imaging waist |
# | **Plain crossing $T(1310\;\text{nm})$ / $\text{IL}$ / $\text{XT}$** | $0.78066$ / $1.0754\;\text{dB}$ / $-14.94\;\text{dB}$ (`L2`) | $0.80355$ / $0.9499\;\text{dB}$ / $-14.86\;\text{dB}$ (`L3`) | Unfocused diffraction into orthogonal cross arms |
# | **Cosine IL advantage vs linear / vs plain ($1310\;\text{nm}$)** | $-0.5784\;\text{dB}$ / $-0.8789\;\text{dB}$ ($+17.51\%$ $T$) | $-0.5599\;\text{dB}$ ($+11.85\%$ $T$) / $-0.8603\;\text{dB}$ ($+17.60\%$ $T$) | $\text{TE}_0 + \text{TE}_2$ self-imaging waist suppresses cross-arm diffraction |
# | **Cosine XT suppression vs plain ($1310\;\text{nm}$)** | $28.21\;\text{dB}$ (`L2`) | $28.61\;\text{dB}$ (`L3`) | $> 20\;\text{dB}$ suppression [1] |
# | **2D Poynting flux-box total $\langle \Phi_{\text{box}} \rangle$** | $0.97681$ (`L2`, stops at $263.3\;\text{fs}$) | $0.99991$ (`L3`, $[0.99975, 1.00007]$, stops at $348.0\;\text{fs}$) | $1.00000$ (exact lossless 2D passivity) |
# | **$D_4$ $C_4$ spread / reciprocity / mirror residual** | $1.07\times10^{-6}$ / $8.17\times10^{-7}$ / $2.71\times10^{-10}$ (`L2`) | $5.60\times10^{-7}$ / $2.96\times10^{-7}$ / $1.13\times10^{-9}$ (`L3`) | $0$ (exact $D_4$ & Lorentz reciprocity) |
# | **Vertical slab $n_{\text{eff,slab}}(1310\;\text{nm})$ & port $n_{\text{eff}}(1310\;\text{nm})$** | $2.9274$ / $2.4351$ (`L2`) | $2.9274$ / $2.4348$ (`L3`, $n_g = 3.3348$) | Analytic 1D TE slab $n_{\text{eff,slab}} = 2.9274$ |
# | **Cosine taper $(a, b, x_{\text{peak}})$ & phase $\Delta\Phi$** | $0.392996\;\mu\text{m}^{-1}$, $0.963732\;\text{rad}$, $-2.4523\;\mu\text{m}$; $3.52\pi$ | $0.392996\;\mu\text{m}^{-1}$, $0.963732\;\text{rad}$, $-2.4523\;\mu\text{m}$; $3.52\pi$ | Exact closed-form endpoints; 3D target $\Delta\Phi \approx 3\pi$ [1] |
#
# ### Key findings, `--quick` truncation audit & convergence verdict
#
# 1. **Why `--full` revises the insertion loss from $0.1965\;\text{dB}$ (`--quick`) to $0.0896\;\text{dB}$ (`L3`) / $0.0945\;\text{dB}$ (`R`) (`10_sparameter_spectra_linear.png`, `16_port_time_trace.png`)**:
#    In the `--quick` run (`energy_decay = 5e-3`), the cosine crossing
#    yielded $\text{IL}(1310\;\text{nm}) = 0.1965\;\text{dB}$ at `L2` and
#    $0.1933\;\text{dB}$ at `L3`, which appears at first glance to be a direct
#    numerical match to the $\approx 0.20\;\text{dB}$ 3D FDTD value of Chandran
#    *et al.* (2020) [1]. The full study (`energy_decay = 1e-4`) shows plainly
#    that **this apparent $0.20\;\text{dB}$ match in `--quick` was a coincidence
#    caused by early pulse-tail truncation**: at `energy_decay = 5e-3`, `kottke_L2`
#    stopped at $t = 263.3\;\text{fs}$ ($8{,}901$ steps) while the trailing edge
#    of the transmitted Gaussian pulse was still passing through the `east` port
#    monitor ($x = +6.35\;\mu\text{m}$; see `16_port_time_trace.png` at
#    $t \approx 263\;\text{fs}$ where $|E_y(t)|^2/|E_y|_{\max}^2 \sim 10^{-2}$).
#    As a result, $2.32\%$ of the pulse energy had not yet exited the flux box
#    ($\langle \Phi_{\text{box}} \rangle = 0.97681$), adding $\approx 0.102\;\text{dB}$
#    of artificial truncation loss at $1310\;\text{nm}$ and producing a spectral
#    DFT truncation ripple that dipped slightly below $0\;\text{dB}$ ($\text{IL} = -0.0259\;\text{dB}$)
#    near the band edges. In `--full` (`energy_decay = 1e-4`), the simulation runs
#    to $t = 348.0\;\text{fs}$ ($16{,}801$ steps at `L3`), where the port power has
#    decayed to $1.93 \times 10^{-7}$ of its peak, the 2D Poynting flux box closes
#    to $\langle \Phi_{\text{box}} \rangle = 0.99991$ ($\min = 0.99975, \max = 1.00007$),
#    the spectral ripple disappears, and the true converged 2D EIM insertion loss is
#    **$0.0896\;\text{dB}$ at `L3` ($20\;\text{cpw}$) and $0.0945\;\text{dB}$ at
#    `R` ($40\;\text{cpw}$)** ($0.0877\text{--}0.1005\;\text{dB}$ across
#    $1265\text{--}1355\;\text{nm}$ at `L3`). Because the same $\approx 0.10\text{--}0.13\;\text{dB}$
#    pulse-tail truncation offset affected all geometries in `--quick`, the
#    **relative taper comparison is unchanged**: at `L3`, the convex cosine taper
#    ($\text{IL} = 0.0896\;\text{dB}$, $\text{XT} = -43.47\;\text{dB}$) beats the
#    monotonic linear taper ($\text{IL} = 0.6495\;\text{dB}$, $\text{XT} = -37.35\;\text{dB}$)
#    by $0.5599\;\text{dB}$ ($+11.85$ percentage points in through power) and the
#    plain untapered crossing ($\text{IL} = 0.9499\;\text{dB}$, $\text{XT} = -14.86\;\text{dB}$)
#    by $0.8603\;\text{dB}$ ($+17.60$ percentage points in through power) and
#    $28.61\;\text{dB}$ in crosstalk suppression (`12_tapered_vs_plain_crossing.png`,
#    `13_multimode_focusing_comparison.png`). At the intersection center $x = 0$,
#    the self-imaged cosine waist contracts to $\text{FWHM} = 0.387\;\mu\text{m}$
#    (from $0.860\;\mu\text{m}$ at the bulge peak $x = -2.4523\;\mu\text{m}$),
#    suppressing the sidewall intensity at $y = \pm 0.55\;\mu\text{m}$ by $42.7\times$
#    ($-16.30\;\text{dB}$) relative to the on-axis peak.
# 2. **Why converged 2D EIM insertion loss ($0.090\text{--}0.095\;\text{dB}$) is lower than 3D FDTD ($\approx 0.20\;\text{dB}$)**:
#    In 2D EIM (`bc_z = periodic`), vertical out-of-plane radiation is identically
#    zero; the $2.04\%$ (`L3`) to $2.15\%$ (`R`) loss is purely in-plane cladding
#    radiation and residual modal mismatch at the intersection corners. In the 3D
#    $161\;\text{nm}$ SOI crossing of Chandran *et al.* (2020) [1], opening the
#    lateral sidewalls over the $1.10 \times 1.10\;\mu\text{m}^2$ crossing box also
#    perturbs vertical confinement along $z$, scattering an additional $\approx 2.3\%$
#    ($\sim 0.105\;\text{dB}$) into out-of-plane oxide cladding modes so that total
#    3D loss is $\approx 0.20\;\text{dB}$. We do not tune any geometric or material
#    parameter to artificially inflate the 2D EIM loss to match the 3D value.
# 3. **Convergence verdict and Kottke vs staircase scaling (`15_kottke_vs_staircase_convergence.png`)**:
#    - **Resolution compliance**: All 10 `--full` tasks (and all 8 `--quick` tasks)
#      satisfy $\text{cpw} \ge 10$ ($\text{cpw} \in \{10, 14, 20, 40\}$) and
#      $\text{cells\_across\_feature} \ge 4$ ($8.150, 11.394, 16.274, 32.547$ cells
#      across $w_{\text{in}} = 350\;\text{nm}$), read directly from `data/task_*.json`.
#    - **Convergence at the accepted level `L3` ($20\;\text{cpw}$)**: Between the
#      last two study levels `L2` ($14\;\text{cpw}$) and `L3` ($20\;\text{cpw}$),
#      Kottke through transmission $T(1310\;\text{nm})$ changes by only
#      $|T_{\text{L3}} - T_{\text{L2}}| / T_{\text{L3}} = 0.0833\%$
#      ($\Delta T = -8.16 \times 10^{-4}$, $\Delta\text{IL} = +0.0036\;\text{dB}$ or
#      $4.03\%$ of $\text{IL}$), and the band-averaged RMS change across
#      $1270\text{--}1350\;\text{nm}$ is $\|T_{\text{L3}} - T_{\text{L2}}\|_{\text{RMS}} / \langle T_{\text{L3}} \rangle = 0.1088\%$
#      ($\text{RMS} = 1.07 \times 10^{-3}$). Between `L3` ($20\;\text{cpw}$) and the
#      $2\times$-finer independent reference `R` ($40\;\text{cpw}$), the relative
#      change in $T(1310\;\text{nm})$ is $|T_{\text{R}} - T_{\text{L3}}| / T_{\text{R}} = 0.1127\%$
#      ($\Delta T = -1.10 \times 10^{-3}$, $\Delta\text{IL} = +0.0049\;\text{dB}$ or
#      $5.18\%$ of $\text{IL}$), the band RMS error is
#      $\|T_{\text{L3}} - T_{\text{ref}}\|_{\text{RMS}} = 1.13 \times 10^{-3}$
#      ($0.1159\%$ of $\langle T_{\text{ref}} \rangle$), and cross-port crosstalk
#      changes by only $0.19\;\text{dB}$ ($-43.47\;\text{dB} \to -43.66\;\text{dB}$).
#      The study is therefore **well converged at `L3`**.
#    - **Kottke vs staircase convergence order**: Against `R = (40, 16)`, Kottke
#      band RMS error decreases monotonically from $3.90 \times 10^{-3}$ (`L1`) to
#      $2.19 \times 10^{-3}$ (`L2`) and $1.13 \times 10^{-3}$ (`L3`), corresponding
#      to an empirical convergence order $p(\text{L1}\to\text{L2}) = 1.72$,
#      $p(\text{L2}\to\text{L3}) = 1.85$, and overall $p(\text{L1}\to\text{L3}) = 1.78$
#      (approaching second-order $\mathcal{O}(\Delta x^2)$). Staircase averaging
#      exhibits non-monotonic error between `L1` and `L2` ($5.85 \times 10^{-3} \to 6.30 \times 10^{-3}$,
#      $p = -0.22$) before decreasing to $3.33 \times 10^{-3}$ at `L3` ($2.93\times$
#      larger error than Kottke at `L3`; overall empirical order $p(\text{L1}\to\text{L3}) = 0.82$,
#      first-order $\mathcal{O}(\Delta x^1)$). Furthermore, Kottke suppresses
#      spurious sidewall Bragg reflection $\langle |S_{11}|^2 \rangle$ by $3.3\times$
#      at `L1` ($5.40 \times 10^{-5}$ vs $1.81 \times 10^{-4}$), $13.1\times$ at `L2`
#      ($3.53 \times 10^{-5}$ vs $4.61 \times 10^{-4}$), and $5.2\times$ at `L3`
#      ($3.55 \times 10^{-5}$ vs $1.85 \times 10^{-4}$, matching the reference
#      $3.78 \times 10^{-5}$).
# 4. **4-port $D_4$ symmetry and reciprocity (`14_symmetry_4fold_audit.png`)**:
#    Driving all four ports (`symmetry_4port` at `L3`) confirms 4-fold rotational
#    through-power invariance across $1265\text{--}1355\;\text{nm}$ to
#    $\max_{i,j}|T_i - T_j| = 5.60 \times 10^{-7}$, Lorentz reciprocity to
#    $\max_{i,j}|S_{ij} - S_{ji}| = 2.96 \times 10^{-7}$, and north/south mirror
#    crosstalk balance $||S_{31}|^2 - |S_{41}|^2|$ to $1.13 \times 10^{-9}$.
#
# ### Physical and numerical limits
#
# Four physical and numerical limits must be kept in mind:
#
# 1. **Out-of-plane radiative diffraction at the intersection**: In a 3D
#    $161\;\text{nm}$ SOI slab, when the lateral sidewalls open into the orthogonal
#    arms at $|x| \le 0.55\;\mu\text{m}$, vertical confinement along $z$ is locally
#    perturbed at the four corners $(\pm 0.55, \pm 0.55)\;\mu\text{m}$, coupling
#    $\approx 2.3\%$ ($\sim 0.105\;\text{dB}$) of power into unguided vertical
#    substrate/cladding radiation. In 2D EIM (`bc_z = periodic`), vertical radiation
#    is absent and all loss ($\approx 2.0\text{--}2.2\%$, or $0.090\text{--}0.095\;\text{dB}$)
#    is restricted to in-plane cladding radiation and cross-port coupling.
# 2. **Lateral effective-index contrast & $\text{TE}_0\text{--}\text{TE}_2$ beat
#    length**: Standard EIM replaces the silicon core by $n_{\text{eff,slab}}(1.31\;\mu\text{m}) = 2.9274$
#    and the etched regions by bulk oxide $n_{\text{SiO}_2} = 1.45$. Because the
#    3D vector boundary condition at vertical sidewalls mixes in-plane electric
#    field components, the 2D EIM lateral phase velocity of the $\text{TE}_2$ mode
#    differs slightly from 3D ($\Delta\Phi = 3.52\pi$ in the 1D 2D-EIM slab integral
#    vs $\approx 3\pi$ in 3D), shifting the optimal self-imaging wavelength and
#    yielding even lower in-plane cross-port crosstalk ($-43.47\;\text{dB}$ at `L3`
#    and $-43.66\;\text{dB}$ at `R` in 2D EIM vs $<-30\;\text{dB}$ in 3D FDTD [1, 4]).
# 3. **Fixed slab dispersion**: Holding $n_{\text{eff,slab}}$ constant at its
#    $1310\;\text{nm}$ value across $1260\text{--}1360\;\text{nm}$ neglects vertical
#    slab dispersion ($\Delta n_{\text{eff,slab}} \approx -0.056$ across the band,
#    shown in `02_eim_vertical_stack.png`).
# 4. **Failure mode when a narrow gap or feature is resolved by only 1–2 cells**:
#    If a narrow waveguide, coupler gap, or grating tooth is sampled by only
#    1–2 Yee cells, the evanescent decay, coupling coefficient, cavity resonance,
#    or Bragg wavelength becomes a strong function of where the interfaces land
#    relative to the grid lines (the staircased width can jump by $50\text{--}100\%$).
#    Subpixel permittivity averaging (Kottke) removes first-order boundary-position
#    steps, but it cannot reconstruct field gradients inside a feature that spans
#    only a single cell—you cannot resolve the physics of a geometry when the grid
#    is coarser than the feature itself, which is why we enforce $\text{cpw} \ge 10$
#    and $N \ge 4$ cells across the smallest feature via `mesh_overrides`.
#
# ## 8. Exercises
#
# 1. **Cosine bulge amplitude sweep**: Vary the peak half-width $w_m \in \{0.60, 0.68, 0.75, 0.82\}\;\mu\text{m}$
#    in `build_crossing(kind="cosine", w_m=...)` and plot the insertion loss and
#    cross-port crosstalk at $1310\;\text{nm}$ vs $w_m$. How sensitive is the
#    self-imaging condition to a $\pm 50\;\text{nm}$ width bias?
# 2. **Taper length & beat-length scaling**: Sweep the taper length
#    $l_t \in [3.5, 6.5]\;\mu\text{m}$ and compare the transmission maxima against
#    the analytical phase integral $\Delta\Phi(l_t) = \int [\beta_0(x) - \beta_2(x)]\,\mathrm{d}x$.
# 3. **Cascaded crossing insertion-loss chain**: Using the extracted complex
#    $4 \times 4$ scattering matrix $S(\lambda)$ from `symmetry_4port`, compute the
#    cascaded transmission $|S_{21}(\lambda)|^{2N}$ through $N = 10$ crossings
#    for the cosine-tapered, linear-tapered, and plain untapered designs.
#
# ## 9. References
#
# 1. S. Chandran, M. Dahlem, Y. Bian, P. Moreira, A. P. Jacob, M. Rakowski,
#    *et al.*, "Beam shaping for ultra-compact waveguide crossings on monolithic
#    silicon photonics platform," *Opt. Lett.* **45**(22), 6230–6233 (2020),
#    DOI: [10.1364/OL.402446](https://doi.org/10.1364/OL.402446).
# 2. L. B. Soldano and E. C. M. Pennings, "Optical multi-mode interference devices
#    based on self-imaging: principles and applications," *J. Lightwave Technol.*
#    **13**(4), 615–627 (1995),
#    DOI: [10.1109/50.372474](https://doi.org/10.1109/50.372474).
# 3. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing of
#    discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 4. M. Hammer and O. V. Ivanova, "Effective index approximations of photonic
#    crystal slabs: a 2-to-1-D assessment," *Opt. Quantum Electron.* **41**(4),
#    267–283 (2009),
#    DOI: [10.1007/s11082-009-9349-3](https://doi.org/10.1007/s11082-009-9349-3).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="03. Waveguide Crossing Based on Cosine Tapers (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run L1/L2/L3-ref 2D study locally (cpw >= 10, N >= 4) and render all figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run full L1/L2/L3/R-ref 2D study across all tasks and render all figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed; uses --full unless --quick is also passed)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for --task (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and generate all figures in figures/",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_tasks:
        tasks = get_task_table(quick=args.quick)
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=args.quick)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    # Default when invoked with no flags (e.g. notebook execution): --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    for extra_idx in range(len(tasks), 20):
        for ext in ("npz", "json"):
            p = DATA_DIR / f"task_{extra_idx}.{ext}"
            if p.exists():
                p.unlink()
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
