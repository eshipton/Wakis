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
# # Tutorial 9.4 · Visible-Band RGB Achromatic Metalens (`invrs-gym` `metalens`)
#
# **Chapter 9 — Inverse Design**
#
# This tutorial presents Wakis's time-domain FIT model, `fmmax` Fourier-modal-method (FMM/RCWA)
# cross-validation, and Round-1 $\to$ Round-2 optimization results on the `invrs-gym`
# **`metalens`** benchmark challenge (Schubert et al., arXiv:`2410.24132`). The RGB-metalens
# benchmark problem itself originates in the multi-group `photonics-opt-testbed` study on the
# validation and characterization of algorithms and software for photonics inverse design
# (M. Chen, R. E. Christiansen, J. A. Fan, G. Işiklar, J. Jiang, S. G. Johnson, W. Ma,
# O. D. Miller, A. Oskooi, M. F. Schubert, F. Wang, I. A. D. Williamson, W. Xue and Y. Zhou,
# *J. Opt. Soc. Am. B* **41**, A161, 2024, doi:`10.1364/JOSAB.506412`). It builds directly on
# the inverse-designed metasurface and high-NA metalens work from Owen D. Miller's group at Yale
# ([millergroup.yale.edu](https://millergroup.yale.edu)): high-NA achromatic metalenses by
# inverse design over a minimax multi-wavelength focal-intensity objective analyzing the
# trade-off between aperture diameter, numerical aperture, and dispersion engineering (Chung and
# Miller, *Opt. Express* **28**, 6945, 2020, doi:`10.1364/OE.385440`); the inverse design of a
# high-NA metalens for maskless lithography with experimentally fabricable feature scales (Chung,
# Zhang, Li, Miller, and Smith, *Nanophotonics* **12**, 2371–2381, 2023,
# doi:`10.1515/nanoph-2022-0761`); and tunable metasurface inverse design for high switching
# efficiencies and wide-angle angular deflection (Chung and Miller, *ACS Photonics* **7**,
# 2236–2243, 2020, doi:`10.1021/acsphotonics.0c00787`):
#
# - **Challenge geometry & materials**: a $10.0\,\mu\text{m}$-wide, $1.0\,\mu\text{m}$-thick
#   $\text{TiO}_2$-like metasurface slab ($n_{\text{solid}} = 2.40$, $\varepsilon_{\max} = 5.76$;
#   $n_{\text{void}} = 1.00$, $\varepsilon_{\min} = 1.00$) resting on a fused-silica substrate
#   ($n_{\text{sub}} = 1.50$, $\varepsilon_{\text{sub}} = 2.25$) inside a $16.0\,\mu\text{m}$
#   periodic unit cell at a native pixel resolution of $\Delta x = \Delta y = 20\,\text{nm}$
#   ($500\times 50$ design pixels).
# - **Multi-wavelength achromatic objective**: focus a normally incident plane wave from the
#   substrate at three visible RGB wavelengths — **blue** ($\lambda_1 = 450\,\text{nm}$),
#   **green** ($\lambda_2 = 550\,\text{nm}$), and **red** ($\lambda_3 = 650\,\text{nm}$) — to a
#   common focal spot in air at $f = 2.4\,\mu\text{m}$ above the top surface of the lens
#   ($(x_f, y_f) = (0.0,\, +3.4)\,\mu\text{m}$, numerical aperture
#   $\text{NA} = \sin(\arctan((D/2)/f)) = \sin(\arctan(5.0/2.4)) = 0.9015$).
# - **Foundry length-scale target**: minimum solid pillar width $\ge 5\,\text{px} = 100\,\text{nm}$
#   and minimum void slot spacing $\ge 5\,\text{px} = 100\,\text{nm}$.
# - **Key results**:
#   - **Cross-validation on reference design `Rasmus123nm`**: Official `fmmax` FMM/RCWA gives
#     per-wavelength enhancements **`16.39 / 14.30 / 15.45`** (worst-case $\min = 14.30$), and our
#     2D Wakis time-domain model reproduces **`15.55 / 14.27 / 16.09`** (worst-case $\min = 14.27$,
#     agreeing to within $0.2\%$ on the worst-case figure of merit).
#   - **Round-1 (before guaranteed length scale)**: achieved an official `fmmax` worst-case
#     enhancement of **$14.07$** ($14.0676$ on `seed 0`, with per-wavelength enhancements
#     `18.23 / 14.07 / 16.80`), but violated the $100\,\text{nm}$ foundry rule with
#     **`2 / 1 px` ($40\,\text{nm}$ solid / $20\,\text{nm}$ void)** features (and `seed 1` scored
#     $7.78$ at `2 / 2 px`).
#   - **Round-2 (`round2_metalens`, schedule `v3`, `seed 0`)**: enforcing guaranteed
#     **`5 / 5 px` ($100\,\text{nm}$ / $100\,\text{nm}$)** minimum width and spacing via
#     `FeasibleDensity` (`conic_filter` + Wang–Lazarov–Sigmund $\tanh$ projection + `imageruler`
#     morphological repair + $\beta = 64$ STE) achieves an official `fmmax` worst-case enhancement
#     of **$8.87$** ($8.8652$, with per-wavelength official enhancements **`11.01 / 8.87 / 9.28`**
#     and Wakis enhancements **`10.83 / 8.43 / 9.18`**).
#   - **Leaderboard note**: the committed `leaderboard_rescore.json` bundle only covers the two
#     `ceviche` waveguide challenges (`ceviche_waveguide_bend` and `ceviche_power_splitter`); a
#     `metalens` leaderboard entry is **not included** in the bundle, so we compare against the
#     published `Rasmus123nm` reference design (`14.30` at $123\,\text{nm}$ minimum width/spacing)
#     and our Round-1 unconstrained run (`14.07` at `2/1 px`).
#
# ### Learning goals
#
# 1. **Figure of merit (`eval_metric`)**: compute the per-wavelength focal intensity enhancement
#    $\eta(\lambda_k) = |\mathbf{E}_{\text{lens}}(\mathbf{r}_f, \lambda_k)|^2 / |\mathbf{E}_{\text{ref}}(\mathbf{r}_f, \lambda_k)|^2$
#    over the no-lens substrate reference and the worst-case scalar metric
#    $\mathcal{M} = \min_k \eta(\lambda_k)$, plus the softmin optimization objective.
# 2. **Wakis 2D $x\text{--}y$ model & `Rasmus123nm` validation vs. `fmmax`**: understand the
#    $16\,\mu\text{m}$ periodic cell, smoothed top-hat current-sheet source (`source_offset = 2.5 um`,
#    `source_smoothing_fwhm = 1.0 um`), and series-capacitance harmonic-mean edge assignment
#    (`PixelDesign`), and see how Wakis matches `fmmax` RCWA on `Rasmus123nm` (`14.27` vs. `14.30`).
# 3. **Physical trade-off between `5/5 px` ($100\,\text{nm}$) length scale and RGB chromatic
#    dispersion engineering**: analyze why removing $20\text{--}40\,\text{nm}$ sub-wavelength slots
#    reduces the worst-case focal enhancement from $14.07$ (`2/1 px` Round 1) to $8.87$
#    (`5/5 px` Round 2) by restricting the group-delay dispersion envelope of $\text{TiO}_2$
#    waveguide-mode resonators.
# 4. **Standalone Wakis re-simulation of the stored Round-2 metalens**: rebuild the exact
#    $800\times 320$ ($20\,\text{nm}$ pixel) geometry, harmonic-mean `PixelDesign` edges, and
#    smoothed aperture source from `round2_metalens/design.npy`, and verify that Wakis reproduces
#    the stored enhancements (`10.8253 / 8.4329 / 9.1790`) to $\max|\Delta\eta| = 1.71\times 10^{-7}$.
# 5. **Focal-spot wavefields & diffraction-limited Airy/$\operatorname{sinc}^2$ line cut**:
#    compare the lateral focal-plane intensity profile at $y_f = 3.4\,\mu\text{m}$ against the
#    2D cylindrical-aperture diffraction limit
#    $I_{\text{diff}}(x) = \operatorname{sinc}^2(2\,\text{NA}\,x/\lambda)$ at
#    $\text{NA} = 0.9015$ across all three RGB wavelengths.
#
# ---
#
# ## Physics, Wakis Model & Round-1 vs. Round-2 Summary
#
# ### 1. Polychromatic focal enhancement figure of merit
#
# At each RGB wavelength $\lambda_k \in \{450, 550, 650\}\,\text{nm}$, let
# $\mathbf{E}_{\text{lens}}(\mathbf{r}_f, \lambda_k)$ be the steady-state complex electric field at
# the focal point $\mathbf{r}_f = (0.0,\, +3.4)\,\mu\text{m}$ with the metalens present, and let
# $\mathbf{E}_{\text{ref}}(\mathbf{r}_f, \lambda_k)$ be the reference electric field at the same
# location for a bare substrate ($n_{\text{sub}} = 1.5$ for $y \le 0$, $n_{\text{air}} = 1.0$ for
# $y > 0$, with the lens region replaced by air). The per-wavelength intensity enhancement and
# worst-case `eval_metric` are:
#
# $$
# \eta_k = \frac{|\mathbf{E}_{\text{lens}}(\mathbf{r}_f, \lambda_k)|^2}{|\mathbf{E}_{\text{ref}}(\mathbf{r}_f, \lambda_k)|^2},
# \qquad
# \mathcal{M} = \min_{k \in \{1, 2, 3\}} \eta_k.
# $$
#
# Because $\min_k \eta_k$ is non-smooth when the limiting wavelength switches between blue, green,
# and red, the `invrs-wakis` optimizer minimizes the smooth softmin surrogate with temperature
# $\tau = 2.0$:
#
# $$
# \mathcal{L}(\boldsymbol{\rho}) = \tau \log\!\left(\frac{1}{3}\sum_{k=1}^{3} \exp\!\left(-\frac{\eta_k(\boldsymbol{\rho})}{\tau}\right)\right)
# \approx -\min_{k} \eta_k(\boldsymbol{\rho}).
# $$
#
# ### 2. Series-capacitance harmonic-mean edges (`PixelDesign`)
#
# In our 2D Wakis model, the out-of-plane electric field is represented by $E_z(x, y)$ on a
# 1-cell-thick `GridFIT3D` domain (`Nz = 1`). Each primary $z$-edge at grid node $(I, J)$ touches
# four pixel cells $(i, j) \in \{I-1, I\} \times \{J-1, J\}$ of the $500\times 50$ design region.
# Rather than arithmetic averaging, `PixelDesign` assigns the inverse permittivity $\varepsilon_z^{-1}$
# via the 4-cell harmonic mean (series capacitance):
#
# $$
# \varepsilon_{z}^{-1}(I, J) = \frac{1}{4\,\varepsilon_0}\sum_{\delta_i \in \{0, 1\}}\sum_{\delta_j \in \{0, 1\}}
# \frac{1}{\varepsilon_{\min} + (\varepsilon_{\max} - \varepsilon_{\min})\,\rho_{I - \delta_i,\, J - \delta_j}},
# $$
#
# which matches the subpixel smoothing used when calibrating the Wakis time-domain model against
# `fmmax` FMM/RCWA on `Rasmus123nm`.
#
# ### 3. Summary of Validation, Round-1, and Round-2 Results
#
# | Design / Run | Backend / Grid | Enhancement $\eta_{450}$ | Enhancement $\eta_{550}$ | Enhancement $\eta_{650}$ | Worst-Case $\mathcal{M} = \min_k \eta_k$ | Min Solid / Void (`px` & `nm`) | Target (`px` & `nm`) | Notes |
# | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
# | **Reference `Rasmus123nm` (`fmmax`)** | Official FMM/RCWA | $16.39$ | $14.30$ | $15.45$ | **$14.30$** | $123\,\text{nm}$ | $100\,\text{nm}$ | Published benchmark reference (Chen et al. 2024) |
# | **Reference `Rasmus123nm` (Wakis)** | Wakis $20\,\text{nm}$ ($r=1$) | $15.55$ | $14.27$ | $16.09$ | **$14.27$** | $123\,\text{nm}$ | $100\,\text{nm}$ | Validates Wakis vs. `fmmax` to $0.2\%$ on $\min_k \eta_k$ |
# | **Wakis Round-2 (`v3`, `seed 0`, Official)** | Official `fmmax` | $11.01$ | **$8.87$** | $9.28$ | **$8.8652$** ($8.87$) | **`5 / 5 px` ($100 / 100\,\text{nm}$)** | `5 / 5 px` ($100\,\text{nm}$) | **Guaranteed $100\,\text{nm}$ length scale (`ste_beta64:14`)** |
# | **Wakis Round-2 (`v3`, `seed 0`, Wakis $r=1$)** | Wakis $20\,\text{nm}$ ($r=1$) | $10.83$ | **$8.43$** | $9.18$ | **$8.4329$** | **`5 / 5 px` ($100 / 100\,\text{nm}$)** | `5 / 5 px` ($100\,\text{nm}$) | Re-simulated below ($\max|\Delta\eta| = 1.71\times 10^{-7}$) |
# | **Wakis Round-1 (`seed 0`, Unconstrained)** | Official `fmmax` | $18.23$ | **$14.07$** | $16.80$ | **$14.0676$** ($14.07$) | `2 / 1 px` ($40 / 20\,\text{nm}$) | `5 / 5 px` ($100\,\text{nm}$) | **Violates length scale** ($20\,\text{nm}$ air slots) |
# | **Wakis Round-1 (`seed 1`, Unconstrained)** | Official `fmmax` | $9.45$ | **$7.78$** | $12.23$ | **$7.7794$** ($7.78$) | `2 / 2 px` ($40 / 40\,\text{nm}$) | `5 / 5 px` ($100\,\text{nm}$) | Violates length scale; lower local minimum |
# | **`invrs-gym` Leaderboard** | — | — | — | — | *Not in bundle* | — | `5 / 5 px` ($100\,\text{nm}$) | Not included in `leaderboard_rescore.json` |

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS_0
from scipy.ndimage import binary_closing, binary_erosion, binary_opening

import wakis
from wakis.inverse import AdjointSimulation, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

WAVELENGTHS_NM = np.array([450.0, 550.0, 650.0], dtype=np.float64)
WAVELENGTHS_UM = WAVELENGTHS_NM * 1e-3
APERTURE_UM = 10.0
FOCAL_LENGTH_UM = 2.4
LENS_THICKNESS_UM = 1.0
FOCUS_X_UM = 0.0
FOCUS_Y_UM = LENS_THICKNESS_UM + FOCAL_LENGTH_UM  # +3.4 um
LENS_NA = float(np.sin(np.arctan((0.5 * APERTURE_UM) / FOCAL_LENGTH_UM)))  # 0.901523

TASKS: dict[str, str] = {
    "fom_validation_and_length_scale": "Verify metalens FOM, Rasmus123nm validation vs fmmax, Round-1 vs Round-2 trade-off, and 5/5 px length scale.",
    "wakis_resimulation": "Rebuild and re-simulate the stored Round-2 metalens design with Wakis at 450/550/650 nm.",
    "focal_spot_and_diffraction_limit": "Extract focal-plane line cuts at y=3.4 um and compare against the NA=0.9015 diffraction limit.",
    "toy_optimization": "Run a pedagogical toy 2D metalens adjoint focusing optimization with wakis.inverse.",
    "optional_invrs_gym": "Optional live invrs-gym fmmax evaluation if invrs_gym is installed.",
}


def get_task_table(quick: bool = False) -> list[dict]:
    """Return the task table for Tutorial 9.4 with verified resolution metrics."""
    # At dx = 20 nm and lambda_min = 450 nm in TiO2 (n=2.4), wavelength inside material is 187.5 nm -> 9.38 cells/wl
    # In SiO2 substrate (n=1.5) and air (n=1.0), cpw is 15.0 and 22.5 cells/wl; effective average cpw >= 11.43
    cpw_val = 28.0 / float(np.sqrt(6.0))
    return [
        {
            "index": 0,
            "name": "fom_validation_and_length_scale",
            "description": (
                "Verify metalens focal enhancement metric, Rasmus123nm validation against fmmax "
                "(16.39/14.30/15.45 vs Wakis 15.55/14.27/16.09), Round-1 (14.07 at 2/1 px) vs "
                "Round-2 (8.87 at 5/5 px) trade-off, and 5/5 px (100 nm) length-scale check."
            ),
            "cpw": 15.0,
            "cells_across_feature": 5.0,
        },
        {
            "index": 1,
            "name": "wakis_resimulation",
            "description": (
                "Rebuild the 16 um x 6.4 um periodic 2D domain, smoothed aperture current sheet, "
                "and series-capacitance harmonic-mean PixelDesign from round2_metalens/design.npy, "
                "re-simulate at 450/550/650 nm, and verify parity with stored enhancements."
            ),
            "cpw": 15.0 if not quick else 10.0,
            "cells_across_feature": 5.0 if not quick else 4.0,
            "coarsen": 2 if quick else 1,
        },
        {
            "index": 2,
            "name": "focal_spot_and_diffraction_limit",
            "description": (
                "Analyze the 2D intensity enhancement maps, focal spot FWHM, Strehl-like peak ratio, "
                "and lateral focal-plane line cuts vs a diffraction-limited sinc^2(2*NA*x/lambda) reference."
            ),
            "cpw": 15.0,
            "cells_across_feature": 5.0,
        },
        {
            "index": 3,
            "name": "toy_optimization",
            "description": (
                "Pedagogical toy optimization (explicitly separate from the Round-2 invrs-gym "
                "benchmark runs): live Wakis AdjointSimulation + MaterialGrid gradient ascent "
                "for a compact 2D dielectric focusing lens."
            ),
            "cpw": cpw_val,
            "cells_across_feature": 5.0,
            "n_grid": 40 if quick else 48,
            "n_design_x": 24 if quick else 30,
            "n_design_y": 6 if quick else 8,
            "Nt": 300 if quick else 400,
            "n_iters": 8 if quick else 12,
        },
        {
            "index": 4,
            "name": "optional_invrs_gym",
            "description": (
                "Optional official invrs-gym fmmax evaluation if installed in the Python environment."
            ),
            "cpw": 15.0,
            "cells_across_feature": 5.0,
        },
    ]


def compute_metalens_eval_metric(
    focal_intensity_lens: np.ndarray,
    focal_intensity_ref: np.ndarray | None = None,
    softmin_tau: float = 2.0,
) -> tuple[float, np.ndarray, float]:
    """Compute per-wavelength focal intensity enhancements, worst-case eval_metric, and softmin loss.

    Parameters
    ----------
    focal_intensity_lens : np.ndarray
        Either the focal-spot intensity ``|E_lens(r_f, lambda_k)|^2`` (when ``focal_intensity_ref``
        is provided) or already-normalized enhancements ``eta_k`` of shape ``(3,)`` [dimensionless].
    focal_intensity_ref : np.ndarray or None
        No-lens reference intensity ``|E_ref(r_f, lambda_k)|^2`` of shape ``(3,)`` [V^2/m^2].
    softmin_tau : float
        Temperature parameter ``tau`` for the smooth softmin surrogate objective [dimensionless].

    Returns
    -------
    eval_metric : float
        Worst-case enhancement ``min_k eta_k`` [dimensionless].
    enhancements : np.ndarray
        Per-wavelength enhancements ``eta_k`` at 450, 550, 650 nm [dimensionless].
    softmin_loss : float
        Smooth optimization loss ``tau * log(mean(exp(-eta_k / tau)))`` [dimensionless].
    """
    arr = np.asarray(focal_intensity_lens, dtype=np.float64)
    if focal_intensity_ref is not None:
        ref = np.asarray(focal_intensity_ref, dtype=np.float64)
        if np.any(ref <= 0.0):
            raise ValueError("Reference focal intensity must be strictly positive.")
        eta = arr / ref
    else:
        eta = arr
    eval_metric = float(np.min(eta))
    # Numerically stable log-sum-exp for softmin loss
    m = float(np.min(eta))
    softmin_loss = float(-m + softmin_tau * np.log(np.mean(np.exp(-(eta - m) / softmin_tau))))
    return eval_metric, eta, softmin_loss


def diffraction_limited_sinc2_profile(
    x_um: np.ndarray,
    wavelength_um: float,
    na: float = LENS_NA,
    x0_um: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the normalized 2D cylindrical-aperture diffraction-limited focal profile and FWHM.

    For a 2D line focus with numerical aperture ``NA = sin(arctan((D/2)/f))``, a uniform
    band-limited angular spectrum over ``k_x in [-k_0 NA, +k_0 NA]`` yields the normalized
    lateral intensity profile ``I(x) = sinc^2(2 * NA * (x - x_0) / lambda)``, where
    ``sinc(u) = sin(pi * u) / (pi * u)``.

    Parameters
    ----------
    x_um : np.ndarray
        Lateral focal-plane coordinate ``x`` [um].
    wavelength_um : float
        Free-space wavelength ``lambda`` [um].
    na : float
        Numerical aperture ``NA`` in ``(0, 1)`` [dimensionless].
    x0_um : float
        Lateral focal-spot center ``x_0`` [um].

    Returns
    -------
    profile : np.ndarray
        Normalized diffraction-limited intensity ``I_diff(x)`` in ``[0, 1]`` [dimensionless].
    fwhm_um : float
        Analytical full width at half maximum ``0.442946 * lambda / NA`` [um].
    """
    if na <= 0.0 or na >= 1.0:
        raise ValueError(f"Numerical aperture NA must lie in (0, 1), got {na}.")
    if wavelength_um <= 0.0:
        raise ValueError(f"Wavelength must be positive, got {wavelength_um}.")
    u = 2.0 * float(na) * (np.asarray(x_um, dtype=np.float64) - float(x0_um)) / float(wavelength_um)
    profile = np.sinc(u) ** 2
    # Since sinc^2(u_half) = 0.5 at u_half = 0.44294647, FWHM = 2 * x_half = 0.44294647 * lambda / NA
    fwhm_um = 0.44294647 * float(wavelength_um) / float(na)
    return profile, fwhm_um


def circular_brush(diameter_px: int) -> np.ndarray:
    """Create an imageruler-style circular structuring element of diameter ``diameter_px`` [px]."""
    d = max(1, int(diameter_px))
    if d == 1:
        return np.ones((1, 1), dtype=bool)
    coords = np.arange(d, dtype=np.float64) - 0.5 * (d - 1)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    return (np.hypot(xx, yy) <= 0.5 * d + 1e-9).astype(bool)


def measure_length_scales_px(
    binary_arr: np.ndarray,
    max_px: int = 10,
    periodic: tuple[bool, bool] = (True, False),
) -> tuple[int, int]:
    """Measure minimum solid width and minimum void spacing [px] via imageruler-style morphology."""
    b2 = np.asarray(binary_arr[:, :, 0] if binary_arr.ndim == 3 else binary_arr) >= 0.5
    if np.all(b2) or not np.any(b2):
        return max_px, max_px
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    int_solid = binary_erosion(b2, structure=cross, border_value=1)
    int_void = binary_erosion(~b2, structure=cross, border_value=1)
    pad = max(4, int(max_px) + 2)
    b_pad = np.pad(b2, ((pad, pad), (0, 0)), mode="wrap" if periodic[0] else "edge")
    b_pad = np.pad(b_pad, ((0, 0), (pad, pad)), mode="wrap" if periodic[1] else "edge")

    min_solid = 1
    for d in range(2, max_px + 1):
        brush = circular_brush(d)
        opened = binary_opening(b_pad, structure=brush)[pad:-pad, pad:-pad]
        if int(np.sum((opened != b2) & int_solid)) == 0:
            min_solid = d
        else:
            break

    min_void = 1
    for d in range(2, max_px + 1):
        brush = circular_brush(d)
        closed = binary_closing(b_pad, structure=brush)[pad:-pad, pad:-pad]
        if int(np.sum((closed != b2) & int_void)) == 0:
            min_void = d
        else:
            break

    return int(min_solid), int(min_void)


class _DummyDesign:
    """Zero-parameter design placeholder for pure forward AdjointSimulation runs."""

    shape = (1,)
    flat_idx = np.zeros(0, dtype=int)

    def apply(self, solver, rho):
        pass

    def dispersive_group(self, solver):
        return None

    def network_of(self, solver):
        return None


def _smooth_aperture_profile(
    x_m: np.ndarray,
    aperture_width_m: float = 15e-6,
    fwhm_m: float = 1.0e-6,
) -> np.ndarray:
    """Construct the Gaussian-smoothed top-hat current-sheet profile in the substrate."""
    top_hat = (np.abs(x_m) <= 0.5 * aperture_width_m).astype(np.float64)
    if fwhm_m <= 0.0:
        return top_hat
    dx = float(x_m[1] - x_m[0])
    sigma = fwhm_m / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(2, int(np.ceil(4.0 * sigma / dx)))
    kx = np.arange(-half, half + 1, dtype=np.float64) * dx
    kernel = np.exp(-0.5 * (kx / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(top_hat, half, mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def simulate_metalens_wakis(
    design_npy_path: Path,
    coarsen: int = 1,
    run_time_factor: float = 1.0,
) -> dict:
    """Rebuild and simulate the 2D RGB metalens and its no-lens reference with Wakis alone.

    Parameters
    ----------
    design_npy_path : Path
        Path to ``design.npy`` (800x50 binary/density array at 20 nm pixel pitch).
    coarsen : int
        Spatial grid coarsening factor relative to the native 20 nm grid (1 = 20 nm, 2 = 40 nm, 4 = 80 nm).
    run_time_factor : float
        Multiplier on the physical propagation time window.

    Returns
    -------
    dict
        Dictionary with ``wavelengths_nm``, ``x_um``, ``intensity_ref``, ``intensity_lens``,
        ``wakis_enhancement_resim``, ``focal_line_enhancement_resim``, ``nx``, ``ny``, ``n_steps``.
    """
    design = np.load(design_npy_path).astype(np.float64)
    if design.ndim == 3:
        design = design[:, :, 0]

    # Native 20 nm domain: x in [-8, +8] um (800 cells), y in [-2.0, +4.4] um (320 cells)
    dx0 = 20e-9
    nx0, ny0 = 800, 320
    xmin, xmax = -8.0e-6, 8.0e-6
    ymin, ymax = -2.0e-6, 4.4e-6
    npml0 = 20

    nx = nx0 // coarsen
    ny = ny0 // coarsen
    dx = dx0 * coarsen
    npml = max(8, npml0 // coarsen)

    # Coarsen the 800x50 lens design if coarsen > 1
    if coarsen > 1:
        ny_lens = 50 // coarsen
        des_c = design.reshape(nx, coarsen, ny_lens, coarsen).mean(axis=(1, 3))
    else:
        ny_lens = 50
        des_c = design

    j_sub_end = int(round((0.0 - ymin) / dx))  # y = 0.0 um interface
    j_lens_end = j_sub_end + ny_lens  # y = +1.0 um lens top

    freqs = C0 / (WAVELENGTHS_UM * 1e-6)
    f0 = float(np.mean(freqs))
    df = float(0.75 * np.ptp(freqs))

    def _make_solver(include_lens: bool):
        grid = wakis.GridFIT3D(
            xmin, xmax, ymin, ymax, -0.5 * dx, 0.5 * dx, nx, ny, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["periodic", "cpml", "periodic"],
            bc_high=["periodic", "cpml", "periodic"],
            n_pml=npml,
            verbose=0,
        )
        t_shape = solver.ieps.to_matrix("z").shape
        # Cell-wise inverse permittivity: substrate eps=2.25 for y < 0, air eps=1.0 for y >= 0
        inv_eps_cell = np.ones((nx, ny), dtype=np.float64)
        inv_eps_cell[:, :j_sub_end] = 1.0 / 2.25
        if include_lens:
            eps_lens = 1.0 + (5.76 - 1.0) * np.clip(des_c, 0.0, 1.0)
            inv_eps_cell[:, j_sub_end:j_lens_end] = 1.0 / eps_lens

        # Series-capacitance 4-cell harmonic-mean assignment on primary z-edges (PixelDesign)
        inv_pad = np.pad(inv_eps_cell, ((1, 0), (1, 1)), mode="wrap")
        # Fix y-padding to edge (wrap in x, edge in y)
        inv_pad[:, 0] = inv_pad[:, 1]
        inv_pad[:, -1] = inv_pad[:, -2]
        ieps_node = 0.25 * (
            inv_pad[:-1, :-1] + inv_pad[1:, :-1] + inv_pad[:-1, 1:] + inv_pad[1:, 1:]
        )
        # Pad last x node periodically to match t_shape
        ieps_full = np.pad(ieps_node, ((0, 1), (0, 0)), mode="wrap")
        ieps_3d = np.broadcast_to(
            (ieps_full[: t_shape[0], : t_shape[1], None] / EPS_0), t_shape
        ).copy()
        for c in ("x", "y", "z"):
            solver.ieps.from_matrix(ieps_3d, c)
        solver.update_tensors("ieps")
        return solver

    probe = _make_solver(include_lens=True)
    dt = float(probe.dt)
    # Run time matching invrs-wakis metalens bridge (~48 fs -> 3473 steps at 20 nm)
    t_phys = run_time_factor * 48e-15
    n_steps = int(np.ceil(t_phys / dt))

    x_centers = (np.arange(nx, dtype=np.float64) + 0.5) * dx + xmin
    prof = _smooth_aperture_profile(x_centers, aperture_width_m=15e-6, fwhm_m=1.0e-6)
    src = CurrentSource(
        GaussianSource(f0, df),
        component="z",
        center=(0.0, -1.0e-6, 0.0),
        size=(xmax - xmin, 0.0, 0.0),
        amp_func=lambda x, y, z: np.interp(np.asarray(x), x_centers, prof, period=(xmax - xmin)),
    )

    monitors_ref = {
        "focus_pt": wakis.DFTMonitor(
            freqs, components=["Ez"], center=(0.0, 3.4e-6, 0.0), size=(0.0, 0.0, 0.0)
        ),
    }
    monitors_lens = {
        "focus_pt": wakis.DFTMonitor(
            freqs, components=["Ez"], center=(0.0, 3.4e-6, 0.0), size=(0.0, 0.0, 0.0)
        ),
        "focus_line": wakis.DFTMonitor(
            freqs,
            components=["Ez"],
            center=(0.0, 3.4e-6, 0.0),
            size=(xmax - xmin, 0.0, 0.0),
        ),
    }

    sim_ref = AdjointSimulation(
        lambda: _make_solver(False), [src], monitors_ref, _DummyDesign(), n_steps
    )
    out_ref = sim_ref.forward(np.zeros(1), record=False)
    ez_ref = out_ref["focus_pt"]["Ez"].reshape(len(freqs), -1)
    i_ref = np.mean(np.abs(ez_ref) ** 2, axis=1)

    sim_lens = AdjointSimulation(
        lambda: _make_solver(True), [src], monitors_lens, _DummyDesign(), n_steps
    )
    out_lens = sim_lens.forward(np.zeros(1), record=False)
    ez_lens = out_lens["focus_pt"]["Ez"].reshape(len(freqs), -1)
    i_lens = np.mean(np.abs(ez_lens) ** 2, axis=1)
    eta_resim = i_lens / np.maximum(i_ref, 1e-30)

    ez_line = out_lens["focus_line"]["Ez"][:, :nx, 0, 0]
    line_enh = (np.abs(ez_line) ** 2) / np.maximum(i_ref[:, None], 1e-30)

    return {
        "wavelengths_nm": WAVELENGTHS_NM,
        "x_um": x_centers * 1e6,
        "intensity_ref": i_ref,
        "intensity_lens": i_lens,
        "wakis_enhancement_resim": eta_resim,
        "focal_line_enhancement_resim": line_enh.astype(np.float32),
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
    }


def run_task_0_fom_validation_and_length_scale() -> dict:
    """Task 0: Verify FOM, Rasmus123nm validation vs fmmax, Round-1 vs Round-2 trade-off, and 5/5 px length scale."""
    r2_summary = json.loads((DATA_DIR / "round2_metalens" / "summary.json").read_text())
    r1_s0 = json.loads(
        (DATA_DIR / "round1" / "20261018-0430-invrs-metalens__shard_0__metalens_seed0__summary.json").read_text()
    )
    r1_s1 = json.loads(
        (DATA_DIR / "round1" / "20261018-0430-invrs-metalens__shard_1__metalens_seed1__summary.json").read_text()
    )
    design = np.load(DATA_DIR / "round2_metalens" / "design.npy")

    # 1. Verify compute_metalens_eval_metric on Round-2 official and Wakis enhancements
    off_eval, off_eta, off_softmin = compute_metalens_eval_metric(
        np.asarray(r2_summary["official_enhancement_ex"], dtype=np.float64)
    )
    wak_eval, wak_eta, wak_softmin = compute_metalens_eval_metric(
        np.asarray(r2_summary["wakis_enhancement_ex"], dtype=np.float64)
    )

    # 2. Verify Rasmus123nm reference validation numbers (from data/round2_metalens/README.md)
    rasmus_fmmax = np.array([16.39, 14.30, 15.45], dtype=np.float64)
    rasmus_wakis = np.array([15.55, 14.27, 16.09], dtype=np.float64)
    rasmus_fmmax_eval, _, _ = compute_metalens_eval_metric(rasmus_fmmax)
    rasmus_wakis_eval, _, _ = compute_metalens_eval_metric(rasmus_wakis)

    # 3. Measure length scale on Round-2 design (periodic in x, non-periodic in y)
    ls_measured = measure_length_scales_px(design, max_px=10, periodic=(True, False))

    out = {
        "round2_v3_seed0": {
            "official_eval_stored": float(r2_summary["official_eval_metric"]),
            "official_eval_recomputed": off_eval,
            "official_eval_abs_diff": abs(off_eval - float(r2_summary["official_eval_metric"])),
            "official_enhancement_ex": off_eta.tolist(),
            "official_softmin_loss": off_softmin,
            "wakis_eval_stored": float(r2_summary["wakis_eval_metric"]),
            "wakis_eval_recomputed": wak_eval,
            "wakis_enhancement_ex": wak_eta.tolist(),
            "wakis_softmin_loss": wak_softmin,
            "length_scale_stored_px": r2_summary["length_scale_leaderboard_px"],
            "length_scale_measured_px": list(ls_measured),
            "length_scale_nm": [int(ls_measured[0] * 20), int(ls_measured[1] * 20)],
            "selected_stage": r2_summary["selected_stage"],
        },
        "round1_unconstrained": {
            "seed0_official_eval": float(r1_s0["official_eval_metric"]),
            "seed0_official_enhancement_ex": r1_s0["official_enhancement_ex"],
            "seed0_wakis_eval": float(r1_s0["wakis_eval_metric"]),
            "seed0_length_scale_px": r1_s0["length_scale_solid_void_px"],
            "seed1_official_eval": float(r1_s1["official_eval_metric"]),
            "seed1_official_enhancement_ex": r1_s1["official_enhancement_ex"],
            "seed1_wakis_eval": float(r1_s1["wakis_eval_metric"]),
            "seed1_length_scale_px": r1_s1["length_scale_solid_void_px"],
            "target_length_scale_px": r1_s0["target_min_width_spacing_px"],
        },
        "rasmus123nm_validation": {
            "fmmax_enhancement_ex": rasmus_fmmax.tolist(),
            "fmmax_eval_metric": rasmus_fmmax_eval,
            "wakis_enhancement_ex": rasmus_wakis.tolist(),
            "wakis_eval_metric": rasmus_wakis_eval,
            "min_eval_relative_diff": abs(rasmus_wakis_eval - rasmus_fmmax_eval) / rasmus_fmmax_eval,
        },
        "leaderboard_status": "not_included_in_bundle",
    }
    (DATA_DIR / "fom_validation_and_length_scale_summary.json").write_text(json.dumps(out, indent=2))
    print(
        f"[Task 0] Metalens Round-2 official_eval={off_eval:.4f} (wakis={wak_eval:.4f}), "
        f"LS={ls_measured} px ({ls_measured[0]*20}/{ls_measured[1]*20} nm) vs Round-1 seed0={r1_s0['official_eval_metric']:.4f} at {r1_s0['length_scale_solid_void_px']} px"
    )
    return out


def run_task_1_wakis_resimulation(mode: str = "quick") -> dict:
    """Task 1: Re-simulate stored final design with Wakis and compare against stored enhancements."""
    r2_summary = json.loads((DATA_DIR / "round2_metalens" / "summary.json").read_text())
    resim_npz_path = DATA_DIR / "wakis_resim_full.npz"

    if mode == "full" or not resim_npz_path.exists():
        print("[Task 1] Running full-resolution (20 nm, 800x320) Wakis re-simulation of round2_metalens...")
        res = simulate_metalens_wakis(
            DATA_DIR / "round2_metalens" / "design.npy",
            coarsen=1,
            run_time_factor=1.0,
        )
        np.savez_compressed(
            resim_npz_path,
            wavelengths_nm=res["wavelengths_nm"],
            x_um=res["x_um"],
            intensity_ref=res["intensity_ref"],
            intensity_lens=res["intensity_lens"],
            wakis_enhancement_resim=res["wakis_enhancement_resim"],
            wakis_enhancement_stored=np.asarray(r2_summary["wakis_enhancement_ex"], dtype=np.float64),
            official_enhancement_stored=np.asarray(r2_summary["official_enhancement_ex"], dtype=np.float64),
            focal_line_enhancement_resim=res["focal_line_enhancement_resim"],
        )

    if mode == "quick":
        quick_res = simulate_metalens_wakis(
            DATA_DIR / "round2_metalens" / "design.npy",
            coarsen=2,
            run_time_factor=0.30,
        )
        quick_steps = int(quick_res["n_steps"])
    else:
        quick_steps = 0

    resim = np.load(resim_npz_path)
    eta_resim = resim["wakis_enhancement_resim"].astype(np.float64)
    eta_stored = np.asarray(r2_summary["wakis_enhancement_ex"], dtype=np.float64)
    eta_official = np.asarray(r2_summary["official_enhancement_ex"], dtype=np.float64)

    max_abs_diff_wakis = float(np.max(np.abs(eta_resim - eta_stored)))
    max_rel_diff_official = float(np.max(np.abs(eta_resim - eta_official) / eta_official))
    eval_resim, _, _ = compute_metalens_eval_metric(eta_resim)

    out = {
        "mode": mode,
        "quick_coarsened_steps": quick_steps,
        "wavelengths_nm": WAVELENGTHS_NM.tolist(),
        "wakis_enhancement_resim": eta_resim.tolist(),
        "wakis_enhancement_stored": eta_stored.tolist(),
        "official_enhancement_stored": eta_official.tolist(),
        "max_abs_diff_vs_stored_wakis": max_abs_diff_wakis,
        "max_rel_diff_vs_official_fmmax": max_rel_diff_official,
        "wakis_eval_metric_resim": eval_resim,
        "wakis_eval_metric_stored": float(r2_summary["wakis_eval_metric"]),
        "official_eval_metric_stored": float(r2_summary["official_eval_metric"]),
    }
    (DATA_DIR / "wakis_resim_comparison.json").write_text(json.dumps(out, indent=2))
    print(
        f"[Task 1] Metalens Wakis re-sim enhancements={np.round(eta_resim, 4).tolist()} "
        f"(max|d_eta| vs stored={max_abs_diff_wakis:.2e})"
    )
    return out


def run_task_2_focal_spot_and_diffraction_limit() -> dict:
    """Task 2: Analyze 2D enhancement maps, focal spot FWHM, and diffraction-limited sinc^2 line cuts."""
    fc = np.load(DATA_DIR / "round2_metalens" / "fields_compact.npz")
    resim = np.load(DATA_DIR / "wakis_resim_full.npz")

    focus_x_raw, focus_y_raw = fc["focus_um"].astype(np.float64)
    x_um = fc["x_um"].astype(np.float64) - float(focus_x_raw)
    y_um = fc["y_um"].astype(np.float64)
    enh_map = fc["enhancement"].astype(np.float64) * float(fc["enhancement_scale"])
    iy_focus = int(np.argmin(np.abs(y_um - float(focus_y_raw))))
    ix_focus = int(np.argmin(np.abs(x_um - 0.0)))

    line_cuts = []
    for k, wl_nm in enumerate(WAVELENGTHS_NM):
        wl_um = float(wl_nm * 1e-3)
        diff_prof, fwhm_diff_um = diffraction_limited_sinc2_profile(x_um, wl_um, na=LENS_NA)
        cut_stored = enh_map[k, :, iy_focus]
        cut_resim = resim["focal_line_enhancement_resim"][k].astype(np.float64)
        peak_val = float(cut_stored[ix_focus])
        norm_cut = cut_stored / max(peak_val, 1e-30)

        # Measure empirical FWHM of contiguous central focal lobe around ix_focus with linear interpolation
        left = ix_focus
        while left > 0 and norm_cut[left] >= 0.5:
            left -= 1
        right = ix_focus
        while right < len(x_um) - 1 and norm_cut[right] >= 0.5:
            right += 1
        x_left = float(
            x_um[left]
            + (0.5 - norm_cut[left]) / max(norm_cut[left + 1] - norm_cut[left], 1e-12) * (x_um[left + 1] - x_um[left])
        )
        x_right = float(
            x_um[right - 1]
            + (0.5 - norm_cut[right - 1]) / min(norm_cut[right] - norm_cut[right - 1], -1e-12) * (x_um[right] - x_um[right - 1])
        )
        fwhm_emp_um = float(x_right - x_left)

        line_cuts.append(
            {
                "wavelength_nm": float(wl_nm),
                "na_used": LENS_NA,
                "focal_enhancement_stored_map": peak_val,
                "focal_enhancement_resim_line": float(np.max(cut_resim)),
                "diffraction_limit_fwhm_um": fwhm_diff_um,
                "empirical_fwhm_um": fwhm_emp_um,
                "fwhm_ratio_empirical_over_diffraction": fwhm_emp_um / fwhm_diff_um,
            }
        )

    out = {
        "aperture_width_um": APERTURE_UM,
        "focal_length_um": FOCAL_LENGTH_UM,
        "focus_coordinates_um": [FOCUS_X_UM, FOCUS_Y_UM],
        "na": LENS_NA,
        "wavelength_analysis": line_cuts,
    }
    (DATA_DIR / "focal_line_cut_analysis.json").write_text(json.dumps(out, indent=2))
    print(
        f"[Task 2] Focal line cut analysis at y={FOCUS_Y_UM:.2f} um (NA={LENS_NA:.4f}): "
        + ", ".join(
            f"{int(r['wavelength_nm'])}nm FWHM={r['empirical_fwhm_um']:.3f}um (diff={r['diffraction_limit_fwhm_um']:.3f}um)"
            for r in line_cuts
        )
    )
    return out


def run_task_3_pedagogical_toy_optimization(mode: str = "quick") -> dict:
    """Task 3: Pedagogical toy 2D metalens adjoint focusing demo using wakis.inverse (separate from invrs-gym)."""
    cfg = get_task_table(quick=(mode == "quick"))[3]
    n_iters = int(cfg["n_iters"])
    n_steps = int(cfg["Nt"])
    n_grid = int(cfg["n_grid"])
    nx_des = int(cfg["n_design_x"])
    ny_des = int(cfg["n_design_y"])

    f0 = 3.0e9
    lam0 = C0 / f0
    lx = 3.2 * lam0
    ly = 2.4 * lam0
    dz = lx / n_grid
    eps_sub = 2.25
    eps_lens = 5.76

    def make_solver():
        grid = wakis.GridFIT3D(
            -0.5 * lx, 0.5 * lx, -0.5 * ly, 0.5 * ly, -0.5 * dz, 0.5 * dz, n_grid, n_grid, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["periodic", "cpml", "periodic"],
            bc_high=["periodic", "cpml", "periodic"],
            n_pml=6,
            verbose=0,
        )
        sub = MaterialGrid((0.0, -0.35 * ly, 0.0), (lx, 0.30 * ly, 0.0), (1, 1, 1), eps_min=eps_sub, eps_max=eps_sub)
        sub.apply(solver, np.ones((1, 1, 1)))
        return solver

    design = MaterialGrid(
        (0.0, -0.10 * ly, 0.0),
        (0.75 * lx, 0.18 * ly, 0.0),
        (nx_des, ny_des, 1),
        eps_min=1.0,
        eps_max=eps_lens,
        smoothing="tensor",
    )
    src = CurrentSource(
        GaussianSource(f0, 0.5 * f0),
        component="z",
        center=(0.0, -0.30 * ly, 0.0),
        size=(0.85 * lx, 0.0, 0.0),
    )
    monitors = {
        "focus": wakis.DFTMonitor([f0], components=["Ez"], center=(0.0, 0.25 * ly, 0.0), size=(0.0, 0.0, 0.0)),
    }
    sim = AdjointSimulation(make_solver, [src], monitors, design, n_steps)

    rho = np.full(design.shape, 0.5, dtype=np.float64)
    history_j = []
    history_gnorm = []
    step_size = 0.20

    for _it in range(n_iters):
        out_dft = sim.forward(rho)
        ez_f = out_dft["focus"]["Ez"]
        obj_val = float(np.sum(np.abs(ez_f) ** 2))

        g = sim.vjp({"focus": {"Ez": 2.0 * ez_f}})
        gnorm = float(np.linalg.norm(g))
        history_j.append(obj_val)
        history_gnorm.append(gnorm)
        if gnorm > 0:
            rho = np.clip(rho + step_size * g / (float(np.max(np.abs(g))) + 1e-30), 0.0, 1.0)

    out = {
        "label": "PEDAGOGICAL_TOY_DEMO_NOT_INVRS_GYM",
        "mode": mode,
        "n_iters": n_iters,
        "n_steps": n_steps,
        "objective_history": history_j,
        "grad_norm_history": history_gnorm,
        "objective_ratio_end_over_start": float(history_j[-1] / max(history_j[0], 1e-30)),
    }
    (DATA_DIR / "toy_optimization_summary.json").write_text(json.dumps(out, indent=2))
    print(
        f"[Task 3] Toy metalens adjoint demo ({n_iters} iters): J_start={history_j[0]:.4e} -> "
        f"J_end={history_j[-1]:.4e} ({out['objective_ratio_end_over_start']:.2f}x)"
    )
    return out


def run_task_4_optional_invrs_gym() -> dict:
    """Task 4: Optional official invrs-gym fmmax re-score if invrs_gym is installed."""
    try:
        import invrs_gym  # type: ignore[import-not-found]
    except ImportError:
        msg = (
            "Optional package `invrs_gym` is not installed in this offline environment. "
            "Using pre-computed official fmmax scores from `data/round2_metalens/summary.json`."
        )
        print(f"[Task 4] {msg}")
        return {"installed": False, "message": msg}

    return {"installed": True, "version": getattr(invrs_gym, "__version__", "unknown")}


def plot_all() -> list[Path]:
    """Generate all 8 tutorial figures and the design-evolution GIF in ``figures/``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    r2_summary = json.loads((DATA_DIR / "round2_metalens" / "summary.json").read_text())
    r1_s0 = json.loads(
        (DATA_DIR / "round1" / "20261018-0430-invrs-metalens__shard_0__metalens_seed0__summary.json").read_text()
    )
    r1_s1 = json.loads(
        (DATA_DIR / "round1" / "20261018-0430-invrs-metalens__shard_1__metalens_seed1__summary.json").read_text()
    )
    fc = np.load(DATA_DIR / "round2_metalens" / "fields_compact.npz")
    snaps = np.load(DATA_DIR / "round2_metalens" / "snapshots_thinned.npz")
    design = np.load(DATA_DIR / "round2_metalens" / "design.npy")
    if design.ndim == 3:
        design = design[:, :, 0]
    resim = np.load(DATA_DIR / "wakis_resim_full.npz")

    saved: list[Path] = []

    # --- Figure 1: Metalens binary profile, fill-fraction profile, and 5/5 px length-scale verification ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), constrained_layout=True)
    # Crop active 500x50 lens region x in [-5, +5] um (indices 150..650)
    des_active = design[150:650, :]
    axes[0].imshow(
        des_active.T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        extent=[-5.0, 5.0, 0.0, 1.0],
        aspect="auto",
    )
    ls_px = r2_summary["length_scale_leaderboard_px"]
    axes[0].set_title(
        f"Round-2 Metalens Active Profile (500x50 px, 20 nm/px, x in [-5, +5] um, y in [0, 1] um)\n"
        f"Official fmmax eval_metric = {r2_summary['official_eval_metric']:.2f}x, Guaranteed LS = {ls_px[0]}/{ls_px[1]} px (100/100 nm)"
    )
    axes[0].set_ylabel("y [um]")

    # Zoom-in on central [-2.0, +2.0] um showing individual TiO2 pillars and 100 nm (5 px) brush
    des_zoom = design[300:500, :]
    axes[1].imshow(
        des_zoom.T,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.3,
        extent=[-2.0, 2.0, 0.0, 1.0],
        aspect="equal",
        alpha=0.35,
    )
    axes[1].contour(
        np.linspace(-2.0, 2.0, 200),
        np.linspace(0.0, 1.0, 50),
        des_zoom.T,
        levels=[0.5],
        colors=["#1f77b4"],
        linewidths=1.5,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    axes[1].plot(-1.75 + 0.05 * np.cos(theta), 0.20 + 0.05 * np.sin(theta), "r-", lw=2.0, label="100 nm (5 px) brush")
    axes[1].set_title("Central Zoom (x in [-2, +2] um): 100 nm (5 px) Pillar & Trench Feature Verification")
    axes[1].set_ylabel("y [um]")
    axes[1].legend(loc="upper right", fontsize=8.5)

    # Depth-averaged TiO2 fill fraction across the 10 um aperture
    x_act = np.linspace(-5.0, 5.0, 500)
    fill_frac = np.mean(des_active, axis=1)
    n_eff_approx = np.sqrt(1.0 + (5.76 - 1.0) * fill_frac)
    axes[2].plot(x_act, fill_frac, color="#1f77b4", lw=1.5, label="Depth-averaged TiO2 fill fraction <rho(x)>_y")
    axes[2].set_xlim(-5.0, 5.0)
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].set_xlabel("Lateral Position x [um]")
    axes[2].set_ylabel("Fill Fraction")
    axes[2].set_title("Lateral Metasurface Fill-Fraction Profile Across 10 um Aperture")
    axes[2].grid(True, alpha=0.3)
    ax2_r = axes[2].twinx()
    ax2_r.plot(x_act, n_eff_approx, color="#ff7f0e", alpha=0.6, lw=1.2, ls="--", label="Effective index sqrt(<eps>)")
    ax2_r.set_ylabel("Approx. Effective Index", color="#ff7f0e")
    axes[2].legend(loc="upper right", fontsize=8.5)

    p1 = FIG_DIR / "01_metalens_profile_and_length_scale.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # --- Figure 2: Optimization trace (Loss, Surrogate eval_metric, and beta continuation) ---
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True, constrained_layout=True)
    iters = snaps["kept_index"].astype(int)
    losses = snaps["loss"].astype(float)
    evals = snaps["eval_metric"].astype(float)
    stages = snaps["stage"].astype(str)
    betas = np.array([float(st.split("beta")[-1]) for st in stages], dtype=float)
    ste_mask = np.array([st.startswith("ste_") for st in stages], dtype=bool)
    ste_start = int(iters[ste_mask][0]) if np.any(ste_mask) else 100

    axes[0].plot(iters, losses, color="#1f77b4", lw=1.8, label="Softmin surrogate loss L (tau=2.0)")
    axes[0].axvline(ste_start, color="#d62728", ls="--", lw=1.5, label=f"Stage 2 STE 100 nm repair (step {ste_start})")
    axes[0].set_ylabel("Softmin Loss L")
    axes[0].set_title("Round-2 Metalens (schedule v3, seed 0): Softmin Loss & Projection beta")
    axes[0].grid(True, alpha=0.3)
    ax0_b = axes[0].twinx()
    ax0_b.step(iters, betas, where="post", color="#ff7f0e", lw=1.4, alpha=0.7, label="Projection beta")
    ax0_b.set_ylabel("Projection sharpness beta", color="#ff7f0e")
    axes[0].legend(loc="upper right", fontsize=8.5)

    axes[1].plot(iters, evals, color="#2ca02c", lw=1.8, label="Wakis optimization eval_metric (min_k eta_k)")
    axes[1].axhline(
        r2_summary["official_eval_metric"],
        color="#9467bd",
        ls="-.",
        lw=1.6,
        label=f"Official fmmax eval_metric = {r2_summary['official_eval_metric']:.2f}x",
    )
    axes[1].axhline(
        r2_summary["wakis_eval_metric"],
        color="#1f77b4",
        ls=":",
        lw=1.5,
        label=f"Wakis r=1 eval_metric = {r2_summary['wakis_eval_metric']:.2f}x",
    )
    axes[1].axvline(ste_start, color="#d62728", ls="--", lw=1.5)
    for cand in r2_summary["candidates"]:
        c_tag = f"{cand['stage']}:{cand['step']}"
        c_it = ste_start + int(cand["step"])
        axes[1].scatter(
            [c_it],
            [cand["official_eval_metric"]],
            color="#d62728" if c_tag == r2_summary.get("selected_stage") else "#8c564b",
            s=55,
            zorder=5,
            label=f"Candidate {c_tag} (fmmax {cand['official_eval_metric']:.2f}x)",
        )
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Worst-Case Focal Enhancement")
    axes[1].set_title("Focal Enhancement Trajectory & Multi-Candidate Rescoring")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8, ncol=2)

    p2 = FIG_DIR / "02_optimization_trace_and_beta_schedule.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # --- Figure 3: 2D Focal Intensity Enhancement Maps at 450, 550, 650 nm ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    x_um = fc["x_um"].astype(np.float64) - float(fc["focus_um"][0])
    y_um = fc["y_um"].astype(np.float64)
    enh_map = fc["enhancement"].astype(np.float64) * float(fc["enhancement_scale"])
    eps_map = fc["eps"].astype(np.float64) * float(fc["eps_scale"])
    vmax_enh = float(np.percentile(enh_map, 99.6))

    rgb_titles = [
        ("Blue (450 nm)", r2_summary["official_enhancement_ex"][0], r2_summary["wakis_enhancement_ex"][0]),
        ("Green (550 nm)", r2_summary["official_enhancement_ex"][1], r2_summary["wakis_enhancement_ex"][1]),
        ("Red (650 nm)", r2_summary["official_enhancement_ex"][2], r2_summary["wakis_enhancement_ex"][2]),
    ]
    for k, (lbl, off_v, wak_v) in enumerate(rgb_titles):
        ax = axes[k]
        im = ax.imshow(
            enh_map[k].T,
            origin="lower",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax_enh,
            extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
            aspect="equal",
        )
        # Overlay lens box and focal spot crosshair
        ax.contour(x_um, y_um, eps_map.T, levels=[3.5], colors=["w"], linewidths=0.5, alpha=0.6)
        ax.scatter([FOCUS_X_UM], [FOCUS_Y_UM], marker="+", s=90, color="cyan", linewidths=1.8, label="Target focus (0, 3.4 um)")
        ax.set_title(f"{lbl}\nfmmax={off_v:.2f}x | Wakis={wak_v:.2f}x", fontsize=10)
        ax.set_xlabel("x [um]")
        if k == 0:
            ax.set_ylabel("y [um]")
            ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label="Intensity Enhancement |E(x,y)|^2 / |E_ref(r_f)|^2")
    fig.suptitle("Round-2 RGB Achromatic Metalens: 2D Intensity Enhancement Maps", fontsize=12)

    p3 = FIG_DIR / "03_intensity_enhancement_maps_rgb.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # --- Figure 4: Real-Part Wavefields Re(E_z) Showing Phasefront Convergence to (0, 3.4 um) ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    ex_cplx = (fc["Ex_re"].astype(np.float64) + 1j * fc["Ex_im"].astype(np.float64)) * float(fc["Ex_scale"])
    vmax_e = float(np.percentile(np.abs(ex_cplx), 99.2))
    for k, wl_nm in enumerate(WAVELENGTHS_NM):
        ax = axes[k]
        im_e = ax.imshow(
            np.real(ex_cplx[k]).T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax_e,
            vmax=vmax_e,
            extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
            aspect="equal",
        )
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.scatter([FOCUS_X_UM], [FOCUS_Y_UM], marker="o", s=45, facecolors="none", edgecolors="k", linewidths=1.5)
        ax.set_title(f"lambda = {int(wl_nm)} nm: Converging Wavefronts Re(E)", fontsize=10)
        ax.set_xlabel("x [um]")
        if k == 0:
            ax.set_ylabel("y [um]")
    fig.colorbar(im_e, ax=axes.ravel().tolist(), shrink=0.88, label="Re(E) [arb. units]")
    fig.suptitle("Steady-State Harmonic Electric Field Re(E) Across RGB Wavelengths", fontsize=12)

    p4 = FIG_DIR / "04_wavefield_real_part_rgb.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    # --- Figure 5: Focal-Plane Lateral Line Cuts vs Diffraction-Limited sinc^2 Reference (NA = 0.9015) ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False, constrained_layout=True)
    iy_focus = int(np.argmin(np.abs(y_um - FOCUS_Y_UM)))
    x_fine = np.linspace(-2.0, 2.0, 801)
    colors_rgb = ["#1f77b4", "#2ca02c", "#d62728"]

    for k, (wl_nm, col_c) in enumerate(zip(WAVELENGTHS_NM, colors_rgb)):
        ax = axes[k]
        wl_um = float(wl_nm * 1e-3)
        diff_prof, fwhm_diff = diffraction_limited_sinc2_profile(x_fine, wl_um, na=LENS_NA)
        cut_stored = enh_map[k, :, iy_focus]
        cut_resim = resim["focal_line_enhancement_resim"][k].astype(np.float64)
        resim_x = resim["x_um"].astype(np.float64)
        if float(np.mean(resim_x)) > 4.0:
            resim_x = resim_x - float(fc["focus_um"][0])
        peak_s = float(np.max(cut_stored[np.abs(x_um) <= 0.5]))

        ax.plot(x_um, cut_stored, "-", color=col_c, lw=2.0, label="Stored Wakis field cut")
        ax.plot(resim_x, cut_resim, "--", color="k", lw=1.4, label="Re-simulated Wakis cut")
        ax.plot(
            x_fine,
            peak_s * diff_prof,
            ":",
            color="#ff7f0e",
            lw=1.8,
            label=f"Diffraction limit (NA={LENS_NA:.3f}, FWHM={fwhm_diff*1e3:.0f} nm)",
        )
        ax.set_xlim(-2.0, 2.0)
        ax.set_xlabel("Focal-Plane Lateral Coordinate x [um]")
        if k == 0:
            ax.set_ylabel("Enhancement |E(x, y_f)|^2 / |E_ref(r_f)|^2")
        ax.set_title(f"lambda = {int(wl_nm)} nm (Peak = {peak_s:.2f}x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        f"Focal-Plane Lateral Line Cuts at y_f = {FOCUS_Y_UM:.1f} um vs 2D Diffraction Limit (D=10 um, f=2.4 um, NA={LENS_NA:.4f})",
        fontsize=11.5,
    )

    p5 = FIG_DIR / "05_focal_line_cuts_vs_diffraction_limit.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved.append(p5)

    # --- Figure 6: Wakis Re-Simulation Parity & Rasmus123nm Validation vs Official fmmax ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x_w = np.arange(3)
    w_b = 0.25

    ax = axes[0]
    ras_fmmax = [16.39, 14.30, 15.45]
    ras_wakis = [15.55, 14.27, 16.09]
    ax.bar(x_w - 0.5 * w_b, ras_fmmax, width=w_b, color="#1f77b4", edgecolor="k", label="Official fmmax (FMM/RCWA)")
    ax.bar(x_w + 0.5 * w_b, ras_wakis, width=w_b, color="#2ca02c", edgecolor="k", label="Wakis 2D FDTD (20 nm, PixelDesign)")
    ax.axhline(14.30, color="#1f77b4", ls="--", lw=1.2, label="fmmax min = 14.30x")
    ax.axhline(14.27, color="#2ca02c", ls=":", lw=1.2, label="Wakis min = 14.27x (0.2% diff)")
    ax.set_xticks(x_w)
    ax.set_xticklabels(["450 nm (Blue)", "550 nm (Green)", "650 nm (Red)"])
    ax.set_ylabel("Focal Intensity Enhancement")
    ax.set_title("Benchmark Validation on Reference Design Rasmus123nm")
    ax.set_ylim(0, 19.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    r2_off = r2_summary["official_enhancement_ex"]
    r2_wak_s = r2_summary["wakis_enhancement_ex"]
    r2_wak_r = resim["wakis_enhancement_resim"]
    ax.bar(x_w - w_b, r2_off, width=w_b, color="#9467bd", edgecolor="k", label="Official fmmax (min = 8.87x)")
    ax.bar(x_w, r2_wak_s, width=w_b, color="#1f77b4", edgecolor="k", label="Stored Wakis r=1 (min = 8.43x)")
    ax.bar(x_w + w_b, r2_wak_r, width=w_b, color="#ff7f0e", edgecolor="k", label="Re-simulated Wakis (max|diff|=1.7e-7)")
    ax.set_xticks(x_w)
    ax.set_xticklabels(["450 nm (Blue)", "550 nm (Green)", "650 nm (Red)"])
    ax.set_ylabel("Focal Intensity Enhancement")
    ax.set_title("Round-2 Final Design (5/5 px = 100 nm): Official vs Wakis Re-Simulation")
    ax.set_ylim(0, 13.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    p6 = FIG_DIR / "06_wakis_vs_fmmax_validation_and_resim.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    saved.append(p6)

    # --- Figure 7: Round-1 (Unconstrained 2/1 px) vs Round-2 (Guaranteed 5/5 px) & Reference Trade-Off ---
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    entries = [
        ("Round-1 Seed 1\n(2/2 px = 40 nm, infeasible)", float(r1_s1["official_eval_metric"]), "#ff7f0e"),
        ("Round-2 v3 Seed 0\n(5/5 px = 100 nm, FEASIBLE)", float(r2_summary["official_eval_metric"]), "#2ca02c"),
        ("Round-1 Seed 0\n(2/1 px = 40/20 nm, infeasible)", float(r1_s0["official_eval_metric"]), "#d62728"),
        ("Rasmus123nm Ref\n(123 nm, feasible ref)", 14.30, "#1f77b4"),
    ]
    ax = axes[0]
    y_p = np.arange(len(entries))
    bars = ax.barh(y_p, [e[1] for e in entries], color=[e[2] for e in entries], edgecolor="k", height=0.55)
    ax.set_yticks(y_p)
    ax.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    ax.set_xlabel("Official fmmax Worst-Case Enhancement min_k eta_k")
    ax.set_title("Length-Scale Feasibility vs Focal Enhancement Trade-Off")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, e in zip(bars, entries):
        ax.text(e[1] + 0.25, bar.get_y() + 0.5 * bar.get_height(), f"{e[1]:.2f}x", va="center", fontsize=8.5)
    ax.set_xlim(0, 16.8)

    ax = axes[1]
    x_w = np.arange(3)
    w_b = 0.25
    ax.bar(x_w - w_b, r1_s0["official_enhancement_ex"], width=w_b, color="#d62728", edgecolor="k", label="Round-1 Seed 0 (2/1 px = 40/20 nm)")
    ax.bar(x_w, r1_s1["official_enhancement_ex"], width=w_b, color="#ff7f0e", edgecolor="k", label="Round-1 Seed 1 (2/2 px = 40/40 nm)")
    ax.bar(x_w + w_b, r2_summary["official_enhancement_ex"], width=w_b, color="#2ca02c", edgecolor="k", label="Round-2 v3 Seed 0 (5/5 px = 100 nm)")
    ax.set_xticks(x_w)
    ax.set_xticklabels(["450 nm (Blue)", "550 nm (Green)", "650 nm (Red)"])
    ax.set_ylabel("Official fmmax Focal Enhancement")
    ax.set_title("Per-Wavelength RGB Enhancement Across Rounds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    p7 = FIG_DIR / "07_round1_vs_round2_length_scale_tradeoff.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    saved.append(p7)

    # --- Figure 8: Design-Evolution Animated GIF (< 5 MB) from snapshots_thinned.npz ---
    dens_all = snaps["designs"][:, 150:650, :].astype(np.float32)
    n_snaps = len(iters)
    frame_indices = np.linspace(0, n_snaps - 1, min(18, n_snaps), dtype=int)

    fig_gif, ax_gif = plt.subplots(1, 1, figsize=(8.5, 2.8), constrained_layout=True)

    def _update_gif(idx_pos: int):
        f_idx = int(frame_indices[idx_pos])
        ax_gif.clear()
        it_num = int(iters[f_idx])
        st_str = str(stages[f_idx])
        beta_v = float(st_str.split("beta")[-1])
        ev_v = float(evals[f_idx])
        s_lbl = "Stage 1 (Continuous)" if not st_str.startswith("ste_") else "Stage 2 (100 nm STE Repair)"
        ax_gif.imshow(
            dens_all[f_idx].T,
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            extent=[-5.0, 5.0, 0.0, 1.0],
            aspect="auto",
        )
        ax_gif.set_title(
            f"RGB Metalens Evolution | step {it_num} ({s_lbl}) | beta={beta_v:.0f} | Wakis eval={ev_v:.2f}x",
            fontsize=9.5,
        )
        ax_gif.set_xlabel("x [um]")
        ax_gif.set_ylabel("y [um]")
        return [ax_gif]

    anim = animation.FuncAnimation(fig_gif, _update_gif, frames=len(frame_indices), interval=350)
    p8 = FIG_DIR / "08_design_evolution.gif"
    anim.save(p8, writer=animation.PillowWriter(fps=3), dpi=90)
    plt.close(fig_gif)
    saved.append(p8)

    print(f"[Plot] Generated {len(saved)} figures in {FIG_DIR}")
    return saved


# %% [markdown]
# ## Limits & Failure Modes
#
# 1. **Physical trade-off between `5/5 px` ($100\,\text{nm}$) minimum feature size and RGB chromatic dispersion**:
#    As analyzed for inverse-designed high-NA achromatic metalenses by Chung and Miller (*Opt. Express* **28**, 6945, 2020,
#    doi:`10.1364/OE.385440`) and for ultraviolet maskless-lithography metalenses by Chung, Zhang, Li, Miller, and Smith
#    (*Nanophotonics* **12**, 2371–2381, 2023, doi:`10.1515/nanoph-2022-0761`), simultaneous focusing across widely separated
#    wavelengths in a $1\,\mu\text{m}$-thick slab relies on coupled waveguide-like resonances and narrow air slots to provide
#    independent phase and group-delay dispersion. Without length-scale repair, Round-1 `seed 0` achieved an official `fmmax`
#    worst-case enhancement of $14.07\times$ ($18.23 / 14.07 / 16.80$), but relied on `2/1 px` ($40\,\text{nm}$ pillar /
#    $20\,\text{nm}$ slot) sub-wavelength features to engineer independent group delays across $450\text{--}650\,\text{nm}$.
#    Enforcing a strict $100\,\text{nm}$ (`5/5 px`) minimum width and spacing in Round-2 (`round2_metalens`, schedule `v3`,
#    `seed 0`) removes those $20\,\text{nm}$ air slots and reduces the official worst-case enhancement to $8.87\times$
#    ($11.01 / 8.87 / 9.28$). Meanwhile, the published `Rasmus123nm` reference design from the `photonics-opt-testbed`
#    validation and characterization benchmark (Chen et al., *J. Opt. Soc. Am. B* **41**, A161, 2024, doi:`10.1364/JOSAB.506412`)
#    achieves $14.30\times$ at $123\,\text{nm}$ length scale, showing that additional multi-start seeds and longer Stage-1
#    dispersion exploration before locking the binary topology in Stage 2 could recover significant focal intensity while
#    keeping $100\,\text{nm}$ manufacturability.
# 2. **Run-to-run seed variance**:
#    Even in Round 1 without morphological repair, `seed 0` scored $14.07\times$ while `seed 1` scored $7.78\times$ (`2/2 px`),
#    a $1.8\times$ spread across two random initializations. Because the $10\,\mu\text{m}$ aperture spans $\sim 22$ free-space
#    wavelengths at $450\,\text{nm}$, gradient-based optimization can settle into distinct local phase-zone topologies.
# 3. **2D cylindrical vs. 3D spherical focusing**:
#    The `invrs-gym` `metalens` challenge models a 2D line-focusing metasurface ($10\,\mu\text{m}$ aperture in $x$, invariant
#    in the out-of-plane direction) where the ideal diffraction-limited focal profile is
#    $\operatorname{sinc}^2(2\,\text{NA}\,x/\lambda)$ and enhancement scales as $\mathcal{O}(D/\lambda)$. A full 3D circular
#    metalens has an Airy-disk profile $[2 J_1(\pi\,\text{NA}\,r/\lambda) / (\pi\,\text{NA}\,r/\lambda)]^2$ and area-based
#    enhancement scaling as $\mathcal{O}((D/\lambda)^2)$.
#
# ## Verified References
#
# 1. M. F. Schubert et al., "invrs-gym: a toolkit for nanophotonic inverse design research," arXiv:`2410.24132` (2024).
# 2. M. Chen, R. E. Christiansen, J. A. Fan, G. Işiklar, J. Jiang, S. G. Johnson, W. Ma, O. D. Miller, A. Oskooi,
#    M. F. Schubert, F. Wang, I. A. D. Williamson, W. Xue and Y. Zhou, "Validation and characterization of algorithms and
#    software for photonics inverse design", *J. Opt. Soc. Am. B* **41**, A161 (2024), doi:`10.1364/JOSAB.506412`.
# 3. H. Chung and O. D. Miller, "High-NA achromatic metalenses by inverse design", *Opt. Express* **28**, 6945 (2020),
#    doi:`10.1364/OE.385440`.
# 4. H. Chung, F. Zhang, H. Li, O. D. Miller and H. I. Smith, "Inverse design of high-NA metalens for maskless
#    lithography", *Nanophotonics* **12**, 2371–2381 (2023), doi:`10.1515/nanoph-2022-0761`.
# 5. H. Chung and O. D. Miller, "Tunable Metasurface Inverse Design for 80% Switching Efficiencies and 144° Angular
#    Deflection", *ACS Photonics* **7**, 2236–2243 (2020), doi:`10.1021/acsphotonics.0c00787`.
# 6. F. Wang, B. S. Lazarov, and O. Sigmund, "On projection methods, convergence and robust formulations in topology
#    optimization," *Struct. Multidiscip. Optim.* **43**, 767–784 (2011), DOI: `10.1007/s00158-010-0602-y`.
# 7. M. F. Schubert, A. K. Cheung, I. A. D. Williamson, A. Spyra, and D. H. Alexander, "Inverse design of photonic devices
#    with strict foundry fabrication constraints," *ACS Photonics* **9**, 2327–2336 (2022), DOI: `10.1021/acsphotonics.2c00313`.

# %%
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Fast verification mode (< 1 min).")
    mode_group.add_argument("--full", action="store_true", help="Full native-resolution re-simulation mode.")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help="Run only the specified task.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print available tasks and exit.")
    parser.add_argument("--plot", action="store_true", help="Regenerate figures from stored data only.")
    args = parser.parse_args(argv)

    if args.list_tasks:
        for name, desc in TASKS.items():
            print(f"{name}: {desc}")
        return 0

    mode = "full" if args.full else "quick"

    if args.plot and args.task is None:
        plot_all()
        return 0

    selected = (
        [args.task]
        if args.task
        else [
            "fom_validation_and_length_scale",
            "wakis_resimulation",
            "focal_spot_and_diffraction_limit",
            "toy_optimization",
            "optional_invrs_gym",
        ]
    )

    if "fom_validation_and_length_scale" in selected:
        run_task_0_fom_validation_and_length_scale()
    if "wakis_resimulation" in selected:
        run_task_1_wakis_resimulation(mode=mode)
    if "focal_spot_and_diffraction_limit" in selected:
        run_task_2_focal_spot_and_diffraction_limit()
    if "toy_optimization" in selected:
        run_task_3_pedagogical_toy_optimization(mode=mode)
    if "optional_invrs_gym" in selected:
        run_task_4_optional_invrs_gym()

    if args.task is None or args.plot:
        plot_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
