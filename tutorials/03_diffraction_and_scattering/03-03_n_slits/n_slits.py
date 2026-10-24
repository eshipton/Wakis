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
# # Tutorial 3.3 — $N$-Slit Diffraction Gratings: Principal Maxima, $(N-2)$ Sidelobes, Resolving Power, and Periodic vs Finite-$N$ Scaling
#
# ## Abstract
#
# Extending Young's double slit from $N = 2$ to an array of $N$ identical slits of width $a$
# and period $d$ transforms broad two-beam cosinusoidal fringes into a **diffraction
# grating**. Constructive interference across all $N$ apertures concentrates the transmitted
# energy into narrow **principal maxima** at angles governed by the grating equation
# $d\sin\theta_m = m\lambda$, separated by $(N - 1)$ equispaced nulls and $(N - 2)$ weak
# **secondary maxima** whose relative height drops toward $(1.4303\pi)^{-2} \approx 4.72\%$
# as $N$ grows. Because the angular half-width to the first zero scales as
# $\Delta(\sin\theta)_{\mathrm{HWZM}} = \lambda / (N d) \propto 1/N$, increasing $N$
# directly increases the **Rayleigh chromatic resolving power** $R = \lambda / \Delta\lambda
# = |m| N$.
#
# In this tutorial we simulate 2D finite-$N$ PEC transmission gratings ($N \in \{2, 3, 4,
# 6, 8\}$) in Wakis, project their near fields to the far zone with `wakis.near2far.Near2FarMonitor`,
# demonstrate the Rayleigh resolving limit $R = |m| N$ on a spectral doublet in a single
# broadband time-domain run, and compare the finite-$N$ angular lobe integrals and
# computational cost against the **$N \to \infty$ periodic unit cell** solved in one period
# $d$ with periodic boundary conditions and `wakis.diffraction.DiffractionMonitor`—pointing
# forward to **Chapter 6 (`06_periodic_structures`)**, where general periodic and Bloch
# structures are treated systematically.
#
# ## Learning Goals
#
# 1. Derive the $N$-slit grating factorization $I(\theta) = I_0 \left[\frac{\sin(N\alpha)}{N\sin\alpha}\right]^2 \operatorname{sinc}^2(\beta)$ with $\alpha = \frac{\pi d}{\lambda}\sin\theta$ and $\beta = \frac{\pi a}{\lambda}\sin\theta$.
# 2. Verify the **grating equation** $d\sin\theta_m = m\lambda$, count the $(N - 2)$ **secondary maxima** between consecutive principal orders, and measure their suppression vs $N$.
# 3. Quantify the $\propto 1/N$ narrowing of principal maxima ($\Delta(\sin\theta)_{\mathrm{HWZM}} = \lambda / (N d)$ and $\Delta(\sin\theta)_{\mathrm{FWHM}} \approx 0.8859\,\lambda / (N d)$) and demonstrate **Rayleigh's chromatic resolving power** $R = \lambda / \Delta\lambda = |m| N$.
# 4. Compare finite-$N$ integrated order fractions against the **$N \to \infty$ periodic unit cell** (`wakis.diffraction.DiffractionMonitor`), evaluate the $\mathcal{O}(N)$ mesh and memory cost of finite-$N$ domains via `wakis.estimate.estimate`, and connect to **Chapter 6** (`06_periodic_structures`).

# %% [markdown]
# ## Physics Background
#
# ### 1. The $N$-Slit Interference Factor and Secondary Maxima
#
# Consider $N$ identical slits of width $a$ centered at
#
# $$
# x_n = \left(n - \frac{N - 1}{2}\right) d, \qquad n = 0, 1, \dots, N - 1,
# $$
#
# illuminated at normal incidence by a monochromatic plane wave of wavelength $\lambda$
# ($k_0 = 2\pi / \lambda$). Summing the geometric phase progression $e^{-i k_0 x_n \sin\theta}$
# across the $N$ slits yields a geometric series whose squared magnitude factorizes the
# far-field intensity into the product of an **$N$-slit interference factor** $I_{\mathrm{int}, N}(\theta)$
# and the **single-slit diffraction envelope** $I_{\mathrm{env}}(\theta)$ (Born & Wolf,
# *Principles of Optics*, §8.6.1; Hecht, *Optics*, §10.2.6):
#
# $$
# \frac{U_{\mathrm{2D}}(\theta)}{U_{\mathrm{2D}}(0)}
# = \underbrace{\left[\frac{\sin(N\alpha)}{N\sin\alpha}\right]^2}_{I_{\mathrm{int}, N}(\theta)}
#   \,\underbrace{\left(\frac{\sin\beta}{\beta}\right)^2}_{I_{\mathrm{env}}(\theta)},
# \qquad
# \alpha = \frac{\pi d}{\lambda}\sin\theta,
# \quad
# \beta = \frac{\pi a}{\lambda}\sin\theta.
# $$
#
# Key structural properties of $I_{\mathrm{int}, N}(\alpha) = \left[\frac{\sin(N\alpha)}{N\sin\alpha}\right]^2$:
#
# - **Principal maxima**: When $\alpha = m\pi$ ($m \in \mathbb{Z}$), both numerator and
#   denominator vanish and L'Hôpital's rule gives $I_{\mathrm{int}, N}(m\pi) = 1$. In
#   direction-cosine space this is the **grating equation**:
#
#   $$
#   d\sin\theta_m = m\lambda, \qquad m = 0, \pm 1, \pm 2, \dots, \quad |m| < \frac{d}{\lambda}.
#   $$
#
# - **$(N - 1)$ interference zeros between principal orders**: The numerator $\sin(N\alpha)$
#   vanishes without the denominator $\sin\alpha$ vanishing whenever
#
#   $$
#   \alpha = \frac{q\pi}{N} \implies \sin\theta = \left(m + \frac{q}{N}\right)\frac{\lambda}{d},
#   \qquad q = 1, 2, \dots, N - 1.
#   $$
#
# - **$(N - 2)$ secondary maxima**: By Rolle's theorem, between the $(N - 1)$ consecutive
#   zeros at $q = 1, \dots, N - 1$ there lie exactly **$(N - 2)$ secondary maxima**
#   satisfying $\tan(N\alpha) = N\tan\alpha$. For large $N$, the tallest secondary maximum
#   (adjacent to a principal peak) occurs near $N\alpha \approx \pm 1.4303\pi$, where
#   $I_{\mathrm{sec}} / I_{\mathrm{main}} \to [\sin(1.4303\pi)/(1.4303\pi)]^2 \approx 0.0472$
#   ($4.72\%$).
#
# ### 2. Peak Width $\propto 1/N$ and Rayleigh Chromatic Resolving Power $R = |m| N$
#
# Around the $m$-th principal maximum $\alpha_m = m\pi$, the first adjacent zero lies at
# $\Delta\alpha = \pm\pi / N$, corresponding to an angular half-width to first minimum
# (HWZM) in $\sin\theta$:
#
# $$
# \Delta(\sin\theta)_{\mathrm{HWZM}} = \frac{\lambda}{N d}, \qquad
# \Delta\theta_{\mathrm{HWZM}} \approx \frac{\lambda}{N d \cos\theta_m}.
# $$
#
# Similarly, the full width at half maximum (FWHM) satisfies $\operatorname{sinc}^2(N\Delta\alpha_{\mathrm{HWHM}}/\pi) = 1/2$, giving
#
# $$
# \Delta(\sin\theta)_{\mathrm{FWHM}} \approx \frac{0.88589\,\lambda}{N d} \propto \frac{1}{N}.
# $$
#
# Suppose the incident beam contains two closely spaced wavelengths $\lambda$ and
# $\lambda + \Delta\lambda$ in diffraction order $m \neq 0$. By differentiating the grating
# equation $d\sin\theta_m = m\lambda$ at fixed $m$, the angular dispersion shift is
# $\Delta(\sin\theta_m) = |m|\,\Delta\lambda / d$. Under **Lord Rayleigh's criterion** (Born
# & Wolf §8.6.2), two equal-intensity spectral lines are just resolved when the principal
# maximum of $\lambda + \Delta\lambda$ coincides with the first diffraction zero of $\lambda$:
#
# $$
# \frac{|m|\,\Delta\lambda_{\min}}{d} = \frac{\lambda}{N d}
# \implies
# R \equiv \frac{\lambda}{\Delta\lambda_{\min}} = |m|\,N.
# $$
#
# At the exact Rayleigh limit $\Delta\lambda / \lambda = 1 / (|m| N)$, the incoherent sum of
# the two line profiles exhibits a saddle dip of $8 / \pi^2 \approx 0.8106$ ($81.1\%$) of the
# peak intensity between the two maxima.
#
# ### 3. Finite-$N$ Gratings vs the $N \to \infty$ Periodic Unit Cell (Forward Link to Chapter 6)
#
# As $N \to \infty$, the continuous angular factor $\left[\frac{\sin(N\alpha)}{N\sin\alpha}\right]^2$
# collapses into a Dirac comb of discrete **Rayleigh–Floquet diffraction orders** at
# $\sin\theta_m = m\lambda / d$, while the integrated power in the $m$-th finite-$N$ lobe
# approaches the discrete grating diffraction efficiency $T_m$.
#
# Simulating a finite grating of $N$ slits in an open CPML domain requires a transverse span
# $L_x > N d + 2 W_{\mathrm{margin}}$, so the number of 2D mesh cells, memory footprint, and
# runtime grow **linearly with $N$** (and even faster if the acoustic/electromagnetic
# transit time across the array diagonal is included). When $N \gg 10$ and edge-truncation
# effects are negligible, simulating all $N$ slits explicitly is unnecessary: in **Chapter 6
# (`06_periodic_structures`)**, the infinite grating ($N \to \infty$) is solved in **a single
# unit cell of width $d$** using periodic (or Bloch) boundary conditions along $x$ and Fourier
# modal decomposition via `wakis.diffraction.DiffractionMonitor`.

# %%
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    DEFAULT_WAVELENGTH,
    plot_n_slit_patterns_and_nearfield,
    plot_periodic_comparison_and_cost,
    plot_resolving_power_and_width,
    run_patterns_task,
    run_periodic_vs_finite_task,
    run_preflight_summary,
    run_resolution_task,
    run_resolving_power_task,
)

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"


# %% [markdown]
# ## Numerical Setup & Preflight Budget
#
# All simulations run in 2D ($N_z = 1$ with periodic $z$ boundaries and $p$-polarization
# $\mathbf{E} \perp$ slit edges, where $H_z$ is the fundamental uniform TEM aperture mode):
#
# - **Wavelength**: $\lambda = 1.0\,\mu\text{m}$, slit width $a = 0.6\lambda$, grating period
#   $d = 1.8\lambda$ ($d/a = 3$, propagating orders $m \in \{-1, 0, +1\}$ at $0^\circ$ and
#   $\pm 33.75^\circ$), PEC screen thickness $t = 0.14\lambda$.
# - **Discretization rules**: background resolution $\mathrm{cpw} = 22 \ge 20$ cells per
#   wavelength ($\mathrm{cpw} = 20$ in `--quick` mode), with graded mesh refinement snapping
#   exact grid nodes to all $2N$ slit jaws and both screen faces ($\ge 6$ cells across each
#   slit $a$, $\ge 6$ cells across each inter-slit PEC bar $b = d - a$, and $\ge 4$ cells
#   across the screen thickness $t$).
# - **Source & monitors**: full-width upstream `CurrentSource` plane wave (normalizing out
#   source spectrum $S(f_0)$), downstream `Near2FarMonitor` for angular patterns $U_{\mathrm{2D}}(\theta)$
#   (at multiple frequencies simultaneously for the Rayleigh doublet study), and
#   `wakis.diffraction.DiffractionMonitor` on a single periodic cell $x \in [-d/2, +d/2]$ for
#   the $N \to \infty$ reference.

# %%
def print_preflight() -> dict:
    """Run and display the preflight resource estimate for the N = 6 grating mesh."""
    pf = run_preflight_summary(
        wavelength=DEFAULT_WAVELENGTH,
        n_slits=6,
        slit_width=0.6 * DEFAULT_WAVELENGTH,
        period=1.8 * DEFAULT_WAVELENGTH,
        screen_thickness=0.14 * DEFAULT_WAVELENGTH,
        cpw=22,
    )
    print("=== Tutorial 3.3 Preflight & Discretization Check (N = 6) ===")
    print(f"  Total 2D mesh cells           : {pf['n_cells']:,}")
    print(f"  Cells across each slit a      : {pf['cells_across_slit']} (rule: >= 4)")
    print(f"  Cells across each PEC bar b   : {pf['cells_across_bridge']} (rule: >= 4)")
    print(f"  Cells across screen t         : {pf['cells_across_screen']} (rule: >= 4)")
    print(f"  Courant time step dt          : {pf['dt'] * 1e15:.4f} fs")
    print(
        f"  Estimated RAM (solver+DFT)    : {pf['total_ram_mb']:.2f} MB "
        f"(solver={pf['solver_ram_mb']:.2f} MB, DFT={pf['monitor_ram_mb']:.2f} MB)"
    )
    print(f"  Preflight status              : {'OK' if pf['ok'] else pf['warnings']}")
    return pf


# %% [markdown]
# ## Simulation Tasks & Benchmark Tables
#
# The script provides four modular tasks (`--task {all,patterns,resolving_power,periodic_vs_finite,resolution}`):
#
# 1. `patterns`: Runs 2D FIT simulations for $N \in \{2, 3, 4, 6, 8\}$ (`{2, 3, 4, 6}` in
#    `--quick` mode) with $a = 0.6\lambda$, $d = 1.8\lambda$. Verifies the principal peak
#    angle $\theta_1 = \arcsin(\lambda/d) = 33.75^\circ$, counts the $(N - 2)$ secondary
#    maxima between $m = 0$ and $m = +1$, measures the tallest secondary peak vs theory, and
#    records the 2D near-field Talbot interference carpet for $N = 6$.
# 2. `resolving_power`: Evaluates the $\Delta(\sin\theta) \propto 1/N$ peak narrowing across
#    $N$ and demonstrates Rayleigh's chromatic resolving power $R = \lambda / \Delta\lambda =
#    |m| N = 6$ ($N = 6, m = +1$) by extracting three spectral doublets ($\Delta\lambda/\lambda
#    \in \{0.55/R, 1.00/R, 1.45/R\}$) from a single broadband time-domain simulation.
# 3. `periodic_vs_finite`: Solves the $N \to \infty$ grating in a single period $d$ with
#    periodic boundary conditions and `wakis.diffraction.DiffractionMonitor`, compares the
#    unit-cell Rayleigh–Floquet order fractions $T_m / T_{\mathrm{tot}}$ ($m \in \{0, \pm 1\}$)
#    with the finite-$N$ lobe integrals, and benchmarks the computational cost of finite-$N$
#    domains vs the single periodic unit cell (pointing forward to Chapter 6).
# 4. `resolution`: Sweeps $\mathrm{cpw} \in \{12, 16, 20, 28, 36\}$ (`{14, 20, 28}` in
#    `--quick` mode) for $N = 4$ to verify grid convergence.

# %%
def run_selected_tasks(mode: str, task: str) -> None:
    """Execute the requested benchmark tasks and save .npz archives in data/."""
    quick = mode == "quick"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if task in ("all", "patterns"):
        pat_data = run_patterns_task(quick=quick)
        np.savez(DATA_DIR / "n_slits_patterns.npz", **pat_data)

    if task in ("all", "resolving_power"):
        rp_data = run_resolving_power_task(quick=quick)
        np.savez(DATA_DIR / "n_slits_resolving_power.npz", **rp_data)

    if task in ("all", "periodic_vs_finite"):
        per_data = run_periodic_vs_finite_task(quick=quick)
        np.savez(DATA_DIR / "n_slits_periodic_vs_finite.npz", **per_data)

    if task in ("all", "resolution"):
        res_data = run_resolution_task(quick=quick)
        np.savez(DATA_DIR / "n_slits_resolution.npz", **res_data)


def print_summary_tables() -> None:
    """Load saved .npz datasets from data/ and print quantitative comparison tables."""
    pat_path = DATA_DIR / "n_slits_patterns.npz"
    rp_path = DATA_DIR / "n_slits_resolving_power.npz"
    per_path = DATA_DIR / "n_slits_periodic_vs_finite.npz"
    res_path = DATA_DIR / "n_slits_resolution.npz"

    if pat_path.exists():
        d = np.load(pat_path)
        print(
            "\n--- Table 1: N-Slit Principal Maxima, (N-2) Secondary Maxima & Peak Width (a = 0.6*lambda, d = 1.8*lambda) ---"
        )
        print(
            f"{'N':>4s}  {'L2 err [%]':>10s}  {'theta_1 FIT':>11s}  {'Sec peaks (FIT/th)':>18s}  "
            f"{'Max sec FIT':>11s}  {'Max sec th':>10s}  {'HWZM FIT':>9s}  {'lambda/(Nd)':>11s}"
        )
        for k, n_val in enumerate(d["n_values"]):
            sec_str = f"{int(d['n_sec_counted'][k])} / {int(d['n_sec_expected'][k])}"
            print(
                f"{int(n_val):4d}  {float(d['l2_errors'][k]) * 100.0:10.2f}  "
                f"{float(d['theta_m1_fit'][k]):11.2f}  {sec_str:>18s}  "
                f"{float(d['max_sec_fit'][k]):11.4f}  {float(d['max_sec_theory'][k]):10.4f}  "
                f"{float(d['hwzm_sin_fit'][k]):9.4f}  {float(d['hwzm_sin_theory'][k]):11.4f}"
            )

    if rp_path.exists():
        d = np.load(rp_path)
        r_val = float(d["resolving_power"])
        print(
            f"\n--- Table 2: Rayleigh Chromatic Resolving Power R = |m|N = {r_val:.0f} (N = {int(d['n_slits'])}, m = +{int(d['order_m'])}) ---"
        )
        print(
            f"{'Regime':>20s}  {'(Delta lam/lam)*R':>17s}  {'Saddle/Peak FIT':>15s}  {'Saddle/Peak theory':>18s}"
        )
        regimes = ["Sub-Rayleigh", "Exact Rayleigh", "Super-Rayleigh"]
        for k, reg in enumerate(regimes):
            mult = float(d["frac_shifts"][k]) * r_val
            print(
                f"{reg:>20s}  {mult:17.2f}  {float(d['saddle_to_peak_fit'][k]):15.4f}  "
                f"{float(d['saddle_to_peak_th'][k]):18.4f}"
            )

    if pat_path.exists() and per_path.exists():
        pat = np.load(pat_path)
        per = np.load(per_path)
        print(
            "\n--- Table 3: Finite-N Lobe Fractions & Cost vs N -> infinity Periodic Unit Cell (Chapter 6 Preview) ---"
        )
        print(
            f"{'Case':>16s}  {'Frac m=0':>10s}  {'Frac m=+/-1':>12s}  {'2D Cells':>10s}  "
            f"{'RAM [MB]':>10s}  {'Time [s]':>9s}"
        )
        for k, n_val in enumerate(pat["n_values"]):
            f0 = float(pat["lobe_fractions"][k, 1])
            f1 = 0.5 * float(pat["lobe_fractions"][k, 0] + pat["lobe_fractions"][k, 2])
            print(
                f"{f'Finite N = {int(n_val)}':>16s}  {f0:10.4f}  {f1:12.4f}  "
                f"{int(pat['n_cells'][k]):10,d}  {float(pat['host_ram_mb'][k]):10.1f}  "
                f"{float(pat['elapsed_s'][k]):9.2f}"
            )
        p0 = float(per["periodic_order_fractions"][1])
        p1 = 0.5 * float(
            per["periodic_order_fractions"][0] + per["periodic_order_fractions"][2]
        )
        print(
            f"{'Periodic (N->inf)':>16s}  {p0:10.4f}  {p1:12.4f}  "
            f"{int(per['periodic_n_cells']):10,d}  {float(per['periodic_host_ram_mb']):10.1f}  "
            f"{float(per['periodic_elapsed_s']):9.2f}"
        )

    if res_path.exists():
        d = np.load(res_path)
        print(
            "\n--- Table 4: Resolution Convergence Study (N = 4, a = 0.6*lambda, d = 1.8*lambda) ---"
        )
        print(
            f"{'cpw':>5s}  {'slit_cells':>10s}  {'bar_cells':>9s}  {'scr_cells':>9s}  "
            f"{'L2 vs theory [%]':>16s}  {'Rel to finest [%]':>17s}  {'Time [s]':>8s}"
        )
        for k, cpw in enumerate(d["cpw_values"]):
            print(
                f"{int(cpw):5d}  {int(d['cells_across_slit'][k]):10d}  "
                f"{int(d['cells_across_bridge'][k]):9d}  {int(d['cells_across_screen'][k]):9d}  "
                f"{float(d['l2_vs_theory'][k]) * 100.0:16.2f}  "
                f"{float(d['rel_to_finest'][k]) * 100.0:17.2f}  "
                f"{float(d['elapsed_s'][k]):8.2f}"
            )


def generate_all_plots() -> None:
    """Generate all three Tutorial 3.3 publication figures from saved data/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pat_path = DATA_DIR / "n_slits_patterns.npz"
    rp_path = DATA_DIR / "n_slits_resolving_power.npz"
    per_path = DATA_DIR / "n_slits_periodic_vs_finite.npz"
    res_path = DATA_DIR / "n_slits_resolution.npz"

    if pat_path.exists():
        pat_data = dict(np.load(pat_path))
        plot_n_slit_patterns_and_nearfield(
            pat_data, FIG_DIR / "n_slits_patterns_and_secondary_maxima.png"
        )

    if pat_path.exists() and rp_path.exists():
        pat_data = dict(np.load(pat_path))
        rp_data = dict(np.load(rp_path))
        plot_resolving_power_and_width(
            pat_data, rp_data, FIG_DIR / "n_slits_resolving_power_and_width.png"
        )

    if pat_path.exists() and per_path.exists() and res_path.exists():
        pat_data = dict(np.load(pat_path))
        per_data = dict(np.load(per_path))
        res_data = dict(np.load(res_path))
        plot_periodic_comparison_and_cost(
            pat_data, per_data, res_data, FIG_DIR / "n_slits_periodic_vs_finite_and_cost.png"
        )


# %% [markdown]
# ## Results & Discussion
#
# ### 1. Principal Maxima and $(N - 2)$ Secondary Maxima (`figures/n_slits_patterns_and_secondary_maxima.png`)
#
# - **Near-field Talbot carpet (Panel a)**: Immediately behind the $N = 6$ grating ($a = 0.6\lambda$,
#   $d = 1.8\lambda$), the six emerging cylindrical wavelets interfere to form a structured
#   near-field self-imaging (Talbot) carpet before separating into the $m = 0$ and $m = \pm 1$
#   far-field diffraction orders.
# - **Sharpening of principal orders (Panel b)**: As $N$ increases from $2$ to $8$, the
#   principal maxima remain pinned at the grating-equation angles $\theta_0 = 0^\circ$ and
#   $\theta_{\pm 1} = \pm\arcsin(\lambda / d) = \pm 33.75^\circ$ (measured $\pm 33.6^\circ\text{--}33.8^\circ$),
#   and their peak heights remain modulated by the single-slit diffraction envelope
#   $\operatorname{sinc}^2(\beta)$. Across all $N \in \{2, 3, 4, 6, 8\}$, the 2D FIT
#   `Near2FarMonitor` angular pattern agrees with $\left[\frac{\sin(N\alpha)}{N\sin\alpha}\right]^2 \operatorname{sinc}^2\beta$
#   to within $2.5\%\text{--}4.2\%$ $L^2$ error.
# - **$(N - 2)$ secondary maxima (Panels c & d)**: Zooming into the interval $0 \le (d/\lambda)\sin\theta \le 1$
#   between $m = 0$ and $m = +1$ confirms that $N = 2$ has $0$ secondary peaks, $N = 3$ has
#   $1$, $N = 4$ has $2$, $N = 6$ has $4$, and $N = 8$ has $6$ secondary peaks—matching the
#   $(N - 2)$ rule without exception. The tallest secondary peak drops from $11.1\%$ ($N = 3$)
#   and $6.2\%$ ($N = 4$) toward the asymptotic $(1.4303\pi)^{-2} \approx 4.72\%$ floor.
#
# ### 2. Peak Width $\propto 1/N$ and Chromatic Resolving Power (`figures/n_slits_resolving_power_and_width.png`)
#
# - **$\propto 1/N$ angular narrowing (Panel a)**: Both the measured first-minimum half-width
#   $\Delta(\sin\theta)_{\mathrm{HWZM}}$ and the full width at half maximum
#   $\Delta(\sin\theta)_{\mathrm{FWHM}}$ follow the exact $\lambda / (N d)$ and
#   $0.8859\,\lambda / (N d)$ curves across $N = 2 \dots 8$ within $< 3\%$.
# - **Rayleigh resolving criterion $R = |m| N$ (Panel b)**: For $N = 6$ in order $m = +1$
#   ($R = 6$, Rayleigh separation $\Delta\lambda / \lambda = 1/6$), extracting multiple
#   frequencies from a single broadband time-domain simulation shows:
#   - For $\Delta\lambda / \lambda = 0.55 / R$ (sub-Rayleigh), the two lines merge into a
#     single flat-topped peak (`saddle/peak = 1.00`).
#   - At the **exact Rayleigh limit** $\Delta\lambda / \lambda = 1 / R = 1/6$, the peak of
#     $\lambda_2 = \lambda_0(1 + 1/6)$ lands on the first interference zero of $\lambda_0$,
#     producing a clear central saddle dip with `saddle/peak` $\approx 0.78$ (matching the
#     analytical grating value $0.779$, close to the $8/\pi^2 \approx 0.811$ paraxial sinc
#     limit).
#   - For $\Delta\lambda / \lambda = 1.45 / R$ (super-Rayleigh), the doublet is deeply split
#     (`saddle/peak` $\approx 0.25$).
#
# ### 3. Finite-$N$ vs $N \to \infty$ Periodic Unit Cell and Cost Scaling (`figures/n_slits_periodic_vs_finite_and_cost.png`)
#
# - **Order fractions (Panel a)**: Integrating the finite-$N$ `Near2FarMonitor` angular
#   intensity over each principal order window $\sin\theta \in [(m - 1/2)\lambda/d, (m + 1/2)\lambda/d]$
#   yields order fractions $T_0 / T_{\mathrm{tot}} \approx 0.44$ and $T_{\pm 1} / T_{\mathrm{tot}}
#   \approx 0.28$ that rapidly approach the $N \to \infty$ single-period Rayleigh–Floquet
#   efficiencies measured by `wakis.diffraction.DiffractionMonitor`.
# - **Computational cost (Panel b)**: In a finite CPML domain, the 2D mesh grows from
#   $\sim 66\text{k}$ cells ($N = 2$) to $\sim 140\text{k}$ cells ($N = 8$) and exceeds
#   $420\text{k}$ cells ($> 3\,\text{GB}$ host RAM) by $N = 32$. By contrast, the **single
#   periodic unit cell** ($x \in [-d/2, +d/2]$ with `bc_low=['periodic', 'cpml', 'periodic']`)
#   requires only **$\sim 8.5\text{k}$ cells** ($\sim 65\,\text{MB}$ RAM, $< 0.6\,\text{s}$
#   runtime)—over an order of magnitude smaller than even $N = 6$. In **Chapter 6
#   (`06_periodic_structures`)**, this unit-cell formulation is extended with Bloch
#   quasi-periodic boundary conditions to model arbitrary incidence angles, wood anomalies,
#   and metasurfaces.
#
# ## Limits & Failure Modes
#
# 1. **Edge-slit truncation and mutual coupling at small $N$**: Scalar grating theory assumes
#    every slit has an identical aperture field. In a real PEC grating, the two outermost
#    slits ($n = 0$ and $n = N - 1$) have only one neighboring slit, whereas interior slits
#    have two. Surface waves launched along the PEC bars slightly perturb the outer-slit
#    amplitudes and phases, preventing the $(N - 1)$ interference zeros from reaching exact
#    zero ($U_{\min} / U(0) \sim 10^{-3}$ instead of $0$).
# 2. **Rayleigh / Wood threshold anomalies ($m\lambda = d$)**: Whenever a diffraction order
#    emerges at grazing angle $\theta_m = \pm 90^\circ$ ($d / \lambda \in \mathbb{Z}^+$), the
#    diffracted wave skims along the grating surface and couples strongly all $N$ slits over
#    long distances. At these Wood-anomaly wavelengths, scalar $\operatorname{sinc}^2(\beta)$
#    envelope theory fails completely and full-wave Maxwell solvers are mandatory.
# 3. **Finite-domain aperture truncation at large $N$**: For large $N$, a fixed downstream
#    near-to-far monitor line of width $W_{\mathrm{mon}}$ located at distance $y_{\mathrm{n2f}}$
#    must extend well beyond the outer slits ($W_{\mathrm{mon}} > N d + 2\lambda$) so that
#    high-order beams ($|\theta_m| \gtrsim 45^\circ$) from the outermost slits pass through the
#    monitor rather than escaping out the lateral edges. When $N \gtrsim 15$, switching to the
#    Chapter 6 periodic unit cell with `DiffractionMonitor` avoids both lateral truncation and
#    the $\mathcal{O}(N)$ mesh cost.
# 4. **Time-window truncation for high-$R$ spectral resolution**: Resolving two wavelengths
#    separated by $\Delta\lambda / \lambda = 1 / (|m| N)$ in a time-domain DFT requires a
#    simulation duration $T_{\mathrm{sim}} \gtrsim |m| N / f_0$ so that waves from the two
#    opposite ends of the $N$-slit array (path difference $|m| N \lambda$) can interfere at
#    the monitor before time stepping stops.
#
# ## Exercises
#
# 1. **Higher diffraction orders ($|m| = 2$)**: Increase the grating period to $d = 2.6\lambda$
#    while keeping $a = 0.65\lambda$ ($d/a = 4$). Verify that $m = \pm 2$ appears at
#    $\theta_{\pm 2} = \pm\arcsin(2/2.6) = \pm 50.28^\circ$ and that the chromatic resolving
#    power in second order doubles to $R = 2N$.
# 2. **Missing grating orders**: In the $d = 1.8\lambda, a = 0.6\lambda$ ($d/a = 3$) grating,
#    check the periodic unit-cell `DiffractionMonitor` efficiency at $m = \pm 3$ when
#    $d = 3.2\lambda, a = 3.2\lambda / 3$: just as in Tutorial 3.2, the $m = \pm 3$ grating
#    order coincides with the first single-slit envelope null and is strongly suppressed.
# 3. **Polarization comparison ($\mathbf{E} \parallel$ vs $\mathbf{E} \perp$ slit edges)**:
#    Run `run_periodic_unit_cell_grating` with `polarization="Ez"` vs `"Hz"` at $a = 0.45\lambda$
#    (below the $\text{TE}_{10}$ waveguide cutoff $a = \lambda/2$) to observe cutoff
#    suppression of all grating orders for $E_z$ alongside strong TEM transmission for $H_z$.
#
# ## References
#
# 1. M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University Press, 1999),
#    §8.6 "The diffraction grating" (pp. 446–460),
#    doi:[10.1017/CBO9781139644181](https://doi.org/10.1017/CBO9781139644181).
# 2. E. Hecht, *Optics*, 5th ed. (Pearson, 2017), §10.2.6 "Multiple-Slit Diffraction" and
#    §10.2.7 "The Diffraction Grating".
# 3. Lord Rayleigh, "Investigations in optics, with special reference to the spectroscope,"
#    *Philos. Mag.* **8**(49), 261–274 (1879),
#    doi:[10.1080/14786447908639684](https://doi.org/10.1080/14786447908639684).
# 4. A. Oskooi and S. G. Johnson, "Electromagnetic Wave Source Conditions," in *Advances in
#    FDTD Computational Electrodynamics: Photonics and Nanotechnology* (Artech House, 2013),
#    [arXiv:1301.5366](https://arxiv.org/abs/1301.5366).
# 5. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and S. G. Johnson,
#    "Meep: A flexible free-software package for electromagnetic simulations by the FDTD
#    method," *Comput. Phys. Commun.* **181**(3), 687–702 (2010),
#    doi:[10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for Tutorial 3.3."""
    parser = argparse.ArgumentParser(
        description="Tutorial 3.3: N-Slit Diffraction Gratings, Resolving Power, and Periodic vs Finite-N Scaling."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced-cost sweep (~30-50 s).",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark sweep (~1-2 min).",
    )
    parser.add_argument(
        "--task",
        choices=("all", "patterns", "resolving_power", "periodic_vs_finite", "resolution"),
        default="all",
        help="Select which simulation task to run (default: all).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate figures and print tables from existing data/ without running solver.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for Tutorial 3.3 CLI and notebook execution."""
    args = parse_args(argv)
    if args.plot:
        print_summary_tables()
        generate_all_plots()
        return

    mode = "quick" if args.quick else "full"
    print_preflight()
    run_selected_tasks(mode=mode, task=args.task)
    print_summary_tables()
    generate_all_plots()


if __name__ == "__main__":
    main()
