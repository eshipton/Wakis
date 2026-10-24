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
# # Tutorial 3.1 · Diffraction by a Single Slit — Fresnel Near Field, Fraunhofer Far Field, and Vectorial Polarization Breakdown
#
# ## Abstract
# This tutorial opens **Chapter 3 (Diffraction and Scattering)** by modeling 2D electromagnetic
# diffraction of a monochromatic plane wave by a single slit of width $a$ in a perfectly
# conducting (PEC) screen of thickness $t$. We introduce three foundational workflows used
# throughout the chapter: (i) plane-wave excitation using a full-width soft current sheet paired
# with an empty-domain normalization run (and why truncating the source plane at the inner CPML
# boundary causes severe edge-diffraction artifacts), (ii) representing thin metallic screens
# either as finite-thickness PEC blocks (`Box(..., "pec")` with $\ge 4$ cells across $t$) or as
# zero-thickness conformal PEC sheets (`wakis.conformal.rectangle_sheet` passed to
# `GridFIT3D(..., sheets=...)`), and (iii) extracting transmitted power via `FluxMonitor` and
# far-field angular patterns via `wakis.near2far.Near2FarMonitor`. We benchmark the Finite
# Integration Technique (FIT) solver across the Fresnel near-field regime ($N_F = a^2/(4\lambda L)
# \in [0.25, 2.0]$), compare far-field angular patterns against Fraunhofer, Kirchhoff, and
# Rayleigh–Sommerfeld I/II theories, and map where scalar diffraction breaks down for narrow slits
# ($a \sim \lambda$) across the two 2D polarizations ($\mathbf{E} \parallel$ vs $\mathbf{E} \perp$
# to the slit edges) and screen thicknesses.
#
# ## Learning Goals
# 1. Inject a clean 2D plane wave in Wakis using a soft `CurrentSource` sheet that extends
#    all the way through the lateral CPMLs, and normalize scattered fields and power fluxes
#    with a matched empty-domain reference run.
# 2. Construct slotted PEC screens using both finite-thickness `Box(..., "pec")` geometries
#    ($\ge 20$ cells per wavelength, $\ge 4$ cells across the slit $a$ and across the screen
#    thickness $t$) and zero-thickness conformal PEC sheets (`rectangle_sheet`).
# 3. Project near-field aperture DFT data to the 2D cylindrical far field using
#    `wakis.near2far.Near2FarMonitor` and verify the 2D Parseval power identity
#    $\int_{-\pi/2}^{\pi/2} U_{\mathrm{2D}}(\theta)\,\mathrm{d}\theta = P'_{\mathrm{n2f}}$ to
#    better than $0.2\%$.
# 4. Distinguish the Fresnel near-field regime ($N_F \gtrsim 1$, double-horned and Cornu-spiral
#    fringe structure) from the Fraunhofer far-field regime ($\operatorname{sinc}^2$ envelope and
#    obliquity factors) and explain why $\mathbf{E} \parallel$ slit edges experiences a waveguide
#    cutoff at $a = \lambda/2$ while $\mathbf{E} \perp$ slit edges supports cutoff-free TEM
#    Fabry–Pérot transmission ($\tau > 1$).

# %% [markdown]
# ## Physics Background
#
# ### 1. Plane-Wave Injection and the Empty-Domain Normalization Run
#
# Scattering and diffraction problems require illuminating an obstacle with a known incident
# plane wave $\mathbf{E}^{\mathrm{inc}}(\mathbf{r}, t)$ while absorbing outgoing scattered waves
# at the domain boundaries. Wakis provides two complementary approaches:
#
# - **Total-Field / Scattered-Field (TF/SF) box** (`wakis.tfsf.TFSF2D` / `TFSF3D`): injects the
#   incident wave on a closed Huygens contour enclosing an isolated scatterer (used in
#   Tutorial 3.4 for a Mie cylinder).
# - **Full-width soft source plane with normalization run**: for an infinite screen that divides
#   the domain into upstream ($y < -t/2$) and downstream ($y > +t/2$) half-spaces, a closed
#   TF/SF contour cannot wrap around an infinite wall. Instead, we place a transparent planar
#   current sheet (`CurrentSource`) at $y = y_{\mathrm{src}} < -t/2$ extending **all the way
#   through the lateral $x$-CPML regions** ($x \in [x_{\min}, x_{\max}]$). As analyzed by
#   Oskooi & Johnson (2013), terminating a plane-wave source at the inner PML interface acts as a
#   hard aperture of width $L_x$ that launches cylindrical edge waves across the test zone;
#   extending both the source sheet and the PEC screen into the lateral CPML eliminates edge
#   diffraction to the $10^{-4}$ level.
#
# Running the identical grid and time step **without** the screen (`with_screen=False`) yields the
# exact discrete incident field $E_z^{\mathrm{inc}}(x, y, \omega)$ (or $H_z^{\mathrm{inc}}$) and
# incident power per unit transverse width and unit $z$-length:
#
# $$
# I_{\mathrm{inc}}(\omega) = \frac{\Phi_{\mathrm{inc}}(\omega)}{W_{\mathrm{ref}}\,L_z},
# $$
#
# where $\Phi_{\mathrm{inc}}(\omega)$ is the spectral Poynting flux recorded by a `FluxMonitor` of
# width $W_{\mathrm{ref}}$ at $y = 0$.
#
# ### 2. Scalar Diffraction Hierarchy: Fraunhofer, Kirchhoff, and Rayleigh–Sommerfeld
#
# Consider a slit of width $a$ occupying $|x| \le a/2$ at $y = 0$, illuminated at normal incidence
# by a monochromatic wave of wavelength $\lambda$ ($k_0 = 2\pi/\lambda$). Under the
# **Kirchhoff–Smythe aperture approximation**, the field inside the slit opening $|x'| \le a/2$
# equals the unperturbed incident wave $\psi_{\mathrm{inc}}(x', 0) = \psi_0$, and vanishes on the
# dark face of the screen ($|x'| > a/2$).
#
# In two dimensions, the exact Green's function for the Helmholtz equation
# $(\nabla_{\perp}^2 + k_0^2)\psi = 0$ is the Hankel function $\frac{\mathrm{i}}{4} H_0^{(1)}(k_0 R)$
# with $R = \sqrt{(x - x')^2 + L^2}$. Applying Green's theorem with either Dirichlet ($G_I = 0$ on
# $y'=0$) or Neumann ($\partial_{y'} G_{II} = 0$ on $y'=0$) boundary conditions yields the two
# **2D Rayleigh–Sommerfeld (RS) diffraction integrals**:
#
# $$
# \psi_{\mathrm{RS\text{-}I}}(x, L)
# = \frac{k_0 L}{2\mathrm{i}} \int_{-a/2}^{a/2} \psi_0\,\frac{H_1^{(1)}(k_0 R)}{R}\,\mathrm{d}x',
# \qquad
# \psi_{\mathrm{RS\text{-}II}}(x, L)
# = \frac{k_0}{2} \int_{-a/2}^{a/2} \psi_0\,H_0^{(1)}(k_0 R)\,\mathrm{d}x'.
# $$
#
# Note how the two scalar boundary conditions map directly onto the two decoupled 2D
# electromagnetic polarizations for a knife-edge PEC screen ($t \to 0$):
# - **$\mathbf{E} \parallel$ slit edges ($\mathrm{TM}_z / s$-polarization, fields $E_z, H_x, H_y$)**:
#   the tangential electric field $E_z$ vanishes on the PEC screen ($E_z = 0$ at $y = 0$,
#   $|x| > a/2$), which is the Dirichlet boundary condition of **RS-I**.
# - **$\mathbf{E} \perp$ slit edges ($\mathrm{TE}_z / p$-polarization, fields $E_x, E_y, H_z$)**:
#   the tangential electric field $E_x = \frac{1}{\mathrm{i}\omega\varepsilon_0}\partial_y H_z$
#   vanishes on the PEC screen ($\partial_y H_z = 0$ at $y = 0$, $|x| > a/2$), which is the
#   Neumann boundary condition of **RS-II**.
#
# #### Fresnel Near-Field Regime ($N_F = a^2 / (4\lambda L)$)
# In the paraxial near-field limit ($L \gg a$, small angles), expanding
# $R \approx L + (x - x')^2/(2L)$ reduces both integrals to the **Fresnel Cornu-spiral profile**:
#
# $$
# \frac{I_{\mathrm{Fresnel}}(x, L)}{I_{\mathrm{inc}}}
# = \frac{1}{2}\Bigl\{\bigl[C(u_2) - C(u_1)\bigr]^2 + \bigl[S(u_2) - S(u_1)\bigr]^2\Bigr\},
# \qquad
# u_{1,2}(x) = \sqrt{\frac{2}{\lambda L}}\left(x \mp \frac{a}{2}\right),
# $$
#
# where $C(u) = \int_0^u \cos(\pi t^2/2)\,\mathrm{d}t$ and
# $S(u) = \int_0^u \sin(\pi t^2/2)\,\mathrm{d}t$ are the Fresnel integrals, and the dimensionless
# **Fresnel number** is
#
# $$
# N_F = \frac{(a/2)^2}{\lambda L} = \frac{a^2}{4\lambda L}.
# $$
#
# When $N_F \gtrsim 1$, interference between the two slit edges produces multi-peaked Fresnel
# fringes across the geometric shadow $|x| \le a/2$ (including an on-axis dip near $N_F \approx 1.5$
# and an on-axis focus $I(0)/I_{\mathrm{inc}} \approx 1.66$ near $N_F \approx 0.74$). For
# $N_F \ll 1$, the beam reorganizes into a single central lobe whose angular shape approaches the
# far-field limit.
#
# #### Far-Field Angular Patterns ($r \to \infty$)
# Using the large-argument asymptotic form $H_n^{(1)}(k_0 R) \sim \sqrt{\frac{2}{\pi k_0 R}}
# \mathrm{e}^{\mathrm{i}(k_0 R - n\pi/2 - \pi/4)}$ with $R \approx r - x'\sin\theta$ and
# $L/r = \cos\theta$, the normalized far-field angular intensities $U(\theta)/U(0)$ become
#
# $$
# \frac{U_{\mathrm{Fraunhofer}}(\theta)}{U(0)} = \operatorname{sinc}^2\beta,
# \qquad
# \frac{U_{\mathrm{Kirchhoff}}(\theta)}{U(0)} = \left(\frac{1 + \cos\theta}{2}\right)^2 \operatorname{sinc}^2\beta,
# $$
#
# $$
# \frac{U_{\mathrm{RS\text{-}I}}(\theta)}{U(0)} = \cos^2\theta\,\operatorname{sinc}^2\beta
# \;\;(\mathbf{E} \parallel \text{slit}),
# \qquad
# \frac{U_{\mathrm{RS\text{-}II}}(\theta)}{U(0)} = \operatorname{sinc}^2\beta
# \;\;(\mathbf{E} \perp \text{slit}),
# $$
#
# where $\beta = \frac{\pi a \sin\theta}{\lambda}$ and $\operatorname{sinc}\beta = \frac{\sin\beta}{\beta}$.
# The diffraction zeros occur at $\sin\theta_m = m\,\frac{\lambda}{a}$ ($m = \pm 1, \pm 2, \dots$)
# whenever $a > \lambda$. Notice the key physical distinction at large diffraction angles
# $|\theta| \gtrsim 30^\circ$: for $\mathbf{E} \parallel$ slit edges, boundary charges/currents force
# the tangential field $E_z$ to zero along the screen ($\theta \to \pm 90^\circ$), giving the
# $\cos^2\theta$ RS-I rolloff; for $\mathbf{E} \perp$ slit edges, $H_z$ can glide along the PEC
# surface right up to grazing angles ($\theta \to \pm 90^\circ$), matching RS-II!
#
# ### 3. 2D Near-to-Far-Field Projection (`wakis.near2far.Near2FarMonitor`)
#
# Rather than extending the FDTD/FIT grid hundreds of wavelengths downstream to reach the
# Fraunhofer zone $L \gg a^2/\lambda$, we record the tangential DFT fields $(\mathbf{E}, \mathbf{H})$
# on a single planar monitor (`Near2FarMonitor`) placed just downstream of the screen at
# $y = y_{\mathrm{n2f}} = t/2 + \delta y$. By the Surface Equivalence Principle (Schelkunoff
# equivalence), the equivalent electric and magnetic surface currents
# $\mathbf{J}_s = \hat{\mathbf{n}} \times \mathbf{H}$ and
# $\mathbf{M}_s = -\hat{\mathbf{n}} \times \mathbf{E}$ (with $\hat{\mathbf{n}} = +\hat{\mathbf{y}}$)
# radiate the exact far-field pattern.
#
# Because Wakis's ` SolverFIT3D` operates on a 3D mesh extruded by one periodic cell of thickness
# $L_z$ along $z$, `Near2FarMonitor.radiation_intensity(theta_sph, phi_sph)` evaluates the 3D
# spherical radiation intensity $U_{\mathrm{3D}}$ $[\mathrm{W\,s^2/sr}]$. Applying stationary-phase
# integration along the periodic $z$-axis converts $U_{\mathrm{3D}}$ in the $xy$-plane
# ($\theta_{\mathrm{sph}} = \pi/2$, $\varphi_{\mathrm{sph}} = \pi/2 - \theta$) to the exact 2D
# cylindrical radiation intensity $U_{\mathrm{2D}}(\theta)$ $[\mathrm{W\,s^2/(m\cdot rad)}]$:
#
# $$
# U_{\mathrm{2D}}(\theta)
# = \frac{2\pi}{k_0 L_z^2}\,U_{\mathrm{3D}}\!\left(\theta_{\mathrm{sph}}=\frac{\pi}{2},\;
# \varphi_{\mathrm{sph}}=\frac{\pi}{2} - \theta\right),
# \qquad
# \int_{-\pi/2}^{\pi/2} U_{\mathrm{2D}}(\theta)\,\mathrm{d}\theta
# = \frac{\Phi_{\mathrm{n2f}}}{L_z} = P'_{\mathrm{n2f}}.
# $$
#
# ### 4. Vectorial Breakdown for Narrow Slits ($a \sim \lambda$) and Finite Screen Thickness $t$
#
# Scalar diffraction theory predicts that the total power transmitted through a slit of width $a$
# is simply the geometric-optics intercept $P'_{\mathrm{geom}} = a\,I_{\mathrm{inc}}$, so the
# normalized transmission coefficient $\tau \equiv P'_{\mathrm{trans}} / (a\,I_{\mathrm{inc}}) = 1$
# for all $a/\lambda$ and all polarizations. Full-wave Maxwell theory reveals a dramatic divergence
# when $a \lesssim \lambda$ (Sommerfeld 1896; Morse & Rubenstein 1938; Bouwkamp 1954; Takakura 2001):
#
# 1. **$\mathbf{E} \parallel$ slit edges ($\mathrm{TM}_z / E_z$)**: Inside a PEC slit of thickness
#    $t > 0$, the boundary condition $E_z(x = \pm a/2) = 0$ quantizes the transverse wavenumber to
#    $k_{x,n} = n\pi/a$ ($n = 1, 2, \dots$). The fundamental $\mathrm{TE}_{10}$ slab-waveguide mode
#    has longitudinal propagation constant
#    $$
#    k_{y,1} = \sqrt{k_0^2 - (\pi/a)^2} = k_0\sqrt{1 - \left(\frac{\lambda}{2a}\right)^2}.
#    $$
#    For **$a < \lambda/2$**, $k_{y,1} = \mathrm{i}\gamma_1$ is purely imaginary: the slit is
#    **below waveguide cutoff**, and transmitted power decays exponentially with screen thickness as
#    $\tau_{\parallel} \propto \mathrm{e}^{-2\gamma_1 t}$! Even for a zero-thickness knife-edge
#    screen ($t = 0$), $\tau_{\parallel} \propto (k_0 a)^2 \to 0$ as $a/\lambda \to 0$.
# 2. **$\mathbf{E} \perp$ slit edges ($\mathrm{TE}_z / H_z$)**: The PEC walls enforce
#    $\partial_x H_z(x = \pm a/2) = 0$, which admits the $n = 0$ **parallel-plate TEM mode** with
#    $k_{x,0} = 0$ and $k_{y,0} = k_0$ for **any** slit width $a > 0$ (no cutoff!). Impedance
#    mismatch at the entrance and exit faces $y = \mp t/2$ forms a Fabry–Pérot cavity that funnels
#    incident energy into the subwavelength slit, producing resonant transmission peaks with
#    $\tau_{\perp} > 1$ (effective collection width larger than the physical opening $a$).

# %%
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    DEFAULT_WAVELENGTH,
    plot_farfield_comparison,
    plot_source_and_fresnel,
    plot_transmission_and_resolution,
    run_farfield_task,
    run_fresnel_task,
    run_preflight_summary,
    run_resolution_task,
    run_source_benchmark_task,
    run_transmission_task,
)


# %% [markdown]
# ## Numerical Setup & Preflight Check
#
# Before launching parameter sweeps, we run `wakis.estimate.estimate` and `preflight` on the
# baseline 2D graded mesh ($\lambda = 600\,\mathrm{nm}$, $a = 2\lambda$, $t = 0.15\lambda$,
# background resolution $\mathrm{cpw} = 24$ cells/$\lambda$, refined to $\ge 8$ cells across the
# slit opening and $\ge 4$ cells across the PEC screen thickness).

# %%
def print_preflight_report(pf: dict) -> None:
    """Print a compact summary of the preflight resource & discretization check."""
    print("=== Tutorial 3.1 Preflight & Discretization Check ===")
    print(f"  Total 2D mesh cells          : {pf['n_cells']:,}")
    print(f"  Cells across slit opening a  : {pf['cells_across_slit']} (rule: >= 4)")
    print(f"  Cells across screen t        : {pf['cells_across_screen']} (rule: >= 4)")
    print(f"  Courant time step dt         : {pf['dt'] * 1e15:.4f} fs")
    print(
        f"  Estimated RAM (solver+DFT)   : {pf['total_ram_mb']:.2f} MB "
        f"(solver={pf['solver_ram_mb']:.2f} MB, DFT={pf['monitor_ram_mb']:.2f} MB)"
    )
    print(f"  Preflight status             : {'OK' if pf['ok'] else 'WARNINGS'}")
    for w in pf["warnings"]:
        print(f"    - {w}")


# %% [markdown]
# ## Tasks 1–4 & Resolution Study Execution
#
# The script supports `--quick` (fast validation for CI and interactive exploration), `--full`
# (complete benchmark tables and publication figures committed to `data/` and `figures/`),
# `--task {all,source,fresnel,farfield,transmission,resolution}`, and `--plot` (regenerate figures
# from saved `.npz` files without re-running the solver).

# %%
def _save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def print_summary_tables(
    src_data: dict,
    fresnel_data: dict,
    ff_data: dict,
    trans_data: dict,
    res_data: dict,
) -> None:
    """Print quantitative validation tables comparing FIT results against diffraction theory."""
    print("\n--- Table 1: Plane-Wave Source & 2D Near2Far Parseval Identity ---")
    print(
        f"  Full-width source transverse ripple (|x| <= 2.5*lambda) : "
        f"{float(src_data['ripple_full']) * 100:.3f} %"
    )
    print(
        f"  PML-truncated source transverse ripple (failure mode)   : "
        f"{float(src_data['ripple_trunc']) * 100:.2f} %"
    )
    print(
        f"  Empty-domain FluxMonitor vs |E_inc|^2/Z_0 discrepancy   : "
        f"{float(src_data['flux_field_rel_diff']) * 100:.3f} %"
    )
    print(
        f"  2D Parseval ratio (integral U_2D dtheta / P'_n2f)       : "
        f"{float(src_data['parseval_ratio_empty']):.4f}"
    )

    print("\n--- Table 2: Fresnel Near-Field Line Profiles (a = 2.8*lambda, E || slit) ---")
    print("  N_F     L / lambda   L2 err vs RS-I [%]   L2 err vs Cornu [%]   On-axis I(0)/I_inc (FIT / RS-I / Cornu)")
    x_oa = fresnel_data["x_over_a"]
    i0 = int(np.argmin(np.abs(x_oa)))
    for k, nf in enumerate(fresnel_data["nf_values"]):
        dl = float(fresnel_data["distances_over_lam"][k])
        e_rs = float(fresnel_data["l2_errors_rs1"][k]) * 100.0
        e_co = float(fresnel_data["l2_errors_cornu"][k]) * 100.0
        i_fit0 = float(fresnel_data["fit_profiles"][k, i0])
        i_rs0 = float(fresnel_data["rs1_profiles"][k, i0])
        i_co0 = float(fresnel_data["cornu_profiles"][k, i0])
        print(
            f"  {nf:4.2f}    {dl:8.2f}       {e_rs:12.2f}         {e_co:13.2f}         "
            f"{i_fit0:5.3f} / {i_rs0:5.3f} / {i_co0:5.3f}"
        )

    print("\n--- Table 3: Far-Field Angular Pattern Comparison (Near2FarMonitor) ---")
    print(
        "  a / lambda   L2(E_z vs Fraun) [%]   L2(E_z vs RS-I) [%]   "
        "L2(H_z vs RS-II) [%]   Parseval (E_z / H_z)"
    )
    for k, ratio in enumerate(ff_data["a_over_lam"]):
        ef = float(ff_data["l2_ez_vs_fraun"][k]) * 100.0
        er1 = float(ff_data["l2_ez_vs_rs1"][k]) * 100.0
        er2 = float(ff_data["l2_hz_vs_rs2"][k]) * 100.0
        p1 = float(ff_data["parseval_ez"][k])
        p2 = float(ff_data["parseval_hz"][k])
        print(
            f"  {ratio:8.2f}       {ef:14.2f}         {er1:13.2f}          "
            f"{er2:14.2f}         {p1:.4f} / {p2:.4f}"
        )

    print("\n--- Table 4: Normalized Transmitted Power tau = P'_trans / (a * I_inc) ---")
    print("  a / lambda   tau_Ez (t=0.2*lam)   tau_Hz (t=0.2*lam)   tau_Ez (t=0 sheet)   tau_Hz (t=0 sheet)")
    for k, ratio in enumerate(trans_data["a_over_lam"]):
        tez_t = float(trans_data["tau_ez_thick"][k])
        thz_t = float(trans_data["tau_hz_thick"][k])
        tez_s = float(trans_data["tau_ez_sheet"][k])
        thz_s = float(trans_data["tau_hz_sheet"][k])
        print(
            f"  {ratio:8.2f}       {tez_t:14.4f}       {thz_t:14.4f}       "
            f"{tez_s:14.4f}       {thz_s:14.4f}"
        )

    print("\n--- Table 5: Resolution Convergence Study (a = 2*lambda, t = 0.15*lambda, E || slit) ---")
    print("  cpw   slit_cells   scr_cells   L2 vs RS-I [%]   Rel to finest [%]   tau       First null [deg]   Time [s]")
    for k, cpw in enumerate(res_data["cpw_values"]):
        sc = int(res_data["cells_across_slit"][k])
        tc = int(res_data["cells_across_screen"][k])
        err = float(res_data["l2_vs_rs1"][k]) * 100.0
        rel = float(res_data["rel_to_finest"][k]) * 100.0
        tau = float(res_data["tau_values"][k])
        nd = float(res_data["null_deg_values"][k])
        el = float(res_data["elapsed_s"][k])
        print(
            f"  {cpw:3d}   {sc:10d}   {tc:9d}   {err:14.2f}   {rel:17.2f}   "
            f"{tau:7.4f}   {nd:16.2f}   {el:8.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tutorial 3.1: Single-slit diffraction in 2D (Fresnel, Fraunhofer, and polarization)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run fast validation preset (~30-45 s).")
    mode.add_argument("--full", action="store_true", help="Run full tutorial benchmark preset.")
    parser.add_argument(
        "--task",
        choices=("all", "source", "fresnel", "farfield", "transmission", "resolution"),
        default="all",
        help="Select a single benchmark task or 'all'.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate figures from existing data/*.npz files without running the solver.",
    )
    args = parser.parse_args(argv)
    quick = bool(args.quick or not args.full)

    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    f_src = data_dir / "single_slit_source.npz"
    f_fres = data_dir / "single_slit_fresnel.npz"
    f_ff = data_dir / "single_slit_farfield.npz"
    f_trans = data_dir / "single_slit_transmission.npz"
    f_res = data_dir / "single_slit_resolution.npz"

    if not args.plot:
        pf = run_preflight_summary(wavelength=DEFAULT_WAVELENGTH, cpw=20 if quick else 24)
        print_preflight_report(pf)

        if args.task in ("all", "source"):
            src_data = run_source_benchmark_task(quick=quick)
            _save_npz(f_src, src_data)
        if args.task in ("all", "fresnel"):
            fresnel_data = run_fresnel_task(quick=quick)
            _save_npz(f_fres, fresnel_data)
        if args.task in ("all", "farfield"):
            ff_data = run_farfield_task(quick=quick)
            _save_npz(f_ff, ff_data)
        if args.task in ("all", "transmission"):
            trans_data = run_transmission_task(quick=quick)
            _save_npz(f_trans, trans_data)
        if args.task in ("all", "resolution"):
            res_data = run_resolution_task(quick=quick)
            _save_npz(f_res, res_data)

    src_data = _load_npz(f_src) if f_src.exists() else None
    fresnel_data = _load_npz(f_fres) if f_fres.exists() else None
    ff_data = _load_npz(f_ff) if f_ff.exists() else None
    trans_data = _load_npz(f_trans) if f_trans.exists() else None
    res_data = _load_npz(f_res) if f_res.exists() else None

    if src_data is not None and fresnel_data is not None:
        plot_source_and_fresnel(
            src_data,
            fresnel_data,
            fig_dir / "single_slit_fresnel_and_source.png",
        )
    if ff_data is not None:
        plot_farfield_comparison(
            ff_data,
            fig_dir / "single_slit_farfield_patterns.png",
        )
    if trans_data is not None and res_data is not None:
        plot_transmission_and_resolution(
            trans_data,
            res_data,
            fig_dir / "single_slit_transmission_and_convergence.png",
        )

    if all(x is not None for x in (src_data, fresnel_data, ff_data, trans_data, res_data)):
        print_summary_tables(src_data, fresnel_data, ff_data, trans_data, res_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %% [markdown]
# ## Results & Validation Summary
#
# Running `single_slit.py --full` generates three benchmark figures and five validation datasets:
#
# 1. **Plane-wave injection & Fresnel near field** (`figures/single_slit_fresnel_and_source.png`):
#    - Extending the soft `CurrentSource` sheet into the lateral $x$-CPML yields a transverse
#      amplitude ripple of $< 0.05\%$ across $|x| \le 2.5\lambda$, whereas truncating the source
#      at the inner CPML edge introduces a $> 30\%$ edge-diffraction standing wave.
#    - Downstream of an $a = 2\lambda$ slit ($\mathbf{E} \parallel$ slit edges), the transverse
#      intensity profiles across $N_F \in \{2.0, 1.0, 0.5, 0.25\}$ ($L/\lambda \in \{0.5, 1.0,
#      2.0, 4.0\}$) track the 2D Rayleigh–Sommerfeld I integral and Cornu-spiral Fresnel profile,
#      reproducing the double-horned near-zone profile at $N_F = 2.0, 1.0$, the on-axis bright
#      focus ($I(0)/I_{\mathrm{inc}} > 1.25$) at $N_F = 0.5$, and the transition to a single main
#      lobe at $N_F = 0.25$.
# 2. **Far-field angular patterns vs Fraunhofer, Kirchhoff, and Rayleigh–Sommerfeld I/II**
#    (`figures/single_slit_farfield_patterns.png`):
#    - For wide slits ($a = 4\lambda, 2\lambda$), both polarizations collapse onto
#      $\operatorname{sinc}^2(\pi a\sin\theta/\lambda)$ across the main lobe and first side lobes,
#      with diffraction nulls at $\arcsin(m\lambda/a)$ ($\pm 14.48^\circ, \pm 30.0^\circ, \dots$).
#    - As the slit narrows ($a = 1.0\lambda, 0.6\lambda$), the two polarizations split at wide
#      angles $|\theta| \gtrsim 35^\circ$: $\mathbf{E} \parallel$ slit edges ($E_z$) follows the
#      Rayleigh–Sommerfeld I $\cos^2\theta\,\operatorname{sinc}^2\beta$ rolloff (vanishing along
#      the PEC screen at $\theta = \pm 90^\circ$), while $\mathbf{E} \perp$ slit edges ($H_z$)
#      maintains broad angular radiation into grazing angles as predicted by Rayleigh–Sommerfeld II.
# 3. **Total transmitted power $\tau(a/\lambda)$ & mesh convergence**
#    (`figures/single_slit_transmission_and_convergence.png`):
#    - Below the $\mathrm{TE}_{10}$ waveguide cutoff $a/\lambda = 0.5$, a finite-thickness screen
#      ($t = 0.20\lambda$) suppresses $\mathbf{E} \parallel$ transmission to $\tau_{\parallel} <
#      0.06$ at $a/\lambda = 0.30$ (compared with $\tau_{\parallel} \approx 0.60$ for a
#      zero-thickness knife-edge sheet $t = 0$).
#    - By contrast, $\mathbf{E} \perp$ slit edges ($H_z$) supports the cutoff-free TEM parallel-plate
#      mode, achieving resonant Fabry–Pérot funneling $\tau_{\perp} > 1.05$ at $a/\lambda \in [0.30,
#      0.45]$ before converging toward $\tau \to 1$ for $a \gg \lambda$.
#
# ## Limits & Failure Modes
#
# 1. **Truncating the plane-wave source or PEC screen inside the lateral CPMLs**:
#    If the `CurrentSource` sheet or the PEC screen blocks stop at the inner CPML boundary
#    ($x = \pm x_{\mathrm{core}}$) instead of extending across the full grid $x \in [x_{\min},
#    x_{\max}]$, the artificial ends radiate cylindrical waves and leak unblocked incident power
#    around the screen edges, corrupting both the aperture field (by $> 30\%$) and the far-field
#    side lobes.
# 2. **Open-plane `Near2FarMonitor` finite-aperture truncation at grazing angles**:
#    In 2D screen diffraction, the downstream `Near2FarMonitor` is an open planar segment of finite
#    width $W_{\mathrm{ref}}$ rather than a closed surface enclosing the entire infinite screen.
#    Angles beyond $|\theta_{\max}| \approx \arctan\bigl(W_{\mathrm{ref}} / (2 y_{\mathrm{n2f}})\bigr)
#    \approx 80^\circ$ suffer mild aperture-truncation ripple, especially for $\mathbf{E} \perp$
#    slit edges ($H_z$) where grazing surface waves propagate along the PEC face. Keep
#    $y_{\mathrm{n2f}} - t/2 \le 0.2\lambda$ so the monitor intercepts nearly the full forward
#    hemisphere.
# 3. **Under-resolving the slit opening or screen thickness**:
#    When $a < \lambda/2$ or $t \ll \lambda$, using a uniform grid at $\mathrm{cpw} = 20$ without
#    local grading can leave only $1$–$2$ Yee cells across the slit or screen thickness, causing
#    staircase capacitance errors and shifting the $\mathrm{TE}_{10}$ cutoff and TEM Fabry–Pérot
#    resonance. Always enforce $\ge 4$ cells across $a$ and $\ge 4$ cells across $t$ (or use
#    `wakis.conformal.rectangle_sheet` for the ideal $t = 0$ knife-edge limit).
# 4. **Scalar Kirchhoff/Fraunhofer breakdown for $a \lesssim \lambda$**:
#    Scalar diffraction ignores polarization and screen thickness altogether: for $a/\lambda < 0.5$
#    and $t = 0.2\lambda$, scalar theory ($\tau = 1$) overpredicts $\mathbf{E} \parallel$
#    transmission by more than an order of magnitude while missing the $\tau_{\perp} > 1$ TEM
#    resonance.
#
# ## Exercises
#
# 1. **Screen-thickness Fabry–Pérot sweep ($\mathbf{E} \perp$ slit edges)**:
#    Fix a subwavelength slit width $a = 0.25\lambda$ with `polarization="Hz"` and sweep the PEC
#    screen thickness $t/\lambda \in [0.1, 1.2]$. Verify that transmission peaks occur near the
#    Fabry–Pérot condition $k_0 t + \varphi_{\mathrm{refl}} = m\pi$ ($m = 1, 2$), consistent with
#    Takakura (2001).
# 2. **Oblique plane-wave incidence**:
#    Use a tilted plane-wave source or phased current sheet at angle $\theta_{\mathrm{inc}} =
#    20^\circ$ and verify that the Fraunhofer main lobe shifts to $\sin\theta_0 =
#    \sin\theta_{\mathrm{inc}}$ while the first nulls follow
#    $\sin\theta_{\pm 1} - \sin\theta_{\mathrm{inc}} = \pm \lambda/a$.
# 3. **Cornu-spiral on-axis extrema**:
#    Place a longitudinal 1D `DFTMonitor` along the symmetry axis $x = 0$, $y \in [0.2\lambda,
#    6\lambda]$ for $a = 3\lambda$ and compare the axial intensity oscillations $|E_z(0, y)|^2 /
#    |E_{\mathrm{inc}}|^2$ against $2\bigl[C(\sqrt{2N_F})^2 + S(\sqrt{2N_F})^2\bigr]$.
#
# ## References
#
# 1. M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University Press, 2019),
#    Ch. 8 (Elements of the Theory of Diffraction) & Ch. 11 (Rigorous Diffraction Theory),
#    DOI: [10.1017/9781108769914](https://doi.org/10.1017/9781108769914).
# 2. E. Hecht, *Optics*, 5th ed. (Pearson, 2017), Ch. 10 (Diffraction).
# 3. A. Oskooi and S. G. Johnson, "Electromagnetic Wave Source Conditions," in *Advances in FDTD
#    Computational Electrodynamics: Photonics and Nanotechnology*, ed. A. Taflove, A. Oskooi, and
#    S. G. Johnson (Artech House, 2013), Ch. 4,
#    arXiv:[1301.5366](https://arxiv.org/abs/1301.5366).
# 4. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and S. G. Johnson,
#    "Meep: A flexible free-software package for electromagnetic simulations by the FDTD method,"
#    *Comput. Phys. Commun.* **181**(3), 687–702 (2010),
#    DOI: [10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).
# 5. A. Sommerfeld, "Mathematische Theorie der Diffraction," *Math. Ann.* **47**(2–3), 317–374
#    (1896), DOI: [10.1007/BF01447273](https://doi.org/10.1007/BF01447273).
# 6. P. M. Morse and P. J. Rubenstein, "The Diffraction of Waves by Ribbons and by Slits,"
#    *Phys. Rev.* **54**(11), 895–898 (1938),
#    DOI: [10.1103/PhysRev.54.895](https://doi.org/10.1103/PhysRev.54.895).
# 7. C. J. Bouwkamp, "Diffraction theory," *Rep. Prog. Phys.* **17**(1), 35–100 (1954),
#    DOI: [10.1088/0034-4885/17/1/302](https://doi.org/10.1088/0034-4885/17/1/302).
# 8. Y. Takakura, "Optical resonance in a narrow slit in a thick metallic screen,"
#    *Phys. Rev. Lett.* **86**(24), 5601–5603 (2001),
#    DOI: [10.1103/PhysRevLett.86.5601](https://doi.org/10.1103/PhysRevLett.86.5601).
