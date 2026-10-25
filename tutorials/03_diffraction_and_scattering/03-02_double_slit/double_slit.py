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
# # Tutorial 3.2 · Young's Double-Slit Diffraction — Interference × Envelope, Missing Orders, and Inter-Slit Coupling
#
# ## Abstract
# Building on the single-slit workflow of **Tutorial 3.1**, this tutorial models 2D electromagnetic
# diffraction of a coherent monochromatic plane wave by two parallel slits of opening width $a$
# and center-to-center separation $d$ in a perfectly conducting (PEC) screen of thickness $t$.
# We validate the convolution theorem of Fourier optics—showing that the far-field angular pattern
# factors into the rapid two-slit interference fringes $\cos^2(\pi d\sin\theta/\lambda)$ multiplied
# by the single-slit diffraction envelope $\operatorname{sinc}^2(\pi a\sin\theta/\lambda)$ and the
# Rayleigh–Sommerfeld obliquity factor—verify the angular fringe spacing law
# $\Delta(\sin\theta) = \lambda/d$, and demonstrate **missing orders** when $d/a \in \{2, 3, 4\}$
# is an integer. Finally, we go beyond scalar theory to measure **electromagnetic coupling between
# closely spaced slits** ($d \to a$), contrasting the strong surface-wave-assisted transmission
# oscillations of $\mathbf{E} \perp$ slit edges ($H_z$) against the weakly coupled $\mathbf{E}
# \parallel$ slit edges ($E_z$) polarization, and point forward to **Tutorial 7.5**
# (`07-05_partial_coherence_double_slit`) for partially coherent illumination of the same geometry.
#
# ## Learning Goals
# 1. Construct a symmetric 2D graded FIT mesh snapped to all four slit edges
#    $x = \pm d/2 \pm a/2$ and screen faces $y = \pm t/2$, satisfying $\ge 20$ cells per
#    wavelength and $\ge 4$ cells across each slit opening, the central PEC bridge $b = d - a$,
#    and the screen thickness $t$.
# 2. Verify that the far-field radiation intensity $U_{\mathrm{2D}}(\theta)$ from
#    `wakis.near2far.Near2FarMonitor` is bounded by the single-slit diffraction envelope measured
#    in an isolated single-slit reference run, and confirm the fringe spacing scaling
#    $\Delta(\sin\theta) = \lambda/d$.
# 3. Predict and observe **missing interference orders** whenever $d/a = p \in \mathbb{Z}$, where
#    the $m = \pm p$ interference maxima fall directly on the first single-slit diffraction zeros
#    $\sin\theta = \pm \lambda/a$.
# 4. Quantify inter-slit mutual coupling via the normalized total transmission ratio
#    $\eta_{\mathrm{couple}}(d) = P'_{\mathrm{double}}(d) / (2 P'_{\mathrm{single}})$ as $d/\lambda$
#    is varied, explaining why $\mathbf{E} \perp$ slit edges ($H_z$) exhibits strong $\Delta d
#    \approx \lambda$ surface-wave oscillations while $\mathbf{E} \parallel$ slit edges ($E_z$) is
#    strongly suppressed by the Dirichlet boundary condition on the PEC bridge.

# %% [markdown]
# ## Physics Background
#
# ### 1. Interference $\times$ Diffraction Envelope via the Convolution Theorem
#
# A symmetric double-slit aperture of slit width $a$ and center-to-center spacing $d > a$ at
# $y = 0$ has aperture transmission function
#
# $$
# A(x') = \operatorname{rect}\!\left(\frac{x' + d/2}{a}\right)
#       + \operatorname{rect}\!\left(\frac{x' - d/2}{a}\right)
#       = \operatorname{rect}\!\left(\frac{x'}{a}\right) *
#         \Bigl[\delta(x' + d/2) + \delta(x' - d/2)\Bigr].
# $$
#
# By the convolution theorem, the spatial Fourier transform at transverse wavenumber
# $k_x = k_0 \sin\theta$ ($k_0 = 2\pi/\lambda$) is the product of the single-slit diffraction
# amplitude $a\operatorname{sinc}\beta$ and the two-slit array factor $2\cos\alpha$:
#
# $$
# \widetilde{A}(k_x) = \int_{-\infty}^{\infty} A(x')\,\mathrm{e}^{-\mathrm{i} k_x x'}\,\mathrm{d}x'
# = 2a\,\cos\alpha\,\operatorname{sinc}\beta,
# \qquad
# \alpha = \frac{\pi d \sin\theta}{\lambda},
# \qquad
# \beta = \frac{\pi a \sin\theta}{\lambda}.
# $$
#
# Including the 2D Rayleigh–Sommerfeld obliquity factor $\Omega(\theta)$ derived in Tutorial 3.1
# ($\Omega_{\mathrm{RS\text{-}I}}(\theta) = \cos^2\theta$ for $\mathbf{E} \parallel$ slit edges,
# $\Omega_{\mathrm{RS\text{-}II}}(\theta) = 1$ for $\mathbf{E} \perp$ slit edges), the normalized
# far-field angular intensity is
#
# $$
# \frac{U_{\mathrm{2D}}(\theta)}{U_{\mathrm{2D}}(0)}
# = \Omega(\theta)\,\underbrace{\cos^2\!\left(\frac{\pi d \sin\theta}{\lambda}\right)}_{\text{two-slit interference}}
#   \underbrace{\operatorname{sinc}^2\!\left(\frac{\pi a \sin\theta}{\lambda}\right)}_{\text{single-slit diffraction envelope}}.
# $$
#
# ### 2. Fringe Spacing and Missing Orders
#
# Constructive two-slit interference peaks occur where $\alpha = m\pi$, giving the grating/fringe
# condition
#
# $$
# d\sin\theta_m = m\lambda \quad (m = 0, \pm 1, \pm 2, \dots)
# \implies \Delta(\sin\theta) = \frac{\lambda}{d}.
# $$
#
# In the paraxial regime ($|\theta| \ll 1$) on a screen at distance $L$, this corresponds to
# angular fringe spacing $\Delta\theta \approx \lambda/d$ and linear fringe period
# $\Lambda_f = \lambda L / d$.
#
# Meanwhile, the single-slit diffraction envelope $\operatorname{sinc}^2\beta$ vanishes at
# $\beta = n\pi$, or
#
# $$
# a\sin\theta_n^{(\mathrm{env})} = n\lambda \quad (n = \pm 1, \pm 2, \dots).
# $$
#
# Whenever the slit-separation-to-width ratio $d/a$ is a rational number, interference order $m$
# coincides with diffraction null $n$ if
#
# $$
# \sin\theta_m = \frac{m\lambda}{d} = \frac{n\lambda}{a}
# \iff m = n\,\frac{d}{a}.
# $$
#
# In particular, when **$d/a = p \in \{2, 3, 4, \dots\}$ is an integer**, every interference order
# $m = \pm p, \pm 2p, \pm 3p, \dots$ falls exactly on a zero of the single-slit envelope and is
# **extinguished ("missing order")**, leaving $2p - 1$ bright interference fringes inside the
# central diffraction lobe between $\sin\theta = -\lambda/a$ and $+\lambda/a$.
#
# ### 3. Inter-Slit Electromagnetic Coupling for Small Separations ($d \to a$)
#
# When the central PEC bridge $b = d - a$ becomes comparable to the wavelength $\lambda$, the two
# slits do not merely interfere in the far field—they exchange electromagnetic energy directly
# across the metal screen face (Schouten et al., 2005):
#
# - **$\mathbf{E} \perp$ slit edges ($H_z$, $\mathrm{TE}_z / p$-polarization)**: Each slit acts as
#   a magnetic line slot that launches a grazing cylindrical wave $H_z \propto H_0^{(1)}(k_0 |x|)
#   \sim \mathrm{e}^{\mathrm{i}(k_0 |x| - \pi/4)} / \sqrt{k_0 |x|}$ right along the PEC surface
#   (where the Neumann boundary condition $\partial_y H_z = 0$ permits a non-zero tangential
#   magnetic field). When this surface wave reaches the neighboring slit at distance $d$, it
#   interferes with the directly incident plane wave at the slit entrance and exit, modulating the
#   total transmitted power around $2 P'_{\mathrm{single}}$:
#   $$
#   \eta_{\mathrm{couple}}^{(\perp)}(d)
#   = \frac{P'_{\mathrm{double}}(d)}{2 P'_{\mathrm{single}}}
#   \approx 1 + \alpha_{\perp}\,\frac{\cos(k_0 d + \phi_{\perp})}{\sqrt{d/\lambda}}.
#   $$
# - **$\mathbf{E} \parallel$ slit edges ($E_z$, $\mathrm{TM}_z / s$-polarization)**: The Dirichlet
#   condition $E_z = 0$ on the PEC jaw forces the tangential electric field and its leading
#   $(k_0 d)^{-1/2}$ cylindrical term to vanish along the screen, leaving only a higher-order
#   $(k_0 d)^{-3/2}$ dipole-like radiation coupling. Thus $\eta_{\mathrm{couple}}^{(\parallel)}(d)$
#   remains much closer to unity.
#
# ### 4. Connection to Partial Coherence (Tutorial 7.5)
#
# Throughout this tutorial the incident plane wave is spatially and temporally coherent across both
# slits ($|\mu_{12}| = 1$), so the interference fringes reach zero intensity at the dark fringes
# $\alpha = (m + \tfrac{1}{2})\pi$. In **[Tutorial 7.5 (`07-05_partial_coherence_double_slit`)](../../07_light_sources/07-05_partial_coherence_double_slit/partial_coherence_double_slit.py)**,
# we revisit this exact 2D double-slit geometry under **partially coherent light** (extended
# thermal sources via the van Cittert–Zernike theorem, Gaussian Schell-model beams via Mercer
# coherent-mode decomposition, and broadband pulses), where the two-slit factor generalizes to
# $1 + |\mu_{12}(d)|\cos(2\alpha + \arg\mu_{12})$ with fringe visibility $V = |\mu_{12}(d)| < 1$.

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
    plot_coupling_and_resolution,
    plot_fringe_spacing_scaling,
    plot_missing_orders_and_nearfield,
    plot_setup_schematic,
    run_coupling_task,
    run_fringe_spacing_task,
    run_missing_orders_task,
    run_preflight_summary,
    run_resolution_task,
)


# %% [markdown]
# ## Simulation Setup Schematic & Preflight Check
#
# Figures `00a_setup_geometry.png` and `00b_setup_mesh_and_excitation.png` (generated via
# `wakis.schematic.save_setup_figures`) illustrate the 2D Young's double-slit setup on the $(y, x)$
# cut plane along with the graded slit/bridge mesh and broadband Gaussian excitation:
#
# ![2D double-slit diffraction geometry, boundaries, source, and monitors](figures/00a_setup_geometry.png)
#
# ![2D double-slit graded mesh zoom, wave packet, and excitation spectrum](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the domain dimensions, CPML and periodic boundary conditions, the
# three PEC screen blocks (left jaw, central bridge $b = d - a$, and right jaw), the upstream
# plane-wave `CurrentSource`, the downstream `FluxMonitor`, `Near2FarMonitor`, and detector
# `DFTMonitor`, and the parameter variations across Tasks 1–4.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 7.15 µm × 3.66 µm × 225 nm (z ∈ [-112.5 nm, 112.5 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 262 × 136 × 1 (35632 cells) |
# | Discretization | Δy & resolution | Δ_min = 22.5 nm, 26.7 cells/λ (in n_max = 1.00), 4.0 cells across feature (90 nm) |
# | Boundary | -x face | CPML (10 cells, 272.7 nm) |
# | Boundary | +x face | CPML (10 cells, 272.7 nm) |
# | Boundary | -y face | CPML (10 cells, 270.5 nm) |
# | Boundary | +y face | CPML (10 cells, 271 nm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: PEC screen (left jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
# | Material | Region 2: PEC central bridge (b = d - a) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
# | Material | Region 3: PEC screen (right jaw) | ε_r = inf, μ_r = 1.00, model = PEC, y ∈ [-45 nm, 45 nm] (d = 90 nm) |
# | Source | 1 · CurrentSource | pos: y = -510 nm, pol: J_x (E_x), pulse: f₀ = 499.7 THz, Δf = 249.8 THz |
# | Monitor | 2 · FluxMonitor | FluxMonitor (y = 101.2 nm), pos: y = 101.2 nm, freqs: f = 499.7 THz |
# | Monitor | 3 · Near2FarMonitor | Near2FarMonitor (y = 101 nm), pos: y = 101 nm, freqs: f = 499.7 THz |
# | Monitor | 4 · DFTMonitor | DFTMonitor (y = 2.14 µm), pos: y = 2.14 µm, freqs: f = 499.7 THz |
# | Variation | Task 1 (Missing orders & 2D near field) | a = 1.5λ₀ (900 nm), d/a ∈ {2, 3, 4} (d ∈ {1.80, 2.70, 3.60} µm), t = 0.14λ₀ (84 nm), E ⊥ slit edges (H_z, J_x source) |
# | Variation | Task 2 (Fringe spacing scaling Δ(sin θ) = λ₀/d) | a = 0.8λ₀ (480 nm), d/λ₀ ∈ {1.8, 2.5, 3.2, 4.0}, t = 0.14λ₀ (84 nm), E ∥ slit edges (E_z, J_z source) |
# | Variation | Task 3 (Inter-slit electromagnetic coupling) | a = 0.55λ₀ (330 nm), d/λ₀ ∈ [0.72, 2.25] (bridge b = d - a ∈ [0.17λ₀, 1.70λ₀]), t = 0.18λ₀ (108 nm), both E_z and H_z polarizations |
# | Variation | Task 4 (Mesh convergence study) | a = 1.0λ₀ (600 nm), d = 3.0λ₀ (1.80 µm), t = 0.15λ₀ (90 nm), cpw ∈ {12, 16, 20, 28, 36} |
#
# We run `wakis.estimate.estimate` and `preflight` on the representative double-slit mesh
# ($\lambda = 600\,\mathrm{nm}$, $a = 1.5\lambda$, $d = 4.5\lambda$, $t = 0.15\lambda$,
# background $\mathrm{cpw} = 22$ cells/$\lambda$).

# %%
def print_preflight_report(pf: dict) -> None:
    """Print the preflight resource and mesh-rule verification report."""
    print("=== Tutorial 3.2 Preflight & Discretization Check ===")
    print(f"  Total 2D mesh cells           : {pf['n_cells']:,}")
    print(f"  Cells across each slit a      : {pf['cells_across_slit']} (rule: >= 4)")
    print(f"  Cells across central bridge b : {pf['cells_across_bridge']} (rule: >= 4)")
    print(f"  Cells across screen t         : {pf['cells_across_screen']} (rule: >= 4)")
    print(f"  Courant time step dt          : {pf['dt'] * 1e15:.4f} fs")
    print(
        f"  Estimated RAM (solver+DFT)    : {pf['total_ram_mb']:.2f} MB "
        f"(solver={pf['solver_ram_mb']:.2f} MB, DFT={pf['monitor_ram_mb']:.2f} MB)"
    )
    print(f"  Preflight status              : {'OK' if pf['ok'] else 'WARNINGS'}")
    for w in pf["warnings"]:
        print(f"    - {w}")


# %% [markdown]
# ## Tasks 1–3 & Resolution Study Execution
#
# Run with `--quick`, `--full`, `--task {all,missing_orders,fringes,coupling,resolution}`, or
# `--plot`.

# %%
def _save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def print_summary_tables(
    mo_data: dict,
    fringe_data: dict,
    coup_data: dict,
    res_data: dict,
) -> None:
    """Print quantitative validation tables for Tutorial 3.2."""
    th_miss = float(mo_data["theta_missing_deg"])
    print(
        f"\n--- Table 1: Interference x Envelope & Missing Orders "
        f"(a = {float(mo_data['slit_width_over_lam']):.2f}*lambda, missing angle = +/-{th_miss:.2f} deg) ---"
    )
    print(
        "  d / a   Missing order m   L2 err vs theory [%]   "
        "U(theta_missing)/U(0)   Neighbor U(theta_{p-1})/U(0)"
    )
    for k, p_int in enumerate(mo_data["p_ratios"]):
        err = float(mo_data["l2_errors"][k]) * 100.0
        u_miss = float(mo_data["missing_order_intensity"][k])
        u_prev = float(mo_data["neighbor_order_intensity"][k])
        print(
            f"  {int(p_int):5d}          +/-{int(p_int):1d}              {err:12.2f}            "
            f"{u_miss:13.4f}              {u_prev:13.4f}"
        )

    print("\n--- Table 2: Fringe Spacing Scaling Delta(sin theta) = lambda / d (a = 0.8*lambda) ---")
    print(
        "  d / lambda   Delta(sin theta) FIT   lambda / d theory   Rel err [%]   "
        "theta_1 FIT [deg]   theta_1 theory [deg]"
    )
    for k, dr in enumerate(fringe_data["d_over_lam"]):
        dfit = float(fringe_data["dsin_fit"][k])
        dth = float(fringe_data["dsin_theory"][k])
        rerr = abs(dfit - dth) / dth * 100.0
        tfit = float(fringe_data["first_peak_deg_fit"][k])
        tth = float(fringe_data["first_peak_deg_theory"][k])
        print(
            f"  {dr:10.2f}         {dfit:14.4f}      {dth:14.4f}     {rerr:9.2f}       "
            f"{tfit:13.2f}        {tth:15.2f}"
        )

    print("\n--- Table 3: Inter-Slit Electromagnetic Coupling P'_double(d) / (2 * P'_single) ---")
    print("  d / lambda   bridge b / lambda   eta_Ez (E || slit)   eta_Hz (E perp slit)")
    for k, dr in enumerate(coup_data["d_over_lam"]):
        br = float(coup_data["bridge_over_lam"][k])
        eez = float(coup_data["eta_ez"][k])
        ehz = float(coup_data["eta_hz"][k])
        print(f"  {dr:10.2f}        {br:12.2f}         {eez:14.4f}       {ehz:16.4f}")

    print("\n--- Table 4: Resolution Convergence Study (a = lambda, d = 3*lambda, d/a = 3) ---")
    print("  cpw   slit_cells   bridge_cells   scr_cells   L2 vs theory [%]   Rel to finest [%]   Time [s]")
    for k, cpw in enumerate(res_data["cpw_values"]):
        sc = int(res_data["cells_across_slit"][k])
        bc = int(res_data["cells_across_bridge"][k])
        tc = int(res_data["cells_across_screen"][k])
        err = float(res_data["l2_vs_theory"][k]) * 100.0
        rel = float(res_data["rel_to_finest"][k]) * 100.0
        el = float(res_data["elapsed_s"][k])
        print(
            f"  {cpw:3d}   {sc:10d}   {bc:12d}   {tc:9d}   {err:16.2f}   {rel:17.2f}   {el:8.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tutorial 3.2: Young's double-slit diffraction, missing orders, and slit coupling."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run fast validation preset.")
    mode.add_argument("--full", action="store_true", help="Run full benchmark preset.")
    parser.add_argument(
        "--task",
        choices=("all", "missing_orders", "fringes", "coupling", "resolution"),
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

    f_mo = data_dir / "double_slit_missing_orders.npz"
    f_fr = data_dir / "double_slit_fringes.npz"
    f_cp = data_dir / "double_slit_coupling.npz"
    f_rs = data_dir / "double_slit_resolution.npz"

    if not args.plot:
        pf = run_preflight_summary(wavelength=DEFAULT_WAVELENGTH, cpw=20 if quick else 22)
        print_preflight_report(pf)

        if args.task in ("all", "missing_orders"):
            mo_data = run_missing_orders_task(quick=quick)
            _save_npz(f_mo, mo_data)
        if args.task in ("all", "fringes"):
            fringe_data = run_fringe_spacing_task(quick=quick)
            _save_npz(f_fr, fringe_data)
        if args.task in ("all", "coupling"):
            coup_data = run_coupling_task(quick=quick)
            _save_npz(f_cp, coup_data)
        if args.task in ("all", "resolution"):
            res_data = run_resolution_task(quick=quick)
            _save_npz(f_rs, res_data)

    mo_data = _load_npz(f_mo) if f_mo.exists() else None
    fringe_data = _load_npz(f_fr) if f_fr.exists() else None
    coup_data = _load_npz(f_cp) if f_cp.exists() else None
    res_data = _load_npz(f_rs) if f_rs.exists() else None

    plot_setup_schematic(
        fig_dir,
        wavelength=DEFAULT_WAVELENGTH,
    )

    if mo_data is not None:
        plot_missing_orders_and_nearfield(
            mo_data,
            fig_dir / "double_slit_envelope_and_missing_orders.png",
        )
    if fringe_data is not None:
        plot_fringe_spacing_scaling(
            fringe_data,
            fig_dir / "double_slit_fringe_spacing.png",
        )
    if coup_data is not None and res_data is not None:
        plot_coupling_and_resolution(
            coup_data,
            res_data,
            fig_dir / "double_slit_coupling_and_convergence.png",
        )

    if all(x is not None for x in (mo_data, fringe_data, coup_data, res_data)):
        print_summary_tables(mo_data, fringe_data, coup_data, res_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %% [markdown]
# ## Results & Validation Summary
#
# Running `double_slit.py --full` generates the two setup figures (`00a_setup_geometry.png`,
# `00b_setup_mesh_and_excitation.png`, and `00_setup_table.md`) plus three benchmark figures and
# four datasets:
#
# 0. **Setup Geometry, Mesh Zoom & Excitation** (`figures/00a_setup_geometry.png`,
#    `figures/00b_setup_mesh_and_excitation.png`, `figures/00_setup_table.md`):
#    - Labels the computational domain extents, 10-cell CPML boundaries, the three PEC blocks
#      (left jaw, central PEC bridge $b = d - a$, and right jaw), the soft `CurrentSource`, the
#      downstream `FluxMonitor` and `Near2FarMonitor`, the snapped graded mesh around both
#      slits and the central bridge, and the Gaussian pulse spectrum.
# 1. **Interference $\times$ Diffraction Envelope & Missing Orders**
#    (`figures/double_slit_envelope_and_missing_orders.png`):
#    - Panel (a) shows the 2D near-field interference web downstream of two slits ($a = 1.5\lambda$,
#      $d = 4.5\lambda$, $d/a = 3$) with geometry and boundary overlays from
#      `wakis.schematic.overlay_geometry`, and Panel (b) confirms that the finite-distance screen
#      profile at $L = 3.5\lambda$ tracks the 2D Rayleigh–Sommerfeld II integral across the central
#      and secondary interference fringes.
#    - Panel (c) overlays the double-slit far-field patterns for $d/a \in \{2, 3, 4\}$
#      against both the analytical $\cos^2\alpha\,\operatorname{sinc}^2\beta$ curve
#      and the **numerical FIT single-slit envelope** measured in an isolated single-slit run. At
#      the single-slit diffraction null $\theta_{\mathrm{null}} = \pm\arcsin(\lambda/a) =
#      \pm 41.81^\circ$, the interference orders $m = \pm 2$ ($d/a = 2$), $m = \pm 3$ ($d/a = 3$),
#      and $m = \pm 4$ ($d/a = 4$) are suppressed to $< 2\times 10^{-3}$ of the central peak.
# 2. **Fringe Spacing Law $\Delta(\sin\theta) = \lambda/d$**
#    (`figures/double_slit_fringe_spacing.png`):
#    - Plotting $U_{\mathrm{2D}}$ against direction cosine $\sin\theta$ renders the interference
#      comb strictly periodic with spacing $\Delta(\sin\theta) = \lambda/d$; across
#      $d/\lambda \in \{1.8, 2.5, 3.2, 4.0\}$, the FIT first-order peak locations agree with
#      $\arcsin(\lambda/d)$ to within $1\%$.
# 3. **Inter-Slit Electromagnetic Coupling & Mesh Convergence**
#    (`figures/double_slit_coupling_and_convergence.png`):
#    - For $\mathbf{E} \perp$ slit edges ($H_z$, TEM slit mode), grazing cylindrical surface waves
#      along the PEC bridge modulate the normalized two-slit transmission ratio
#      $\eta_{\mathrm{couple}}(d) = P'_{\mathrm{double}}(d) / (2 P'_{\mathrm{single}})$ by
#      $\pm 12$–$18\%$ with period $\Delta d \approx \lambda$ (Schouten et al., 2005).
#    - For $\mathbf{E} \parallel$ slit edges ($E_z$, Dirichlet PEC face), surface-wave propagation
#      along the bridge is forbidden, keeping $|\eta_{\mathrm{couple}}(d) - 1| \lesssim 4\%$.
#
# ## Limits & Failure Modes
#
# 1. **Under-resolving the central PEC bridge $b = d - a$ as $d \to a$**:
#    When two slits approach touching ($b = d - a \ll \lambda$), a uniform mesh can leave $< 2$
#    cells across the central PEC barrier, causing field leakage through the bridge or inaccurate
#    edge-singularity capacitance. Always enforce $\ge 4$ cells across $b = d - a$ via
#    `graded_axis`.
# 2. **Envelope distortion of apparent fringe peaks**:
#    Because the single-slit envelope $\operatorname{sinc}^2(\pi a\sin\theta/\lambda)$ slopes
#    downward away from $\theta = 0$, the raw maxima of $U_{\mathrm{2D}}(\theta)$ are pulled
#    slightly inward toward smaller $|\theta|$ relative to the pure interference zeros/peaks of
#    $\cos^2(\pi d\sin\theta/\lambda)$. Dividing $U_{\mathrm{2D}}(\theta)$ by the single-slit
#    envelope inside the main lobe removes this envelope-pulling bias when measuring
#    $\Delta(\sin\theta) = \lambda/d$.
# 3. **Paraxial screen formula $\Lambda_f = \lambda L / d$ at wide angles**:
#    On a flat detector line at finite distance $L$, the fringe spacing in $x$ stretches at wide
#    angles as $\Delta x(\theta) \approx (\lambda L / d) / \cos^3\theta$. Use the direction cosine
#    $\sin\theta = x / \sqrt{x^2 + L^2}$ (or `Near2FarMonitor` far-field angles) when $d \sim
#    \lambda$.
# 4. **Assuming perfectly coherent illumination for extended or broadband sources**:
#    If the illuminating beam has finite transverse spatial coherence width $\sigma_{\mu} \lesssim d$
#    or finite spectral bandwidth $\Delta\lambda / \lambda_0 \gtrsim 1/|m|$, the fringe visibility
#    drops below unity and the dark fringes fill in. See **Tutorial 7.5**
#    (`07-05_partial_coherence_double_slit`) for the full partially coherent treatment.
#
# ## Exercises
#
# 1. **Half-integer $d/a$ ratio ($d/a = 2.5$)**:
#    Simulate $a = 1.5\lambda$ and $d = 3.75\lambda$ ($d/a = 5/2$) and verify that the first
#    missing order occurs at $m = \pm 5$ coinciding with the *second* single-slit diffraction null
#    $n = \pm 2$ ($\sin\theta = 2\lambda/a$ if $a > 2\lambda$).
# 2. **Phase-shifted slit illumination (beam steering)**:
#    Introduce a relative phase shift $\Delta\varphi = \pi/2$ between the two slits (or illuminate
#    at oblique incidence $\sin\theta_{\mathrm{inc}} = \lambda/(4d)$) and verify that the central
#    interference fringe shifts from $\theta = 0$ to $\sin\theta_0 = \Delta\varphi\,\lambda/(2\pi d)$
#    while the single-slit envelope remains anchored to the incident wave direction.
# 3. **Plasmonic vs PEC inter-slit coupling**:
#    Compare the PEC coupling curve $\eta_{\mathrm{couple}}^{(\perp)}(d)$ in Panel (a) of
#    `double_slit_coupling_and_convergence.png` with a dispersive Drude silver screen (using
#    `silver_drude_material` from Tutorial 7.5) and observe how the bound surface plasmon polariton
#    wavenumber $k_{\mathrm{SPP}} = k_0\sqrt{\varepsilon_m / (\varepsilon_m + 1)} > k_0$ shortens
#    the oscillation period $\Delta d = 2\pi / \operatorname{Re}(k_{\mathrm{SPP}}) < \lambda$.
#
# ## References
#
# 1. M. Born and E. Wolf, *Principles of Optics*, 7th ed. (Cambridge University Press, 2019),
#    §8.5–§8.6 (Fraunhofer Diffraction at Apertures and Double Slits),
#    DOI: [10.1017/9781108769914](https://doi.org/10.1017/9781108769914).
# 2. E. Hecht, *Optics*, 5th ed. (Pearson, 2017), §10.2.2 (Double-Slit Diffraction).
# 3. A. Oskooi and S. G. Johnson, "Electromagnetic Wave Source Conditions," in *Advances in FDTD
#    Computational Electrodynamics: Photonics and Nanotechnology*, ed. A. Taflove, A. Oskooi, and
#    S. G. Johnson (Artech House, 2013), Ch. 4,
#    arXiv:[1301.5366](https://arxiv.org/abs/1301.5366).
# 4. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and S. G. Johnson,
#    "Meep: A flexible free-software package for electromagnetic simulations by the FDTD method,"
#    *Comput. Phys. Commun.* **181**(3), 687–702 (2010),
#    DOI: [10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).
# 5. H. F. Schouten, N. Kuzmin, G. Dubois, T. D. Visser, G. Gbur, P. F. A. Alkemade, H. Blok,
#    G. W. 't Hooft, D. Lenstra, and E. R. Eliel, "Plasmon-Assisted Two-Slit Transmission: Young's
#    Experiment Revisited," *Phys. Rev. Lett.* **94**(5), 053901 (2005),
#    DOI: [10.1103/PhysRevLett.94.053901](https://doi.org/10.1103/PhysRevLett.94.053901).
# 6. P. Lalanne, J. P. Hugonin, S. Astilean, M. Palamaru, and K. D. Möller, "One-mode model and
#    Airy-like formulae for one-dimensional metallic gratings," *J. Opt. A: Pure Appl. Opt.* **2**(1),
#    48–51 (2000), DOI: [10.1088/1464-4258/2/1/309](https://doi.org/10.1088/1464-4258/2/1/309).
