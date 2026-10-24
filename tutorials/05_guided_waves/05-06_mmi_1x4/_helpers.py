# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Device-specific helpers for Tutorial 5.6 (1x4 Si-GaN-Si vertical-slot MMI
power splitter) that extend `wakis.templates.photonics` without modifying
`wakis/`:

1. 5-layer symmetric vertical-slot slab dispersion solver (`slot_slab_neff`),
   1D vertical mode profile (`vertical_slot_mode`), equivalent 3-layer core
   index mapping (`equivalent_3layer_core_index`), and 3-panel EIM stack figure
   (`plot_slot_eim_stack`).
2. Analytical Soldano & Pennings (1995) self-imaging beat-length estimates
   (`mmi_analytic_estimates`), staircase MMI width quantization and numerical
   dispersion diagnostics (`mmi_discretization_diagnostics`), and 1D lateral
   eigenmode expansion (EME) sweep (`mmi_eme_length_sweep`).
"""

from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from wakis.templates.photonics import slab_neff


def _slot_slab_residual(
    neff, n_rail, n_slot, n_clad, h_rail, h_slot, wavelength, polarization="TM"
):
    """
    Even-mode transfer-matrix residual at the top rail/cladding interface for a
    symmetric 5-layer stack `clad | rail(h_rail) | slot(h_slot) | rail(h_rail) | clad`.

    Parameters
    ----------
    neff : float
        Trial effective index (dimensionless).
    n_rail, n_slot, n_clad : float
        Refractive indices of the high-index rails (Si), central low-index slot
        (GaN), and surrounding cladding (SiO2) (dimensionless).
    h_rail, h_slot : float
        Single-rail thickness and central slot thickness [m].
    wavelength : float
        Vacuum wavelength [m].
    polarization : {'TM', 'TE'}, optional
        Vertical polarization: `'TM'` (dominant `E_z` normal to the horizontal
        slot interfaces, continuous `H_y` and `(1/n^2) dH_y/dz`) or `'TE'`
        (dominant in-plane `E_y`, continuous `E_y` and `dE_y/dz`).

    Returns
    -------
    residual : float
        Boundary-matching residual at `z = +h_slot/2 + h_rail`.
    """
    pol = polarization.upper()
    k0 = 2.0 * np.pi / float(wavelength)
    v = np.array([1.0, 0.0], dtype=float)
    for n_layer, d_layer in ((n_slot, 0.5 * h_slot), (n_rail, h_rail)):
        w = float(n_layer) ** 2 if pol == "TM" else 1.0
        if neff < n_layer:
            k = k0 * np.sqrt(max(n_layer**2 - neff**2, 0.0))
            c, s = np.cos(k * d_layer), np.sin(k * d_layer)
            m_mat = np.array([[c, (w / k) * s], [-(k / w) * s, c]], dtype=float)
        else:
            g = k0 * np.sqrt(max(neff**2 - n_layer**2, 1e-30))
            ch, sh = np.cosh(g * d_layer), np.sinh(g * d_layer)
            m_mat = np.array([[ch, (w / g) * sh], [(g / w) * sh, ch]], dtype=float)
        v = m_mat @ v
    w_clad = float(n_clad) ** 2 if pol == "TM" else 1.0
    g_clad = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
    return float(v[1] + (g_clad / w_clad) * v[0])


def slot_slab_neff(
    n_rail=3.48,
    n_slot=2.305,
    n_clad=1.444,
    h_rail=0.30e-6,
    h_slot=0.10e-6,
    wavelength=1.55e-6,
    polarization="TM",
):
    """
    Fundamental even vertical mode effective index of a symmetric 5-layer
    `clad / rail / slot / rail / clad` sandwich waveguide.

    Parameters
    ----------
    n_rail, n_slot, n_clad : float, optional
        Refractive indices of the rails (Si), central slot (GaN), and cladding
        (SiO2) (dimensionless).
    h_rail, h_slot : float, optional
        Single-rail thickness and central slot thickness [m].
    wavelength : float, optional
        Vacuum wavelength [m].
    polarization : {'TM', 'TE'}, optional
        Vertical polarization (`'TM'` for the vertical-slot mode with dominant
        `E_z` confined in the GaN slot; `'TE'` for the in-plane `E_y` mode).

    Returns
    -------
    neff : float
        Fundamental even-mode effective index (dimensionless).
    """
    zs = np.linspace(float(n_clad) + 1e-5, float(n_rail) - 1e-5, 1200)
    vals = [
        _slot_slab_residual(
            z, n_rail, n_slot, n_clad, h_rail, h_slot, wavelength, polarization
        )
        for z in zs
    ]
    roots = []
    for i in range(len(zs) - 1):
        if vals[i] == 0.0:
            roots.append(float(zs[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            r = brentq(
                _slot_slab_residual,
                zs[i],
                zs[i + 1],
                args=(
                    n_rail,
                    n_slot,
                    n_clad,
                    h_rail,
                    h_slot,
                    wavelength,
                    polarization,
                ),
            )
            roots.append(float(r))
    if not roots:
        raise RuntimeError(
            f"No guided even {polarization} mode found for 5-layer slot stack at lambda={wavelength:.3e} m."
        )
    return float(max(roots))


def equivalent_3layer_core_index(
    n_eff_target, n_clad=1.444, h_total=0.70e-6, wavelength=1.55e-6
):
    """
    Find the equivalent 3-layer core refractive index `n_core_eq` on a slab of
    thickness `h_total` such that `slab_neff(n_core_eq, n_clad, h_total,
    wavelength, polarization='TE') == n_eff_target`.

    Parameters
    ----------
    n_eff_target : float
        Target 2D effective index (e.g. from `slot_slab_neff`) (dimensionless).
    n_clad : float, optional
        Cladding refractive index (dimensionless).
    h_total : float, optional
        Total vertical stack thickness `2 * h_rail + h_slot` [m].
    wavelength : float, optional
        Reference vacuum wavelength [m].

    Returns
    -------
    n_core_eq : float
        Equivalent 3-layer core index (dimensionless) to pass to `PICSimulation`.
    """
    return float(
        brentq(
            lambda nc: (
                slab_neff(nc, n_clad, h_total, wavelength, polarization="TE")
                - n_eff_target
            ),
            float(n_eff_target) + 1e-6,
            float(n_eff_target) + 1.5,
        )
    )


def vertical_slot_mode(
    n_rail=3.48,
    n_slot=2.305,
    n_clad=1.444,
    h_rail=0.30e-6,
    h_slot=0.10e-6,
    wavelength=1.55e-6,
    z_span=2.0e-6,
    n_pts=601,
):
    """
    Analytical 1D vertical-slot TM mode profiles `E_z(z)`, `H_y(z)`, and
    refractive-index profile `n(z)` across the 5-layer `SiO2 / Si / GaN / Si / SiO2`
    sandwich stack.

    Parameters
    ----------
    n_rail, n_slot, n_clad : float, optional
        Refractive indices of the Si rails, GaN slot, and SiO2 cladding
        (dimensionless).
    h_rail, h_slot : float, optional
        Single-rail thickness and central slot thickness [m].
    wavelength : float, optional
        Vacuum wavelength [m].
    z_span : float, optional
        Total vertical coordinate span `[-z_span/2, +z_span/2]` [m].
    n_pts : int, optional
        Number of vertical grid points.

    Returns
    -------
    result : dict
        Dictionary with keys `'z'` [m], `'n_profile'`, `'ez'` (normalized normal
        electric field `E_z(z) / max|E_z|`), `'hy'` (normalized transverse
        magnetic field `H_y(z) / max|H_y|`), `'neff'` (fundamental TM slot mode
        effective index), `'neff_te'` (fundamental TE mode effective index),
        `'slot_confinement'` (fraction of `|E_z|^2` inside the GaN slot), and
        `'stack_confinement'` (fraction of `|E_z|^2` inside the entire
        `Si-GaN-Si` sandwich).
    """
    neff = slot_slab_neff(
        n_rail=n_rail,
        n_slot=n_slot,
        n_clad=n_clad,
        h_rail=h_rail,
        h_slot=h_slot,
        wavelength=wavelength,
        polarization="TM",
    )
    neff_te = slot_slab_neff(
        n_rail=n_rail,
        n_slot=n_slot,
        n_clad=n_clad,
        h_rail=h_rail,
        h_slot=h_slot,
        wavelength=wavelength,
        polarization="TE",
    )
    k0 = 2.0 * np.pi / float(wavelength)
    z = np.linspace(-0.5 * float(z_span), 0.5 * float(z_span), int(n_pts))
    za = np.abs(z)
    z1 = 0.5 * float(h_slot)
    z2 = z1 + float(h_rail)

    w_s = float(n_slot) ** 2
    w_r = float(n_rail) ** 2
    g_s = k0 * np.sqrt(max(neff**2 - n_slot**2, 1e-30))
    k_r = k0 * np.sqrt(max(n_rail**2 - neff**2, 1e-30))
    g_c = k0 * np.sqrt(max(neff**2 - n_clad**2, 1e-30))

    # At z = z1: H_y(z1) = cosh(g_s * z1), (1/w) dH_y/dz(z1) = (g_s / w_s) sinh(g_s * z1)
    hy_z1 = np.cosh(g_s * z1)
    q_z1 = (g_s / w_s) * np.sinh(g_s * z1)
    # For z in [z1, z2]: H_y(z) = hy_z1 cos(k_r(z-z1)) + (w_r / k_r) q_z1 sin(k_r(z-z1))
    hy_z2 = hy_z1 * np.cos(k_r * h_rail) + (w_r / k_r) * q_z1 * np.sin(k_r * h_rail)

    in_slot = za <= z1
    in_rail = (za > z1) & (za <= z2)
    in_clad = za > z2

    hy = np.empty_like(z)
    hy[in_slot] = np.cosh(g_s * za[in_slot])
    hy[in_rail] = hy_z1 * np.cos(k_r * (za[in_rail] - z1)) + (
        w_r / k_r
    ) * q_z1 * np.sin(k_r * (za[in_rail] - z1))
    hy[in_clad] = hy_z2 * np.exp(-g_c * (za[in_clad] - z2))

    n_prof = np.where(in_slot, n_slot, np.where(in_rail, n_rail, n_clad))
    ez = hy / (n_prof**2)
    ez /= np.max(np.abs(ez))
    hy /= np.max(np.abs(hy))

    dz = z[1] - z[0]
    tot_e2 = float(np.sum(ez**2) * dz)
    slot_conf = float(np.sum(ez[in_slot] ** 2) * dz / tot_e2)
    stack_conf = float(np.sum(ez[in_slot | in_rail] ** 2) * dz / tot_e2)

    return {
        "z": z,
        "n_profile": n_prof,
        "ez": ez,
        "hy": hy,
        "neff": neff,
        "neff_te": neff_te,
        "slot_confinement": slot_conf,
        "stack_confinement": stack_conf,
    }


def plot_slot_eim_stack(
    n_rail=3.48,
    n_slot=2.305,
    n_clad=1.444,
    h_rail=0.30e-6,
    h_slot=0.10e-6,
    wavelength_ref=1.55e-6,
    wavelengths=None,
    out_path=None,
):
    """
    Plot the 5-layer `SiO2 / Si / GaN / Si / SiO2` vertical slot stack, the 1D
    vertical-slot TM mode profile (`|E_z(z)|` and `|H_y(z)|`), and the slab
    effective-index dispersion `n_eff(lambda)` across the C-band.
    """
    if wavelengths is None:
        wavelengths = np.linspace(1.50e-6, 1.60e-6, 51)
    vm = vertical_slot_mode(
        n_rail=n_rail,
        n_slot=n_slot,
        n_clad=n_clad,
        h_rail=h_rail,
        h_slot=h_slot,
        wavelength=wavelength_ref,
    )
    neff_tm_band = np.array(
        [
            slot_slab_neff(n_rail, n_slot, n_clad, h_rail, h_slot, w, polarization="TM")
            for w in wavelengths
        ]
    )
    neff_te_band = np.array(
        [
            slot_slab_neff(n_rail, n_slot, n_clad, h_rail, h_slot, w, polarization="TE")
            for w in wavelengths
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), dpi=150)
    z1_nm = 0.5 * h_slot * 1e9
    z2_nm = (0.5 * h_slot + h_rail) * 1e9
    ax0, ax1, ax2 = axes

    ax0.axhspan(-750, -z2_nm, color="#d9e6f2", label=f"SiO₂ clad (n={n_clad:.3f})")
    ax0.axhspan(
        -z2_nm,
        -z1_nm,
        color="#2b5c8f",
        alpha=0.88,
        label=f"Si rails ({h_rail * 1e9:.0f} nm × 2, n={n_rail:.2f})",
    )
    ax0.axhspan(
        -z1_nm,
        z1_nm,
        color="#d95f02",
        alpha=0.85,
        label=f"GaN slot ({h_slot * 1e9:.0f} nm, n={n_slot:.3f})",
    )
    ax0.axhspan(z1_nm, z2_nm, color="#2b5c8f", alpha=0.88)
    ax0.axhspan(z2_nm, 750, color="#d9e6f2")
    for zb in (-z2_nm, -z1_nm, z1_nm, z2_nm):
        ax0.axhline(zb, color="#16324f", lw=1.1)
    ax0.set(
        ylim=(-750, 750),
        xlim=(0, 1),
        xticks=[],
        ylabel="Vertical coordinate z [nm]",
        title="(a) 5-layer Si–GaN–Si slot stack",
    )
    ax0.legend(loc="upper right", fontsize=7.8)

    z_nm = vm["z"] * 1e9
    ax1.axvspan(-z2_nm, -z1_nm, color="#2b5c8f", alpha=0.14, label="Si rails")
    ax1.axvspan(-z1_nm, z1_nm, color="#d95f02", alpha=0.22, label="GaN slot")
    ax1.axvspan(z1_nm, z2_nm, color="#2b5c8f", alpha=0.14)
    ax1.plot(
        z_nm,
        vm["ez"],
        color="#d95f02",
        lw=2.1,
        label=r"$|E_z(z)|$ (TM slot mode)",
    )
    ax1.plot(
        z_nm,
        vm["ez"] ** 2,
        color="#1b9e77",
        lw=1.6,
        ls="--",
        label=f"$|E_z|^2$ (Γ_stack={vm['stack_confinement'] * 100:.1f}%)",
    )
    ax1.plot(
        z_nm,
        vm["hy"],
        color="#2b5c8f",
        lw=1.4,
        ls=":",
        label=r"$|H_y(z)|$ (continuous)",
    )
    ax1.set(
        xlim=(-750, 750),
        ylim=(0, 1.08),
        xlabel="Vertical coordinate z [nm]",
        ylabel="Normalized modal amplitude",
        title=f"(b) Vertical slot mode (n_eff = {vm['neff']:.4f} @ 1550 nm)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.5)

    wl_nm = np.asarray(wavelengths) * 1e9
    ax2.plot(
        wl_nm,
        neff_tm_band,
        color="#d95f02",
        lw=2.1,
        label=r"$n_{\mathrm{eff,slot}}(\lambda)$ (TM₀ slot)",
    )
    ax2.plot(
        wl_nm,
        neff_te_band,
        color="#2b5c8f",
        lw=1.8,
        ls="-.",
        label=r"$n_{\mathrm{eff,TE}}(\lambda)$ (TE₀ rail)",
    )
    ax2.axhline(
        vm["neff"],
        color="0.35",
        ls=":",
        lw=1.4,
        label=f"EIM @ 1550 nm ({vm['neff']:.4f})",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Vertical slab effective index $n_{\mathrm{eff}}$",
        title="(c) Slot & rail slab dispersion",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="center right", fontsize=7.8)

    fig.tight_layout()
    if out_path is not None:
        FilePath(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, axes


def mmi_analytic_estimates(
    w_mmi=4.90e-6, n_r=2.984168, n_clad=1.444, wavelength=1.55e-6, n_splits=4
):
    """
    Analytical Soldano & Pennings (1995) beat length `L_pi` and first `N`-fold
    symmetric self-imaging length `L_MMI = 3 * L_pi / (4 * N)` for both lateral
    TM (2D EIM `Ex, Ey, Hz`) and lateral TE (3D vertical-slot `E_z`) Goos-Hänchen
    effective widths `W_e`.

    Parameters
    ----------
    w_mmi : float, optional
        Physical MMI width `W_MMI` [m].
    n_r, n_clad : float, optional
        Slab effective index and cladding index (dimensionless).
    wavelength : float, optional
        Design vacuum wavelength [m].
    n_splits : int, optional
        Number of symmetric output images `N` (default 4).

    Returns
    -------
    est : dict
        Dictionary with keys `'w_e_tm'`, `'l_pi_tm'`, `'l_mmi_tm'`, `'w_e_te'`,
        `'l_pi_te'`, `'l_mmi_te'` [m].
    """
    dn = np.sqrt(max(n_r**2 - n_clad**2, 1e-30))
    dw_te = (wavelength / np.pi) / dn
    dw_tm = dw_te * (n_clad / n_r) ** 2
    w_e_tm = float(w_mmi + dw_tm)
    w_e_te = float(w_mmi + dw_te)
    l_pi_tm = float(4.0 * n_r * w_e_tm**2 / (3.0 * wavelength))
    l_pi_te = float(4.0 * n_r * w_e_te**2 / (3.0 * wavelength))
    l_mmi_tm = float(3.0 * l_pi_tm / (4.0 * n_splits))
    l_mmi_te = float(3.0 * l_pi_te / (4.0 * n_splits))
    return {
        "w_e_tm": w_e_tm,
        "l_pi_tm": l_pi_tm,
        "l_mmi_tm": l_mmi_tm,
        "w_e_te": w_e_te,
        "l_pi_te": l_pi_te,
        "l_mmi_te": l_mmi_te,
    }


def mmi_discretization_diagnostics(
    dx,
    dy,
    ny,
    w_mmi=4.90e-6,
    l_mmi=11.15e-6,
    n_r=2.984168,
    n_clad=1.444,
    wavelength=1.55e-6,
    n_splits=4,
):
    """
    Quantify staircase MMI width quantization (`W_stair(Ey)` and `W_stair(Ex)`),
    the resulting Soldano-Pennings beat-length and imaging-length shifts
    (`dL_pi`, `dL_MMI`), and the 2nd-order Yee/FIT numerical-dispersion phase
    error `L * k * (k * dx)^2 / 24` and axial image-plane shift over `l_mmi`.

    Parameters
    ----------
    dx, dy : float
        Axial (`x`) and transverse (`y`) grid cell sizes [m].
    ny : int
        Total number of transverse grid cells across the symmetric `y` domain.
    w_mmi, l_mmi : float, optional
        Nominal physical MMI width and length [m].
    n_r, n_clad : float, optional
        Slab effective index and cladding refractive index (dimensionless).
    wavelength : float, optional
        Design vacuum wavelength [m].
    n_splits : int, optional
        Number of symmetric output images `N` (default 4).

    Returns
    -------
    diag : dict
        Dictionary with keys `'n_ey'`, `'n_ex'`, `'w_stair_ey'`, `'w_stair_ex'`,
        `'dw_ey'`, `'dw_ex'`, `'dl_pi_ey'`, `'dl_mmi_ey'`, `'dl_pi_ex'`,
        `'dl_mmi_ex'` [m], `'k_dx'` [rad], `'phase_err_rad'` [rad], and
        `'image_shift_disp'` [m].
    """
    dx = float(dx)
    dy = float(dy)
    ny_int = int(round((float(w_mmi) + 2.0e-6) / dy))
    y_nodes = np.linspace(-0.5 * ny_int * dy, 0.5 * ny_int * dy, ny_int + 1)
    y_centers = y_nodes[:-1] + 0.5 * dy
    n_ey = int(np.sum(np.abs(y_centers) <= 0.5 * float(w_mmi) + 1e-12))
    n_ex = int(np.sum(np.abs(y_nodes[:-1]) <= 0.5 * float(w_mmi) + 1e-12))
    w_stair_ey = float(n_ey * dy)
    w_stair_ex = float(n_ex * dy)

    est_nom = mmi_analytic_estimates(
        w_mmi=w_mmi, n_r=n_r, n_clad=n_clad, wavelength=wavelength, n_splits=n_splits
    )
    est_ey = mmi_analytic_estimates(
        w_mmi=w_stair_ey,
        n_r=n_r,
        n_clad=n_clad,
        wavelength=wavelength,
        n_splits=n_splits,
    )
    est_ex = mmi_analytic_estimates(
        w_mmi=w_stair_ex,
        n_r=n_r,
        n_clad=n_clad,
        wavelength=wavelength,
        n_splits=n_splits,
    )

    k_core = 2.0 * np.pi * float(n_r) / float(wavelength)
    k_dx = float(k_core * dx)
    phase_err_rad = float(l_mmi * k_core * (k_dx**2) / 24.0)
    image_shift_disp = float(l_mmi * (k_dx**2) / 24.0)

    return {
        "n_ey": n_ey,
        "n_ex": n_ex,
        "w_stair_ey": w_stair_ey,
        "w_stair_ex": w_stair_ex,
        "dw_ey": float(w_stair_ey - w_mmi),
        "dw_ex": float(w_stair_ex - w_mmi),
        "dl_pi_ey": float(est_ey["l_pi_tm"] - est_nom["l_pi_tm"]),
        "dl_mmi_ey": float(est_ey["l_mmi_tm"] - est_nom["l_mmi_tm"]),
        "dl_pi_ex": float(est_ex["l_pi_tm"] - est_nom["l_pi_tm"]),
        "dl_mmi_ex": float(est_ex["l_mmi_tm"] - est_nom["l_mmi_tm"]),
        "k_dx": k_dx,
        "phase_err_rad": phase_err_rad,
        "image_shift_disp": image_shift_disp,
    }


def mmi_eme_length_sweep(
    w_mmi=4.90e-6,
    w_taper=0.80e-6,
    pitch=1.30e-6,
    n_r=2.984168,
    n_clad=1.444,
    wavelength=1.55e-6,
    lengths=None,
    n_modes_max=10,
):
    """
    1D Eigenmode Expansion (EME) over the symmetric lateral modes `m = 0, 2, 4, ...`
    of the MMI region to evaluate `T_inner(L)`, `T_outer(L)`, total guided
    transmission `T_tot(L) = 2 * (T_inner + T_outer)`, and the 2D interference
    intensity carpet `I(x, y)` inside the MMI box.

    Parameters
    ----------
    w_mmi : float, optional
        MMI box width [m].
    w_taper : float, optional
        Input and output taper width `W_2` at the MMI interfaces [m].
    pitch : float, optional
        Center-to-center pitch `W_1 + g_1` of the 4 output waveguides [m].
    n_r, n_clad : float, optional
        Slab effective index and cladding index (dimensionless).
    wavelength : float, optional
        Vacuum wavelength [m].
    lengths : array_like, optional
        MMI lengths `L_MMI` [m] at which to evaluate output coupling.
    n_modes_max : int, optional
        Maximum number of even lateral modes to include.

    Returns
    -------
    eme : dict
        Dictionary containing `'lengths'`, `'t_inner'`, `'t_outer'`, `'t_tot'`,
        `'imbalance'`, `'l_opt_eme'`, `'mode_neffs'`, `'mode_weights'`, `'L_pi_exact'`,
        `'carpet_x'`, `'carpet_y'`, and `'carpet_I'`.
    """
    if lengths is None:
        lengths = np.linspace(8.0e-6, 14.0e-6, 241)
    lengths = np.asarray(lengths, dtype=float)
    k0 = 2.0 * np.pi / float(wavelength)
    y = np.linspace(-0.75 * w_mmi, 0.75 * w_mmi, 2401)
    dy = float(y[1] - y[0])

    def _solve_even_modes(width, n_max):
        modes = []
        for m_even in range(n_max):

            def _disp(neff):
                k = k0 * np.sqrt(max(n_r**2 - neff**2, 0.0))
                g = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
                return (
                    0.5 * k * width
                    - np.arctan2((n_r / n_clad) ** 2 * g, k)
                    - m_even * np.pi
                )

            lo, hi = n_clad + 1e-7, n_r - 1e-7
            if _disp(lo) * _disp(hi) < 0.0:
                neff = float(brentq(_disp, lo, hi))
                k = k0 * np.sqrt(n_r**2 - neff**2)
                g = k0 * np.sqrt(neff**2 - n_clad**2)
                hz = np.where(
                    np.abs(y) <= 0.5 * width,
                    np.cos(k * y),
                    np.cos(0.5 * k * width) * np.exp(-g * (np.abs(y) - 0.5 * width)),
                )
                eps_y = np.where(np.abs(y) <= 0.5 * width, n_r**2, n_clad**2)
                ey = hz / eps_y
                norm = np.sqrt(float(np.sum(ey * hz) * dy))
                modes.append((neff, ey / norm, hz / norm))
        return modes

    # Also solve first odd mode (m=1) for exact L_pi = pi / (beta_0 - beta_1)
    def _disp_odd1(neff):
        k = k0 * np.sqrt(max(n_r**2 - neff**2, 0.0))
        g = k0 * np.sqrt(max(neff**2 - n_clad**2, 0.0))
        return 0.5 * k * w_mmi - np.arctan2((n_r / n_clad) ** 2 * g, k) - 0.5 * np.pi

    neff_m1 = float(brentq(_disp_odd1, n_clad + 1e-7, n_r - 1e-7))

    in_mode = _solve_even_modes(w_taper, 1)[0]
    mmi_modes = _solve_even_modes(w_mmi, n_modes_max)
    neff_m0 = mmi_modes[0][0]
    l_pi_exact = float(wavelength / (2.0 * (neff_m0 - neff_m1)))

    c_m = np.array([float(np.sum(in_mode[1] * m[2]) * dy) for m in mmi_modes])
    neffs = np.array([m[0] for m in mmi_modes])
    hz_stack = np.stack([m[2] for m in mmi_modes], axis=0)  # (M, Ny)

    # Output taper fundamental modes centered at y_inner = 0.5*pitch, y_outer = 1.5*pitch
    neff_w2 = in_mode[0]
    k_w2 = k0 * np.sqrt(n_r**2 - neff_w2**2)
    g_w2 = k0 * np.sqrt(neff_w2**2 - n_clad**2)

    def _shifted_taper_ey(y0):
        ym = y - y0
        hz = np.where(
            np.abs(ym) <= 0.5 * w_taper,
            np.cos(k_w2 * ym),
            np.cos(0.5 * k_w2 * w_taper) * np.exp(-g_w2 * (np.abs(ym) - 0.5 * w_taper)),
        )
        eps_y = np.where(np.abs(ym) <= 0.5 * w_taper, n_r**2, n_clad**2)
        ey = hz / eps_y
        norm = np.sqrt(float(np.sum(ey * hz) * dy))
        return ey / norm

    ey_inner = _shifted_taper_ey(0.5 * pitch)
    ey_outer = _shifted_taper_ey(1.5 * pitch)

    # Modal overlap integrals with each MMI mode m
    ov_inner = np.array([float(np.sum(ey_inner * m[2]) * dy) for m in mmi_modes])
    ov_outer = np.array([float(np.sum(ey_outer * m[2]) * dy) for m in mmi_modes])

    phases = np.exp(-1j * k0 * neffs[:, None] * lengths[None, :])  # (M, NL)
    amp_in = (c_m * ov_inner) @ phases
    amp_out = (c_m * ov_outer) @ phases
    t_inner = np.abs(amp_in) ** 2
    t_outer = np.abs(amp_out) ** 2
    t_tot = 2.0 * (t_inner + t_outer)
    imbalance = np.abs(t_inner - t_outer)
    l_opt_eme = float(lengths[int(np.argmax(t_tot))])

    # 2D interference carpet I(x, y) over x in [0, 14 um]
    carpet_x = np.linspace(0.0, float(np.max(lengths)), 180)
    y_sub = y[::8]
    hz_sub = hz_stack[:, ::8]
    eps_sub = np.where(np.abs(y_sub) <= 0.5 * w_mmi, n_r**2, n_clad**2)
    ph_x = np.exp(-1j * k0 * neffs[:, None] * carpet_x[None, :])
    hz_xy = (c_m[:, None] * ph_x).T @ hz_sub  # (Nx, Ny_sub)
    ey_xy = hz_xy / eps_sub[None, :]
    carpet_i = np.abs(ey_xy) ** 2
    carpet_i /= max(float(np.max(carpet_i)), 1e-30)

    return {
        "lengths": lengths,
        "t_inner": t_inner,
        "t_outer": t_outer,
        "t_tot": t_tot,
        "imbalance": imbalance,
        "l_opt_eme": l_opt_eme,
        "mode_neffs": neffs,
        "mode_weights": np.abs(c_m) ** 2,
        "L_pi_exact": l_pi_exact,
        "carpet_x": carpet_x,
        "carpet_y": y_sub,
        "carpet_I": carpet_i,
    }
