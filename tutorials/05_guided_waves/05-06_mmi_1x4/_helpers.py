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
        Equivalent 3-layer core index (dimensionless) to pass to `PhotonicCircuit`.
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

    from wakis import figstyle

    figstyle.apply_style()
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=12.2, height=4.0)
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
    ax0.legend(loc="upper right", fontsize=8.5)

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
    ax1.legend(loc="upper right", fontsize=8.5)

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
    ax2.legend(loc="center right", fontsize=8.5)

    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
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


# ---------------------------------------------------------------------------
# Graded non-uniform Cartesian mesh & schematic/overlay helpers (Tutorial 5.6)
# ---------------------------------------------------------------------------

from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, Rectangle  # noqa: E402
from scipy.constants import c as c_light  # noqa: E402

from wakis import figstyle, schematic  # noqa: E402
from wakis.templates import Simulation, meshing  # noqa: E402
from wakis.templates.photonics import PhotonicCircuit, n_to_eps  # noqa: E402


def build_mmi_1x4_graded_axes(
    domain,
    l1: float = 2.00e-6,
    l_mmi: float = 11.15e-6,
    w1: float = 0.40e-6,
    g2: float = 0.50e-6,
    w_mmi: float = 4.90e-6,
    n_pml: int = 10,
    cpw: float = 20.0,
    dx_junc: float | None = None,
    dx_mmi: float | None = None,
    dx_lead: float | None = None,
    dy_in: float | None = None,
    dy_mmi: float | None = None,
    dy_wall: float | None = None,
    dy_clad: float | None = None,
    grade_ratio: float = 1.22,
):
    """
    Construct mirror-symmetric graded non-uniform, non-square Cartesian node
    arrays ``(x, y, z)`` for the 1x4 Si-GaN-Si vertical-slot MMI splitter:

    - Exact transverse grid nodes at ``y = +-0.5 * g2`` (central output taper gap)
      and ``y = +-0.5 * w_mmi`` (MMI sidewalls), eliminating the ``-7.3 nm``
      uniform-grid sidewall rounding error,
    - Fine transverse spacing ``dy_in`` across the input waveguide and central
      output gap ``|y| <= 0.5 * g2`` and ``dy_wall`` across the outer taper
      shoulder and MMI sidewall ``|y| in [0.5 * w_mmi - 0.20 um, 0.5 * w_mmi]``,
    - Standard ``>= 20 cpw`` spacing ``dy_mmi`` across the interior MMI core and
      smoothly graded coarse cells ``dy_clad`` in the outer ``SiO2`` cladding
      ``|y| > 0.5 * w_mmi``,
    - Fine axial spacing ``dx_junc`` across the input-taper-to-MMI entrance
      (``x ~ l1``) and 4-port MMI exit junction (``x ~ l1 + l_mmi``), paired
      with ``dx_mmi`` inside the multimode cavity and stretched non-square axial
      cells ``dx_lead > dy_in`` along the smooth input/output leads and tapers.
    """
    if len(domain) == 4:
        xmin, xmax, ymin, ymax = domain
        zmin_2d, zmax_2d = -0.5e-6, 0.5e-6
    else:
        xmin, xmax, ymin, ymax, zmin_2d, zmax_2d = domain

    scale = 20.0 / max(float(cpw), 5.0)
    x_in_junc = float(l1)
    x_out_junc = float(l1) + float(l_mmi)
    y_gap = 0.5 * max(float(g2), float(w1))
    y_wall = 0.5 * float(w_mmi)
    y_shoulder = max(y_gap + 0.10e-6, y_wall - 0.20e-6)

    dx_j = float(24.0e-9 * scale if dx_junc is None else dx_junc)
    dx_m = float(25.5e-9 * scale if dx_mmi is None else dx_mmi)
    dx_ld = float(38.0e-9 * scale if dx_lead is None else dx_lead)

    dy_i = float(22.73e-9 * scale if dy_in is None else dy_in)
    dy_c = float(25.0e-9 * scale if dy_mmi is None else dy_mmi)
    dy_w = float(22.22e-9 * scale if dy_wall is None else dy_wall)
    dy_cl = float(75.0e-9 * scale if dy_clad is None else dy_clad)

    x_int = meshing.graded_axis(
        [
            xmin,
            x_in_junc - 0.20e-6,
            x_in_junc + 0.20e-6,
            x_out_junc - 0.20e-6,
            x_out_junc + 0.25e-6,
            xmax,
        ],
        [dx_ld, dx_j, dx_m, dx_j, dx_ld],
        grading=grade_ratio,
    )
    y_pos = meshing.graded_axis(
        [0.0, y_gap, y_shoulder, y_wall, ymax],
        [dy_i, dy_c, dy_w, dy_cl],
        grading=grade_ratio,
    )
    y_int = np.concatenate([-y_pos[:0:-1], y_pos])

    n_pml = int(n_pml)

    def _pad_pml(nodes: np.ndarray) -> np.ndarray:
        lo_pml = nodes[0] - np.arange(n_pml, 0, -1) * (nodes[1] - nodes[0])
        hi_pml = nodes[-1] + np.arange(1, n_pml + 1) * (nodes[-1] - nodes[-2])
        return np.concatenate([lo_pml, nodes, hi_pml])

    return (
        _pad_pml(x_int),
        _pad_pml(y_int),
        np.array([zmin_2d, zmax_2d], dtype=float),
    )


def measure_mmi_1x4_mesh(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    w1: float = 0.40e-6,
    g2: float = 0.50e-6,
    w_mmi: float = 4.90e-6,
    has_ghost: bool = False,
) -> dict:
    """
    Measure cell counts, min/max steps, feature cell counts across ``w1`` and
    ``g2``, and sidewall node alignment error on ``(x_nodes, y_nodes)``.
    """
    xv = np.asarray(x_nodes[:-1] if has_ghost else x_nodes, dtype=float)
    yv = np.asarray(y_nodes[:-1] if has_ghost else y_nodes, dtype=float)
    dx = np.diff(xv)
    dy = np.diff(yv)

    def _count_span(nodes: np.ndarray, lo: float, hi: float) -> float:
        ov = np.maximum(0.0, np.minimum(nodes[1:], hi) - np.maximum(nodes[:-1], lo))
        return float(np.sum(ov / np.maximum(np.diff(nodes), 1e-30)))

    cells_w1 = _count_span(yv, -0.5 * float(w1), +0.5 * float(w1))
    cells_g2 = _count_span(yv, -0.5 * float(g2), +0.5 * float(g2))
    sidewall_err_nm = float(np.min(np.abs(yv - 0.5 * float(w_mmi))) * 1e9)
    return {
        "nx": int(len(dx)),
        "ny": int(len(dy)),
        "n_cells": int(len(dx) * len(dy)),
        "dx_min_nm": float(np.min(dx) * 1e9),
        "dx_max_nm": float(np.max(dx) * 1e9),
        "dy_min_nm": float(np.min(dy) * 1e9),
        "dy_max_nm": float(np.max(dy) * 1e9),
        "cells_across_w1": cells_w1,
        "cells_across_g2": cells_g2,
        "sidewall_err_nm": sidewall_err_nm,
        "aspect_max": float(np.max(dx) / np.min(dy)),
    }


class NonUniformMMI1x4Circuit(PhotonicCircuit):
    """
    2D EIM ``PhotonicCircuit`` subclass for the 1x4 MMI power splitter using
    ``build_mmi_1x4_graded_axes`` (fine transverse cells across the waveguide
    core, taper gaps, and exact MMI sidewalls; graded coarse cells in the outer
    ``SiO2`` cladding; fine axial cells at the MMI entrance/exit throats and
    stretched non-square axial cells along the input/output leads and tapers).
    """

    def __init__(
        self,
        *args,
        l1: float = 2.00e-6,
        l_mmi: float = 11.15e-6,
        w1: float = 0.40e-6,
        g2: float = 0.50e-6,
        w_mmi: float = 4.90e-6,
        grade_ratio: float = 1.22,
        **kwargs,
    ):
        self.l1 = float(l1)
        self.l_mmi = float(l_mmi)
        self.w1 = float(w1)
        self.g2 = float(g2)
        self.w_mmi = float(w_mmi)
        self.grade_ratio = float(grade_ratio)
        kwargs["mesh_overrides"] = ()
        super().__init__(*args, **kwargs)

    def build(self):
        structs, domain = self._build_2d_structures_and_domain()
        boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=[],
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        n_pml = int(self.n_pml or 10)
        self.sim.n_pml = n_pml
        axes_xyz = build_mmi_1x4_graded_axes(
            domain,
            l1=self.l1,
            l_mmi=self.l_mmi,
            w1=self.w1,
            g2=self.g2,
            w_mmi=self.w_mmi,
            n_pml=n_pml,
            cpw=self.cells_per_wavelength,
            grade_ratio=self.grade_ratio,
        )
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


class _PolygonSchematicProxy:
    """
    Lightweight polygon structure proxy exposing ``.vertices``, ``.material``,
    and ``.name`` (without ``.bounds`` or ``.radius``) so ``wakis.schematic``
    classifies a 2D ``PolySlab`` as ``kind='polygon'`` rather than a bounding-box
    rectangle.
    """

    def __init__(self, slab, material_eps: float, name: str):
        self.vertices = np.asarray(slab.vertices, dtype=float)
        self.material = float(material_eps)
        self.name = str(name)


class _SchematicSolverProxy:
    """
    Lightweight proxy around ``SolverFIT3D`` that hides ``ieps`` from
    ``wakis.schematic.extract_setup_info`` so explicit ``structures`` are used.
    """

    def __init__(self, solver, eps_clad: float):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["pml", "pml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["pml", "pml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_mmi_1x4_sources_and_monitors(circuit, include_flux_box: bool = True):
    """Build ``(sources, monitors)`` dicts for ``wakis.schematic`` from ``circuit``."""
    xmin, xmax = float(circuit.grid.xmin), float(circuit.grid.xmax)
    ymin, ymax = float(circuit.grid.ymin), float(circuit.grid.ymax)
    dx_bg = float(np.median(np.diff(circuit.grid.x[:-1])))
    dy_bg = float(np.median(np.diff(circuit.grid.y[:-1])))
    n_pml = int(getattr(circuit.solver, "n_pml", 10))
    pulse_obj = circuit.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)

    sources_s = []
    monitors_s = []
    out_ports = [pt for pt in circuit.ports if pt.name.startswith("out") and "_m1" not in pt.name]

    for pt in circuit.ports:
        if pt.name == "in":
            cx = float(pt.center[0])
            hy = 0.5 * float(pt.size[1])
            x_src = cx - 0.15e-6
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "short_word": "ModeSrc",
                    "label": "Port 'in' ModeSource (TM₀ slot mode, +x)",
                    "desc": f"Port 'in' ModeSource at x = {x_src * 1e6:.2f} µm (TE₀/EIM, +x)",
                    "h_bounds": (x_src, x_src),
                    "v_bounds": (-hy, +hy),
                    "e_vec": (0.0, 1.0, 0.0),
                    "k_vec": (1.0, 0.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(4.0 / (np.pi * fwidth)),
                        "t_end": float(pulse_obj.end_time),
                        "src_time": pulse_obj,
                    },
                }
            )
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "in mon",
                    "label": "Port 'in' ModeMonitor (S₁₁)",
                    "desc": f"Port 'in' ModeMonitor at x = {cx * 1e6:.2f} µm (S₁₁)",
                    "h_bounds": (cx, cx),
                    "v_bounds": (-hy, +hy),
                }
            )

    if out_ports:
        cx_out = float(out_ports[0].center[0])
        y_lo = float(min(pt.center[1] - 0.5 * pt.size[1] for pt in out_ports))
        y_hi = float(max(pt.center[1] + 0.5 * pt.size[1] for pt in out_ports))
        monitors_s.append(
            {
                "kind": "plane",
                "type_name": "ModeMonitor",
                "short_word": "out1–4",
                "label": "Ports 'out1'–'out4' ModeMonitors (S₂₁..S₅₁)",
                "desc": f"Ports 'out1'–'out4' ModeMonitors at x = {cx_out * 1e6:.2f} µm (y = ±0.65, ±1.95 µm)",
                "h_bounds": (cx_out, cx_out),
                "v_bounds": (y_lo, y_hi),
            }
        )

    if include_flux_box:
        pad_x = (n_pml + 2) * dx_bg
        pad_y = (n_pml + 2) * dy_bg
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxBox",
                "short_word": "FluxBox",
                "label": "2D Poynting FluxBox",
                "desc": "Closed 2D Poynting FluxBox (4 faces inside CPML)",
                "h_bounds": (xmin + pad_x, xmax - pad_x),
                "v_bounds": (ymin + pad_y, ymax - pad_y),
            }
        )
    return sources_s, monitors_s


def save_mmi_1x4_setup_figures(
    circuit,
    structures,
    primary_data: dict,
    out_dir: str | FilePath,
    *,
    w1: float = 0.40e-6,
    w2: float = 0.80e-6,
    l1: float = 2.00e-6,
    l2: float = 5.00e-6,
    l_mmi: float = 11.15e-6,
    w_mmi: float = 4.90e-6,
    pitch: float = 1.30e-6,
) -> dict:
    """
    Generate ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``,
    and ``00_setup_table.md`` for Tutorial 5.6 using
    ``wakis.schematic.save_setup_figures``, enhanced with the launched input
    mode profile ``|E_y(y)|`` on the port plane and the 34-vertex polygon on the
    mesh and 1x zoom inset.
    """
    out_path = FilePath(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eps_clad = n_to_eps(circuit.n_clad)
    eps_slab = n_to_eps(circuit.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(circuit.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"Si–GaN–Si slot EIM (n_eff = {circuit.n_eff_slab:.2f})",
    )
    mat_names = {
        "background": f"SiO₂ cladding (n = {circuit.n_clad:.3f}, ε_r = {eps_clad:.2f})",
        "region_0": f"Si–GaN–Si slot EIM (n_eff = {circuit.n_eff_slab:.2f}, ε_eff = {eps_slab:.2f})",
    }
    sources_s, monitors_s = _build_mmi_1x4_sources_and_monitors(circuit, include_flux_box=True)
    variations = [
        {
            "name": "kottke_coarse..fine (tasks 0–5)",
            "param": "cpw ∈ {10, 14, 20}, averaging ∈ {kottke, staircase}",
            "note": "Uniform resolution ladder at (W_MMI, L_MMI) = (4.90, 11.15) µm",
        },
        {
            "name": "kottke_reference (task 6)",
            "param": "cpw = 40 (1705 × 571 × 1 = 973,555 cells)",
            "note": "Independent 2× fine Kottke reference",
        },
        {
            "name": "sweep_W* (tasks 7–36)",
            "param": "W_MMI ∈ [4.80, 5.20] µm × L_MMI ∈ [11.10, 11.35] µm",
            "note": "30-point 5×6 design sweep at 20 cpw (min EL at 5.00×11.20, min imb at 5.00×11.30 µm)",
        },
        {
            "name": "kottke_nonuniform (task 37)",
            "param": "Graded non-square Cartesian mesh (731 × 256 × 1 = 187,136 cells)",
            "note": "Exact sidewall nodes (y = ±2.45 µm), dy_min = 20.0 nm in gaps, dx_max = 38.0 nm in leads",
        },
    ]

    res = schematic.save_setup_figures(
        solver_proxy,
        out_path,
        prefix="00",
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=circuit.wavelength_ref,
        min_feature=w1,
        title_geometry=(
            "1×4 Si–GaN–Si Vertical-Slot MMI Power Splitter (2D EIM Setup Schematic)"
        ),
        title_mesh=(
            "Discretization (Graded Non-Uniform Mesh & 1× Zoom) and Broadband Excitation"
        ),
        variations=variations,
    )

    # Enhance 00a_setup_geometry.png with top/bottom CPML shading and launched mode profile |Ey(y)|
    fig_a, ax_a = schematic.plot_schematic(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=circuit.wavelength_ref,
        title="1×4 Si–GaN–Si Vertical-Slot MMI Power Splitter (2D EIM Setup Schematic)",
        width_in=7.6,
        height_in=4.6,
    )
    x0_um = float(circuit.grid.xmin * 1e6)
    x1_um = float(circuit.grid.xmax * 1e6)
    y0_um = float(circuit.grid.ymin * 1e6)
    y1_um = float(circuit.grid.ymax * 1e6)
    n_pml = int(getattr(circuit.solver, "n_pml", 10))
    th_y_low = float((circuit.grid.y[n_pml] - circuit.grid.y[0]) * 1e6)
    th_y_high = float((circuit.grid.y[-2] - circuit.grid.y[-2 - n_pml]) * 1e6)
    for yb_start, th_y in ((y0_um, th_y_low), (y1_um - th_y_high, th_y_high)):
        ax_a.add_patch(
            Rectangle(
                (x0_um, yb_start),
                x1_um - x0_um,
                th_y,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="///",
                alpha=0.36,
                lw=0.8,
                zorder=2,
            )
        )
    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        y_m = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        ey_m = np.abs(np.asarray(primary_data["port_ey_in"]))
        ey_m = ey_m / max(float(np.max(ey_m)), 1e-30)
        x_src_um = float(sources_s[0]["h_bounds"][0] * 1e6)
        x_prof = x_src_um + 1.15 * ey_m
        ax_a.fill_betweenx(
            y_m, x_src_um, x_prof, color="#D55E00", alpha=0.28, zorder=7.5
        )
        ax_a.plot(
            x_prof,
            y_m,
            color="#D55E00",
            lw=1.8,
            zorder=8,
            label="Launched mode |Ey(y)|",
        )
    ax_a.text(
        float((l1 + 0.5 * l_mmi) * 1e6),
        0.0,
        f"MMI cavity: L_MMI = {l_mmi * 1e6:.2f} µm, W_MMI = {w_mmi * 1e6:.2f} µm",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=figstyle.INK,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="#FFFFFFE8",
            edgecolor="#B0B8C2",
            lw=0.6,
        ),
        zorder=9,
    )
    for k_port, yc_m in enumerate((-1.5 * pitch, -0.5 * pitch, 0.5 * pitch, 1.5 * pitch), start=1):
        ax_a.text(
            float((l1 + l_mmi + l2 + 0.95e-6) * 1e6),
            float(yc_m * 1e6),
            f"out{k_port}",
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#006B4E",
            zorder=9,
        )
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with the 34-vertex polygon on ax_mesh and zoom inset
    x_exit = float(l1 + l_mmi)
    zoom_bounds = (x_exit - 0.35e-6, x_exit + 0.45e-6, -0.45e-6, +0.45e-6)
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_proxy,
        plane="xy",
        structures=[poly_proxy],
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        mesh_every=8,
        zoom_bounds=zoom_bounds,
        wavelength=circuit.wavelength_ref,
        min_feature=w1,
        title="Discretization (Graded Non-Uniform Mesh & 1× Zoom) and Broadband Excitation",
        width_in=7.6,
        height_in=5.6,
    )
    ax_mesh = axes_b[0]
    verts_um = np.asarray(structures[0].vertices, dtype=float) * 1e6
    ax_mesh.add_patch(
        MplPolygon(
            verts_um,
            closed=True,
            facecolor="#DCEAF7",
            edgecolor="#1E293B",
            lw=1.1,
            alpha=0.80,
            zorder=2,
        )
    )
    for child_ax in fig_b.axes:
        if getattr(child_ax, "_is_inset", False):
            child_ax.add_patch(
                MplPolygon(
                    verts_um,
                    closed=True,
                    facecolor="#DCEAF7",
                    edgecolor="#1E293B",
                    lw=1.0,
                    alpha=0.80,
                    zorder=1.5,
                )
            )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res


def plot_mesh_upgrade_comparison_mmi_1x4(
    circuit_uniform,
    circuit_nonuniform,
    structures,
    d_uniform: dict,
    d_nonuniform: dict,
    d_reference: dict,
    meta_uniform: dict,
    meta_nonuniform: dict,
    meta_reference: dict,
    out_path: str | FilePath,
    xlim_um: tuple[float, float] = (12.6, 15.0),
    ylim_um: tuple[float, float] = (-2.65, 2.65),
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels):
    (a) Baseline uniform 20 cpw Kottke mesh zoomed on the 1x4 MMI exit junction,
    (b) Upgraded graded non-uniform, non-square Cartesian Kottke mesh on the same
        window showing fine cells across the gaps/sidewall and coarse cladding,
    (c) Broadband total transmission and excess loss comparison against the
        independent 40 cpw reference, with cell counts and wall time annotated.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.3)
    verts_um = np.asarray(structures[0].vertices, dtype=float) * 1e6

    for ax, circuit_obj, meta, tag in (
        (ax0, circuit_uniform, meta_uniform, "(a) Baseline uniform 20 cpw Kottke"),
        (ax1, circuit_nonuniform, meta_nonuniform, "(b) Graded non-uniform Kottke"),
    ):
        g = circuit_obj.grid
        iz = circuit_obj.solver.Nz // 2
        ieps_y = np.asarray(circuit_obj.solver.ieps[:, :, iz, "y"], dtype=float)
        with np.errstate(divide="ignore"):
            eps_y = np.where(ieps_y > 0.0, 1.0 / ieps_y, 1.0)
        xu = np.asarray(g.x, dtype=float) * 1e6
        yu = np.asarray(g.y, dtype=float) * 1e6
        nx_c = len(xu) - 1
        ny_c = len(yu) - 1
        im = ax.pcolormesh(
            xu,
            yu,
            eps_y[:nx_c, :ny_c].T,
            cmap="Blues",
            vmin=1.8,
            vmax=9.5,
            shading="flat",
            alpha=0.88,
        )
        ax.plot(
            np.append(verts_um[:, 0], verts_um[0, 0]),
            np.append(verts_um[:, 1], verts_um[0, 1]),
            color="#D55E00",
            lw=1.35,
            label="Exact 34-vertex polygon",
        )
        xs = xu[(xu >= xlim_um[0]) & (xu <= xlim_um[1])]
        ys = yu[(yu >= ylim_um[0]) & (yu <= ylim_um[1])]
        segs = [[(x, ylim_um[0]), (x, ylim_um[1])] for x in xs] + [
            [(xlim_um[0], y), (xlim_um[1], y)] for y in ys
        ]
        ax.add_collection(
            LineCollection(segs, colors="0.15", linewidths=0.42, alpha=0.62)
        )
        m_stats = measure_mmi_1x4_mesh(g.x, g.y, has_ghost=True)
        ax.set(
            xlim=xlim_um,
            ylim=ylim_um,
            xlabel="x [µm]",
            ylabel="y [µm]",
            title=(
                f"{tag}\n"
                f"{meta['grid_shape'][0]}×{meta['grid_shape'][1]} = {meta['n_cells']:,} cells "
                f"({meta['elapsed_s']:.1f} s)"
            ),
        )
        ax.text(
            0.03,
            0.04,
            (
                f"Δx ∈ [{m_stats['dx_min_nm']:.1f}, {m_stats['dx_max_nm']:.1f}] nm\n"
                f"Δy ∈ [{m_stats['dy_min_nm']:.1f}, {m_stats['dy_max_nm']:.1f}] nm\n"
                f"sidewall node err = {m_stats['sidewall_err_nm']:.1f} nm"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92),
        )
        if ax is ax1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(r"$\varepsilon_{yy} = (\mathbf{\varepsilon}^{-1})_{yy}^{-1}$")

    def _tot_and_in_out(d):
        w_nm = np.asarray(d["wavelengths"], dtype=float) * 1e9
        o = np.argsort(w_nm)
        wl_s = w_nm[o]
        m = (wl_s >= 1505.0) & (wl_s <= 1595.0)
        t_out = np.abs(d["S21"][o][m]) ** 2
        t_in = np.abs(d["S31"][o][m]) ** 2
        t_tot = (
            t_out
            + t_in
            + np.abs(d["S41"][o][m]) ** 2
            + np.abs(d["S51"][o][m]) ** 2
        )
        return wl_s[m], t_tot, t_in, t_out

    wl_u, t_u, _, _ = _tot_and_in_out(d_uniform)
    wl_nu, t_nu, _, _ = _tot_and_in_out(d_nonuniform)
    wl_r, t_r, _, _ = _tot_and_in_out(d_reference)
    i0 = int(np.argmin(np.abs(wl_r - 1550.0)))
    rms_u = float(np.sqrt(np.mean((t_u - t_r) ** 2)))
    rms_nu = float(np.sqrt(np.mean((t_nu - t_r) ** 2)))
    cell_red = 100.0 * (1.0 - meta_nonuniform["n_cells"] / meta_uniform["n_cells"])
    time_red = 100.0 * (1.0 - meta_nonuniform["elapsed_s"] / meta_uniform["elapsed_s"])

    ax2.plot(
        wl_r,
        t_r,
        ":",
        color="black",
        lw=2.1,
        label=f"40 cpw ref ({meta_reference['n_cells']:,} cells, T={t_r[i0]:.4f})",
    )
    ax2.plot(
        wl_u,
        t_u,
        "--",
        color="#0072B2",
        lw=1.9,
        label=f"Uniform 20 cpw ({meta_uniform['n_cells']:,} cells, RMS={rms_u:.4f})",
    )
    ax2.plot(
        wl_nu,
        t_nu,
        "-",
        color="#009E73",
        lw=2.0,
        label=f"Graded non-uniform ({meta_nonuniform['n_cells']:,} cells, RMS={rms_nu:.4f})",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Total guided transmission $T_{\mathrm{tot}} = \sum_{i=1}^4 T_i$",
        ylim=(0.84, 0.93),
        title=(
            f"(c) Equal-accuracy validation (−{cell_red:.1f}% cells, −{time_red:.1f}% time)"
        ),
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="lower center", fontsize=8.5)
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, (ax0, ax1, ax2)


def render_mmi_1x4_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field_2d: np.ndarray,
    circuit,
    structures,
    *,
    kind: str = "intensity_log",
    title: str = "",
    out_path: str | FilePath | None = None,
):
    """
    Render a 2D steady-state field map (``intensity_log`` or ``real``) with
    ``wakis.schematic.overlay_geometry`` (material outlines + labels, CPML
    boundary bands + labels, and port monitors) and ``wakis.figstyle``.
    """
    figstyle.apply_style()
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=9.6, height=4.4)
    x_um = np.asarray(x, dtype=float) * 1e6
    y_um = np.asarray(y, dtype=float) * 1e6

    if kind == "intensity_log":
        mag2 = np.abs(np.asarray(field_2d))
        peak = max(float(np.max(mag2)), 1e-30)
        db = 10.0 * np.log10(np.maximum(mag2 / peak, 1e-4))
        im = ax.pcolormesh(
            x_um, y_um, db.T, cmap="inferno", vmin=-35.0, vmax=0.0, shading="auto"
        )
        cb_label = r"Normalized intensity $10\log_{10}(|E|^2 / |E|_{\max}^2)$ [dB]"
        outline_col = "white"
    else:
        re_val = np.real(np.asarray(field_2d, dtype=complex))
        vmax = max(float(np.percentile(np.abs(re_val), 99.2)), 1e-30)
        im = ax.pcolormesh(
            x_um, y_um, re_val.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )
        cb_label = r"$\mathrm{Re}(E_y)$ [arb. units]"
        outline_col = "#1E293B"

    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.025)
    cb.set_label(cb_label)

    eps_clad = n_to_eps(circuit.n_clad)
    eps_slab = n_to_eps(circuit.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(circuit.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"Si–GaN–Si slot EIM (n={circuit.n_eff_slab:.2f})",
    )
    sources_s, monitors_s = _build_mmi_1x4_sources_and_monitors(circuit, include_flux_box=False)
    schematic.overlay_geometry(
        ax,
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=[poly_proxy],
        material_names={
            "background": f"SiO₂ (n={circuit.n_clad:.3f})",
            "region_0": f"Si–GaN–Si slot (n_eff={circuit.n_eff_slab:.2f})",
        },
        sources=sources_s,
        monitors=monitors_s,
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
        fill_materials=False,
        outline_color=outline_col,
        label_materials=True,
    )
    # Also shade top and bottom CPML strips
    n_pml = int(getattr(circuit.solver, "n_pml", 10))
    y0_um, y1_um = float(circuit.grid.ymin * 1e6), float(circuit.grid.ymax * 1e6)
    x0_um, x1_um = float(circuit.grid.xmin * 1e6), float(circuit.grid.xmax * 1e6)
    th_y = float((circuit.grid.y[n_pml] - circuit.grid.y[0]) * 1e6)
    for yb_s in (y0_um, y1_um - th_y):
        ax.add_patch(
            Rectangle(
                (x0_um, yb_s),
                x1_um - x0_um,
                th_y,
                facecolor="#94A3B8",
                edgecolor="#475569",
                hatch="///",
                alpha=0.18,
                lw=0.6,
                zorder=5,
            )
        )
    ax.set(
        xlim=(float(x_um[0]), float(x_um[-1])),
        ylim=(float(y_um[0]), float(y_um[-1])),
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    ax.set_aspect("equal", adjustable="box")
    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, ax


def render_mmi_1x4_movie_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    circuit,
    structures,
    out_path: str | FilePath,
    *,
    fps: int = 12,
):
    """
    Render ``09_pulse_propagation.gif`` with ``wakis.schematic.overlay_geometry``
    (34-vertex MMI polygon outline + material/CPML/source/monitor labels) on
    every frame, keeping total GIF size well below 5 MB.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path = FilePath(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_um = np.asarray(x, dtype=float) * 1e6
    y_um = np.asarray(y, dtype=float) * 1e6
    f_arr = np.asarray(frames, dtype=np.float32)
    t_fs = np.asarray(times, dtype=float) * 1e15
    vmax = max(float(np.percentile(np.abs(f_arr), 99.5)), 1e-12)

    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=110)
    im = ax.pcolormesh(
        x_um,
        y_um,
        f_arr[0].T,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.025)
    cb.set_label(r"$E_y(x, y, t)$ [arb. units]")

    eps_clad = n_to_eps(circuit.n_clad)
    eps_slab = n_to_eps(circuit.n_eff_slab)
    solver_proxy = _SchematicSolverProxy(circuit.solver, eps_clad=eps_clad)
    poly_proxy = _PolygonSchematicProxy(
        structures[0],
        material_eps=eps_slab,
        name=f"Si–GaN–Si slot (n_eff={circuit.n_eff_slab:.2f})",
    )
    sources_s, monitors_s = _build_mmi_1x4_sources_and_monitors(circuit, include_flux_box=False)
    schematic.overlay_geometry(
        ax,
        solver_proxy,
        plane="xy",
        unit="µm",
        structures=[poly_proxy],
        material_names={
            "background": f"SiO₂ (n={circuit.n_clad:.3f})",
            "region_0": f"Si–GaN–Si slot (n_eff={circuit.n_eff_slab:.2f})",
        },
        sources=sources_s,
        monitors=monitors_s,
        show_boundaries=True,
        show_materials=True,
        show_sources=True,
        show_monitors=True,
        fill_materials=False,
        outline_color="#1E293B",
        label_materials=True,
    )
    ax.set(
        xlim=(float(x_um[0]), float(x_um[-1])),
        ylim=(float(y_um[0]), float(y_um[-1])),
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ax.set_aspect("equal", adjustable="box")
    title_obj = ax.set_title(
        f"1×4 Si–GaN–Si MMI pulse propagation: t = {t_fs[0]:.1f} fs",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()

    def _update(frame_idx: int):
        im.set_array(f_arr[frame_idx].T.ravel())
        title_obj.set_text(
            f"1×4 Si–GaN–Si MMI pulse propagation: t = {t_fs[frame_idx]:.1f} fs"
        )
        return (im, title_obj)

    anim = FuncAnimation(fig, _update, frames=len(f_arr), interval=1000 // max(fps, 1), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path

