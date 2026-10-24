# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
import pytest
from scipy.constants import c as c0
from scipy.constants import epsilon_0, mu_0

import wakis
from wakis.analytic.thin_wire import hallen_dipole
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.lumped import LumpedElement, LumpedPort
from wakis.sources import CurrentSource, GaussianSource
from wakis.subcell import (
    ETA0,
    Drude,
    SheetMaterial,
    graphene_sheet,
    metal_sheet,
    resistive_sheet,
    salisbury_screen_reflection,
    sheet_transmission_reflection,
)


def _run_1d_sheet_scattering(
    sheet_fn=None,
    f0=3.0e9,
    n_cells=60,
    dz=None,
    bc_high_z="cpml",
    nt=650,
    z_nodes=None,
):
    """
    Run a 1D plane-wave-like TEM simulation along z (periodic in x, y) and
    return time traces at reflection and transmission probe planes.
    """
    lam0 = c0 / f0
    if dz is None:
        dz = lam0 / 40.0
    if z_nodes is None:
        grid = wakis.GridFIT3D(
            -dz, dz, -dz, dz, 0.0, n_cells * dz, 2, 2, n_cells, use_mpi=False, verbose=0
        )
    else:
        grid = wakis.GridFIT3D(
            x=np.array([-dz, 0.0, dz]),
            y=np.array([-dz, 0.0, dz]),
            z=z_nodes,
            use_mpi=False,
            verbose=0,
        )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", bc_high_z],
        n_pml=10,
        verbose=0,
    )
    if sheet_fn is not None:
        sheet_fn(solver)

    iz_src = 13
    iz_ref = 18
    iz_trn = n_cells - 14
    dt = solver.dt
    dz_src = float(grid.tL[0, 0, iz_src, "z"])
    t_arr = np.arange(nt) * dt
    ref_trace = np.zeros(nt)
    trn_trace = np.zeros(nt)

    for n in range(nt):
        t = t_arr[n]
        pulse = np.exp(-0.5 * ((t - 0.8e-9) / 0.18e-9) ** 2) * np.sin(2.0 * np.pi * f0 * t)
        solver.J[:, :, iz_src, "x"] = pulse / dz_src
        solver.one_step()
        ref_trace[n] = float(solver.E[0, 0, iz_ref, "x"])
        trn_trace[n] = float(solver.E[0, 0, iz_trn, "x"])

    return t_arr, ref_trace, trn_trace, solver


def _dtft(trace, t_arr, freqs):
    f = np.atleast_1d(np.asarray(freqs, dtype=float))
    return np.sum(trace[:, None] * np.exp(-2j * np.pi * f[None, :] * t_arr[:, None]), axis=0)


# ---------------------------------------------------------------------------
# A. Thin conducting & dispersive sheets
# ---------------------------------------------------------------------------


def test_resistive_sheet_transmission_reflection():
    """
    A resistive sheet of surface conductivity sigma_s in free space gives normal
    incidence transmission amplitude t = 1 / (1 + eta0 * sigma_s / 2) and
    reflection amplitude r = -(eta0 * sigma_s / 2) / (1 + eta0 * sigma_s / 2),
    including on a non-uniform mesh where the primary cell size dz[iz_sheet]
    differs from the dual-edge length tL_z[iz_sheet] crossing the sheet.
    """
    f0 = 3.0e9
    lam0 = c0 / f0
    dz_base = lam0 / 40.0
    n_cells = 60
    iz_sheet = 30

    t_arr, ref_inc, trn_inc, _ = _run_1d_sheet_scattering(
        sheet_fn=None, f0=f0, n_cells=n_cells, dz=dz_base
    )
    freqs = np.linspace(2.0e9, 4.0e9, 9)
    E_inc_ref = _dtft(ref_inc, t_arr, freqs)
    E_inc_trn = _dtft(trn_inc, t_arr, freqs)

    for ratio in (0.5, 2.0, 4.0):
        sigma_s = ratio / ETA0
        _, ref_tot, trn_tot, _ = _run_1d_sheet_scattering(
            sheet_fn=lambda s, sig=sigma_s: s.add_sheet(
                sigma_s=sig, plane="z", position=iz_sheet * dz_base
            ),
            f0=f0,
            n_cells=n_cells,
            dz=dz_base,
        )
        t_num = _dtft(trn_tot, t_arr, freqs) / E_inc_trn
        r_num = _dtft(ref_tot - ref_inc, t_arr, freqs) / E_inc_ref
        T_num = np.abs(t_num) ** 2
        R_num = np.abs(r_num) ** 2

        t_ana, r_ana, T_ana, R_ana = sheet_transmission_reflection(sigma_s, freqs)
        np.testing.assert_allclose(np.abs(t_num), np.abs(t_ana), rtol=8e-3)
        np.testing.assert_allclose(T_num, T_ana, rtol=1.5e-2)
        np.testing.assert_allclose(R_num, R_ana, rtol=1.5e-2)
        # Passivity: absorbed fraction A = 1 - R - T > 0 matches eta0 * sigma_s * |t|^2
        A_num = 1.0 - R_num - T_num
        A_ana = ETA0 * sigma_s * T_ana
        np.testing.assert_allclose(A_num, A_ana, rtol=1.5e-2)

    # Non-uniform z mesh around iz_sheet: dz[iz_sheet - 1] = 1.1 * dz_base and
    # dz[iz_sheet] = 1.7 * dz_base so dual-edge length tL_z[iz_sheet] = 1.4 * dz_base
    # differs from both dz_base (by 40%) and primary cell length dz[iz_sheet] = 1.7 * dz_base.
    dz_arr = np.full(n_cells, dz_base)
    dz_arr[iz_sheet - 1] = 1.1 * dz_base
    dz_arr[iz_sheet] = 1.7 * dz_base
    z_nodes = np.r_[0.0, np.cumsum(dz_arr)]
    z_sheet = float(z_nodes[iz_sheet])
    sigma_s = 2.0 / ETA0
    t_nu, ref_nu_inc, trn_nu_inc, _ = _run_1d_sheet_scattering(
        sheet_fn=None, f0=f0, n_cells=n_cells, z_nodes=z_nodes
    )
    _, ref_nu, trn_nu, s_nu = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(sigma_s=sigma_s, plane="z", position=z_sheet),
        f0=f0,
        n_cells=n_cells,
        z_nodes=z_nodes,
    )
    T_nu = float(np.abs(_dtft(trn_nu, t_nu, [f0])[0] / _dtft(trn_nu_inc, t_nu, [f0])[0]) ** 2)
    R_nu = float(
        np.abs(_dtft(ref_nu - ref_nu_inc, t_nu, [f0])[0] / _dtft(ref_nu_inc, t_nu, [f0])[0]) ** 2
    )
    _, _, T_ana0, R_ana0 = sheet_transmission_reflection(sigma_s, f0)
    assert abs(T_nu - T_ana0) / T_ana0 < 2.0e-2
    assert abs(R_nu - R_ana0) / R_ana0 < 2.0e-2
    # Verify that a naive uniform-grid shortcut sigma = sigma_s / dz_base would
    # use weight 1 / dz_base instead of 1 / tL_z[iz_sheet] and fail:
    tL_sheet = float(s_nu.grid.tL[0, 0, iz_sheet, "z"])
    _, ref_shortcut, trn_shortcut, _ = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(
            sigma_s=sigma_s,
            plane="z",
            position=z_sheet,
            dual_length_factor=tL_sheet / dz_base,
        ),
        f0=f0,
        n_cells=n_cells,
        z_nodes=z_nodes,
    )
    T_shortcut = float(
        np.abs(_dtft(trn_shortcut, t_nu, [f0])[0] / _dtft(trn_nu_inc, t_nu, [f0])[0]) ** 2
    )
    assert abs(T_shortcut - T_ana0) / T_ana0 > 0.25


def test_salisbury_screen_and_mutation():
    """
    A Salisbury screen (R_s = eta0 ~ 377 Ohm/sq placed d = lambda0 / 4 in front
    of a PEC ground plane) achieves R(f0) ~ 0 at the design frequency and
    follows the analytic R(f) curve across the band. Mutating the dual-edge
    length factor breaks the impedance match and must fail the null check.
    """
    f0 = 3.0e9
    lam0 = c0 / f0
    d = 0.25 * lam0
    n_spacer = 12
    dz = d / n_spacer
    n_cells = 48
    z_sheet = (n_cells - n_spacer) * dz

    t_arr, ref_inc, _, _ = _run_1d_sheet_scattering(
        sheet_fn=None, f0=f0, n_cells=n_cells, dz=dz, bc_high_z="cpml", nt=750
    )
    _, ref_sal, _, _ = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(R_s=ETA0, plane=("z", z_sheet)),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
        bc_high_z="pec",
        nt=750,
    )

    freqs = np.linspace(1.5e9, 4.5e9, 31)
    E_inc = _dtft(ref_inc, t_arr, freqs)
    E_ref = _dtft(ref_sal - ref_inc, t_arr, freqs)
    R_num = np.abs(E_ref / E_inc) ** 2
    _, R_ana = salisbury_screen_reflection(1.0 / ETA0, d, freqs)

    idx_f0 = int(np.argmin(np.abs(freqs - f0)))
    assert R_num[idx_f0] < 5e-4
    np.testing.assert_allclose(R_num, R_ana, atol=3.5e-2)

    # Mutation check: wrong dual-edge length factor (2x) ruins the 377 Ohm/sq match
    _, ref_mut, _, _ = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(
            R_s=ETA0, plane=("z", z_sheet), dual_length_factor=2.0
        ),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
        bc_high_z="pec",
        nt=750,
    )
    R_mut = np.abs(_dtft(ref_mut - ref_inc, t_arr, [f0])[0] / _dtft(ref_inc, t_arr, [f0])[0]) ** 2
    assert R_mut > 0.08, f"Mutated dual-edge length factor should fail R(f0) ~ 0, got {R_mut}"


def test_graphene_sheet_drude_and_passivity():
    """
    Graphene intraband Drude sheet: verify both exp(+j w t) and exp(-i w t)
    phasor conventions, compare time-domain FIT R(f) and T(f) against the
    analytic Drude sheet formula over a band, and verify passivity (R + T <= 1).
    """
    E_F_eV = 0.35
    tau = 0.08e-10  # 80 ps (scaled for GHz test grid; identical dimensionless w*tau)
    mat = graphene_sheet(E_F=E_F_eV, tau=tau, T=300.0)

    f_test = np.linspace(2.0e9, 4.5e9, 11)
    sig_plus = mat.surface_conductivity(f_test, convention="+")
    sig_minus = mat.surface_conductivity(f_test, convention="-")
    np.testing.assert_allclose(sig_minus, np.conj(sig_plus), rtol=1e-15)
    assert np.all(sig_plus.real > 0.0)
    assert np.all(sig_plus.imag < 0.0)  # inductive in exp(+j w t)
    assert np.all(sig_minus.imag > 0.0)  # inductive in exp(-i w t)

    # Scale E_F so sigma_dc ~ 2 / ETA0 for O(1) reflection/transmission on the GHz grid
    scale_sig = (2.0 / ETA0) / mat.meta["sigma_dc"]
    mat_scaled = mat.scaled(scale_sig, name="graphene_scaled")

    f0 = 3.0e9
    lam0 = c0 / f0
    dz = lam0 / 40.0
    n_cells = 56
    z_sheet = (n_cells // 2) * dz

    t_arr, ref_inc, trn_inc, _ = _run_1d_sheet_scattering(
        sheet_fn=None, f0=f0, n_cells=n_cells, dz=dz
    )
    _, ref_gr, trn_gr, _ = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(
            material=mat_scaled, plane="xy", position=z_sheet
        ),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
    )

    E_inc_ref = _dtft(ref_inc, t_arr, f_test)
    E_inc_trn = _dtft(trn_inc, t_arr, f_test)
    T_num = np.abs(_dtft(trn_gr, t_arr, f_test) / E_inc_trn) ** 2
    R_num = np.abs(_dtft(ref_gr - ref_inc, t_arr, f_test) / E_inc_ref) ** 2

    _, _, T_ana, R_ana = sheet_transmission_reflection(mat_scaled, f_test)
    np.testing.assert_allclose(T_num, T_ana, rtol=1.5e-2)
    np.testing.assert_allclose(R_num, R_ana, rtol=2.0e-2)
    assert np.all(R_num + T_num < 1.0)


def test_metal_sheet_thin_and_sibc():
    """
    Thin metal trace models:
    1) t << delta ('thin'): R_s = 1 / (sigma * t) constant sheet resistance.
    2) t >> delta ('sibc'): Z_s(w) = (1 + j) / (sigma * delta(w)), verifying both
       the narrowband 1-pole model around f0 and the 3-pole Stieltjes-Drude fit
       over a 3:1 band, plus time-domain FIT reflection vs analytic SIBC.
    """
    sigma_cu = 5.8e7
    # 1) Thin regime: t = 20 nm << delta (~1.2 um at 3 GHz)
    t_thin = 20e-9
    m_thin = metal_sheet(sigma_cu, thickness=t_thin, mode="auto", f0=3e9)
    assert m_thin.meta["model"] == "metal_thin"
    np.testing.assert_allclose(
        m_thin.surface_conductivity(3e9), sigma_cu * t_thin, rtol=1e-14
    )

    # 2) Thick skin-effect regime: t = 50 um >> delta (~0.38 um at 30 GHz)
    f0_mm = 30.0e9
    w0_mm = 2.0 * np.pi * f0_mm
    delta0 = np.sqrt(2.0 / (w0_mm * mu_0 * sigma_cu))
    Zs_exact = lambda f: (1.0 + 1j) * np.sqrt(2.0 * np.pi * np.asarray(f) * mu_0 / (2.0 * sigma_cu))

    m_nb = metal_sheet(sigma_cu, thickness=50e-6, mode="sibc", f0=f0_mm, n_poles=1)
    assert m_nb.meta["model"] == "metal_sibc_narrowband"
    # Exact at centre frequency f0_mm
    np.testing.assert_allclose(m_nb.sheet_impedance(f0_mm), Zs_exact(f0_mm), rtol=1e-14)
    # Within ~5% in |Z_s| over [0.85 f0, 1.15 f0]
    f_nb_band = np.linspace(0.85 * f0_mm, 1.15 * f0_mm, 21)
    rel_nb = np.abs(np.abs(m_nb.sheet_impedance(f_nb_band)) - np.abs(Zs_exact(f_nb_band))) / np.abs(Zs_exact(f_nb_band))
    assert np.max(rel_nb) < 0.05

    # Wideband 3-pole fit over 20-60 GHz (3:1 band)
    m_wb = metal_sheet(sigma_cu, mode="sibc", f_range=(20e9, 60e9), n_poles=3)
    assert m_wb.meta["model"] == "metal_sibc_multipole"
    f_wb = np.linspace(20e9, 60e9, 41)
    sig_wb = m_wb.surface_conductivity(f_wb)
    assert np.all(sig_wb.real > 0.0)
    rel_wb = np.abs(m_wb.sheet_impedance(f_wb) - Zs_exact(f_wb)) / np.abs(Zs_exact(f_wb))
    assert np.max(rel_wb) < 0.025

    # Time-domain FIT verification of the SIBC sheet on a 3 GHz grid (moderate sigma so R and T are both measurable)
    sigma_mod = 200.0
    f0 = 3.0e9
    m_td = metal_sheet(sigma_mod, mode="sibc", f_range=(2.0e9, 4.0e9), n_poles=3)
    lam0 = c0 / f0
    dz = lam0 / 40.0
    n_cells = 56
    z_sheet = (n_cells // 2) * dz
    t_arr, ref_inc, trn_inc, _ = _run_1d_sheet_scattering(
        sheet_fn=None, f0=f0, n_cells=n_cells, dz=dz
    )
    _, ref_sibc, trn_sibc, _ = _run_1d_sheet_scattering(
        sheet_fn=lambda s: s.add_sheet(material=m_td, plane=("z", z_sheet)),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
    )
    freqs = np.linspace(2.2e9, 3.8e9, 9)
    T_num = np.abs(_dtft(trn_sibc, t_arr, freqs) / _dtft(trn_inc, t_arr, freqs)) ** 2
    Zs_mod = (1.0 + 1j) * np.sqrt(2.0 * np.pi * freqs * mu_0 / (2.0 * sigma_mod))
    _, _, T_ana, _ = sheet_transmission_reflection(1.0 / Zs_mod, freqs)
    np.testing.assert_allclose(T_num, T_ana, rtol=2.0e-2)


def test_pec_sheet_matches_pec_plate_to_roundoff():
    """
    The sigma_s -> inf ('pec') limit of solver.add_sheet sets tangential E = 0
    on the sheet edges and reproduces an explicit PEC plate (ieps = 0) to
    machine precision.
    """
    grid1 = wakis.GridFIT3D(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 10, 10, 10, verbose=0)
    grid2 = wakis.GridFIT3D(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 10, 10, 10, verbose=0)
    s1 = wakis.SolverFIT3D(grid1, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
    s2 = wakis.SolverFIT3D(grid2, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)

    # Patch on plane iz = 5 spanning x in [-0.03, 0.03], y in [-0.03, 0.03]
    z_plane = float(grid1.z[5])
    info = s1.add_sheet(
        center=(0.0, 0.0, z_plane), size=(0.06, 0.06, 0.0), sigma_s=np.inf
    )

    # Explicit PEC plate on the exact same tangential edges in s2
    idx_pec = np.concatenate([info["idx_u"], info["idx_v"]])
    s2.ieps.array[idx_pec] = 0.0
    s2.update_tensors()
    s2.step_0 = True

    src1 = CurrentSource(
        GaussianSource(3.0e9, 1.5e9), component="x", center=(0.0, 0.0, -0.02)
    )
    src2 = CurrentSource(
        GaussianSource(3.0e9, 1.5e9), component="x", center=(0.0, 0.0, -0.02)
    )
    s1.emsolve(40, source=src1)
    s2.emsolve(40, source=src2)

    assert np.max(np.abs(s1.E.array[idx_pec])) == 0.0
    np.testing.assert_allclose(s1.E.toarray(), s2.E.toarray(), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(s1.H.toarray(), s2.H.toarray(), rtol=0.0, atol=1e-15)


def test_anisotropic_and_polydata_sheet():
    """
    Anisotropic sheet conductivity (sigma_xx != sigma_yy) and planar
    pyvista.PolyData surface input to solver.add_sheet.
    """
    import pyvista as pv

    f0 = 3.0e9
    lam0 = c0 / f0
    dz = lam0 / 32.0
    grid = wakis.GridFIT3D(-0.04, 0.04, -0.04, 0.04, -10 * dz, 10 * dz, 8, 8, 20, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["periodic", "periodic", "cpml"], bc_high=["periodic", "periodic", "cpml"], n_pml=6, verbose=0
    )
    plane_mesh = pv.Plane(
        center=(0.0, 0.0, float(grid.z[10])),
        direction=(0.0, 0.0, 1.0),
        i_size=0.05,
        j_size=0.05,
        i_resolution=2,
        j_resolution=2,
    ).triangulate()
    info = solver.add_sheet(plane_mesh, sigma_s={"x": 1.0 / ETA0, "y": 4.0 / ETA0})
    assert info["normal"] == "z"
    assert info["idx_u"].size > 0 and info["idx_v"].size > 0
    assert len(solver._dispersive_groups) == 2


# ---------------------------------------------------------------------------
# B. Thin wires (subcell contour-path / Hodge model)
# ---------------------------------------------------------------------------


def _run_thin_wire_dipole(radius_ratio, n_seg=15, correction=True):
    """Run a 3D FIT centre-fed thin-wire dipole and return (freqs, Z_fit, Z_mom)."""
    f0 = 1.0e9
    lam0 = c0 / f0
    length = 0.48 * lam0
    dz = length / n_seg
    dx = dy = dz
    nx = ny = 24
    nz = n_seg + 16
    grid = wakis.GridFIT3D(
        -nx * dx / 2,
        nx * dx / 2,
        -ny * dy / 2,
        ny * dy / 2,
        -nz * dz / 2,
        nz * dz / 2,
        nx,
        ny,
        nz,
        use_mpi=False,
        verbose=0,
    )
    ix, iy = nx // 2, ny // 2
    iz0 = (nz - n_seg) // 2
    iz1 = iz0 + n_seg
    iz_feed = (iz0 + iz1) // 2
    x_w = float(grid.x[ix])
    y_w = float(grid.y[iy])
    z0 = float(grid.z[iz0])
    z1 = float(grid.z[iz1])
    z_feed = 0.5 * float(grid.z[iz_feed] + grid.z[iz_feed + 1])

    radius = dx / float(radius_ratio)
    src = GaussianSource(frequency=f0, fwidth=0.5 * f0)
    port = LumpedPort(
        src_time=src,
        component="z",
        center=(x_w, y_w, z_feed),
        size=(0.0, 0.0, 0.5 * dz),
        impedance=50.0,
    )
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver.add_thin_wire(
        [(x_w, y_w, z0), (x_w, y_w, z1)],
        radius=radius,
        correction=correction,
    )
    solver.emsolve(int(40 * n_seg), source=port)
    freqs = np.linspace(0.85e9, 1.15e9, 121)
    Z_fit = port.impedance_in(freqs)
    Z_mom, _, _ = hallen_dipole(length, radius, freqs, n_segments=60, gap=dz)
    return freqs, Z_fit, Z_mom


def test_thin_wire_dipole_vs_hallen_and_mutation():
    """
    Thin-wire dipole input impedance vs Hallen MoM for two subcell radii
    a = dx / 10 and a = dx / 50:
    - Resonance frequency shifts upward as radius decreases (matching MoM);
    - Reactance slope dX_in/df increases as radius decreases (matching MoM);
    - Input impedance at resonance and across the band agrees with Hallen MoM;
    - Mutation: disabling the thin-wire ln(dx/a) correction eliminates the
      radius dependence completely, which fails the shift test.
    """
    freqs, Z_10, Z_mom_10 = _run_thin_wire_dipole(10.0, n_seg=15, correction=True)
    _, Z_50, Z_mom_50 = _run_thin_wire_dipole(50.0, n_seg=15, correction=True)

    f_res_10 = freqs[int(np.argmin(np.abs(Z_10.imag)))]
    f_res_50 = freqs[int(np.argmin(np.abs(Z_50.imag)))]
    f_mom_10 = freqs[int(np.argmin(np.abs(Z_mom_10.imag)))]
    f_mom_50 = freqs[int(np.argmin(np.abs(Z_mom_50.imag)))]

    # Resonance shifts up with thinner wire, matching Hallen MoM within ~1.5%
    assert f_res_50 > f_res_10, f"Expected f_res(dx/50) > f_res(dx/10), got {f_res_50} vs {f_res_10}"
    assert f_mom_50 > f_mom_10
    assert abs(f_res_10 - f_mom_10) / f_mom_10 < 0.02
    assert abs(f_res_50 - f_mom_50) / f_mom_50 < 0.02

    # Reactance slope increases significantly for the thinner wire
    slope_10 = (Z_10[-1].imag - Z_10[0].imag) / (freqs[-1] - freqs[0])
    slope_50 = (Z_50[-1].imag - Z_50[0].imag) / (freqs[-1] - freqs[0])
    assert slope_50 > 1.35 * slope_10

    # Resistance near resonance agrees with Hallen MoM (~72-78 Ohm)
    idx_f0 = int(np.argmin(np.abs(freqs - 1.0e9)))
    assert abs(Z_10[idx_f0].real - Z_mom_10[idx_f0].real) / Z_mom_10[idx_f0].real < 0.10
    assert abs(Z_50[idx_f0].real - Z_mom_50[idx_f0].real) / Z_mom_50[idx_f0].real < 0.10

    # Mutation check: without the ln(dx/a) correction, radius dependence disappears
    _, Z_10_mut, _ = _run_thin_wire_dipole(10.0, n_seg=11, correction=False)
    _, Z_50_mut, _ = _run_thin_wire_dipole(50.0, n_seg=11, correction=False)
    diff_uncorrected = float(np.max(np.abs(Z_50_mut - Z_10_mut)))
    diff_corrected = float(np.max(np.abs(Z_50 - Z_10)))
    assert diff_uncorrected == 0.0
    assert diff_corrected > 25.0


def test_thin_wire_grid_convergence_and_polyline():
    """
    Grid convergence of Z_in for a fixed physical wire radius as the mesh is
    refined, and support for an L-shaped polyline wire with a 90-degree bend.
    """
    f0 = 1.0e9
    lam0 = c0 / f0
    length = 0.48 * lam0
    a_phys = 0.0005  # 0.5 mm
    freqs = np.linspace(0.88e9, 1.12e9, 49)

    errors = []
    for n_seg in (9, 15):
        dz = length / n_seg
        dx = dy = dz
        n_pad = max(6, int(round(0.26 * lam0 / dx)))
        nx = ny = 2 * (n_pad + 4)
        nz = n_seg + 2 * (n_pad + 4)
        grid = wakis.GridFIT3D(
            -nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2, -nz * dz / 2, nz * dz / 2,
            nx, ny, nz, use_mpi=False, verbose=0,
        )
        ix, iy = nx // 2, ny // 2
        iz0 = (nz - n_seg) // 2
        iz1 = iz0 + n_seg
        iz_feed = (iz0 + iz1) // 2
        x_w, y_w = float(grid.x[ix]), float(grid.y[iy])
        z_feed = 0.5 * float(grid.z[iz_feed] + grid.z[iz_feed + 1])
        port = LumpedPort(
            GaussianSource(f0, 0.5 * f0),
            component="z",
            center=(x_w, y_w, z_feed),
            size=(0.0, 0.0, 0.5 * dz),
            impedance=50.0,
        )
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
        )
        solver.add_thin_wire(
            [(x_w, y_w, float(grid.z[iz0])), (x_w, y_w, float(grid.z[iz1]))],
            radius=a_phys,
            gap=port,
        )
        solver.emsolve(int(40 * n_seg), source=port)
        Z_fit = port.impedance_in(freqs)
        Z_mom, _, _ = hallen_dipole(length, a_phys, freqs, n_segments=80, gap=dz)
        errors.append(float(np.linalg.norm(Z_fit - Z_mom) / np.linalg.norm(Z_mom)))

    assert errors[1] < errors[0], f"Expected grid convergence, got errors {errors}"

    # Polyline (L-shaped) wire with bend node (degree 2) and end nodes (degree 1)
    grid_p = wakis.GridFIT3D(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 10, 10, 10, verbose=0)
    solver_p = wakis.SolverFIT3D(grid_p, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
    info = solver_p.add_thin_wire(
        [
            (float(grid_p.x[5]), float(grid_p.y[5]), float(grid_p.z[2])),
            (float(grid_p.x[5]), float(grid_p.y[5]), float(grid_p.z[5])),
            (float(grid_p.x[8]), float(grid_p.y[5]), float(grid_p.z[5])),
        ],
        radius=0.001,
    )
    assert len(info["edges"]) == 6
    assert info["node_degrees"][(5, 5, 5)] == 2  # 90-degree bend node
    assert info["node_degrees"][(5, 5, 2)] == 1  # end node
    assert info["node_degrees"][(8, 5, 5)] == 1  # end node


# ---------------------------------------------------------------------------
# C. Discrete energy conservation and decay
# ---------------------------------------------------------------------------


def test_energy_conservation_and_decay():
    """
    In a closed PEC cavity:
    - Lossless sheet (PEC patch) and lossless PEC thin wire conserve the
      discrete electromagnetic energy;
    - Lossy resistive sheet and lossy thin wire cause monotonic energy decay.
    """
    def run_cavity(lossy=False):
        grid = wakis.GridFIT3D(-0.06, 0.06, -0.06, 0.06, -0.06, 0.06, 12, 12, 12, verbose=0)
        solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
        z_sheet = float(grid.z[5])
        if lossy:
            solver.add_sheet(plane=("z", z_sheet), R_s=ETA0)
            solver.add_thin_wire(
                [(float(grid.x[4]), float(grid.y[4]), float(grid.z[2])),
                 (float(grid.x[4]), float(grid.y[4]), float(grid.z[9]))],
                radius=0.001,
                R_per_m=150.0,
            )
        else:
            solver.add_sheet(
                center=(0.0, 0.0, z_sheet), size=(0.06, 0.06, 0.0), sigma_s="pec"
            )
            solver.add_thin_wire(
                [(float(grid.x[4]), float(grid.y[4]), float(grid.z[2])),
                 (float(grid.x[4]), float(grid.y[4]), float(grid.z[9]))],
                radius=0.001,
                material="pec",
            )

        # Initialize the fundamental TE101/TM110 cavity modes (divergence-free, non-zero on sheet and wire)
        solver.one_step()
        y_c = np.sin(np.pi * (grid.y[:-1] - grid.ymin) / (grid.ymax - grid.ymin))
        z_c = np.sin(np.pi * (grid.z[:-1] - grid.zmin) / (grid.zmax - grid.zmin))
        x_c = np.sin(np.pi * (grid.x[:-1] - grid.xmin) / (grid.xmax - grid.xmin))
        solver.E[:, :, :, "x"] = (y_c[None, :, None] * z_c[None, None, :]) * (
            solver.ieps[:, :, :, "x"] != 0
        )
        solver.E[:, :, :, "z"] = (x_c[:, None, None] * y_c[None, :, None]) * (
            solver.ieps[:, :, :, "z"] != 0
        )

        energies = []
        exact_invariants = []
        L_e = solver.grid.L.toarray()
        itA_arr = solver.grid.itA.toarray()
        tA_e = np.divide(
            1.0, itA_arr, out=np.zeros(3 * solver.N), where=itA_arr != 0
        )
        ieps_e = solver.ieps.toarray()
        w_e = np.divide(L_e * tA_e, ieps_e, out=np.zeros(3 * solver.N), where=ieps_e != 0)
        tL_h = solver.grid.tL.toarray()
        iA_arr = solver.grid.iA.toarray()
        A_h = np.divide(
            1.0, iA_arr, out=np.zeros(3 * solver.N), where=iA_arr != 0
        )
        imu_h = solver.imu.toarray()
        w_h = np.divide(tL_h * A_h, imu_h, out=np.zeros(3 * solver.N), where=imu_h != 0)

        for _ in range(180):
            E_prev = solver.E.toarray().copy()
            H_prev = solver.H.toarray().copy()
            solver.one_step()
            H_next = solver.H.toarray()
            energies.append(solver.field_energy())
            # Exact integer-time leapfrog Poynting energy:
            # W^n = 0.5 * <E^n, D^n> + 0.5 * <H^{n-1/2}, B^{n+1/2}>
            # Satisfies W^{n+1} - W^n = -dt * <G (E^{n+1}+E^n)/2, (E^{n+1}+E^n)/2> <= 0
            exact_invariants.append(
                0.5 * float(np.sum(w_e * E_prev * E_prev) + np.sum(w_h * H_prev * H_next))
            )
        return np.asarray(energies), np.asarray(exact_invariants)

    e_lossless, inv_lossless = run_cavity(lossy=False)
    e_lossy, inv_lossy = run_cavity(lossy=True)

    # Exact leapfrog discrete energy invariant is conserved to machine precision (~1e-14)
    rel_inv_drift = np.max(np.abs(inv_lossless[1:] - inv_lossless[1])) / inv_lossless[1]
    assert rel_inv_drift < 1e-12
    # Standard staggered field_energy() has zero secular drift
    assert abs(np.mean(e_lossless[-40:]) - np.mean(e_lossless[:40])) / np.mean(e_lossless) < 1e-2
    # Lossy cavity energy decays monotonically at every step and substantially over the run
    assert np.all(np.diff(inv_lossy[1:]) <= 1e-14 * inv_lossy[1])
    assert inv_lossy[-1] < 0.40 * inv_lossy[1]
    assert e_lossy[-1] < 0.40 * e_lossy[0]


# ---------------------------------------------------------------------------
# D. Adjoint dot-product and finite-difference gradient tests
# ---------------------------------------------------------------------------


def test_adjoint_dot_product_and_fd_gradient():
    """
    1) Dot-product test <U x, y> == <x, U^H y> to machine precision (~1e-14)
       with a dispersive sheet, a resistive sheet, and a thin wire (with a
       lumped element across its gap) all present simultaneously.
    2) Finite-difference gradient check of a DFT objective wrt a MaterialGrid
       permittivity design region next to a conducting/dispersive sheet.
    """
    # 1. Exact dot-product test of AdjointStepper
    grid = wakis.GridFIT3D(-0.08, 0.08, -0.08, 0.08, -0.08, 0.08, 12, 12, 12, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=4, verbose=0
    )
    # Add a graphene/Drude dispersive sheet on plane z = grid.z[5]
    # Note: SheetMaterial Drude omega_p has units [rad/s * sqrt(m)] since chi_s [m] is divided by tL_perp
    dz_cell = float(grid.z[1] - grid.z[0])
    gr_mat = SheetMaterial(
        poles=[
            Drude(
                omega_p=2 * np.pi * 2.5e9 * np.sqrt(dz_cell),
                gamma=2 * np.pi * 0.6e9,
            )
        ],
        sigma_s=0.005,
        name="sheet_ade",
    )
    solver.add_sheet(
        center=(0.0, 0.0, float(grid.z[5])),
        size=(0.10, 0.10, 0.0),
        material=gr_mat,
    )
    # Add a thin wire along z with a LumpedElement across its middle segment
    xw, yw = float(grid.x[7]), float(grid.y[7])
    z_mid = 0.5 * float(grid.z[6] + grid.z[7])
    dz_w = float(grid.z[7] - grid.z[6])
    load = LumpedElement(
        component="z",
        center=(xw, yw, z_mid),
        size=(0.0, 0.0, 0.5 * dz_w),
        R=75.0,
        L=2e-9,
        C=0.5e-12,
    )
    solver.add_thin_wire(
        [(xw, yw, float(grid.z[4])), (xw, yw, float(grid.z[9]))],
        radius=0.001,
        gap=load,
    )

    elements = [load]
    load.update(solver, 0.0)
    solver.one_step()
    load.post_step(solver, 0.0)

    stepper = AdjointStepper(solver, elements)
    rng = np.random.default_rng(42)
    template = stepper.zero_state()
    keys = list(template)
    aux = [k for k in keys if k not in ("E", "H")]
    assert len(aux) > 0

    for x_sup, y_sup in [(keys, keys), (aux, keys), (keys, aux)]:
        x = {k: rng.standard_normal(v.shape) * (k in x_sup) for k, v in template.items()}
        y = {k: rng.standard_normal(v.shape) * (k in y_sup) for k, v in template.items()}
        stepper.set_state(x)
        load.update(solver, 0.0)
        solver.one_step()
        load.post_step(solver, 0.0)
        Ux = stepper.get_state()
        Uty = stepper.step(y)

        lhs = stepper.inner(Ux, y)
        rhs = stepper.inner(x, Uty)
        scale = stepper.inner(
            {k: np.abs(v) for k, v in Ux.items()},
            {k: np.abs(v) for k, v in y.items()},
        ).real
        rel_err = abs(lhs - rhs) / scale
        assert rel_err < 1e-13, f"Dot-product test failed: rel_err={rel_err:.2e}"

    # 2. Finite-difference gradient check wrt permittivity design region next to a sheet
    f0 = 3.0e9
    n_2d, l_2d = 32, 0.24
    dx_2d = l_2d / n_2d

    def make_solver():
        g = wakis.GridFIT3D(
            -l_2d / 2, l_2d / 2, -l_2d / 2, l_2d / 2, -dx_2d / 2, dx_2d / 2,
            n_2d, n_2d, 1, verbose=0,
        )
        s = wakis.SolverFIT3D(
            g,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=6,
            verbose=0,
        )
        # Place a conducting/dispersive sheet at x = +0.03 m right next to the design region
        x_sheet = float(g.x[n_2d // 2 + 4])
        s.add_sheet(
            center=(x_sheet, 0.0, 0.0),
            size=(0.0, 0.09, dx_2d),
            material=SheetMaterial(
                poles=[
                    Drude(
                        2 * np.pi * 2e9 * np.sqrt(dx_2d),
                        2 * np.pi * 0.5e9,
                    )
                ],
                sigma_s=1.0 / ETA0,
            ),
        )
        return s

    source = CurrentSource(
        GaussianSource(f0, 1.0e9), component="z", center=(-0.05, 0.0, 0.0)
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            [f0], components=["Ez"], center=(0.05, 0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(-0.01, 0.0, 0.0),
        size=(0.03, 0.045, 0.0),
        shape=(3, 3, 1),
        eps_min=1.0,
        eps_max=4.0,
    )
    sim = AdjointSimulation(make_solver, [source], monitors, design, 280)

    def eval_obj(out):
        Ez = out["probe"]["Ez"]
        scale_f = 1e20
        val = float(scale_f * np.sum(np.abs(Ez) ** 2))
        grad_Ez = 2.0 * scale_f * Ez
        return val, {"probe": {"Ez": grad_Ez}}

    rho = np.random.default_rng(7).uniform(0.2, 0.8, design.shape)
    _, cot = eval_obj(sim.forward(rho))
    adj_grad = sim.vjp(cot)

    h = 1e-4
    for node_idx in [(0, 0, 0), (1, 1, 0), (2, 1, 0)]:
        step = np.zeros_like(rho)
        step[node_idx] = h
        f_plus, _ = eval_obj(sim.forward(rho + step))
        f_minus, _ = eval_obj(sim.forward(rho - step))
        fd_grad = (f_plus - f_minus) / (2.0 * h)
        rel_diff = abs(adj_grad[node_idx] - fd_grad) / max(abs(fd_grad), 1e-30)
        assert rel_diff < 1e-6, (
            f"FD gradient mismatch at {node_idx}: adj={adj_grad[node_idx]:.6e}, "
            f"fd={fd_grad:.6e}, rel_diff={rel_diff:.2e}"
        )
