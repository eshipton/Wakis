# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Tests for broadband oblique-incidence sweeps at constant angle and Floquet ports."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

import wakis
from wakis.floquet import (
    AngleSweep,
    FloquetCell,
    FloquetMode,
    FloquetPort,
    floquet_s_matrix,
    fresnel_rt,
    grating_angles,
    k_parallel_grid,
    rayleigh_sinusoidal_grating,
    rcwa_1d_grating,
    reciprocal_s_matrix,
    resample_constant_angle,
    s_parameters_from_ports,
    sampling_density,
    slab_rt,
)
from wakis.sources import CurrentSource, GaussianSource


def _make_slab_callback(thickness: float, eps_slab: float):
    """Subpixel-smoothed planar dielectric slab centered at z = 0."""

    def _fill(solver):
        z = solver.grid.z
        dz = float(solver.grid.dz[0])
        tol = 1e-6 * dz
        zc = z[:-1]
        for comp in ("x", "y"):
            inside = (zc > -0.5 * thickness + tol) & (zc < 0.5 * thickness - tol)
            on_face = (np.abs(zc + 0.5 * thickness) <= tol) | (
                np.abs(zc - 0.5 * thickness) <= tol
            )
            solver.ieps[:, :, inside, comp] = 1.0 / (epsilon_0 * eps_slab)
            solver.ieps[:, :, on_face, comp] = 2.0 / (epsilon_0 * (1.0 + eps_slab))
        inside_z = (solver.z > -0.5 * thickness) & (solver.z < 0.5 * thickness)
        solver.ieps[:, :, inside_z, "z"] = 1.0 / (epsilon_0 * eps_slab)

    return _fill


def test_slab_constant_angle_te_tm_and_brewster():
    """Dielectric slab R(f), T(f) at constant angles 30 and 60 deg (TE & TM) + Brewster zero.

    Also verifies the required mutation check: shifting the k_par resampling lookup
    by one index (index_shift=1) causes the R(f)/T(f) comparison to fail.
    """
    freqs = np.linspace(1.0e9, 1.3e9, 7)
    d_slab = 0.08
    eps_slab = 2.25
    brewster_rad = float(np.arctan(np.sqrt(eps_slab)))  # ~56.31 deg
    angles = np.array([np.deg2rad(30.0), brewster_rad, np.deg2rad(60.0)])

    cell = FloquetCell(
        Lx=0.02,
        Ly=0.02,
        zmin=-0.35,
        zmax=0.35,
        nx=2,
        ny=2,
        nz=140,
        z_source=-0.18,
        z_refl=-0.10,
        z_tran=0.10,
        z_ref_below=-0.5 * d_slab,
        z_ref_above=0.5 * d_slab,
        orders_x=(0,),
        orders_y=(0,),
        n_pml=15,
        material_callback=_make_slab_callback(d_slab, eps_slab),
        run_time_cycles=22.0,
    )

    for pol in ("TE", "TM"):
        sweep = AngleSweep(cell, freqs, angles, polarization=pol, n_k=7)
        res = sweep.run()
        out = res.resample()
        exact = slab_rt(freqs, angles, d_slab, eps_slab, pol=pol)

        # 30 deg (index 0) and 60 deg (index 2) R(f) and T(f) vs transfer matrix
        for ia in (0, 2):
            np.testing.assert_allclose(out["R_00"][ia], exact["R"][ia], atol=6.5e-3)
            np.testing.assert_allclose(out["T_00"][ia], exact["T"][ia], atol=6.5e-3)
            np.testing.assert_allclose(
                out["R_total"][ia] + out["T_total"][ia], 1.0, atol=6.5e-3
            )

        # Complex reflection S-parameter phase referenced to the slab faces
        np.testing.assert_allclose(
            out[f"S_refl_{pol}"][0, :, 0, 0], exact["r"][0], atol=6.5e-3
        )

        # Mutation check: shifting resampling by one k_par index must fail tolerance
        mutated = res.resample(index_shift=1)
        err_mutated = np.nanmax(np.abs(mutated["R_00"][[0, 2]] - exact["R"][[0, 2]]))
        assert err_mutated > 1.0e-2, (
            f"Mutation index_shift=1 did not fail for {pol}: max error {err_mutated}"
        )

        if pol == "TM":
            # At Brewster angle, TM reflection vanishes across the entire band
            assert np.all(out["R_00"][1] < 2e-3)
            # Scan a fine angular grid around Brewster angle and locate minimum R_TM
            scan_deg = np.linspace(45.0, 62.0, 35)
            scan_rad = np.deg2rad(scan_deg)
            scan_r = res.order_efficiency(0, 0, side="refl", angles=scan_rad)
            # Mid-band frequency index 3
            imin = int(np.nanargmin(scan_r[:, 3]))
            est_brewster_deg = scan_deg[imin]
            dtheta_deg = sweep.density["dtheta_deg"]
            assert abs(est_brewster_deg - np.rad2deg(brewster_rad)) <= max(
                1.5, dtheta_deg
            )


def test_total_internal_reflection_from_denser_medium():
    """Total internal reflection above the critical angle from n1 = 1.5 into air n2 = 1.0."""
    freqs = np.linspace(1.0e9, 1.15e9, 5)
    eps_below = 2.25  # n1 = 1.5 -> critical angle arcsin(1/1.5) = 41.81 deg
    eps_above = 1.0
    theta_tir = np.deg2rad(52.0)  # strictly above critical angle across all (k_par, f)

    def _fill_interface(solver):
        z = solver.grid.z
        dz = float(solver.grid.dz[0])
        tol = 1e-6 * dz
        zc = z[:-1]
        for comp in ("x", "y"):
            solver.ieps[:, :, zc < -tol, comp] = 1.0 / (epsilon_0 * eps_below)
            solver.ieps[:, :, zc > tol, comp] = 1.0 / (epsilon_0 * eps_above)
            solver.ieps[:, :, np.abs(zc) <= tol, comp] = 2.0 / (
                epsilon_0 * (eps_below + eps_above)
            )
        solver.ieps[:, :, solver.z < 0.0, "z"] = 1.0 / (epsilon_0 * eps_below)
        solver.ieps[:, :, solver.z >= 0.0, "z"] = 1.0 / (epsilon_0 * eps_above)

    cell = FloquetCell(
        Lx=0.02,
        Ly=0.02,
        zmin=-0.35,
        zmax=0.35,
        nx=2,
        ny=2,
        nz=140,
        z_source=-0.18,
        z_refl=-0.10,
        z_tran=0.10,
        eps_below=eps_below,
        eps_above=eps_above,
        orders_x=(0,),
        orders_y=(0,),
        n_pml=15,
        material_callback=_fill_interface,
        run_time_cycles=22.0,
    )

    sweep = AngleSweep(cell, freqs, [theta_tir], polarization="TE", n_k=4)
    res = sweep.run()
    out = res.resample()
    exact = fresnel_rt(theta_tir, np.sqrt(eps_below), np.sqrt(eps_above), pol="TE")

    assert exact["tir"]
    np.testing.assert_allclose(out["R_00"][0], 1.0, atol=5e-3)
    assert np.all(np.isnan(out["T_00"][0]))



def test_lossless_grating_angles_energy_and_rayleigh_perturbation():
    """Lossless shallow sinusoidal grating: sum(efficiencies)=1, grating angles, Rayleigh theory."""
    lam0 = 0.30
    f0 = c / lam0
    freqs = np.linspace(0.95 * f0, 1.12 * f0, 6)
    period = 1.35 * lam0
    eps1 = 1.0
    eps2 = 2.25
    nx = 24
    nz = 120
    zmin, zmax = -0.60, 0.60
    dz = (zmax - zmin) / nz
    # Shallow sinusoidal interface z(x) = a * cos(2*pi*x / P) with k0 * a << 1
    amp = 0.15 * dz
    theta_inc = np.deg2rad(15.0)

    def _fill_sinusoidal_grating(solver):
        x_nodes = solver.grid.x[:-1]
        z_nodes = solver.grid.z[:-1]
        iz0 = int(np.argmin(np.abs(z_nodes - 0.0)))
        profile_cos = np.cos(2.0 * np.pi * x_nodes / period)[:, None]
        # Subpixel-averaged permittivity on the interface plane z = 0
        eps_interface = 0.5 * (eps1 + eps2) + (eps2 - eps1) * (amp / dz) * profile_cos
        for comp in ("x", "y"):
            solver.ieps[:, :, :iz0, comp] = 1.0 / (epsilon_0 * eps1)
            solver.ieps[:, :, iz0 + 1 :, comp] = 1.0 / (epsilon_0 * eps2)
            solver.ieps[:, :, iz0, comp] = 1.0 / (epsilon_0 * eps_interface)
        solver.ieps[:, :, solver.z < 0.0, "z"] = 1.0 / (epsilon_0 * eps1)
        solver.ieps[:, :, solver.z >= 0.0, "z"] = 1.0 / (epsilon_0 * eps2)

    orders_x = (-1, 0, 1)
    cell = FloquetCell(
        Lx=period,
        Ly=dz,
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=1,
        nz=nz,
        z_source=-0.32,
        z_refl=-0.18,
        z_tran=0.22,
        eps_below=eps1,
        eps_above=eps2,
        orders_x=orders_x,
        orders_y=(0,),
        n_pml=15,
        material_callback=_fill_sinusoidal_grating,
        run_time_cycles=22.0,
    )

    sweep = AngleSweep(cell, freqs, [theta_inc], polarization="TE", n_k=5)
    res = sweep.run()
    out = res.resample()

    # 1. Energy conservation: sum of all propagating order efficiencies = 1
    total_eff = out["R_total"][0] + out["T_total"][0]
    np.testing.assert_allclose(total_eff, 1.0, atol=6e-3)

    # 2. Compare first-order efficiencies against Rayleigh perturbation theory
    rayleigh = rayleigh_sinusoidal_grating(
        freqs, theta_inc, period, amp, eps1=eps1, eps2=eps2, pol="TE"
    )
    t_m1 = res.order_efficiency(-1, 0, side="tran")[0]
    t_p1 = res.order_efficiency(+1, 0, side="tran")[0]
    r_m1 = res.order_efficiency(-1, 0, side="refl")[0]
    np.testing.assert_allclose(t_m1, rayleigh["T_m1"], rtol=8e-2, atol=1.5e-4)
    np.testing.assert_allclose(t_p1, rayleigh["T_p1"], rtol=8e-2, atol=1.5e-4)
    np.testing.assert_allclose(r_m1, rayleigh["R_m1"], rtol=1.0e-1, atol=1.5e-4)

    # 3. Verify diffraction order angles against the grating equation
    k_mid = float(sweep.k_par[2])
    port_below, port_above = wakis.floquet._run_single_Solver(
        cell, k_par=k_mid, phi=0.0, polarization="TE", frequencies=freqs
    )
    th_mon_above, _ = port_above.monitor.order_angles()
    th_inc_k = np.arcsin(c * k_mid / (2.0 * np.pi * freqs * np.sqrt(eps1)))
    for jf in range(freqs.size):
        expected = grating_angles(
            freqs[jf],
            th_inc_k[jf],
            period,
            port_above.orders_b,
            n_inc=np.sqrt(eps1),
            n_out=np.sqrt(eps2),
        )
        actual = th_mon_above[jf, :, 0]
        valid = np.isfinite(expected) & np.isfinite(actual)
        np.testing.assert_allclose(actual[valid], np.abs(expected[valid]), atol=1e-12)


def test_interpolation_convergence_rayleigh_masking_and_batch_cli(tmp_path):
    """Fourth-order O(dk^4) cubic convergence, Rayleigh masking, no extrapolation, and CLI."""
    freqs = np.linspace(1.0e9, 1.4e9, 9)
    angles = np.deg2rad([20.0, 35.0, 50.0])
    d_slab = 0.09
    eps_slab = 2.5

    exact_r = slab_rt(freqs, angles[1], d_slab, eps_slab, pol="TE")["r"]
    errors = []
    dks = []
    for n_k in (5, 9, 17):
        k_grid = k_parallel_grid(freqs, angles, n=1.0, n_k=n_k)
        dens = sampling_density(k_grid, freqs, angles=angles, n=1.0)
        dks.append(dens["dk"])
        k0 = 2.0 * np.pi * freqs[None, :] / c
        sin_th = np.clip(k_grid[:, None] / k0, 0.0, 0.9999)
        prop = (k_grid[:, None] / k0) < 0.9999
        vals = np.full((n_k, freqs.size), np.nan + 0j, dtype=complex)
        for ik in range(n_k):
            for jf in range(freqs.size):
                if prop[ik, jf]:
                    vals[ik, jf] = slab_rt(
                        freqs[jf], np.arcsin(sin_th[ik, jf]), d_slab, eps_slab, pol="TE"
                    )["r"]
        interp = resample_constant_angle(
            k_grid, freqs, vals, angles[1], n=1.0, propagating=prop
        )
        errors.append(float(np.max(np.abs(interp - exact_r))))

    # Fourth-order convergence: halving dk reduces error by ~16x (assert > 11x)
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 11.0, f"Expected O(dk^4) convergence ratio > 11, got {ratio_1}"
    assert ratio_2 > 11.0, f"Expected O(dk^4) convergence ratio > 11, got {ratio_2}"

    # Never extrapolate past sampled k_par range
    k_grid = k_parallel_grid(freqs, angles, n=1.0, n_k=7)
    ones = np.ones((7, freqs.size), dtype=float)
    out_low = resample_constant_angle(k_grid, freqs, ones, np.deg2rad(5.0), n=1.0)
    out_high = resample_constant_angle(k_grid, freqs, ones, np.deg2rad(75.0), n=1.0)
    assert np.isnan(out_low[0])
    assert np.isnan(out_high[-1])

    # Rayleigh anomaly masking: order appearing mid-grid must not be interpolated across cutoff
    prop_step = np.zeros((7, freqs.size), dtype=bool)
    prop_step[:3, :] = True  # Propagating for indices 0, 1, 2; evanescent for >= 3
    vals_step = np.where(prop_step, 1.0, 1e6)  # Large spike on evanescent side
    # Target angle whose k_target falls in the cutoff interval [k_2, k_3]
    k_mid_cutoff = 0.5 * (k_grid[2] + k_grid[3])
    th_cutoff = float(np.arcsin(k_mid_cutoff / (2.0 * np.pi * freqs[4] / c)))
    res_masked = resample_constant_angle(
        k_grid, [freqs[4]], vals_step[:, 4:5], th_cutoff, propagating=prop_step[:, 4:5]
    )
    assert np.isnan(res_masked[0])

    # Cutoff-straddling k_z interpolation and multi-order energy conservation in AngleSweepResult
    period_cut = 0.45
    f_cut_test = np.array([c / 0.32967])
    k0_cut = float(2.0 * np.pi * f_cut_test[0] / c)
    G_cut = 2.0 * np.pi / period_cut
    k_grid_cut = np.linspace(3.1277, 7.5930, 7)
    kz_p1 = np.sqrt(np.maximum(0.0, k0_cut**2 - (k_grid_cut + G_cut) ** 2))
    prop_p1 = (k_grid_cut + G_cut) < k0_cut  # True for indices 0, 1, 2; False for 3..6
    t_p1_exact_slope = 0.012
    t_p1_grid = np.where(prop_p1, t_p1_exact_slope * kz_p1, 0.0)
    t_0_grid = 1.0 - t_p1_grid  # Lossless sum = 1.0 at every k_par node

    t_te_cut = np.zeros((7, 1, 2, 1), dtype=float)
    t_te_cut[:, 0, 0, 0] = t_0_grid
    t_te_cut[:, 0, 1, 0] = t_p1_grid
    prop_cut = np.ones((7, 1, 2, 1), dtype=bool)
    prop_cut[:, 0, 1, 0] = prop_p1
    z_cut = np.zeros_like(t_te_cut)
    res_cut_obj = wakis.floquet.AngleSweepResult(
        k_par=k_grid_cut,
        frequencies=f_cut_test,
        angles=np.array([np.deg2rad(15.0)]),
        polarization="TE",
        phi=0.0,
        n_inc=1.0,
        n_trn=1.0,
        orders_x=(0, 1),
        orders_y=(0,),
        r_te=z_cut,
        r_tm=z_cut,
        t_te=t_te_cut,
        t_tm=z_cut,
        s_refl_te=z_cut.astype(complex),
        s_refl_tm=z_cut.astype(complex),
        s_tran_te=np.sqrt(t_te_cut).astype(complex),
        s_tran_tm=z_cut.astype(complex),
        prop_refl=prop_cut,
        prop_tran=prop_cut,
        inc_power=np.ones((7, 1)),
        Lx=period_cut,
        Ly=1.0,
    )
    out_cut = res_cut_obj.resample()
    kt_15 = k0_cut * np.sin(np.deg2rad(15.0))
    kz_p1_target = float(np.sqrt(k0_cut**2 - (kt_15 + G_cut) ** 2))
    np.testing.assert_allclose(
        out_cut["T_order"][0, 0, 1, 0], t_p1_exact_slope * kz_p1_target, atol=1e-12
    )
    np.testing.assert_allclose(
        out_cut["R_total"][0, 0] + out_cut["T_total"][0, 0], 1.0, atol=1e-12
    )
    # Mutation check: shifting the lookup index into the evanescent side masks m=+1
    out_cut_mut = res_cut_obj.resample(index_shift=1)
    assert np.isnan(out_cut_mut["T_order"][0, 0, 1, 0])

    sweep = AngleSweep(
        _mock_slab_builder,
        freqs,
        angles,
        polarization="TE",
        n_k=5,
        orders_x=(0,),
        orders_y=(0,),
    )
    tasks = sweep.cli(["--list-tasks"])
    assert len(tasks) == 5
    task_files = []
    for t_info in tasks:
        p = tmp_path / f"task_{t_info['index']}.npz"
        sweep.cli(["--task", str(t_info["index"]), "--output", str(p)])
        task_files.append(str(p))
    combined_path = tmp_path / "combined.npz"
    res_cli = sweep.cli(["--collect", *task_files, "--output", str(combined_path)])
    res_mp = sweep.run(processes=2)
    np.testing.assert_allclose(
        res_cli.resample()["R_00"], res_mp.resample()["R_00"], atol=1e-12
    )


def _mock_slab_builder(k_par, f_arr, pol, phi):
    """Module-level callable builder for multiprocessing / CLI tests."""
    d_slab = 0.09
    eps_slab = 2.5
    k0 = 2.0 * np.pi * f_arr / c
    s = k_par / k0
    prop = (s < 0.999)[:, None, None]
    th = np.arcsin(np.clip(s, 0.0, 0.999))
    rt = slab_rt(f_arr, th, d_slab, eps_slab, pol=pol)
    r_diag = (
        np.diag(rt["r"])[:, None, None]
        if rt["r"].ndim == 2
        else rt["r"][:, None, None]
    )
    t_diag = (
        np.diag(rt["t"])[:, None, None]
        if rt["t"].ndim == 2
        else rt["t"][:, None, None]
    )
    z = np.zeros_like(r_diag)
    return {
        "r_te": np.abs(r_diag) ** 2,
        "r_tm": np.zeros_like(np.abs(r_diag) ** 2),
        "t_te": np.abs(t_diag) ** 2,
        "t_tm": np.zeros_like(np.abs(t_diag) ** 2),
        "s_refl_te": r_diag,
        "s_refl_tm": z,
        "s_tran_te": t_diag,
        "s_tran_tm": z,
        "prop_refl": prop,
        "prop_tran": prop,
        "inc_power": np.ones(f_arr.size),
    }


def test_floquet_s_matrix_unitarity_and_reciprocity():
    """Generalized Floquet S-matrix unitarity (S^H S = I) and Lorentz reciprocity (S_ij = S_ji)."""
    lam0 = 0.30
    f0 = c / lam0
    freqs = np.array([f0])
    period = 1.65 * lam0  # Orders m = -1, 0, +1 all propagate well away from cutoff
    thickness = 0.054
    nx, nz = 24, 100
    zmin, zmax = -0.45, 0.45
    dz = (zmax - zmin) / nz

    # Smooth asymmetric lossless dielectric grating profile across the cell period
    def _fill_asymmetric_grating(solver, nx_cur=nx):
        x = np.mod(solver.grid.x[:-1], period) / period
        z = solver.grid.z[:-1]
        eps_x = (
            1.0
            + 0.65 * np.cos(2.0 * np.pi * x)
            + 0.35 * np.sin(4.0 * np.pi * x)
        )
        in_layer = (z >= -0.5 * thickness) & (z <= 0.5 * thickness)
        for comp in ("x", "y", "z"):
            for ix in range(nx_cur):
                solver.ieps[ix, :, in_layer, comp] = 1.0 / (epsilon_0 * eps_x[ix])

    kx0 = 2.0 * np.pi / lam0 * np.sin(np.deg2rad(5.0))
    G = 2.0 * np.pi / period

    def _run_port_excitation(
        k_bloch_x: float,
        mode: FloquetMode,
        nx_cur: int = nx,
        nz_cur: int = nz,
        n_pml_cur: int = 14,
    ):
        dz_cur = (zmax - zmin) / nz_cur
        grid = wakis.GridFIT3D(
            -0.5 * period,
            0.5 * period,
            -0.5 * dz_cur,
            0.5 * dz_cur,
            zmin,
            zmax,
            nx_cur,
            1,
            nz_cur,
            verbose=0,
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["bloch", "bloch", "cpml"],
            bc_high=["bloch", "bloch", "cpml"],
            n_pml=n_pml_cur,
            k_bloch=(-k_bloch_x, 0.0, 0.0),
            dtype=np.complex128,
            verbose=0,
        )
        _fill_asymmetric_grating(solver, nx_cur=nx_cur)

        kt_m = k_bloch_x + mode.m * G
        z_src = -0.22 if mode.port == 1 else 0.22
        comp = "y" if mode.pol == "TE" else "x"
        src = GaussianSource(frequency=f0, fwidth=0.06 * f0)
        sheet = CurrentSource(
            src,
            component=comp,
            center=(0.0, 0.0, z_src),
            size=(period, dz_cur, 0.0),
            amp_func=lambda x, y, z: np.exp(-1j * kt_m * x),
        )
        mon_b = wakis.DiffractionMonitor(
            freqs, center=(0.0, 0.0, -0.12), size=(period, dz_cur, 0.0)
        )
        mon_a = wakis.DiffractionMonitor(
            freqs, center=(0.0, 0.0, 0.12), size=(period, dz_cur, 0.0)
        )
        nt = int((src.end_time + 20.0 / f0) / solver.dt)
        solver.emsolve(nt, source=sheet, monitors=[mon_b, mon_a])
        p1 = FloquetPort(
            mon_b, side="below", reference_position=-0.04, port_id=1, dn=dz_cur
        )
        p2 = FloquetPort(
            mon_a, side="above", reference_position=0.04, port_id=2, dn=dz_cur
        )
        return [p1, p2]

    modes_te = [
        FloquetMode(port, m, 0, "TE") for port in (1, 2) for m in (-1, 0, 1)
    ]
    exc_pos = {m: _run_port_excitation(+kx0, m) for m in modes_te}
    s_pos, _ = floquet_s_matrix(exc_pos, modes=modes_te)

    # 1. Energy conservation (column norms sum_i |S_ij|^2 = 1) and full unitarity S^H S = I.
    # At the coarse baseline resolution (nx=24, nz=100, n_pml=14; dx ~ lam0/14.5,
    # dz ~ lam0/33, 6 cells across the grating thickness), the lossless column sums
    # reach 1.0013 (+1.3e-3 deviation from 1). This slight apparent gain comes from
    # two spatial/temporal discretization effects at the port monitors (placed at
    # z = +/-0.12 m, only 0.093 m = 0.31*lam0 from the grating faces z = +/-0.027 m):
    #   (a) Second-order FIT spatial dispersion across the 6-cell-thick grating layer
    #       and finite DFT window (20 / f0 after source end) shift the raw monitor
    #       flux ratio (Phi_above - Phi_below) / Phi_inc by +8.8e-4.
    #   (b) Evanescent higher harmonics (|m| >= 2) at 0.31*lam0 from the grating carry
    #       a small negative reactive cross-flux (-4.1e-4 of Phi_inc), so restricting
    #       S to the propagating subset m in (-1, 0, +1) adds +4.1e-4 (+8.8e-4 + 4.1e-4 = +1.3e-3).
    s_mat = s_pos[0]
    col_powers = np.sum(np.abs(s_mat) ** 2, axis=0)
    np.testing.assert_allclose(col_powers, np.ones(len(modes_te)), atol=2e-3)
    np.testing.assert_allclose(
        s_mat.conj().T @ s_mat, np.eye(len(modes_te)), atol=2e-3
    )

    # Verify that doubling the grid resolution (nx=48, nz=200, n_pml=20) reduces
    # the column-sum discretization error by more than 3x (from 1.29e-3 to 4.2e-4).
    m_col = FloquetMode(1, -1, 0, "TE")
    ports_fine = _run_port_excitation(+kx0, m_col, nx_cur=48, nz_cur=200, n_pml_cur=20)
    s_col_fine = s_parameters_from_ports(ports_fine, incident_mode=m_col, modes=modes_te)
    col_power_coarse = float(col_powers[modes_te.index(m_col)])
    col_power_fine = float( sum(abs(s_col_fine[m_i][0]) ** 2 for m_i in modes_te) )
    err_coarse = abs(col_power_coarse - 1.0)
    err_fine = abs(col_power_fine - 1.0)
    assert err_fine < 0.5 * err_coarse
    np.testing.assert_allclose(col_power_fine, 1.0, atol=5e-4)

    # 2. Lorentz reciprocity: S_{i, j}(+kx0) = S_{-j, -i}(-kx0)
    m_inc = FloquetMode(1, 0, 0, "TE")
    m_out = FloquetMode(2, -1, 0, "TE")
    ports_neg = _run_port_excitation(-kx0, m_out.reciprocal_partner())
    s_col_neg = s_parameters_from_ports(
        ports_neg, incident_mode=m_out.reciprocal_partner(), modes=modes_te
    )
    s_21_pos = s_pos[0, modes_te.index(m_out), modes_te.index(m_inc)]
    s_12_neg = s_col_neg[m_inc.reciprocal_partner()][0]
    np.testing.assert_allclose(s_21_pos, s_12_neg, atol=5e-3)

    # Verify reciprocal_s_matrix maps S(-kx0) column entries to S(+kx0) row entries
    s_neg_partial = np.zeros_like(s_pos)
    j_rec = modes_te.index(m_out.reciprocal_partner())
    for i_idx, m_i in enumerate(modes_te):
        s_neg_partial[:, i_idx, j_rec] = s_col_neg[m_i]
    s_rec_from_neg = reciprocal_s_matrix(s_neg_partial, modes_te)
    np.testing.assert_allclose(
        s_pos[:, modes_te.index(m_out), :],
        s_rec_from_neg[:, modes_te.index(m_out), :],
        atol=5e-3,
    )


def test_grating_oblique_rcwa_crosscheck():
    """Cross-check oblique lamellar grating efficiencies over a band vs 1D RCWA and FMMAX."""
    lam0 = 0.30
    f0 = c / lam0
    period = 1.5 * lam0
    height = 0.5 * lam0
    eps_sub = 2.25
    eps_tooth = 4.0

    # 1. Verify rcwa_1d_grating matches the published FMMAX 1.7.1 benchmark at 15 deg
    x_fine = np.linspace(0.0, period, 1024, endpoint=False)
    eps_prof = np.where(x_fine < 0.5 * period, eps_tooth, 1.0)
    rcwa_f0 = rcwa_1d_grating(
        [f0],
        np.deg2rad(15.0),
        period,
        height,
        eps_prof,
        eps_below=eps_sub,
        eps_above=1.0,
        pol="TE",
        n_harmonics=25,
        orders=(-2, -1, 0, 1),
    )
    np.testing.assert_allclose(
        rcwa_f0["R_orders"][0], [0.02005, 0.07602, 0.24072, 0.16239], atol=2e-4
    )
    np.testing.assert_allclose(
        rcwa_f0["T_orders"][0, :3], [0.04229, 0.44821, 0.01033], atol=2e-4
    )

    # 2. Broadband constant-angle Wakis sweep vs RCWA on a dielectric lamellar grating
    period_band = 1.40 * lam0
    freqs = np.linspace(0.94 * f0, 1.06 * f0, 5)
    theta_inc = np.deg2rad(10.0)
    h_grating = 0.25 * lam0
    eps_ridge = 2.25
    nx, nz = 28, 120
    zmin, zmax = -0.45, 0.45
    dz = (zmax - zmin) / nz

    def _lamellar_eps(x, z):
        tooth = (np.mod(x, period_band) < 0.5 * period_band) & (z > 0.0) & (z < h_grating)
        return np.where(tooth, eps_ridge, 1.0)

    def _fill_lamellar(solver):
        grid, d = solver.grid, 1e-6 * lam0
        nodes = (grid.x[:-1], grid.y[:-1], grid.z[:-1])
        for ci, comp in enumerate("xyz"):
            pos = [
                (n + np.diff(g) / 2) if k == ci else n
                for k, (n, g) in enumerate(zip(nodes, (grid.x, grid.y, grid.z)))
            ]
            X, _, Z = np.meshgrid(*pos, indexing="ij")
            offsets = [
                (ox, oz)
                for ox in ((0.0,) if ci == 0 else (-d, d))
                for oz in ((0.0,) if ci == 2 else (-d, d))
            ]
            eps = sum(_lamellar_eps(X + ox, Z + oz) for ox, oz in offsets) / len(offsets)
            solver.ieps[:, :, :, comp] = 1.0 / (epsilon_0 * eps)

    cell = FloquetCell(
        Lx=period_band,
        Ly=dz,
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=1,
        nz=nz,
        z_source=-0.27,
        z_refl=-0.15,
        z_tran=0.225,
        z_ref_below=0.0,
        z_ref_above=h_grating,
        eps_below=1.0,
        eps_above=1.0,
        orders_x=(-1, 0, 1),
        orders_y=(0,),
        n_pml=15,
        material_callback=_fill_lamellar,
        run_time_cycles=36.0,
    )

    sweep = AngleSweep(cell, freqs, [theta_inc], polarization="TE", n_k=5)
    res = sweep.run()
    out = res.resample()

    x_band = np.linspace(0.0, period_band, 1024, endpoint=False)
    eps_lam_prof = np.where(x_band < 0.5 * period_band, eps_ridge, 1.0)
    rcwa_band = rcwa_1d_grating(
        freqs,
        theta_inc,
        period_band,
        h_grating,
        eps_lam_prof,
        eps_below=1.0,
        eps_above=1.0,
        pol="TE",
        n_harmonics=15,
        orders=(-1, 0, 1),
    )

    np.testing.assert_allclose(
        out["T_order"][0, :, :, 0], rcwa_band["T_orders"], atol=1e-2
    )
    np.testing.assert_allclose(
        out["R_order"][0, :, :, 0], rcwa_band["R_orders"], atol=1e-2
    )
    np.testing.assert_allclose(
        out["R_total"][0] + out["T_total"][0], 1.0, atol=8e-3
    )


@pytest.mark.skipif(
    importlib.util.find_spec("fmmax") is None,
    reason="Optional dependency fmmax is not installed",
)
def test_grating_oblique_fmmax_live():
    """Live RCWA cross-check against fmmax when installed in the test environment."""
    import fmmax.basis as fbasis
    import fmmax.fields as ffields
    import fmmax.fmm as ffmm
    import fmmax.scattering as fscat
    import jax.numpy as jnp

    lam0 = 0.30
    f0 = c / lam0
    period = 1.5 * lam0
    h_grating = 0.25 * lam0
    theta_inc = np.deg2rad(15.0)
    freqs = np.linspace(0.90 * f0, 1.00 * f0, 3)

    x_fine = np.linspace(0.0, period, 1024, endpoint=False)
    eps_lam_prof = np.where(x_fine < 0.5 * period, 2.25, 1.0)
    rcwa_ref = rcwa_1d_grating(
        freqs,
        theta_inc,
        period,
        h_grating,
        eps_lam_prof,
        pol="TE",
        n_harmonics=15,
        orders=(-1, 0, 1),
    )

    primitive_lattice_vectors = fbasis.LatticeVectors(
        u=jnp.array([period, 0.0]), v=jnp.array([0.0, 1.0])
    )
    # 1D grating expansion along x (ny = 0) matching benchmarks/diffraction_fmmax_reference.py
    n_harm = 15
    coeffs = np.stack(
        [np.arange(-n_harm, n_harm + 1), np.zeros(2 * n_harm + 1, dtype=int)],
        axis=-1,
    )
    expansion = fbasis.Expansion(basis_coefficients=coeffs)
    n_terms = expansion.num_terms
    idx0 = int(np.flatnonzero(coeffs[:, 0] == 0)[0])

    for jf, freq in enumerate(freqs):
        wl = c / freq
        in_plane_wavevector = fbasis.plane_wave_in_plane_wavevector(
            wavelength=jnp.array(wl),
            polar_angle=jnp.array(theta_inc),
            azimuthal_angle=jnp.array(0.0),
            permittivity=jnp.array(1.0),
        )
        eig_bg = ffmm.eigensolve_isotropic_media(
            wavelength=jnp.array(wl),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=jnp.ones((1, 1), dtype=jnp.complex64),
            expansion=expansion,
            formulation=ffmm.Formulation.FFT,
        )
        eig_gr = ffmm.eigensolve_isotropic_media(
            wavelength=jnp.array(wl),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=jnp.asarray(eps_lam_prof[:, None], dtype=jnp.complex64),
            expansion=expansion,
            formulation=ffmm.Formulation.FFT,
        )
        s_mat = fscat.stack_s_matrix(
            layer_solve_results=[eig_bg, eig_gr, eig_bg],
            layer_thicknesses=[jnp.array(0.0), jnp.array(h_grating), jnp.array(0.0)],
        )
        # TE (E along y for the xz plane of incidence) is the first amplitude block
        # in fmmax, as in benchmarks/diffraction_fmmax_reference.py
        inc_fwd = (
            jnp.zeros((2 * n_terms, 1), dtype=jnp.complex64)
            .at[idx0, 0]
            .set(1.0)
        )
        inc_flux, _ = ffields.amplitude_poynting_flux(
            forward_amplitude=inc_fwd,
            backward_amplitude=jnp.zeros_like(inc_fwd),
            layer_solve_result=eig_bg,
        )
        inc_power = float(jnp.sum(inc_flux))

        bwd_amp = s_mat.s21 @ inc_fwd
        fwd_amp = s_mat.s11 @ inc_fwd
        _, bwd_flux = ffields.amplitude_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amp),
            backward_amplitude=bwd_amp,
            layer_solve_result=eig_bg,
        )
        fwd_flux, _ = ffields.amplitude_poynting_flux(
            forward_amplitude=fwd_amp,
            backward_amplitude=jnp.zeros_like(fwd_amp),
            layer_solve_result=eig_bg,
        )
        np.testing.assert_allclose(
            float(jnp.sum(fwd_flux)) / inc_power, rcwa_ref["T_total"][jf], atol=1e-3
        )
        np.testing.assert_allclose(
            float(jnp.sum(-bwd_flux)) / inc_power, rcwa_ref["R_total"][jf], atol=1e-3
        )
