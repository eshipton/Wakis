# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Fast analytical, Wakis 2D FIT solver, and mutation tests for Tutorial 4.3:
`tutorials/04_radiation_and_antennas/04-03_dipole_near_mirror/`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0

from wakis import GridFIT3D, SolverFIT3D
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.shapes import assign_materials

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-03_dipole_near_mirror"
)


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "_emit02_helpers", TUTORIAL_DIR / "_helpers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_emit02_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_tutorial():
    spec = importlib.util.spec_from_file_location(
        "_emit02_tutorial", TUTORIAL_DIR / "dipole_near_mirror.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_emit02_tutorial"] = mod
    spec.loader.exec_module(mod)
    return mod


hlp = _load_helpers()
dnm = _load_tutorial()


def _free_space_power_2d(
    freq: float, s_f: complex, pol: str = "y", n_bg: float = 1.0
) -> float:
    """Exact 2D homogeneous-medium power per unit y-length on FluxMonitor.get_flux()/ly scale."""
    k = 2.0 * np.pi * n_bg * freq / c0
    eta = float(np.sqrt(mu0 / eps0)) / n_bg
    p0_te = 0.25 * k * eta * (np.abs(s_f) ** 2)
    return float(p0_te if pol == "y" else 0.5 * p0_te)


def test_analytic_pec_mirror_identities_and_integral():
    """Verify 2D PEC mirror Bessel limits, angular pattern integrals, and 2D vs 3D decay."""
    # 1. Exact kh -> 0 limits: TE_par -> 0, TM_par -> 0, TM_perp -> 2
    assert float(hlp.pec_mirror_purcell_2d(0.0, "y")) == pytest.approx(0.0, abs=1e-12)
    assert float(hlp.pec_mirror_purcell_2d(0.0, "x")) == pytest.approx(0.0, abs=1e-12)
    assert float(hlp.pec_mirror_purcell_2d(0.0, "z")) == pytest.approx(2.0, abs=1e-12)

    assert float(hlp.pec_mirror_purcell_3d(0.0, "par")) == pytest.approx(0.0, abs=1e-12)
    assert float(hlp.pec_mirror_purcell_3d(0.0, "perp")) == pytest.approx(
        2.0, abs=1e-12
    )

    # 2. Angular integral of (1/P'_0) dP'/dtheta over (-pi/2, pi/2) equals F_P(kh)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    for pol in ("y", "x", "z"):
        for kh in (0.7, 1.57079632679, 3.14159265359, 5.2):
            pat = hlp.pec_mirror_pattern_2d(theta, kh, pol)
            f_int = float(np.trapezoid(pat, theta))
            f_ana = float(hlp.pec_mirror_purcell_2d(kh, pol))
            assert f_int == pytest.approx(f_ana, rel=1e-5)

    # 3. 2D cylindrical O((kh)^-1/2) oscillation envelope decays slower than 3D spherical O((kh)^-1)
    kh_window = np.linspace(18.0, 22.0, 201)
    amp_2d = float(np.max(np.abs(hlp.pec_mirror_purcell_2d(kh_window, "y") - 1.0)))
    amp_3d = float(np.max(np.abs(hlp.pec_mirror_purcell_3d(kh_window, "par") - 1.0)))
    assert amp_2d > 2.2 * amp_3d


def test_cps_sommerfeld_pec_limit_and_drude_silver():
    """Verify CPS Sommerfeld solver in the PEC limit and on Drude silver at 600 nm."""
    kh_pts = np.array([0.8, 1.5708, 3.1416])
    eps_pec_like = -1.0e8 + 0.1j
    for pol in ("y", "x", "z"):
        cps = hlp.cps_sommerfeld_2d(kh_pts, eps_pec_like, pol=pol, u_max=18.0)
        f_pec = hlp.pec_mirror_purcell_2d(kh_pts, pol)
        np.testing.assert_allclose(cps["F_tot"], f_pec, rtol=3e-3)
        assert np.all(cps["eta_rad"] > 0.998)

    # Drude silver at lambda_0 = 600 nm: verify skin depth, SPP pole, and TM quenching
    eps_ag, _, delta_e = hlp.drude_silver_permittivity(600.0e-9)
    assert float(np.real(eps_ag)) == pytest.approx(-15.301, rel=2e-3)
    assert float(delta_e * 1e9) == pytest.approx(24.41, rel=2e-3)

    u_spp = float(np.real(np.sqrt(eps_ag / (eps_ag + 1.0))))
    assert 1.03 < u_spp < 1.04

    cps_te = hlp.cps_sommerfeld_2d(0.30, eps_ag, pol="y", u_max=25.0)
    cps_tm = hlp.cps_sommerfeld_2d(0.30, eps_ag, pol="z", u_max=25.0)
    # Near the silver surface (kh = 0.30), TM_perp strongly excites the SPP pole
    assert float(cps_tm["F_nonrad"][0]) > 20.0 * float(cps_te["F_nonrad"][0])
    assert float(cps_te["eta_rad"][0]) > 0.95
    assert float(cps_tm["eta_rad"][0]) < 0.40


def test_planar_cavity_dos_2d_vs_3d_and_tmm_rcled():
    """Verify 2D vs 3D planar PEC cavity DOS formulas and Lee et al. (2024) DBR TMM."""
    # 1. Sub-cutoff inhibition for d < 0.5 in TE_par and TM_par, while TM_perp has TEM_0 mode
    d_sub = np.array([0.20, 0.35, 0.48])
    assert np.allclose(hlp.planar_cavity_purcell_2d(d_sub, "y"), 0.0)
    assert np.allclose(hlp.planar_cavity_purcell_2d(d_sub, "x"), 0.0)
    assert np.allclose(hlp.planar_cavity_purcell_3d(d_sub, "par"), 0.0)
    np.testing.assert_allclose(
        hlp.planar_cavity_purcell_2d(d_sub, "z"), 2.0 / (np.pi * d_sub), rtol=1e-12
    )
    np.testing.assert_allclose(
        hlp.planar_cavity_purcell_3d(d_sub, "perp"), 1.0 / (2.0 * d_sub), rtol=1e-12
    )

    # 2. Abram et al. (1998) Eq. (7) 3D midplane step jump at d = 0.5+ equals 3/(2*0.5) = 3.0,
    #    whereas 2D Van Hove DOS diverges as 1/sqrt(d^2 - 0.25)
    assert float(hlp.planar_cavity_purcell_3d(0.5001, "par")) == pytest.approx(
        3.0, rel=1e-3
    )
    assert float(hlp.planar_cavity_purcell_2d(0.505, "y")) > 4.0

    # 3. TMM reflectance of Lee et al. (2024) blue InGaN DBRs at lambda_0 = 450 nm
    lam0 = 450.0e-9
    n_gan, n_npgan, n_sio2, n_ta2o5 = 2.39, 1.70, 1.46, 2.15
    bot_dbr = [(lam0 / (4 * n_npgan), n_npgan), (lam0 / (4 * n_gan), n_gan)] * 10
    top_dbr = [(lam0 / (4 * n_sio2), n_sio2), (lam0 / (4 * n_ta2o5), n_ta2o5)] * 5
    r_bot = float(
        hlp.tmm_stack_reflectance(lam0, bot_dbr, n_inc=n_gan, n_sub=n_gan)["R"][0]
    )
    r_top = float(
        hlp.tmm_stack_reflectance(lam0, top_dbr, n_inc=n_gan, n_sub=1.0)["R"][0]
    )
    assert r_bot > 0.99
    assert r_top > 0.95


def _simulate_pec_mirror_2d(
    h_rel: float, cpw: int = 16, ground_plane: str | None = "-z"
):
    """Helper running a fast 2D Wakis FIT line dipole at height h_rel * lambda_0 above PEC."""
    lam0 = 1.0e-6
    f0 = c0 / lam0
    dx = lam0 / cpw
    Ny = 3
    ly = dx
    x = np.arange(-20, 21) * dx
    y = np.arange(Ny + 1) * dx
    y_mid = 0.5 * Ny * dx
    z = np.arange(0, 34) * dx
    h_val = round(h_rel * cpw) * dx
    kh = (2.0 * np.pi / lam0) * h_val

    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "pec"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        use_stl=False,
        verbose=0,
    )
    pulse = GaussianSource(f0, 0.35 * f0, cutoff=3.8)
    Nt = int(np.ceil((pulse.end_time + 2.6 * lam0 / c0) / solver.dt))
    src = CurrentSource(
        pulse,
        component="y",
        center=(0.0, y_mid, h_val),
        size=(0.0, 1.0, 0.0),
        amplitude=1.0,
    )
    dft_pt = DFTMonitor(
        [f0], ["Ey"], center=(0.0, y_mid, h_val), size=(0.0, 0.0, 0.0)
    )
    box = FluxMonitor(
        [f0], center=(0.0, y_mid, h_val), size=(4 * dx, ly, 4 * dx)
    )
    z_bot_n2f = 0.0 if ground_plane == "-z" else 1.0 * dx
    z_top_n2f = round((h_rel + 0.50) * cpw) * dx
    n2f = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, y_mid, 0.5 * (z_bot_n2f + z_top_n2f)),
        size=(20 * dx, ly, z_top_n2f - z_bot_n2f),
        ground_plane=ground_plane,
    )
    solver.emsolve(Nt, source=[src], monitors=[dft_pt, box, n2f])

    s_f = pulse.fourier_transform(f0)
    p0 = _free_space_power_2d(f0, s_f, "y")
    e_src = complex(dft_pt.get_field("Ey")[0].squeeze())
    f_work = float(-np.real(e_src * np.conj(s_f)) / p0)
    f_flux = float((box.get_flux()[0] / ly) / p0)

    theta = np.linspace(-0.46 * np.pi, 0.46 * np.pi, 91)
    pat_num = hlp.farfield_2d_from_monitor(n2f, theta, ly=ly) / p0
    pat_ana = hlp.pec_mirror_pattern_2d(theta, kh, "y")
    return {
        "kh": kh,
        "f_work": f_work,
        "f_flux": f_flux,
        "f_ana": float(hlp.pec_mirror_purcell_2d(kh, "y")),
        "theta": theta,
        "pat_num": pat_num,
        "pat_ana": pat_ana,
    }


def test_wakis_2d_pec_mirror_drexhage_and_near2far():
    """Verify Wakis 2D FIT Purcell factor and Near2Far pattern for constructive/destructive heights."""
    r_con = _simulate_pec_mirror_2d(0.25, cpw=16, ground_plane="-z")
    r_des = _simulate_pec_mirror_2d(0.50, cpw=16, ground_plane="-z")

    # Constructive enhancement at h = 0.25 lambda_0 (F_P = 1.3042) vs destructive at h = 0.50 lambda_0 (F_P = 0.7797)
    assert r_con["f_work"] == pytest.approx(r_con["f_ana"], rel=0.03)
    assert r_con["f_flux"] == pytest.approx(r_con["f_ana"], rel=0.03)
    assert r_des["f_work"] == pytest.approx(r_des["f_ana"], rel=0.03)
    assert r_des["f_flux"] == pytest.approx(r_des["f_ana"], rel=0.03)
    assert r_con["f_flux"] > 1.55 * r_des["f_flux"]

    # Angular radiation pattern from Near2FarMonitor(ground_plane="-z") matches exact 2D pattern
    for res in (r_con, r_des):
        l2_err = float(
            np.sqrt(np.mean((res["pat_num"] - res["pat_ana"]) ** 2))
            / np.sqrt(np.mean(res["pat_ana"] ** 2))
        )
        assert l2_err < 0.045


def test_wakis_2d_planar_cavity_inhibition_and_kottke_subpixel():
    """Verify 2D planar PEC microcavity sub-cutoff inhibition and SlabXZ Kottke subpixel accuracy."""
    # 1. Subpixel dielectric conservation on off-grid DBR layers: Kottke vs staircase
    lam0 = 450.0e-9
    f0 = c0 / lam0
    n_gan, n_sio2 = 2.39, 1.46
    dz = lam0 / (14 * n_gan)
    z_surf = -0.172 * lam0
    d_hi = lam0 / (4.0 * n_gan)
    d_lo = lam0 / (4.0 * n_sio2)
    intervals = [
        (z_surf - 2 * d_hi - 2 * d_lo, z_surf - 2 * d_hi - d_lo, n_sio2**2),
        (z_surf - 2 * d_hi - d_lo, z_surf - d_hi - d_lo, n_gan**2),
        (z_surf - d_hi - d_lo, z_surf - d_hi, n_sio2**2),
        (z_surf - d_hi, z_surf, n_gan**2),
    ]
    exact_excess = sum((z1 - z0) * (eps_v - 1.0) for (z0, z1, eps_v) in intervals)

    errs_avg = {}
    for avg in ("kottke", "staircase"):
        z = np.arange(-30, 10) * dz
        x = np.arange(-3, 4) * dz
        y = np.arange(4) * dz
        grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["cpml", "periodic", "cpml"],
            bc_high=["cpml", "periodic", "cpml"],
            n_pml=6,
            use_stl=False,
            verbose=0,
        )
        shapes = [
            hlp.SlabXZ(
                -1.0,
                1.0,
                z0,
                z1 + (1.5 * dz if idx < len(intervals) - 1 else 0.0),
                material=eps_v,
                snap=False,
            )
            for idx, (z0, z1, eps_v) in enumerate(intervals)
        ]
        assign_materials(solver, shapes, f0, averaging=avg)
        ieps_x = np.asarray(solver.ieps[grid.Nx // 2, 1, :-1, "x"], dtype=float)
        eps_x = 1.0 / (eps0 * ieps_x)
        num_excess = float(np.sum(eps_x - 1.0) * dz)
        errs_avg[avg] = abs(num_excess - exact_excess) / exact_excess

    assert errs_avg["kottke"] < 1e-12
    assert errs_avg["staircase"] > 0.05

    # 2. Planar PEC cavity sub-cutoff inhibition (d = 0.32 < 0.5) vs above-cutoff (d = 0.80 > 0.5)
    n_cav = 2.4
    L_cav = 1.0e-6
    Nz = 16
    dz_c = L_cav / Nz
    x_c = np.arange(-30, 31) * dz_c
    y_c = np.arange(4) * dz_c
    y_mid_c = 1.5 * dz_c
    z_c = np.linspace(0.0, L_cav, Nz + 1)

    f_purcell = {}
    for d_val in (0.32, 0.80):
        fc = d_val * c0 / (n_cav * L_cav)
        grid_c = GridFIT3D(x=x_c, y=y_c, z=z_c, verbose=0)
        solver_c = SolverFIT3D(
            grid_c,
            bc_low=["cpml", "periodic", "pec"],
            bc_high=["cpml", "periodic", "pec"],
            bg=[n_cav**2, 1.0],
            n_pml=8,
            use_stl=False,
            verbose=0,
        )
        pulse_c = GaussianSource(fc, 0.20 * fc, cutoff=4.0)
        Nt_c = int(np.ceil((pulse_c.end_time + 11.0 * n_cav * L_cav / c0) / solver_c.dt))
        src_c = CurrentSource(
            pulse_c,
            component="y",
            center=(0.0, y_mid_c, 0.5 * L_cav),
            size=(0.0, 1.0, 0.0),
            amplitude=1.0,
        )
        dft_c = DFTMonitor(
            [fc], ["Ey"], center=(0.0, y_mid_c, 0.5 * L_cav), size=(0.0, 0.0, 0.0)
        )
        solver_c.emsolve(Nt_c, source=[src_c], monitors=[dft_c])
        s_fc = pulse_c.fourier_transform(fc)
        e_c = complex(dft_c.get_field("Ey")[0].squeeze())
        p0_c = _free_space_power_2d(fc, s_fc, "y")
        f_purcell[d_val] = max(0.0, float(-np.real(e_c * np.conj(s_fc)) / p0_c))

    assert f_purcell[0.32] < 0.02
    assert f_purcell[0.80] == pytest.approx(
        float(hlp.planar_cavity_purcell_2d(0.80, "y")), rel=0.04
    )


def _assert_task_mesh_rules(task_cfg: dict, full_mode: bool = False) -> dict:
    """Build only the mesh for ``task_cfg`` and assert resolution Rules 1-3."""
    mesh = dnm.build_task_mesh(task_cfg)
    if mesh.get("is_demo", False):
        return mesh
    # Rule 1: Never fewer than 10 cells per wavelength in highest-index material at lambda_min
    # (and >= 20 cpw for the primary --full physics tasks 0..3)
    min_cpw_req = (
        20.0 if (full_mode and int(task_cfg["index"]) < 4) else 10.0
    )
    assert mesh["cpw_worst"] >= min_cpw_req - 1e-6, (
        f"Task {task_cfg['name']} violates Rule 1: cpw_worst={mesh['cpw_worst']} < {min_cpw_req}"
    )
    # Rule 2: >= 4 cells across the smallest geometric feature (DBR layer or dipole-to-mirror gap)
    assert mesh["cells_across_feature"] >= 4.0 - 1e-6, (
        f"Task {task_cfg['name']} violates Rule 2: cells_across_feature={mesh['cells_across_feature']} < 4.0"
    )
    assert mesh["dx_min_nm"] <= mesh["min_feature_nm"] / 4.0 + 1e-6, (
        f"Task {task_cfg['name']} violates Rule 2: dx_min_nm={mesh['dx_min_nm']} > min_feature_nm/4"
    )
    # Rule 3: Metals/plasmonics require >= 4 cells per skin depth at non-demo levels
    if mesh["cells_per_skin_depth"] is not None:
        assert mesh["cells_per_skin_depth"] >= 4.0 - 1e-6, (
            f"Task {task_cfg['name']} violates Rule 3: cells_per_skin_depth={mesh['cells_per_skin_depth']} < 4.0"
        )
    assert mesh["n_cells"] > 0 and mesh["n_steps"] > 0
    return mesh


def test_task_table_resolution_rules_quick_and_full():
    """Verify every non-demo task in --quick and --full satisfies Rules 1-3 (mesh-only)."""
    for full_mode, quick_flag in ((False, True), (True, False)):
        tasks = dnm.get_task_table(quick=quick_flag)
        assert len(tasks) == 11
        for t_cfg in tasks:
            mesh = _assert_task_mesh_rules(t_cfg, full_mode=full_mode)
            # Also check that graded RCLED stack conserves exact dielectric integral on non-uniform grid
            if t_cfg["kind"] == "rcled" and quick_flag:
                grid = mesh["grid"]
                st = dnm._build_rcled_stack(d_opt=1.00, n_bot_pairs=4, n_top_pairs=2)
                dz_arr = np.diff(grid.z)
                tL_z = np.zeros(grid.Nz)
                tL_z[1:] = 0.5 * (dz_arr[:-1] + dz_arr[1:])
                tL_z[0] = 0.5 * dz_arr[0]
                Z = np.broadcast_to(grid.z[:-1].reshape(1, 1, -1), (grid.Nx, grid.Ny, grid.Nz))
                Hz = np.broadcast_to(tL_z.reshape(1, 1, -1), (grid.Nx, grid.Ny, grid.Nz))
                X = np.zeros_like(Z)
                Hx = np.ones_like(Z) * 15e-9
                z0, z1, nv = st["slabs"][1]  # 47.07 nm GaN quarter-wave layer on graded grid
                slab = hlp.SlabXZ(-1.0, 1.0, z0, z1, material=nv**2)
                w = slab.fill(X, X, Z, Hx, Hx, Hz)[0, 0, :]
                assert float(np.sum(w * tL_z)) == pytest.approx(z1 - z0, rel=1e-12)


def test_mutation_checks_provably_fail():
    """Mutation checks: mutating image-dipole sign, ground_plane, coherent averaging, or mesh resolution MUST fail."""
    # Mutation 1: Omitting ground_plane="-z" on Near2FarMonitor above a PEC mirror destroys
    # the destructive normal-incidence node at h = 0.50 lambda_0
    r_mut_no_ground = _simulate_pec_mirror_2d(0.50, cpw=12, ground_plane=None)
    with pytest.raises(AssertionError):
        l2_err = float(
            np.sqrt(
                np.mean((r_mut_no_ground["pat_num"] - r_mut_no_ground["pat_ana"]) ** 2)
            )
            / np.sqrt(np.mean(r_mut_no_ground["pat_ana"] ** 2))
        )
        assert l2_err < 0.05

    # Mutation 2: Swapping the PEC image-dipole sign (1 - J_0(2kh) -> 1 + J_0(2kh)) must fail
    r_con = _simulate_pec_mirror_2d(0.25, cpw=12, ground_plane="-z")
    f_mut_pmc = 2.0 - r_con["f_ana"]  # 1 + J_0(2kh) instead of 1 - J_0(2kh)
    with pytest.raises(AssertionError):
        assert r_con["f_flux"] == pytest.approx(f_mut_pmc, rel=0.05)

    # Mutation 3: Coherent field sum |sum_m E_m(theta)|^2 vs incoherent power sum sum_m |E_m(theta)|^2
    theta = np.linspace(-0.45 * np.pi, 0.45 * np.pi, 181)
    k0 = 2.0 * np.pi / 450.0e-9
    x_pos = np.linspace(-0.54e-6, 0.54e-6, 7)
    envelope = np.cos(theta)
    e_dipoles = np.array(
        [envelope * np.exp(-1j * k0 * xm * np.sin(theta)) for xm in x_pos]
    )
    pat_incoh = np.mean(np.abs(e_dipoles) ** 2, axis=0)
    pat_incoh /= np.max(pat_incoh)
    pat_coh_mut = np.abs(np.sum(e_dipoles, axis=0)) ** 2
    pat_coh_mut /= np.max(pat_coh_mut)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pat_coh_mut, pat_incoh, atol=0.10)

    # Mutation 4: Under-resolving an off-grid DBR task (cpw=12 -> 11.1 cpw >= 10, but 3.0 cells across 47.07 nm GaN layer)
    # must fail _assert_task_mesh_rules (violating Rule 2: < 4 cells across feature)
    mut_dbr_task = dict(dnm.get_task_table(quick=True)[4], cpw=12, is_demo=False)
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_task_mesh_rules(mut_dbr_task, full_mode=False)

    # Mutation 5: Under-resolving the PEC mirror (cpw=8 -> 6.4 cpw at lambda_min=800 nm)
    # must fail _assert_task_mesh_rules (violating Rule 1: < 10 cpw)
    mut_pec_task = dict(dnm.get_task_table(quick=True)[0], cpw=8, is_demo=False)
    with pytest.raises(AssertionError, match="Rule 1"):
        _assert_task_mesh_rules(mut_pec_task, full_mode=False)

    # Mutation 6: Under-resolving Drude silver (dz_ag_nm=10.0 -> 2.0 cells across h_min=20 nm
    # and 2.44 cells per skin depth) must fail _assert_task_mesh_rules
    mut_ag_task = dict(dnm.get_task_table(quick=True)[1], dz_ag_nm=10.0, is_demo=False)
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_task_mesh_rules(mut_ag_task, full_mode=False)


def test_committed_full_study_data_and_convergence():
    """Verify the committed --full study data in data/task_0..10.{json,npz} satisfies all resolution & physics checks."""
    import json

    data_dir = TUTORIAL_DIR / "data"
    full_tasks = dnm.get_task_table(quick=False)
    assert len(full_tasks) == 11

    for t_cfg in full_tasks:
        idx = int(t_cfg["index"])
        meta_path = data_dir / f"task_{idx}.json"
        npz_path = data_dir / f"task_{idx}.npz"
        assert meta_path.is_file(), f"Missing {meta_path}"
        assert npz_path.is_file(), f"Missing {npz_path}"
        meta = json.loads(meta_path.read_text())
        assert meta["task_index"] == idx
        assert meta["name"] == t_cfg["name"]
        assert meta["cpw"] == t_cfg["cpw"]
        assert meta["cells_across_feature"] >= 4.0 - 1e-6
        assert meta["dx_min_nm"] <= meta["min_feature_nm"] / 4.0 + 1e-6
        if meta["cells_per_skin_depth"] is not None:
            assert meta["cells_per_skin_depth"] >= 4.0 - 1e-6

    # Check Part A (task_0): all 9 monochromatic Purcell factors <= 1.2% error and Near2Far patterns <= 1.5% L2 error
    t0 = np.load(data_dir / "task_0.npz")
    for pol in ("y", "x", "z"):
        mono_ana = hlp.pec_mirror_purcell_2d(t0[f"mono_kh_{pol}"], pol)
        rel_err_mono = np.abs(t0[f"mono_flux_{pol}"] - mono_ana) / mono_ana
        assert np.all(rel_err_mono < 0.012)
        for tag in ("025", "050", "075"):
            p_num = t0[f"pat_num_{pol}_{tag}"]
            p_ana = t0[f"pat_ana_{pol}_{tag}"]
            l2 = float(np.sqrt(np.mean((p_num - p_ana) ** 2)) / np.sqrt(np.mean(p_ana**2)))
            assert l2 < 0.015

    # Check Part B (task_1): TE_par (Jy) mean CPS error < 0.5%
    t1 = np.load(data_dir / "task_1.npz")
    eps_ag, _, _ = hlp.drude_silver_permittivity(600.0e-9)
    cps_tot_y = hlp.cps_sommerfeld_2d(t1["kh_num_y"], eps_ag, pol="y")["F_tot"]
    err_ag_y = np.abs(t1["F_tot_num_y"] - cps_tot_y) / cps_tot_y
    assert float(np.mean(err_ag_y)) < 0.005

    # Check Part C (task_2): Planar PEC cavity sub-cutoff inhibition and smooth-band agreement
    t2 = np.load(data_dir / "task_2.npz")
    assert float(np.median(t2["F_sim_y"][t2["d_sim_y"] < 0.42])) < 0.02
    mask_mid = (t2["d_sim_y"] > 0.60) & (t2["d_sim_y"] < 1.40)
    f_cav_2d = hlp.planar_cavity_purcell_2d(t2["d_sim_y"][mask_mid], "y")
    err_cav_y = np.abs(t2["F_sim_y"][mask_mid] - f_cav_2d) / f_cav_2d
    assert float(np.mean(err_cav_y)) < 0.01

    # Check Convergence (tasks 4..10): Kottke fine -> reference step change < 0.25% and beats staircase by > 10x at fine
    t8 = np.load(data_dir / "task_8.npz")  # kottke_fine (cpw=32)
    t9 = np.load(data_dir / "task_9.npz")  # staircase_fine (cpw=32)
    t10 = np.load(data_dir / "task_10.npz")  # kottke_reference (cpw=48)
    f_ref_dbr = float(t10["f_dbr_at_f0"])
    err_kot_fine = abs(float(t8["f_dbr_at_f0"]) - f_ref_dbr) / f_ref_dbr
    err_st_fine = abs(float(t9["f_dbr_at_f0"]) - f_ref_dbr) / f_ref_dbr
    assert err_kot_fine < 0.0025
    assert err_st_fine > 10.0 * err_kot_fine

    # Mutation check on committed data: Perturbing monochromatic Purcell factor by +5% must fail the < 1.2% check
    with pytest.raises(AssertionError):
        mono_ana_y = hlp.pec_mirror_purcell_2d(t0["mono_kh_y"], "y")
        bad_flux = t0["mono_flux_y"] * 1.05
        assert np.all(np.abs(bad_flux - mono_ana_y) / mono_ana_y < 0.012)


def test_setup_schematic_and_all_figures_exist():
    """Verify that 00_setup_schematic.png and all 14 diagnostic figures exist and are non-empty."""
    fig_dir = TUTORIAL_DIR / "figures"
    expected_figures = [
        "00_setup_schematic.png",
        "01_geometry_and_mesh_overview.png",
        "02_pec_mirror_drexhage_purcell.png",
        "03_pec_mirror_angular_patterns.png",
        "04_pec_mirror_field_snapshots.png",
        "05_dipole_mirror_interference.gif",
        "06_drude_silver_drexhage_cps.png",
        "07_sommerfeld_spp_kspace_spectrum.png",
        "08_planar_cavity_purcell_2d_vs_3d.png",
        "09_cavity_cutoff_ringing_diagnostic.png",
        "10_rcled_dbr_reflectance_and_cavity_field.png",
        "11_rcled_angular_extraction_beaming.png",
        "12_incoherent_quantum_well_averaging.png",
        "13_incoherent_monte_carlo_convergence.png",
        "14_mesh_and_subpixel_convergence.png",
    ]
    for fname in expected_figures:
        p = fig_dir / fname
        assert p.is_file(), f"Missing expected figure: {p}"
        size_bytes = p.stat().st_size
        assert size_bytes > 10_000, f"Figure {p} is unexpectedly small ({size_bytes} bytes)"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, f"GIF {p} exceeds 5 MB limit ({size_bytes} bytes)"

    # Verify set_monitor_bounds helper and its mutation check
    fm = FluxMonitor([300.0e12], center=(1.0e-6, 2.0e-6, 3.0e-6), size=(0.4e-6, 0.2e-6, 0.8e-6))
    hlp.set_monitor_bounds(fm, label="TestFluxBox")
    assert fm.name == "TestFluxBox"
    assert np.allclose(fm.bounds, (0.8e-6, 1.2e-6, 1.9e-6, 2.1e-6, 2.6e-6, 3.4e-6))
    with pytest.raises(AssertionError):
        # Mutation check: omitting the 0.5 half-span factor must fail the bounds check
        bad_bounds = (
            fm.center[0] - fm.size[0],
            fm.center[0] + fm.size[0],
            fm.center[1] - fm.size[1],
            fm.center[1] + fm.size[1],
            fm.center[2] - fm.size[2],
            fm.center[2] + fm.size[2],
        )
        assert np.allclose(fm.bounds, bad_bounds)



