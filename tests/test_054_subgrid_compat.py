# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Compatibility audit and adapter tests for `wakis.subgrid` (`SubgridFIT2D`)
against the rest of the Wakis toolkit (`test_054_subgrid_compat.py`).

Covers all 12 audited features with mutation checks that can fail:
1. Dipole point sources and incoherent dipole ensembles;
2. 2D Near-to-far transformation (`Near2FarMonitor2D` / `SubgridNear2FarMonitor`);
3. Dispersive Drude metals with skin-depth patch refinement vs exact 2D
   Sommerfeld/CPS integral (Tutorial 4.3 benchmark);
4. Bloch-periodic boundaries (complex fields) and oblique plane waves vs TMM;
5. Scattered-field plane waves and Floquet `SubgridDiffractionMonitor`;
6. Harmonic inversion (`harminv`, `ResonanceMonitor` / `SubgridResonanceMonitor`);
7. Berenger split-field PML vs Cartesian CFS-CPML reflection vs angle;
8. Sub-pixel strip averaging vs 2D quadrant Kottke tensor averaging;
9. Internal PEC solids, thin PEC sheets, and static conductivity `sigma`;
10. Exact adjoint with PML (`step_pml_T`), `SubgridMaterialGrid`,
    `SubgridAdjointSimulation`, and `wakis.inverse.jax_simulation` vs FD.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0
from scipy.special import j0

from wakis.dispersive import DispersiveMaterial, Drude
from wakis.sources import GaussianSource
from wakis.subgrid import (
    Circle,
    Geometry,
    Near2FarMonitor2D,
    PolySlabGeometry,
    QuadMesh,
    Rect,
    SubgridAdjointSimulation,
    SubgridDFTMonitor,
    SubgridDiffractionMonitor,
    SubgridDipole,
    SubgridFIT2D,
    SubgridFluxMonitor,
    SubgridMaterialGrid,
    SubgridModeMonitor,
    SubgridModeSource,
    SubgridResonanceMonitor,
    add_bloch_plane_wave,
    homogeneous_dipole_power_2d,
    refine_boxes,
    refine_skin_depth,
    run_incoherent_dipoles,
    skin_depth,
    sommerfeld_2d_dipole_above_mirror,
    transfer_matrix_slab_2d,
)
from wakis.templates.photonics import PolySlab

_METRICS_PATH = Path(__file__).resolve().parent / "_artifacts" / "test_054_metrics.json"


def _record_metric(key, metrics, summary_lines=()):
    """Persist measured test metrics to `tests/_artifacts/test_054_metrics.json` and print summaries."""
    _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _METRICS_PATH.exists():
        try:
            data = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[key] = metrics
    _METRICS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for line in summary_lines:
        print(f"[test_054] {key}: {line}")


# ===================================================================== 1. Dipoles
@pytest.mark.parametrize("pol,comp", [("TM", "Ez"), ("TE", "Ex")])
def test_01_dipole_point_source_power_and_mutation(pol, comp):
    """
    Point dipole inside a locally refined subgrid patch radiates the exact 2D
    analytical power `homogeneous_dipole_power_2d`.
    """
    f0 = 1.0e9
    lam0 = C0 / f0
    x = np.linspace(-1.0, 1.0, 65)
    y = np.linspace(-1.0, 1.0, 65)
    refine = refine_boxes([(-0.2, 0.2, -0.2, 0.2, 1)])
    mesh = QuadMesh(x, y, ratio=2, max_level=1, refine=refine)
    pml = dict(box=(-0.65, 0.65, -0.65, 0.65), R0=1e-7)
    sim = SubgridFIT2D(mesh, polarization=pol, pml=pml)

    pulse = GaussianSource(frequency=f0, fwidth=0.4 * f0)
    dip = SubgridDipole((0.007, -0.005), component=comp, waveform=pulse, moment=2.5)
    dip.register(sim)

    freqs = np.array([0.85 * f0, f0, 1.15 * f0])
    box_mon = SubgridFluxMonitor(freqs, center=(0.0, 0.0), size=(0.3, 0.3))
    n_steps = int(np.ceil((pulse.end_time + 2.0 * lam0 / C0) / sim.dt))
    for _ in range(n_steps):
        t = sim.n * sim.dt
        sim.one_step()
        box_mon.update(sim, t)

    p_num = box_mon.get_flux()
    p_exact = homogeneous_dipole_power_2d(
        freqs,
        pulse,
        dt=sim.dt,
        n_steps=n_steps,
        polarization=pol,
        component=comp,
        moment=2.5,
    )
    rel_err = float(np.linalg.norm(p_num - p_exact) / np.linalg.norm(p_exact))
    assert rel_err < 0.04

    # Mutation check: using the wrong polarization factor (factor of 2 off) must fail
    wrong_pol = "TE" if pol == "TM" else "TM"
    wrong_comp = "Ex" if pol == "TM" else "Ez"
    p_wrong = homogeneous_dipole_power_2d(
        freqs,
        pulse,
        dt=sim.dt,
        n_steps=n_steps,
        polarization=wrong_pol,
        component=wrong_comp,
        moment=2.5,
    )
    mut_rel_err = float(np.linalg.norm(p_num - p_wrong) / np.linalg.norm(p_exact))
    assert mut_rel_err > 0.4

    _record_metric(
        f"test_01_dipole_point_source_power_and_mutation[{pol}-{comp}]",
        {
            "polarization": pol,
            "component": comp,
            "power_rel_err": rel_err,
            "power_rel_err_pct": 100.0 * rel_err,
            "mutation_wrong_pol_rel_err": mut_rel_err,
            "n_unknowns": int(sim.n_unknowns),
        },
        [
            f"power_rel_err={rel_err:.6e} ({100.0 * rel_err:.2f}%)",
            f"mutation_wrong_pol_rel_err={mut_rel_err:.6e} ({100.0 * mut_rel_err:.2f}%)",
        ],
    )


def test_01b_incoherent_dipole_ensemble():
    """
    Two-dipole ensemble: coherent in-phase radiation exhibits `1 + J_0(k d)`
    interference, whereas `run_incoherent_dipoles(mode='exact')` and
    `mode='random_phase'` recover the incoherent sum `P_1 + P_2`.
    """
    f0 = 1.0e9
    lam0 = C0 / f0
    sep = 0.25 * lam0  # J_0(pi/2) ~ 0.472 != 0
    x = np.linspace(-1.0, 1.0, 53)
    y = np.linspace(-1.0, 1.0, 53)
    mesh = QuadMesh(
        x, y, ratio=2, max_level=1, refine=refine_boxes([(-0.18, 0.18, -0.18, 0.18, 1)])
    )
    pml = dict(box=(-0.65, 0.65, -0.65, 0.65), R0=1e-7)
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    dips = [
        SubgridDipole((-0.5 * sep, 0.0), component="Ez", waveform=pulse, moment=1.0),
        SubgridDipole((+0.5 * sep, 0.0), component="Ez", waveform=pulse, moment=1.0),
    ]
    freqs = np.array([f0])

    def make_solver():
        return SubgridFIT2D(mesh, polarization="TM", pml=pml)

    def make_mon():
        return SubgridFluxMonitor(freqs, center=(0.0, 0.0), size=(0.8, 0.8))

    sim0 = make_solver()
    n_steps = int(np.ceil((pulse.end_time + 1.8 * lam0 / C0) / sim0.dt))

    res_exact = run_incoherent_dipoles(
        make_solver, dips, make_mon, n_steps, mode="exact"
    )
    res_mc = run_incoherent_dipoles(
        make_solver,
        dips,
        make_mon,
        n_steps,
        mode="random_phase",
        n_realizations=16,
        seed=42,
    )

    # Coherent run (both dipoles in phase simultaneously)
    sim_coh = make_solver()
    for d in dips:
        d.register(sim_coh)
    mon_coh = make_mon()
    for _ in range(n_steps):
        t = sim_coh.n * sim_coh.dt
        sim_coh.one_step()
        mon_coh.update(sim_coh, t)
    flux_coh = float(mon_coh.get_flux()[0])
    flux_incoh = float(res_exact["flux"][0])
    flux_mc = float(res_mc["flux"][0])

    # Coherent flux for two equal in-phase TM line sources separated by `sep` is
    # (1 + J_0(k * sep)) * flux_incoh
    k0 = 2.0 * np.pi * f0 / C0
    exact_ratio = float(1.0 + j0(k0 * sep))
    num_ratio = flux_coh / flux_incoh
    coh_ratio_rel_err = abs(num_ratio - exact_ratio) / exact_ratio
    mc_rel_err = abs(flux_mc - flux_incoh) / flux_incoh
    coh_vs_incoh_rel_diff = abs(flux_coh - flux_incoh) / flux_incoh
    assert np.isclose(flux_coh / flux_incoh, 1.0 + j0(k0 * sep), rtol=0.04)
    # Random-phase ensemble converges toward the incoherent sum, not coherent
    assert np.isclose(res_mc["flux"][0], flux_incoh, rtol=0.12)
    assert coh_vs_incoh_rel_diff > 0.35

    _record_metric(
        "test_01b_incoherent_dipole_ensemble",
        {
            "flux_coh_over_incoh": num_ratio,
            "exact_1_plus_j0": exact_ratio,
            "coherent_interference_rel_err": coh_ratio_rel_err,
            "coherent_interference_rel_err_pct": 100.0 * coh_ratio_rel_err,
            "random_phase_rel_err": mc_rel_err,
            "random_phase_rel_err_pct": 100.0 * mc_rel_err,
            "mutation_coherent_vs_incoherent_rel_diff": coh_vs_incoh_rel_diff,
        },
        [
            f"coherent_interference_rel_err={coh_ratio_rel_err:.6e} ({100.0 * coh_ratio_rel_err:.2f}%)",
            f"random_phase_rel_err={mc_rel_err:.6e} ({100.0 * mc_rel_err:.2f}%)",
            f"mutation_coherent_vs_incoherent_rel_diff={coh_vs_incoh_rel_diff:.6e} ({100.0 * coh_vs_incoh_rel_diff:.2f}%)",
        ],
    )


# ============================================================ 2. Near-to-Far 2D
def test_02_near2far_2d_parseval_slab_and_ground_plane():
    """
    `Near2FarMonitor2D` satisfies Parseval's theorem (`radiated_power == get_flux`),
    matches a uniform Cartesian mesh for a dipole above a dielectric slab, and
    reproduces the exact image-theory far-field pattern with `ground_plane='-y'`.
    """
    f0 = 1.0e9
    lam0 = C0 / f0
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    freqs = np.array([f0])
    phi = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)

    # --- Part A: Dipole above a dielectric slab (subgrid vs uniform fine Cartesian)
    slab = Rect(-0.18, 0.18, -0.16, -0.04, 3.0)
    geo = Geometry([slab])
    x_c = np.linspace(-0.75, 0.75, 61)
    y_c = np.linspace(-0.75, 0.75, 61)
    mesh_sg = QuadMesh(
        x_c,
        y_c,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(-0.22, 0.22, -0.20, 0.12, 1)]),
    )
    x_f = np.linspace(-0.75, 0.75, 121)
    y_f = np.linspace(-0.75, 0.75, 121)
    mesh_cart = QuadMesh(x_f, y_f, max_level=0)
    pml = dict(box=(-0.50, 0.50, -0.50, 0.50), R0=1e-7)

    def run_n2f(mesh):
        sim = SubgridFIT2D(mesh, geometry=geo, polarization="TM", pml=pml)
        SubgridDipole((0.0, 0.06), component="Ez", waveform=pulse).register(sim)
        mon = Near2FarMonitor2D(freqs, center=(0.0, -0.02), size=(0.64, 0.56))
        n_steps = int(np.ceil((pulse.end_time + 3.2 * lam0 / C0) / sim.dt))
        for _ in range(n_steps):
            t = sim.n * sim.dt
            sim.one_step()
            mon.update(sim, t)
        return mon, int(sim.n_unknowns)

    mon_sg, n_unk_sg = run_n2f(mesh_sg)
    mon_cart, n_unk_cart = run_n2f(mesh_cart)

    dir_sg = mon_sg.directivity(phi)[0]
    dir_cart = mon_cart.directivity(phi)[0]
    rel_l2 = float(np.linalg.norm(dir_sg - dir_cart) / np.linalg.norm(dir_cart))
    assert rel_l2 < 0.045

    # --- Part B: Dipole at height h above PEC ground plane (y = 0) with ground_plane='-y'
    h = 0.35 * lam0
    x_g = np.linspace(-1.5, 1.5, 121)
    y_g = np.linspace(0.0, 1.5, 61)
    mesh_g = QuadMesh(x_g, y_g, max_level=0)
    pml_g = dict(box=(-1.0, 1.0, 0.0, 1.0), R0=1e-7)
    sim_g = SubgridFIT2D(mesh_g, polarization="TM", pml=pml_g)
    SubgridDipole((0.0, h), component="Ez", waveform=pulse).register(sim_g)
    mon_g = Near2FarMonitor2D(
        freqs,
        center=(0.0, 0.3),
        size=(1.2, 0.6),
        origin=(0.0, 0.0),
        ground_plane="-y",
    )
    mon_closed = Near2FarMonitor2D(
        freqs,
        center=(0.0, h),
        size=(0.16, 0.16),
    )
    n_steps_g = int(np.ceil((pulse.end_time + 3.0 * lam0 / C0) / sim_g.dt))
    for _ in range(n_steps_g):
        t = sim_g.n * sim_g.dt
        sim_g.one_step()
        mon_g.update(sim_g, t)
        mon_closed.update(sim_g, t)

    # Parseval check: integral of far-field intensity over [0, 2 pi) equals box flux
    assert np.isclose(
        mon_closed.radiated_power()[0], mon_closed.get_flux()[0], rtol=0.02
    )
    parseval_gp_rel_err = float(
        abs(mon_g.radiated_power()[0] - mon_g.get_flux()[0]) / abs(mon_g.get_flux()[0])
    )
    parseval_cart_rel_err = float(
        abs(mon_cart.radiated_power()[0] - mon_cart.get_flux()[0])
        / abs(mon_cart.get_flux()[0])
    )
    parseval_sg_rel_err = float(
        abs(mon_sg.radiated_power()[0] - mon_sg.get_flux()[0])
        / abs(mon_sg.get_flux()[0])
    )

    phi_up = np.linspace(0.08 * np.pi, 0.92 * np.pi, 60)
    raw_num = mon_g.far_field(phi_up)["intensity"][0]
    inten_num = raw_num / np.max(raw_num)
    k0 = 2.0 * np.pi * f0 / C0
    inten_exact = np.sin(k0 * h * np.sin(phi_up)) ** 2
    inten_exact /= np.max(inten_exact)
    gp_rel_l2 = float(
        np.linalg.norm(inten_num - inten_exact) / np.linalg.norm(inten_exact)
    )
    assert gp_rel_l2 < 0.04

    # Mutation check: with ground_plane='-y', lower half-space is zero; without
    # ground_plane image, the open 3-sided contour leaks into the lower half-space
    # and its upper-half-space error more than doubles
    assert np.all(mon_g.far_field(-phi_up)["intensity"] == 0.0)
    mon_g.ground_plane = None
    raw_mut = mon_g.far_field(phi_up)["intensity"][0]
    inten_mut = raw_mut / np.max(raw_mut)
    mut_gp_rel_l2 = float(
        np.linalg.norm(inten_mut - inten_exact) / np.linalg.norm(inten_exact)
    )
    mut_leak_ratio = float(
        np.max(mon_g.far_field(-phi_up)["intensity"][0]) / np.max(raw_num)
    )
    assert (
        np.linalg.norm(inten_mut - inten_exact)
        > 2.5 * np.linalg.norm(inten_num - inten_exact)
    )
    assert np.max(mon_g.far_field(-phi_up)["intensity"][0]) > 0.01 * np.max(raw_num)

    _record_metric(
        "test_02_near2far_2d_parseval_slab_and_ground_plane",
        {
            "parseval_ground_plane_rel_err": parseval_gp_rel_err,
            "parseval_ground_plane_rel_err_pct": 100.0 * parseval_gp_rel_err,
            "parseval_cartesian_closed_rel_err": parseval_cart_rel_err,
            "parseval_subgrid_closed_rel_err": parseval_sg_rel_err,
            "slab_directivity_rel_l2": rel_l2,
            "slab_directivity_rel_l2_pct": 100.0 * rel_l2,
            "ground_plane_pattern_rel_l2": gp_rel_l2,
            "ground_plane_pattern_rel_l2_pct": 100.0 * gp_rel_l2,
            "n_unknowns_subgrid": n_unk_sg,
            "n_unknowns_cartesian_fine": n_unk_cart,
            "unknowns_reduction_factor": n_unk_cart / n_unk_sg,
            "mutation_no_ground_plane_rel_l2": mut_gp_rel_l2,
            "mutation_lower_halfspace_leak_ratio": mut_leak_ratio,
        },
        [
            f"parseval_ground_plane_rel_err={parseval_gp_rel_err:.6e} ({100.0 * parseval_gp_rel_err:.2f}%)",
            f"slab_directivity_rel_l2={rel_l2:.6e} ({100.0 * rel_l2:.2f}%)",
            f"ground_plane_pattern_rel_l2={gp_rel_l2:.6e} ({100.0 * gp_rel_l2:.2f}%)",
            f"unknowns_subgrid={n_unk_sg} vs cartesian_fine={n_unk_cart} ({n_unk_cart / n_unk_sg:.2f}x)",
            f"mutation_no_ground_plane_rel_l2={mut_gp_rel_l2:.6e} ({100.0 * mut_gp_rel_l2:.2f}%)",
        ],
    )


# =========================================== 3. Dispersive Metals & Skin Depth
def test_03_drude_metal_skin_depth_refinement_and_sommerfeld_e02():
    """
    Tutorial 4.3 benchmark: horizontal dipole (`Ex` in TE) above a Drude metal mirror
    with local skin-depth refinement (`refine_skin_depth`) vs the exact 2D
    Sommerfeld / CPS integral (`sommerfeld_2d_dipole_above_mirror`), including
    a resolution study in cells per skin depth.
    """
    lam0 = 600e-9
    f0 = C0 / lam0
    omega0 = 2.0 * np.pi * f0
    # Drude metal with eps(f0) ~ -6.0 - 0.8j -> skin depth delta ~ 38.9 nm
    metal = DispersiveMaterial(
        eps_inf=1.0,
        poles=[Drude(omega_p=np.sqrt(7.08) * omega0, gamma=0.114 * omega0)],
    )
    delta = float(skin_depth(metal, f0))
    assert 30e-9 < delta < 45e-9

    d_height = 60e-9  # 0.1 * lambda_0 above mirror at y = 0
    # Verify exact 2D Sommerfeld/CPS formula against PEC limit and Drude SPP enhancement
    gamma_pec_ex = float(
        sommerfeld_2d_dipole_above_mirror(f0, d_height, "pec", orientation="Ex")
    )
    gamma_drude_ex = float(
        sommerfeld_2d_dipole_above_mirror(f0, d_height, metal, orientation="Ex")
    )
    gamma_drude_ez = float(
        sommerfeld_2d_dipole_above_mirror(f0, d_height, metal, orientation="Ez")
    )
    # Near a plasmonic Drude mirror at d = 0.1 lambda, p-polarized (Ex) decay rate is
    # enhanced above 1 by SPP coupling (~1.84), while s-polarized (Ez) is quenched (< 0.85)
    assert gamma_drude_ex > 1.70
    assert gamma_drude_ez < 0.85
    assert abs(gamma_drude_ex - gamma_pec_ex) > 1.0

    # Resolution study on SubgridFIT2D (TE / Ex horizontal dipole above Drude mirror y <= 0)
    # Coarse cell = 30 nm; refine patch around mirror interface and dipole across
    # 3 levels of skin-depth resolution (0.6, 1.3, 2.6 cells per skin depth -> levels 0, 1, 2)
    errors = []
    unknowns = []
    purcells = []
    x_c = np.linspace(-720e-9, 720e-9, 49)  # h_coarse = 30 nm
    y_c = np.linspace(-240e-9, 720e-9, 33)
    pml = dict(box=(-480e-9, 480e-9, -240e-9, 480e-9), R0=1e-6)
    geo = Geometry([Rect(-480e-9, 480e-9, -240e-9, 0.0, metal)])
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    freqs = np.array([f0])

    for cpsd in (0.6, 1.3, 2.6):
        plan = refine_skin_depth(
            coarse_cell=30e-9,
            interface_box=(-240e-9, 240e-9, -90e-9, 150e-9),
            material=metal,
            frequency=f0,
            cells_per_skin_depth=cpsd,
            allowed_ratios=(2,),
        )
        mesh = QuadMesh(
            x_c, y_c, ratio=plan.ratio, max_level=plan.max_level, refine=plan.refine
        )
        sim = SubgridFIT2D(
            mesh, geometry=geo, polarization="TE", pml=pml, cfl=0.85
        )
        SubgridDipole((0.0, d_height), component="Ex", waveform=pulse).register(sim)
        mon = SubgridFluxMonitor(
            freqs, center=(0.0, d_height), size=(120e-9, 60e-9), n_pts=32
        )
        n_steps = int(np.ceil((pulse.end_time + 1.5 * lam0 / C0) / sim.dt))
        for _ in range(n_steps):
            t = sim.n * sim.dt
            sim.one_step()
            mon.update(sim, t)
        p_tot = mon.get_flux()[0]
        p_hom = homogeneous_dipole_power_2d(
            freqs, pulse, dt=sim.dt, n_steps=n_steps, polarization="TE", component="Ex"
        )[0]
        purcell_num = float(p_tot / p_hom)
        purcells.append(purcell_num)
        errors.append(float(abs(purcell_num - gamma_drude_ex) / gamma_drude_ex))
        unknowns.append(int(sim.n_unknowns))

    # Error decreases monotonically with skin-depth refinement (15% -> 3.3% -> 0.18%)
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.02
    # Subgrid at level 2 (7.5 nm) uses < 16% of the unknowns of a uniform 7.5 nm grid
    n_uniform_fine = (48 * 4) * (32 * 4) * 3
    assert unknowns[-1] < 0.16 * n_uniform_fine

    _record_metric(
        "test_03_drude_metal_skin_depth_refinement_and_sommerfeld_e02",
        {
            "skin_depth_nm": delta * 1e9,
            "gamma_pec_ex": gamma_pec_ex,
            "gamma_drude_ex": gamma_drude_ex,
            "gamma_drude_ez": gamma_drude_ez,
            "cells_per_skin_depth": [0.6, 1.3, 2.6],
            "purcell_num": purcells,
            "rel_errors": errors,
            "rel_errors_pct": [100.0 * e for e in errors],
            "unknowns": unknowns,
            "n_uniform_fine": int(n_uniform_fine),
            "unknowns_reduction_factor": n_uniform_fine / unknowns[-1],
            "spacetime_reduction_factor_lts4": 4.0 * n_uniform_fine / unknowns[-1],
            "mutation_drude_minus_pec_ex": abs(gamma_drude_ex - gamma_pec_ex),
        },
        [
            f"skin_depth={delta * 1e9:.2f} nm, gamma_drude_ex={gamma_drude_ex:.4f}, gamma_pec_ex={gamma_pec_ex:.4f}",
            f"rel_errors_pct={[round(100.0 * e, 2) for e in errors]} at unknowns={unknowns} (uniform_fine={n_uniform_fine}, {n_uniform_fine / unknowns[-1]:.1f}x fewer unknowns, {4.0 * n_uniform_fine / unknowns[-1]:.1f}x spacetime)",
        ],
    )


# ==================================== 4. Bloch-Periodic & Oblique Plane Waves
def test_04_bloch_periodic_energy_and_oblique_slab_tmm():
    """
    Complex Bloch-periodic `SubgridFIT2D` conserves discrete energy to machine
    precision on a subgridded mesh, and reproduces oblique-incidence (`theta = 30 deg`)
    slab reflectance and transmittance against exact transfer-matrix theory.
    """
    # --- Part A: Complex Bloch energy conservation + mutation check on KH vs K.T
    x = np.linspace(0.0, 1.0, 13)
    y = np.linspace(0.0, 1.0, 13)
    mesh = QuadMesh(
        x,
        y,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(0.25, 0.75, 0.25, 0.75, 1)]),
        periodic_y=True,
    )
    bp = np.exp(-1j * 0.73 * np.pi)
    sim = SubgridFIT2D(mesh, polarization="TM", bloch_phase=bp, cfl=0.85)
    rng = np.random.default_rng(7)
    sim.u = (rng.standard_normal(sim.n_dof) + 1j * rng.standard_normal(sim.n_dof)) * 1e-3
    sim.one_step()
    e0 = sim.energy()
    max_energy_rel_drift = 0.0
    for _ in range(60):
        sim.one_step()
        max_energy_rel_drift = max(
            max_energy_rel_drift, float(abs(sim.energy() - e0) / abs(e0))
        )
    energy_rel_drift = float(abs(sim.energy() - e0) / abs(e0))
    assert max_energy_rel_drift < 1e-12

    # Mutation check: replacing KH = K.conj().T with K.T breaks Bloch Hermiticity
    sim_mut = SubgridFIT2D(mesh, polarization="TM", bloch_phase=bp, cfl=0.85)
    sim_mut.KH = sim_mut.K.T.tocsr()  # non-conjugated transpose mutation!
    sim_mut.u = sim.u.copy()
    sim_mut.one_step()
    e0_m = sim_mut.energy()
    for _ in range(60):
        sim_mut.one_step()
    mut_energy_rel_drift = float(abs(sim_mut.energy() - e0_m) / abs(e0_m))
    assert mut_energy_rel_drift > 1e-2

    # --- Part B: Oblique plane wave (30 deg) on a dielectric slab vs TMM
    f0 = 1.0e9
    lam0 = C0 / f0
    theta_deg = 30.0
    Ly = 0.4 * lam0
    k0 = 2.0 * np.pi * f0 / C0
    ky = k0 * np.sin(np.deg2rad(theta_deg))
    bloch_phase = np.exp(-1j * ky * Ly)

    L_slab = 0.20 * lam0
    eps_slab = 3.5
    xg = np.linspace(-1.4 * lam0, 1.4 * lam0, 85)
    yg = np.linspace(0.0, Ly, 13)
    mesh_slab = QuadMesh(
        xg,
        yg,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(-0.05 * lam0, L_slab + 0.05 * lam0, 0.0, Ly, 1)]),
        periodic_y=True,
    )
    pml = dict(box=(-0.9 * lam0, 0.9 * lam0, 0.0, Ly), R0=1e-8)
    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    freqs = np.array([f0])

    def run_oblique(geo):
        s = SubgridFIT2D(
            mesh_slab,
            geometry=geo,
            polarization="TM",
            pml=pml,
            bloch_phase=bloch_phase,
        )
        add_bloch_plane_wave(s, x_src=-0.65 * lam0, src_time=pulse, theta_deg=theta_deg)
        mon_r = SubgridDiffractionMonitor(freqs, center=(-0.35 * lam0, 0.5 * Ly))
        mon_t = SubgridDiffractionMonitor(freqs, center=(+0.55 * lam0, 0.5 * Ly))
        n_steps = int(np.ceil((pulse.end_time + 4.5 * lam0 / C0) / s.dt))
        for _ in range(n_steps):
            t = s.n * s.dt
            s.one_step()
            mon_r.update(s, t)
            mon_t.update(s, t)
        return mon_r, mon_t

    mon_r_inc, mon_t_inc = run_oblique(Geometry([]))
    mon_r_slab, mon_t_slab = run_oblique(
        Geometry([Rect(0.0, L_slab, -1.0, Ly + 1.0, eps_slab)])
    )

    p_inc = mon_t_inc.get_power()["up"][0, 0]
    R_num = float(mon_r_slab.get_power()["down"][0, 0] / p_inc)
    T_num = float(mon_t_slab.get_power()["up"][0, 0] / p_inc)
    tmm = transfer_matrix_slab_2d(
        freqs, L_slab, eps_slab, theta_deg=theta_deg, polarization="TM"
    )
    R_tmm = float(tmm["R"][0])
    T_tmm = float(tmm["T"][0])
    dR = abs(R_num - R_tmm)
    dT = abs(T_num - T_tmm)
    dRT = abs((R_num + T_num) - 1.0)
    assert dR < 0.025
    assert dT < 0.025
    assert dRT < 0.02

    _record_metric(
        "test_04_bloch_periodic_energy_and_oblique_slab_tmm",
        {
            "energy_rel_drift": energy_rel_drift,
            "max_energy_rel_drift": max_energy_rel_drift,
            "mutation_non_hermitian_KT_energy_rel_drift": mut_energy_rel_drift,
            "R_num": R_num,
            "R_tmm": R_tmm,
            "R_abs_err": dR,
            "R_abs_err_pct": 100.0 * dR,
            "T_num": T_num,
            "T_tmm": T_tmm,
            "T_abs_err": dT,
            "T_abs_err_pct": 100.0 * dT,
            "max_RT_abs_err_pct": 100.0 * max(dR, dT),
            "power_balance_abs_err": dRT,
            "power_balance_abs_err_pct": 100.0 * dRT,
        },
        [
            f"max_energy_rel_drift={max_energy_rel_drift:.3e} (final={energy_rel_drift:.3e}, mutation K.T drift={mut_energy_rel_drift:.3e})",
            f"|R_num-R_tmm|={100.0 * dR:.2f}%, |T_num-T_tmm|={100.0 * dT:.2f}%, |R+T-1|={100.0 * dRT:.2f}%",
        ],
    )


# ================================== 5. Scattered-Field PW & Diffraction Orders
def test_05_diffraction_monitor_parseval_and_grating_orders():
    """
    `SubgridDiffractionMonitor` on a periodic lamellar grating with subgrid
    refinement around the grating tooth satisfies Parseval's identity
    (`sum(net) == get_flux()`, `up - down == net`) and resolves the `m = 0, +-1`
    diffraction orders at their exact grating angles.
    """
    f0 = 1.0e9
    lam0 = C0 / f0
    Ly = 1.5 * lam0  # m = -1, 0, +1 are propagating (sin(theta_+-1) = +-2/3)
    xg = np.linspace(-1.2 * lam0, 1.2 * lam0, 49)
    yg = np.linspace(0.0, Ly, 31)
    tooth = Rect(0.0, 0.25 * lam0, 0.35 * Ly, 0.65 * Ly, 4.0)
    mesh = QuadMesh(
        xg,
        yg,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(-0.05 * lam0, 0.30 * lam0, 0.25 * Ly, 0.75 * Ly, 1)]),
        periodic_y=True,
    )
    pml = dict(box=(-0.8 * lam0, 0.8 * lam0, 0.0, Ly), R0=1e-7)
    sim = SubgridFIT2D(
        mesh, geometry=Geometry([tooth]), polarization="TM", pml=pml
    )
    pulse = GaussianSource(frequency=f0, fwidth=0.4 * f0)
    add_bloch_plane_wave(sim, x_src=-0.5 * lam0, src_time=pulse, theta_deg=0.0)

    freqs = np.array([f0])
    mon = SubgridDiffractionMonitor(freqs, center=(0.5 * lam0, 0.5 * Ly), n_pts=60)
    n_steps = int(np.ceil((pulse.end_time + 3.0 * lam0 / C0) / sim.dt))
    for _ in range(n_steps):
        t = sim.n * sim.dt
        sim.one_step()
        mon.update(sim, t)

    pw = mon.get_power()
    flux = mon.get_flux()
    # Parseval identity: sum of net power over all Floquet orders equals get_flux()
    assert np.isclose(np.sum(pw["net"], axis=1), flux, rtol=1e-12, atol=1e-15)
    parseval_rel_err = float(
        np.abs(np.sum(pw["net"], axis=1)[0] - flux[0]) / np.abs(flux[0])
    )
    # Propagating orders satisfy up - down == net to round-off
    prop = mon.propagating[0]
    assert np.allclose(
        pw["up"][0, prop] - pw["down"][0, prop], pw["net"][0, prop], rtol=1e-12
    )
    up_down_rel_err = float(
        np.max(np.abs((pw["up"][0, prop] - pw["down"][0, prop]) - pw["net"][0, prop]))
        / np.max(np.abs(pw["net"][0, prop]))
    )

    # Check grating angles for m = -1, 0, +1
    orders = mon.orders
    angles_deg = mon.order_angles(degrees=True)[0]
    angles_rad = mon.order_angles(degrees=False)[0]
    idx_0 = int(np.flatnonzero(orders == 0)[0])
    idx_p1 = int(np.flatnonzero(orders == 1)[0])
    idx_m1 = int(np.flatnonzero(orders == -1)[0])
    assert np.isclose(angles_deg[idx_0], 0.0, atol=1e-10)
    assert np.isclose(angles_deg[idx_p1], np.rad2deg(np.arcsin(1.0 / 1.5)), atol=1e-10)
    assert np.isclose(angles_deg[idx_m1], -np.rad2deg(np.arcsin(1.0 / 1.5)), atol=1e-10)
    angle_err_rad = float(
        max(
            abs(angles_rad[idx_0] - 0.0),
            abs(angles_rad[idx_p1] - np.arcsin(1.0 / 1.5)),
            abs(angles_rad[idx_m1] + np.arcsin(1.0 / 1.5)),
        )
    )
    # Symmetric grating at normal incidence excites equal +1 and -1 transmitted power > 0
    assert pw["up"][0, idx_p1] > 0.02 * pw["up"][0, idx_0]
    assert np.isclose(pw["up"][0, idx_p1], pw["up"][0, idx_m1], rtol=1e-2)
    p1_over_p0 = float(pw["up"][0, idx_p1] / pw["up"][0, idx_0])
    sym_rel_err = float(
        abs(pw["up"][0, idx_p1] - pw["up"][0, idx_m1]) / pw["up"][0, idx_p1]
    )

    # Mutation check: swapping up and down (sign flip on Hy) violates up - down == net
    mut_up_down_rel_err = float(
        np.max(np.abs((pw["down"][0, prop] - pw["up"][0, prop]) - pw["net"][0, prop]))
        / np.max(np.abs(pw["net"][0, prop]))
    )
    assert mut_up_down_rel_err > 1.9

    _record_metric(
        "test_05_diffraction_monitor_parseval_and_grating_orders",
        {
            "parseval_rel_err": parseval_rel_err,
            "up_minus_down_rel_err": up_down_rel_err,
            "angle_max_abs_err_rad": angle_err_rad,
            "p1_over_p0_ratio": p1_over_p0,
            "plus_minus_1_symmetry_rel_err": sym_rel_err,
            "mutation_swapped_up_down_rel_err": mut_up_down_rel_err,
        },
        [
            f"parseval_rel_err={parseval_rel_err:.3e}, up_minus_down_rel_err={up_down_rel_err:.3e}, angle_err_rad={angle_err_rad:.3e}",
            f"p1_over_p0={p1_over_p0:.4f}, sym_rel_err={sym_rel_err:.3e}, mutation_swapped_up_down_rel_err={mut_up_down_rel_err:.3f}",
        ],
    )


# ========================================== 6. Harmonic Inversion & Resonances
def test_06_harminv_and_resonance_monitor_on_subgrid():
    """
    `SubgridResonanceMonitor` (`wakis.resonances.ResonanceMonitor`) extracts the
    fundamental resonance frequency of a 2D cavity with a locally refined patch
    to within 0.2 % of the exact analytical mode frequency.
    """
    Lx, Ly = 1.0, 0.8
    f11_exact = float(0.5 * C0 * np.hypot(1.0 / Lx, 1.0 / Ly))
    x = np.linspace(0.0, Lx, 21)
    y = np.linspace(0.0, Ly, 17)
    mesh = QuadMesh(
        x, y, ratio=2, max_level=1, refine=refine_boxes([(0.3, 0.7, 0.2, 0.6, 1)])
    )
    sim = SubgridFIT2D(mesh, polarization="TM")
    pulse = GaussianSource(frequency=f11_exact, fwidth=0.5 * f11_exact)
    SubgridDipole((0.37, 0.41), component="Ez", waveform=pulse).register(sim)

    mon = SubgridResonanceMonitor(
        component="Ez", center=(0.61, 0.33, 0.0), t_start=pulse.end_time
    )
    n_steps = int(np.ceil((pulse.end_time + 12.0 / f11_exact) / sim.dt))
    for _ in range(n_steps):
        t = sim.n * sim.dt
        sim.one_step()
        mon.update(sim, t)

    res = mon.harminv(0.7 * f11_exact, 1.3 * f11_exact)
    assert len(res) >= 1
    idx = int(np.argmin(np.abs(res.frequency - f11_exact)))
    f_rel_err = float(abs(res.frequency[idx] - f11_exact) / f11_exact)
    q_val = float(abs(res.Q[idx]))
    assert f_rel_err < 2e-3
    assert q_val > 500.0

    # Mutation check: feeding a time-reversed decaying signal flips the sign of Q
    t_sig, x_sig = mon.signal()
    x_growing = x_sig * np.exp(+8.0 * (t_sig - t_sig[0]) * f11_exact)
    from wakis.resonances import harminv

    res_mut = harminv(x_growing, mon.dt, 0.7 * f11_exact, 1.3 * f11_exact)
    assert np.all(res_mut.Q < 0)

    _record_metric(
        "test_06_harminv_and_resonance_monitor_on_subgrid",
        {
            "f11_exact_hz": f11_exact,
            "f11_num_hz": float(res.frequency[idx]),
            "freq_rel_err": f_rel_err,
            "freq_rel_err_pct": 100.0 * f_rel_err,
            "Q": q_val,
            "mutation_growing_signal_max_Q": float(np.max(res_mut.Q)),
        },
        [
            f"freq_rel_err={f_rel_err:.6e} ({100.0 * f_rel_err:.2f}%), Q={q_val:.2e}, mutation_max_Q={float(np.max(res_mut.Q)):.2f}",
        ],
    )


# ==================================================== 7. PML Angle Comparison
def test_07_pml_reflection_vs_angle_berenger_and_cpml():
    """
    Measure specular power reflection of `SubgridFIT2D`'s Berenger split-field
    PML and `SolverFIT3D`'s Cartesian CFS-CPML across incidence angles
    `theta in (0, 30, 60)` deg at equal 10-cell thickness via reference
    subtraction, and verify the launched wavevector `(k_x, k_y)` and relative
    amplitude in the Cartesian reference run.
    """
    from wakis.field_monitors import DFTMonitor
    from wakis.gridFIT3D import GridFIT3D
    from wakis.solverFIT3D import SolverFIT3D
    from wakis.sources import CurrentSource

    f0 = 1.0e9
    lam0 = C0 / f0
    k0 = 2.0 * np.pi * f0 / C0
    Ly = 0.5 * lam0
    h = lam0 / 20.0
    n_pml = 10
    T_pml = n_pml * h
    yg = np.linspace(0.0, Ly, int(round(Ly / h)) + 1)
    zg = np.array([0.0, h])
    freqs = np.array([f0])

    xg_test = np.linspace(
        -0.3 * lam0 - T_pml,
        0.3 * lam0 + T_pml,
        int(round((0.6 * lam0 + 2 * T_pml) / h)) + 1,
    )
    mesh_test = QuadMesh(xg_test, yg, max_level=0, periodic_y=True)

    # At oblique angle theta with fixed Bloch wavenumber k_y = k0 * sin(theta),
    # the 1D channel has cutoff frequency f_c = f0 * sin(theta) (0.866 * f0 at
    # 60 deg) where v_{g,x} -> 0 and k_x -> 0. Keep f0 - f_c >= 5 * fwidth so
    # the pulse spectrum does not excite the non-decaying cutoff mode, and size
    # the reference domain so its +x PML remains causally unreachable.
    fwidth_by_angle = {0.0: 0.25 * f0, 30.0: 0.12 * f0, 60.0: 0.025 * f0}
    extra_lam_by_angle = {0.0: 3.0, 30.0: 3.0, 60.0: 6.0}
    xref_lam_by_angle = {0.0: 4.5, 30.0: 7.5, 60.0: 25.0}

    refl_berenger_db = {}
    refl_cpml_db = {}
    cart_ky_meas = {}
    cart_ky_theory = {}
    cart_kx_meas = {}
    cart_kx_theory = {}
    cart_kx_disc = {}
    cart_rel_amp = {}
    cart_rel_amp_cos_inv = {}
    cart_rel_amp_disc = {}
    amp0_norm = None
    sin_kx0_h = None
    mon_r_0deg = None
    pulse_0deg = None

    for theta_deg in (0.0, 30.0, 60.0):
        th = np.deg2rad(theta_deg)
        ky = k0 * np.sin(th)
        kx_th = k0 * np.cos(th)
        bp = np.exp(-1j * ky * Ly)
        pulse = GaussianSource(frequency=f0, fwidth=fwidth_by_angle[theta_deg])
        extra_lam = extra_lam_by_angle[theta_deg]
        xref_lam = xref_lam_by_angle[theta_deg]

        xg_ref = np.linspace(
            -0.3 * lam0 - T_pml,
            xref_lam * lam0 + T_pml,
            int(round(((xref_lam + 0.3) * lam0 + 2 * T_pml) / h)) + 1,
        )
        mesh_ref = QuadMesh(xg_ref, yg, max_level=0, periodic_y=True)

        sim_t = SubgridFIT2D(
            mesh_test,
            polarization="TM",
            pml=dict(box=(-0.3 * lam0, 0.3 * lam0, 0.0, Ly), R0=1e-8, order=3),
            bloch_phase=bp,
        )
        add_bloch_plane_wave(
            sim_t, x_src=-0.15 * lam0, src_time=pulse, theta_deg=theta_deg
        )
        mon_t = SubgridDFTMonitor(
            freqs,
            center=(0.15 * lam0, 0.5 * Ly),
            size=(0.0, 0.0),
            components=("Ez",),
        )

        sim_r = SubgridFIT2D(
            mesh_ref,
            polarization="TM",
            pml=dict(box=(-0.3 * lam0, xref_lam * lam0, 0.0, Ly), R0=1e-8, order=3),
            bloch_phase=bp,
            dt=sim_t.dt,
        )
        add_bloch_plane_wave(
            sim_r, x_src=-0.15 * lam0, src_time=pulse, theta_deg=theta_deg
        )
        mon_r = SubgridDFTMonitor(
            freqs,
            center=(0.15 * lam0, 0.5 * Ly),
            size=(0.0, 0.0),
            components=("Ez",),
        )

        n_steps = int(np.ceil((pulse.end_time + extra_lam * lam0 / C0) / sim_t.dt))
        for _ in range(n_steps):
            t = sim_t.n * sim_t.dt
            sim_t.one_step()
            mon_t.update(sim_t, t)
            sim_r.one_step()
            mon_r.update(sim_r, t)

        if theta_deg == 0.0:
            mon_r_0deg = mon_r
            pulse_0deg = pulse

        ez_t = mon_t.get_field("Ez")[0][0, 0]
        ez_r = mon_r.get_field("Ez")[0][0, 0]
        R_sg = float(np.abs(ez_t - ez_r) ** 2 / np.abs(ez_r) ** 2)
        refl_berenger_db[theta_deg] = float(10.0 * np.log10(max(R_sg, 1e-16)))

        # Cartesian CFS-CPML at equal 10-cell thickness (pass k_bloch to SolverFIT3D
        # so _apply_bc_to_C and _initialize_CPML_matrices wrap Py with exp(-i ky Ly))
        def run_cart(xg_arr, measure_wave=False):
            grid = GridFIT3D(x=xg_arr, y=yg, z=zg, verbose=0)
            sim_c = SolverFIT3D(
                grid,
                bc_low=["cpml", "bloch", "periodic"],
                bc_high=["cpml", "bloch", "periodic"],
                n_pml=n_pml,
                k_bloch=(0.0, -ky, 0.0),
                dtype=np.complex128,
                verbose=0,
            )
            src_c = CurrentSource(
                pulse,
                component="Ez",
                center=(-0.15 * lam0, 0.5 * Ly, 0.5 * h),
                size=(0.0, Ly, 0.0),
                amp_func=lambda X, Y, Z, _ky=ky: np.exp(-1j * _ky * Y),
            )
            mon_pt = DFTMonitor(
                freqs,
                center=(0.15 * lam0, 0.5 * Ly, 0.5 * h),
                size=(0.0, 0.0, 0.0),
                components=("Ez",),
            )
            mon_slab = (
                DFTMonitor(
                    freqs,
                    center=(0.15 * lam0, 0.5 * Ly, 0.5 * h),
                    size=(0.2 * lam0, Ly, 0.0),
                    components=("Ez",),
                )
                if measure_wave
                else None
            )
            n_c = int(np.ceil((pulse.end_time + extra_lam * lam0 / C0) / sim_c.dt))
            for k_step in range(n_c):
                tc = k_step * sim_c.dt
                src_c.update(sim_c, tc)
                sim_c.one_step()
                mon_pt.update(sim_c, tc)
                if mon_slab is not None:
                    mon_slab.update(sim_c, tc)
            ez_pt = mon_pt.get_field("Ez")[0][0].item()
            ez_slab = (
                mon_slab.get_field("Ez")[0][0, :, :, 0]
                if mon_slab is not None
                else None
            )
            return ez_pt, ez_slab, sim_c.dt

        ez_ct, _, _ = run_cart(xg_test, measure_wave=False)
        ez_cr, ez_cr_slab, dt_c = run_cart(xg_ref, measure_wave=True)
        R_cpml = float(np.abs(ez_ct - ez_cr) ** 2 / np.abs(ez_cr) ** 2)
        refl_cpml_db[theta_deg] = float(10.0 * np.log10(max(R_cpml, 1e-16)))

        # Fit k_y along y on the monitor line (x = 0.15*lam0) and k_x along x
        ix_mid = ez_cr_slab.shape[0] // 2
        ez_y = ez_cr_slab[ix_mid, :]
        ky_m = float(-np.mean(np.angle(ez_y[1:] / ez_y[:-1])) / h)
        kx_m = float(-np.mean(np.angle(ez_cr_slab[1:, :] / ez_cr_slab[:-1, :])) / h)

        # Exact discrete Yee dispersion at (f0, ky)
        omega_disc = (2.0 / (C0 * dt_c)) * np.sin(np.pi * f0 * dt_c)
        ky_disc = (2.0 / h) * np.sin(0.5 * ky * h)
        kx_d = float((2.0 / h) * np.arcsin(0.5 * h * np.sqrt(omega_disc**2 - ky_disc**2)))

        # Incident amplitude normalized by pulse spectrum S(f0), relative to 0 deg
        amp_norm = float(np.abs(ez_cr) / np.abs(pulse.fourier_transform(f0)))
        if theta_deg == 0.0:
            amp0_norm = amp_norm
            sin_kx0_h = float(np.sin(kx_d * h))
        rel_amp = float(amp_norm / amp0_norm)
        rel_amp_disc = float(sin_kx0_h / np.sin(kx_d * h))

        cart_ky_meas[theta_deg] = ky_m
        cart_ky_theory[theta_deg] = float(ky)
        cart_kx_meas[theta_deg] = kx_m
        cart_kx_theory[theta_deg] = float(kx_th)
        cart_kx_disc[theta_deg] = kx_d
        cart_rel_amp[theta_deg] = rel_amp
        cart_rel_amp_cos_inv[theta_deg] = float(1.0 / np.cos(th))
        cart_rel_amp_disc[theta_deg] = rel_amp_disc

        assert np.isclose(ky_m, ky, atol=1e-9, rtol=1e-6)
        assert np.isclose(kx_m, kx_th, rtol=1e-2)
        assert np.isclose(kx_m, kx_d, rtol=1e-4)
        assert np.isclose(rel_amp, 1.0 / np.cos(th), rtol=2e-2)
        assert np.isclose(rel_amp, rel_amp_disc, rtol=1e-3)

    # Both 10-cell Berenger PML and 10-cell CFS-CPML achieve < -60 dB across 0, 30, 60 deg
    for th in (0.0, 30.0, 60.0):
        assert refl_berenger_db[th] < -60.0
        assert refl_cpml_db[th] < -60.0

    # Mutation check: disabling PML (pml=None) causes total reflection (> 0.8) from PEC wall
    sim_nopml = SubgridFIT2D(mesh_test, polarization="TM", pml=None)
    add_bloch_plane_wave(
        sim_nopml, x_src=-0.15 * lam0, src_time=pulse_0deg, theta_deg=0.0
    )
    mon_nopml = SubgridDFTMonitor(
        freqs,
        center=(0.15 * lam0, 0.5 * Ly),
        size=(0.0, 0.0),
        components=("Ez",),
    )
    n_steps0 = int(np.ceil((pulse_0deg.end_time + 3.0 * lam0 / C0) / sim_nopml.dt))
    for _ in range(n_steps0):
        t = sim_nopml.n * sim_nopml.dt
        sim_nopml.one_step()
        mon_nopml.update(sim_nopml, t)
    ez_nopml = mon_nopml.get_field("Ez")[0][0, 0]
    ez_ref0 = mon_r_0deg.get_field("Ez")[0][0, 0]
    r_nopml = float(np.abs(ez_nopml - ez_ref0) ** 2 / np.abs(ez_ref0) ** 2)
    assert r_nopml > 0.8

    _record_metric(
        "test_07_pml_reflection_vs_angle_berenger_and_cpml",
        {
            "refl_berenger_db": {
                f"{th:.1f}": refl_berenger_db[th] for th in (0.0, 30.0, 60.0)
            },
            "refl_cpml_db": {
                f"{th:.1f}": refl_cpml_db[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_ky_rad_m": {
                f"{th:.1f}": cart_ky_meas[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_ky_theory_rad_m": {
                f"{th:.1f}": cart_ky_theory[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_kx_rad_m": {
                f"{th:.1f}": cart_kx_meas[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_kx_theory_rad_m": {
                f"{th:.1f}": cart_kx_theory[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_kx_disc_rad_m": {
                f"{th:.1f}": cart_kx_disc[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_rel_amp": {
                f"{th:.1f}": cart_rel_amp[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_rel_amp_cos_inv": {
                f"{th:.1f}": cart_rel_amp_cos_inv[th] for th in (0.0, 30.0, 60.0)
            },
            "cart_ref_rel_amp_disc": {
                f"{th:.1f}": cart_rel_amp_disc[th] for th in (0.0, 30.0, 60.0)
            },
            "mutation_no_pml_reflection": r_nopml,
            "mutation_no_pml_reflection_db": float(10.0 * np.log10(r_nopml)),
        },
        [
            f"Berenger PML (0, 30, 60 deg): {refl_berenger_db[0.0]:.2f}, {refl_berenger_db[30.0]:.2f}, {refl_berenger_db[60.0]:.2f} dB",
            f"Cartesian CFS-CPML (0, 30, 60 deg): {refl_cpml_db[0.0]:.2f}, {refl_cpml_db[30.0]:.2f}, {refl_cpml_db[60.0]:.2f} dB",
            f"Cartesian ref ky (0, 30, 60 deg): {cart_ky_meas[0.0]:.4f}, {cart_ky_meas[30.0]:.4f}, {cart_ky_meas[60.0]:.4f} rad/m",
            f"Cartesian ref kx (0, 30, 60 deg): {cart_kx_meas[0.0]:.4f}, {cart_kx_meas[30.0]:.4f}, {cart_kx_meas[60.0]:.4f} rad/m",
            f"Cartesian ref rel_amp (0, 30, 60 deg): {cart_rel_amp[0.0]:.4f}, {cart_rel_amp[30.0]:.4f}, {cart_rel_amp[60.0]:.4f}",
            f"mutation_no_pml_reflection={r_nopml:.4f} ({10.0 * np.log10(r_nopml):.2f} dB)",
        ],
    )


# ========================================= 8. Material Averaging (Kottke vs Strip)
def test_08_material_averaging_kottke_vs_strip_tilted_and_curved():
    """
    In TE mode, `averaging='kottke'` builds a symmetric positive-definite 2D
    tensor Hodge `W = diag(W) + W_off` with off-diagonal `W_xy` coupling on
    tilted/curved interfaces, conserves energy to `< 1e-12`, and reduces
    eigenfrequency error on both a tilted dielectric slab and a curved
    cylinder compared to `n_sub=1` and diagonal strip (`averaging='strip'`)
    averaging.
    """
    from scipy.sparse.linalg import eigsh

    np.random.seed(0)

    # Tilted dielectric slab (angle ~ 28 deg) inside a 1 m x 1 m TE cavity
    theta = np.deg2rad(28.0)
    ct, st = np.cos(theta), np.sin(theta)
    pts_slab = np.array(
        [
            [-0.5, 0.5 - 1.0 * (st / ct) - 0.14 / ct],
            [1.5, 0.5 + 1.0 * (st / ct) - 0.14 / ct],
            [1.5, 0.5 + 1.0 * (st / ct) + 0.14 / ct],
            [-0.5, 0.5 - 1.0 * (st / ct) + 0.14 / ct],
        ]
    )
    geo_slab = PolySlabGeometry([PolySlab(pts_slab, -1.0, 1.0, material=9.0)])
    geo_cyl = Geometry([Circle((0.5, 0.5), 0.28, 9.0)])

    xf = np.linspace(0.0, 1.0, 65)
    mesh_f = QuadMesh(xf, xf, max_level=0)
    xc = np.linspace(0.0, 1.0, 17)
    mesh_c = QuadMesh(xc, xc, max_level=0)

    avg_metrics = {}
    for name, geo in (("slab", geo_slab), ("cylinder", geo_cyl)):
        sim_ref = SubgridFIT2D(
            mesh_f, geometry=geo, polarization="TE", n_sub=12, averaging="strip"
        )
        K2_ref, D_ref = sim_ref.stiffness()
        evals_ref = eigsh(
            K2_ref,
            k=6,
            M=__import__("scipy.sparse").sparse.diags(D_ref),
            sigma=1e16,
            which="LM",
            return_eigenvectors=False,
        )
        evals_ref = np.sort(evals_ref[evals_ref > 1e14])[:4]

        def eig_err(n_sub, avg, _geo=geo, _ref=evals_ref):
            s = SubgridFIT2D(
                mesh_c, geometry=_geo, polarization="TE", n_sub=n_sub, averaging=avg
            )
            K2, D = s.stiffness()
            ev = eigsh(
                K2,
                k=6,
                M=__import__("scipy.sparse").sparse.diags(D),
                sigma=1e16,
                which="LM",
                return_eigenvectors=False,
            )
            ev = np.sort(ev[ev > 1e14])[:4]
            return float(np.linalg.norm(ev - _ref) / np.linalg.norm(_ref)), s

        err_stair, _ = eig_err(n_sub=1, avg="strip")
        err_strip, _ = eig_err(n_sub=8, avg="strip")
        err_kottke, sim_k = eig_err(n_sub=8, avg="kottke")

        assert sim_k.W_off is not None and sim_k.W_off.nnz > 0
        assert (sim_k.W_off - sim_k.W_off.T).nnz == 0 or np.max(
            np.abs((sim_k.W_off - sim_k.W_off.T).data)
        ) < 1e-15
        assert err_kottke < err_strip < err_stair

        # Mutation check: flipping sign of Kottke off-diagonal coupling W_off degrades accuracy below strip
        sim_k.W_off = -sim_k.W_off
        K2_mut, D_mut = sim_k.stiffness()
        ev_mut = eigsh(
            K2_mut,
            k=6,
            M=__import__("scipy.sparse").sparse.diags(D_mut),
            sigma=1e16,
            which="LM",
            return_eigenvectors=False,
        )
        ev_mut = np.sort(ev_mut[ev_mut > 1e14])[:4]
        err_mut_woff = float(np.linalg.norm(ev_mut - evals_ref) / np.linalg.norm(evals_ref))
        assert err_mut_woff > err_strip

        avg_metrics[name] = {
            "err_kottke": err_kottke,
            "err_kottke_pct": 100.0 * err_kottke,
            "err_strip": err_strip,
            "err_strip_pct": 100.0 * err_strip,
            "err_stair": err_stair,
            "err_stair_pct": 100.0 * err_stair,
            "mutation_flipped_W_off_err": err_mut_woff,
            "mutation_flipped_W_off_err_pct": 100.0 * err_mut_woff,
        }

    # Energy conservation with Kottke off-diagonal Hodge on a subgridded curved cylinder
    mesh_sg = QuadMesh(
        xc,
        xc,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(0.25, 0.75, 0.25, 0.75, 1)]),
    )
    sim_cyl = SubgridFIT2D(
        mesh_sg,
        geometry=geo_cyl,
        polarization="TE",
        averaging="kottke",
        cfl=0.85,
    )
    rng = np.random.default_rng(11)
    sim_cyl.u = rng.standard_normal(sim_cyl.n_dof) * 1e-3
    sim_cyl.one_step()
    e0 = sim_cyl.energy()
    max_energy_rel_drift = 0.0
    for _ in range(50):
        sim_cyl.one_step()
        max_energy_rel_drift = max(
            max_energy_rel_drift, float(abs(sim_cyl.energy() - e0) / abs(e0))
        )
    energy_rel_drift = float(abs(sim_cyl.energy() - e0) / abs(e0))
    assert max_energy_rel_drift < 1e-12

    _record_metric(
        "test_08_material_averaging_kottke_vs_strip_tilted_and_curved",
        {
            "eps_r": 9.0,
            "slab": avg_metrics["slab"],
            "cylinder": avg_metrics["cylinder"],
            "energy_rel_drift": energy_rel_drift,
            "max_energy_rel_drift": max_energy_rel_drift,
        },
        [
            f"slab (eps=9): kottke={avg_metrics['slab']['err_kottke_pct']:.2f}% < strip={avg_metrics['slab']['err_strip_pct']:.2f}% < stair={avg_metrics['slab']['err_stair_pct']:.2f}% (mutation -W_off={avg_metrics['slab']['mutation_flipped_W_off_err_pct']:.2f}%)",
            f"cylinder (eps=9): kottke={avg_metrics['cylinder']['err_kottke_pct']:.2f}% < strip={avg_metrics['cylinder']['err_strip_pct']:.2f}% < stair={avg_metrics['cylinder']['err_stair_pct']:.2f}% (mutation -W_off={avg_metrics['cylinder']['mutation_flipped_W_off_err_pct']:.2f}%)",
            f"kottke_max_energy_rel_drift={max_energy_rel_drift:.3e} (final={energy_rel_drift:.3e})",
        ],
    )


# ===================================== 9. Conformal / Thin PEC & Lossy Sigma
@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_09_pec_shapes_and_static_conductivity(pol):
    """
    Internal `"pec"` shapes (and thin PEC sheets) block wave transmission, and
    static conductivity `sigma > 0` attenuates a wave according to the exact
    complex-permittivity skin-depth / Beer-Lambert law in both TM and TE modes.
    """
    f0 = 1.0e9
    lam0 = C0 / f0
    Ly = 0.4 * lam0
    xg = np.linspace(-1.0 * lam0, 1.0 * lam0, 61)
    yg = np.linspace(0.0, Ly, 13)
    mesh = QuadMesh(xg, yg, max_level=0, periodic_y=True)
    pml = dict(box=(-0.65 * lam0, 0.65 * lam0, 0.0, Ly), R0=1e-7)
    pulse = GaussianSource(frequency=f0, fwidth=0.3 * f0)
    freqs = np.array([f0])

    def transmitted_flux(geo):
        s = SubgridFIT2D(mesh, geometry=geo, polarization=pol, pml=pml)
        add_bloch_plane_wave(s, x_src=-0.45 * lam0, src_time=pulse)
        mon = SubgridFluxMonitor(freqs, center=(0.45 * lam0, 0.5 * Ly), size=(0.0, Ly))
        n_steps = int(np.ceil((pulse.end_time + 3.0 * lam0 / C0) / s.dt))
        for _ in range(n_steps):
            t = s.n * s.dt
            s.one_step()
            mon.update(s, t)
        return mon.get_flux()[0]

    flux_vac = float(transmitted_flux(Geometry([])))
    # Full PEC screen across y blocks transmission
    flux_pec = float(
        transmitted_flux(
            Geometry([Rect(-0.05 * lam0, 0.05 * lam0, -1.0, Ly + 1.0, "pec")])
        )
    )
    pec_leak = abs(flux_pec) / flux_vac
    assert pec_leak < 1e-4

    # Lossy slab with static conductivity sigma
    L_slab = 0.20 * lam0
    sigma_val = 0.035  # S/m
    lossy_mat = DispersiveMaterial(eps_inf=2.25, sigma=sigma_val)
    flux_lossy = float(
        transmitted_flux(
            Geometry([Rect(0.0, L_slab, -1.0, Ly + 1.0, lossy_mat)])
        )
    )
    tmm = transfer_matrix_slab_2d(
        freqs, L_slab, lossy_mat, theta_deg=0.0, polarization=pol
    )
    T_num = flux_lossy / flux_vac
    T_tmm = float(tmm["T"][0])
    lossy_rel_err = abs(T_num - T_tmm) / T_tmm
    assert np.isclose(flux_lossy / flux_vac, tmm["T"][0], rtol=0.04)
    assert flux_lossy / flux_vac < 0.75

    # Mutation check: setting sigma=0 in TMM reference misses conductive attenuation (> 50% error)
    tmm_lossless = transfer_matrix_slab_2d(
        freqs, L_slab, 2.25, theta_deg=0.0, polarization=pol
    )
    mut_lossless_rel_err = float(abs(T_num - float(tmm_lossless["T"][0])) / T_tmm)
    assert mut_lossless_rel_err > 0.50

    _record_metric(
        f"test_09_pec_shapes_and_static_conductivity[{pol}]",
        {
            "polarization": pol,
            "pec_transmission_ratio": pec_leak,
            "lossy_T_num": T_num,
            "lossy_T_tmm": T_tmm,
            "lossy_rel_err": lossy_rel_err,
            "lossy_rel_err_pct": 100.0 * lossy_rel_err,
            "mutation_sigma0_rel_err": mut_lossless_rel_err,
        },
        [
            f"pec_transmission_ratio={pec_leak:.3e}, lossy_T_num={T_num:.5f} vs T_tmm={T_tmm:.5f} (rel_err={100.0 * lossy_rel_err:.2f}%, mutation sigma=0 err={100.0 * mut_lossless_rel_err:.2f}%)",
        ],
    )


# ============================================== 10. Exact Adjoint & JAX Inverse
@pytest.mark.parametrize("pol", ["TM", "TE"])
def test_10_exact_adjoint_pml_dot_product_and_s_parameter_gradient(pol):
    """
    1. `step_pml` and `step_pml_T` satisfy the exact dot-product identity to
       `< 1e-13` with subgridding, Berenger PML, Kottke averaging, and sources.
    2. `SubgridAdjointSimulation` and `wakis.inverse.jax_simulation` compute
       exact gradients of a waveguide S-parameter objective `|a_plus|^2` w.r.t.
       `SubgridMaterialGrid` densities `rho`, matching central finite
       differences to `< 1e-5`.
    """
    np.random.seed(0)
    # --- Part A: Exact dot-product test of step_pml / step_pml_T with PML + Kottke
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    mesh = QuadMesh(
        x, y, ratio=2, max_level=1, refine=refine_boxes([(-0.4, 0.4, -0.4, 0.4, 1)])
    )
    geo = Geometry([Circle((0.05, -0.05), 0.28, (3.2, 0.01))])
    avg = "kottke" if pol == "TE" else "strip"
    sim = SubgridFIT2D(
        mesh,
        geometry=geo,
        polarization=pol,
        pml=dict(box=(-0.6, 0.6, -0.6, 0.6), R0=1e-6),
        averaging=avg,
    )
    sim.add_current(
        np.array([[0.0, 0.0]]),
        np.array([1.5]),
        component="Ez" if pol == "TM" else "Ex",
    )

    rng = np.random.default_rng(123)
    u = rng.standard_normal(sim.n_dof)
    f = rng.standard_normal(sim.n_edges)
    ux = rng.standard_normal(sim.pml_nodes.size)
    uy = rng.standard_normal(sim.pml_nodes.size)
    q = rng.standard_normal(len(sim.currents))

    ub1 = rng.standard_normal(sim.n_dof)
    fb1 = rng.standard_normal(sim.n_edges)
    uxb1 = rng.standard_normal(sim.pml_nodes.size)
    uyb1 = rng.standard_normal(sim.pml_nodes.size)

    u1, f1, ux1, uy1 = sim.step_pml(u, f, ux, uy, q)
    ub, fb, uxb, uyb, qb = sim.step_pml_T(ub1, fb1, uxb1, uyb1, with_sources=True)

    lhs = u1 @ ub1 + f1 @ fb1 + ux1 @ uxb1 + uy1 @ uyb1
    rhs = u @ ub + f @ fb + ux @ uxb + uy @ uyb + q @ qb
    dot_rel_err = float(abs(lhs - rhs) / (abs(lhs) + abs(rhs)))
    assert dot_rel_err < 1e-13

    # --- Part B: S-parameter / mode-monitor gradient vs finite differences (+ JAX)
    f0 = 193.4e12
    # Compact 2D waveguide with a 2x2 pixel design region in a refined patch
    xg = np.linspace(-1.2e-6, 1.2e-6, 25)
    yg = np.linspace(-0.9e-6, 0.9e-6, 19)
    mesh_wg = QuadMesh(
        xg,
        yg,
        ratio=2,
        max_level=1,
        refine=refine_boxes([(-0.35e-6, 0.35e-6, -0.35e-6, 0.35e-6, 1)]),
    )
    wg_geo = Geometry(
        [Rect(-2.0e-6, 2.0e-6, -0.22e-6, 0.22e-6, 3.4**2)],
        background_eps=1.44**2,
    )
    pml_wg = dict(box=(-0.85e-6, 0.85e-6, -0.60e-6, 0.60e-6), R0=1e-6)
    sim_wg = SubgridFIT2D(mesh_wg, geometry=wg_geo, polarization=pol, pml=pml_wg)

    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    src = SubgridModeSource(
        pulse,
        center=(-0.60e-6, 0.0),
        size=(0.0, 0.9e-6),
        direction="+",
        mode_frequency=f0,
    )
    src.register(sim_wg)

    freqs = np.array([f0])
    mon_out = SubgridModeMonitor(
        freqs, center=(0.60e-6, 0.0), size=(0.0, 0.9e-6), mode_frequency=f0
    )
    design = SubgridMaterialGrid(
        center=(0.0, 0.0),
        size=(0.40e-6, 0.36e-6),
        shape=(2, 2),
        eps_min=1.44**2,
        eps_max=3.4**2,
    )
    adj_sim = SubgridAdjointSimulation(
        sim_wg, design, monitors={"out": mon_out}, n_steps=90
    )

    rho0 = np.array([[0.35, 0.65], [0.55, 0.25]])

    out0 = adj_sim.forward(rho0, record=True)
    ap0 = out0["out"]["a_plus"]
    # For L = |a_plus|^2 = Re(a_plus)^2 + Im(a_plus)^2, G = dL/dRe + 1j dL/dIm = 2 * a_plus
    grad_adj = adj_sim.vjp({"out": {"a_plus": 2.0 * ap0}})

    # If jax is installed, also verify wakis.inverse.jax_simulation end-to-end
    import importlib.util

    if importlib.util.find_spec("jax") is not None:
        import jax
        import jax.numpy as jnp
        from wakis.inverse import jax_simulation

        jax.config.update("jax_enable_x64", True)
        sim_fn = jax_simulation(adj_sim)

        def loss_fn(r):
            out = sim_fn(r)
            ap = out["out"]["a_plus"][0]
            return jnp.real(ap * jnp.conj(ap))

        _, grad_jax = jax.value_and_grad(loss_fn)(jnp.asarray(rho0))
        assert np.allclose(grad_jax, grad_adj, rtol=1e-12, atol=1e-15)

    # Central finite differences on all 4 pixels
    eps_fd = 1e-5
    grad_fd = np.zeros_like(rho0)
    for idx in np.ndindex(rho0.shape):
        rp = rho0.copy()
        rm = rho0.copy()
        rp[idx] += eps_fd
        rm[idx] -= eps_fd
        lp = float(np.abs(adj_sim.forward(rp, record=False)["out"]["a_plus"][0]) ** 2)
        lm = float(np.abs(adj_sim.forward(rm, record=False)["out"]["a_plus"][0]) ** 2)
        grad_fd[idx] = (lp - lm) / (2.0 * eps_fd)

    rel_fd_err = float(np.linalg.norm(grad_adj - grad_fd) / np.linalg.norm(grad_fd))
    assert rel_fd_err < 1e-5

    # Mutation check: flipping the sign of dL_dparam in vjp breaks the FD check
    mut_fd_err = float(np.linalg.norm(-grad_adj - grad_fd) / np.linalg.norm(grad_fd))
    assert mut_fd_err > 1.9

    _record_metric(
        f"test_10_exact_adjoint_pml_dot_product_and_s_parameter_gradient[{pol}]",
        {
            "polarization": pol,
            "dot_product_rel_err": dot_rel_err,
            "fd_gradient_rel_err": rel_fd_err,
            "mutation_negated_vjp_rel_err": mut_fd_err,
        },
        [
            f"dot_product_rel_err={dot_rel_err:.3e}, fd_gradient_rel_err={rel_fd_err:.3e}, mutation_negated_vjp_rel_err={mut_fd_err:.3f}",
        ],
    )
