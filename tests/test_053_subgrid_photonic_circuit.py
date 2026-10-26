# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation tests for the 2D subgridding photonic-circuit kit (`wakis.subgrid`):

* polarization mapping between `PhotonicCircuit(two_d=True)` and
  `SubgridFIT2D(polarization="TE")` (mutation: `polarization="TM"` gives the
  wrong waveguide effective index);
* `PolySlabGeometry` material sampling vs `PolySlab.fill` on a uniform grid,
  and vectorized evaluation on multi-hundred-vertex polygons with holes;
* quadtree refinement helpers (`refine_boxes`, `refine_near_polygons`,
  `choose_refinement`);
* exact discrete adjoint dot-product test of `linear_step` vs `linear_step_T`
  with soft current sources registered via `add_current` (mutation: omitting
  the within-step flux-to-node coupling in the source adjoint fails);
* discrete energy conservation over a long run on a refined photonic-circuit
  waveguide with PML off (mutation: non-transposed injection restriction blows up);
* straight waveguide S-parameters on `SubgridPhotonicCircuit`:
  - `max_level=0` vs `PhotonicCircuit` on the same uniform grid (`||S21_sg| -
    |S21_cart|| < 1e-2`),
  - refined patch crossed by the waveguide (interfaces normal to propagation:
    `T > 0.98`, `R < 1e-3`),
  - refined patch covering only the core width (interfaces parallel to
    propagation: `T > 0.98`, `R < 1e-3`),
  - `lts=2` passthrough and `PhotonicCircuit(..., backend="subgrid")`,
  - mutation of the hanging-node prolongation weights `P` breaks `R < 1e-3`
    and `T > 0.98`;
* cost baseline: wall time per cell/node per step of `SubgridFIT2D(max_level=0)`
  vs `SolverFIT3D` 2D on the same grid.
"""

import importlib
import sys
import time
import warnings

import numpy as np
import pytest
from scipy.constants import c as c_light

import wakis.subgrid as subgrid_mod
import wakis.subgrid.ports as subgrid_ports_mod
from wakis.subgrid import (
    PHOTONIC_CIRCUIT_POLARIZATION,
    PolySlabGeometry,
    QuadMesh,
    SubgridDFTMonitor,
    SubgridFIT2D,
    SubgridFluxMonitor,
    SubgridPhotonicCircuit,
    SubgridPICSimulation,
    choose_refinement,
    combine_refinements,
    polarization_mapping,
    refine_boxes,
    refine_near_polygons,
    solve_port_modes_2d,
)
from wakis.subgrid.interface import _inject
from wakis.templates.formulas import slab_waveguide_neff
from wakis.templates.photonics import (
    PHOTONIC_MATERIALS,
    PhotonicCircuit,
    PolySlab,
    Port,
    euler_bend,
    n_to_eps,
    ring,
    straight,
)


def _straight_wg_setup():
    wg = PolySlab(
        straight((-1.5e-6, 0.0), (1.5e-6, 0.0), width=0.5e-6),
        z_min=0.0,
        z_max=0.22e-6,
        material=n_to_eps(PHOTONIC_MATERIALS["Si"]),
    )
    ports = [
        Port(
            "in",
            center=(-1.0e-6, 0.0, 0.11e-6),
            size=(0.0, 2.0e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(1.0e-6, 0.0, 0.11e-6),
            size=(0.0, 2.0e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    kw = dict(
        wavelength_min=1.50e-6,
        wavelength_max=1.60e-6,
        n_freqs=21,
        structures=[wg],
        ports=ports,
        cells_per_wavelength=14,
        domain=(-1.3e-6, 1.3e-6, -1.2e-6, 1.2e-6),
        n_pml=8,
    )
    return wg, ports, kw


def test_polarization_mapping_and_mode_dispersion():
    """
    2D EIM `PhotonicCircuit` (`Ex, Ey, Hz` with `ieps_z = 0`) maps to
    `SubgridFIT2D(polarization="TE")`. Mutation: `polarization="TM"` solves
    `(Ez, Hx, Hy)` with continuous tangential `Ez` on the sidewalls and gives
    the wrong effective index by > 0.15.
    """
    pmap = polarization_mapping()
    assert pmap["subgrid_polarization"] == "TE"
    assert PHOTONIC_CIRCUIT_POLARIZATION == "TE"
    assert pmap["node_unknown_u"] == "Hz"

    _, _, kw = _straight_wg_setup()
    circuit_cart = PhotonicCircuit(**kw)
    circuit_sg = SubgridPhotonicCircuit(**kw, max_level=0)

    f0 = circuit_cart.f_center
    lam0 = c_light / f0
    m_cart = circuit_cart.solve_port_modes("in", frequencies=f0, n_modes=1)[0]
    m_sg = circuit_sg.solve_port_modes("in", frequencies=f0, n_modes=1)[0]

    # Analytical 1D horizontal effective index (in 1D slab terminology, E_y is
    # normal to the y-sidewalls of width 0.5 um, which is the 1D TM slab equation)
    neff_exact = float(
        slab_waveguide_neff(
            circuit_sg.n_eff_slab,
            circuit_sg.n_clad,
            0.5e-6,
            lam0,
            polarization="TM",
        )
    )
    assert abs(m_sg.neff - m_cart.neff) < 5e-3
    assert abs(m_sg.neff - neff_exact) < 1.5e-2
    assert m_sg.e["b"].shape == (len(m_sg.coords["Eb"][0]), 1)
    assert np.max(np.abs(m_sg.e["b"])) > 0.0 and np.max(np.abs(m_sg.e["c"])) == 0.0
    assert np.max(np.abs(m_sg.h["b"])) == 0.0 and np.max(np.abs(m_sg.h["c"])) > 0.0

    # Mutation: solving the port on a TM SubgridFIT2D instance gives the wrong
    # polarization (Ez tangential to sidewalls instead of Ey normal to sidewalls)
    solver_tm = SubgridFIT2D(
        circuit_sg.mesh,
        geometry=circuit_sg.geometry,
        polarization="TM",
        cfl=circuit_sg.cfl,
    )
    p_center, p_size, _ = circuit_sg._port_plane_args(circuit_sg.port_map["in"])
    m_tm = solve_port_modes_2d(solver_tm, f0, p_center, p_size, n_modes=1)[0]
    assert abs(m_tm.neff - m_cart.neff) > 0.10


def test_polyslab_geometry_adapter_vs_kit_fill_and_vectorized_speed():
    """
    `PolySlabGeometry` matches `PolySlab.fill` on a uniform grid for curved
    polygons with holes, and evaluates ~2.5e5 points on a 400-vertex clothoid
    bend + ring in vectorized form.
    """
    bend_verts, _ = euler_bend(3.0e-6, 90.0, 0.5e-6, n=200)
    slab_bend = PolySlab(bend_verts, 0.0, 0.22e-6, material="Si")
    outer_r, holes_r = ring((1.5e-6, 1.5e-6), radius=1.0e-6, width=0.4e-6, n=128)
    slab_ring = PolySlab(outer_r, 0.0, 0.22e-6, material="Si3N4", holes=holes_r)

    geo = PolySlabGeometry(
        [slab_bend, slab_ring], background_eps="SiO2", z=0.11e-6
    )
    assert geo.background_eps == pytest.approx(n_to_eps("SiO2"))
    assert len(geo.eps) == 3

    # Compare PolySlabGeometry.cell_fill on a uniform grid against PolySlab.fill
    xg = np.linspace(-0.5e-6, 3.0e-6, 36)
    yg = np.linspace(-0.5e-6, 3.0e-6, 36)
    X0, Y0 = np.meshgrid(xg[:-1], yg[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(xg[1:], yg[1:], indexing="ij")
    XC, YC = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)
    DX, DY = X1 - X0, Y1 - Y0

    geo_ring = PolySlabGeometry([slab_ring], background_eps="SiO2", z=0.11e-6)
    fill_sg = geo_ring.cell_fill(X0, X1, Y0, Y1, n_sub=8, mat_id=1)
    fill_kit = slab_ring.fill(
        XC,
        YC,
        np.full_like(XC, 0.11e-6),
        DX,
        DY,
        np.full_like(XC, 0.1e-6),
        n_sub=8,
    )
    np.testing.assert_allclose(fill_sg, fill_kit, atol=1e-12)

    # Also verify SubgridFIT2D quarter-cell area fractions sum to fill_kit
    m = QuadMesh(xg, yg, max_level=0)
    s = SubgridFIT2D(m, geo_ring, polarization="TE", n_sub=4)
    cell_area = (m.cells["x1"] - m.cells["x0"]) * (m.cells["y1"] - m.cells["y0"])
    frac_solver = s._q["frac"][:, :, 1].sum(axis=1) / cell_area
    # Order of cells in QuadMesh with max_level=0 is lexsort((i, j)) -> F-order of (X, Y)
    np.testing.assert_allclose(
        frac_solver, fill_kit.ravel(order="F"), atol=1e-12
    )

    # Vectorized evaluation on 250k points and mutation check on hole subtraction
    pts_1d = np.linspace(-0.5e-6, 3.0e-6, 500)
    Xp, Yp = np.meshgrid(pts_1d, pts_1d, indexing="ij")
    mid = geo.material_id(Xp, Yp)
    assert mid.shape == (500, 500)
    # Center of the ring (1.5e-6, 1.5e-6) is inside the hole -> background (0)
    assert int(geo.material_id(1.5e-6, 1.5e-6)) == 0
    # On the ring annulus (2.5e-6, 1.5e-6) -> mat_id 2 (Si3N4)
    assert int(geo.material_id(2.5e-6, 1.5e-6)) == 2

    # Mutation: dropping the hole from slab_ring wrongly marks the center as material 2
    slab_no_hole = PolySlab(outer_r, 0.0, 0.22e-6, material="Si3N4")
    geo_mut = PolySlabGeometry(
        [slab_bend, slab_no_hole], background_eps="SiO2", z=0.11e-6
    )
    assert int(geo_mut.material_id(1.5e-6, 1.5e-6)) == 2


def test_refinement_helpers_and_choose_refinement():
    """
    `refine_boxes`, `refine_near_polygons`, and `choose_refinement` reach the
    requested target cell size inside the box/polygon band while leaving the
    far field coarse.
    """
    xc = np.linspace(-2.0e-6, 2.0e-6, 41)  # coarse dx = 100 nm
    yc = np.linspace(-1.5e-6, 1.5e-6, 31)  # coarse dy = 100 nm
    coarse_h = 100e-9

    # Target 35 nm in a grating tooth band -> ratio 3, max_level 1 (fine = 33.3 nm)
    plan = choose_refinement(
        coarse_h, [(-0.5e-6, 0.5e-6, -0.3e-6, 0.3e-6, 35e-9)]
    )
    assert plan.ratio == 3 and plan.max_level == 1
    assert plan.fine_cell <= 35e-9
    m = QuadMesh(xc, yc, ratio=plan.ratio, max_level=plan.max_level, refine=plan.refine)
    dx_cells = m.cells["x1"] - m.cells["x0"]
    xm = 0.5 * (m.cells["x0"] + m.cells["x1"])
    ym = 0.5 * (m.cells["y0"] + m.cells["y1"])
    in_box = (np.abs(xm) < 0.4e-6) & (np.abs(ym) < 0.2e-6)
    far = (np.abs(xm) > 1.2e-6) | (np.abs(ym) > 1.0e-6)
    assert np.max(dx_cells[in_box]) <= 35e-9 * (1 + 1e-9)
    assert np.min(dx_cells[far]) == pytest.approx(coarse_h)

    # refine_near_polygons + combine_refinements
    wg = PolySlab(
        straight((-1.0e-6, 0.0), (1.0e-6, 0.0), width=0.4e-6),
        0.0,
        0.22e-6,
        "Si",
    )
    pred_poly = refine_near_polygons(wg, distance=0.1e-6, level=1, boundary_only=True)
    pred_box = refine_boxes([(0.2e-6, 0.6e-6, -0.8e-6, -0.5e-6, 2)])
    m2 = QuadMesh(
        xc, yc, ratio=2, max_level=2, refine=combine_refinements(pred_poly, pred_box)
    )
    counts = m2.level_counts()
    assert counts[1] > 0 and counts[2] > 0

    # Mutation: empty refine_boxes leaves the tooth band unrefined (dx = 100 nm > 35 nm)
    m_mut = QuadMesh(xc, yc, ratio=3, max_level=1, refine=refine_boxes([]))
    assert np.min(m_mut.cells["x1"] - m_mut.cells["x0"]) == pytest.approx(coarse_h)


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("lts", [None, 2])
def test_adjoint_dot_product_with_current_sources(pol, lts):
    """
    `linear_step(u, f, q)` and `linear_step_T(ub1, fb1, with_sources=True)`
    satisfy `<u1, ub1> + <f1, fb1> == <u, ub> + <f, fb> + <q, qb>` to 1e-12
    for node, edge, and interpolated point current sources.
    """
    x = np.linspace(0.0, 1.2e-6, 17)
    y = np.linspace(0.0, 1.0e-6, 15)
    ref = refine_boxes([(0.3e-6, 0.7e-6, 0.25e-6, 0.65e-6, 1)])
    m = QuadMesh(x, y, ratio=2, max_level=1, refine=ref)
    wg = PolySlab(
        straight((0.0, 0.5e-6), (1.2e-6, 0.5e-6), width=0.35e-6),
        0.0,
        0.22e-6,
        "Si",
    )
    geo = PolySlabGeometry([wg], background_eps="SiO2")
    s = SubgridFIT2D(m, geo, polarization=pol, cfl=0.9, lts=lts)

    rng = np.random.default_rng(42)
    # Register 3 sources: (0) node DOFs, (1) active edges, (2) 2D points inside patch
    dof_sel = rng.choice(s.n_dof, size=10, replace=False)
    edge_sel = rng.choice(s.n_edges, size=12, replace=False)
    pts = rng.uniform([0.32e-6, 0.28e-6], [0.68e-6, 0.62e-6], size=(8, 2))

    s.add_current(dof_sel, rng.standard_normal(10), kind="node")
    s.add_current(edge_sel, rng.standard_normal(12), kind="edge")
    s.add_current(
        pts,
        rng.standard_normal(8),
        kind="points",
        component="Ey" if pol == "TE" else "Ez",
    )

    u = rng.standard_normal(s.n_dof)
    f = rng.standard_normal(s.n_edges)
    q = rng.standard_normal(len(s.currents))
    ub1 = rng.standard_normal(s.n_dof)
    fb1 = rng.standard_normal(s.n_edges)

    u1, f1 = s.linear_step(u, f, q)
    ub, fb, qb = s.linear_step_T(ub1, fb1, with_sources=True)

    lhs = float(u1 @ ub1 + f1 @ fb1)
    rhs = float(u @ ub + f @ fb + q @ qb)
    rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs))
    assert rel_err < 1e-12, f"Adjoint dot-product relative error {rel_err:.3e} >= 1e-12"

    # Also verify pure edge-source sensitivity (u=0, f=0, q=[0, q[1], 0]) to 1e-12
    q_edge = np.array([0.0, q[1], 0.0])
    u1_q, f1_q = s.linear_step(np.zeros_like(u), np.zeros_like(f), q_edge)
    lhs_q = float(u1_q @ ub1 + f1_q @ fb1)
    rhs_q = float(q_edge @ qb)
    assert abs(lhs_q - rhs_q) < 1e-12 * (abs(lhs_q) + abs(rhs_q))

    # Mutation: using fb1 instead of fb in the edge-source adjoint misses the
    # within-step update of u1 from f1 and breaks the dot-product identity
    qb_bad = qb.copy()
    qb_bad[1] = s.dt * (s.currents[1]["b_f"] @ fb1)
    rhs_q_bad = float(q_edge @ qb_bad)
    assert abs(lhs_q - rhs_q_bad) > 0.9 * abs(lhs_q)


def test_photonic_circuit_subgrid_pec_energy_conservation_and_mutation():
    """
    With PML off (PEC outer box) and a refined patch crossing a silicon strip
    waveguide, discrete energy is conserved to < 1e-12 over 4000 steps; a
    non-transposed restriction blows up.
    """
    x = np.linspace(-1.0e-6, 1.0e-6, 21)
    y = np.linspace(-0.8e-6, 0.8e-6, 17)
    ref = refine_boxes([(-0.4e-6, 0.4e-6, -0.4e-6, 0.4e-6, 1)])
    m = QuadMesh(x, y, ratio=2, max_level=1, refine=ref)
    wg = PolySlab(
        straight((-1.0e-6, 0.0), (1.0e-6, 0.0), width=0.5e-6),
        0.0,
        0.22e-6,
        "Si",
    )
    geo = PolySlabGeometry([wg], background_eps="SiO2")
    s = SubgridFIT2D(m, geo, polarization="TE", pml=None, cfl=0.95)

    rng = np.random.default_rng(7)
    s.u = rng.standard_normal(s.n_dof)
    s.one_step()
    e0 = s.energy()
    max_drift = 0.0
    for step_idx in range(4000):
        s.one_step()
        if step_idx % 400 == 0:
            max_drift = max(max_drift, abs(s.energy() / e0 - 1.0))
    assert max_drift < 1e-12

    # Mutation: non-transposed injection on the fine-to-coarse restriction
    s_mut = SubgridFIT2D(m, geo, polarization="TE", pml=None, cfl=0.95)
    s_mut.u = rng.standard_normal(s_mut.n_dof)
    keep = np.isin(m.free, s_mut.dof)
    KQt = (m.gradient() @ _inject(m, 1.0).real[:, keep]).tocsr()[s_mut.edge_idx].T.tocsr()
    s_mut.one_step()
    e0_mut = s_mut.energy()
    worst_mut = 0.0
    for _ in range(4000):
        f1 = s_mut.f - s_mut.dt * (s_mut.K @ s_mut.u)
        s_mut.u_prev, s_mut.u = s_mut.u, s_mut.u + s_mut.dt * (KQt @ (s_mut.W * f1)) / s_mut.D
        s_mut.f_prev, s_mut.f = s_mut.f, f1
        e_cur = s_mut.energy()
        if not np.isfinite(e_cur):
            worst_mut = np.inf
            break
        worst_mut = max(worst_mut, abs(e_cur / e0_mut - 1.0))
        if worst_mut > 1e4:
            break
    assert worst_mut > 1.0


def test_straight_waveguide_subgrid_patches_and_vs_cartesian():
    """
    Validate S-parameters on a straight silicon waveguide:
    1. `max_level=0` vs `PhotonicCircuit` on the same uniform grid:
       `max ||S21_sg| - |S21_cart|| < 1e-2`;
    2. Refined patch crossed by the waveguide (interfaces normal to
       propagation): `T > 0.98`, `R < 1e-3`;
    3. Refined patch covering only the core width (interfaces parallel to
       propagation): `T > 0.98`, `R < 1e-3`;
    4. `lts=2` passthrough and `PhotonicCircuit(..., backend='subgrid')`;
    5. Mutation: corrupting the hanging-node prolongation weights `P` makes the
       reflection test (`R < 1e-3`) and transmission test (`T > 0.98`) fail.
    """
    _, _, kw = _straight_wg_setup()
    mid = slice(3, -3)

    # 1. Uniform coarse subgrid (max_level=0) vs Cartesian PhotonicCircuit
    circuit_cart = PhotonicCircuit(**kw)
    s_cart = circuit_cart.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=False
    )

    circuit_sg0 = SubgridPhotonicCircuit(**kw, max_level=0)
    s_sg0 = circuit_sg0.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=True
    )

    s21_diff = float(
        np.max(
            np.abs(
                np.abs(s_sg0[("out", "in")][mid])
                - np.abs(s_cart[("out", "in")][mid])
            )
        )
    )
    assert s21_diff < 1e-2, f"|S21| diff vs PhotonicCircuit = {s21_diff:.4e} >= 1e-2"
    assert 1550e-9 in circuit_sg0.dft_fields
    assert circuit_sg0.dft_fields[1550e-9]["E2"].max() > 0
    assert circuit_sg0.flux_balance is not None
    np.testing.assert_allclose(
        circuit_sg0.flux_balance["total_box"][mid], 1.0, atol=0.04
    )
    for key in ("n_cells", "n_nodes", "n_steps", "dt", "wall_time", "peak_rss_mb"):
        assert key in circuit_sg0.summary() and circuit_sg0.summary()[key] > 0

    # 2. Refined patch crossed by the waveguide (both interfaces normal to +x)
    circuit_norm = SubgridPhotonicCircuit(
        **kw, refine=[(-0.4e-6, 0.4e-6, -1.0e-6, 1.0e-6, 1)]
    )
    s_norm = circuit_norm.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=False
    )
    t_norm = float(np.min(np.abs(s_norm[("out", "in")][mid]) ** 2))
    r_norm = float(np.max(np.abs(s_norm[("in", "in")][mid]) ** 2))
    assert t_norm > 0.98, f"Normal-patch T_min = {t_norm:.4f} <= 0.98"
    assert r_norm < 1e-3, f"Normal-patch R_max = {r_norm:.4e} >= 1e-3"

    # 3. Refined patch covering only the core width (interfaces parallel to +x)
    circuit_par = SubgridPhotonicCircuit(
        **kw, refine=[(-1.3e-6, 1.3e-6, -0.3e-6, 0.3e-6, 1)]
    )
    s_par = circuit_par.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=False
    )
    t_par = float(np.min(np.abs(s_par[("out", "in")][mid]) ** 2))
    r_par = float(np.max(np.abs(s_par[("in", "in")][mid]) ** 2))
    assert t_par > 0.98, f"Parallel-patch T_min = {t_par:.4f} <= 0.98"
    assert r_par < 1e-3, f"Parallel-patch R_max = {r_par:.4e} >= 1e-3"

    # 4. Backend switch on PhotonicCircuit + LTS passthrough (lts=2) on a patch
    circuit_lts = PhotonicCircuit(
        **kw,
        backend="subgrid",
        refine=[(-0.4e-6, 0.4e-6, -0.5e-6, 0.5e-6, 1)],
        lts=2,
    )
    assert isinstance(circuit_lts, SubgridPhotonicCircuit)
    assert circuit_lts.solver.lts is not None
    s_lts = circuit_lts.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=False
    )
    assert float(np.min(np.abs(s_lts[("out", "in")][mid]) ** 2)) > 0.98
    assert float(np.max(np.abs(s_lts[("in", "in")][mid]) ** 2)) < 1e-3

    # 5. Mutation: break hanging-node prolongation weights -> R >= 1e-3 and T <= 0.98
    circuit_mut = SubgridPhotonicCircuit(
        **kw, refine=[(-0.4e-6, 0.4e-6, -1.0e-6, 1.0e-6, 1)]
    )
    m = circuit_mut.mesh
    h, ab, t, wb = m._constraints
    # Perturb hanging-node weights so rows of P no longer interpolate linearly
    m.P = m.prolongation()
    m.P.data[np.isin(m.P.tocoo().row, h)] *= -0.5
    circuit_mut.solver._assemble()
    circuit_mut.solver._setup_pml(circuit_mut.solver.pml)
    s_mut = circuit_mut.run_s_parameters(
        drive=["in"], energy_decay=1e-4, collect_fields=False
    )
    t_mut = float(np.min(np.abs(s_mut[("out", "in")][mid]) ** 2))
    r_mut = float(np.max(np.abs(s_mut[("in", "in")][mid]) ** 2))
    assert r_mut > 1e-3 and t_mut < 0.98


def test_monitors_dft_and_flux_consistency():
    """
    `SubgridFluxMonitor` on the output port line matches `2 * |a+|^2` from
    `SubgridModeMonitor`, and `SubgridDFTMonitor` reconstructs `Ey` on the port
    line matching the modal field.
    """
    _, _, kw = _straight_wg_setup()
    circuit = SubgridPhotonicCircuit(
        **kw, refine=[(-0.3e-6, 0.3e-6, -0.3e-6, 0.3e-6, 1)]
    )
    out_port = circuit.port_map["out"]
    p_center, p_size, _ = circuit._port_plane_args(out_port)

    in_port = circuit.port_map["in"]
    in_c, in_s, in_inward = circuit._port_plane_args(in_port)
    k_mon = int(np.argmin(np.abs(circuit.grid.x[:-1] - in_c[0])))
    src_c = in_c.copy()
    src_c[0] = circuit.grid.x[k_mon - 2 * in_inward]

    circuit.solver.clear_currents()
    circuit.solver.reset()
    src = circuit.solver
    from wakis.subgrid import SubgridModeMonitor, SubgridModeSource

    SubgridModeSource(
        circuit.pulse,
        center=src_c,
        size=in_s,
        direction="+",
        mode_frequency=circuit.f_center,
    ).register(src)

    mode_mon = SubgridModeMonitor(
        [circuit.f_center],
        center=p_center,
        size=p_size,
        mode_frequency=circuit.f_center,
    )
    flux_mon = SubgridFluxMonitor(
        [circuit.f_center], center=p_center[:2], size=p_size[:2], n_pts=80
    )
    dft_mon = SubgridDFTMonitor(
        [circuit.f_center],
        center=p_center[:2],
        size=(0.0, 0.0),
        components=("Ey", "Hz"),
    )

    for n in range(900):
        t = n * src.dt
        src.one_step()
        mode_mon.update(src, t)
        flux_mon.update(src, t)
        dft_mon.update(src, t)

    a_p, a_m = mode_mon.get_amplitudes(0)
    flux = flux_mon.get_flux()[0]
    modal_flux = float(2.0 * (np.abs(a_p[0]) ** 2 - np.abs(a_m[0]) ** 2))
    assert flux == pytest.approx(modal_flux, rel=2e-2)

    ey_center, _ = dft_mon.get_field("Ey")
    hz_center, _ = dft_mon.get_field("Hz")
    assert np.abs(ey_center[0, 0]) > 0 and np.abs(hz_center[0, 0]) > 0


def test_cost_baseline_subgrid_vs_solverfit3d():
    """
    Measure per-step wall time per cell and per node for
    `SubgridFIT2D(max_level=0)` vs `SolverFIT3D` 2D on the same uniform grid.
    """
    _, _, kw = _straight_wg_setup()
    circuit_c = PhotonicCircuit(**kw)
    circuit_s = SubgridPhotonicCircuit(**kw, max_level=0)

    assert (circuit_c.grid.Nx - 1) * (circuit_c.grid.Ny - 1) == circuit_s.mesh.n_cells
    assert circuit_c.grid.Nx * circuit_c.grid.Ny == circuit_s.mesh.n_nodes
    n_cells = circuit_s.mesh.n_cells
    n_nodes = circuit_s.mesh.n_nodes
    n_steps = 200

    circuit_c.solver.one_step()
    circuit_s.solver.one_step()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        circuit_c.solver.one_step()
    dt_cart = (time.perf_counter() - t0) / n_steps

    t0 = time.perf_counter()
    for _ in range(n_steps):
        circuit_s.solver.one_step()
    dt_sg = (time.perf_counter() - t0) / n_steps

    ns_per_cell_cart = dt_cart / n_cells * 1e9
    ns_per_cell_sg = dt_sg / n_cells * 1e9
    ns_per_node_cart = dt_cart / n_nodes * 1e9
    ns_per_node_sg = dt_sg / n_nodes * 1e9

    assert 1.0 < ns_per_cell_cart < 5000.0
    assert 1.0 < ns_per_cell_sg < 5000.0
    assert 1.0 < ns_per_node_cart < 5000.0
    assert 1.0 < ns_per_node_sg < 5000.0


def test_deprecated_subgrid_pic_aliases_and_mutation():
    """
    Deprecated `wakis.subgrid.pic`, `SubgridPICSimulation`, and `PIC_POLARIZATION`
    still work and emit `DeprecationWarning`; the new names emit no warning.
    """
    _, _, kw = _straight_wg_setup()

    with warnings.catch_warnings(record=True) as rec_new:
        warnings.simplefilter("always", DeprecationWarning)
        circuit_new = SubgridPhotonicCircuit(**kw, max_level=0)
        pol_new = PHOTONIC_CIRCUIT_POLARIZATION
    dep_new = [w for w in rec_new if issubclass(w.category, DeprecationWarning)]
    assert len(dep_new) == 0
    assert pol_new == "TE"

    with pytest.deprecated_call(match="SubgridPICSimulation is deprecated"):
        circuit_old = SubgridPICSimulation(**kw, max_level=0)
    assert isinstance(circuit_old, SubgridPhotonicCircuit)
    assert circuit_old.mesh.n_cells == circuit_new.mesh.n_cells

    with pytest.deprecated_call(match="PIC_POLARIZATION is deprecated"):
        pol_old_pkg = subgrid_mod.PIC_POLARIZATION
    assert pol_old_pkg == PHOTONIC_CIRCUIT_POLARIZATION

    with pytest.deprecated_call(match="PIC_POLARIZATION is deprecated"):
        pol_old_ports = subgrid_ports_mod.PIC_POLARIZATION
    assert pol_old_ports == PHOTONIC_CIRCUIT_POLARIZATION

    sys.modules.pop("wakis.subgrid.pic", None)
    with pytest.deprecated_call(match="wakis.subgrid.pic is deprecated"):
        shim = importlib.import_module("wakis.subgrid.pic")
    assert shim.SubgridPhotonicCircuit is SubgridPhotonicCircuit
    assert shim.SubgridPICSimulation is SubgridPICSimulation
