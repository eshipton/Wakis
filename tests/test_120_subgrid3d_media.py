# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification of 3D subgrid dispersive (Drude/Lorentz ADE) and conformal
(subpixel) media inside patches (`wakis.subgrid3d.media`).
"""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from wakis.dispersive import DispersiveMaterial, Drude, Lorentz
from wakis.subgrid3d import (
    Box,
    SubgridFIT3D,
    SubgridLTSFIT3D,
    SubgridMesh3D,
    dispersive_dt_max,
    dispersive_eigenfrequencies,
    dispersive_pencil,
)

C0 = 299792458.0


def _td_peak_frequency(trace, dt, f_expect):
    """Parabolic-interpolated Hann-DFT peak near `f_expect` [Hz]."""
    y = np.asarray(trace, float) - np.mean(trace)
    w = np.hanning(y.size)
    n_pad = 8 * y.size
    spec = np.abs(np.fft.rfft(y * w, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)
    mask = np.abs(freqs - f_expect) < 0.25 * f_expect
    idx = np.flatnonzero(mask)[np.argmax(spec[mask])]
    s0, s1, s2 = np.log(spec[idx - 1 : idx + 2] + 1e-300)
    delta = 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2)
    return float(freqs[idx] + delta * (freqs[1] - freqs[0]))


def _non_pow2_patched_mesh():
    """1.0 x 0.7 x 0.9 m PEC box, 10 x 7 x 9 coarse cells, interior 2:1 patch."""
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 0.7, 8)
    z = np.linspace(0.0, 0.9, 10)
    m = SubgridMesh3D(x, y, z)
    m.add_patch((2, 2, 2), (8, 5, 7), ratio=2)
    return m


def _closed_nested_mesh():
    """Graded coarse grid with nested 2:1 + 3:1 patches and interior PEC/PMC cells (as test_104)."""
    x = np.array([0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0])
    m = SubgridMesh3D(x, np.linspace(0, 0.8, 9), np.linspace(0, 0.6, 7))
    p1 = m.add_patch((2, 2, 1), (7, 6, 5), ratio=2)
    m.add_patch((3, 2, 2), (7, 6, 6), ratio=3, parent=p1)
    pec = Box(((0.42, 0.55), (0.35, 0.45), (0.25, 0.35)))
    pmc = Box(((0.75, 0.85), (0.1, 0.2), (0.0, 0.1)))
    return m, pec, pmc


# ===================================================================== Part 1
def test_d1_homogeneous_lorentz_mass_identity():
    """D1: One Lorentz material fills a 2:1-patched cavity: M_k * eps_inf == D."""
    mesh = _non_pow2_patched_mesh()
    eps_inf = 2.25
    mat = DispersiveMaterial(
        eps_inf=eps_inf, poles=[Lorentz(1.5, 2 * np.pi * 180e6, 0.0)]
    )
    full_box = Box(((-1.0, 2.0), (-1.0, 2.0), (-1.0, 2.0)))

    solver = SubgridFIT3D(mesh, dispersive=[(full_box, mat)])
    M_k = solver.dispersive_groups[0]["M_full"]
    rel_err = float(np.max(np.abs(M_k * eps_inf / solver.D - 1.0)))

    solver_mut = SubgridFIT3D(
        mesh, dispersive=[(full_box, mat)], mutate="lump_no_rowsum"
    )
    M_k_mut = solver_mut.dispersive_groups[0]["M_full"]
    rel_err_mut = float(np.max(np.abs(M_k_mut * eps_inf / solver_mut.D - 1.0)))

    print(
        f"NUMBERS test_d1_homogeneous_lorentz_mass_identity: "
        f"rel_err={rel_err:.3e} mut_lump_no_rowsum={rel_err_mut:.3e}"
    )
    assert rel_err <= 1e-14
    assert rel_err_mut >= 0.1


def test_d2_homogeneous_lorentz_cavity_mode_and_perturbation():
    """D2: Lowest mode of a homogeneous Lorentz 2:1-patched cavity via Hann DFT vs
    2x2-Lambda and epsilon_discrete roots, plus lump_no_rowsum perturbation shift."""
    mesh = _non_pow2_patched_mesh()
    eps_inf = 1.5
    delta_eps = 3.0
    f0 = 320e6
    w0 = 2.0 * np.pi * f0
    s = delta_eps * w0**2
    mat = DispersiveMaterial(eps_inf=eps_inf, poles=[Lorentz(delta_eps, w0, 0.0)])
    full_box = Box(((-1.0, 2.0), (-1.0, 2.0), (-1.0, 2.0)))

    # Vacuum eigenmode (lowest cavity mode) on the exact same patched mesh
    s_vac = SubgridFIT3D(mesh, eps_r=1.0)
    S_vac = (s_vac.KH @ sp.diags(s_vac.W) @ s_vac.K).tocsc()
    sigma_vac = (2.0 * np.pi * 100e6) ** 2
    lu_vac = sla.splu(
        (S_vac - sigma_vac * sp.diags(s_vac.D)).tocsc(), permc_spec="MMD_AT_PLUS_A"
    )
    op_vac = sla.LinearOperator(
        (s_vac.n_free, s_vac.n_free), matvec=lu_vac.solve, dtype=float
    )
    lam_all, vec_all = sla.eigsh(
        S_vac,
        k=4,
        M=sp.diags(s_vac.D).tocsc(),
        sigma=sigma_vac,
        OPinv=op_vac,
        which="LA",
        tol=1e-13,
        v0=np.random.default_rng(0).standard_normal(s_vac.n_free),
    )
    order = np.argsort(lam_all)
    lam_vac = float(lam_all[order[0]])
    v_vac = vec_all[:, order[0]]
    v_vac /= np.sqrt(float(v_vac @ (s_vac.D * v_vac)))

    solver = SubgridFIT3D(mesh, dispersive=[(full_box, mat)], cfl=0.9)
    dt = solver.dt

    # Reference path 1: 2x2 semi-discrete matrix Lambda -> time-discrete arcsin
    A2 = np.array(
        [[(lam_vac + s) / eps_inf, -(w0**2) / eps_inf], [-s, w0**2]], dtype=float
    )
    Lam = np.sort(np.real(la.eigvals(A2)))
    Lam_low = float(Lam[0])
    omega_lam = (2.0 / dt) * np.arcsin(0.5 * dt * np.sqrt(Lam_low))
    f_lam = float(omega_lam / (2.0 * np.pi))

    # Reference path 2: root of (2 sin(omega dt / 2) / dt)^2 * epsilon_discrete(f, dt) == lam_vac
    def disp_res(omega):
        om_hat = 2.0 * np.sin(0.5 * omega * dt) / dt
        f_hz = omega / (2.0 * np.pi)
        eps_d = float(np.real(mat.epsilon_discrete(np.array([f_hz]), dt)[0]))
        return om_hat**2 * eps_d - lam_vac

    omega_ed = brentq(disp_res, 0.2 * omega_lam, 0.999 * w0, xtol=1e-15, rtol=1e-15)
    f_ed = float(omega_ed / (2.0 * np.pi))
    ref_diff = abs(f_lam - f_ed) / f_lam

    # Initialize exclusively in the lower branch mode along v_vac and probe
    om_hat_low = 2.0 * np.sin(0.5 * omega_lam * dt) / dt
    p_ratio = s / (w0**2 - om_hat_low**2)

    def _run_probe(slv):
        slv.reset()
        slv.e[:] = v_vac
        g = slv.dispersive_groups[0]
        g["p"][0][:] = p_ratio * v_vac[g["idx"]]
        g["p_old"][0][:] = p_ratio * np.cos(omega_lam * slv.dt) * v_vac[g["idx"]]
        slv.b[:] = -(np.sin(0.5 * omega_lam * slv.dt) / om_hat_low) * (slv.K @ v_vac)
        # Pick edge with maximum |v_vac| for high SNR
        i_max = int(np.argmax(np.abs(v_vac)))
        n_steps = 4096
        tr = np.empty(n_steps, float)
        for step in range(n_steps):
            tr[step] = slv.e[i_max]
            slv.one_step()
        return _td_peak_frequency(tr, slv.dt, f_lam)

    f_td = _run_probe(solver)
    rel_err_lam = abs(f_td - f_lam) / f_lam
    rel_err_ed = abs(f_td - f_ed) / f_ed

    # Mutation: lump_no_rowsum. First predict shift by 1st-order perturbation theory.
    solver_mut = SubgridFIT3D(
        mesh, dispersive=[(full_box, mat)], dt=dt, mutate="lump_no_rowsum"
    )
    M_ok = solver.dispersive_groups[0]["M_full"]
    M_mut = solver_mut.dispersive_groups[0]["M_full"]
    rel_dM = (M_mut - M_ok) / M_ok
    if_mask = np.abs(rel_dM) > 1e-12
    e_weights = (solver.D * v_vac**2) / float(v_vac @ (solver.D * v_vac))
    if_energy_share = float(np.sum(e_weights[if_mask]))
    alpha_if = float(np.sum(rel_dM * e_weights))
    chi_pole = s / (w0**2 - Lam_low)
    d_disp = eps_inf + s * w0**2 / (w0**2 - Lam_low) ** 2
    pred_rel_shift = abs(-0.5 * (chi_pole / d_disp) * alpha_if)

    f_td_mut = _run_probe(solver_mut)
    meas_rel_shift = abs(f_td_mut - f_td) / f_lam

    print(
        f"NUMBERS test_d2_homogeneous_lorentz_cavity_mode_and_perturbation: "
        f"f_lam_hz={f_lam:.6e} f_ed_hz={f_ed:.6e} ref_diff={ref_diff:.3e} "
        f"f_td_hz={f_td:.6e} rel_err_lam={rel_err_lam:.3e} rel_err_ed={rel_err_ed:.3e} "
        f"if_energy_share={if_energy_share:.3e} pred_shift={pred_rel_shift:.3e} "
        f"mut_lump_no_rowsum_shift={meas_rel_shift:.3e}"
    )
    assert ref_diff <= 1e-12
    assert rel_err_lam <= 1e-5
    assert rel_err_ed <= 1e-5
    assert pred_rel_shift > 1e-3
    assert meas_rel_shift > 1e-4


def test_d3_lossless_and_damped_stability_and_energy():
    """D3: Lossless Lorentz + Drude media crossing nested 2:1 + 3:1 patches on
    graded mesh with PEC/PMC cells: 4000-step energy conservation at 0.99 dt_max,
    instability above dt_max, strength mutation drift, and damped spectral radius."""
    mesh, pec, pmc = _closed_nested_mesh()
    mat_lor = DispersiveMaterial(
        eps_inf=1.8, poles=[Lorentz(1.2, 2 * np.pi * 280e6, 0.0)]
    )
    mat_dru = DispersiveMaterial(eps_inf=2.1, poles=[Drude(2 * np.pi * 240e6, 0.0)])
    box_lor = Box(((0.15, 0.55), (0.15, 0.55), (0.10, 0.60)))
    box_dru = Box(((0.45, 0.75), (0.20, 0.65), (0.15, 0.65)))
    disp = [(box_lor, mat_lor), (box_dru, mat_dru)]

    probe_s = SubgridFIT3D(mesh, pec=pec, pmc=pmc, dispersive=disp)
    dt_lim = probe_s.dispersive_dt_max()
    dt_safe = 0.99 * dt_lim

    def _init_random(slv, seed=42):
        rng = np.random.default_rng(seed)
        slv.e[:] = rng.standard_normal(slv.n_free)
        slv.b[:] = rng.standard_normal(slv.n_faces) * (slv.dt * 1e-9)
        for g in slv.dispersive_groups:
            for k in range(len(g["poles"])):
                g["p"][k][:] = rng.standard_normal(g["idx"].size) * 1e-1
                g["p_old"][k][:] = rng.standard_normal(g["idx"].size) * 1e-1

    solver = SubgridFIT3D(mesh, pec=pec, pmc=pmc, dispersive=disp, dt=dt_safe)
    _init_random(solver)
    n_steps = 4000
    hist = solver.run(n_steps, energy=True)
    rel_drift = float(np.max(np.abs(hist - hist[0])) / abs(hist[0]))
    bound = float(50.0 * np.finfo(float).eps * np.sqrt(n_steps))

    # Mutation 1: dt = 1.02 * dt_max -> raises ValueError on init, and grows > 1e3 if unguarded
    with pytest.raises(ValueError, match="dispersive limit"):
        SubgridFIT3D(mesh, pec=pec, pmc=pmc, dispersive=disp, dt=1.02 * dt_lim)

    s_over = SubgridFIT3D(
        mesh, pec=pec, pmc=pmc, dispersive=disp, dt=1.02 * dt_lim, check_stability=False
    )
    _init_random(s_over)
    e0_norm = float(np.linalg.norm(s_over.e))
    growth_over = 1.0
    for _ in range(600):
        s_over.one_step()
        en = float(np.linalg.norm(s_over.e))
        if not np.isfinite(en) or en / e0_norm > 1e6:
            growth_over = float(en / e0_norm) if np.isfinite(en) else float("inf")
            break
        growth_over = max(growth_over, en / e0_norm)

    # Mutation 2: mutate="strength" (zeta uses 1.01 * s, M unchanged) -> drift > 1e3 * bound
    s_mut = SubgridFIT3D(
        mesh, pec=pec, pmc=pmc, dispersive=disp, dt=dt_safe, mutate="strength"
    )
    _init_random(s_mut)
    hist_mut = s_mut.run(400, energy=True)
    drift_mut = float(np.max(np.abs(hist_mut - hist_mut[0])) / abs(hist_mut[0]))

    # Damped poles (gamma > 0): dense spectral radius of one-step matrix on a small patched mesh
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 0.7, 4)
    zs = np.linspace(0.0, 0.9, 4)
    ms = SubgridMesh3D(xs, ys, zs)
    ms.add_patch((1, 1, 1), (3, 2, 2), ratio=2)
    mat_damped = DispersiveMaterial(
        eps_inf=1.6,
        poles=[
            Lorentz(1.1, 2 * np.pi * 250e6, 2 * np.pi * 40e6),
            Drude(2 * np.pi * 220e6, 2 * np.pi * 30e6),
        ],
    )
    mat_lossless = DispersiveMaterial(
        eps_inf=1.6,
        poles=[
            Lorentz(1.1, 2 * np.pi * 250e6, 0.0),
            Drude(2 * np.pi * 220e6, 0.0),
        ],
    )
    s_ll = SubgridLTSFIT3D(
        ms,
        lts=False,
        dispersive=[(Box(((0.2, 0.8), (0.1, 0.6), (0.1, 0.8))), mat_lossless)],
    )
    dt_ll = s_ll.dispersive_dt_max()
    s_dmp = SubgridLTSFIT3D(
        ms,
        lts=False,
        dispersive=[(Box(((0.2, 0.8), (0.1, 0.6), (0.1, 0.8))), mat_damped)],
        dt=0.99 * dt_ll,
    )
    n_st = s_dmp.n_state
    assert n_st <= 4000
    cols = [s_dmp.step_vec(row) for row in np.eye(n_st)]
    M_step = np.column_stack(cols)
    rho_damped = float(np.max(np.abs(la.eigvals(M_step))))

    print(
        f"NUMBERS test_d3_lossless_and_damped_stability_and_energy: "
        f"rel_drift={rel_drift:.3e} bound={bound:.3e} "
        f"mut_dt_1p02_growth={growth_over:.3e} mut_strength_drift={drift_mut:.3e} "
        f"drift_ratio={drift_mut / bound:.3e} rho_damped={rho_damped:.12f}"
    )
    assert rel_drift <= bound
    assert not np.isfinite(growth_over) or growth_over > 1e3
    assert drift_mut > 1e3 * bound
    assert rho_damped <= 1.0 + 1e-12


def test_d4_lanczos_dt_max_vs_dense_and_drude_identity():
    """D4: dispersive_dt_max (Lanczos) vs dense eigh on a tiny patched mesh,
    and homogeneous Drude cavity vs courant^2 + (omega_p dt / 2)^2 = eps_inf."""
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 0.7, 4)
    z = np.linspace(0.0, 0.9, 4)
    mesh = SubgridMesh3D(x, y, z)
    mesh.add_patch((1, 1, 1), (3, 2, 2), ratio=2)
    mat_lor = DispersiveMaterial(
        eps_inf=1.7, poles=[Lorentz(1.3, 2 * np.pi * 310e6, 0.0)]
    )
    mat_dru = DispersiveMaterial(eps_inf=2.2, poles=[Drude(2 * np.pi * 260e6, 0.0)])
    disp = [
        (Box(((0.1, 0.6), (0.1, 0.5), (0.1, 0.6))), mat_lor),
        (Box(((0.4, 0.9), (0.2, 0.6), (0.2, 0.8))), mat_dru),
    ]
    solver = SubgridFIT3D(mesh, dispersive=disp)
    dt_lanczos = dispersive_dt_max(solver, tol=1e-13)

    K_mat, M_vec, _ = dispersive_pencil(solver)
    lam_dense = la.eigvalsh(K_mat.toarray(), np.diag(M_vec))
    dt_dense = float(2.0 / np.sqrt(np.max(lam_dense)))
    rel_err_dense = abs(dt_lanczos - dt_dense) / dt_dense

    # Homogeneous Drude identity: (dt / dt_vac)^2 + (omega_p dt / 2)^2 == eps_inf
    eps_inf = 2.4
    omega_p = 2.0 * np.pi * 350e6
    s_vac = SubgridFIT3D(mesh, eps_r=1.0)
    dt_vac = s_vac.dt_max
    mat_h = DispersiveMaterial(eps_inf=eps_inf, poles=[Drude(omega_p, 0.0)])
    s_dru = SubgridFIT3D(
        mesh, dispersive=[(Box(((-1.0, 2.0), (-1.0, 2.0), (-1.0, 2.0))), mat_h)]
    )
    dt_dru = s_dru.dispersive_dt_max()
    dt_formula = float(
        1.0 / np.sqrt((1.0 / (dt_vac**2 * eps_inf)) + (omega_p**2 / (4.0 * eps_inf)))
    )
    courant_sq_plus = (dt_dru / dt_vac) ** 2 + (0.5 * omega_p * dt_dru) ** 2
    rel_err_drude = abs(dt_dru - dt_formula) / dt_formula
    rel_err_id = abs(courant_sq_plus - eps_inf) / eps_inf

    print(
        f"NUMBERS test_d4_lanczos_dt_max_vs_dense_and_drude_identity: "
        f"rel_err_dense={rel_err_dense:.3e} rel_err_drude={rel_err_drude:.3e} "
        f"rel_err_identity={rel_err_id:.3e}"
    )
    assert rel_err_dense <= 1e-10
    assert rel_err_drude <= 1e-10
    assert rel_err_id <= 1e-10


def _dot_defect(s):
    """Relative dot-product defect of `step_vec_T` with b and i scaled to e."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(s.n_state)
    y = rng.standard_normal(s.n_state)
    i = rng.standard_normal(s.mesh.n_edges) * (np.median(s.D) / s.dt)
    W = np.asarray(s._Wb)
    sb = np.sqrt(np.median(s.D) / np.median(W[W > 0]))
    x[: s._nb] *= sb
    y[: s._nb] *= sb
    fx = s.step_vec(x, i)
    xb, ib = s.step_vec_T(y)
    scale = float(np.linalg.norm(fx) * np.linalg.norm(y))
    return float(abs(np.dot(fx, y) - (np.dot(x, xb) + np.dot(i, ib))) / scale)


def test_d5_step_vec_transpose_dot_product():
    """D5: Dot-product test <y, M x> == <M^T y, x> of step_vec_T with ADE,
    with and without CPML, plus ade_transpose_no_feedback mutation."""
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(0.0, 0.7, 6)
    z = np.linspace(0.0, 0.9, 6)
    mesh = SubgridMesh3D(x, y, z)
    mesh.add_patch((2, 2, 2), (4, 4, 4), ratio=2)
    mat_lor = DispersiveMaterial(
        eps_inf=1.9, poles=[Lorentz(1.4, 2 * np.pi * 270e6, 2 * np.pi * 35e6)]
    )
    mat_dru = DispersiveMaterial(
        eps_inf=2.3, poles=[Drude(2 * np.pi * 220e6, 2 * np.pi * 25e6)]
    )
    disp = [
        (Box(((0.2, 0.6), (0.2, 0.5), (0.2, 0.6))), mat_lor),
        (Box(((0.4, 0.8), (0.3, 0.6), (0.3, 0.7))), mat_dru),
    ]

    s_nopml = SubgridLTSFIT3D(mesh, lts=False, dispersive=disp)
    err_nopml = _dot_defect(s_nopml)

    # With CPML (patch is at cells 2..4, 1-cell CPML layer is at cells 0..1 and N-1..N)
    th = [(0.16, 0.16), (0.13, 0.13), (0.15, 0.15)]
    s_pml = SubgridLTSFIT3D(
        mesh, lts=False, dispersive=disp, pml=dict(thickness=th, R0=1e-6)
    )
    err_pml = _dot_defect(s_pml)

    # Mutation: omit ADE feedback term in transpose
    s_mut = SubgridLTSFIT3D(
        mesh, lts=False, dispersive=disp, mutate="ade_transpose_no_feedback"
    )
    err_mut = _dot_defect(s_mut)

    print(
        f"NUMBERS test_d5_step_vec_transpose_dot_product: "
        f"err_nopml={err_nopml:.3e} err_pml={err_pml:.3e} "
        f"mut_no_feedback={err_mut:.3e}"
    )
    assert err_nopml <= 1e-15
    assert err_pml <= 1e-15
    assert err_mut >= 1e-8


# ------------------------------------------------ 1D slab shooting helper (D6/C2)
def _shoot_1d_slab_modes(
    a, b, c, z_s, eps_of_omega, n_modes=6, f_min=120e6, f_max=400e6
):
    """
    Compute the lowest `n_modes` exact 1D TE_z and TM_z resonances [Hz] of a PEC box
    [0, a] x [0, b] x [0, c] with slab 0 < z < z_s of permittivity `eps_of_omega(omega)`
    and vacuum z_s < z < c, via `solve_ivp` (rtol=1e-12) + `brentq`.
    """
    candidates = []
    w_grid = np.linspace(2 * np.pi * f_min, 2 * np.pi * f_max, 25)

    def res_te(omega, kt2):
        k02 = (omega / C0) ** 2
        eps1 = float(eps_of_omega(omega))
        q1 = eps1 * k02 - kt2
        q2 = k02 - kt2
        sol1 = solve_ivp(
            lambda _z, y: [y[1], -q1 * y[0]],
            (0.0, z_s),
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
        )
        y_mid = [float(sol1.y[0, -1]), float(sol1.y[1, -1])]
        sol2 = solve_ivp(
            lambda _z, y: [y[1], -q2 * y[0]],
            (z_s, c),
            y_mid,
            rtol=1e-12,
            atol=1e-14,
        )
        return float(sol2.y[0, -1])

    def res_tm(omega, kt2):
        k02 = (omega / C0) ** 2
        eps1 = float(eps_of_omega(omega))
        sol1 = solve_ivp(
            lambda _z, y: [-(k02 - kt2 / eps1) * y[1], eps1 * y[0]],
            (0.0, z_s),
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
        )
        y_mid = [float(sol1.y[0, -1]), float(sol1.y[1, -1])]
        sol2 = solve_ivp(
            lambda _z, y: [-(k02 - kt2) * y[1], y[0]],
            (z_s, c),
            y_mid,
            rtol=1e-12,
            atol=1e-14,
        )
        return float(sol2.y[0, -1])

    for m in range(0, 3):
        for n in range(0, 3):
            if m == 0 and n == 0:
                continue
            kt2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            # TE_z(m, n)
            vals = [res_te(w, kt2) for w in w_grid]
            for i in range(len(w_grid) - 1):
                if vals[i] * vals[i + 1] <= 0:
                    w_root = brentq(
                        res_te,
                        w_grid[i],
                        w_grid[i + 1],
                        args=(kt2,),
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                    candidates.append((w_root / (2 * np.pi), f"TE_{m}{n}"))
            # TM_z(m, n) requires m >= 1 and n >= 1
            if m >= 1 and n >= 1:
                vals = [res_tm(w, kt2) for w in w_grid]
                for i in range(len(w_grid) - 1):
                    if vals[i] * vals[i + 1] <= 0:
                        w_root = brentq(
                            res_tm,
                            w_grid[i],
                            w_grid[i + 1],
                            args=(kt2,),
                            xtol=1e-13,
                            rtol=1e-13,
                        )
                        candidates.append((w_root / (2 * np.pi), f"TM_{m}{n}"))

    candidates.sort(key=lambda item: item[0])
    freqs = np.array([item[0] for item in candidates[:n_modes]], float)
    labels = [item[1] for item in candidates[:n_modes]]
    return freqs, labels


def _analytic_empty_box_modes(a, b, c, n_modes=6):
    """Lowest `n_modes` analytic cavity frequencies [Hz] of an empty PEC box."""
    modes = []
    for m in range(0, 4):
        for n in range(0, 4):
            for p in range(0, 4):
                f = 0.5 * C0 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)
                if m >= 1 and n >= 1 and p >= 0:
                    modes.append(f)  # TM_mnp
                if p >= 1 and (m >= 1 or n >= 1):
                    modes.append(f)  # TE_mnp
    return np.sort(modes)[:n_modes]


def _run_d6_slab_convergence(nz_list):
    """Run D6 Lorentz and Drude slab convergence across `nz_list`."""
    a, b, c = 0.38, 0.31, 0.90
    z_s = 0.30

    # 1. Validate 1D shooting reference on the empty box against analytic box modes
    f_1d_vac, lbl_vac = _shoot_1d_slab_modes(
        a, b, c, z_s, lambda _w: 1.0, n_modes=6, f_min=300e6, f_max=850e6
    )
    f_ana_vac = _analytic_empty_box_modes(a, b, c, n_modes=6)
    vac_1d_err = float(np.max(np.abs(f_1d_vac - f_ana_vac) / f_ana_vac))
    assert vac_1d_err <= 1e-11

    # 2. Lorentz slab (f0 = 520 MHz > lowest 6 cavity modes -> all 6 sit on lower polariton branch)
    f0_lor = 520e6
    w0_lor = 2 * np.pi * f0_lor
    mat_lor = DispersiveMaterial(eps_inf=1.0, poles=[Lorentz(6.0, w0_lor, 0.0)])
    s_lor = 6.0 * w0_lor**2
    f_ref_lor, lbl_lor = _shoot_1d_slab_modes(
        a,
        b,
        c,
        z_s,
        lambda w: 1.0 + s_lor / (w0_lor**2 - w**2),
        n_modes=6,
        f_min=160e6,
        f_max=420e6,
    )

    # 3. Lossless Drude slab (fp = 380 MHz, eps_inf = 1.0 -> propagating branch above 440 MHz)
    wp_dru = 2 * np.pi * 380e6
    mat_dru = DispersiveMaterial(eps_inf=1.0, poles=[Drude(wp_dru, 0.0)])
    f_ref_dru, lbl_dru = _shoot_1d_slab_modes(
        a,
        b,
        c,
        z_s,
        lambda w: 1.0 - (wp_dru / w) ** 2,
        n_modes=6,
        f_min=400e6,
        f_max=750e6,
    )

    slab_box = Box(((-0.1, a + 0.1), (-0.1, b + 0.1), (-0.1, z_s)))

    err_lor_1d, err_lor_fine, err_lor_mut = [], [], []
    err_dru_1d, err_dru_fine, err_dru_mut = [], [], []

    for nz in nz_list:
        s = nz // 9
        nx, ny = 3 * s, 3 * s
        x = np.linspace(0.0, a, nx + 1)
        y = np.linspace(0.0, b, ny + 1)
        z = np.linspace(0.0, c, nz + 1)
        # Patch side faces (i=1s..2s, j=1s..2s) and bottom face (k=1s < 3s) cut the slab 0 < z < 0.3
        mesh_p = SubgridMesh3D(x, y, z)
        mesh_p.add_patch((1 * s, 1 * s, 1 * s), (2 * s, 2 * s, 5 * s), ratio=2)

        xf = np.linspace(0.0, a, 2 * nx + 1)
        yf = np.linspace(0.0, b, 2 * ny + 1)
        zf = np.linspace(0.0, c, 2 * nz + 1)
        mesh_f = SubgridMesh3D(xf, yf, zf)

        # Lorentz
        slv_lp = SubgridFIT3D(
            mesh_p, dispersive=[(slab_box, mat_lor)], dt=1e-11, check_stability=False
        )
        f_lp = dispersive_eigenfrequencies(slv_lp, 6, f_ref_lor[0])
        slv_lf = SubgridFIT3D(
            mesh_f, dispersive=[(slab_box, mat_lor)], dt=1e-11, check_stability=False
        )
        f_lf = dispersive_eigenfrequencies(slv_lf, 6, f_ref_lor[0])
        slv_lm = SubgridFIT3D(
            mesh_p,
            dispersive=[(slab_box, mat_lor)],
            dt=1e-11,
            check_stability=False,
            mutate="lump_no_rowsum",
        )
        f_lm = dispersive_eigenfrequencies(slv_lm, 6, f_ref_lor[0])

        err_lor_1d.append(float(np.mean(np.abs(f_lp - f_ref_lor) / f_ref_lor)))
        err_lor_fine.append(float(np.mean(np.abs(f_lp - f_lf) / f_lf)))
        err_lor_mut.append(float(np.mean(np.abs(f_lm - f_ref_lor) / f_ref_lor)))

        # Drude (shift above the bulk longitudinal plasma frequency fp = 252 MHz)
        f_guess_dru = 1.85 * f_ref_dru[0]
        slv_dp = SubgridFIT3D(
            mesh_p, dispersive=[(slab_box, mat_dru)], dt=1e-11, check_stability=False
        )
        f_dp = dispersive_eigenfrequencies(slv_dp, 6, f_guess_dru)
        slv_df = SubgridFIT3D(
            mesh_f, dispersive=[(slab_box, mat_dru)], dt=1e-11, check_stability=False
        )
        f_df = dispersive_eigenfrequencies(slv_df, 6, f_guess_dru)
        slv_dm = SubgridFIT3D(
            mesh_p,
            dispersive=[(slab_box, mat_dru)],
            dt=1e-11,
            check_stability=False,
            mutate="lump_no_rowsum",
        )
        f_dm = dispersive_eigenfrequencies(slv_dm, 6, f_guess_dru)

        err_dru_1d.append(float(np.mean(np.abs(f_dp - f_ref_dru) / f_ref_dru)))
        err_dru_fine.append(float(np.mean(np.abs(f_dp - f_df) / f_df)))
        err_dru_mut.append(float(np.mean(np.abs(f_dm - f_ref_dru) / f_ref_dru)))

    def _orders(errs):
        orders = []
        for i in range(len(nz_list) - 1):
            r = nz_list[i + 1] / nz_list[i]
            orders.append(float(np.log(errs[i] / errs[i + 1]) / np.log(r)))
        return orders

    return dict(
        vac_1d_err=vac_1d_err,
        lbl_lor=lbl_lor,
        lbl_dru=lbl_dru,
        err_lor_1d=err_lor_1d,
        err_lor_fine=err_lor_fine,
        err_lor_mut=err_lor_mut,
        err_dru_1d=err_dru_1d,
        err_dru_fine=err_dru_fine,
        err_dru_mut=err_dru_mut,
        ord_lor_1d=_orders(err_lor_1d),
        ord_lor_fine=_orders(err_lor_fine),
        ord_lor_mut=_orders(err_lor_mut),
        ord_dru_1d=_orders(err_dru_1d),
        ord_dru_fine=_orders(err_dru_fine),
        ord_dru_mut=_orders(err_dru_mut),
    )


def test_d6_slab_resonances_fast():
    """D6 (fast, N_z in {9, 18}): Lorentz and lossless Drude slab crossing a 2:1 patch."""
    res = _run_d6_slab_convergence([9, 18])
    print(
        f"NUMBERS test_d6_slab_resonances_fast: "
        f"vac_1d_err={res['vac_1d_err']:.3e} "
        f"lor_modes={','.join(res['lbl_lor'])} dru_modes={','.join(res['lbl_dru'])} "
        f"lor_err_1d={res['err_lor_1d'][0]:.3e}->{res['err_lor_1d'][1]:.3e} "
        f"lor_ord_1d={res['ord_lor_1d'][0]:.3f} lor_ord_fine={res['ord_lor_fine'][0]:.3f} "
        f"lor_mut_ord={res['ord_lor_mut'][0]:.3f} "
        f"dru_err_1d={res['err_dru_1d'][0]:.3e}->{res['err_dru_1d'][1]:.3e} "
        f"dru_ord_1d={res['ord_dru_1d'][0]:.3f} dru_ord_fine={res['ord_dru_fine'][0]:.3f} "
        f"dru_mut_ord={res['ord_dru_mut'][0]:.3f}"
    )
    assert 1.8 <= res["ord_lor_1d"][0] <= 2.2
    assert 1.8 <= res["ord_lor_fine"][0] <= 2.2
    assert res["ord_lor_mut"][0] < 1.5
    assert 1.8 <= res["ord_dru_1d"][0] <= 2.2
    assert 1.8 <= res["ord_dru_fine"][0] <= 2.2
    assert res["ord_dru_mut"][0] < 1.5


@pytest.mark.slow
def test_d6_slab_resonances_slow():
    """D6 (slow, N_z in {9, 18, 27}): 3-level convergence of Lorentz and Drude slabs."""
    res = _run_d6_slab_convergence([9, 18, 27])
    print(
        f"NUMBERS test_d6_slab_resonances_slow: "
        f"lor_ord_1d={res['ord_lor_1d'][0]:.3f},{res['ord_lor_1d'][1]:.3f} "
        f"lor_ord_fine={res['ord_lor_fine'][0]:.3f},{res['ord_lor_fine'][1]:.3f} "
        f"lor_mut_ord={res['ord_lor_mut'][0]:.3f},{res['ord_lor_mut'][1]:.3f} "
        f"dru_ord_1d={res['ord_dru_1d'][0]:.3f},{res['ord_dru_1d'][1]:.3f} "
        f"dru_ord_fine={res['ord_dru_fine'][0]:.3f},{res['ord_dru_fine'][1]:.3f} "
        f"dru_mut_ord={res['ord_dru_mut'][0]:.3f},{res['ord_dru_mut'][1]:.3f}"
    )
    for o in (
        res["ord_lor_1d"]
        + res["ord_lor_fine"]
        + res["ord_dru_1d"]
        + res["ord_dru_fine"]
    ):
        assert 1.8 <= o <= 2.2
    for o in res["ord_lor_mut"] + res["ord_dru_mut"]:
        assert o < 1.5


# ===================================================================== Part 2
def test_c1_subpixel_node_planes_control():
    """C1: Control test with dielectric boxes strictly on node planes across a
    graded coarse grid and nested 2:1 + 3:1 patches: subpixel masses equal
    staircase masses to machine precision; count_weights mutation differs by O(grading)."""
    mesh, pec, pmc = _closed_nested_mesh()
    # Node planes of _closed_nested_mesh:
    # x = [0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0]
    # y = 0.0, 0.1, ..., 0.8; z = 0.0, 0.1, ..., 0.6
    eps_boxes = [
        Box(((0.1, 0.75), (0.1, 0.6), (0.1, 0.5)), value=4.0),
        Box(((0.3, 0.55), (0.2, 0.5), (0.2, 0.4)), value=2.5),
    ]

    def eps_callable(X, Y, Z):
        out = np.ones_like(X, dtype=float)
        m1 = (
            (X >= 0.1) & (X <= 0.75) & (Y >= 0.1) & (Y <= 0.6) & (Z >= 0.1) & (Z <= 0.5)
        )
        out[m1] = 4.0
        m2 = (
            (X >= 0.3) & (X <= 0.55) & (Y >= 0.2) & (Y <= 0.5) & (Z >= 0.2) & (Z <= 0.4)
        )
        out[m2] = 2.5
        return out

    s_st = SubgridFIT3D(
        mesh,
        eps_r=eps_boxes,
        pec=pec,
        pmc=pmc,
        averaging="staircase",
        dt=1e-11,
        check_stability=False,
    )
    s_sub_box = SubgridFIT3D(
        mesh,
        eps_r=eps_boxes,
        pec=pec,
        pmc=pmc,
        averaging="subpixel",
        dt=1e-11,
        check_stability=False,
    )
    s_sub_fn = SubgridFIT3D(
        mesh,
        eps_r=eps_callable,
        pec=pec,
        pmc=pmc,
        averaging="subpixel",
        n_sub=4,
        dt=1e-11,
        check_stability=False,
    )
    s_mut = SubgridFIT3D(
        mesh,
        eps_r=eps_boxes,
        pec=pec,
        pmc=pmc,
        averaging="subpixel",
        mutate="count_weights",
        dt=1e-11,
        check_stability=False,
    )

    rel_err_box = float(np.max(np.abs(s_sub_box.D / s_st.D - 1.0)))
    rel_err_fn = float(np.max(np.abs(s_sub_fn.D / s_st.D - 1.0)))
    rel_err_mut = float(np.max(np.abs(s_mut.D / s_st.D - 1.0)))

    print(
        f"NUMBERS test_c1_subpixel_node_planes_control: "
        f"rel_err_box={rel_err_box:.3e} rel_err_callable={rel_err_fn:.3e} "
        f"mut_count_weights={rel_err_mut:.3e}"
    )
    assert rel_err_box <= 1e-15
    assert rel_err_fn <= 1e-15
    assert rel_err_mut >= 1e-3


def _run_c2_subpixel_slab(nz_list):
    """Run C2 subpixel vs staircase vs swap_projection slab convergence on `nz_list`."""
    a, b, c = 1.0, 0.7, 0.9
    r = 0.361
    z_s = (
        r * c
    )  # 0.3249 m: not on coarse or 2:1 fine node planes for N_z in {9,13,17,21,27}
    f_ref, lbl = _shoot_1d_slab_modes(
        a, b, c, z_s, lambda _w: 4.0, n_modes=4, f_min=100e6, f_max=260e6
    )
    slab = Box(((-0.1, a + 0.1), (-0.1, b + 0.1), (-0.1, z_s)), value=4.0)

    err_sub, err_st, err_mut = [], [], []
    for nz in nz_list:
        nx = max(6, int(round(nz * 6 / 9)))
        ny = max(6, int(round(nz * 6 / 9)))
        x = np.linspace(0.0, a, nx + 1)
        y = np.linspace(0.0, b, ny + 1)
        z = np.linspace(0.0, c, nz + 1)
        mesh = SubgridMesh3D(x, y, z)
        k1 = max(1, int(np.floor(r * nz)) - 1)
        k2 = min(nz - 1, int(np.ceil(r * nz)) + 1)
        mesh.add_patch((1, 1, k1), (nx // 2 + 1, ny // 2 + 1, k2), ratio=2)

        f_sub = SubgridFIT3D(
            mesh, eps_r=slab, averaging="subpixel", dt=1e-11, check_stability=False
        ).eigenfrequencies(4, f_ref[0])
        f_st = SubgridFIT3D(
            mesh, eps_r=slab, averaging="staircase", dt=1e-11, check_stability=False
        ).eigenfrequencies(4, f_ref[0])
        f_mut = SubgridFIT3D(
            mesh,
            eps_r=slab,
            averaging="subpixel",
            mutate="swap_projection",
            dt=1e-11,
            check_stability=False,
        ).eigenfrequencies(4, f_ref[0])

        err_sub.append(float(np.mean(np.abs(f_sub - f_ref) / f_ref)))
        err_st.append(float(np.mean(np.abs(f_st - f_ref) / f_ref)))
        err_mut.append(float(np.mean(np.abs(f_mut - f_ref) / f_ref)))

    pairwise_sub = [
        float(np.log(err_sub[i] / err_sub[i + 1]) / np.log(nz_list[i + 1] / nz_list[i]))
        for i in range(len(nz_list) - 1)
    ]
    ls_sub = float(-np.polyfit(np.log(nz_list), np.log(err_sub), 1)[0])
    ls_mut = float(-np.polyfit(np.log(nz_list), np.log(err_mut), 1)[0])
    return dict(
        modes=lbl,
        err_sub=err_sub,
        err_st=err_st,
        err_mut=err_mut,
        pairwise_sub=pairwise_sub,
        ls_sub=ls_sub,
        ls_mut=ls_mut,
    )


def test_c2_subpixel_slab_convergence_fast():
    """C2 (fast, N_z in {9, 13, 17}): eps=4 slab not on node planes crossing a 2:1 patch."""
    res = _run_c2_subpixel_slab([9, 13, 17])
    pw_str = ",".join(f"{o:.3f}" for o in res["pairwise_sub"])
    sub_str = ",".join(f"{e:.3e}" for e in res["err_sub"])
    st_str = ",".join(f"{e:.3e}" for e in res["err_st"])
    print(
        f"NUMBERS test_c2_subpixel_slab_convergence_fast: "
        f"modes={','.join(res['modes'])} err_sub={sub_str} err_st={st_str} "
        f"pairwise_ord={pw_str} ls_ord_sub={res['ls_sub']:.3f} ls_ord_mut={res['ls_mut']:.3f}"
    )
    for es, est in zip(res["err_sub"], res["err_st"]):
        assert es < est
    assert res["ls_sub"] >= 1.7
    assert res["ls_mut"] < 1.5


@pytest.mark.slow
def test_c2_subpixel_slab_convergence_slow():
    """C2 (slow, N_z in {9, 13, 17, 21, 27}): 5-level subpixel slab convergence."""
    res = _run_c2_subpixel_slab([9, 13, 17, 21, 27])
    pw_str = ",".join(f"{o:.3f}" for o in res["pairwise_sub"])
    sub_str = ",".join(f"{e:.3e}" for e in res["err_sub"])
    st_str = ",".join(f"{e:.3e}" for e in res["err_st"])
    print(
        f"NUMBERS test_c2_subpixel_slab_convergence_slow: "
        f"modes={','.join(res['modes'])} err_sub={sub_str} err_st={st_str} "
        f"pairwise_ord={pw_str} ls_ord_sub={res['ls_sub']:.3f} ls_ord_mut={res['ls_mut']:.3f}"
    )
    for es, est in zip(res["err_sub"], res["err_st"]):
        assert es < est
    assert res["ls_sub"] >= 1.7
    assert res["ls_mut"] < 1.5
