# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification tests for single-block 3D curvilinear hexahedral FIT
(`wakis.curvilinear.mesh3d.HexBlock3D` and
`wakis.curvilinear.solver3d.CurvilinearFIT3D`).
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from wakis.curvilinear.mesh3d import HexBlock3D
from wakis.curvilinear.solver3d import CurvilinearFIT3D


def _deformed_box(nx=4, ny=4, nz=4, lx=1.0, ly=0.9, lz=0.8, amp=0.22):
    """Curvilinear hex block in a box [0, lx] x [0, ly] x [0, lz] with smooth
    sinusoidal interior node displacements up to ``amp * h`` (<= 0.3 h)."""
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    z = np.linspace(0.0, lz, nz + 1)
    X = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    hx, hy, hz = lx / nx, ly / ny, lz / nz
    sx = np.sin(np.pi * X[..., 0] / lx)
    sy = np.sin(np.pi * X[..., 1] / ly)
    sz = np.sin(np.pi * X[..., 2] / lz)
    envelope = sx * sy * sz
    X[..., 0] += amp * hx * envelope * np.cos(0.5 * np.pi * X[..., 1] / ly)
    X[..., 1] += amp * hy * envelope * np.cos(0.5 * np.pi * X[..., 2] / lz)
    X[..., 2] += amp * hz * envelope * np.cos(0.5 * np.pi * X[..., 0] / lx)
    return HexBlock3D(X)


# -------------------------------------------------------------------- test 1
def test_hex_geometry():
    """Unit cube and affine parallelepiped volumes, face areas, corner volumes
    V_c = V_hex / 8 exact; inverted hex raises ValueError."""
    # Unit cube (single cell)
    cube = HexBlock3D.cartesian([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    assert cube.cell_volumes.shape == (1, 1, 1)
    assert np.isclose(cube.cell_volumes[0, 0, 0], 1.0, rtol=0.0, atol=1e-15)
    assert np.allclose(cube.corner_volumes, 0.125, rtol=0.0, atol=1e-15)
    assert np.allclose(cube.corner_jacobians, 1.0, rtol=0.0, atol=1e-15)
    # 6 faces of the unit cube: 2 fx, 2 fy, 2 fz
    expected_af = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(cube.face_areas, expected_af, rtol=0.0, atol=1e-15)
    # 12 boundary edges of the unit cube each have 1 quarter dual face = 0.25
    assert np.allclose(
        np.linalg.norm(cube.dual_face_areas, axis=1), 0.25, rtol=0.0, atol=1e-15
    )

    # Multi-cell parallelepiped via a general invertible affine map x -> A x + b0
    cart = HexBlock3D.cartesian([0.0, 0.4, 1.1], [0.0, 0.5, 0.9, 1.5], [0.0, 0.7, 1.3])
    A = np.array(
        [
            [1.3, 0.25, -0.15],
            [-0.2, 1.1, 0.3],
            [0.1, -0.18, 0.95],
        ]
    )
    b0 = np.array([0.5, -1.2, 0.8])
    detA = float(np.linalg.det(A))
    assert detA > 0.0
    cofA = detA * np.linalg.inv(A).T  # cofactor matrix for area vectors

    para = HexBlock3D(cart.X @ A.T + b0)
    vol_err = np.max(
        np.abs(para.cell_volumes - detA * cart.cell_volumes)
        / (detA * cart.cell_volumes)
    )
    vc_err = np.max(np.abs(para.corner_volumes - para.cell_volumes / 8.0))
    af_ref = cart.face_areas @ cofA.T
    af_err = np.max(np.abs(para.face_areas - af_ref)) / np.max(np.abs(af_ref))
    se_ref = cart.dual_face_areas @ cofA.T
    se_err = np.max(np.abs(para.dual_face_areas - se_ref)) / np.max(np.abs(se_ref))
    print(
        f"[test_hex_geometry] vol_err={vol_err:.3e}, vc_err={vc_err:.3e}, "
        f"af_err={af_err:.3e}, se_err={se_err:.3e}"
    )
    assert vol_err <= 1e-14
    assert vc_err == 0.0
    assert af_err <= 1e-14
    assert se_err <= 1e-14

    # Inverted hex: swap two corners of cell (0, 0, 0)
    X_inv = cube.X.copy()
    X_inv[0, 0, 0], X_inv[1, 0, 0] = (
        X_inv[1, 0, 0].copy(),
        X_inv[0, 0, 0].copy(),
    )
    with pytest.raises(ValueError, match="inverted or degenerate"):
        HexBlock3D(X_inv)

    # Non-finite coordinates must also fail
    X_nan = cube.X.copy()
    X_nan[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HexBlock3D(X_nan)


# -------------------------------------------------------------------- test 2
def test_orthogonal_hodges_equal_yee():
    """Nonuniform Cartesian grid (6x5x4 cells, random spacings) with random
    per-hex eps_r and mu_r (and uniform media): W_nu and W_eps match explicit
    Yee formulas computed directly from the 1D grid spacings to round-off."""
    rng = np.random.default_rng(42)
    nx, ny, nz = 6, 5, 4
    hx = rng.uniform(0.08, 0.25, size=nx)
    hy = rng.uniform(0.09, 0.22, size=ny)
    hz = rng.uniform(0.07, 0.20, size=nz)
    x = np.concatenate([[0.0], np.cumsum(hx)])
    y = np.concatenate([[0.0], np.cumsum(hy)])
    z = np.concatenate([[0.0], np.cumsum(hz)])
    block = HexBlock3D.cartesian(x, y, z)

    # 1) Uniform eps_r, mu_r: W_nu,ff = L_dual_f / (mu A_f) and
    #    W_eps,ee = L_e / (eps A_dual_e)
    eps_u, mu_u = 2.3, 1.7
    s_uni = CurvilinearFIT3D(block, eps_r=eps_u, mu_r=mu_u)

    h_axes = (hx, hy, hz)
    # Dual edge length per 1D node index: half cell at boundary, average at interior
    l_dual_1d = [
        np.concatenate([[0.5 * h[0]], 0.5 * (h[:-1] + h[1:]), [0.5 * h[-1]]])
        for h in h_axes
    ]

    w_nu_yee_uni = []
    w_eps_yee_uni = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        # Faces of normal a: L_dual along a, area along (b, c)
        L_dual_f = l_dual_1d[a][:, None, None]
        A_f = h_axes[b][None, :, None] * h_axes[c][None, None, :]
        # Reorder axes from (a, b, c) to (0, 1, 2)
        perm = np.argsort([a, b, c])
        w_nu_a = np.transpose(L_dual_f / (mu_0 * mu_u * A_f), perm).ravel()
        w_nu_yee_uni.append(w_nu_a)

        # Edges of direction a: length along a, dual area along (b, c)
        L_e = h_axes[a][:, None, None]
        A_dual_e = l_dual_1d[b][None, :, None] * l_dual_1d[c][None, None, :]
        w_eps_a = np.transpose(L_e / (eps_0 * eps_u * A_dual_e), perm).ravel()
        w_eps_yee_uni.append(w_eps_a)

    w_nu_yee_uni = np.concatenate(w_nu_yee_uni)
    w_eps_yee_uni = np.concatenate(w_eps_yee_uni)

    W_nu_u = s_uni.hodge_magnetic()
    W_eps_full_u = s_uni.hodge_electric(restricted=False)
    off_nu_u = (W_nu_u - sp.diags(W_nu_u.diagonal())).nnz
    off_eps_u = (W_eps_full_u - sp.diags(W_eps_full_u.diagonal())).nnz
    err_nu_uni = np.max(np.abs(W_nu_u.diagonal() - w_nu_yee_uni) / w_nu_yee_uni)
    err_eps_uni = np.max(
        np.abs(W_eps_full_u.diagonal() - w_eps_yee_uni) / w_eps_yee_uni
    )

    # 2) Random per-hex eps_r and mu_r
    eps_r = rng.uniform(1.1, 4.5, size=(nx, ny, nz))
    mu_r = rng.uniform(1.0, 3.2, size=(nx, ny, nz))
    solver = CurvilinearFIT3D(block, eps_r=eps_r, mu_r=mu_r)
    W_nu = solver.hodge_magnetic()
    W_eps_full = solver.hodge_electric(restricted=False)
    W_eps_rest = solver.hodge_electric(restricted=True)

    # Explicit orthogonal cell-by-cell Yee formulas from (hx, hy, hz, eps_r, mu_r)
    w_nu_yee = []
    w_eps_yee = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        f_shape = [nx, ny, nz]
        f_shape[a] += 1
        w_f = np.zeros(f_shape, dtype=float)
        for side in (0, 1):
            sl_f = [slice(None), slice(None), slice(None)]
            sl_f[a] = slice(1, f_shape[a]) if side == 1 else slice(0, f_shape[a] - 1)
            # cell thickness along a, face area along (b, c)
            ha = (
                hx[:, None, None]
                if a == 0
                else (hy[None, :, None] if a == 1 else hz[None, None, :])
            )
            hb = (
                hx[:, None, None]
                if b == 0
                else (hy[None, :, None] if b == 1 else hz[None, None, :])
            )
            hc = (
                hx[:, None, None]
                if c == 0
                else (hy[None, :, None] if c == 1 else hz[None, None, :])
            )
            w_f[tuple(sl_f)] += (0.5 * ha) / (mu_0 * mu_r * (hb * hc))
        w_nu_yee.append(w_f.ravel())

        e_shape = [nx, ny, nz]
        e_shape[b] += 1
        e_shape[c] += 1
        a_dual = np.zeros(e_shape, dtype=float)
        inv_eps_a = np.zeros(e_shape, dtype=float)
        ha = (
            hx[:, None, None]
            if a == 0
            else (hy[None, :, None] if a == 1 else hz[None, None, :])
        )
        hb = (
            hx[:, None, None]
            if b == 0
            else (hy[None, :, None] if b == 1 else hz[None, None, :])
        )
        hc = (
            hx[:, None, None]
            if c == 0
            else (hy[None, :, None] if c == 1 else hz[None, None, :])
        )
        quarter_area = 0.25 * hb * hc
        for db in (0, 1):
            for dc in (0, 1):
                sl_e = [slice(None), slice(None), slice(None)]
                sl_e[b] = slice(db, e_shape[b] - 1 + db)
                sl_e[c] = slice(dc, e_shape[c] - 1 + dc)
                a_dual[tuple(sl_e)] += quarter_area
                inv_eps_a[tuple(sl_e)] += quarter_area / (eps_0 * eps_r)
        # Broadcast edge length ha to e_shape
        w_e = (ha / (a_dual**2)) * inv_eps_a
        w_eps_yee.append(w_e.ravel())

    w_nu_yee = np.concatenate(w_nu_yee)
    w_eps_yee = np.concatenate(w_eps_yee)

    off_nu = float(np.max(np.abs((W_nu - sp.diags(W_nu.diagonal())).data), initial=0.0))
    off_eps = float(
        np.max(
            np.abs((W_eps_full - sp.diags(W_eps_full.diagonal())).data),
            initial=0.0,
        )
    )
    err_nu = np.max(np.abs(W_nu.diagonal() - w_nu_yee) / w_nu_yee)
    err_eps_full = np.max(np.abs(W_eps_full.diagonal() - w_eps_yee) / w_eps_yee)
    err_eps_rest = np.max(
        np.abs(W_eps_rest.diagonal() - w_eps_yee[block.unknown_edges])
        / w_eps_yee[block.unknown_edges]
    )
    print(
        f"[test_orthogonal_hodges_equal_yee] err_nu_uni={err_nu_uni:.3e}, "
        f"err_eps_uni={err_eps_uni:.3e}, err_nu={err_nu:.3e}, "
        f"err_eps_full={err_eps_full:.3e}, err_eps_rest={err_eps_rest:.3e}, "
        f"off_nu={off_nu:.3e}, off_eps={off_eps:.3e}"
    )
    assert off_nu_u == 0 and off_eps_u == 0
    assert err_nu_uni <= 1e-12
    assert err_eps_uni <= 1e-12
    assert off_nu <= 1e-12 * np.max(w_nu_yee)
    assert off_eps <= 1e-12 * np.max(w_eps_yee)
    assert err_nu <= 1e-12
    assert err_eps_full <= 1e-12
    assert err_eps_rest <= 1e-12

    # Non-orthogonal check: displacing an interior node produces non-zero
    # off-diagonal Hodge couplings (ensuring the test would fail if diagonal
    # and non-diagonal paths were confused).
    X_skew = block.X.copy()
    X_skew[2, 2, 2] += np.array([0.02, -0.015, 0.018])
    s_skew = CurvilinearFIT3D(HexBlock3D(X_skew), eps_r=eps_r, mu_r=mu_r)
    W_nu_skew = s_skew.hodge_magnetic()
    off_skew = np.max(np.abs((W_nu_skew - sp.diags(W_nu_skew.diagonal())).data))
    assert off_skew > 1e-3 * np.max(W_nu_skew.diagonal())


# -------------------------------------------------------------------- test 3
def test_curl_topology():
    """C entries in {-1, 0, 1}, every row has 4 nonzeros, C_u shape consistent,
    and exact discrete sequence identities D C = 0 and C_u G_u = 0 hold."""
    block = _deformed_box(nx=5, ny=4, nz=3)
    solver = CurvilinearFIT3D(block)
    C = solver.curl(restricted=False)
    C_u = solver.curl(restricted=True)

    assert C.shape == (block.n_faces, block.n_edges)
    assert C_u.shape == (block.n_faces, block.n_unknown_edges)
    assert np.all(np.isin(C.data, [-1.0, 1.0]))
    row_nnz = np.diff(C.indptr)
    assert np.all(row_nnz == 4)
    assert np.all(np.asarray(C.sum(axis=1)).ravel() == 0.0)
    assert (C_u != C[:, block.unknown_edges]).nnz == 0

    D = block.divergence()
    G_u = block.gradient_unknown()
    dc_norm = float(np.max(np.abs((D @ C).data), initial=0.0))
    cg_norm = float(np.max(np.abs((C_u @ G_u).data), initial=0.0))
    # Mutation check: flipping one sign in C breaks D @ C == 0
    C_mut = C.copy()
    C_mut.data[0] *= -1.0
    dc_mut_norm = float(np.max(np.abs((D @ C_mut).data), initial=0.0))
    print(
        f"[test_curl_topology] C.shape={C.shape}, C_u.shape={C_u.shape}, "
        f"max|D C|={dc_norm:.3e}, max|C_u G_u|={cg_norm:.3e}, "
        f"max|D C_mut|={dc_mut_norm:.3e}"
    )
    assert dc_norm == 0.0
    assert cg_norm == 0.0
    assert dc_mut_norm > 0.0


# -------------------------------------------------------------------- test 4
def test_dot_product_step():
    """Exact transpose dot-product test for the full leapfrog step S and for
    the W_eps, W_nu, and curl pieces separately, plus finite-difference check."""
    rng = np.random.default_rng(7)
    block = _deformed_box(nx=4, ny=4, nz=3)
    eps_r = rng.uniform(1.0, 3.5, size=block.nc)
    mu_r = rng.uniform(1.0, 2.5, size=block.nc)
    solver = CurvilinearFIT3D(block, eps_r=eps_r, mu_r=mu_r)
    dt = 0.7 * solver.max_stable_dt()

    b = rng.standard_normal(solver.n_faces)
    yb = rng.standard_normal(solver.n_faces)
    eta_inv = np.sqrt(eps_0 / mu_0)
    d = rng.standard_normal(solver.n_unknown_edges) * eta_inv
    yd = rng.standard_normal(solver.n_unknown_edges) * eta_inv

    # 1) Full step S and transpose_step S^T
    sb, sd = solver.step(b, d, dt=dt)
    tb, td = solver.transpose_step(yb, yd, dt=dt)
    lhs_step = float(yb @ sb + yd @ sd)
    rhs_step = float(tb @ b + td @ d)
    denom_step = np.hypot(np.linalg.norm(yb), np.linalg.norm(yd)) * np.hypot(
        np.linalg.norm(sb), np.linalg.norm(sd)
    )
    rel_step = abs(lhs_step - rhs_step) / denom_step

    # 2) W_eps piece
    W_eps = solver.hodge_electric()
    w_d = W_eps @ d
    wt_yd = W_eps.T @ yd
    rel_weps = abs(float(yd @ w_d - wt_yd @ d)) / (
        np.linalg.norm(yd) * np.linalg.norm(w_d)
    )
    rel_weps_sym = abs(float(yd @ w_d - (W_eps @ yd) @ d)) / (
        np.linalg.norm(yd) * np.linalg.norm(w_d)
    )

    # 3) W_nu piece
    W_nu = solver.hodge_magnetic()
    w_b = W_nu @ b
    wt_yb = W_nu.T @ yb
    rel_wnu = abs(float(yb @ w_b - wt_yb @ b)) / (
        np.linalg.norm(yb) * np.linalg.norm(w_b)
    )
    rel_wnu_sym = abs(float(yb @ w_b - (W_nu @ yb) @ b)) / (
        np.linalg.norm(yb) * np.linalg.norm(w_b)
    )

    # 4) Curl piece C_u and C_u^T
    C_u = solver.curl(restricted=True)
    c_d = C_u @ d
    ct_yb = C_u.T @ yb
    rel_curl = abs(float(yb @ c_d - ct_yb @ d)) / (
        np.linalg.norm(yb) * np.linalg.norm(c_d)
    )

    # 5) Central finite-difference check of the quadratic functional
    #    J(b, d) = 1/2 (||sb||^2 + ||sd||^2) whose gradient is S^T S (b, d)
    gb, gd = solver.transpose_step(sb, sd, dt=dt)
    db_dir = rng.standard_normal(solver.n_faces)
    dd_dir = rng.standard_normal(solver.n_unknown_edges) * eta_inv
    eps_fd = 1e-7
    sb_p, sd_p = solver.step(b + eps_fd * db_dir, d + eps_fd * dd_dir, dt=dt)
    sb_m, sd_m = solver.step(b - eps_fd * db_dir, d - eps_fd * dd_dir, dt=dt)
    j_p = 0.5 * (np.dot(sb_p, sb_p) + np.dot(sd_p, sd_p))
    j_m = 0.5 * (np.dot(sb_m, sb_m) + np.dot(sd_m, sd_m))
    fd_grad = (j_p - j_m) / (2.0 * eps_fd)
    adj_grad = float(gb @ db_dir + gd @ dd_dir)
    rel_fd = abs(fd_grad - adj_grad) / abs(adj_grad)

    # 6) Mutation check: applying transpose sub-steps in un-reversed order fails
    d_bad = yd - dt * (W_eps.T @ (C_u.T @ yb))
    b_bad = yb + dt * (W_nu.T @ (C_u @ d_bad))
    rel_bad = abs(lhs_step - float(b_bad @ b + d_bad @ d)) / denom_step

    print(
        f"[test_dot_product_step] rel_step={rel_step:.3e}, "
        f"rel_weps={rel_weps:.3e}, rel_weps_sym={rel_weps_sym:.3e}, "
        f"rel_wnu={rel_wnu:.3e}, rel_wnu_sym={rel_wnu_sym:.3e}, "
        f"rel_curl={rel_curl:.3e}, rel_fd={rel_fd:.3e}, "
        f"rel_bad_mutation={rel_bad:.3e}"
    )
    assert rel_step <= 1e-14
    assert rel_weps <= 1e-14 and rel_weps_sym <= 1e-14
    assert rel_wnu <= 1e-14 and rel_wnu_sym <= 1e-14
    assert rel_curl <= 1e-14
    assert rel_fd <= 1e-7
    assert rel_bad > 1e-2


# -------------------------------------------------------------------- test 5
def test_energy_conserved_pec_box():
    """PEC box with sinusoidally displaced interior nodes, random initial
    state, 1e4 steps at 0.5 * max_stable_dt(): relative energy drift <= 1e-13."""
    rng = np.random.default_rng(123)
    block = _deformed_box(nx=4, ny=4, nz=4, amp=0.25)
    solver = CurvilinearFIT3D(block)
    dt = 0.5 * solver.max_stable_dt()

    # Scale b and d so magnetic and electric energies have comparable magnitude
    b0 = rng.standard_normal(solver.n_faces)
    d0 = rng.standard_normal(solver.n_unknown_edges) * np.sqrt(
        (b0 @ (solver.W_nu @ b0))
        / (
            d0_raw := rng.standard_normal(solver.n_unknown_edges),
            d0_raw @ (solver.W_eps_u @ d0_raw),
        )[1]
    )
    b, d = b0.copy(), d0.copy()
    e0 = solver.energy(b, d, dt=dt)
    assert e0 > 0.0

    n_steps = 10000
    max_drift = 0.0
    for step_idx in range(n_steps):
        b, d = solver.step(b, d, dt=dt)
        if (step_idx + 1) % 100 == 0 or step_idx == n_steps - 1:
            en = solver.energy(b, d, dt=dt)
            drift = abs(en - e0) / abs(e0)
            max_drift = max(max_drift, drift)

    # Mutation check: an asymmetric Hodge fails to conserve energy
    W_asym = solver.W_eps_u + 0.25 * sp.triu(solver.W_eps_u, 1)
    s_mut = CurvilinearFIT3D(block, W_eps_override=W_asym)
    bm, dm = b0.copy(), d0.copy()
    em0 = s_mut.energy(bm, dm, dt=dt)
    mut_drift = 0.0
    for _ in range(500):
        bm, dm = s_mut.step(bm, dm, dt=dt)
        em = s_mut.energy(bm, dm, dt=dt)
        mut_drift = max(mut_drift, abs(em - em0) / abs(em0))

    print(
        f"[test_energy_conserved_pec_box] max_rel_energy_drift={max_drift:.3e} "
        f"(over {n_steps} steps), asymmetric_mutation_drift={mut_drift:.3e}"
    )
    assert max_drift <= 1e-13
    assert mut_drift > 1e-3


# -------------------------------------------------------------------- test 6
def test_stability_bound():
    """1e4 steps at 0.95 * max_stable_dt() stay bounded (max E <= 1.01 E0);
    at 1.05 * max_stable_dt() energy grows by at least 10x."""
    rng = np.random.default_rng(99)
    block = _deformed_box(nx=4, ny=4, nz=4, amp=0.2)
    solver = CurvilinearFIT3D(block)
    dt_max = solver.max_stable_dt()

    b0 = rng.standard_normal(solver.n_faces)
    d0 = rng.standard_normal(solver.n_unknown_edges)
    scale = np.sqrt((b0 @ (solver.W_nu @ b0)) / (d0 @ (solver.W_eps_u @ d0)))
    d0 = d0 * scale

    n_steps = 10000

    # 1) Stable: dt = 0.95 * dt_max
    dt_stable = 0.95 * dt_max
    b, d = b0.copy(), d0.copy()
    e0_stable = solver.energy(b, d, dt=dt_stable)
    max_ratio_stable = 1.0
    for step_idx in range(n_steps):
        b, d = solver.step(b, d, dt=dt_stable)
        if (step_idx + 1) % 200 == 0 or step_idx == n_steps - 1:
            en = solver.energy(b, d, dt=dt_stable)
            max_ratio_stable = max(max_ratio_stable, en / e0_stable)

    # 2) Unstable: dt = 1.05 * dt_max
    dt_unstable = 1.05 * dt_max
    b, d = b0.copy(), d0.copy()
    e0_unstable = abs(solver.energy(b, d, dt=dt_unstable))
    max_ratio_unstable = 1.0
    for step_idx in range(n_steps):
        b, d = solver.step(b, d, dt=dt_unstable)
        if (step_idx + 1) % 50 == 0 or step_idx == n_steps - 1:
            en = abs(solver.energy(b, d, dt=dt_unstable))
            if not np.isfinite(en):
                max_ratio_unstable = np.inf
                break
            max_ratio_unstable = max(max_ratio_unstable, en / e0_unstable)
            if max_ratio_unstable > 1e6:
                break

    print(
        f"[test_stability_bound] dt_max={dt_max:.4e} s, "
        f"max_ratio_stable(0.95)={max_ratio_stable:.6f}, "
        f"max_ratio_unstable(1.05)={max_ratio_unstable:.3e}"
    )
    assert max_ratio_stable <= 1.01
    assert max_ratio_unstable >= 10.0


# -------------------------------------------------------------------- test 7
def test_cavity_te101_orthogonal():
    """
    PEC box of size a x b x d (a = 1.0 m, b = 0.7 m, d = 0.8 m).

    Analytic fundamental TE101 frequency (since a > d > b):
        omega_exact = c * pi * sqrt(1 / a^2 + 1 / d^2).

    Closed-form discrete Yee/FIT eigenvalue derivation on a uniform Nx x Ny x Nz
    grid with spacings dx = a / Nx, dy = b / Ny, dz = d / Nz:
    The TE101 mode has only the y-component of electric field non-zero,
        e_y(i, j, k) = sin(i * pi / Nx) * sin(k * pi / Nz),
    which vanishes on the PEC walls i in {0, Nx} and k in {0, Nz} and is
    independent of j. Applying the discrete curl C_u produces magnetic fluxes
    b_x and b_z on interior faces, where the diagonal Hodges are uniform:
    (W_eps)_yy = dy / (eps_0 dx dz), (W_nu)_xx = dx / (mu_0 dy dz), and
    (W_nu)_zz = dz / (mu_0 dx dy). The second-difference operators along x
    and z act on sin(i * pi / Nx) and sin(k * pi / Nz) with eigenvalues
        kx_tilde^2 = ( (2 / dx) * sin(pi * dx / (2 * a)) )^2,
        kz_tilde^2 = ( (2 / dz) * sin(pi * dz / (2 * d)) )^2,
    yielding the exact discrete eigenvalue of K = C_u^T W_nu C_u W_eps_u:
        omega_h = c * sqrt(kx_tilde^2 + kz_tilde^2).
    By Taylor expansion of sin(theta) = theta - theta^3 / 6 + O(theta^5),
        omega_h - omega_exact = O(dx^2 + dz^2),
    so the relative error against omega_exact decreases at second order (p = 2)
    under uniform grid refinement.
    """
    a, b, d = 1.0, 0.7, 0.8
    c_em = 1.0 / np.sqrt(eps_0 * mu_0)
    omega_analytic = C0 * np.pi * np.hypot(1.0 / a, 1.0 / d)

    def yee_te101_omega(nx, nz):
        dx, dz = a / nx, d / nz
        kx_t = (2.0 / dx) * np.sin(np.pi * dx / (2.0 * a))
        kz_t = (2.0 / dz) * np.sin(np.pi * dz / (2.0 * d))
        return c_em * np.hypot(kx_t, kz_t)

    # 1) Compare 8x6x8 grid against the closed-form discrete Yee/FIT formula
    block_868 = HexBlock3D.cartesian(
        np.linspace(0.0, a, 9),
        np.linspace(0.0, b, 7),
        np.linspace(0.0, d, 9),
    )
    solver_868 = CurvilinearFIT3D(block_868)
    omega_num_868 = solver_868.cavity_frequencies(k=2)[0]
    omega_yee_868 = yee_te101_omega(8, 8)
    rel_disc_err = abs(omega_num_868 - omega_yee_868) / omega_yee_868

    # 2) Convergence against analytic frequency across two refinements:
    #    (4, 3, 4) -> (8, 6, 8) -> (16, 12, 16)
    grids = [(4, 3, 4), (8, 6, 8), (16, 12, 16)]
    analytic_errs = []
    for nx, ny, nz in grids:
        if (nx, ny, nz) == (8, 6, 8):
            w_num = omega_num_868
        else:
            blk = HexBlock3D.cartesian(
                np.linspace(0.0, a, nx + 1),
                np.linspace(0.0, b, ny + 1),
                np.linspace(0.0, d, nz + 1),
            )
            w_num = CurvilinearFIT3D(blk).cavity_frequencies(k=2)[0]
        assert abs(w_num - yee_te101_omega(nx, nz)) / omega_analytic <= 1e-12
        analytic_errs.append(abs(w_num - omega_analytic) / omega_analytic)

    analytic_errs = np.asarray(analytic_errs)
    orders = np.log2(analytic_errs[:-1] / analytic_errs[1:])
    print(
        f"[test_cavity_te101_orthogonal] omega_analytic={omega_analytic:.6e}, "
        f"omega_8x6x8={omega_num_868:.6e}, rel_disc_err_8x6x8={rel_disc_err:.3e}, "
        f"analytic_errs={analytic_errs.tolist()}, orders={orders.tolist()}"
    )
    assert rel_disc_err <= 1e-12
    assert np.all(np.diff(analytic_errs) < 0.0)
    assert np.all(orders >= 1.8)

    # Mutation check: perturbing eps_r shifts the fundamental frequency
    s_mut = CurvilinearFIT3D(block_868, eps_r=1.05)
    w_mut = s_mut.cavity_frequencies(k=2)[0]
    assert abs(w_mut - omega_yee_868) / omega_yee_868 > 1e-2


# -------------------------------------------------------------------- test 8
def test_mapping_cylinder_corner_jacobian():
    """
    Minimum corner Jacobian of the elliptic cube-to-disk cylinder mapping
    ``x = a u sqrt(1 - v^2 / 2)``, ``y = a v sqrt(1 - u^2 / 2)``, ``z = z``.

    Derivation:
    The continuous Jacobian determinant is
        J_cont(u, v) = a^2 * (1 - (u^2 + v^2) / 2) / sqrt((1 - u^2 / 2)(1 - v^2 / 2)),
    which equals ``0.0`` at the four corners ``(u, v) = (+-1, +-1)`` where the
    adjacent sides of ``[-1, 1]^2`` meet smoothly along the circle ``r = a``
    with interior angle ``pi``. On a finite ``Nx x Ny x Nz`` grid with
    ``du = 2 / Nx``, ``dv = 2 / Ny``, ``dz = L / Nz``, the trilinear corner
    Jacobian at ``(u, v) = (+-1, +-1)`` is given by the secant-chord triple
    product
        J_corner,min = a^2 * dz * (du * dv / 2 - delta(du) * delta(dv)),
    where ``delta(h) = sqrt(1 - (1 - h)^2 / 2) - 1 / sqrt(2)``.
    Measured values on ``a = 1.0 m, L = 1.2 m``:
    - ``N = 4``: ``J_corner,min = 2.186269665969e-02 m^3`` (rel_diff ``9.522e-16``)
    - ``N = 6``: ``J_corner,min = 4.999381340485e-03 m^3`` (rel_diff ``3.470e-16``)
    - ``N = 8``: ``J_corner,min = 1.718682124227e-03 m^3`` (rel_diff ``1.514e-15``)
    """
    from wakis.curvilinear.mappings3d import (
        cylinder_continuous_jacobian,
        cylinder_corner_jacobian_analytic,
        cylinder_nodes,
        min_normalised_corner_det,
    )

    a, length = 1.0, 1.2
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            j_c = cylinder_continuous_jacobian(su, sv, a=a)
            assert j_c == 0.0
    assert np.isclose(
        cylinder_continuous_jacobian(0.0, 0.0, a=a), a * a, rtol=0.0, atol=1e-15
    )

    for n in (4, 6, 8):
        X = cylinder_nodes(n, n, n, a=a, length=length)
        # All lateral boundary nodes lie on x^2 + y^2 = a^2
        for bnd_slice in (X[0, :, :], X[-1, :, :], X[:, 0, :], X[:, -1, :]):
            r_bnd = np.hypot(bnd_slice[..., 0], bnd_slice[..., 1])
            assert np.allclose(r_bnd, a, rtol=0.0, atol=1e-14)

        block = HexBlock3D(X)
        j_min_meas = float(np.min(block.corner_jacobians))
        j_min_pred = cylinder_corner_jacobian_analytic(n, n, n, a=a, length=length)
        rel_diff = abs(j_min_meas - j_min_pred) / j_min_pred
        min_det = min_normalised_corner_det(block)
        print(
            f"[test_mapping_cylinder_corner_jacobian] N={n}: "
            f"j_min_meas={j_min_meas:.12e}, j_min_pred={j_min_pred:.12e}, "
            f"rel_diff={rel_diff:.3e}, min_normalised_det={min_det:.6e}"
        )
        assert j_min_meas > 0.0
        assert rel_diff <= 1e-14

    # Mutation check: perturbing the corner coordinate breaks the analytic match
    X_mut = cylinder_nodes(8, 8, 8, a=a, length=length)
    X_mut[0, 0, :, 0] *= 0.98
    j_mut = float(np.min(HexBlock3D(X_mut).corner_jacobians))
    j_ref = cylinder_corner_jacobian_analytic(8, 8, 8, a=a, length=length)
    assert abs(j_mut - j_ref) / j_ref > 1e-2


# -------------------------------------------------------------------- test 9
def test_perturbed_box_order_small():
    """
    Convergence of the fundamental ``TE101`` cavity frequency on a smoothly
    perturbed box ``[0, a] x [0, b] x [0, d]`` (``a = 1.0 m, b = 0.7 m, d = 0.8 m``)
    with interior displacement ``dx_i = amp * h_i * sin(pi x/a) sin(pi y/b) sin(pi z/d)``
    at amplitude ``amp = 0.3`` from ``4 x 3 x 4`` to ``8 x 6 x 8`` cells.

    Measured values:
    - ``omega_analytic = 1.50765688e+09 rad/s``
    - ``4 x 3 x 4``: ``omega_h = 1.48137014e+09 rad/s``, ``rel_error = 1.743549e-02``
    - ``8 x 6 x 8``: ``omega_h = 1.50088898e+09 rad/s``, ``rel_error = 4.489016e-03``
    - Observed order ``log2(err_434 / err_868) = 1.957555 >= 1.8``.
    """
    from wakis.curvilinear.mappings3d import (
        box_te101_omega,
        min_normalised_corner_det,
        perturbed_box_nodes,
    )

    a, b, d, amp = 1.0, 0.7, 0.8, 0.3
    omega_ana = box_te101_omega(a=a, d=d)

    X_434 = perturbed_box_nodes(4, 3, 4, a=a, b=b, d=d, amp=amp)
    X_868 = perturbed_box_nodes(8, 6, 8, a=a, b=b, d=d, amp=amp)

    # Boundary nodes must remain fixed on the 6 rectangular walls
    for X_grid, (nx, ny, nz) in ((X_434, (4, 3, 4)), (X_868, (8, 6, 8))):
        assert np.allclose(X_grid[0, :, :, 0], 0.0, rtol=0.0, atol=1e-15)
        assert np.allclose(X_grid[-1, :, :, 0], a, rtol=0.0, atol=1e-15)
        assert np.allclose(X_grid[:, 0, :, 1], 0.0, rtol=0.0, atol=1e-15)
        assert np.allclose(X_grid[:, -1, :, 1], b, rtol=0.0, atol=1e-15)
        assert np.allclose(X_grid[:, :, 0, 2], 0.0, rtol=0.0, atol=1e-15)
        assert np.allclose(X_grid[:, :, -1, 2], d, rtol=0.0, atol=1e-15)

    # At the center node (4, 3, 4) of the 8x6x8 grid, sin(pi/2)^3 == 1
    assert np.allclose(
        X_868[4, 3, 4] - np.array([0.5 * a, 0.5 * b, 0.5 * d]),
        np.array([amp * a / 8.0, amp * b / 6.0, amp * d / 8.0]),
        rtol=0.0,
        atol=1e-15,
    )

    blk_434 = HexBlock3D(X_434)
    blk_868 = HexBlock3D(X_868)
    w_434 = float(CurvilinearFIT3D(blk_434).cavity_frequencies(k=2)[0])
    w_868 = float(CurvilinearFIT3D(blk_868).cavity_frequencies(k=2)[0])
    err_434 = abs(w_434 - omega_ana) / omega_ana
    err_868 = abs(w_868 - omega_ana) / omega_ana
    order = float(np.log2(err_434 / err_868))

    print(
        f"[test_perturbed_box_order_small] omega_ana={omega_ana:.8e}, "
        f"w_434={w_434:.8e} (err={err_434:.6e}, min_det={min_normalised_corner_det(blk_434):.6e}), "
        f"w_868={w_868:.8e} (err={err_868:.6e}, min_det={min_normalised_corner_det(blk_868):.6e}), "
        f"order={order:.6f}"
    )
    assert err_868 < err_434
    assert order >= 1.8

    # Mutation check: displacing the outer wall ruins convergence to omega_ana
    X_868_mut = X_868.copy()
    X_868_mut[-1, :, :, 0] += 0.05 * a
    w_868_mut = float(
        CurvilinearFIT3D(HexBlock3D(X_868_mut)).cavity_frequencies(k=2)[0]
    )
    assert abs(w_868_mut - omega_ana) / omega_ana > err_434


# -------------------------------------------------------------------- test 10
def test_cylinder_tm010_coarse():
    """
    Coarse-mesh ``TM010`` resonance on the circular cylinder (``a = 1.0 m``,
    ``L = 1.2 m``) compared with ``omega_ana = c * x_01 / a``.

    Measured values across two refinement levels ``N = 4`` and ``N = 8`` (and
    intermediate ``N = 6``):
    - ``omega_analytic = 7.20948565e+08 rad/s``
    - ``4 x 4 x 4``: ``omega_4 = 6.66164437e+08 rad/s``, ``rel_err_4 = 7.598895e-02``
    - ``6 x 6 x 6``: ``omega_6 = 6.95843079e+08 rad/s``, ``rel_err_6 = 3.482285e-02``
    - ``8 x 8 x 8``: ``omega_8 = 7.06542605e+08 rad/s``, ``rel_err_8 = 1.998195e-02``
    - Observed two-level order: ``p = log2(rel_err_4 / rel_err_8) = 1.927092 >= 1.8``.

    Tolerance justification from measured order:
    With asymptotic convergence order ``p >= 1.8``, halving the mesh spacing
    from ``N = 4`` to ``N = 8`` reduces the discretisation error by at least
    ``2^1.8``, giving the derived upper bound ``tol_8 = rel_err_4 * 2^(-1.8) = 2.182210e-02``,
    and the two-level power-law interpolant ``rel_err_4 * (4 / 6)^p = 3.478615e-02``
    predicts ``rel_err_6`` to within ``2e-3`` relative difference.
    """
    from wakis.curvilinear.mappings3d import cylinder_nodes, cylinder_tm010_omega

    a, length = 1.0, 1.2
    omega_ana = cylinder_tm010_omega(a=a)

    w_4 = float(
        CurvilinearFIT3D(
            HexBlock3D(cylinder_nodes(4, 4, 4, a=a, length=length))
        ).cavity_frequencies(k=2)[0]
    )
    w_6 = float(
        CurvilinearFIT3D(
            HexBlock3D(cylinder_nodes(6, 6, 6, a=a, length=length))
        ).cavity_frequencies(k=2)[0]
    )
    w_8 = float(
        CurvilinearFIT3D(
            HexBlock3D(cylinder_nodes(8, 8, 8, a=a, length=length))
        ).cavity_frequencies(k=2)[0]
    )

    err_4 = abs(w_4 - omega_ana) / omega_ana
    err_6 = abs(w_6 - omega_ana) / omega_ana
    err_8 = abs(w_8 - omega_ana) / omega_ana
    order_4_8 = float(np.log2(err_4 / err_8))
    tol_8 = float(err_4 * (0.5**1.8))
    pred_err_6 = float(err_4 * ((4.0 / 6.0) ** order_4_8))

    print(
        f"[test_cylinder_tm010_coarse] omega_ana={omega_ana:.8e}, "
        f"w_4={w_4:.8e} (err_4={err_4:.6e}), "
        f"w_6={w_6:.8e} (err_6={err_6:.6e}, pred_err_6={pred_err_6:.6e}), "
        f"w_8={w_8:.8e} (err_8={err_8:.6e}, tol_8={tol_8:.6e}), "
        f"order_4_8={order_4_8:.6f}"
    )
    assert err_8 < err_6 < err_4
    assert order_4_8 >= 1.8
    assert err_8 <= tol_8
    assert abs(err_6 - pred_err_6) / pred_err_6 <= 2e-3

    # Mutation check: perturbing cylinder radius violates tol_8
    w_8_mut = float(
        CurvilinearFIT3D(
            HexBlock3D(cylinder_nodes(8, 8, 8, a=1.06 * a, length=length))
        ).cavity_frequencies(k=2)[0]
    )
    assert abs(w_8_mut - omega_ana) / omega_ana > tol_8


# -------------------------------------------------------------------- test 11
def test_sphere_lowest_roots():
    """
    Verify that the spherical Bessel root finder reproduces the lowest ``l = 1``
    roots of ``d/dx[x j_1(x)] = 0`` and ``j_1(x) = 0`` to characteristic
    equation residual ``<= 1e-10``, and that the 3D elliptic cube-to-ball
    mapping ``sphere_nodes`` maps all 6 boundary faces onto ``r = a`` with
    positive corner Jacobians.

    Measured values:
    - ``x_d_xj1 = 2.743707269992269``, ``|d/dx[x j_1(x)]| = 2.220e-16 <= 1e-10``
    - ``x_j1 = 4.493409457909064``, ``|j_1(x)| = 6.177e-18 <= 1e-10``
    """
    from scipy.special import spherical_jn

    from wakis.curvilinear.mappings3d import (
        min_normalised_corner_det,
        sphere_bessel_residual,
        sphere_lowest_omegas,
        sphere_lowest_roots,
        sphere_nodes,
        sphere_riccati_bessel_deriv_residual,
    )

    x1, x2 = sphere_lowest_roots(l=1)
    res1 = abs(float(spherical_jn(1, x1) + x1 * spherical_jn(1, x1, derivative=True)))
    res2 = abs(float(spherical_jn(1, x2)))
    assert np.isclose(
        res1,
        abs(float(sphere_riccati_bessel_deriv_residual(x1, l=1))),
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        res2, abs(float(sphere_bessel_residual(x2, l=1))), rtol=0.0, atol=1e-15
    )

    a = 1.0
    w1, w2 = sphere_lowest_omegas(a=a, l=1)
    assert np.isclose(w1, C0 * x1 / a, rtol=0.0, atol=1e-6)
    assert np.isclose(w2, C0 * x2 / a, rtol=0.0, atol=1e-6)

    X_sph = sphere_nodes(6, 6, 6, a=a)
    for face in (
        X_sph[0, :, :],
        X_sph[-1, :, :],
        X_sph[:, 0, :],
        X_sph[:, -1, :],
        X_sph[:, :, 0],
        X_sph[:, :, -1],
    ):
        r_face = np.linalg.norm(face, axis=-1)
        assert np.allclose(r_face, a, rtol=0.0, atol=1e-14)

    blk_sph = HexBlock3D(X_sph)
    min_jac_sph = float(np.min(blk_sph.corner_jacobians))
    min_det_sph = min_normalised_corner_det(blk_sph)
    print(
        f"[test_sphere_lowest_roots] x1={x1:.15f} (res1={res1:.3e}), "
        f"x2={x2:.15f} (res2={res2:.3e}), "
        f"sphere_6x6x6_min_jac={min_jac_sph:.6e}, min_norm_det={min_det_sph:.6e}"
    )
    assert 2.0 < x1 < np.pi < x2 < 2.0 * np.pi
    assert res1 <= 1e-10
    assert res2 <= 1e-10
    assert min_jac_sph > 0.0
    assert 0.0 < min_det_sph < 1.0

    # Mutation check: perturbing the roots makes the residuals exceed 1e-4
    assert abs(float(sphere_riccati_bessel_deriv_residual(x1 + 1e-3, l=1))) > 1e-4
    assert abs(float(sphere_bessel_residual(x2 + 1e-3, l=1))) > 1e-4
