# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Brief W4-B diagnostic and regression tests for 3D curvilinear hexahedral FIT:
1. Node-level magnetic wall normal residual identity ``m1 <= g + 1e-12``
   (where ``m1 = |B_c . n_v| / ||B_c||`` and ``g = ||n_v x n_f||``) and
   smooth cavity eigenmode ``m3`` convergence across node classes.
2. Exact semi-discrete Yee ``TE101`` dispersion identity at ``amp = 0`` on
   ``F1`` ``n in {6, 8}`` and ``F2`` ``k = 1`` within ``1e-9`` relative error.
3. Perturbed-box error decomposition ``e(amp) = e_Yee + e_dist`` into
   opposite-sign Yee dispersion (``e_Yee < 0``) and mesh-distortion
   (``e_dist > 0``) contributions.
"""

from __future__ import annotations

import numpy as np

from studies.curvilinear3d.bn_wall_diagnosis import (
    classify_boundary_nodes,
    evaluate_eigenmode_m3,
    evaluate_wall_octant_metrics,
)
from studies.curvilinear3d.perturbed_box_order import (
    fit_order,
    local_orders,
    te101_errors,
    yee_te101_omega,
)
from wakis.curvilinear import CurvilinearFIT3D, HexBlock3D
from wakis.curvilinear.mappings3d import (
    box_te101_omega,
    butterfly_disk_nodes,
    cubed_sphere_ball_nodes,
    perturbed_box_nodes,
    sphere_nodes,
)


def test_wall_bn_facet_node_angle_identity():
    """
    Verify Hypothesis H1 for the node-level magnetic wall residual
    ``m1 = |B_c . n_v| / ||B_c||``:

    Derivation
    ----------
    At any boundary octant ``(h, c)`` with a PEC wall face ``f_w`` where
    ``b_{f_w} = 0``, the corner reconstruction ``B_c = R_c b_c`` (Eq. 2.1,
    ``R_c = M_c^{-1} diag(s_c)``) satisfies ``A_{f_w} . B_c = s_{f_w} b_{f_w} = 0``
    to machine precision, so ``m0 = |B_c . n_f| / ||B_c|| <= 1e-12``.
    Decomposing the unit node normal ``n_v`` into components parallel and
    orthogonal to the unit facet normal ``n_f`` gives

    .. math::

        |B_c \\cdot n_v|
        = |B_c \\cdot n_f \\, (n_v \\cdot n_f)
           + B_c \\cdot (n_f \\times (n_v \\times n_f))|
        \\le ||B_c|| \\, (m_0 + ||n_v \\times n_f||)
        = ||B_c|| \\, (m_0 + g),

    hence ``m1 <= g + 1e-12`` on every 1-wall octant (and on multi-wall octants
    where ``B_c`` is orthogonal to all incident wall faces of ``h``). Because
    ``m1 <= m0 + g`` is a Cauchy-Schwarz inequality that holds identically for
    any non-zero vector ``B_c``, it carries no test content beyond ``m0 = 0``
    (``m0 <= 1e-12``); physical evidence for ``B . n -> 0`` on curved walls is
    provided by the decay of the node-averaged eigenmode residual ``m3`` with
    ``n`` in :func:`test_wall_bn_cavity_eigenmode_m3_decay`.

    Mutations (derived before execution)
    ------------------------------------
    - Mutation (a): Do not zero ``b`` on PEC boundary faces (``zero_wall_b=False``).
      Then ``A_{f_w} . B_c = s_{f_w} b_{f_w} != 0``, so ``B_c`` acquires an
      ``O(1)`` normal component along ``n_f``, driving ``m0 = 9.998492e-01 >> 1e-12``
      and ``m1 - g = 8.294986e-01 >> 1e-12`` on ``sphere_nodes(6, 6, 6)``.
    - Mutation (b): Replace ``R_c`` by ``R_c^T`` (``transpose_rc=True``).
      On non-orthogonal ``sphere_nodes(6, 6, 6)`` corners, ``M_c`` is not
      orthogonal (``M_c M_c^T`` is non-diagonal), so ``A_{f_w} . (R_c^T b_c) != 0``
      even when ``b_{f_w} = 0``, giving ``m0 = 1.997607e-01 >> 1e-12``.
    """
    g_sphere_1w = {}
    m1_sphere_1w = {}
    for n in (4, 6, 8):
        sol = CurvilinearFIT3D(HexBlock3D(sphere_nodes(n, n, n, a=1.0)))
        cls, _ = classify_boundary_nodes(sol.mesh)
        stats = evaluate_wall_octant_metrics(sol, cls, seed=2026)
        ow = stats["one_wall_octants"]
        ab = stats["all_boundary_octants"]
        g_sphere_1w[n] = ow["max_g"]
        m1_sphere_1w[n] = ow["max_m1"]
        print(
            f"[test_wall_bn_facet_node_angle_identity] sphere N={n}: "
            f"1-wall m0={ow['max_m0']:.3e}, m1={ow['max_m1']:.6e}, "
            f"g={ow['max_g']:.6e}, m1-g={ow['max_m1_minus_g']:.3e}, "
            f"all_bnd m1-g={ab['max_m1_minus_g']:.3e}"
        )
        assert ab["max_m0"] <= 1e-12
        assert ab["max_m1_minus_g"] <= 1e-12
        assert ow["max_m1"] <= ow["max_g"] + 1e-12

    # Reproduce the exact N=4, 6, 8 facet-to-node normal angle bounds and N=6 m1
    assert np.isclose(g_sphere_1w[4], 2.651614e-01, rtol=1e-5, atol=0.0)
    assert np.isclose(g_sphere_1w[6], 1.845888e-01, rtol=1e-5, atol=0.0)
    assert np.isclose(g_sphere_1w[8], 1.414062e-01, rtol=1e-5, atol=0.0)
    assert np.isclose(m1_sphere_1w[6], 1.845750e-01, rtol=1e-5, atol=0.0)
    assert g_sphere_1w[8] < g_sphere_1w[6] < g_sphere_1w[4]
    assert m1_sphere_1w[8] < m1_sphere_1w[6] < m1_sphere_1w[4]

    # Multi-block HexMesh generators and flat-wall perturbed-box control
    for name, msh_in, is_flat in (
        ("cubed_sphere_4", cubed_sphere_ball_nodes(4, a=1.0), False),
        ("butterfly_disk_4", butterfly_disk_nodes(4, a=1.0, length=1.2), False),
        (
            "perturbed_box_654",
            HexBlock3D(perturbed_box_nodes(6, 5, 4, a=1.0, b=0.7, d=0.8, amp=0.3)),
            True,
        ),
    ):
        sol = CurvilinearFIT3D(msh_in)
        cls, _ = classify_boundary_nodes(sol.mesh)
        stats = evaluate_wall_octant_metrics(sol, cls, seed=2026)
        ab = stats["all_boundary_octants"]
        ow = stats["one_wall_octants"]
        print(
            f"[test_wall_bn_facet_node_angle_identity] {name}: "
            f"1-wall m0={ow['max_m0']:.3e}, m1={ow['max_m1']:.6e}, "
            f"g={ow['max_g']:.6e}, m1-g={ow['max_m1_minus_g']:.3e}"
        )
        assert ab["max_m0"] <= 1e-12
        assert ab["max_m1_minus_g"] <= 1e-12
        if is_flat:
            assert ow["max_m1"] <= 1e-12
            assert ow["max_g"] <= 1e-12

    # Mutation (a): do not zero b on wall faces -> m0, m1 >> g, identity fails
    sol_sph6 = CurvilinearFIT3D(HexBlock3D(sphere_nodes(6, 6, 6, a=1.0)))
    cls_sph6, _ = classify_boundary_nodes(sol_sph6.mesh)
    mut_a = evaluate_wall_octant_metrics(
        sol_sph6, cls_sph6, seed=2026, zero_wall_b=False, transpose_rc=False
    )["one_wall_octants"]
    # Mutation (b): replace R_c by R_c^T -> m0 >> 1e-12 on non-orthogonal corners
    mut_b = evaluate_wall_octant_metrics(
        sol_sph6, cls_sph6, seed=2026, zero_wall_b=True, transpose_rc=True
    )["one_wall_octants"]
    print(
        f"[test_wall_bn_facet_node_angle_identity] mutations on sphere_6: "
        f"mut_a m0={mut_a['max_m0']:.6e}, m1={mut_a['max_m1']:.6e}, "
        f"m1-g={mut_a['max_m1_minus_g']:.6e}; "
        f"mut_b m0={mut_b['max_m0']:.6e}"
    )
    assert mut_a["max_m0"] > 0.5
    assert mut_a["max_m1_minus_g"] > 0.5
    assert mut_b["max_m0"] > 0.1


def test_wall_bn_cavity_eigenmode_m3_decay():
    """
    Verify that the node-averaged magnetic wall normal residual
    ``m3 = |B_v . n| / max_v ||B_v||`` of smooth cavity eigenmodes (``b = C_u W_eps_u d``)
    decreases with resolution ``n`` on ``smooth_wall`` nodes for both
    ``n = n_v`` and ``n = n_exact`` on ``cubed_sphere_ball_nodes`` and
    ``butterfly_disk_nodes``.

    Mutation check: replacing ``b = C_u W_eps_u d`` with an unzeroed wall flux
    breaks ``m3`` smallness (producing ``m3 > 0.25``).
    """
    for fam, gen, kind in (
        (
            "cubed_sphere",
            lambda n: cubed_sphere_ball_nodes(n, a=1.0),
            "sphere",
        ),
        (
            "butterfly_disk",
            lambda n: butterfly_disk_nodes(n, a=1.0, length=1.2),
            "cylinder",
        ),
    ):
        res_by_n = {}
        for n in (4, 6):
            sol = CurvilinearFIT3D(gen(n))
            cls, _ = classify_boundary_nodes(sol.mesh)
            res_by_n[n] = evaluate_eigenmode_m3(
                sol, cls, geometry_kind=kind, a=1.0, length=1.2
            )
        for mode_key in ("TM_lowest", "TE_lowest"):
            sw4 = res_by_n[4][mode_key]["per_class"]["smooth_wall"]
            sw6 = res_by_n[6][mode_key]["per_class"]["smooth_wall"]
            print(
                f"[test_wall_bn_cavity_eigenmode_m3_decay] {fam} {mode_key}: "
                f"n=4 nv/ex={sw4['max_m3_nv']:.4e}/{sw4['max_m3_exact']:.4e} -> "
                f"n=6 nv/ex={sw6['max_m3_nv']:.4e}/{sw6['max_m3_exact']:.4e}"
            )
            assert sw6["max_m3_nv"] < sw4["max_m3_nv"]
            assert sw6["max_m3_exact"] < sw4["max_m3_exact"]

    # Mutation check: adding a non-zero wall-face flux component to b_face
    # destroys the smallness of m3_nv on smooth_wall
    from studies.curvilinear3d.bn_wall_diagnosis import compute_node_averaged_B

    sol4 = CurvilinearFIT3D(butterfly_disk_nodes(4, a=1.0, length=1.2))
    cls4, _ = classify_boundary_nodes(sol4.mesh)
    _, vecs4 = sol4.cavity_frequencies(k=4, return_vectors=True)
    b_mut = sol4.C_u @ (sol4.W_eps_u @ vecs4[:, 0])
    b_scale = float(np.max(np.abs(b_mut)))
    b_mut[sol4.mesh.boundary_faces] = b_scale
    B_v_mut = compute_node_averaged_B(sol4, b_mut)
    sw_nodes = np.array(
        [v for v, c in cls4.items() if c == "smooth_wall"], dtype=np.int64
    )
    B_max_mut = float(np.max(np.linalg.norm(B_v_mut, axis=1)))
    m3_nv_mut = float(
        np.max(
            np.abs(np.sum(B_v_mut[sw_nodes] * sol4.mesh.node_normals[sw_nodes], axis=1))
        )
        / B_max_mut
    )
    print(f"[test_wall_bn_cavity_eigenmode_m3_decay] mutation m3_nv={m3_nv_mut:.4e}")
    assert m3_nv_mut > 0.25


def test_yee_te101_semidiscrete_dispersion_identity():
    """
    Second computation path for ``amp = 0``: on an unperturbed ``nx x ny x nz``
    Cartesian grid over ``[0, a] x [0, b] x [0, d]``, ``CurvilinearFIT3D``
    reduces identically to the diagonal Yee FIT operators (Proposition in
    guide §3), whose exact semi-discrete ``TE101`` angular eigenfrequency is

    .. math::

        \\omega_h^2 = c_0^2 \\left[
            \\left(\\frac{2}{h_x}\\sin\\frac{\\pi h_x}{2 a}\\right)^2
            + \\left(\\frac{2}{h_z}\\sin\\frac{\\pi h_z}{2 d}\\right)^2
        \\right].

    Justification of the ``1e-9`` relative tolerance
    ------------------------------------------------
    ``CurvilinearFIT3D.cavity_frequencies`` solves the generalized symmetric
    eigenvalue problem via shift-invert Lanczos (``scipy.sparse.linalg.eigsh``)
    with default stopping tolerance ``tol = eps_mach ~ 2.22e-16`` on
    ``omega_h^2`` and corner Hodge assembly accurate to ``~1e-15``, yielding
    relative frequency agreement ``|omega_num - omega_h| / omega_h ~ 1e-15``,
    well below ``1e-9``.

    Mutation check
    --------------
    Comparing ``omega_num`` against the continuum frequency
    ``omega_ana = c_0 * pi * sqrt(1/a^2 + 1/d^2)`` instead of ``omega_h``
    yields the second-order Yee dispersion error itself (``1.138407e-02`` at
    ``F1 n = 6``, ``6.413149e-03`` at ``F1 n = 8``, and ``2.550464e-02`` at
    ``F2 k = 1``), all exceeding ``1e-9`` by more than six orders of magnitude.
    """
    a, b, d = 1.0, 0.7, 0.8
    omega_ana = box_te101_omega(a=a, d=d)
    cases = [
        ("F1_n6", (6, 6, 6)),
        ("F1_n8", (8, 8, 8)),
        ("F2_k1", (4, 3, 4)),
    ]

    for label, (nx, ny, nz) in cases:
        _, omegas_num = te101_errors(
            [(nx, ny, nz)],
            amp=0.0,
            amp_mode="cell",
            a=a,
            b=b,
            d=d,
            return_omegas=True,
        )
        w_num = float(omegas_num[0])
        w_yee = yee_te101_omega(nx, nz, a=a, d=d)
        rel_mismatch = abs(w_num - w_yee) / w_yee
        continuum_mismatch = abs(w_num - omega_ana) / omega_ana
        print(
            f"[test_yee_te101_semidiscrete_dispersion_identity] {label} {(nx, ny, nz)}: "
            f"w_num={w_num:.12e}, w_yee={w_yee:.12e}, "
            f"rel_mismatch={rel_mismatch:.3e}, "
            f"mutation_continuum_mismatch={continuum_mismatch:.6e}"
        )
        assert rel_mismatch <= 1e-9
        # Mutation: comparing against continuum omega_ana fails the 1e-9 bound
        assert continuum_mismatch > 5e-3


def test_perturbed_box_error_decomposition():
    """
    Verify the perturbed-box ``TE101`` error decomposition
    ``e(amp) = e_Yee + e_dist`` on ``F1`` ``n in {6, 8, 12}`` at ``amp = 0.3``:

    Derivation of convergence-order bounds
    --------------------------------------
    1. Distortion error ``e_dist = e(amp) - e(0) = O(h^3)`` (``ord_dist >= 3.0``):
       In ``"cell"`` mode (``perturbed_box_nodes``), the smooth interior node
       displacement ``dx_i(x) = amp * h_i * sin(pi x/a) sin(pi y/b) sin(pi z/d)``
       scales with the local cell size ``h_i = L_i / n = O(h)`` and vanishes on
       ``dOmega``. Differentiating ``x -> x + dx(x)`` yields a smooth metric
       (Jacobian) perturbation ``dJ(x) = grad(dx) = O(amp * h) = O(h)``. Because
       the continuum ``TE101`` eigenvalue is invariant under a boundary-fixing
       diffeomorphism, the discrete shift ``e_dist(n)`` arises from the ``O(h^2)``
       second-order corner Hodge truncation error acting on the ``O(h)`` metric
       perturbation, giving ``e_dist(n) = O(h) * O(h^2) = O(h^3) = O(n^{-3})``
       and hence ``ord_dist >= 3.0`` (measured ``ord_dist = 3.404419``, local
       orders ``[3.1308, 3.5777]``).
    2. Total error ``e(amp) = e_Yee + e_dist = O(h^2)`` (``ord_pert >= 1.95``):
       On the unperturbed isotropic grid ``F1`` (``h_x = a/n``, ``h_z = d/n``),
       Eq. (3.3) gives the closed-form semi-discrete Yee dispersion error
       ``e_Yee(n) = sin(pi/(2n)) / (pi/(2n)) - 1 = -(pi^2 / 24) n^{-2} + O(n^{-4}) < 0``
       with second-order rate ``ord_zero = 1.996362`` on ``n in {6, 8, 12}``.
       Since ``0 < e_dist(n) = O(n^{-3})`` has opposite sign to ``e_Yee(n) < 0``
       with relative ratio ``delta(n) = e_dist(n) / |e_Yee(n)| = O(n^{-1})``
       (``delta(6) = 2.7970e-02``, ``delta(12) = 1.0628e-02``), expanding
       ``ln|e(amp, n)| = ln|e_Yee(n)| - delta(n) + O(delta(n)^2)`` shifts the
       effective log-log slope over ``[n_min, n_max] = [6, 12]`` from ``ord_zero``
       by ``-(delta(6) - delta(12)) / ln(12 / 6) = -0.0250`` (and at most
       ``-delta(6) / ln(12 / 6) = -0.0404``), yielding ``ord_pert >= 2.0 - 0.05 = 1.95``
       (measured ``ord_pert = 1.970955``).

    Mutation check (``pec_reconstruction="drop_only"``)
    ---------------------------------------------------
    Switching to ``pec_reconstruction="drop_only"`` leaves unconstrained boundary
    dual-face skew in the ``O(h)`` boundary cell layer, which strictly increases
    ``e_dist`` on ``ns = [6, 8, 12]`` (at ``n = 6``, ``e_dist`` rises from
    ``+3.184109e-04`` to ``+3.228137e-03``, a printed ``10.14x`` ratio) and
    degrades the orders to ``ord_dist_drop = 2.750425 < 3.0`` and
    ``ord_pert_drop = 1.782195 < 1.95`` (and four-point ``F1`` ``{6, 8, 12, 16}``
    slope ``D_drop_only = 1.795576 < 1.8`` vs ``D = 1.974416``).
    """
    ns = [6, 8, 12]
    e_pert = te101_errors(ns, amp=0.3, amp_mode="cell")
    e_zero = te101_errors(ns, amp=0.0, amp_mode="cell")
    e_dist = e_pert - e_zero

    ord_pert = fit_order(ns, e_pert)
    ord_zero = fit_order(ns, e_zero)
    ord_dist = fit_order(ns, e_dist)
    loc_dist = local_orders(ns, e_dist)

    print(
        f"[test_perturbed_box_error_decomposition] F1 ns={ns}: "
        f"e_pert={[f'{x:+.6e}' for x in e_pert]} (LS={ord_pert:.6f}), "
        f"e_Yee={[f'{x:+.6e}' for x in e_zero]} (LS={ord_zero:.6f}), "
        f"e_dist={[f'{x:+.6e}' for x in e_dist]} (LS={ord_dist:.6f}, local={[round(x, 4) for x in loc_dist]})"
    )

    assert np.all(e_zero < 0.0)
    assert np.all(e_dist > 0.0)
    assert np.all(np.abs(e_pert) < np.abs(e_zero))
    assert np.all(np.diff(np.abs(e_dist)) < 0.0)
    assert ord_dist >= 3.0
    assert ord_pert >= 1.95

    # Mutation check: drop_only PEC reconstruction increases e_dist and breaks
    # the derived O(h^3) distortion and O(h^2) total convergence order bounds.
    # The empirical n=6 inflation ratio (10.14x) is printed as a diagnostic.
    e_drop = te101_errors(ns, amp=0.3, amp_mode="cell", pec_reconstruction="drop_only")
    e_dist_drop = e_drop - e_zero
    ord_pert_drop = fit_order(ns, e_drop)
    ord_dist_drop = fit_order(ns, e_dist_drop)
    e_dist_drop_6 = float(e_dist_drop[0])
    print(
        f"[test_perturbed_box_error_decomposition] drop_only ns={ns}: "
        f"e_pert_drop={[f'{x:+.6e}' for x in e_drop]} (LS={ord_pert_drop:.6f}), "
        f"e_dist_drop={[f'{x:+.6e}' for x in e_dist_drop]} (LS={ord_dist_drop:.6f}), "
        f"n=6 e_dist constrained={e_dist[0]:+.6e} vs drop_only={e_dist_drop_6:+.6e} "
        f"(ratio={e_dist_drop_6 / e_dist[0]:.2f}x)"
    )
    assert np.all(e_dist_drop > e_dist)
    assert ord_dist_drop < 3.0
    assert ord_pert_drop < 1.95
