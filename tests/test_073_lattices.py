# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for triangular/honeycomb lattice tools, PWE solver, unfolding, and Berry phase."""

from __future__ import annotations

import numpy as np
import pytest

from wakis.lattices import (
    CircularInclusion,
    HexagonalLattice,
    PWE2D,
    berry_curvature_from_envelopes,
    dirac_berry_curvature,
    fit_berry_curvature_patch,
    fit_supercell_bands_at_k,
    translate_supercell_field,
    unfold_supercell_weight,
)


def test_hexagonal_lattice_geometry_and_bloch_phases():
    """Verify hexagonal Bravais/reciprocal vectors, supercell geometry, and Bloch phases."""
    a = 1.5e-6
    lat = HexagonalLattice(a=a)

    # Primitive and reciprocal orthogonality: a_i . b_j = 2*pi * delta_ij
    mat_a = np.column_stack([lat.a1, lat.a2])
    mat_b = np.column_stack([lat.b1, lat.b2])
    prod = mat_a.T @ mat_b
    np.testing.assert_allclose(prod, 2.0 * np.pi * np.eye(2), atol=1e-12)

    # Centred-rectangular supercell (a x sqrt(3)*a) has twice the primitive area
    a_prim = abs(lat.a1[0] * lat.a2[1] - lat.a1[1] * lat.a2[0])
    lx, ly = lat.supercell_size
    assert np.isclose(lx, a)
    assert np.isclose(ly, np.sqrt(3.0) * a)
    assert np.isclose(lx * ly, 2.0 * a_prim)

    # Folding vector G_fold satisfies exp(i G_fold . tau) = -1
    phase_fold = np.exp(1j * float(np.dot(lat.g_fold, lat.tau)))
    assert np.isclose(phase_fold, -1.0, atol=1e-12)

    # Bloch phases at high-symmetry points
    hs = lat.high_symmetry_points()
    phi_gamma = lat.bloch_phases(hs["Gamma"])
    assert np.allclose(phi_gamma, (1.0 + 0.0j, 1.0 + 0.0j))

    phi_k = lat.bloch_phases(hs["K"])
    assert np.isclose(phi_k[0], np.exp(1j * 4.0 * np.pi / 3.0))
    assert np.isclose(phi_k[1], 1.0 + 0.0j)

    # Gamma-M-K-Gamma path
    path = lat.bz_path(("Gamma", "M", "K", "Gamma"), n_per_segment=8)
    assert path.k_points.shape == (25, 2)
    assert path.tick_labels == ("Gamma", "M", "K", "Gamma")
    assert len(path.tick_positions) == 4
    # Irreducible wedge edge lengths: |Gamma-M| = 2*pi/(sqrt(3)*a), |M-K| = 2*pi/(3*a), |K-Gamma| = 4*pi/(3*a)
    expected_ticks = np.cumsum(
        [
            0.0,
            2.0 * np.pi / (np.sqrt(3.0) * a),
            2.0 * np.pi / (3.0 * a),
            4.0 * np.pi / (3.0 * a),
        ]
    )
    np.testing.assert_allclose(path.tick_positions, expected_ticks, rtol=1e-12)

    # Placement helpers
    tri_sites = lat.triangular_supercell_sites(radius=0.35 * a, eps_in=1.0)
    assert len(tri_sites) == 2
    hc_sites = lat.honeycomb_supercell_sites(r_a=0.2 * a, r_b=0.16 * a, eps_in=12.0)
    assert len(hc_sites) == 4
    gen_sites = lat.generate_lattice_inclusions(
        range(-1, 2),
        range(-1, 2),
        lat.triangular_primitive_sites(0.3 * a, 1.0),
        clip_bounds=(-0.1 * a, 1.1 * a, -0.1 * a, 1.1 * a),
    )
    assert len(gen_sites) >= 2

    with pytest.raises(ValueError, match="positive"):
        HexagonalLattice(a=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        CircularInclusion(x=0.0, y=0.0, radius=-0.1, eps_in=2.0)


def test_pwe_supercell_unfolding_and_mutation():
    """Verify that supercell band unfolding reproduces primitive-cell bands and fails on sign mutation."""
    lat = HexagonalLattice(a=1.0)
    r_hole = 0.35
    eps_bg = 12.0

    path = lat.bz_path(("Gamma", "M", "K", "Gamma"), n_per_segment=4)
    A1, A2 = lat.supercell_vectors

    for pol in ("TM", "TE"):
        pwe_prim = PWE2D(
            lat.a1,
            lat.a2,
            lat.triangular_primitive_sites(r_hole, eps_in=1.0, offset=(0.25, 0.25 * np.sqrt(3.0))),
            eps_bg=eps_bg,
            n_pw=5,
            polarization=pol,
        )
        pwe_sup = PWE2D(
            A1,
            A2,
            lat.triangular_supercell_sites(r_hole, eps_in=1.0, origin=(0.25, 0.25 * np.sqrt(3.0))),
            eps_bg=eps_bg,
            n_pw=5,
            polarization=pol,
        )

        res_prim = pwe_prim.solve_path(path.k_points, n_bands=3)["frequencies"]
        res_sup = pwe_sup.solve_path(
            path.k_points, n_bands=6, unfold_supercell=True, lattice_a=1.0, phase_sign=1
        )
        np.testing.assert_allclose(res_sup["unfolded_frequencies"], res_prim, atol=1e-7)

    # Also test spatial field unfolding weight on a reconstructed supercell Bloch wave at K
    k_test = lat.high_symmetry_points()["K"]
    pwe_sup_tm = PWE2D(
        A1,
        A2,
        lat.triangular_supercell_sites(r_hole, eps_in=1.0),
        eps_bg=eps_bg,
        n_pw=5,
        polarization="TM",
    )
    freqs_k, vecs_k = pwe_sup_tm.solve_k(k_test, n_bands=4, return_vectors=True)
    _, _, psi_0 = pwe_sup_tm.reconstruct_field(vecs_k[:, 0], 24, 40, k=k_test, bloch=True)
    _, _, psi_1 = pwe_sup_tm.reconstruct_field(vecs_k[:, 1], 24, 40, k=k_test, bloch=True)

    w0 = unfold_supercell_weight(psi_0, k_test, a=1.0, is_envelope=False, phase_sign=1)
    w1 = unfold_supercell_weight(psi_1, k_test, a=1.0, is_envelope=False, phase_sign=1)
    # At K, one of the lowest two supercell modes is primitive (w ~ 1) and the other is folded from Gamma (w ~ 0)
    assert max(w0, w1) > 0.999
    assert min(w0, w1) < 0.001

    # Mutation check: flipping phase_sign=-1 swaps unfolded and folded subspaces,
    # causing unfolded_frequencies to mismatch the primitive bands (at K the folded mode is at omega=0!)
    res_mut = pwe_sup_tm.solve_path(
        path.k_points, n_bands=6, unfold_supercell=True, lattice_a=1.0, phase_sign=-1
    )
    err_mut = float(np.max(np.abs(res_mut["unfolded_frequencies"] - pwe_prim.solve_path(path.k_points, n_bands=3)["frequencies"])))
    assert err_mut > 0.05, f"Mutated unfolding phase should fail to match primitive bands, got err={err_mut}"


def test_triangular_air_holes_te_tm_gaps_and_fit_vs_pwe():
    """Validate triangular-lattice air-hole TE/TM bandgaps and 2D FIT (harminv) vs PWE convergence."""
    lat = HexagonalLattice(a=1.0)
    # 1. Verify complete overlapping TE and TM bandgap for r/a = 0.43, eps_bg = 13.0
    path = lat.bz_path(("Gamma", "M", "K", "Gamma"), n_per_segment=5)
    inc_prim = lat.triangular_primitive_sites(radius=0.43, eps_in=1.0)
    pwe_te = PWE2D(lat.a1, lat.a2, inc_prim, eps_bg=13.0, n_pw=5, polarization="TE")
    pwe_tm = PWE2D(lat.a1, lat.a2, inc_prim, eps_bg=13.0, n_pw=5, polarization="TM")

    bands_te = pwe_te.solve_path(path.k_points, n_bands=4)["frequencies"]
    bands_tm = pwe_tm.solve_path(path.k_points, n_bands=4)["frequencies"]

    # TE fundamental gap between band 1 and band 2
    te_gap_low = float(np.max(bands_te[:, 0]))
    te_gap_high = float(np.min(bands_te[:, 1]))
    assert te_gap_high > te_gap_low + 0.08

    # TM gap between band 2 and band 3 overlaps with TE gap
    tm_gap_low = float(np.max(bands_tm[:, 1]))
    tm_gap_high = float(np.min(bands_tm[:, 2]))
    assert tm_gap_high > tm_gap_low + 0.02
    overlap = min(te_gap_high, tm_gap_high) - max(te_gap_low, tm_gap_low)
    assert overlap > 0.02

    # 2. Validate 2D FIT supercell + unfolding vs PWE at K for r/a = 0.35, eps_bg = 12.0
    k_pt = lat.high_symmetry_points()["K"]
    inc_sup = lat.triangular_supercell_sites(radius=0.35, eps_in=1.0)
    pwe_ref_tm = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(0.35, 1.0), eps_bg=12.0, n_pw=9, polarization="TM")
    pwe_ref_te = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(0.35, 1.0), eps_bg=12.0, n_pw=9, polarization="TE")
    f_ref_tm = float(pwe_ref_tm.solve_k(k_pt, n_bands=2)[0])
    f_ref_te = float(pwe_ref_te.solve_k(k_pt, n_bands=2)[0])

    fit_coarse_tm = fit_supercell_bands_at_k(
        lat, inc_sup, k_pt, polarization="TM", nx=10, ny=16, eps_bg=12.0, f_min=0.15, f_max=0.45, n_steps=800, n_sub=1
    )
    fit_fine_tm = fit_supercell_bands_at_k(
        lat, inc_sup, k_pt, polarization="TM", nx=24, ny=42, eps_bg=12.0, f_min=0.15, f_max=0.45, n_steps=1920, n_sub=8
    )
    fit_fine_te = fit_supercell_bands_at_k(
        lat, inc_sup, k_pt, polarization="TE", nx=24, ny=42, eps_bg=12.0, f_min=0.15, f_max=0.45, n_steps=1920, n_sub=8
    )

    err_coarse_tm = float(np.min(np.abs(fit_coarse_tm["unfolded_frequencies"] - f_ref_tm)) / f_ref_tm)
    err_fine_tm = float(np.min(np.abs(fit_fine_tm["unfolded_frequencies"] - f_ref_tm)) / f_ref_tm)
    err_fine_te = float(np.min(np.abs(fit_fine_te["unfolded_frequencies"] - f_ref_te)) / f_ref_te)

    assert err_fine_tm < 0.008, f"Fine TM FIT band error {err_fine_tm:.4f} exceeds 0.8%"
    assert err_fine_te < 0.010, f"Fine TE FIT band error {err_fine_te:.4f} exceeds 1.0%"
    assert err_fine_tm < err_coarse_tm

    # Mutation check: flipping phase_sign=-1 in fit_supercell_bands_at_k projects onto the folded
    # branch instead of the primitive branch, so the fundamental primitive mode at f_ref_tm is absent
    fit_mut_tm = fit_supercell_bands_at_k(
        lat, inc_sup, k_pt, polarization="TM", nx=24, ny=42, eps_bg=12.0, f_min=0.15, f_max=0.45, n_steps=2000, phase_sign=-1
    )
    if fit_mut_tm["unfolded_frequencies"].size > 0:
        err_mut = float(np.min(np.abs(fit_mut_tm["unfolded_frequencies"] - f_ref_tm)) / f_ref_tm)
        assert err_mut > 0.03, f"Mutated FIT signal unfolding should fail (< 0.8% check), got err={err_mut}"


def test_honeycomb_dirac_cone_and_valley_hall_gap_scaling():
    """Verify massless Dirac point at K for delta=0 and linear gap opening vs |delta|."""
    lat = HexagonalLattice(a=1.0)
    k_pt = lat.high_symmetry_points()["K"]
    r_0 = 0.18
    eps_rod = 12.0

    # Symmetric honeycomb (delta = 0): Dirac degeneracy at K between bands 1 and 2
    pwe_sym = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r_0, r_0, eps_rod),
        eps_bg=1.0,
        n_pw=5,
        polarization="TM",
    )
    f_sym = pwe_sym.solve_k(k_pt, n_bands=3)
    assert abs(f_sym[1] - f_sym[0]) < 1e-4

    # Gap opening vs asymmetry delta
    deltas = np.array([0.04, 0.08, 0.12, 0.16])
    gaps = []
    for d in deltas:
        pwe_d = PWE2D(
            lat.a1,
            lat.a2,
            lat.honeycomb_primitive_sites(r_0 * (1.0 + d), r_0 * (1.0 - d), eps_rod),
            eps_bg=1.0,
            n_pw=5,
            polarization="TM",
        )
        fk = pwe_d.solve_k(k_pt, n_bands=2)
        gaps.append(float(fk[1] - fk[0]))
    gaps_arr = np.asarray(gaps)

    # Strictly increasing and linear through the origin
    assert np.all(np.diff(gaps_arr) > 0.0)
    slope = float(np.dot(deltas, gaps_arr) / np.dot(deltas, deltas))
    r_sq = 1.0 - float(np.sum((gaps_arr - slope * deltas) ** 2) / np.sum((gaps_arr - np.mean(gaps_arr)) ** 2))
    assert r_sq > 0.998


def _assert_berry_patch_matches_pwe(
    c_fit_pos: float,
    c_fit_neg: float,
    c_fit_zero: float,
    c_pwe_pos: float,
    c_pwe_neg: float,
    rtol: float = 0.05,
) -> None:
    """Assert 2D FIT Berry patch flux matches PWE within rtol and vanishes at delta=0."""
    rel_pos = abs(c_fit_pos - c_pwe_pos) / max(abs(c_pwe_pos), 1e-12)
    rel_neg = abs(c_fit_neg - c_pwe_neg) / max(abs(c_pwe_neg), 1e-12)
    assert rel_pos < rtol, (
        f"FIT C_patch(+delta) = {c_fit_pos:+.4f} deviates from PWE {c_pwe_pos:+.4f} by {rel_pos:.2%} (>= {rtol:.2%})"
    )
    assert rel_neg < rtol, (
        f"FIT C_patch(-delta) = {c_fit_neg:+.4f} deviates from PWE {c_pwe_neg:+.4f} by {rel_neg:.2%} (>= {rtol:.2%})"
    )
    assert np.isclose(c_fit_pos, -c_fit_neg, rtol=1e-5)
    assert abs(c_fit_zero) < 1e-4, f"Offset delta=0 patch should have ~0 Berry flux, got {c_fit_zero:+.4e}"


def test_berry_curvature_and_valley_chern_sign_flip_pwe_and_fit(monkeypatch: pytest.MonkeyPatch):
    """Verify Berry curvature magnitude & sign flip in Dirac model, PWE, and 2D FIT, plus code-path mutations."""
    lat = HexagonalLattice(a=1.0)
    hs = lat.high_symmetry_points()
    k_val = hs["K"]
    kp_val = hs["K'"]
    r_0 = 0.18
    eps_rod = 12.0

    # 1. Analytical Dirac Berry curvature sign flip
    q_lin = np.linspace(-0.5, 0.5, 21)
    QX, QY = np.meshgrid(q_lin, q_lin, indexing="ij")
    om_pos = dirac_berry_curvature(QX, QY, v_d=0.176, delta_omega=+0.056, valley=+1)
    om_neg = dirac_berry_curvature(QX, QY, v_d=0.176, delta_omega=-0.056, valley=+1)
    om_kp = dirac_berry_curvature(QX, QY, v_d=0.176, delta_omega=+0.056, valley=-1)
    assert np.all(om_pos < 0.0)
    np.testing.assert_allclose(om_neg, -om_pos, rtol=1e-12)
    np.testing.assert_allclose(om_kp, -om_pos, rtol=1e-12)

    # 2. PWE Berry curvature around K and K'
    dk = 0.10 * (2.0 * np.pi)
    kx_k = k_val[0] + np.linspace(-dk, dk, 5)
    ky_k = k_val[1] + np.linspace(-dk, dk, 5)
    kx_kp = kp_val[0] + np.linspace(-dk, dk, 5)
    ky_kp = kp_val[1] + np.linspace(-dk, dk, 5)

    pwe_pos = PWE2D(
        lat.a1, lat.a2, lat.honeycomb_primitive_sites(r_0 * 1.12, r_0 * 0.88, eps_rod), eps_bg=1.0, n_pw=4, polarization="TM"
    )
    pwe_neg = PWE2D(
        lat.a1, lat.a2, lat.honeycomb_primitive_sites(r_0 * 0.88, r_0 * 1.12, eps_rod), eps_bg=1.0, n_pw=4, polarization="TM"
    )

    om_pwe_pos, c_pwe_pos = pwe_pos.berry_curvature(kx_k, ky_k, band_index=0)
    om_pwe_neg, c_pwe_neg = pwe_neg.berry_curvature(kx_k, ky_k, band_index=0)
    _, c_pwe_kp = pwe_pos.berry_curvature(kx_kp, ky_kp, band_index=0)

    assert c_pwe_pos < -0.15
    assert c_pwe_neg > +0.15
    assert np.isclose(c_pwe_pos, -c_pwe_neg, rtol=1e-2)
    assert np.isclose(c_pwe_pos, -c_pwe_kp, rtol=1e-2)

    # 3. 2D FIT Berry curvature patch around K for delta = +0.12, -0.12, and offset 0.0
    fit_pos = fit_berry_curvature_patch(lat, r_0=r_0, delta=+0.12, eps_rod=eps_rod, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000)
    fit_neg = fit_berry_curvature_patch(lat, r_0=r_0, delta=-0.12, eps_rod=eps_rod, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000)
    fit_zero = fit_berry_curvature_patch(lat, r_0=r_0, delta=0.0, eps_rod=eps_rod, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000)

    # Evaluate PWE on the exact same dk_frac=0.08 patch and assert FIT matches PWE within 5%
    _, c_pwe_patch_pos = pwe_pos.berry_curvature(fit_pos["kx_grid"], fit_pos["ky_grid"], band_index=0)
    _, c_pwe_patch_neg = pwe_neg.berry_curvature(fit_neg["kx_grid"], fit_neg["ky_grid"], band_index=0)
    _assert_berry_patch_matches_pwe(
        float(fit_pos["c_patch"]),
        float(fit_neg["c_patch"]),
        float(fit_zero["c_patch"]),
        c_pwe_patch_pos,
        c_pwe_patch_neg,
        rtol=0.05,
    )
    assert float(np.max(np.abs(fit_zero["omega"]))) < 5e-3

    # Mutation check 1: short ring-down (n_steps=1600) in fit_berry_curvature_patch mixes
    # the two valley bands (producing C_patch ~ -0.238 instead of -0.118) and must fail the < 5% PWE check.
    fit_short = fit_berry_curvature_patch(
        lat, r_0=r_0, delta=+0.12, eps_rod=eps_rod, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=1600
    )
    with pytest.raises(AssertionError):
        _assert_berry_patch_matches_pwe(
            float(fit_short["c_patch"]),
            float(fit_neg["c_patch"]),
            float(fit_zero["c_patch"]),
            c_pwe_patch_pos,
            c_pwe_patch_neg,
            rtol=0.05,
        )

    # Mutation check 2: mutating the FHS inner product in berry_curvature_from_envelopes
    # (dropping complex conjugation in np.vdot) breaks gauge invariance and must fail the PWE check.
    with monkeypatch.context() as m:
        m.setattr(np, "vdot", lambda a, b: np.sum(np.asarray(a) * np.asarray(b)))
        _, c_mut_vdot = pwe_pos.berry_curvature(fit_pos["kx_grid"], fit_pos["ky_grid"], band_index=0)
    with pytest.raises(AssertionError):
        _assert_berry_patch_matches_pwe(
            c_mut_vdot,
            float(fit_neg["c_patch"]),
            float(fit_zero["c_patch"]),
            c_pwe_patch_pos,
            c_pwe_patch_neg,
            rtol=0.05,
        )
