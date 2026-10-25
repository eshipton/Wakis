# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and numerical validation tests for Tutorial 6.1:
Sidewall-Corrugated Silicon Waveguide Bragg Gratings (2D EIM).
"""

from pathlib import Path
import sys

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "06_periodic_structures"
    / "06-01_bragg_grating"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    FastPolySlab,
    bragg_grating,
    build_bragg_mesh_overrides,
    build_tuned_bragg_mesh_overrides,
    build_uniform_bragg_mesh_overrides,
    cmt_bragg_spectrum,
    cmt_coupling_from_neff_step,
    compare_bragg_mesh_strategies,
    extract_stopband_metrics,
    load_graded_vs_uniform_study_summary,
    measure_grating_mesh_metrics,
)
from bragg_grating import (  # noqa: E402
    DUTY_CYCLE,
    DW_CORRUG,
    H_SI,
    N_AIR,
    N_SI,
    N_SIO2,
    PITCH,
    W_CORE,
    WL_MIN,
    WL_REF,
    build_bragg_grating,
    build_task_mesh_metrics,
    get_task_table,
)
from wakis.templates.photonics import (  # noqa: E402
    PICSimulation,
    PolySlab,
    Port,
    n_to_eps,
    slab_neff,
)


def _shoelace_area(vertices):
    """Signed shoelace area [m^2] of a 2D polygon."""
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


from functools import lru_cache


def _build_compact_grating(n_periods, pitch=PITCH, w_core=W_CORE, dw=DW_CORRUG, shift=0.0):
    """Build a compact-lead Bragg grating for fast unit tests (< 60 s total)."""
    length = float(n_periods) * float(pitch)
    pad_x = 1.1e-6
    verts = bragg_grating(
        n_periods=n_periods,
        pitch=pitch,
        w_core=w_core,
        dw=dw,
        duty_cycle=DUTY_CYCLE,
        shift=shift,
        apodization="uniform",
        y0=0.0,
        pad_x=pad_x,
    )
    slab = FastPolySlab(verts, z_min=0.0, z_max=H_SI, material=n_to_eps(N_SI))
    x_port = 0.5 * length + 0.30e-6
    domain_x = 0.5 * length + 0.60e-6
    domain_y = 0.68e-6
    ports = [
        Port("in", center=(-x_port, 0.0, 0.5 * H_SI), size=(0.0, 1.05e-6, 0.8e-6), direction=+1),
        Port("out", center=(+x_port, 0.0, 0.5 * H_SI), size=(0.0, 1.05e-6, 0.8e-6), direction=+1),
    ]
    domain = (-domain_x, +domain_x, -domain_y, +domain_y)
    return [slab], ports, domain


@lru_cache(maxsize=8)
def _run_compact_case(n_per, pitch_val=PITCH, shift=0.0):
    """Run a compact 2D EIM Bragg grating at L1 = (10, 4) with mesh_overrides."""
    cpw = 10
    n_feat = 4
    neff_slab = slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
    dx_bg = WL_MIN / (neff_slab * cpw)
    structs, ports, domain = _build_compact_grating(
        n_periods=n_per, pitch=pitch_val, w_core=W_CORE, dw=DW_CORRUG, shift=shift
    )
    overrides = build_bragg_mesh_overrides(
        n_periods=n_per,
        pitch=pitch_val,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=DUTY_CYCLE,
        shift=shift,
        n_feature=n_feat,
        dx_background=dx_bg,
    )
    pic = PICSimulation(
        wavelength_min=1.51e-6,
        wavelength_max=1.63e-6,
        n_freqs=47,
        structures=structs,
        ports=ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw,
        domain=domain,
        n_pml=6,
        cfln=0.85,
        mesh_overrides=overrides,
    )
    mesh_m = measure_grating_mesh_metrics(
        pic.grid.x,
        pic.grid.y,
        pic.grid.z,
        dw=DW_CORRUG,
        n_periods=n_per,
        pitch=pitch_val,
        w_core=W_CORE,
        duty_cycle=DUTY_CYCLE,
        shift=shift,
        n_pml=6,
    )
    s_mat = pic.run_s_parameters(
        drive=["in"],
        energy_decay=3e-2,
        max_time=0.42e-12,
        collect_fields=False,
    )
    r_pow = np.abs(s_mat[("in", "in")]) ** 2
    t_pow = np.abs(s_mat[("out", "in")]) ** 2
    metrics = extract_stopband_metrics(
        pic.wavelengths, r_pow, t_pow, n_per * pitch_val
    )
    return metrics, r_pow, t_pow, mesh_m


def test_bragg_grating_geometry_and_fast_polyslab():
    """
    `bragg_grating` constructs a counter-clockwise polygon with exact analytical
    area and bounds, and `FastPolySlab` matches `PolySlab.fill` and `PolySlab.normal`
    to machine precision.
    """
    n_per = 20
    pad_x = 1.5e-6
    length = n_per * PITCH
    verts_al = bragg_grating(
        n_periods=n_per,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=0.5,
        shift=0.0,
        pad_x=pad_x,
    )
    verts_mis = bragg_grating(
        n_periods=n_per,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=0.5,
        shift=0.5 * PITCH,
        pad_x=pad_x,
    )

    # Exact area = core strip + 2 * N * (0.5 * pitch * dw)
    expected_area = (length + 2.0 * pad_x) * W_CORE + 2.0 * n_per * (
        0.5 * PITCH * DW_CORRUG
    )
    assert _shoelace_area(verts_al) == pytest.approx(expected_area, rel=1e-12)
    assert _shoelace_area(verts_mis) == pytest.approx(expected_area, rel=1e-12)

    # FastPolySlab vs PolySlab bit-for-bit agreement on both uniform and apodized gratings
    for apod in ("uniform", "gaussian"):
        v = bragg_grating(n_periods=24, apodization=apod, pad_x=1.0e-6)
        ps_ref = PolySlab(v, 0.0, H_SI, material=12.0409)
        ps_fast = FastPolySlab(v, 0.0, H_SI, material=12.0409)

        xg = np.linspace(-5.0e-6, 5.0e-6, 120)
        yg = np.linspace(-0.45e-6, 0.45e-6, 36)
        X, Y, Z = np.meshgrid(xg, yg, [0.5 * H_SI], indexing="ij")
        hx = np.full_like(X, xg[1] - xg[0])
        hy = np.full_like(Y, yg[1] - yg[0])
        hz = np.full_like(Z, H_SI)

        f_ref = ps_ref.fill(X, Y, Z, hx, hy, hz)
        f_fast = ps_fast.fill(X, Y, Z, hx, hy, hz)
        np.testing.assert_allclose(f_fast, f_ref, atol=1e-14)

        mask = f_ref > 0
        n_ref = ps_ref.normal(X[mask], Y[mask], Z[mask])
        n_fast = ps_fast.normal(X[mask], Y[mask], Z[mask])
        np.testing.assert_allclose(n_fast, n_ref, atol=1e-14)


def test_cmt_analytical_identities_and_phase_shift():
    """
    Analytical Coupled-Mode Theory (CMT) conserves energy (R + T == 1), matches
    R_max = tanh^2(kappa * L) at lambda_B = 2 * n_eff * Lambda, and obeys the
    Wang et al. (2014) cosine cancellation law under sidewall phase shift.
    """
    wl = np.linspace(1.52e-6, 1.60e-6, 401)
    neff = 2.415
    ng = 3.45
    kappa = 7.5e4  # 750 cm^-1
    length = 100 * PITCH
    cmt = cmt_bragg_spectrum(wl, neff, PITCH, kappa, length, n_g=ng)

    assert cmt["lambda_bragg"] == pytest.approx(2.0 * neff * PITCH, rel=1e-12)
    np.testing.assert_allclose(cmt["R"] + cmt["T"], 1.0, atol=1e-13)
    assert float(np.max(cmt["R"])) == pytest.approx(
        float(np.tanh(kappa * length) ** 2), rel=1e-4
    )

    metrics = extract_stopband_metrics(wl, cmt["R"], cmt["T"], length, n_g=ng)
    assert metrics["lambda_bragg"] == pytest.approx(cmt["lambda_bragg"], abs=0.5e-9)
    assert metrics["kappa_from_tmin"] == pytest.approx(kappa, rel=2e-2)

    # Wang et al. (2014): kappa(shift) = kappa_0 |cos(pi * shift / Lambda)|
    k0 = cmt_coupling_from_neff_step(0.15, 1.56e-6, duty_cycle=0.5, shift_over_pitch=0.0)
    k_quarter = cmt_coupling_from_neff_step(
        0.15, 1.56e-6, duty_cycle=0.5, shift_over_pitch=0.25
    )
    k_half = cmt_coupling_from_neff_step(
        0.15, 1.56e-6, duty_cycle=0.5, shift_over_pitch=0.50
    )
    assert k0 > 0.0
    assert k_quarter == pytest.approx(k0 / np.sqrt(2.0), rel=1e-12)
    assert k_half == pytest.approx(0.0, abs=1e-9)


def test_task_table_resolution_and_cells_across_feature():
    """
    Every non-demo task in both `--quick` and `--full` must have `cpw >= 10`
    and `cells_across_feature >= 4.0` measured directly from the generated mesh.
    Includes mutation checks verifying that removing/coarsening the override
    fails the check and that sweeping `dw` scales the override cell size.
    """
    tasks_q = get_task_table(quick=True)
    tasks_f = get_task_table(quick=False)
    assert len(tasks_q) == 13
    assert len(tasks_f) == 15

    for mode_name, table in (("quick", tasks_q), ("full", tasks_f)):
        demo_names = [t["name"] for t in table if t.get("is_demo", False)]
        assert demo_names == ["underresolved_kottke", "underresolved_staircase"]

        for t in table:
            mesh_m = build_task_mesh_metrics(t)
            assert t["cpw"] >= 10, (
                f"[{mode_name}] Task {t['name']} has cpw={t['cpw']} < 10"
            )
            if t.get("is_demo", False):
                assert t["n_feature"] == 0
                assert mesh_m["cells_across_feature"] < 2.0, (
                    f"[{mode_name}] Demo task {t['name']} should be under-resolved "
                    f"(< 2 cells across tooth), got {mesh_m['cells_across_feature']:.2f}"
                )
            else:
                assert t["n_feature"] >= 4, (
                    f"[{mode_name}] Non-demo task {t['name']} has n_feature={t['n_feature']} < 4"
                )
                assert mesh_m["cells_across_feature"] >= 4.0 - 1e-6, (
                    f"[{mode_name}] Non-demo task {t['name']} has only "
                    f"{mesh_m['cells_across_feature']:.2f} < 4 cells across smallest feature"
                )
                assert mesh_m["cells_across_tooth_x"] >= 4.0 - 1e-6, (
                    f"[{mode_name}] Non-demo task {t['name']} has only "
                    f"{mesh_m['cells_across_tooth_x']:.2f} < 4 cells across 162 nm tooth length"
                )

    # Mutation check 1: coarsening n_feature from 4 to 2 on L1 must yield cells_across_feature < 4
    mutated_coarse = dict(tasks_q[2])
    mutated_coarse["n_feature"] = 2
    m_coarse = build_task_mesh_metrics(mutated_coarse)
    assert m_coarse["cells_across_feature"] == pytest.approx(2.0, abs=0.05)
    assert m_coarse["cells_across_feature"] < 4.0

    # Mutation check 2: depth_dw25nm vs depth_dw75nm scales dy_min_nm proportionally to dw
    m_dw25 = build_task_mesh_metrics(next(t for t in tasks_q if t["name"] == "depth_dw25nm"))
    m_dw75 = build_task_mesh_metrics(next(t for t in tasks_q if t["name"] == "depth_dw75nm"))
    assert m_dw25["min_feature_nm"] == pytest.approx(25.0)
    assert m_dw75["min_feature_nm"] == pytest.approx(75.0)
    assert m_dw25["cells_across_feature"] == pytest.approx(6.0, abs=0.02)
    assert m_dw75["cells_across_feature"] == pytest.approx(6.0, abs=0.02)
    assert m_dw75["dy_min_nm"] == pytest.approx(3.0 * m_dw25["dy_min_nm"], rel=1e-3)


def test_bragg_wavelength_and_destructive_interference_2d_eim():
    """
    In a 2D EIM FIT simulation at L1 = (10, 4) with per-axis `mesh_overrides`
    (4 cells across the 50 nm tooth), an aligned Bragg grating produces a strong
    stopband near the analytical 2 * n_eff(w_avg) * Lambda prediction, whereas
    a half-period-shifted (Lambda/2) grating cancels the fundamental Bragg
    reflection (R_max < 0.03, T_min > 0.85).
    """
    n_per = 32
    neff_slab = slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
    neff_ch = slab_neff(
        neff_slab, N_AIR, W_CORE + DW_CORRUG, 1.560e-6, polarization="TM"
    )
    lam_b_analytic = 2.0 * neff_ch * PITCH

    m_al, r_al, t_al, mesh_al = _run_compact_case(n_per, PITCH, 0.0)
    m_mis, r_mis, t_mis, mesh_mis = _run_compact_case(n_per, PITCH, 0.5 * PITCH)

    assert mesh_al["cells_across_feature"] >= 4.0 - 1e-6
    assert mesh_mis["cells_across_feature"] >= 4.0 - 1e-6

    # 1. Bragg wavelength within 8 nm of 2 * n_eff(w_avg) * Lambda at L1 = (10, 4)
    err_nm = abs(m_al["lambda_bragg"] - lam_b_analytic) * 1e9
    assert err_nm < 8.0, (
        f"Bragg wavelength {m_al['lambda_bragg'] * 1e9:.2f} nm differs from "
        f"2*n_eff*Lambda = {lam_b_analytic * 1e9:.2f} nm by {err_nm:.2f} nm >= 8 nm"
    )

    # 2. Aligned grating exhibits strong Bragg reflection; misaligned grating suppresses it
    assert m_al["R_max"] > 0.55, f"Aligned R_max = {m_al['R_max']:.4f} <= 0.55"
    assert m_al["T_min"] < 0.45, f"Aligned T_min = {m_al['T_min']:.4f} >= 0.45"
    assert m_mis["R_max"] < 0.03, (
        f"Misaligned R_max = {m_mis['R_max']:.4f} >= 0.03 (destructive interference failed)"
    )
    assert m_mis["T_min"] > 0.85, f"Misaligned T_min = {m_mis['T_min']:.4f} <= 0.85"
    assert m_mis["T_min"] > m_al["T_min"] + 0.40

    # 3. Energy balance / passivity on interior frequencies
    mid = slice(8, -8)
    assert 0.90 <= float(np.mean(r_al[mid] + t_al[mid])) <= 1.02
    assert 0.90 <= float(np.mean(r_mis[mid] + t_mis[mid])) <= 1.02


def test_mutation_period_count_and_pitch_shift():
    """
    Mutation checks at L1 = (10, 4) (cpw >= 10, cells_across_feature >= 4):
    1. Increasing period count from N = 14 to N = 32 strictly increases stopband
       reflectivity R_max and decreases minimum transmission T_min.
    2. Perturbing grating pitch Lambda by +8 nm (from 324 nm to 332 nm) shifts
       the simulated Bragg stopband wavelength upward by > 18 nm.
    """
    m_short, _, _, _ = _run_compact_case(14, PITCH, 0.0)
    m_long, _, _, _ = _run_compact_case(32, PITCH, 0.0)
    m_redshifted, _, _, _ = _run_compact_case(24, PITCH + 8.0e-9, 0.0)

    # 1. R_max increases and T_min decreases with N
    assert m_long["R_max"] > m_short["R_max"] + 0.15, (
        f"Expected R_max(N=32)={m_long['R_max']:.3f} > R_max(N=14)={m_short['R_max']:.3f} + 0.15"
    )
    assert m_long["T_min"] < m_short["T_min"] - 0.15, (
        f"Expected T_min(N=32)={m_long['T_min']:.3f} < T_min(N=14)={m_short['T_min']:.3f} - 0.15"
    )

    # 2. Pitch mutation (+8 nm) shifts Bragg stopband upward by > 18 nm
    delta_lam_nm = (m_redshifted["lambda_bragg"] - m_long["lambda_bragg"]) * 1e9
    assert delta_lam_nm > 18.0, (
        f"Expected pitch mutation (+8 nm) to shift lambda_B by > +18 nm, got {delta_lam_nm:+.2f} nm"
    )


def test_bragg_grating_full_study_data_convergence_cmt_and_apodization():
    """
    Verify the 15-task ``--full`` study dataset in ``tutorials/06_periodic_structures/06-01_bragg_grating/data/``:
    1. Every non-demo task has ``cpw >= 10`` and ``cells_across_feature >= 4.0`` in both
       the JSON metadata and NPZ payload.
    2. Bragg wavelength, stopband FWHM bandwidth, and transmission spectrum RMS error
       converge monotonically across L1 -> L2 -> L3 against the R = (40, 16) reference,
       with Kottke strictly outperforming staircase at every level and relative change
       < 0.2% in ``lambda_B`` and < 2.0% in ``fwhm_bandwidth`` between L2/L3 and L3/R.
    3. Coupled-Mode Theory (CMT) coupling coefficients ``kappa_from_tmin`` and
       ``kappa_from_bw`` agree within 3% at L3 and R, CMT stopband saturation
       ``tanh(kappa * N * Lambda)**2`` matches the period sweep ``N in {25, 50, 100}``
       with < 1.5% spread in ``kappa``, and ``misaligned_half_period`` cancels Bragg
       reflection (``R_max < 1e-3``, ``T_min > 0.99``).
    4. Gaussian apodization suppresses reflection sidelobes by > 10 dB relative to the
       uniform grating.
    5. Mutation checks verify that perturbing the resolution order or swapping uniform
       and apodized spectra causes the convergence and sidelobe assertions to fail.
    """
    import json

    data_dir = TUTORIAL_DIR / "data"
    expected_tasks = get_task_table(quick=False)
    assert len(expected_tasks) == 15

    tasks_npz = {}
    for exp in expected_tasks:
        idx = exp["index"]
        json_path = data_dir / f"task_{idx}.json"
        npz_path = data_dir / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file(), f"Missing {npz_path}"

        meta = json.loads(json_path.read_text())
        d = dict(np.load(npz_path, allow_pickle=False))
        assert meta["name"] == exp["name"] == str(d["name"])
        assert float(meta["cpw"]) == pytest.approx(float(exp["cpw"]))
        assert float(meta["cpw"]) >= 10.0
        assert float(d["cpw"]) >= 10.0

        if exp.get("is_demo", False):
            assert float(meta["cells_across_feature"]) < 2.0
        else:
            assert float(meta["cells_across_feature"]) >= 4.0
            assert float(d["cells_across_feature"]) >= 4.0
            assert float(meta["cells_across_tooth_x"]) >= 4.0
            assert float(meta["cells_across_feature"]) == pytest.approx(
                float(exp["n_feature"]), rel=1e-2
            )
        tasks_npz[exp["name"]] = d

    def _interior_spectrum(d):
        w = d["wavelengths"] * 1e9
        order = np.argsort(w)
        w = w[order]
        m = (w >= 1505.0) & (w <= 1615.0)
        r = np.clip(np.abs(d["S11"][order][m]) ** 2, 0.0, 1.0)
        t = np.clip(np.abs(d["S21"][order][m]) ** 2, 0.0, 1.0)
        return w[m], r, t

    L_100 = 100.0 * PITCH
    prim = tasks_npz["kottke_L3"]
    assert "dft_E2" in prim and "dft_Ey" in prim and "flux_total_box" in prim
    ng_est = float(np.mean(prim["disp_ng"]))
    assert 3.20 <= ng_est <= 3.30

    # 2D Poynting flux box conservation on primary L3 and misaligned L3 runs
    w_all = prim["wavelengths"] * 1e9
    ord_idx = np.argsort(w_all)
    m_band = (w_all[ord_idx] >= 1505.0) & (w_all[ord_idx] <= 1615.0)
    assert float(np.mean(prim["flux_total_box"][ord_idx][m_band])) == pytest.approx(
        1.0, abs=5e-3
    )
    assert float(
        np.mean(tasks_npz["misaligned_half_period"]["flux_total_box"][ord_idx][m_band])
    ) == pytest.approx(1.0, abs=5e-3)

    # Convergence across L1 -> L2 -> L3 against kottke_reference R = (40, 16)
    w_ref, r_ref, t_ref = _interior_spectrum(tasks_npz["kottke_reference"])
    m_ref = extract_stopband_metrics(w_ref * 1e-9, r_ref, t_ref, L_100, n_g=ng_est)
    lam_ref_nm = m_ref["lambda_bragg"] * 1e9
    bw_ref_nm = m_ref["fwhm_bandwidth"] * 1e9

    err_lam_k, err_lam_s = [], []
    err_bw_k, err_bw_s = [], []
    err_t_k, err_t_s = [], []
    lam_k_list, bw_k_list = [], []

    for lvl in ("L1", "L2", "L3"):
        wk, rk, tk = _interior_spectrum(tasks_npz[f"kottke_{lvl}"])
        ws, rs, ts = _interior_spectrum(tasks_npz[f"staircase_{lvl}"])
        mk = extract_stopband_metrics(wk * 1e-9, rk, tk, L_100, n_g=ng_est)
        ms = extract_stopband_metrics(ws * 1e-9, rs, ts, L_100, n_g=ng_est)
        lam_k_list.append(mk["lambda_bragg"] * 1e9)
        bw_k_list.append(mk["fwhm_bandwidth"] * 1e9)
        err_lam_k.append(abs(mk["lambda_bragg"] * 1e9 - lam_ref_nm))
        err_lam_s.append(abs(ms["lambda_bragg"] * 1e9 - lam_ref_nm))
        err_bw_k.append(abs(mk["fwhm_bandwidth"] * 1e9 - bw_ref_nm))
        err_bw_s.append(abs(ms["fwhm_bandwidth"] * 1e9 - bw_ref_nm))
        err_t_k.append(float(np.sqrt(np.mean((tk - t_ref) ** 2))))
        err_t_s.append(float(np.sqrt(np.mean((ts - t_ref) ** 2))))

    assert err_lam_k[2] < err_lam_k[1] < err_lam_k[0]
    assert err_bw_k[2] < err_bw_k[1] < err_bw_k[0]
    assert err_t_k[2] < err_t_k[1] < err_t_k[0]
    for idx_l in range(3):
        assert err_lam_k[idx_l] < err_lam_s[idx_l]
        assert err_bw_k[idx_l] < err_bw_s[idx_l]
        assert err_t_k[idx_l] < err_t_s[idx_l]

    # Relative change between L2/L3 and L3/R
    assert abs(lam_k_list[2] - lam_k_list[1]) / lam_k_list[2] < 2e-3
    assert err_lam_k[2] / lam_ref_nm < 2e-3
    assert abs(bw_k_list[2] - bw_k_list[1]) / bw_k_list[2] < 2e-2
    assert err_bw_k[2] / bw_ref_nm < 2e-2

    # Coupled-Mode Theory (CMT) kappa and bandwidth consistency at L3 and R
    w_l3, r_l3, t_l3 = _interior_spectrum(prim)
    m_l3 = extract_stopband_metrics(w_l3 * 1e-9, r_l3, t_l3, L_100, n_g=ng_est)
    for m_cur in (m_l3, m_ref):
        assert m_cur["kappa_from_tmin"] == pytest.approx(
            m_cur["kappa_from_bw"], rel=0.03
        )
        cmt_c = cmt_bragg_spectrum(
            w_ref * 1e-9,
            n_eff=m_cur["lambda_bragg"] / (2.0 * PITCH),
            pitch=PITCH,
            kappa=m_cur["kappa_from_tmin"],
            length=L_100,
            n_g=ng_est,
            wavelength_bragg=m_cur["lambda_bragg"],
        )
        m_cmt = extract_stopband_metrics(
            w_ref * 1e-9, cmt_c["R"], cmt_c["T"], L_100, n_g=ng_est
        )
        assert (m_cur["fwhm_bandwidth"] * 1e9) == pytest.approx(
            m_cmt["fwhm_bandwidth"] * 1e9, abs=1.0
        )
        assert (m_cur["null_bandwidth"] * 1e9) == pytest.approx(
            cmt_c["null_bandwidth"] * 1e9, abs=0.6
        )

    # Period sweep N in {25, 50, 100}: CMT tanh^2(kappa * N * Lambda) and invariant kappa
    kappas_n = []
    for n_per, t_name in ((25, "periods_N25"), (50, "periods_N50"), (100, "kottke_L3")):
        wn, rn, tn = _interior_spectrum(tasks_npz[t_name])
        mn = extract_stopband_metrics(wn * 1e-9, rn, tn, n_per * PITCH, n_g=ng_est)
        kappas_n.append(mn["kappa_from_tmin"])
        r_cmt_n = float(np.tanh(m_l3["kappa_from_tmin"] * n_per * PITCH) ** 2)
        assert (1.0 - mn["T_min"]) == pytest.approx(r_cmt_n, abs=0.01)
    assert max(kappas_n) / min(kappas_n) - 1.0 < 0.015

    # Corrugation depth sweep dw in {25, 50, 75} nm: monotonic lambda_B, FWHM, and kappa
    w25, r25, t25 = _interior_spectrum(tasks_npz["depth_dw25nm"])
    w75, r75, t75 = _interior_spectrum(tasks_npz["depth_dw75nm"])
    m25 = extract_stopband_metrics(w25 * 1e-9, r25, t25, L_100, n_g=ng_est)
    m75 = extract_stopband_metrics(w75 * 1e-9, r75, t75, L_100, n_g=ng_est)
    assert m25["lambda_bragg"] < m_l3["lambda_bragg"] < m75["lambda_bragg"]
    assert m25["fwhm_bandwidth"] < m_l3["fwhm_bandwidth"] < m75["fwhm_bandwidth"]
    assert m25["kappa_from_tmin"] < m_l3["kappa_from_tmin"] < m75["kappa_from_tmin"]

    # Misaligned (Lambda/2) destructive interference and Gaussian apodization sidelobe suppression
    w_mis, r_mis, t_mis = _interior_spectrum(tasks_npz["misaligned_half_period"])
    assert float(np.max(r_mis)) < 1e-3
    assert float(np.min(t_mis)) > 0.99

    w_ap, r_ap, t_ap = _interior_spectrum(tasks_npz["apodized_gaussian"])
    m_ap = extract_stopband_metrics(w_ap * 1e-9, r_ap, t_ap, L_100, n_g=ng_est)
    sidelobe_suppression_db = m_l3["sidelobe_db"] - m_ap["sidelobe_db"]
    assert sidelobe_suppression_db > 10.0, (
        f"Expected Gaussian apodization to suppress sidelobes by > 10 dB, got {sidelobe_suppression_db:.2f} dB"
    )

    # Mutation check: reversing L1 and L3 breaks monotonic convergence, and swapping
    # uniform and apodized spectra breaks sidelobe suppression
    mutated_err_lam = [err_lam_k[2], err_lam_k[1], err_lam_k[0]]
    assert not (mutated_err_lam[2] < mutated_err_lam[1] < mutated_err_lam[0])
    assert not ((m_ap["sidelobe_db"] - m_l3["sidelobe_db"]) > 10.0)


def test_subgrid_vs_graded_bragg_full_analysis_and_verdict():
    """
    Verify the `--full` analysis of `studies/subgrid-vs-graded-bragg/` against
    `data/analysis_full.json` (Richardson O(h^2) limits, neutral reference,
    log-log interpolated work_eff to 1.0 nm and 0.5 nm, uniform_N4 comparison,
    LTS3 vs no-LTS shift at equal mesh, and non-monotone cpw in subgrid), and
    verify that mutating the reference wavelength by +1.0 nm fails the verdict check.
    """
    import importlib.util
    import json

    study_dir = (
        Path(__file__).resolve().parents[1] / "studies" / "subgrid-vs-graded-bragg"
    )
    study_path = study_dir / "bragg_subgrid_study.py"
    analysis_json_path = study_dir / "data" / "analysis_full.json"
    assert study_path.exists(), f"Missing study script: {study_path}"
    assert analysis_json_path.exists(), f"Missing analysis JSON: {analysis_json_path}"

    spec = importlib.util.spec_from_file_location("bragg_subgrid_study", study_path)
    study_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study_mod)

    saved = json.loads(analysis_json_path.read_text(encoding="utf-8"))
    recomputed = study_mod.compute_summary_and_errors(
        data_dir=study_dir / "data", write_json=False
    )

    for summary in (saved, recomputed):
        study_mod.assert_full_verdict(summary)

        lims = summary["richardson_limits_nm"]
        assert lims["graded_tuned"] == pytest.approx(1551.311, abs=1e-3)
        assert lims["graded_t18"] == pytest.approx(1551.418, abs=1e-3)
        assert lims["subgrid_lts3"] == pytest.approx(1551.197, abs=1e-3)

        ref = summary["reference"]
        assert ref["lambda_B_ref_nm"] == pytest.approx(1551.308, abs=1e-3)
        assert ref["sigma_ref_nm"] == pytest.approx(0.110, abs=1e-3)
        assert ref["BW_ref_nm"] == pytest.approx(30.409, abs=1e-2)

        w1 = summary["work_eff_to_1p0nm"]
        assert w1["graded_tuned"] == pytest.approx(9.31e10, rel=5e-3)
        assert w1["graded_t18"] == pytest.approx(3.83e11, rel=5e-3)
        assert w1["subgrid"] == pytest.approx(6.87e10, rel=5e-3)
        assert w1["subgrid_lts3"] == pytest.approx(2.58e10, rel=5e-3)
        assert w1["graded_tuned_staircase"] == "not reached"

        w05 = summary["work_eff_to_0p5nm"]
        assert w05["graded_tuned"] == pytest.approx(2.49e11, rel=5e-3)
        assert w05["graded_t18"] == "not reached"
        assert w05["subgrid"] == pytest.approx(1.02e11, rel=5e-3)
        assert w05["subgrid_lts3"] == pytest.approx(3.86e10, rel=5e-3)
        assert w05["graded_tuned_staircase"] == "not reached"

        u4 = summary["uniform_N4"]
        assert u4["dy_tooth_nm"] == pytest.approx(12.3077, abs=1e-3)
        assert u4["err_lambda_B_nm"] == pytest.approx(0.1145, abs=5e-3)
        assert u4["work_eff"] == pytest.approx(7.86e10, rel=2e-3)

        # LTS3 vs no-LTS shift at equal mesh (tasks 12 vs 15) is ~0.11 nm ~ sigma_ref
        lts_pairs = {
            (p["task_no_lts"], p["task_lts3"]): p
            for p in summary["lts3_vs_no_lts_equal_mesh"]
        }
        assert (12, 15) in lts_pairs
        assert abs(lts_pairs[(12, 15)]["delta_lambda_B_nm"]) == pytest.approx(
            0.1051, abs=5e-3
        )
        assert lts_pairs[(12, 15)]["abs_delta_over_sigma_ref"] == pytest.approx(
            0.953, abs=5e-2
        )

        # Non-monotone cpw in subgrid arm (task 11 cpw=15.6 -> task 12 cpw=10.6)
        sg_cpw = summary["subgrid_non_monotone_cpw"]
        assert sg_cpw["task_11"]["cpw"] == pytest.approx(15.6)
        assert sg_cpw["task_12"]["cpw"] == pytest.approx(10.6)
        assert sg_cpw["task_12"]["rel_err_bw"] > sg_cpw["task_11"]["rel_err_bw"]

    # Mutation check: perturbing the reference wavelength by +1.0 nm must fail assert_full_verdict
    ref_lam = saved["reference"]["lambda_B_ref_nm"]
    mutated = study_mod.compute_summary_and_errors(
        data_dir=study_dir / "data",
        write_json=False,
        ref_override_nm=ref_lam + 1.0,
    )
    with pytest.raises(AssertionError):
        study_mod.assert_full_verdict(mutated)


def test_setup_schematic_and_graded_mesh_comparison_with_mutation():
    """
    Verify that:
    1. ``00_setup_schematic.png`` and all 16 restyled figures exist in
       ``tutorials/06_periodic_structures/06-01_bragg_grating/figures/``, with
       ``09_pulse_propagation.gif`` under 5 MB.
    2. ``compare_bragg_mesh_strategies`` confirms that at equal tooth resolution
       ``n_feature = 4``, both ``sidewall_graded`` and ``tuned_graded`` resolve the
       50 nm tooth with >= 4 cells while reducing total 2D cell count vs
       ``uniform_fine`` (``cell_reduction_vs_uniform_fine > 1.5``).
    3. ``load_graded_vs_uniform_study_summary`` loads the benchmark comparison
       between ``graded_t18``, ``graded_tuned``, and ``uniform_N4``.
    4. Mutation check: replacing ``build_tuned_bragg_mesh_overrides`` with an
       empty override list drops ``cells_across_feature`` below 4.0 and fails
       the tooth-resolution requirement.
    """
    fig_dir = TUTORIAL_DIR / "figures"
    expected_figures = [
        "00_setup_schematic.png",
        "01_bragg_grating_layout.png",
        "02_eim_vertical_stack.png",
        "03_corrugation_and_apodization_profiles.png",
        "04_permittivity_mesh_zoom.png",
        "05_port_mode_profiles.png",
        "06_port_neff_dispersion.png",
        "07_field_intensity_bragg.png",
        "08_field_real_ey_bragg.png",
        "09_pulse_propagation.gif",
        "10_aligned_vs_misaligned_spectra.png",
        "11_periods_sweep_and_cmt.png",
        "12_corrugation_depth_sweep.png",
        "13_apodized_vs_uniform_sidelobes.png",
        "14_kottke_vs_staircase_convergence.png",
        "15_port_time_trace_and_flux_budget.png",
    ]
    for fname in expected_figures:
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing expected figure {fpath}"
        assert fpath.stat().st_size > 10_000, f"Figure {fpath} is unexpectedly small"
    assert (fig_dir / "09_pulse_propagation.gif").stat().st_size < 5 * 1024 * 1024

    cmp = compare_bragg_mesh_strategies(
        n_periods=20,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        cpw=10,
        n_feature=4,
    )
    assert cmp["uniform_coarse"]["cells_across_feature"] < 2.0
    assert cmp["sidewall_graded"]["cells_across_feature"] >= 4.0 - 1e-6
    assert cmp["tuned_graded"]["cells_across_feature"] >= 4.0 - 1e-6
    assert cmp["uniform_fine"]["cells_across_feature"] >= 4.0 - 1e-6
    assert cmp["sidewall_graded"]["cell_reduction_vs_uniform_fine"] > 4.0
    assert cmp["tuned_graded"]["cell_reduction_vs_uniform_fine"] > 1.8
    assert cmp["tuned_graded"]["dy_core_nm"] == pytest.approx(
        2.0 * cmp["tuned_graded"]["dy_min_nm"], rel=5e-2
    )

    study_sum = load_graded_vs_uniform_study_summary()
    assert len(study_sum["graded_t18"]) == 4
    assert len(study_sum["graded_tuned"]) == 3
    assert study_sum["uniform_N4"]["n_cells"] == 379353
    assert study_sum["graded_tuned"][0]["err_lambda_B_nm"] < study_sum["graded_t18"][0]["err_lambda_B_nm"]

    # Mutation check: omitting mesh_overrides drops tooth resolution below 4 cells/tooth
    neff_slab = slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
    dx_bg = WL_MIN / (neff_slab * 10.0)
    ov_tuned = build_tuned_bragg_mesh_overrides(
        n_periods=12, pitch=PITCH, w_core=W_CORE, dw=DW_CORRUG, n_feature=4, dx_background=dx_bg
    )
    ov_unif = build_uniform_bragg_mesh_overrides(
        dw=DW_CORRUG, n_feature=4, dx_background=dx_bg
    )
    assert len(ov_tuned) == 4
    assert len(ov_unif) == 1
    structs, ports, domain = _build_compact_grating(12)
    pic_mut = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=1.63e-6,
        structures=structs,
        ports=ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=domain,
        n_pml=6,
        mesh_overrides=[],
    )
    m_mut = measure_grating_mesh_metrics(
        pic_mut.grid.x, pic_mut.grid.y, pic_mut.grid.z, dw=DW_CORRUG, n_periods=12, pitch=PITCH, w_core=W_CORE, n_pml=6
    )
    assert m_mut["cells_across_feature"] < 2.0


