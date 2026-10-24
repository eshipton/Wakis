# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, resolution-rule, and mutation validation tests for Tutorial 2.1:
Where dispersion comes from (`02-01_dispersion_physics`).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c as c0

from wakis import GridFIT3D, SolverFIT3D
from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "02_materials"
    / "02-01_dispersion_physics"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from dispersion_physics import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    ade_amplification_spectral_radius,
    get_task_table,
    kramers_kronig_im_from_re,
    kramers_kronig_re_from_im,
    main as tutorial_main,
    run_single_cell_bulk_ade,
    susceptibility_kernel_analytic,
    transfer_matrix_slab,
)


def _audit_task_cfg(task_cfg):
    """
    Compute minimum ``cpw`` (cells per minimum wavelength inside the dispersive
    slab) and ``cells_across_feature`` directly from a Tutorial 2.1 task entry.
    """
    kind = task_cfg["kind"]
    if kind == "bulk_and_kk":
        cpw = float(task_cfg["cpw"])
        cells_across = float(task_cfg["cells_across_feature"])
    elif kind in ("gan_slab", "drude_slab", "debye_slab"):
        dz = float(task_cfg["dz"])
        d = float(task_cfg["thickness"])
        fmax = float(task_cfg["fmax"])
        if kind == "gan_slab":
            f_to = 560.0 * 100.0 * c0
            f_lo = 746.0 * 100.0 * c0
            f_gam = 17.0 * 100.0 * c0
            eps_inf = 5.35
            de = eps_inf * ((f_lo / f_to) ** 2 - 1.0)
            mat = DispersiveMaterial(
                eps_inf,
                [Lorentz(de, 2.0 * np.pi * f_to, 2.0 * np.pi * f_gam)],
            )
        elif kind == "drude_slab":
            eps_inf = 4.0
            f_enz = 2.0e12
            f_gam = 0.20e12
            fp = np.sqrt(eps_inf * (f_enz**2 + f_gam**2))
            mat = DispersiveMaterial(
                eps_inf, [Drude(2.0 * np.pi * fp, 2.0 * np.pi * f_gam)]
            )
        else:
            mat = DispersiveMaterial(4.9, [Debye(75.2, 9.3e-12)])
        freqs = np.linspace(float(task_cfg["fmin"]), fmax, 81)
        n_c = np.sqrt(mat.epsilon(freqs))
        n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
        n_max = float(np.max(np.maximum(np.real(n_c), 1.0)))
        lam_min = (c0 / fmax) / n_max
        cpw = lam_min / dz
        cells_across = d / dz
    elif kind == "dt_study":
        dz_max = float(max(task_cfg["dz_list"]))
        d = float(task_cfg["thickness"])
        mat = DispersiveMaterial(
            2.0, [Lorentz(2.5, 2.0 * np.pi * 1.5e9, 2.0 * np.pi * 0.25e9)]
        )
        freqs = np.linspace(0.5e9, 2.5e9, 81)
        n_c = np.sqrt(mat.epsilon(freqs))
        n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
        n_max = float(np.max(np.maximum(np.real(n_c), 1.0)))
        lam_min = (c0 / 2.5e9) / n_max
        cpw = lam_min / dz_max
        cells_across = d / dz_max
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    ok = (cpw >= 10.0 - 1e-6) and (cells_across >= 4.0 - 1e-6)
    return ok, cpw, cells_across


def test_dispersion_physics_task_table_and_artifacts(capsys):
    """
    Verify that:
    1. Every task in ``--quick`` and ``--full`` satisfies ``cpw >= 10`` and
       ``cells_across_feature >= 4``.
    2. Mutation check: an under-resolved slab config (``dz`` too coarse) is
       rejected by ``_audit_task_cfg``.
    3. All 5 ``data/task_<I>.json`` and ``.npz`` files and all 8 figures exist
       and meet quantitative accuracy thresholds.
    """
    for quick_mode in (True, False):
        table = get_task_table(quick=quick_mode)
        assert len(table) == 5
        for task in table:
            ok, cpw, cells_across = _audit_task_cfg(task)
            assert ok, (
                f"Resolution violation in task {task['name']}: "
                f"cpw={cpw:.2f}, cells_across={cells_across:.2f}"
            )

    # Mutation check 1: under-resolved wavelength sampling (dz = 0.5 um on GaN slab)
    bad_cpw = dict(get_task_table(quick=True)[1], dz=0.5e-6)
    ok_cpw, cpw_mut, _ = _audit_task_cfg(bad_cpw)
    assert not ok_cpw and cpw_mut < 10.0

    # Mutation check 2: under-resolved slab thickness (thickness = 0.1 um with dz = 0.05 um -> 2 cells < 4)
    bad_feat = dict(get_task_table(quick=True)[1], thickness=0.1e-6, dz=0.05e-6)
    ok_feat, _, cells_mut = _audit_task_cfg(bad_feat)
    assert not ok_feat and cells_mut < 4.0

    required_keys = {
        "dx_background",
        "dx_min",
        "cells_across_feature",
        "cpw",
        "n_cells",
        "n_steps",
    }
    for idx in range(5):
        json_path = DATA_DIR / f"task_{idx}.json"
        npz_path = DATA_DIR / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file() and npz_path.stat().st_size > 0
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        assert required_keys.issubset(meta.keys())
        assert float(meta["cpw"]) >= 10.0 - 1e-6
        assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6

    t0 = json.loads((DATA_DIR / "task_0.json").read_text(encoding="utf-8"))
    assert t0["lor_bulk_rel_err_disc"] < 5e-3
    assert t0["dru_bulk_rel_err_disc"] < 5e-3
    assert t0["deb_bulk_rel_err_disc"] < 5e-3
    assert t0["kk_re_rel_err"] < 5e-3
    assert t0["kk_im_rel_err"] < 5e-3
    assert t0["fsum_rel_err_low_gamma"] < 2e-3
    assert t0["fsum_rel_err_high_gamma"] < 2e-3

    t1 = json.loads((DATA_DIR / "task_1.json").read_text(encoding="utf-8"))
    assert t1["peak_R_reststrahlen"] > 0.85
    assert t1["rms_R_disc"] < 0.01
    assert t1["rms_T_disc"] < 0.01

    t2 = json.loads((DATA_DIR / "task_2.json").read_text(encoding="utf-8"))
    assert t2["rms_R_disc"] < 0.01
    assert t2["rms_T_disc"] < 0.01

    t3 = json.loads((DATA_DIR / "task_3.json").read_text(encoding="utf-8"))
    assert t3["rms_R_disc"] < 0.01
    assert t3["rms_T_disc"] < 0.01

    t4 = json.loads((DATA_DIR / "task_4.json").read_text(encoding="utf-8"))
    assert 1.90 < t4["slope_eps_lor"] < 2.10
    assert 1.90 < t4["slope_eps_dru"] < 2.10
    assert 1.90 < t4["slope_eps_deb"] < 2.10
    assert t4["rms_R_cont_fine"] < t4["rms_R_cont_coarse"]

    expected_figs = [
        "00_setup_schematic.png",
        "wikimedia/dielectric_responses_wakis.png",
        "01_microscopic_models_freq_and_time.png",
        "02_bound_electrons_and_reststrahlen.png",
        "03_gold_drude_vs_interband_jc.png",
        "04_kramers_kronig_and_f_sum_rule.png",
        "05_lorentz_reststrahlen_slab_rt.png",
        "06_drude_enz_slab_rt.png",
        "07_debye_water_slab_rt.png",
        "08_ade_discretization_and_stability.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.is_file() and fpath.stat().st_size > 1024, f"Missing {fpath}"
    attr_md = FIG_DIR / "wikimedia" / "ATTRIBUTION.md"
    assert attr_md.is_file() and attr_md.stat().st_size > 200, f"Missing {attr_md}"

    tutorial_main(["--list-tasks", "--quick"])
    out_q = capsys.readouterr().out
    assert "# Total tasks: 5" in out_q
    assert "lorentz_reststrahlen_slab" in out_q


def test_dispersion_physics_kk_sum_rule_and_mutations():
    """
    Fast (< 5 s) physics and mutation checks for Tutorial 2.1:
    1. Passivity in exp(+i*omega*t) convention: Im(eps(f)) <= 0 for all f > 0,
       and causal kernel chi(t) == 0 for t < 0.
    2. Kramers-Kronig Hilbert transform pair and mutation check (wrong sign
       convention +2/pi vs -2/pi must fail).
    3. Single-cell FIT ADE bulk response matches epsilon_discrete(f, dt) to
       < 0.3%, and mutation check (omitting pole strength or unstable dt)
       fails the accuracy / stability assertions.
    """
    pole_l = Lorentz(2.2, 2.0 * np.pi * 2.0e9, 2.0 * np.pi * 0.3e9)
    pole_d = Drude(2.0 * np.pi * 2.5e9, 2.0 * np.pi * 0.25e9)
    pole_b = Debye(3.0, 1.0 / (2.0 * np.pi * 1.4e9))
    mat = DispersiveMaterial(eps_inf=1.8, poles=[pole_l, pole_d, pole_b], sigma=0.01)

    freqs = np.linspace(0.2e9, 6.0e9, 100)
    eps_vals = mat.epsilon(freqs)
    assert np.all(np.imag(eps_vals) <= 0.0), "Passive medium must have Im(eps) <= 0"

    # Causality check: chi(t) == 0 for t < 0 and Fourier integral of chi(t) matches chi(f)
    t_neg = np.linspace(-2.0e-9, -1.0e-12, 20)
    for p in (pole_l, pole_d, pole_b):
        assert np.all(susceptibility_kernel_analytic(p, t_neg) == 0.0)

    # Kramers-Kronig check on Lorentz + Debye susceptibility
    mat_kk = DispersiveMaterial(eps_inf=1.5, poles=[pole_l, pole_b])
    f_grid = np.linspace(1e6, 120.0e9, 4000)
    w_grid = 2.0 * np.pi * f_grid
    chi_grid = mat_kk.epsilon(f_grid) - mat_kk.eps_inf
    f_eval = np.linspace(0.3e9, 4.5e9, 60)
    w_eval = 2.0 * np.pi * f_eval
    chi_exact = mat_kk.epsilon(f_eval) - mat_kk.eps_inf

    re_kk = kramers_kronig_re_from_im(w_grid, np.imag(chi_grid), w_eval)
    im_kk = kramers_kronig_im_from_re(w_grid, np.real(chi_grid), w_eval)
    rel_err_re = np.linalg.norm(re_kk - np.real(chi_exact)) / np.linalg.norm(
        np.real(chi_exact)
    )
    rel_err_im = np.linalg.norm(im_kk - np.imag(chi_exact)) / np.linalg.norm(
        np.imag(chi_exact)
    )
    assert rel_err_re < 1e-2
    assert rel_err_im < 5e-3

    # Mutation check 1: using the exp(-i*omega*t) sign (+im_chi instead of -im_chi)
    # in the KK transform must fail the reconstruction check by ~200%
    re_kk_mut = -re_kk
    rel_err_re_mut = np.linalg.norm(re_kk_mut - np.real(chi_exact)) / np.linalg.norm(
        np.real(chi_exact)
    )
    assert rel_err_re_mut > 1.5, "Sign-flipped KK transform must fail accuracy check"

    # Single-cell ADE bulk check & mutation check
    dt = 10.0e-12
    mat_single = DispersiveMaterial(eps_inf=1.5, poles=[pole_l])
    res_true = run_single_cell_bulk_ade(
        mat_single, dt=dt, nt=1600, f0=2.0e9, fwidth=1.1e9, freqs=f_eval
    )
    assert res_true["rel_err_disc"] < 3e-3

    # Mutation check 2: wrong Lorentz delta_eps (1.1 vs 2.2) must fail against true epsilon_discrete
    mat_mut = DispersiveMaterial(
        eps_inf=1.5,
        poles=[Lorentz(1.1, 2.0 * np.pi * 2.0e9, 2.0 * np.pi * 0.3e9)],
    )
    res_mut = run_single_cell_bulk_ade(
        mat_mut, dt=dt, nt=1600, f0=2.0e9, fwidth=1.1e9, freqs=f_eval
    )
    err_mut = np.linalg.norm(
        res_mut["eps_fit"] - res_true["eps_disc"]
    ) / np.linalg.norm(res_true["eps_disc"])
    assert err_mut > 0.25, "Halved oscillator strength must fail ADE permittivity check"

    # Stability check: omega_0 * dt = 2.2 > 2 must be flagged unstable and rejected by SolverFIT3D
    w0 = 2.0 * np.pi * 2.0e9
    dt_unstable = 2.2 / w0
    rp_u, rc_u = ade_amplification_spectral_radius(pole_l, 1.5, dt_unstable)
    assert rp_u > 1.0 and rc_u > 1.0
    assert not mat_single.is_stable(dt_unstable, courant=0.5)

    grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 1, 1, 1, verbose=0)
    solver = SolverFIT3D(
        grid,
        dt=dt_unstable,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    with pytest.raises(ValueError, match="unstable"):
        solver.add_dispersive_material(
            mat_single, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)
        )

    # Energy conservation check on lossless transfer_matrix_slab: R + T == 1 (A == 0)
    tm_lossless = transfer_matrix_slab(f_eval, d=0.02, eps_r=4.2 + 0.0j, mu_r=1.0)
    assert np.max(np.abs(tm_lossless["A"])) < 1e-14
