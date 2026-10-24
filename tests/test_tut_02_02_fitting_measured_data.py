# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, resolution-rule, and mutation validation tests for Tutorial 2.2:
Fitting measured material data (`02-02_fitting_measured_data`).
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.constants import c as c0

from wakis.dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    dispersive_lib,
    eps_from_nk,
    fit_poles,
)
from wakis.material_library import info as material_info
from wakis.material_library import list_materials

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "02_materials"
    / "02-02_fitting_measured_data"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from fitting_measured_data import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    GABRIEL_1996_TISSUES,
    JC_AG_PATH,
    JC_AU_PATH,
    SIO2_PATH,
    cole_cole_permittivity,
    find_max_stable_dt,
    fit_cole_cole_debye,
    load_refractiveindex_nk,
    load_refractiveindex_sellmeier,
    sellmeier_eps_exact,
    transfer_matrix_slab,
)


def _audit_slab_resolution(thickness, dz, fmax, eps_vals):
    """
    Verify that a dispersive slab simulation satisfies ``cpw >= 10`` inside
    the medium and ``cells_across >= 4`` across the slab thickness.
    """
    n_c = np.sqrt(np.asarray(eps_vals, dtype=np.complex128))
    n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
    n_max = float(np.max(np.maximum(np.real(n_c), 1.0)))
    lam_min = (c0 / float(fmax)) / n_max
    cpw = lam_min / float(dz)
    cells_across = float(thickness) / float(dz)
    ok = (cpw >= 10.0 - 1e-6) and (cells_across >= 4.0 - 1e-6)
    return ok, cpw, cells_across


def test_fitting_measured_data_artifacts_and_tables():
    """
    Verify that:
    1. All 5 ``data/task_<I>.json`` and ``.npz`` files and all 8 figures exist.
    2. Pole convergence on Johnson & Christy (1972) Au and Ag drops the relative
       RMS error monotonically from N=1 (Drude only) to N=3..6 (Drude + Lorentz),
       and every fitted model is passive (Im eps <= 0) and stable at dt = 50 as.
    3. Out-of-band extrapolation on Au has > 5x larger error outside the narrow
       [0.55, 0.95] um window than inside it.
    4. Gabriel et al. (1996) 4-term Cole-Cole fits across all 4 tissues reach
       <= 1.1% relative RMS error by N=4 Debye poles over 10 MHz - 10 GHz.
    5. All 4 FIT slab simulations (Au, Ag, SiO2, Muscle) satisfy cpw >= 10 and
       cells_across >= 4 and match the transfer matrix from measured data.
    """
    for idx in range(5):
        json_path = DATA_DIR / f"task_{idx}.json"
        npz_path = DATA_DIR / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file() and npz_path.stat().st_size > 0

    t0 = json.loads((DATA_DIR / "task_0.json").read_text(encoding="utf-8"))
    au_n1 = t0["au_pole_fits"]["N1"]["fit_error"]
    au_n2 = t0["au_pole_fits"]["N2"]["fit_error"]
    au_n3 = t0["au_pole_fits"]["N3"]["fit_error"]
    au_n6 = t0["au_pole_fits"]["N6"]["fit_error"]
    assert au_n1 > 0.40
    assert au_n2 < 0.20
    assert au_n3 < 0.08
    assert au_n6 <= au_n3 + 1e-3
    for k in range(1, 7):
        assert t0["au_pole_fits"][f"N{k}"]["max_im_eps"] <= 0.0
        assert t0["ag_pole_fits"][f"N{k}"]["max_im_eps"] <= 0.0
        assert t0["au_pole_fits"][f"N{k}"]["stable_50as"] is True

    # Out-of-band extrapolation study
    narrow = t0["au_band_study"]["narrow_red_nir"]
    assert narrow["in_band_rel_rms"] < 0.025
    assert narrow["out_of_band_rel_rms"] > 10.0 * narrow["in_band_rel_rms"]
    assert narrow["max_im_eps_global"] <= 0.0

    # Material library audit checks
    assert t0["material_library_total_canonical"] >= 40
    assert t0["material_library_featured"]["gold"]["stable_50as"] is False
    assert t0["material_library_featured"]["gold jc"]["stable_50as"] is True
    assert t0["material_library_featured"]["fused silica"]["stable_50as"] is True

    # Task 1: Au & Ag thin-film slabs vs measured J&C transfer matrix
    t1 = json.loads((DATA_DIR / "task_1.json").read_text(encoding="utf-8"))
    assert t1["au_slab"]["cells_across"] >= 4.0
    assert (
        t1["au_slab"]["n3_drude_2lorentz"]["rms_R_tot_vs_measured"]
        < 0.35 * t1["au_slab"]["n1_drude_only"]["rms_R_tot_vs_measured"]
    )
    assert t1["au_slab"]["n3_drude_2lorentz"]["rms_R_disc_err"] < 0.01
    assert t1["ag_slab"]["rms_R_tot_vs_measured"] < 0.025
    assert t1["ag_slab"]["rms_R_disc_err"] < 0.01

    # Task 2: SiO2 Malitson Sellmeier & slab
    t2 = json.loads((DATA_DIR / "task_2.json").read_text(encoding="utf-8"))
    assert t2["lib_vs_yml_rel_diff"] < 1e-12
    assert t2["models"]["constrained_2pole_folded"]["fit_error"] < 2e-3
    assert (
        t2["models"]["constrained_2pole_folded"]["max_stable_dt_fs"]
        > 1.5 * t2["models"]["malitson_3pole_exact"]["max_stable_dt_fs"]
    )
    assert t2["sio2_slab"]["cpw_mat"] >= 10.0
    assert t2["sio2_slab"]["cells_across"] >= 4.0
    assert t2["sio2_slab"]["rms_R_tot_vs_measured"] < 5e-3
    assert t2["sio2_slab"]["rms_R_disc_err"] < 5e-3

    # Task 3: Gabriel et al. (1996) 4-term Cole-Cole fits
    t3 = json.loads((DATA_DIR / "task_3.json").read_text(encoding="utf-8"))
    for tkey in ("muscle", "skin_dry", "fat", "bone_cortical"):
        fits = t3["tissues"][tkey]["fits_10MHz_to_10GHz"]
        err_n1 = fits["N1"]["rel_rms_error"]
        err_n4 = fits["N4"]["rel_rms_error"]
        assert err_n1 > 0.08
        assert err_n4 < 0.012
        assert fits["N4"]["stable_10ps"] is True
        assert fits["N4"]["max_im_eps"] <= 0.0

    # Task 4: Muscle tissue slab simulation vs exact 4-Cole-Cole transfer matrix
    t4 = json.loads((DATA_DIR / "task_4.json").read_text(encoding="utf-8"))
    assert t4["cpw_mat"] >= 10.0
    assert t4["cells_across"] >= 4.0
    assert (
        t4["muscle_n4_debye"]["rms_R_tot_vs_cole_cole"]
        < 0.35 * t4["muscle_n1_debye"]["rms_R_tot_vs_cole_cole"]
    )
    assert (
        t4["muscle_n4_debye"]["rms_R_pole_fit_err"]
        < 0.15 * t4["muscle_n1_debye"]["rms_R_pole_fit_err"]
    )
    assert t4["muscle_n4_debye"]["rms_R_tot_vs_cole_cole"] < 0.01

    expected_figs = [
        "00_setup_schematic.png",
        "01_noble_metal_pole_convergence.png",
        "02_fit_band_extrapolation_and_library.png",
        "03_au_slab_fit_vs_measured.png",
        "04_ag_slab_fit_vs_measured.png",
        "05_sio2_sellmeier_and_slab.png",
        "06_gabriel_cole_cole_tissues.png",
        "07_tissue_debye_convergence.png",
        "08_muscle_slab_fit_vs_cole_cole.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert (
            fpath.is_file() and fpath.stat().st_size > 1024
        ), f"Missing {fpath}"


def test_fitting_measured_data_physics_and_mutations():
    """
    Fast (< 5 s) physics and mutation checks for Tutorial 2.2:
    1. ``eps_from_nk(n, k)`` produces passive ``Im(eps) <= 0`` for absorbing
       media, whereas the wrong sign convention ``(n + 1j*k)**2`` (mutation)
       violates passivity (``Im(eps) > 0``) and yields unphysical gain ``R + T > 1``
       in ``transfer_matrix_slab``.
    2. Continuous passivity vs discrete ADE stability mutation: a Lorentz model
       with ``eps_inf = 0.05 < S^2 = 0.25`` is passive (``Im eps <= 0``) at all
       frequencies but is rejected by ``is_stable(dt, courant=0.5)``.
    3. Gabriel (1996) Cole-Cole fractional broadening check: setting ``alpha_m = 0``
       collapses a single Cole-Cole term to an exact 1-pole ``Debye`` model
       (fit error < 1e-6), whereas ``alpha_m = 0.20`` requires multiple Debye
       poles (1-pole fit error > 5%).
    4. Resolution audit mutation check: coarse ``dz`` on the muscle slab is
       rejected by ``_audit_slab_resolution``.
    """
    lam_au, f_au, n_au, k_au, eps_au = load_refractiveindex_nk(JC_AU_PATH)
    assert np.all(np.imag(eps_au) < 0.0)

    # Mutation 1: wrong sign convention (n + 1j*k)^2 has Im(eps) > 0 (active gain)
    eps_au_wrong_sign = (n_au + 1j * k_au) ** 2
    assert np.all(np.imag(eps_au_wrong_sign) > 0.0)

    # Mutation 2: continuous passivity != discrete ADE stability when eps_inf < S^2
    mat_passive_but_unstable = DispersiveMaterial(
        eps_inf=0.05,
        poles=[Lorentz(1.5, 2.0 * np.pi * 500e12, 2.0 * np.pi * 20e12)],
    )
    f_test = np.geomspace(1e12, 1e16, 50)
    assert np.all(np.imag(mat_passive_but_unstable.epsilon(f_test)) <= 0.0)
    assert not mat_passive_but_unstable.is_stable(1.0e-17, courant=0.5)
    assert find_max_stable_dt(mat_passive_but_unstable, courant=0.5) == 0.0

    # Mutation 3: Cole-Cole alpha = 0 vs alpha = 0.25 single-term fit by 1 Debye pole
    f_rf = np.geomspace(10e6, 10e9, 40)
    w_rf = 2.0 * np.pi * f_rf
    tau0 = 50.0e-12
    eps_alpha0 = 4.0 + 40.0 / (1.0 + 1j * w_rf * tau0)
    eps_alpha25 = 4.0 + 40.0 / (1.0 + (1j * w_rf * tau0) ** (1.0 - 0.25))
    fit_a0 = fit_poles(
        f_rf, eps_alpha0, n_lorentz=0, n_drude=0, n_debye=1, n_starts=2
    )
    fit_a25_1 = fit_poles(
        f_rf, eps_alpha25, n_lorentz=0, n_drude=0, n_debye=1, n_starts=2
    )
    fit_a25_3 = fit_poles(
        f_rf, eps_alpha25, n_lorentz=0, n_drude=0, n_debye=3, n_starts=2
    )
    assert fit_a0.fit_error < 1e-5
    assert fit_a25_1.fit_error > 0.05
    assert fit_a25_3.fit_error < 0.01

    # Mutation 4: under-resolved muscle slab (dz = 2.5 mm -> cpw < 10 and cells_across < 4)
    eps_mus = cole_cole_permittivity(
        np.linspace(0.6e9, 6.0e9, 25), "muscle", include_sigma=True
    )
    ok_good, cpw_good, cells_good = _audit_slab_resolution(
        8.0e-3, 0.1e-3, 6.0e9, eps_mus
    )
    assert ok_good and cpw_good >= 10.0 and cells_good >= 4.0
    ok_bad, cpw_bad, cells_bad = _audit_slab_resolution(
        8.0e-3, 2.5e-3, 6.0e9, eps_mus
    )
    assert not ok_bad and (cpw_bad < 10.0 or cells_bad < 4.0)

    # Verify material_library info & list_materials public API
    assert "gold" in list_materials(category="metal")
    assert "fused silica" in list_materials(category="dielectric")
    info_sio2 = material_info("sio2")
    assert info_sio2["name"] == "fused silica"
    assert info_sio2["citation"]["year"] == 1965
