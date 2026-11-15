"""Measured values behind the wave-2 verification table (prints JSON).

Uses the helpers of tests/test_104_subgrid3d_lts.py; run from the repo root.
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_104_subgrid3d_lts as T

from studies.subgrid3d.plot_figures_lts import analyse_cost, analyse_gpu, load
from studies.subgrid3d.run_study_lts import matrix_pencil
from wakis.subgrid3d import SubgridLTS, SubgridLTSFIT3D
from wakis.subgrid3d.lts import lts_dt_max

out = {}
failures = []


def check(name, value, ok, expected=""):
    out[name] = value
    if not ok:
        msg = f"FAIL {name}: measured={value}" + (
            f" (expected {expected})" if expected else ""
        )
        print(msg, flush=True)
        failures.append(msg)


s = SubgridLTSFIT3D(T.closed_mesh(), lts=False, **T.closed_materials())
L = SubgridLTS(s, 2.5 * s.dt_max)
e = np.random.default_rng(0).standard_normal(s.n_free)
full = L.apply_full(e)
check(
    "fast_vs_full",
    float(np.abs(L.apply(e) - full).max() / np.abs(full).max()),
    np.abs(L.apply(e) - full).max() / np.abs(full).max() < 1e-13,
    "< 1e-13",
)
check(
    "matrix_vs_full",
    float(np.abs(L.matrix() @ e - full).max() / np.abs(full).max()),
    np.abs(L.matrix() @ e - full).max() / np.abs(full).max() < 1e-13,
    "< 1e-13",
)
L.levels[1].kappa = 1.1 * L.levels[1].kappa
check(
    "mut_kappa",
    float(np.abs(L.apply(e) - full).max() / np.abs(full).max()),
    np.abs(L.apply(e) - full).max() / np.abs(full).max() > 1e-6,
    "> 1e-6",
)

for nested in (False, True):
    s0 = SubgridLTSFIT3D(T.cavity_mesh(16, patch=False), lts=False)
    s1 = SubgridLTSFIT3D(T.cavity_mesh(16, nested=nested), lts=True)
    k = "nested" if nested else "single"
    r_coarse = float(s1.lts_info["dt_max"] / s0.dt_max)
    r_global = float(s1.lts_info["dt_max"] / s1.dt_max)
    fl = float(T.damping_floor(s1.lts))
    pred = float(T.global_limit_ratio(s1.lts))
    check(
        f"dt_{k}_over_coarse",
        r_coarse,
        fl * (1 - 2e-3) <= r_coarse <= 1.05,
        f"in [{fl * (1 - 2e-3):.4f}, 1.05]",
    )
    check(
        f"dt_{k}_over_global",
        r_global,
        abs(r_global / pred - 1) < 2e-3,
        f"within 2e-3 of {pred:.4f}",
    )
    out[f"floor_{k}"] = fl
    if not nested:
        dm, _ = lts_dt_max(s1, 2.2 * s1.dt_max, interface_parents=False)
        check(
            "mut_parents_over_coarse",
            float(dm / s0.dt_max),
            dm / s0.dt_max < fl * (1 - 2e-3),
            f"< {fl * (1 - 2e-3):.4f}",
        )

s = SubgridLTSFIT3D(T.closed_mesh(), lts=True, cfl=0.99, **T.closed_materials())
s.e[:] = np.random.default_rng(1).standard_normal(s.n_free)
E = s.run(4000, energy=True)
drift_4000 = float(np.abs(E / E[0] - 1).max())
tol_4000 = float(50 * T.EPS * np.sqrt(len(E)))
check("energy_drift_4000", drift_4000, drift_4000 < tol_4000, f"< {tol_4000:.2e}")
out["energy_tol"] = tol_4000
check(
    "closed_dt_over_global",
    float(s.lts_info["dt_max"] / s.dt_max),
    abs(s.lts_info["dt_max"] / s.dt_max - 5.82) < 0.05,
    "~5.82",
)
mut_103_err = None
try:
    SubgridLTSFIT3D(
        T.closed_mesh(),
        lts=True,
        dt=1.03 * s.lts_info["dt_max"],
        **T.closed_materials(),
    )
except ValueError as exc:
    mut_103_err = str(exc)
check("mut_103_rejected", mut_103_err, mut_103_err is not None, "ValueError at setup")
m = SubgridLTSFIT3D(
    T.closed_mesh(),
    lts=False,
    dt=0.5 * s.dt_max,
    mutate="inject",
    **T.closed_materials(),
)
m.lts = SubgridLTS(m, m.dt)
m.e[:] = np.random.default_rng(1).standard_normal(m.n_free)
Em = m.run(500, energy=True)
check(
    "mut_inject_drift",
    float(np.abs(Em / Em[0] - 1).max()),
    np.abs(Em / Em[0] - 1).max() > 1e-2,
    "> 1e-2",
)

for name, val, thr, op in (
    (
        "dot_closed",
        T._dot(
            SubgridLTSFIT3D(
                T.closed_mesh(), lts=True, dt=1.0e-10, **T.closed_materials()
            )
        ),
        1e-15,
        "<",
    ),
    ("dot_cpml_csr", T._dot(T._open_box(dt=7.0e-11)), 1e-15, "<"),
    ("dot_cpml_block", T._dot(T._open_box(dt=7.0e-11, kernels="block")), 1e-15, "<"),
    ("mut_dot_R", T._dot(T._open_box(dt=7.0e-11), mutate="R"), 1e-10, ">"),
    ("mut_dot_pml", T._dot(T._open_box(dt=7.0e-11), mutate="pml"), 1e-10, ">"),
):
    check(name, float(val), (val < thr if op == "<" else val > thr), f"{op} {thr:.1e}")

e8, e16 = float(T._lts_error(8)), float(T._lts_error(16))
m8, m16 = float(T._lts_error(8, "consistency")), float(T._lts_error(16, "consistency"))
cav_ord = float(np.log2(e8 / e16))
mut_ord = float(np.log2(m8 / m16))
out.update(cav_e8=e8, cav_e16=e16, mut_e8=m8, mut_e16=m16)
check("cav_order", cav_ord, 1.8 <= cav_ord <= 2.2, "in [1.8, 2.2]")
check(
    "mut_order",
    mut_ord,
    mut_ord < 1.0 and m16 > 5 * e16,
    "< 1.0 and mut_e16 > 5 * cav_e16",
)
for lts in (False, True):
    r_ok = float(T._residual(T._open_box(lts=lts)))
    r_mut = float(T._residual(T._open_box(lts=lts, R0=1.0)))
    check(f"cpml_residual_lts{lts}", r_ok, r_ok < 1e-6, "< 1e-6")
    check(f"mut_cpml_residual_lts{lts}", r_mut, r_mut > 1e-3, "> 1e-3")

kw = T.closed_materials()
sc = SubgridLTSFIT3D(T.closed_mesh(), lts=True, cfl=0.95, kernels="csr", **kw)
sb = SubgridLTSFIT3D(T.closed_mesh(), lts=True, dt=sc.dt, kernels="block", **kw)
e = np.random.default_rng(5).standard_normal(sc.n_free)
for x in (sc, sb):
    x.e[:] = e
    x.run(500)
rel_500 = float(T._rel(sb.e, sc.e))
bnd_500 = float(T._roundoff_bound(sc, 500))
check("block_vs_csr_500", rel_500, rel_500 < bnd_500, f"< {bnd_500:.2e}")
out["roundoff_bound_500"] = bnd_500

# Matrix pencil accuracy and mutation check
t_mp = np.arange(400) * 2.5e-10
f_true = np.array([2.0e8, 2.35e8])
sig_mp = np.sin(2 * np.pi * f_true[0] * t_mp) + 0.7 * np.cos(
    2 * np.pi * f_true[1] * t_mp
)
f_est = matrix_pencil(sig_mp, 2.5e-10, n_modes=2)
f_mut = matrix_pencil(
    np.sin(2 * np.pi * (1.05 * f_true[0]) * t_mp)
    + 0.7 * np.cos(2 * np.pi * f_true[1] * t_mp),
    2.5e-10,
    n_modes=2,
)
check(
    "matrix_pencil_rel_err",
    float(np.abs(f_est / f_true - 1).max()),
    np.abs(f_est / f_true - 1).max() < 1e-12,
    "< 1e-12",
)
check(
    "mut_matrix_pencil_rel_err",
    float(np.abs(f_mut / f_true - 1).max()),
    np.abs(f_mut / f_true - 1).max() > 0.04,
    "> 0.04",
)

# Guide numbers from results_D.json, results_E.json, results_F.json, and results_E_quick.json
NEW_COLS = (
    "err_semi",
    "err_td",
    "work_gain",
    "work_gain_semi",
    "wall_gain",
    "wall_gain_semi",
    "wall_gain_uniform",
    "work_gain_global",
    "work_gain_global_semi",
    "wall_gain_global",
    "wall_gain_global_semi",
    "wall_gain_vs_global",
    "probe_agreement_ratio",
)
res_d = load("D")
assert res_d is not None, "results_D.json missing"
rows_d, cost_d = analyse_cost(res_d)
for arm in cost_d["arms"]:
    for col in NEW_COLS:
        if col not in arm or arm[col] is None or not np.isfinite(arm[col]):
            check(
                f"D_{arm['kind']}_{arm['ratio']}to1_H{arm['h_coarse']:.4f}_{col}",
                arm.get(col),
                False,
                "finite",
            )
check(
    "cost_D_f_star_MHz",
    round(cost_d["f_star"] / 1e6, 3),
    abs(cost_d["f_star"] / 1e6 - 190.717) < 0.001,
    "190.717 MHz",
)
check(
    "cost_D_observed_order",
    round(cost_d["observed_order"], 2),
    abs(cost_d["observed_order"] - 1.46) < 0.01,
    "1.46",
)
check(
    "cost_D_max_probe_agreement_ratio",
    float(cost_d["max_probe_agreement_ratio"]),
    cost_d["max_probe_agreement_ratio"] <= 0.2,
    "<= 0.2",
)
for kind, ratio, exp_wg_td, exp_wall_td, exp_wg_semi, exp_wall_semi in (
    ("global", 2, (1.78, 2.21), (1.70, 2.55), (1.82, 2.22), (1.73, 2.57)),
    ("lts", 2, (2.04, 2.58), (1.17, 2.13), (1.49, 2.23), (0.90, 1.77)),
    ("global", 3, (1.36, 1.93), (1.22, 2.23), (1.42, 1.95), (1.26, 2.27)),
    ("lts", 3, (1.59, 2.18), (1.01, 1.93), (0.95, 1.59), (0.61, 1.30)),
):
    arms = [a for a in cost_d["arms"] if a["kind"] == kind and a["ratio"] == ratio]
    for col, exp_rng in (
        ("work_gain_td", exp_wg_td),
        ("wall_gain_td", exp_wall_td),
        ("work_gain_semi", exp_wg_semi),
        ("wall_gain_semi", exp_wall_semi),
    ):
        rng = (round(min(a[col] for a in arms), 2), round(max(a[col] for a in arms), 2))
        check(
            f"cost_D_{kind}_{ratio}to1_{col}_range",
            f"{rng[0]:.2f}..{rng[1]:.2f}",
            rng == exp_rng,
            f"{exp_rng[0]:.2f}..{exp_rng[1]:.2f}",
        )
out["cost_D_lts_2to1_wall_gain_semi"] = float(
    next(
        a["wall_gain_semi"]
        for a in cost_d["arms"]
        if a["kind"] == "lts" and a["ratio"] == 2
    )
)
out["cost_D_lts_2to1_wall_gain_global_semi"] = float(
    next(
        a["wall_gain_global_semi"]
        for a in cost_d["arms"]
        if a["kind"] == "lts" and a["ratio"] == 2
    )
)

# GPU numbers in results_D.json (part G)
gvc = res_d["gpu_vs_cpu"]
check(
    "gpu_vs_cpu_rel_err",
    float(gvc["rel_err"]),
    gvc["rel_err"] < gvc["bound"] and abs(gvc["rel_err"] - 1.73e-14) < 0.05e-14,
    f"< {gvc['bound']:.2e} (~1.73e-14)",
)
check(
    "gpu_vs_cpu_bound",
    float(gvc["bound"]),
    abs(gvc["bound"] - 6.68e-12) < 0.05e-12,
    "~6.68e-12",
)
gpu_rows = analyse_gpu(res_d, rows_d)
uni_gpu = [g for g in gpu_rows if g["kind"] == "uniform"]
t_sf_rng = (
    round(min(g["t_solverfit3d"] for g in uni_gpu), 2),
    round(max(g["t_solverfit3d"] for g in uni_gpu), 1),
)
t_sff_rng = (
    round(min(g["t_solverfit3d_fused"] for g in uni_gpu), 3),
    round(max(g["t_solverfit3d_fused"] for g in uni_gpu), 1),
)
check(
    "gpu_t_solverfit3d_s_range",
    f"{t_sf_rng[0]}..{t_sf_rng[1]}",
    t_sf_rng == (1.56, 23.7),
    "1.56..23.7",
)
check(
    "gpu_t_solverfit3d_fused_s_range",
    f"{t_sff_rng[0]}..{t_sff_rng[1]}",
    t_sff_rng == (0.253, 29.0),
    "0.253..29.0",
)

# CPML numbers in results_F.json
res_f = load("F")
assert res_f is not None, "results_F.json missing"
for v, exp_hi, exp_lo in (
    ("nopatch", 2.1e-2, 1.6e-5),
    ("patch_global", 1.1e-2, 1.6e-5),
    ("patch_lts", 2.1e-2, 1.6e-5),
    ("through", 1.2e-1, 1.7e-2),
):
    errs = [
        r["err"]
        for r in sorted(res_f["cpml"], key=lambda x: x["n_pml"])
        if r["variant"] == v
    ]
    check(
        f"cpml_F_{v}_err_4_to_12",
        f"{errs[0]:.2e}..{errs[-1]:.2e}",
        abs(errs[0] - exp_hi) < 0.15 * exp_hi
        and abs(errs[-1] - exp_lo) < 0.15 * exp_lo,
        f"{exp_hi:.1e}..{exp_lo:.1e}",
    )

# Part E full and quick runs
res_e_full = load("E")
assert res_e_full is not None and not res_e_full["meta"].get("quick", False), (
    "results_E.json was missing or overwritten by --quick"
)
e_full_raised = None
try:
    analyse_cost(res_e_full)
except RuntimeError as exc:
    e_full_raised = str(exc)
max_e_full_ratio = float(max(r["probe_agreement_ratio"] for r in res_e_full["cost"]))
check(
    "cost_E_full_probe_agreement_ratio",
    max_e_full_ratio,
    e_full_raised is not None and max_e_full_ratio > 0.2,
    "> 0.2 (raises RuntimeError)",
)
g3_e = next(
    r
    for r in res_e_full["cost"]
    if r["kind"] == "global" and r["ratio"] == 3 and abs(r["h_coarse"] - 0.025) < 1e-6
)
l3_e = next(
    r
    for r in res_e_full["cost"]
    if r["kind"] == "lts" and r["ratio"] == 3 and abs(r["h_coarse"] - 0.025) < 1e-6
)
check(
    "cost_E_3to1_H025_work_ratio",
    round(g3_e["work"] / l3_e["work"], 2),
    round(g3_e["work"] / l3_e["work"], 2) == 1.80,
    "1.80",
)
check(
    "cost_E_3to1_H025_wall_ratio",
    round(g3_e["wall"] / l3_e["wall"], 2),
    1.28 <= g3_e["wall"] / l3_e["wall"] <= 1.31,
    "~1.29-1.30",
)

res_eq = load("E_quick")
if res_eq is None:
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "studies", "subgrid3d", "run_study_lts.py"),
            "--quick",
            "--parts",
            "E",
        ],
        check=True,
    )
    res_eq = load("E_quick")
assert res_eq is not None and res_eq["meta"].get("quick", False), (
    "results_E_quick.json missing"
)
kinds_eq = {r["kind"] for r in res_eq["cost"]}
assert {"uniform", "graded", "global", "lts"}.issubset(kinds_eq), kinds_eq
for r in res_eq["cost"]:
    assert "probes_dec" in r and len(r["probes_dec"]) == 2
    assert "modes_td_probes" in r and all(len(m) == 2 for m in r["modes_td_probes"])
    assert "modes_semi_probes" in r and all(len(m) == 2 for m in r["modes_semi_probes"])
    samples_per_period = 1.0 / (res_eq["f1_guess"] * r["dt_dec"])
    assert 15.0 <= samples_per_period <= 25.0, samples_per_period
timing_graded = [t for t in res_eq["timing"] if t["kind"] == "graded"]
assert timing_graded and all("solverfit3d_s_per_step" in t for t in timing_graded)

raised_msg = None
try:
    analyse_cost(res_eq)
except RuntimeError as exc:
    raised_msg = str(exc)
assert raised_msg is not None, (
    "Expected analyse_cost(res_eq) to raise RuntimeError on probe disagreement"
)
eq_arms = [r for r in res_eq["cost"] if r["kind"] != "uniform"]
for arm in eq_arms:
    for col in NEW_COLS:
        assert col in arm and arm[col] is not None and np.isfinite(arm[col]), (
            col,
            arm["kind"],
        )
max_eq_ratio = float(max(r["probe_agreement_ratio"] for r in res_eq["cost"]))
check(
    "quick_E_probe_agreement_ratio",
    max_eq_ratio,
    max_eq_ratio > 0.2,
    "> 0.2 (raises RuntimeError)",
)
out["quick_E_probe_agreement_error"] = raised_msg
out["quick_E_graded_wall_gain_semi"] = float(
    next(r["wall_gain_semi"] for r in eq_arms if r["kind"] == "graded")
)
out["quick_E_lts_wall_gain_semi"] = float(
    next(r["wall_gain_semi"] for r in eq_arms if r["kind"] == "lts")
)
out["quick_E_lts_wall_gain_global_semi"] = float(
    next(r["wall_gain_global_semi"] for r in eq_arms if r["kind"] == "lts")
)
print(json.dumps(out, indent=1, default=float))
if failures:
    sys.exit(1)
