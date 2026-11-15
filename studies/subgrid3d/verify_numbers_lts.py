"""Measured values behind the wave-2 verification table (prints JSON).

Uses the helpers of tests/test_104_subgrid3d_lts.py; run from the repo root.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "tests")
import test_104_subgrid3d_lts as T  # noqa: E402

from studies.subgrid3d.plot_figures_lts import analyse_cost, load  # noqa: E402
from studies.subgrid3d.run_study_lts import matrix_pencil  # noqa: E402
from wakis.subgrid3d import SubgridLTS, SubgridLTSFIT3D  # noqa: E402
from wakis.subgrid3d.lts import lts_dt_max  # noqa: E402

out = {}
s = SubgridLTSFIT3D(T.closed_mesh(), lts=False, **T.closed_materials())
L = SubgridLTS(s, 2.5 * s.dt_max)
e = np.random.default_rng(0).standard_normal(s.n_free)
full = L.apply_full(e)
out["fast_vs_full"] = float(np.abs(L.apply(e) - full).max() / np.abs(full).max())
out["matrix_vs_full"] = float(np.abs(L.matrix() @ e - full).max() / np.abs(full).max())
L.levels[1].kappa = 1.1 * L.levels[1].kappa
out["mut_kappa"] = float(np.abs(L.apply(e) - full).max() / np.abs(full).max())

for nested in (False, True):
    s0 = SubgridLTSFIT3D(T.cavity_mesh(16, patch=False), lts=False)
    s1 = SubgridLTSFIT3D(T.cavity_mesh(16, nested=nested), lts=True)
    k = "nested" if nested else "single"
    out[f"dt_{k}_over_coarse"] = s1.lts_info["dt_max"] / s0.dt_max
    out[f"dt_{k}_over_global"] = s1.lts_info["dt_max"] / s1.dt_max
    out[f"floor_{k}"] = T.damping_floor(s1.lts)
    if not nested:
        dm, _ = lts_dt_max(s1, 2.2 * s1.dt_max, interface_parents=False)
        out["mut_parents_over_coarse"] = dm / s0.dt_max

s = SubgridLTSFIT3D(T.closed_mesh(), lts=True, cfl=0.99, **T.closed_materials())
s.e[:] = np.random.default_rng(1).standard_normal(s.n_free)
E = s.run(4000, energy=True)
out["energy_drift_4000"] = float(np.abs(E / E[0] - 1).max())
out["energy_tol"] = float(50 * T.EPS * np.sqrt(len(E)))
out["closed_dt_over_global"] = s.lts_info["dt_max"] / s.dt_max
m = SubgridLTSFIT3D(
    T.closed_mesh(), lts=True, dt=1.03 * s.lts_info["dt_max"], **T.closed_materials()
)
m.e[:] = np.random.default_rng(1).standard_normal(m.n_free)
with np.errstate(all="ignore"):
    Em = m.run(4000, energy=True)
out["mut_103_drift"] = (
    float(np.nanmax(np.abs(Em / Em[0] - 1))) if np.all(np.isfinite(Em)) else "overflow"
)

out["dot_closed"] = T._dot(
    SubgridLTSFIT3D(T.closed_mesh(), lts=True, dt=1.0e-10, **T.closed_materials())
)
out["dot_cpml_csr"] = T._dot(T._open_box(dt=7.0e-11))
out["dot_cpml_block"] = T._dot(T._open_box(dt=7.0e-11, kernels="block"))
out["mut_dot_R"] = T._dot(T._open_box(dt=7.0e-11), mutate="R")
out["mut_dot_pml"] = T._dot(T._open_box(dt=7.0e-11), mutate="pml")

e8, e16 = T._lts_error(8), T._lts_error(16)
m8, m16 = T._lts_error(8, "consistency"), T._lts_error(16, "consistency")
out.update(
    cav_e8=e8,
    cav_e16=e16,
    cav_order=float(np.log2(e8 / e16)),
    mut_e8=m8,
    mut_e16=m16,
    mut_order=float(np.log2(m8 / m16)),
)
for lts in (False, True):
    out[f"cpml_residual_lts{lts}"] = T._residual(T._open_box(lts=lts))
    out[f"mut_cpml_residual_lts{lts}"] = T._residual(T._open_box(lts=lts, R0=1.0))

kw = T.closed_materials()
sc = SubgridLTSFIT3D(T.closed_mesh(), lts=True, cfl=0.95, kernels="csr", **kw)
sb = SubgridLTSFIT3D(T.closed_mesh(), lts=True, dt=sc.dt, kernels="block", **kw)
e = np.random.default_rng(5).standard_normal(sc.n_free)
for x in (sc, sb):
    x.e[:] = e
    x.run(500)
out["block_vs_csr_500"] = T._rel(sb.e, sc.e)
out["roundoff_bound_500"] = float(T._roundoff_bound(sc, 500))

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
out["matrix_pencil_rel_err"] = float(np.abs(f_est / f_true - 1).max())
out["mut_matrix_pencil_rel_err"] = float(np.abs(f_mut / f_true - 1).max())
assert out["matrix_pencil_rel_err"] < 1e-12
assert out["mut_matrix_pencil_rel_err"] > 0.04

# Cost study: check new columns on D and exercise probe-agreement raise path on E_quick
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
_, cost_d = analyse_cost(res_d)
assert cost_d["max_probe_agreement_ratio"] <= 0.2
for arm in cost_d["arms"]:
    for col in NEW_COLS:
        assert col in arm and arm[col] is not None and np.isfinite(arm[col]), (col, arm)
out["cost_D_max_probe_agreement_ratio"] = float(cost_d["max_probe_agreement_ratio"])
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

res_e_full = load("E")
assert res_e_full is not None and not res_e_full["meta"].get("quick", False), (
    "results_E.json was missing or overwritten by --quick"
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
assert max_eq_ratio > 0.2
out["quick_E_probe_agreement_ratio"] = max_eq_ratio
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
