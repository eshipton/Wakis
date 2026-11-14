"""Measured values behind the wave-2 verification table (prints JSON).

Uses the helpers of tests/test_104_subgrid3d_lts.py; run from the repo root.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "tests")
import test_104_subgrid3d_lts as T  # noqa: E402

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
m = SubgridLTSFIT3D(T.closed_mesh(), lts=True, dt=1.03 * s.lts_info["dt_max"], **T.closed_materials())
m.e[:] = np.random.default_rng(1).standard_normal(m.n_free)
with np.errstate(all="ignore"):
    Em = m.run(4000, energy=True)
out["mut_103_drift"] = float(np.nanmax(np.abs(Em / Em[0] - 1))) if np.all(np.isfinite(Em)) else "overflow"

out["dot_closed"] = T._dot(SubgridLTSFIT3D(T.closed_mesh(), lts=True, dt=1.0e-10, **T.closed_materials()))
out["dot_cpml_csr"] = T._dot(T._open_box(dt=7.0e-11))
out["dot_cpml_block"] = T._dot(T._open_box(dt=7.0e-11, kernels="block"))
out["mut_dot_R"] = T._dot(T._open_box(dt=7.0e-11), mutate="R")
out["mut_dot_pml"] = T._dot(T._open_box(dt=7.0e-11), mutate="pml")

e8, e16 = T._lts_error(8), T._lts_error(16)
m8, m16 = T._lts_error(8, "consistency"), T._lts_error(16, "consistency")
out.update(cav_e8=e8, cav_e16=e16, cav_order=float(np.log2(e8 / e16)), mut_e8=m8, mut_e16=m16,
           mut_order=float(np.log2(m8 / m16)))
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
print(json.dumps(out, indent=1, default=float))
