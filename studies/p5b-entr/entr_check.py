"""Falsifier for the shape-p PEC-wall entrance-amplitude term (open_questions 2026-11-05 item 1).

Usage: python entr_check.py LABEL OUT.json, LABEL in {p3, p1, nx10nz948, nx20nz474, cfl025, base}.
Prints ENTR <label> <json> with |I1(z0)|/I1_0 and its excess over sinc(be dz/2)^p cos(bq dz/2).
"""
import json
import sys

import numpy as np

import examples._space_charge_waves_common as s

RUNS = {
    "base": dict(nx=20, nz=948, shape=2, cfln=s.CFL_NUMBER),
    "p3": dict(nx=20, nz=948, shape=3, cfln=s.CFL_NUMBER),
    "p1": dict(nx=20, nz=948, shape=1, cfln=s.CFL_NUMBER),
    "nx10nz948": dict(nx=10, nz=948, shape=2, cfln=s.CFL_NUMBER),
    "nx20nz474": dict(nx=20, nz=474, shape=2, cfln=s.CFL_NUMBER),
    "cfl025": dict(nx=20, nz=948, shape=2, cfln=0.25),
}
label, out = sys.argv[1], sys.argv[2]
r = RUNS[label]
params = s.compute_analytic_parameters()
beam_ref = s._build_reference_linearized_beam(r["nx"], r["nz"])
res = s.run_particle_beam_drift(
    r["nx"], r["nz"], beam_ref=beam_ref, qm_scale=1.0, seed=42, m_mod=s.MODULATION_INDEX,
    profile="sheet_lattice", cfln=r["cfln"], shape=r["shape"],
)
dz = s.TUNNEL_LENGTH_M / r["nz"]
pred = float(np.sinc(params["beta_e"] * dz / 2 / np.pi) ** r["shape"] * np.cos(params["beta_q"] * dz / 2))
amp = np.abs(np.asarray(res["i1_part"])[:4]) / params["I1_0"]
row = {**r, "amp": amp.tolist(), "pred": pred, "excess0": float(amp[0] - pred), "dt": float(res["dt"])}
print("ENTR", label, json.dumps(row), flush=True)
json.dump(row, open(out, "w"), indent=1)
