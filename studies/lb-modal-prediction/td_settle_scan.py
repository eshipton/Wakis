"""TD LinearizedBeam runs with DFT windows at several settle times vs the exact FD steady state (D21 study).

Usage: python studies/lb-modal-prediction/td_settle_scan.py NX NZ MULT_MAX OUT.npz
Runs the `run_linearized_beam_drift` setup for n_settle_base * MULT_MAX + 6 periods and
accumulates the integer-period DFT of I1(z) on windows starting at n_settle_base * mult
for mult in {1, 1.25, ..., MULT_MAX}; saves them with the exact FD phasor.
"""

import json
import sys
import time

import numpy as np

from examples._space_charge_waves_common import (
    DRIVE_FREQ_HZ,
    compute_analytic_parameters,
    settle_steps,
    fit_beta_q_from_envelope,
    fit_window,
    _build_reference_linearized_beam,
)
from examples._lb_modal_prediction import lb_exact_phasor

nx, nz, mult_max, out = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    float(sys.argv[3]),
    sys.argv[4],
)
p = compute_analytic_parameters()
omega = p["omega"]
beam = _build_reference_linearized_beam(nx, nz)
solver = beam._solver
dt = float(solver.dt)
n_spp = int(round(1.0 / (DRIVE_FREQ_HZ * dt)))
n_base = settle_steps("lb", nx, nz, p, dt)
mults = np.arange(1.0, mult_max + 1e-9, 0.25)
starts = [int(round(n_base * m_)) for m_ in mults]
n_dft = 6 * n_spp
n_total = max(starts) + n_dft
acc = np.zeros((len(starts), beam.M), dtype=np.complex128)
t0 = time.time()
for step in range(n_total):
    solver.one_step()
    tm = (step + 0.5) * dt
    for w, s0 in enumerate(starts):
        if s0 <= step < s0 + n_dft:
            acc[w] += np.exp(-1j * omega * tm) * (2.0 / n_dft) * beam.axial_current()
t_td = time.time() - t0
fd = lb_exact_phasor(nx, nz)
z = fd["z_coords"]
zmin, zmax = fit_window(p)
res = {
    "nx": nx,
    "nz": nz,
    "n_base": n_base,
    "n_spp": n_spp,
    "dt": dt,
    "t_td_s": t_td,
    "fd_solve_resid": fd["solve_resid"],
    "windows": [],
}
bq_fd = fit_beta_q_from_envelope(z, fd["i1"], p["beta_q"], z_min=zmin, z_max=zmax)[
    "beta_q"
]
res["bq_fd"] = bq_fd
for w, m_ in enumerate(mults):
    f = fit_beta_q_from_envelope(z, acc[w], p["beta_q"], z_min=zmin, z_max=zmax)
    d = np.abs(acc[w] - fd["i1"]) / p["I1_0"]
    res["windows"].append(
        {
            "mult": float(m_),
            "start": starts[w],
            "bq": f["beta_q"],
            "rel_vs_fd": f["beta_q"] / bq_fd - 1.0,
            "amp0": float(abs(acc[w][0]) / p["I1_0"]),
            "max_abs_diff_window": float(np.max(d[(z >= zmin) & (z <= zmax)])),
            "max_abs_diff_all": float(np.max(d)),
        }
    )
    print(json.dumps(res["windows"][-1]), flush=True)
print(json.dumps({k: v for k, v in res.items() if k != "windows"}), flush=True)
np.savez(out, z=z, acc=acc, mults=mults, starts=np.array(starts), fd=fd["i1"])
with open(out.replace(".npz", ".json"), "w") as fh:
    json.dump(res, fh, indent=1)
