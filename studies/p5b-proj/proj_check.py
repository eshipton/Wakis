"""Falsifier for the measurement-projection hypothesis (V7 shape-2 beta_q residual).

Runs the particle drift twice per grid: observable integrated over the LinearizedBeam top-hat
lines (beam_ref, as in test_087) and over every interior E_z line (full-width reference beam).
Prediction: beta_meas(all lines) - beta_meas(top-hat) ~ +2.79e-4 rel (coarse), ~ +1.6e-4 rel (fine).
"""
import json
import sys

import examples._space_charge_waves_common as s

out = {}
for nx, nz, n in ((10, 474, 16), (20, 948, 32)):
    params = s.compute_analytic_parameters()
    z_min, z_max = s.fit_window(params)
    lb = s.run_linearized_beam_drift(nx, nz)
    # Measurement-only reference: b = a makes every interior Ez line active.
    # compute_analytic_parameters() is singular at b = a, so hand it the true params.
    orig_b, orig_cap = s.BEAM_HALF_THICKNESS_M, s.compute_analytic_parameters
    try:
        s.BEAM_HALF_THICKNESS_M = s.TUNNEL_HALF_WIDTH_M
        s.compute_analytic_parameters = lambda: params
        full = s._build_reference_linearized_beam(nx, nz)
    finally:
        s.BEAM_HALF_THICKNESS_M, s.compute_analytic_parameters = orig_b, orig_cap
    row = {"N_lines_tophat": int(lb["beam"].N_lines), "N_lines_full": int(full.N_lines)}
    for tag, ref in (("tophat", lb["beam"]), ("all", full)):
        res = s.run_particle_beam_drift(nx, nz, beam_ref=ref, qm_scale=1.0, seed=42,
                                        m_mod=s.MODULATION_INDEX, profile="sheet_lattice",
                                        n_macro_per_step=n)
        fit = s.fit_beta_q_from_envelope(ref.s_coords, res["i1_part"], params["beta_q"],
                                         z_min=z_min, z_max=z_max)
        row[f"bq_{tag}"] = float(fit["beta_q"])
    row["rel_shift"] = row["bq_all"] / row["bq_tophat"] - 1.0
    out[f"{nx}x{nz}_n{n}"] = row
    print("PROJ", nx, nz, n, json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "proj_check.json", "w"), indent=1)
