"""Numbers behind the D21 figures: first-pass prediction, slack, error decomposition, production LB run.

Usage: python studies/lb-modal-prediction/d21_numbers.py [OUT_DIR]
Runs the coarse (10 x 474) and fine (20 x 948) grids in parallel (one process each) and
writes ``OUT_DIR/d21_numbers_<nx>x<nz>.json`` (default ``studies/lb-modal-prediction/out``).
Plotted by ``plot_figures.py`` together with the ``td_settle_scan.py`` outputs.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

GRIDS = ((10, 474), (20, 948))


def _one(grid: tuple[int, int]) -> str:
    import numpy as np

    from examples._lb_modal_prediction import (
        lb_error_decomposition,
        lb_first_pass_prediction,
    )
    from examples._space_charge_waves_common import (
        compute_analytic_parameters,
        fit_beta_q_from_envelope,
        fit_window,
        run_linearized_beam_drift,
    )

    nx, nz = grid
    t0 = time.time()
    d21 = lb_first_pass_prediction(nx, nz)
    dec = lb_error_decomposition(nx, nz)
    t_pred = time.time() - t0
    params = compute_analytic_parameters()
    z_min, z_max = fit_window(params)
    t0 = time.time()
    lb = run_linearized_beam_drift(nx, nz)
    bq_lb = float(
        fit_beta_q_from_envelope(
            lb["z_coords"], lb["i1_lb"], params["beta_q"], z_min=z_min, z_max=z_max
        )["beta_q"]
    )
    t_td = time.time() - t0
    out = {
        "nx": nx,
        "nz": nz,
        "d21": {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in d21.items()
            if not isinstance(v, np.ndarray)
        },
        "decomposition": {k: float(v) for k, v in dec.items()},
        "bq_lb_production": bq_lb,
        "resid_d21": bq_lb / d21["bq_pred"] - 1.0,
        "resid_legacy": bq_lb / dec["bq_legacy"] - 1.0,
        "t_end_over_t_reach0": d21["t_end"] / d21["t_reach0"],
        "t_pred_s": t_pred,
        "t_td_s": t_td,
    }
    return json.dumps(out, indent=1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "studies/lb-modal-prediction/out"
    os.makedirs(out_dir, exist_ok=True)
    with Pool(len(GRIDS)) as pool:
        for (nx, nz), txt in zip(GRIDS, pool.map(_one, GRIDS)):
            path = os.path.join(out_dir, f"d21_numbers_{nx}x{nz}.json")
            with open(path, "w") as fh:
                fh.write(txt)
            print(path, txt, flush=True)
