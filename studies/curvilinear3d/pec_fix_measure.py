"""Measure PEC-fix effect: TE111 (cylinder) and TE_l1 (sphere) vs N, both PEC reconstructions.

Mode selection: the discrete eigenvalue nearest the analytic omega (positive, > 1e3 rad/s).
Prints one JSON list to stdout.
"""

import json

import numpy as np

from wakis.curvilinear.mappings3d import (
    cylinder_nodes,
    cylinder_te111_omega,
    sphere_lowest_omegas,
    sphere_nodes,
)
from wakis.curvilinear.mesh3d import HexBlock3D
from wakis.curvilinear.solver3d import CurvilinearFIT3D


def main():
    rows = []
    omega_te111 = cylinder_te111_omega(a=1.0, length=1.2)
    omega_te_l1, _ = sphere_lowest_omegas(a=1.0, l=1)
    cases = [
        (
            "cylinder",
            "TE111",
            omega_te111,
            lambda n: cylinder_nodes(n, n, n, a=1.0, length=1.2),
        ),
        ("sphere", "TE_l1", omega_te_l1, lambda n: sphere_nodes(n, n, n, a=1.0)),
    ]
    for geo, mode, omega_ana, nodes in cases:
        for pec in ("constrained", "drop_only"):
            for n in (6, 8, 12, 16):
                blk = HexBlock3D(nodes(n))
                s = CurvilinearFIT3D(blk, _pec_reconstruction=pec)
                w = np.asarray(s.cavity_frequencies(k=8), dtype=float)
                w = w[w > 1e3]
                w_disc = float(w[np.argmin(np.abs(w - omega_ana))])
                rows.append(
                    {
                        "geometry": geo,
                        "mode": mode,
                        "N": n,
                        "pec_reconstruction": pec,
                        "omega_disc": w_disc,
                        "omega_ana": float(omega_ana),
                        "rel_err": (w_disc - omega_ana) / omega_ana,
                        "n_wall_octants": int(getattr(s, "n_wall_octants", -1)),
                        "n_sharp_octants": int(getattr(s, "n_sharp_octants", -1)),
                    }
                )
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
