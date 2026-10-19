"""
Mirror-symmetry check of the tutorial-01b sphere (n = 2, r = 1 um, TF/SF
plane wave along +z, E along x).

The structure and the incidence are symmetric under x -> -x and y -> -y,
so the total field must satisfy (M = mirror matrix of the plane)

    x -> -x:  M E(M r) = -E(r):  E_x even, E_y odd,  E_z odd   in x
    y -> -y:  M E(M r) = +E(r):  E_x even, E_y odd,  E_z even  in y

(the incident field E = x exp(-ikz) is odd under the x mirror and even
under the y mirror). This script measures

    A = || F(r) - p F(M r) || / || F ||   (p = parity above)

for every component on the raw staggered DFT samples (each component at its
own edge positions, which form a mirror-symmetric set on the centred grid),
for the permittivity of every edge (the material model), and for |E|
assembled (a) as in the original tutorial (components trimmed to a common
shape from the low side, no interpolation) and (b) after interpolating all
components to the cell centres.

Usage (GPU VM; one run per material model):
    python check_symmetry.py --resolution 20 --conformal all --gpu --out sym_all.json
"""

import argparse
import json
import time

import numpy as np
import pyvista as pv
from scipy.constants import c as c0
from scipy.constants import epsilon_0

import wakis
from wakis.sources import GaussianSource

R, N_SPH, N_PML = 1e-6, 2.0, 10
X_MIN, X_MAX, X_SNAP = 2.0, 10.0, 6.0
PARITY = {  # (component, mirror axis) -> parity
    ("x", "x"): 1, ("y", "x"): -1, ("z", "x"): -1,
    ("x", "y"): 1, ("y", "y"): -1, ("z", "y"): 1,
}  # fmt: skip


def mirror_asymmetry(F, coords, axis, parity):
    """||F - p F(mirrored)|| / ||F|| along `axis` (0, 1, 2) of a 3D array."""
    c = np.asarray(coords[axis])
    if not np.allclose(c, -c[::-1], atol=1e-6 * np.ptp(c)):
        raise ValueError(f"samples are not mirror-symmetric along axis {axis}")
    G = np.flip(F, axis=axis)
    return float(np.linalg.norm(F - parity * G) / np.linalg.norm(F))


def to_cell_centres(field, comp):
    """Average an E component from its edge positions to the cell centres."""
    a = "xyz".index(comp)
    out = field
    for d in range(3):
        if d != a:  # node-located along d: average neighbouring nodes
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[d], sl1[d] = slice(0, -1), slice(1, None)
            out = 0.5 * (out[tuple(sl0)] + out[tuple(sl1)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=20)
    ap.add_argument("--conformal", default="all")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--t-run", type=float, default=300.0, help="fs")
    ap.add_argument("--out", default="sym.json")
    a = ap.parse_args()

    dx = 1e-6 / a.resolution
    i_r = int(np.ceil(R / dx - 1e-9))
    i_tf = max(i_r + 3, int(round(1.25e-6 / dx)))
    i_flux = i_tf + max(2, int(round(0.2e-6 / dx)))
    i_pml = i_flux + max(6, int(round(0.6e-6 / dx)))
    n_half = i_pml + N_PML
    h = n_half * dx
    sphere = pv.Sphere(radius=R, theta_resolution=180, phi_resolution=180)
    t0 = time.time()
    grid = wakis.GridFIT3D(
        -h, h, -h, h, -h, h, 2 * n_half, 2 * n_half, 2 * n_half,
        stl_solids={"s": sphere}, stl_materials={"s": [N_SPH**2, 1.0]},
        conformal=False if a.conformal == "staircase" else a.conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=N_PML,
        use_stl=True, use_gpu=a.gpu, verbose=0,
    )  # fmt: skip
    out = dict(
        resolution=a.resolution,
        conformal=a.conformal,
        cells=grid.N if hasattr(grid, "N") else None,
    )
    # Grid symmetry
    out["node_asym"] = float(
        max(np.abs(q + q[::-1]).max() for q in (grid.x, grid.y, grid.z))
    )
    # Material symmetry: eps of each E-edge family under both mirrors
    ieps = solver.ieps.copy()
    if getattr(ieps, "on_gpu", False):
        ieps.from_gpu()
    mat = {}
    nc = 2 * n_half
    for comp in "xyz":
        m = np.asarray(ieps.to_matrix(comp))
        k = "xyz".index(comp)
        sl = [slice(0, nc + 1)] * 3
        sl[k] = slice(0, nc)  # drop the ghost edge beyond the last node
        eps = 1 / (epsilon_0 * m[tuple(sl)])
        for ax in "xy":
            d = "xyz".index(ax)
            mat[f"eps_{comp}_mirror_{ax}"] = float(
                np.abs(eps - np.flip(eps, axis=d)).max()
            )
    out["material_asym_max_abs"] = mat

    f_snap = X_SNAP * c0 / (2 * np.pi * R)
    freqs = np.linspace(X_MIN, X_MAX, 81) * c0 / (2 * np.pi * R)
    pulse = GaussianSource(frequency=freqs.mean(), fwidth=0.25 * np.ptp(freqs))
    src = wakis.TFSFSource(pulse, size=(2 * i_tf * dx,) * 3, direction="+z",
                           polarization="x")  # fmt: skip
    vol = wakis.DFTMonitor([f_snap], components=("Ex", "Ey", "Ez"),
                           size=(2 * i_tf * dx,) * 3)  # fmt: skip
    t_setup = time.time() - t0
    solver.emsolve(int(a.t_run * 1e-15 / solver.dt), source=[src], monitors=[vol])
    out["run_s"] = time.time() - t0 - t_setup

    raw, F = {}, {}
    for comp in "xyz":
        val, coords = vol.get_field("E" + comp)
        F[comp] = val[0]
        for ax in "xy":
            raw[f"E{comp}_mirror_{ax}"] = mirror_asymmetry(
                val[0], coords, "xyz".index(ax), PARITY[(comp, ax)]
            )
        raw[f"E{comp}_shape"] = list(val[0].shape)
    out["field_asym"] = raw

    # |E| (a) as in the original tutorial: trim to a common shape, no interpolation
    shape = np.min([F[c].shape for c in "xyz"], axis=0)
    mag_old = np.sqrt(
        sum(np.abs(F[c][: shape[0], : shape[1], : shape[2]]) ** 2 for c in "xyz")
    )
    # (b) all components averaged to the cell centres
    mag_new = np.sqrt(sum(np.abs(to_cell_centres(F[c], c)) ** 2 for c in "xyz"))
    sym = lambda m, d: float(np.linalg.norm(m - np.flip(m, d)) / np.linalg.norm(m))  # noqa: E731
    out["absE_old"] = {"mirror_x": sym(mag_old, 0), "mirror_y": sym(mag_old, 1),
                       "shape": list(mag_old.shape)}  # fmt: skip
    out["absE_centres"] = {"mirror_x": sym(mag_new, 0), "mirror_y": sym(mag_new, 1),
                           "shape": list(mag_new.shape)}  # fmt: skip
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
