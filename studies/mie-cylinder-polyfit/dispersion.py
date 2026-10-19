"""
Numerical dispersion of 2D FIT (scalar node field, diagonal Hodge, leapfrog)
on the three periodic lattices behind the study's meshes, compared at equal
field unknowns (nodes + edges) per unit area:

* squares, edge h:                    D = 3 / h^2           (Yee / Cartesian FIT)
* equilateral triangles, edge h:      D = 4 / (sqrt(3)/2 h^2)  (Delaunay primal,
  hexagonal Voronoi dual; nodes on the triangular lattice)
* regular hexagons, edge h:           D = 5 / (3 sqrt(3)/2 h^2) (Voronoi primal,
  triangular dual; nodes on the honeycomb)

Bloch analysis: K(k) = sum_e w_e b_e^H b_e with b_e = e^{i k.R} at the far
node and -1 at the near node, w_e = (dual length) / (edge length), node
masses = dual areas (c = 1). The leapfrog frequency is
sin(omega dt / 2) = dt sqrt(lambda) / 2 with dt = cfl * 2 / sqrt(max lambda).
The phase-velocity error omega / |k| - 1 of the acoustic branch is taken at
its worst propagation angle.

    python dispersion.py      # writes figures/e_dispersion.png and data/dispersion.json
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
S3 = np.sqrt(3)


def lattices(h):
    """name -> (masses, edges [(i, j, R, w)], unknowns per area, BZ vectors)."""
    sq = (np.array([h * h]), [(0, 0, np.array([h, 0.0]), 1.0), (0, 0, np.array([0.0, h]), 1.0)],
          3 / h**2)  # fmt: skip
    tri = (np.array([S3 / 2 * h * h]),
           [(0, 0, h * np.array(v), 1 / S3) for v in ((1, 0), (0.5, S3 / 2), (-0.5, S3 / 2))],
           4 / (S3 / 2 * h * h))  # fmt: skip
    s = S3 * h
    a1, a2 = s * np.array([1.0, 0.0]), s * np.array([0.5, S3 / 2])
    hexa = (np.full(2, S3 / 4 * s * s),
            [(1, 0, np.zeros(2), S3), (1, 0, a1, S3), (1, 0, a2, S3)],
            5 / (S3 / 2 * s * s))  # fmt: skip
    return {"squares": sq, "triangles": tri, "hexagons": hexa}


def bloch_eigs(masses, edges, k):
    n = len(masses)
    K = np.zeros((n, n), complex)
    for i, j, R, w in edges:
        b = np.zeros(n, complex)
        b[i] -= 1
        b[j] += np.exp(1j * k @ R)
        K += w * np.outer(b.conj(), b)
    Mi = 1 / np.sqrt(masses)
    return np.sort(np.linalg.eigvalsh(Mi[:, None] * K * Mi[None, :]))


def max_lambda(masses, edges, h):
    kk = np.linspace(-np.pi / h * 1.2, np.pi / h * 1.2, 41)
    return max(bloch_eigs(masses, edges, np.array([kx, ky]))[-1] for kx in kk for ky in kk)


def phase_error(name, h, wavelength, cfl):
    masses, edges, D = lattices(h)[name]
    k0 = 2 * np.pi / wavelength
    lam_max = max_lambda(masses, edges, h)
    dt = cfl * 2 / np.sqrt(lam_max)
    errs = []
    for th in np.linspace(0, np.pi / 2, 31):
        k = k0 * np.array([np.cos(th), np.sin(th)])
        lam = bloch_eigs(masses, edges, k)[0]
        om = 2 / dt * np.arcsin(min(1.0, dt * np.sqrt(lam) / 2)) if cfl > 0 else np.sqrt(lam)
        errs.append(om / k0 - 1)
    errs = np.array(errs)
    return D * wavelength**2, errs[np.argmax(np.abs(errs))], dt


def main():
    out = {}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    style = {"squares": ("#E69F00", "s"), "triangles": ("#0072B2", "o"), "hexagons": ("#CC79A7", "h")}
    for cfl, ls in ((0.95, "-"), (0.0, ":")):
        for name in ("squares", "triangles", "hexagons"):
            # sweep h so that unknowns per wavelength^2 span 300..30000
            pts = []
            for D_target in np.geomspace(300, 30000, 9):
                D1 = lattices(1.0)[name][2]
                h = np.sqrt(D1 / D_target)
                d, e, dt = phase_error(name, h, 1.0, cfl if cfl > 0 else 1e-9)
                pts.append((d, abs(e), dt / h))
            pts = np.array(pts)
            out[f"{name}_cfl{cfl}"] = pts.tolist()
            c, m = style[name]
            ax.loglog(pts[:, 0], pts[:, 1], ls=ls, color=c, marker=m if cfl else None, ms=4,
                      label=f"{name} (leapfrog, {cfl} CFL)" if cfl else f"{name} (dt -> 0)")  # fmt: skip
    ax.set_xlabel("field unknowns per wavelength²")
    ax.set_ylabel("max |phase-velocity error|")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7)
    ax.set_title("2D FIT dispersion at equal unknowns (derived, Bloch analysis)", fontsize=9)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e_dispersion.png", dpi=150)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "dispersion.json").write_text(json.dumps(out))
    for k, v in out.items():
        v = np.array(v)
        i = np.argmin(np.abs(v[:, 0] - 3000))
        print(f"{k:22s} at {v[i, 0]:.0f} unknowns/λ²: error {v[i, 1]:.2e}, c dt / h = {v[i, 2]:.3f}")


if __name__ == "__main__":
    main()
