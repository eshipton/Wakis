"""Figure e: the curved DG meshes (B1 at 2 elements per dielectric
wavelength, B3 dimer gap) with the curved element edges drawn exactly."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def draw(ax, solver, unit=1e-6, lw=0.4):
    T = solver.ref
    t = np.linspace(-1, 1, 12)
    for f in range(3):
        rf, sf = T.face_points(f, t)
        L = T.interp_matrix(rf, sf)
        X, Y = L @ solver.x, L @ solver.y
        ax.plot(X / unit, Y / unit, color="0.25", lw=lw)
    for k in solver.curved_idx:
        f = solver.curved_face[k]
        rf, sf = T.face_points(f, t)
        L = T.interp_matrix(rf, sf)
        ax.plot(L @ solver.x[:, k] / unit, L @ solver.y[:, k] / unit, color="C3", lw=1.2)
    ax.set_aspect("equal")


def main():
    import ho_study as H

    from wakis.highorder import DGTD2D, circles_mesh

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    h = H.S.LAM_MIN_DIEL / 2
    mesh = circles_mesh([(0, 0, H.A)], H.dg_size(h), outer_circle=(0, 0, 1.7e-6))
    s = DGTD2D(mesh, 4, {1: 4.0}, dt=1e-17)
    draw(axs[0], s)
    for r in H.GEOM["compact"]["radii"]:
        axs[0].add_patch(plt.Circle((0, 0), r / 1e-6, fill=False, ls="--", color="C0", lw=0.8))
    axs[0].set_title(f"B1, DGTD p = 4, h = λ_d/2: {s.K} triangles, {s.n_dofs:,} unknowns\n"
                     "(red: curved faces; dashed: extraction circles)")  # fmt: skip
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel("y [µm]")
    from wakis.benchmarks import get_case
    from wakis.highorder.bench_arm import DGTD2DArm

    case = get_case("B3-g5")
    arm = DGTD2DArm(p=4)
    _, s3, _ = arm.build(case, 20e-9 / 4)
    draw(axs[1], s3, unit=1e-9, lw=0.3)
    axs[1].set_xlim(-50, 50)
    axs[1].set_ylim(-30, 30)
    axs[1].set_title(f"B3 Au dimer, 5 nm gap, h = a/4: {s3.K} triangles (zoom)")
    axs[1].set_xlabel("x [nm]")
    axs[1].set_ylabel("y [nm]")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e_curved_meshes.png", dpi=150)


if __name__ == "__main__":
    main()
