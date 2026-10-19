"""
Tutorial visualization helpers (wakis.visualize): the data they extract
(edge permittivities, slices, recorded frames) and that the figures render
off screen. Rendering tests are skipped when no OpenGL context is available.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import epsilon_0

import wakis
from wakis import visualize as wv
from wakis.sources import CurrentSource, GaussianSource

R = 0.5
EPS = 4.0


def _solver(conformal, bc="pec"):
    sphere = pv.Sphere(radius=R, theta_resolution=60, phi_resolution=60)
    grid = wakis.GridFIT3D(
        -1, 1, -1, 1, -1, 1, 20, 20, 20,
        stl_solids={"ball": sphere}, stl_materials={"ball": [EPS, 1.0]},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_stl=True, n_pml=4, verbose=0
    )


def test_edge_permittivity_staircase_vs_conformal():
    stair = _solver(False)
    conf = _solver("all")
    for s in (stair, conf):
        eps, (cx, cy) = wv.edge_permittivity(s, "x", axis="z", position=0.0)
        assert eps.shape == (cx.size, cy.size)
        # Edge midpoints along x, nodes along y
        assert np.allclose(cx[:-1], 0.5 * (s.x[:-1] + s.x[1:]))
        assert np.allclose(cy, s.y)
        ix, iy = np.argmin(np.abs(cx[:-1])), np.argmin(np.abs(cy))
        assert eps[ix, iy] == pytest.approx(EPS)  # centre of the sphere
        assert eps[1, 1] == pytest.approx(1.0)  # far outside
    e_s, _ = wv.edge_permittivity(stair, "x", "z", 0.0)
    e_c, _ = wv.edge_permittivity(conf, "x", "z", 0.0)
    finite = np.isfinite(e_s)
    # Staircase: only the two values; conformal: intermediate values at the
    # surface (this fails if the conformal materials were not applied)
    assert set(np.round(e_s[finite], 9)) <= {1.0, EPS}
    inter = (e_c > 1.01) & (e_c < EPS - 0.01)
    assert inter.sum() >= 8
    # The ieps -> eps_r conversion is the solver's own
    assert np.allclose(1 / (epsilon_0 * stair.ieps.to_matrix("x")[5, 5, 10]), e_s[5, 5])


def test_cpml_bounds_slices_and_recorder(tmp_path):
    s = _solver(False, bc="cpml")
    b = wv.cpml_inner_bounds(s)
    assert b[0] == pytest.approx(s.x[4]) and b[1] == pytest.approx(s.x[-5])
    assert wv.box_bounds((0, 0, 1), (2, 4, 6)) == (-1, 1, -2, 2, -2, 4)
    flat, style = wv._split_spec(((0, 1), (2, 3), (4, 5)))
    assert flat == (0, 1, 2, 3, 4, 5) and style is None

    pulse = GaussianSource(frequency=3e8, fwidth=1e8)
    src = CurrentSource(pulse, component="x", center=(0.3, 0.0, 0.0))
    rec = wv.SliceRecorder("Ex", axis="y", position=0.0, every=5)
    dft = wakis.DFTMonitor([3e8], components=("Ex",), size=(1.0, 0.0, 1.0))
    s.emsolve(41, source=[src], monitors=[rec, dft])
    assert (
        len(rec.frames) == 9
    )  # steps 0, 5, ..., 40 and rec.frames[0].shape == (21, 21)
    assert np.abs(rec.frames[-1]).max() > 0
    values, (cu, cv) = wv.field_slice(dft, "y", 0.0, "Ex")
    assert values.shape == (1, cu.size, cv.size)
    snap, _ = wv.field_slice(s.E, "y", 0.0, "x", solver=s)
    assert np.allclose(snap, rec.frames[-1], rtol=0, atol=0)  # last frame = final field
    assert rec.times[-1] == pytest.approx(41 * s.dt)
    late = wv.SliceRecorder("Ex", axis="y", every=1, t_stop=10 * s.dt)
    for n in range(20):
        late.update(s, n * s.dt)
    assert len(late.frames) == 11  # t = 0 ... 10 dt
    gif = wv.save_animation(rec, str(tmp_path / "m.gif"), max_frames=4, dpi=40)
    assert (tmp_path / "m.gif").stat().st_size > 1000 and gif.endswith(".gif")


def test_matplotlib_figures(tmp_path):
    import matplotlib.pyplot as plt

    s = _solver("all")
    fig, ax = plt.subplots()
    lc = wv.plot_material_slice(s, "z", 0.0, surface="ball", ax=ax, vmax=EPS)
    assert lc.get_array().max() == pytest.approx(EPS)
    fig, ax = plt.subplots()
    u = np.linspace(-1, 1, 11)
    wv.plot_field_slice(np.outer(u, u) + 1j, (u, u), kind="log", ax=ax,
                        boxes={"tfsf": wv.box_bounds((0, 0, 0), (1, 1, 1))},
                        surface=((0, 0, 0), R), length_unit="m")  # fmt: skip
    fig.savefig(tmp_path / "f.png")
    plt.close("all")


def test_render_scene(tmp_path):
    s = _solver(False, bc="cpml")
    try:
        pl = wv.plot_scene(
            s.grid,
            boxes={"tfsf": wv.box_bounds((0, 0, 0), (1.4,) * 3)},
            planes={"source": ((0, 0, -0.7), "z", (1.4, 1.4))},
            cpml=(s.grid.grid.bounds, wv.cpml_inner_bounds(s)),
            length_unit="m",
        )
        img = wv.save_plotter(pl, str(tmp_path / "scene.png"))
    except RuntimeError as err:  # no OpenGL context on this machine
        pytest.skip(f"off-screen rendering unavailable: {err}")
    assert img.shape[0] > 100 and img.std() > 1
