"""
Explicit TE/TM scheme (``SolverFIT3D(scheme="tete")``, `wakis.tete`).

References: M. Dohlus, I. Zagorodnov, J. Comput. Phys. 228, 2822 (2009),
arXiv:0901.1053 (explicit scheme, eqs. 23, 25, 27); I. Zagorodnov,
T. Weiland, Phys. Rev. ST Accel. Beams 8, 042001 (2005).

Every test has a failing counterpart (Yee, a mutated coefficient or a time
step above the stated limit) so that a pass is evidence.
"""

import numpy as np
import pytest
from scipy.constants import c as c_light

import wakis
from wakis.inverse import AdjointStepper

# ----------------------------------------------------------------- helpers


def _box(nx, ny, nz, dx, dy, dz, bc_low, bc_high, scheme="tete", use_gpu=False,
         **kw):  # fmt: skip
    grid = wakis.GridFIT3D(0, nx * dx, 0, ny * dy, 0, nz * dz, nx, ny, nz, verbose=0)
    return wakis.SolverFIT3D(
        grid, bc_low=bc_low, bc_high=bc_high, scheme=scheme, use_gpu=use_gpu,
        verbose=0, **kw,
    )  # fmt: skip


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _measured_omega(solver, comp, field, kvec, n_steps=60):
    """
    omega of a single standing Fourier mode: initialise one component with
    cos(k . r), record it at one sample and fit the exact two-term recurrence
    f(n+1) + f(n-1) = 2 cos(omega dt) f(n).
    """
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    dx, dy, dz = solver.grid.dx[0], solver.grid.dy[0], solver.grid.dz[0]
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    phase = kvec[0] * i * dx + kvec[1] * j * dy + kvec[2] * k * dz
    solver.one_step()  # initialisation (ghosts, operator cleanup)
    xp = solver.E.xp
    solver.E.fromarray(xp.zeros_like(solver.E.array))
    solver.H.fromarray(xp.zeros_like(solver.H.array))
    getattr(solver, field)[:, :, :, comp] = np.cos(phase)
    f = []
    for _ in range(n_steps):
        solver.one_step()
        f.append(float(_host(getattr(solver, field)[1, 1, 1, comp])))
    f = np.array(f)
    cos_wdt = np.dot(f[1:-1], f[2:] + f[:-2]) / (2 * np.dot(f[1:-1], f[1:-1]))
    return np.arccos(cos_wdt) / solver.dt


def _dz27(kvec, d, dt):
    """Dohlus & Zagorodnov eq. (27): omega of the explicit TE/TM scheme."""
    K = 0.5 * np.asarray(kvec) * np.asarray(d)
    rhs = np.sin(K[2]) ** 2 / d[2] ** 2 + (
        np.sin(K[0]) ** 2 / d[0] ** 2 + np.sin(K[1]) ** 2 / d[1] ** 2
    ) * (1 - (c_light * dt / d[2]) ** 2 * np.sin(K[2]) ** 2)
    return 2 / dt * np.arcsin(c_light * dt * np.sqrt(rhs))


# --------------------------------------------------- dispersion relation

D = (2e-3, 2e-3, 1e-3)  # dx = dy = 2 dz: magic step c dt = dz is stable


def _periodic(nx, ny, nz, scheme="tete", use_gpu=False):
    return _box(nx, ny, nz, *D, ["periodic"] * 3, ["periodic"] * 3, scheme,
                use_gpu)  # fmt: skip


@pytest.mark.parametrize("m", [1, 3, 6])
def test_dispersion_free_along_z(m, use_gpu):
    """
    A wave along z travels at exactly c for every kz at c dt = dz
    (|v_ph/c - 1| ~ 1e-13 even at 5.3 cells per wavelength), while Yee at its
    own stable step lags by 1e-3 to 6e-2 (eq. 11 of Dohlus & Zagorodnov).
    """
    nz = 32
    kz = 2 * np.pi * m / (nz * D[2])
    errs = {}
    for scheme in ("tete", "yee"):
        s = _periodic(2, 2, nz, scheme, use_gpu)
        w = _measured_omega(s, "x", "E", (0, 0, kz))
        errs[scheme] = abs(w / (c_light * kz) - 1)
    assert errs["tete"] < 1e-11
    assert errs["yee"] > 1e-3


def test_dispersion_mutation_fails():
    """Perturbing the longitudinal coupling by 1e-3 is detected (~5e-4 error)."""
    nz = 32
    kz = 2 * np.pi * 3 / (nz * D[2])
    s = _periodic(2, 2, nz)
    s._tete["LH"] = s._tete["LH"] * (1 + 1e-3)
    w = _measured_omega(s, "x", "E", (0, 0, kz))
    assert abs(w / (c_light * kz) - 1) > 1e-4


@pytest.mark.parametrize("deg", [0, 30, 60, 90])
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_dispersion_relation_vs_angle(deg, pol):
    """
    Oblique plane waves (k in the xz plane, both polarisations) follow
    eq. (27) of Dohlus & Zagorodnov to 1e-10; the same measurement on the
    Yee scheme deviates from eq. (27) (it follows the Yee relation instead).
    """
    nx, nz = 16, 32
    th = np.radians(deg)
    # Lattice wave vector closest to the requested angle
    mx = int(round(3 * np.sin(th)))
    mz = int(round(3 * np.cos(th) * 2))
    k = (2 * np.pi * mx / (nx * D[0]), 0.0, 2 * np.pi * mz / (nz * D[2]))
    comp, field = ("y", "E") if pol == "TE" else ("y", "H")
    s = _periodic(nx, 2, nz)
    w = _measured_omega(s, comp, field, k)
    assert abs(w / _dz27(k, D, s.dt) - 1) < 1e-10
    if 0 < deg < 90:  # along z and in the xy plane (27) is the Yee relation
        y = _periodic(nx, 2, nz, "yee")
        wy = _measured_omega(y, comp, field, k)
        assert abs(wy / _dz27(k, D, y.dt) - 1) > 1e-4


# ------------------------------------------------------------- stability


def _spectral_radius(solver):
    """Largest |eigenvalue| of the one-step map on (H, E) (dense, tiny grid)."""
    stepper = AdjointStepper(solver)
    zero = stepper.zero_state()
    n = zero["H"].size
    cols = []
    for q in range(2 * n):
        x = {k: np.zeros_like(v) for k, v in zero.items()}
        (x["H"] if q < n else x["E"])[q % n] = 1.0
        stepper.set_state(x)
        solver.one_step()
        y = stepper.get_state()
        cols.append(np.concatenate([y["H"], y["E"]]))
    return np.abs(np.linalg.eigvals(np.array(cols).T)).max()


def test_stable_at_magic_step():
    """PEC box, dx = dy = 2 dz, c dt = dz: |eigenvalues| = 1 (lossless)."""
    s = _box(4, 4, 6, *D, ["pec"] * 3, ["pec"] * 3)
    s.one_step()
    assert s.dt == pytest.approx(D[2] / c_light, rel=1e-12)
    assert _spectral_radius(s) < 1 + 1e-9


def test_unstable_above_transverse_limit():
    """
    dx = dy = dz: the transverse limit c dt <= dx / sqrt(2) of eq. (25) is
    sharp. The magic step is refused (ValueError); at 0.99 of the limit the
    step is stable, at 1.03 it is not.
    """
    d = (1e-3, 1e-3, 1e-3)
    # periodic x, y with even N: the transverse Nyquist mode (the one that
    # sets the limit) exists on the grid
    lo, hi = ["periodic", "periodic", "pec"], ["periodic", "periodic", "pec"]
    with pytest.raises(ValueError, match="exceeds the stable limit"):
        _box(4, 4, 6, *d, lo, hi)
    limit = d[0] / np.sqrt(2) / c_light
    s = _box(4, 4, 6, *d, lo, hi, dt=0.99 * limit)
    s.one_step()
    assert _spectral_radius(s) < 1 + 1e-9
    s.dt = 1.03 * limit
    assert _spectral_radius(s) > 1 + 1e-3


def test_unsupported_features_raise():
    with pytest.raises(NotImplementedError, match="CPML on x/y"):
        _box(4, 4, 6, *D, ["cpml"] * 3, ["cpml"] * 3, n_pml=2)
    with pytest.raises(ValueError, match="Invalid scheme"):
        _box(4, 4, 6, *D, ["pec"] * 3, ["pec"] * 3, scheme="abc")


# ------------------------------------------------------- exact adjoint


@pytest.mark.parametrize(
    "bc, bg, dtype",
    [
        ("pec", (1.0, 1.0, 0.0), np.float64),
        ("pec", (2.0, 1.0, 0.05), np.float64),
        ("periodic", (1.0, 1.0, 0.0), np.float64),
        ("cpml", (1.0, 1.0, 0.0), np.float64),
        ("bloch", (1.0, 1.0, 0.0), np.complex128),
    ],
    ids=["pec", "pec-lossy", "periodic", "cpml-z", "bloch-complex"],
)
def test_transposed_step(bc, bg, dtype, use_gpu):
    """
    Dot-product test <U x, y> = <x, U^H y> of the TE/TM step including the
    z-CPML psi variables (also with psi-only supports): ~1e-16 relative.
    """
    if bc == "cpml":
        lo, hi = ["pec", "pec", "cpml"], ["pec", "pec", "cpml"]
    elif bc == "bloch":
        lo, hi = ["bloch"] * 3, ["bloch"] * 3
    else:
        lo, hi = [bc] * 3, [bc] * 3
    kw = {"k_bloch": (30.0, -20.0, 50.0)} if bc == "bloch" else {}
    s = _box(5, 4, 10, *D, lo, hi, bg=list(bg), n_pml=3, dtype=dtype,
             use_gpu=use_gpu, **kw)  # fmt: skip
    s.one_step()
    st = AdjointStepper(s)
    xp = st.xp
    rng = np.random.default_rng(3)
    keys = list(st.zero_state())
    psi = [k for k in keys if k.startswith("tete_psi")]
    assert bool(psi) == (bc == "cpml")

    def rand(support):
        out = {}
        for k, v in st.zero_state().items():
            r = rng.standard_normal(v.shape)
            if np.iscomplexobj(v):
                r = r + 1j * rng.standard_normal(v.shape)
            out[k] = xp.asarray(r * (k in support))
        return out

    supports = [(keys, keys)] + ([(psi, keys), (keys, psi)] if psi else [])
    for xs, ys in supports:
        x = rand(xs)
        st.set_state(x)
        s.one_step()
        Ux = st.get_state()
        y = rand(ys)
        Uty = st.step(y)
        lhs, rhs = st.inner(Ux, y), st.inner(x, Uty)
        scale = st.inner({k: abs(v) for k, v in Ux.items()},
                         {k: abs(v) for k, v in y.items()}).real  # fmt: skip
        assert abs(lhs - rhs) < 1e-13 * scale


def test_transposed_step_mutation_fails():
    """A 1e-3 error in one transposed block breaks the dot-product test."""
    s = _box(5, 4, 10, *D, ["pec"] * 3, ["pec"] * 3)
    s.one_step()
    st = AdjointStepper(s)
    st.tete.t["TH"] = st.tete.t["TH"] * (1 + 1e-3)
    rng = np.random.default_rng(4)
    x = {k: rng.standard_normal(v.shape) for k, v in st.zero_state().items()}
    y = {k: rng.standard_normal(v.shape) for k, v in st.zero_state().items()}
    st.set_state(x)
    s.one_step()
    lhs, rhs = st.inner(st.get_state(), y), st.inner(x, st.step(y))
    assert abs(lhs - rhs) > 1e-9 * abs(lhs)


# --------------------------------------- smooth pipe: no Cherenkov wake


def _pipe_wake(scheme, tmp_path, length=0.1, mutate=None, use_gpu=False):
    """
    Beam (sigma = 5 mm, beta = 1) through a smooth rectangular PEC pipe
    (2 cm x 2 cm, dx = dy = 2 mm, dz = 1 mm), CPML at both z ends with the
    TFSF injection of the bunch self-field; longitudinal wake potential from
    the Wakis post-processing.
    """
    npml, dz, a = 10, 1e-3, 0.02
    nz = int(round(length / dz)) + 2 * npml
    grid = wakis.GridFIT3D(-a / 2, a / 2, -a / 2, a / 2, 0, nz * dz, 10, 10, nz,
                           verbose=0)  # fmt: skip
    wake = wakis.WakeSolver(
        q=1e-9, sigmaz=5e-3, beta=1.0, xsource=0, ysource=0, xtest=0, ytest=0,
        skip_cells=npml + 2, results_folder=str(tmp_path / scheme) + "/",
        Ez_file=str(tmp_path / f"{scheme}_Ez.h5"), save=False, verbose=0,
    )  # fmt: skip
    s = wakis.SolverFIT3D(grid, wake, bc_low=["pec", "pec", "cpml"],
                          bc_high=["pec", "pec", "cpml"], n_pml=npml, verbose=0,
                          scheme=scheme, use_gpu=use_gpu)  # fmt: skip
    if mutate is not None:
        s._tete[mutate] = s._tete[mutate] * (1 + 1e-3)
    s.wakesolve(wakelength=0.05, compute_plane="longitudinal")
    return np.abs(wake.WP).max()


@pytest.fixture
def long_tfsf(monkeypatch):
    # Inject until the tail is exp(-32) below the peak: the default 5 sigma
    # truncation leaves a 4e-6 residual field (same for both schemes)
    monkeypatch.setattr(wakis.sources.Beam, "tfsf_truncation", 8.0)


def test_smooth_pipe_no_cherenkov(tmp_path, long_tfsf, use_gpu):
    """
    Uniform PEC pipe: the exact wake is zero. TE/TM gives |W| < 1e-8 V/pC
    (observed 7e-11), the Yee scheme 0.37 V/pC (numerical Cherenkov
    radiation), growing with the pipe length.
    """
    tete = _pipe_wake("tete", tmp_path, use_gpu=use_gpu)
    yee = _pipe_wake("yee", tmp_path, use_gpu=use_gpu)
    assert tete < 1e-8
    assert yee > 0.1
    assert _pipe_wake("yee", tmp_path, length=0.2, use_gpu=use_gpu) > 1.5 * yee


@pytest.mark.parametrize("block", ["LH", "ZE"])
def test_smooth_pipe_mutation_fails(tmp_path, long_tfsf, block):
    """A 1e-3 error in one TE/TM coefficient block gives a visible wake."""
    assert _pipe_wake("tete", tmp_path, mutate=block) > 1e-5


def _round_pipe_wake(scheme, tmp_path, template=None):
    """Conformal round pipe (r = 6 mm, dx = dy = 2 dz = 1.6 mm), CPML + TFSF."""
    import pyvista as pv

    npml, dz, a, length = 6, 8e-4, 0.016, 0.03
    nz = int(round(length / dz)) + 2 * npml
    zmax = nz * dz
    cyl = pv.Cylinder(center=(0, 0, zmax / 2), direction=(0, 0, 1), radius=0.006,
                      height=2 * zmax, resolution=360).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(-a / 2, a / 2, -a / 2, a / 2, 0, zmax, 10, 10, nz,
                           stl_solids={"pipe": cyl},
                           stl_materials={"pipe": "vacuum"}, conformal=True,
                           verbose=0)  # fmt: skip
    wake = wakis.WakeSolver(
        q=1e-9, sigmaz=3.2e-3, beta=1.0, xsource=0, ysource=0, xtest=0, ytest=0,
        skip_cells=npml + 2, results_folder=str(tmp_path / scheme) + "/",
        Ez_file=str(tmp_path / f"rd_{scheme}_Ez.h5"), save=False, verbose=0,
    )  # fmt: skip
    s = wakis.SolverFIT3D(grid, wake, bc_low=["pec", "pec", "cpml"],
                          bc_high=["pec", "pec", "cpml"], n_pml=npml, bg="pec",
                          use_stl=True, verbose=0, scheme=scheme)  # fmt: skip
    wakis.sources.Beam.tfsf_template = template
    try:
        s.wakesolve(wakelength=0.03, compute_plane="longitudinal")
    finally:
        wakis.sources.Beam.tfsf_template = None
    return np.abs(wake.WP).max(), s


def test_conformal_round_pipe_no_cherenkov(tmp_path, long_tfsf, monkeypatch):
    """
    Conformal round pipe at the magic step: the trapezoidal x/y face areas,
    no x/y face merging and the e_z mass scaling keep the bunch self-field an
    exact discrete solution (|W| ~ 1e-10 V/pC; Yee: ~0.5 V/pC). Merging the
    small x/y faces as in the Yee path (threshold 0.5) couples (x, y) columns
    and gives a visible wake.
    """
    tete, s = _round_pipe_wake("tete", tmp_path)
    assert s.tete_dt_limits["longitudinal"] >= s.dt * (1 - 1e-6)
    assert tete < 1e-8
    yee, _ = _round_pipe_wake("yee", tmp_path)
    assert yee > 0.1
    monkeypatch.setattr(wakis.conformal, "TETE_XY_THRESHOLD", 0.5)
    merged, _ = _round_pipe_wake("tete", tmp_path)
    assert merged > 1e-5
