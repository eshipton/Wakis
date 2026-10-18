"""
Subpixel (Kottke) averaging at interfaces with dispersive materials
(`wakis.tensor_dispersive`): conformal 'all'/'tensor' STL solids with a
`DispersiveMaterial` and `MaterialGrid(material=..., smoothing='tensor')`.

The resonances are measured in the time domain (ring-down, harmonic
inversion). For lossless Drude and Lorentz media the leapfrog error is
removed exactly by sin(w dt / 2) = w_s dt / 2, since the discrete scheme
equals the continuous one at the frequency 2 sin(w dt / 2) / dt.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource
from wakis.unstructured import analytic


class _Probe:
    """Time series of E_c at the sample nearest to `point`."""

    def __init__(self, solver, point, comp):
        coords = [solver.grid.x[:-1], solver.grid.y[:-1], solver.grid.z[:-1]]
        a = "xyz".index(comp)
        coords[a] = coords[a] + [solver.grid.dx, solver.grid.dy, solver.grid.dz][a] / 2
        self.index = (
            *(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)),
            comp,
        )
        self.values = []

    def update(self, solver, t):
        self.values.append(float(solver.E[self.index]))


def _resonance(solver, f0, points, comp, periods, band=0.1):
    """
    Dominant resonance near f0 of the weighted probe sum after a pulse
    driven by the same weighted sources (a symmetry filter), with the
    leapfrog error removed.
    """
    pulse = GaussianSource(frequency=f0, fwidth=0.3 * f0)
    srcs = [CurrentSource(pulse, component=comp, center=p, amplitude=w)
            for p, w in points]  # fmt: skip
    probes = [_Probe(solver, p, comp) for p, _ in points]
    Nt = int((pulse.end_time + periods / f0) / solver.dt)
    solver.emsolve(Nt, source=srcs, monitors=probes)
    start = int(np.ceil(pulse.end_time / solver.dt))
    sig = sum(w * np.array(p.values[start:]) for p, (_, w) in zip(probes, points))
    step = max(1, int(1 / (8 * f0 * solver.dt)))
    res = harminv(sig[::step], step * solver.dt, (1 - band) * f0, (1 + band) * f0,
                  t0=(start + 1) * solver.dt)  # fmt: skip
    i = np.argmax(np.abs(res.amplitude))
    f = 2 / solver.dt * np.sin(np.pi * res.frequency * solver.dt) / (2 * np.pi)
    return f[i], f, np.abs(res.amplitude) / np.abs(res.amplitude[i])


# ------------------------------------------- Drude rod in a PEC cavity (2D)
# Lossless Drude rod (radius b = a/2, k_p a = 4) concentric in a circular PEC
# cavity (radius a), TE: the n = 1 surface plasmon at k a = 3.64691
# (eps = -0.20), exact Bessel/modified-Bessel matching.
CAV_A, ROD_B, KP = 1.0, 0.5, 4.0
CAV_CENTER = np.array([0.0031, -0.0017])


def _drude_rod_solver(n, conformal, use_gpu=False):
    h = 2.2 * CAV_A / n
    lo, hi = CAV_CENTER - 1.1 * CAV_A, CAV_CENTER + 1.1 * CAV_A

    def cyl(r):
        return pv.Cylinder(center=(*CAV_CENTER, 0), direction=(0, 0, 1), radius=r,
                           height=20 * h, resolution=2880).triangulate()  # fmt: skip

    drude = DispersiveMaterial(1.0, [Drude(KP * c, 0.0)])
    grid = wakis.GridFIT3D(
        lo[0], hi[0], lo[1], hi[1], -h / 2, h / 2, n, n, 1,
        stl_solids={"cavity": cyl(CAV_A), "rod": cyl(ROD_B)},
        stl_materials={"cavity": "vacuum", "rod": drude},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        bg="pec", use_stl=True, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


def _plasmon_error(n, conformal, use_gpu=False, periods=100):
    k_sp = analytic.drude_rod_in_cavity_te(CAV_A, ROD_B, KP, n=1)[1]
    f_sp = k_sp * c / (2 * np.pi)
    solver = _drude_rod_solver(n, conformal, use_gpu)
    r = 0.62  # x-dipole filter: E_x at (+-r, 0) (weight 1) and (0, +-r) (-1/2)
    pts = [((*(CAV_CENTER + d), 0.0), w) for d, w in
           (([r, 0], 1.0), ([-r, 0], 1.0), ([0, r], -0.5), ([0, -r], -0.5))]  # fmt: skip
    f, modes, amps = _resonance(solver, f_sp, pts, "x", periods, band=0.15)
    return f / f_sp - 1, modes / f_sp - 1, amps


def test_drude_rod_plasmon(use_gpu):
    """
    Surface plasmon (E crossing the metal surface) of a lossless Drude rod at
    48 and 96 cells across 2.2 a (~11 and ~22 cells per vacuum wavelength).
    With the dispersive Kottke average the spectrum near f_sp holds one
    dominant mode, within -1.0e-2 and -3.9e-3 of the exact one (observed;
    first order: the full tensor is not available for dispersive pixels).
    The previous behaviour (poles on the staircased cell mask, eps_inf
    averaged), emulated by conformal='pec', gives -13.8 % with several
    spurious modes of comparable amplitude at 48 cells.
    """
    e48, _, amps48 = _plasmon_error(48, "all", use_gpu)
    e96, _, _ = _plasmon_error(96, "all", use_gpu)
    assert abs(e48) < 1.5e-2
    assert abs(e96) < 5e-3
    assert abs(e96) < 0.6 * abs(e48)
    # one dominant mode: the n = 2 mode (+3.3 %) is the strongest other one
    assert np.sort(amps48)[-2] < 0.5
    stair, _, amps = _plasmon_error(48, "pec", use_gpu)
    assert abs(stair) > 5e-2
    assert np.sort(amps)[-2] > 0.5


# --------------------------------- Lorentz dielectric rod (test_035 geometry)
# eps(w) = 1 + d w0^2 / (w0^2 - w^2) with w0 = 3 w_ref and eps(w_ref) = 6: the
# dispersive problem has the static eps = 6 problem's TE n = 0 mode as its
# fixed point, k R = 1.9746767234 (Delaunay FIT reference, test_035).
LOR_R, LOR_KR = 0.1, 1.9746767234


def _lorentz_rod_error(n, conformal, use_gpu=False):
    f0 = LOR_KR / LOR_R * c / (2 * np.pi)
    w = 2 * np.pi * f0
    mat = DispersiveMaterial(1.0, [Lorentz(5 * (1 - 1 / 9), 3 * w, 0.0)])
    h = 2 * LOR_R / n
    rod = pv.Cylinder(center=(0.0031, -0.0017, 0), direction=(0, 0, 1),
                      radius=LOR_R / 2, height=20 * h, resolution=2880).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -LOR_R, LOR_R, -LOR_R, LOR_R, -h / 2, h / 2, n, n, 1,
        stl_solids={"rod": rod}, stl_materials={"rod": mat},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        use_stl=True, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    src = CurrentSource(pulse, component="x", center=(0.013, 0.071, 0.0))
    probe = _Probe(solver, (-0.067, 0.021, 0.0), "y")
    Nt = int((pulse.end_time + 60 / f0) / solver.dt)
    solver.emsolve(Nt, source=[src], monitors=[probe])
    start = int(np.ceil(pulse.end_time / solver.dt))
    step = max(1, int(1 / (8 * f0 * solver.dt)))
    sig = np.array(probe.values[start:])[::step]
    res = harminv(
        sig, step * solver.dt, 0.95 * f0, 1.05 * f0, t0=(start + 1) * solver.dt
    )
    f = res.frequency[np.argmax(np.abs(res.amplitude))]
    return 2 / solver.dt * np.sin(np.pi * f * solver.dt) / (2 * np.pi) / f0 - 1


def test_lorentz_rod_matches_static_average(use_gpu):
    """
    A Lorentz rod with eps(w_ref) = 6 at 32 cells across the box (21 cells
    per wavelength in the rod) reproduces the accuracy of the static
    diagonal Kottke average (static eps = 6: +6.6e-3; observed +6.1e-3, the
    difference is the group-index factor of the dispersion), against -4.2 %
    for the staircased poles (conformal='pec').
    """
    err = _lorentz_rod_error(32, "all", use_gpu)
    assert 4e-3 < err < 8e-3
    assert abs(_lorentz_rod_error(32, "pec", use_gpu)) > 3e-2


# ------------------------------------------- planar Debye slab (tangential)
def _slab_reflectance(conformal, z0, thickness, dz, use_gpu=False):
    """Reflectance of a conductive Debye slab (STL box between z0 and
    z0 + thickness, faces between the nodes) at normal incidence."""
    f = np.linspace(0.6e9, 1.4e9, 9)
    mat = DispersiveMaterial(3.0, [Debye(6.0, 1 / (2 * np.pi * 1e9))], sigma=0.05)

    def below(with_slab):
        solids, mats = {}, {}
        if with_slab:
            solids = {"slab": pv.Box(bounds=(-1, 1, -1, 1, z0, z0 + thickness))}
            mats = {"slab": mat}
        grid = wakis.GridFIT3D(
            -0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, int(round(1.2 / dz)),
            stl_solids=solids or None, stl_materials=mats or None,
            conformal=conformal if with_slab else False, verbose=0,
        )  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"], n_pml=15,
            use_stl=with_slab, use_gpu=use_gpu, verbose=0,
        )  # fmt: skip
        pulse = GaussianSource(frequency=1e9, fwidth=0.4e9)
        sheet = CurrentSource(pulse, component="x", center=(0, 0, -0.3),
                              size=(0.02, 0.02, 0.0))  # fmt: skip
        plane = wakis.FluxMonitor(f, center=(0, 0, -0.15), size=(0.02, 0.02, 0.0))
        solver.emsolve(int((pulse.end_time + 20e-9) / solver.dt), source=sheet,
                       monitors=[plane])  # fmt: skip
        return plane.get_flux()

    R = 1 - below(True) / below(False)
    n = np.sqrt(mat.epsilon(f).astype(complex))
    n = np.where(n.imag > 0, -n, n)
    r12 = (1 - n) / (1 + n)
    phase = np.exp(-2j * 2 * np.pi * f / c * n * thickness)
    r = r12 * (1 - phase) / (1 - r12**2 * phase)
    return np.max(np.abs(R - np.abs(r) ** 2))


def test_debye_slab_between_nodes(use_gpu):
    """
    E is tangential to a planar slab, so the parallel (arithmetic) branch of
    the network is exact for the partially filled edges: a conductive Debye
    slab whose faces lie 0.3 cells off the nodes (dz = 5 mm, >= 20
    cells per wavelength in the slab) matches the analytic Fabry-Perot
    reflectance to < 6e-3 (observed 2.9e-3), against 1.6e-2 for the
    staircased slab.
    """
    dz = 5e-3
    err = _slab_reflectance("all", 0.3 * dz, 0.1 + 0.3 * dz, dz, use_gpu)
    stair = _slab_reflectance(False, 0.3 * dz, 0.1 + 0.3 * dz, dz, use_gpu)
    assert err < 6e-3
    assert stair > 4 * err


# ------------------------------------------------------- bulk equivalence
def _filled_solver(conformal, bc="pec", use_gpu=False):
    """2D box completely filled with a multi-pole dispersive solid."""
    w = 2 * np.pi * 2e9
    mat = DispersiveMaterial(
        1.5, [Drude(1.2 * w, 0.05 * w), Lorentz(0.8, 1.5 * w, 0.02 * w),
              Debye(0.6, 1 / w)], sigma=0.02,
    )  # fmt: skip
    L, n = 0.2, 16
    h = L / n
    box = pv.Box(bounds=(-L, L, -L, L, -L, L)).triangulate()
    grid = wakis.GridFIT3D(
        -L / 2, L / 2, -L / 2, L / 2, -h / 2, h / 2, n, n, 1,
        stl_solids={"fill": box}, stl_materials={"fill": mat},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc, bc, "periodic"], bc_high=[bc, bc, "periodic"],
        use_stl=True, n_pml=4, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


def test_bulk_equals_ade(use_gpu):
    """
    Inside a material (fill fraction 1, no interface) the sub-capacitor
    network is one tangential capacitor per edge with the material's own
    model, i.e. the ADE update of `wakis.dispersive`: a box filled with a
    Drude + Lorentz + Debye + conductive medium gives the same fields with
    conformal='all' (network) and the staircase (ADE groups) to rounding.
    """
    out = []
    for conformal in ("all", False):
        solver = _filled_solver(conformal, use_gpu=use_gpu)
        assert (getattr(solver, "dispersive_network", None) is not None) == bool(
            conformal
        )
        pulse = GaussianSource(frequency=2e9, fwidth=1e9)
        src = CurrentSource(pulse, component="x", center=(0.013, -0.021, 0.0))
        solver.emsolve(400, source=[src])
        out.append(solver.E.toarray())
    out = [np.asarray(o.get() if hasattr(o, "get") else o) for o in out]
    np.testing.assert_allclose(
        out[0], out[1], rtol=0, atol=1e-10 * np.abs(out[1]).max()
    )


# ------------------------------------------------- stability and passivity
def _mixed_rod_solver(n, mat, conformal="all"):
    R = 0.1
    h = 2 * R / n
    rod = pv.Cylinder(center=(0.0031, -0.0017, 0), direction=(0, 0, 1), radius=R / 2,
                      height=20 * h, resolution=1440).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -R, R, -R, R, -h / 2, h / 2, n, n, 1, stl_solids={"rod": rod},
        stl_materials={"rod": mat}, conformal=conformal, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        use_stl=True, verbose=0,
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["drude", "lorentz", "drude_lorentz"])
def test_lossless_long_run(kind):
    """
    Lossless metals and dielectrics at the interface cells (negative
    eps(w) included) stay bounded for 50 000 steps after a pulse: the
    stored energy is a sum of positive quadratic forms (one per
    sub-capacitor and pole) and the time step satisfies the a-priori bound.
    The edge-pixel full tensor (studies/tensor-dispersive/coupled.py) grows
    by >1e10 in 20 000 steps in the same Drude setup.
    """
    w = 2 * np.pi * 1.9746767234 / 0.1 * c / (2 * np.pi)
    poles = {
        "drude": [Drude(2 * w, 0.0)],
        "lorentz": [Lorentz(4.95, 3 * w, 0.0)],
        "drude_lorentz": [Drude(2 * w, 0.0), Lorentz(1.0, 1.5 * w, 0.0)],
    }[kind]
    solver = _mixed_rod_solver(24, DispersiveMaterial(1.0, poles))
    assert solver.dispersive_network.n_caps > 0
    pulse = GaussianSource(frequency=w / (2 * np.pi), fwidth=0.5 * w / (2 * np.pi))
    src = CurrentSource(pulse, component="x", center=(0.013, 0.071, 0.0))
    solver.emsolve(int(pulse.end_time / solver.dt) + 1, source=[src])
    norms = []
    for _ in range(10):
        solver.emsolve(5000)
        norms.append(np.linalg.norm(solver.E.toarray()))
    norms = np.array(norms)
    assert np.all(np.isfinite(norms))
    assert norms.max() < 3 * norms.min()


def test_lossy_media_decay():
    """
    With Drude, Lorentz and Debye losses and a conductivity the field decays
    (passivity): the E norm drops by more than 10x within 30 000 steps and
    never grows above its value after the pulse.
    """
    w = 2 * np.pi * 1.9746767234 / 0.1 * c / (2 * np.pi)
    mat = DispersiveMaterial(
        1.5, [Drude(2 * w, 0.2 * w), Lorentz(1.0, 1.5 * w, 0.2 * w), Debye(2.0, 1 / w)],
        sigma=0.5,
    )  # fmt: skip
    solver = _mixed_rod_solver(24, mat)
    pulse = GaussianSource(frequency=w / (2 * np.pi), fwidth=0.5 * w / (2 * np.pi))
    src = CurrentSource(pulse, component="x", center=(0.013, 0.071, 0.0))
    solver.emsolve(int(pulse.end_time / solver.dt) + 1, source=[src])
    n0 = np.linalg.norm(solver.E.toarray())
    norms = []
    for _ in range(6):
        solver.emsolve(5000)
        norms.append(np.linalg.norm(solver.E.toarray()))
    assert max(norms) <= n0
    assert norms[-1] < 0.1 * n0


def test_stability_bound():
    """
    The a-priori bound r^2 + (dt W / 2)^2 / (1 - (dt w0 / 2)^2) <= 1 equals the
    exact von Neumann limit of a single lossless Drude pole in bulk
    (DispersiveMaterial.is_stable), and a time step above it is rejected.
    """
    w = 2 * np.pi * 1.9746767234 / 0.1 * c / (2 * np.pi)
    mat = DispersiveMaterial(1.0, [Drude(2 * w, 0.0)])
    solver = _mixed_rod_solver(16, mat)
    net = solver.dispersive_network
    dt_static = solver.conformal_dt_max
    lhs = net.stability_bound(dt_static)
    r2 = (solver.dt / dt_static) ** 2
    assert lhs == pytest.approx(r2 + (solver.dt * 2 * w / 2) ** 2)
    # the bound at r = C (vacuum Courant number of a uniform grid) is the
    # bulk criterion (w_p dt / 2)^2 + C^2 <= eps_inf
    for scale in (0.95, 1.05):
        dt = scale * 2 / (2 * w)  # with r = 0: (w_p dt / 2)^2 = scale^2
        assert mat.is_stable(dt, courant=0.0) == (scale < 1)
    with pytest.raises(ValueError):
        _ = wakis.SolverFIT3D(
            solver.grid, bc_low=["pec", "pec", "periodic"],
            bc_high=["pec", "pec", "periodic"], use_stl=True, verbose=0,
            dt=1.2 * dt_static,
        )  # fmt: skip


# ------------------------------------------------------------------ adjoint
@pytest.mark.parametrize("conformal", ["all", "tensor"])
@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_adjoint_dot_product(conformal, bc):
    """
    The network correction is linear in (E*, E^n, u, p) and `AdjointStepper`
    applies its exact transpose: <U x, y> = <x, U^T y> to rounding, for a
    rod with Drude, Lorentz, Debye poles and conductivity (eps_inf = 1.5, so
    that 'tensor' also has static couplings to the dielectric neighbours).
    Perturbing one transposed coefficient breaks the identity.
    """
    w = 2 * np.pi * 2e9
    mat = DispersiveMaterial(
        1.5, [Drude(1.2 * w, 0.05 * w), Lorentz(0.5, 0.9 * w, 0.02 * w),
              Debye(0.4, 1 / w)], sigma=1e-3,
    )  # fmt: skip
    R = 0.1
    h = 2 * R / 16
    rod = pv.Cylinder(center=(0.0031, -0.0017, 0), direction=(0, 0, 1), radius=R / 2,
                      height=20 * h, resolution=720).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(-R, R, -R, R, -h / 2, h / 2, 16, 16, 1,
                           stl_solids={"rod": rod}, stl_materials={"rod": mat},
                           conformal=conformal, verbose=0)  # fmt: skip
    solver = wakis.SolverFIT3D(grid, bc_low=[bc, bc, "periodic"],
                               bc_high=[bc, bc, "periodic"], use_stl=True,
                               n_pml=6, verbose=0)  # fmt: skip
    solver.one_step()
    stepper = AdjointStepper(solver)
    rng = np.random.default_rng(1)
    keys = stepper.zero_state().keys()
    assert any(k.startswith("net_pD") for k in keys)

    def random_state():
        return {
            k: rng.standard_normal(v.shape) for k, v in stepper.zero_state().items()
        }

    def mismatch():
        x = random_state()
        stepper.set_state(x)
        solver.one_step()
        ux = stepper.get_state()
        y = random_state()
        uty = stepper.step(y)
        lhs = sum(np.dot(ux[k], y[k]) for k in keys)
        rhs = sum(np.dot(x[k], uty[k]) for k in keys)
        scale = sum(np.abs(ux[k]) @ np.abs(y[k]) for k in keys)
        return abs(lhs - rhs) / scale

    assert mismatch() < 1e-14
    # mutation: the transposed network with den perturbed by 1e-3
    net = solver.dispersive_network
    ext = next(e for e in stepper.dispersive if getattr(e, "net", None) is net)
    transpose = ext.transpose_post

    def perturbed(*args):
        den = net.den
        net.den = den * 1.001
        transpose(*args)
        net.den = den

    ext.transpose_post = perturbed
    assert mismatch() > 1e-8


DES_L, DES_N, DES_F0 = 0.2, 30, 2e9


def _design_solver(bc, use_gpu=False):
    h = DES_L / DES_N
    grid = wakis.GridFIT3D(
        -DES_L / 2, DES_L / 2, -DES_L / 2, DES_L / 2, -h / 2, h / 2,
        DES_N, DES_N, 1, verbose=0,
    )  # fmt: skip
    return wakis.SolverFIT3D(
        grid, bc_low=[bc, bc, "periodic"], bc_high=[bc, bc, "periodic"],
        n_pml=6, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip


@pytest.mark.parametrize("bc", ["pec", "cpml"])
def test_dispersive_tensor_design_gradient(bc, use_gpu):
    """
    MaterialGrid(material=..., smoothing='tensor') between a dielectric
    (eps 1.5) and a Drude + Lorentz + Debye + conductive medium on a 9 x 9
    node design (smoothed disk plus noise), 2D TE: the adjoint gradient
    through the fill fractions and the density-gradient normals of the
    sub-capacitor network (ieps, weights c, eps_inf, pole multipliers mu)
    matches central finite differences (h = 1e-5) to < 1e-6 relative
    (observed 2e-10 to 9e-9).
    """
    w = 2 * np.pi * DES_F0
    mat = DispersiveMaterial(
        2.0, [Drude(1.5 * w, 0.1 * w), Lorentz(1.0, 1.2 * w, 0.05 * w),
              Debye(0.5, 1 / w)], sigma=0.01,
    )  # fmt: skip
    source = CurrentSource(
        GaussianSource(DES_F0, 1e9), component="x", center=(-0.05, 0.013, 0.0)
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            [DES_F0, 1.2 * DES_F0], components=["Ex", "Ey"], center=(0.06, -0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(9, 9, 1),
        eps_min=1.5, material=mat, smoothing="tensor",
    )  # fmt: skip
    sim = AdjointSimulation(
        lambda: _design_solver(bc, use_gpu), [source], monitors, design, Nt=350
    )
    u = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(u, u, indexing="ij")
    rho = 0.5 + 0.5 * np.tanh(4 * (0.6 - np.hypot(X - 0.1, Y + 0.05)))
    rho = rho + 0.05 * np.random.default_rng(0).standard_normal(rho.shape)
    rho = np.clip(rho, 0.02, 0.98)[:, :, None]

    def objective(out):
        Ex, Ey = out["probe"]["Ex"], out["probe"]["Ey"]
        a = 1e20
        F = a * np.sum(np.abs(Ex) ** 2 + 0.5 * np.abs(Ey) ** 2)
        return F, {"probe": {"Ex": 2 * a * Ex, "Ey": a * Ey}}

    F, G = objective(sim.forward(rho))
    assert sim._solver.dispersive_network.n_caps == 2 * design.flat_idx.size
    grad = sim.vjp(G)
    h = 1e-5
    for k in [(4, 4, 0), (2, 5, 0), (6, 3, 0), (5, 7, 0)]:
        step = np.zeros_like(rho)
        step[k] = h
        Fp, _ = objective(sim.forward(rho + step))
        Fm, _ = objective(sim.forward(rho - step))
        fd = (Fp - Fm) / (2 * h)
        assert abs(grad[k] - fd) < 1e-6 * abs(fd), (k, grad[k], fd)
