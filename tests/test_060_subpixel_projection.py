"""
Subpixel-smoothed projection (SSP, Hammond et al., Opt. Express 33, 33620
(2025)): NumPy/JAX implementations, exact VJPs, MaterialGrid integration
and the adjoint chain. All runs are small (< 1 s each).
"""

import numpy as np
import pytest

import wakis
from wakis.inverse import AdjointSimulation, MaterialGrid
from wakis.inverse import projection as ssp
from wakis.sources import CurrentSource, GaussianSource

from test_020_adjoint import F0, _make_2d_solver, _objective


# ---------------------------------------------------------------- the formula
def test_fill_fraction_is_the_parabolic_kernel_fill():
    """
    F_2 (Eq. 10) is C^2, F(-x) = 1 - F(x), and equals the fill fraction of
    the half space beyond a plane at distance d inside a sphere with the
    parabolic kernel k(r) ~ 1 - (r/R)^2 (Eq. 9, by direct quadrature), as
    the paper states. Mutation: the top-hat kernel k = 1 gives a different
    F (differs by > 0.01).
    """
    x = np.linspace(-1, 1, 2001)
    F = ssp.fill_fraction(x)
    np.testing.assert_allclose(F + ssp.fill_fraction(-x), 1.0, atol=1e-15)
    assert F[0] == 1.0 and F[-1] == 0.0 and np.all(np.diff(F) < 0)
    # dF/dx and d2F/dx2 vanish at |x| = 1 (C^2 joints with the constants)
    assert abs(ssp.fill_fraction_derivative(np.array([0.999999]))[0]) < 1e-10
    h = 1e-4
    d2 = (ssp.fill_fraction(1 - h) - 2 * ssp.fill_fraction(1 - 2 * h)
          + ssp.fill_fraction(1 - 3 * h)) / h**2  # fmt: skip
    # F'' = 15/4 x (1 - x^2) -> 0 linearly (1.5e-3 at x = 1 - 2e-4) vs O(1) inside
    assert abs(d2) < 2e-3
    mid = (ssp.fill_fraction(0.5 + h) - 2 * ssp.fill_fraction(0.5)
           + ssp.fill_fraction(0.5 - h)) / h**2  # fmt: skip
    assert abs(mid - 15 / 4 * 0.5 * 0.75) < 1e-4

    def kernel_fill(d, kernel):
        r = np.linspace(0, 1, 3001)
        theta = np.linspace(0, np.pi / 2, 1501)
        inner = []
        for t in theta:
            lo = d / max(np.cos(t), 1e-300)
            rr = r[r >= lo]
            inner.append(np.trapezoid(kernel(rr) * rr**2, rr) if rr.size > 1 else 0.0)
        total = 2 * np.trapezoid(kernel(r) * r**2, r)  # both hemispheres
        return np.trapezoid(np.array(inner) * np.sin(theta), theta) / total

    for d in (0.1, 0.4, 0.7):
        parabolic = kernel_fill(d, lambda r: 1 - r**2)
        assert abs(parabolic - ssp.fill_fraction(d)) < 2e-4, d
        top_hat = kernel_fill(d, lambda r: np.ones_like(r))
        assert abs(top_hat - ssp.fill_fraction(d)) > 0.01


def test_limits_and_threshold():
    """
    beta -> 0+: SSP = rho_f (no projection); beta = inf: SSP = F(d) within R
    of the level set and exactly 0/1 elsewhere; for every beta, SSP > 1/2
    iff rho_f > 1/2 (eta = 1/2), so thresholding and length-scale tools see
    the same binary design as with the tanh projection.
    """
    rng = np.random.default_rng(3)
    rho_f = (
        0.5
        + 0.4
        * np.sin(np.linspace(0, 9, 60))[:, None]
        * np.cos(np.linspace(0, 7, 50))[None, :]
    )
    rho_f += 1e-3 * rng.normal(size=rho_f.shape)
    np.testing.assert_allclose(ssp.smoothed_projection(rho_f, 1e-7), rho_f, atol=1e-7)
    out = ssp.smoothed_projection(rho_f, np.inf)
    g = np.hypot(*np.gradient(rho_f))
    d = (0.5 - rho_f) / g
    band = np.abs(d) < 0.55
    np.testing.assert_allclose(out[band], ssp.fill_fraction(d[band] / 0.55), atol=1e-15)
    assert np.all(np.isin(out[~band], (0.0, 1.0))) and band.any()
    for beta in (2.0, 16.0, 128.0, np.inf):
        o = ssp.smoothed_projection(rho_f, beta)
        assert np.array_equal(o > 0.5, rho_f > 0.5)
    # SSP(beta) -> SSP(inf) as beta grows, in the mean but not uniformly:
    # this field has saddle points on the level set (|grad| -> 0, a
    # topology change), where SSP falls back to the tanh projection
    errs = [np.abs(ssp.smoothed_projection(rho_f, b) - out) for b in (16, 64, 256)]
    means = [e.mean() for e in errs]
    assert means[0] > means[1] > means[2] and means[2] < 2e-3
    assert errs[2].max() > 0.3


@pytest.mark.parametrize("beta", [4.0, 32.0, np.inf])
@pytest.mark.parametrize("periodic", [False, (True, False)])
def test_vjp_matches_jax_and_finite_differences(beta, periodic):
    """
    The hand-derived NumPy VJP equals jax.vjp of the independent JAX
    implementation to rounding, and both match central finite differences.
    Mutation: a wrong dF/dx (15/17 instead of 15/16) breaks the FD match.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from wakis.inverse import conic_filter, smoothed_projection

    rng = np.random.default_rng(0)
    rho_f = np.asarray(
        conic_filter(jax.numpy.asarray(rng.uniform(0, 1, (24, 20))), 4.0)
    )
    rho_f = (rho_f - rho_f.min()) / np.ptp(rho_f)
    spacing = (1.0, 1.5)
    kw = dict(spacing=spacing, periodic=periodic)
    np.testing.assert_allclose(
        ssp.smoothed_projection(rho_f, beta, **kw),
        smoothed_projection(jax.numpy.asarray(rho_f), beta, **kw),
        atol=1e-15,
    )
    c = rng.normal(size=rho_f.shape)
    v_np = ssp.smoothed_projection_vjp(rho_f, c, beta, **kw)
    _, pullback = jax.vjp(lambda r: smoothed_projection(r, beta, **kw), rho_f)
    v_jax = np.asarray(pullback(c)[0])
    np.testing.assert_allclose(v_np, v_jax, atol=1e-13 * np.abs(v_jax).max())
    assert np.abs(v_np).max() > 0

    def directional_fd(v):
        h = 1e-6
        plus = np.sum(c * ssp.smoothed_projection(rho_f + h * v, beta, **kw))
        minus = np.sum(c * ssp.smoothed_projection(rho_f - h * v, beta, **kw))
        return (plus - minus) / (2 * h)

    v = rng.normal(size=rho_f.shape)
    fd = directional_fd(v)
    assert abs(np.sum(v_np * v) - fd) < 1e-6 * abs(fd)

    good = ssp.fill_fraction_derivative
    try:
        ssp.fill_fraction_derivative = lambda x: good(x) * 16 / 17
        bad = np.sum(ssp.smoothed_projection_vjp(rho_f, c, beta, **kw) * v)
    finally:
        ssp.fill_fraction_derivative = good
    assert abs(bad - fd) > 1e-4 * abs(fd)  # 100x the tolerance above


def test_convergence_to_binary_design():
    """
    A disk level set (filtered-density width fixed in physical units) on
    grids of 32..512 pixels: at beta = inf the gray-pixel fraction and the
    L1 distance to the binary indicator fall as O(h) (fitted order 1 +- 0.1),
    i.e. SSP converges to the binary design. Baseline: tanh projection at
    beta = 32 keeps ~6 % of the pixels in (1e-3, 1 - 1e-3) at every resolution.
    """
    gray_inf, l1_inf, gray_32 = [], [], []
    Ns = np.array([32, 64, 128, 256, 512])
    for N in Ns:
        h = 1 / N
        x = (np.arange(N) + 0.5) * h - 0.5
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho_f = 0.5 + (0.3 - np.hypot(X - 0.013, Y + 0.021)) / 0.15
        binary = rho_f > 0.5
        o = ssp.smoothed_projection(rho_f, np.inf, spacing=h)
        gray_inf.append(np.mean((o > 0) & (o < 1)))
        l1_inf.append(np.abs(o - binary).sum() * h * h)
        o32 = ssp.tanh_projection(rho_f, 32.0)
        gray_32.append(np.mean((o32 > 1e-3) & (o32 < 1 - 1e-3)))
    order_gray = -np.polyfit(np.log(Ns), np.log(gray_inf), 1)[0]
    order_l1 = -np.polyfit(np.log(Ns), np.log(l1_inf), 1)[0]
    assert abs(order_gray - 1) < 0.1 and abs(order_l1 - 1) < 0.1, (order_gray, order_l1)
    assert gray_inf[-1] < 0.005
    assert min(gray_32) > 0.05 and gray_32[-1] > 0.9 * gray_32[0]  # no convergence


# ------------------------------------------------------ MaterialGrid + adjoint
def _sim(beta, radius=None, shape=(17, 17, 1), Nt=400):
    source = CurrentSource(GaussianSource(F0, 1e9), component="z", center=(-0.06, 0, 0))
    monitors = {
        "probe": wakis.DFTMonitor(
            [F0, 1.1 * F0], components=["Ez", "Hy"], center=(0.07, 0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.06, 0.06, 0.0), shape=shape, eps_min=1.0,
        eps_max=4.0, projection_beta=beta, projection_radius=radius,
    )  # fmt: skip
    return AdjointSimulation(lambda: _make_2d_solver(), [source], monitors, design, Nt)


_X = np.linspace(-0.03, 0.03, 17)
_XX, _YY = np.meshgrid(_X, _X, indexing="ij")


def _disk(r0=0.017, w=0.02):
    return np.clip(0.5 + (r0 - np.hypot(_XX - 0.004, _YY + 0.002)) / w, 0, 1)[..., None]


@pytest.mark.parametrize("beta", [8.0, np.inf])
def test_material_grid_ssp_adjoint_gradient(beta):
    """
    MaterialGrid(projection_beta=...) evaluates SSP at every E edge (from the
    interpolated filtered density and gradient, R = 0.55 mesh steps): the
    exact-adjoint gradient matches central finite differences along random
    directions. The design is binary except a one-cell layer at beta = inf.
    Mutation: dropping the |grad rho| path of the VJP breaks the match.
    """
    sim = _sim(beta)
    rho = _disk()
    F, G = _objective(sim.forward(rho))
    grad = sim.vjp(G)
    f = sim.design._fill(rho)
    assert sim.design.projection_radius == pytest.approx(0.55 * 0.3 / 40)
    if np.isinf(beta):
        assert 0.05 < np.mean((f > 0) & (f < 1)) < 0.35
        assert np.all(np.isin(f[(f <= 0) | (f >= 1)], (0.0, 1.0)))
    rng = np.random.default_rng(0)
    dirs = [rng.normal(size=rho.shape) for _ in range(2)]
    fds = []
    for v in dirs:
        h = 1e-5
        Fp, _ = _objective(sim.forward(rho + h * v))
        Fm, _ = _objective(sim.forward(rho - h * v))
        fds.append((Fp - Fm) / (2 * h))
        assert abs(np.sum(grad * v) - fds[-1]) < 1e-5 * abs(fds[-1])

    # Mutation: no contribution through |grad rho| (d_g = 0)
    from wakis.inverse import design as design_module

    good = ssp.smoothed_projection_pointwise

    def no_gradient_path(*args):
        out, d_rho, d_g = good(*args)
        return out, d_rho, 0 * d_g

    sim.forward(rho)
    try:
        ssp.smoothed_projection_pointwise = no_gradient_path
        bad = sim.vjp(G)
    finally:
        ssp.smoothed_projection_pointwise = good
    assert design_module is not None
    assert abs(np.sum(bad * dirs[0]) - fds[0]) > 1e-3 * abs(fds[0])


def test_jax_chain_equals_numpy_adjoint_chain():
    """
    latent -> conic filter -> SSP(beta = inf) -> Wakis (jax_simulation):
    jax.grad equals the NumPy chain sim.vjp -> smoothed_projection_vjp ->
    filter pullback, and central finite differences of the whole chain.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    from wakis.inverse import conic_filter, jax_simulation, smoothed_projection

    sim = _sim(None, Nt=300)  # MaterialGrid nodes take the projected density
    simulate = jax_simulation(sim)
    spacing = 0.06 / 16

    def objective(out):
        Ez = out["probe"]["Ez"]
        return 1e20 * jnp.sum(jnp.abs(Ez[0]) ** 2)

    latent = jnp.asarray(_disk(0.014, 0.008) + 0.05 * np.random.default_rng(1).normal(
        size=(17, 17, 1)))  # fmt: skip

    def filtered(lat):
        return conic_filter(lat, 2.0)

    def chain(lat):
        y = filtered(lat)[..., 0]
        return objective(
            simulate(smoothed_projection(y, np.inf, spacing=spacing)[..., None])
        )

    value, grad = jax.value_and_grad(chain)(latent)

    y, pull_filter = jax.vjp(filtered, latent)
    y = np.asarray(y)[..., 0]
    rho = ssp.smoothed_projection(y, np.inf, spacing=spacing)
    assert 0 < np.mean((rho > 0) & (rho < 1)) < 0.5
    out = sim.forward(rho[..., None])
    Ez = out["probe"]["Ez"]
    G_Ez = np.zeros_like(Ez)
    G_Ez[0] = 2e20 * Ez[0]
    g_rho = sim.vjp({"probe": {"Ez": G_Ez}})[..., 0]
    g_y = ssp.smoothed_projection_vjp(y, g_rho, np.inf, spacing=spacing)
    expected = np.asarray(pull_filter(jnp.asarray(g_y[..., None]))[0])
    np.testing.assert_allclose(
        np.asarray(grad), expected, atol=1e-10 * np.abs(expected).max()
    )

    v = np.random.default_rng(2).normal(size=latent.shape)
    h = 1e-6
    fd = (chain(latent + h * v) - chain(latent - h * v)) / (2 * h)
    assert abs(float(fd) - np.sum(expected * v)) < 1e-5 * abs(float(fd))


def test_objective_moves_smoothly_with_the_interface():
    """
    A straight level set x = x0 (rho_f = 1/2 + (x0 - x)/w) moved by a
    fraction of a cell at beta = inf: with SSP the objective changes and the
    adjoint dF/dx0 matches the finite difference of F(x0); with the step
    projection (R -> 0) F does not change at all within the cell and the
    gradient is exactly zero.
    """
    w = 0.1

    def slab(x0):
        return np.broadcast_to((0.5 + (x0 - _XX) / w)[..., None], (17, 17, 1)).copy()

    def F_and_slope(sim, x0):
        F, G = _objective(sim.forward(slab(x0)))
        return F, np.sum(sim.vjp(G)) / w  # d rho / d x0 = 1/w at every node

    dx = 0.3 / 40
    x0, h = 0.31 * dx, 1e-3 * dx
    sim = _sim(np.inf)
    F0_, slope = F_and_slope(sim, x0)
    Fp, _ = F_and_slope(sim, x0 + h)
    Fm, _ = F_and_slope(sim, x0 - h)
    fd = (Fp - Fm) / (2 * h)
    assert abs(Fp - Fm) > 1e-4 * abs(F0_)
    assert abs(slope - fd) < 1e-4 * abs(fd)
    step = _sim(np.inf, radius=1e-12)
    Fs0, slope_s = F_and_slope(step, 0.12 * dx)  # no edge sample in (0.12, 0.27) dx
    Fs1, _ = F_and_slope(step, 0.27 * dx)
    assert slope_s == 0.0 and Fs1 == Fs0


@pytest.mark.parametrize("beta", [16.0, np.inf])
def test_ssp_with_tensor_smoothing_gradient(beta):
    """
    SSP fill fractions combined with tensor subpixel averaging
    (MaterialGrid(smoothing='tensor', projection_beta=...)) in a 2D TE run
    (in-plane E, off-diagonal couplings active): adjoint = central finite
    differences along random directions.
    """
    from test_035_tensor_averaging import DES_F0, _design_objective, _design_solver

    source = CurrentSource(GaussianSource(DES_F0, 1e9), component="x",
                           center=(-0.05, 0.013, 0.0))  # fmt: skip
    monitors = {
        "probe": wakis.DFTMonitor(
            [DES_F0, 1.2 * DES_F0], components=["Ex", "Ey"], center=(0.06, -0.01, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0, 0, 0), size=(0.08, 0.08, 0.0), shape=(9, 9, 1), eps_min=1.0,
        eps_max=4.0, smoothing="tensor", projection_beta=beta,
    )  # fmt: skip
    sim = AdjointSimulation(
        lambda: _design_solver("pec"), [source], monitors, design, Nt=350
    )
    u = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(u, u, indexing="ij")
    rho = np.clip(0.5 + (0.6 - np.hypot(X - 0.1, Y + 0.05)) / 0.8, 0, 1)[:, :, None]
    F, G = _design_objective(sim.forward(rho))
    assert sim._solver.tensor_operator.rows.size > 0
    grad = sim.vjp(G)
    rng = np.random.default_rng(5)
    for _ in range(2):
        v = rng.normal(size=rho.shape)
        h = 1e-6
        Fp, _ = _design_objective(sim.forward(rho + h * v))
        Fm, _ = _design_objective(sim.forward(rho - h * v))
        fd = (Fp - Fm) / (2 * h)
        assert abs(np.sum(grad * v) - fd) < 1e-5 * abs(fd), (np.sum(grad * v), fd)
