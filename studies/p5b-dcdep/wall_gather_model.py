"""1D static model of the first-cell entrance error for the masked and image wall stencils.

Predicts, per sheet-lattice line and for the coarse (10 x 474) and fine (20 x 948) grids of
``examples/_space_charge_waves_common.py``, the energy excess a wall-born macroparticle carries
after its first cells, for two treatments of the order-p B-spline stencil at the ``z = 0`` PEC
emitter wall:

``masked``  (released default): exterior ghost nodes/edges are dropped; the birth closure current
            and the D17 birth kick restore charge and energy consistency.
``image``   (opt-in ``wall_stencil="image"``): the exterior part of the stencil is folded back
            with the PEC image symmetry (normal E even, tangential E odd, charge odd). A particle
            on the wall then has no visible charge and no effective potential, so the closure
            current and the birth kick vanish identically.

Physics (derived in ``first_principles.md``): the leapfrog energy gain of a particle in a static
field is the trapezoid rule of ``E_g`` on the step positions ``z^n``; it is exact on linear pieces
of the gathered field and makes the error ``(|q| dz / m c^2) * theta (1 - theta) / 2 * dzeta^2 * |J|``
when a step crosses a slope jump ``J`` of ``E_g`` at crossing phase ``theta``. The order-2
Galerkin gather of ``E_z`` is piecewise linear with knots at ``zeta = k + 1/2``. With masked ghosts,
``E_g = (1/2 + zeta) E0`` for ``zeta < 1/2`` and the knot at 1/2 jumps by ``E1 - 2 E0`` (first
order in the field). With the image ghost ``E_{-1} = E_0``, ``E_g = E0`` for ``zeta < 1/2`` and the
jump is ``E1 - E0`` (a first difference, O(dz) smaller). The interior knots jump by second
differences in both cases.

The static field is the discrete 2D (x, z) Poisson field of a z-uniform beam (the model used by
``tests/test_087_particles_vs_linearized_beam.py::_entrance_first_cell_delta_bound``), extended to
``K`` axial edges with the discrete axial decay ``lambda_m^k``. The particle moves along ``z`` only
(50 T guide field), with the library's birth sequence: kick (masked) -> back-push ``-dt/2`` ->
leapfrog. The excess is ``gamma(t_n) - gamma_0 + (q / m c^2) Phi_g(z^n)`` with the synchronised
``gamma(t_n)`` and the potential ``Phi_g`` whose gradient is the gathered field of each scheme.

Run (laptop, < 5 s, 1 core)::

    PYTHONPATH=. python studies/p5b-dcdep/wall_gather_model.py

Prints one ``WGM {json}`` line per grid, density case and scheme.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0, m_e
from scipy.linalg import solve_banded

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _space_charge_waves_common as s  # noqa: E402

N_EDGES = 128  # axial E_z edges kept in the static profile [-] (covers zeta_far = 64)


def deposit_node_profile(scheme: str, dzeta: float, n_nodes: int = 8) -> np.ndarray:
    """Steady-state near-wall node charge of a wall-born order-2 stream, per interior node charge.

    Every step one sheet of macroparticles is born at ``z = 0``. Its birth charge reaches the
    grid through the closure current (masked) or is zero (image), and the field at ``E^{n+1}``
    sees the sheets born at steps ``n, n - 1, ...`` at ``zeta_j = j * dzeta``, ``j >= 1``; the
    sheet born at ``n + 1`` is not yet part of ``E^{n+1}`` (its closure current and first
    Esirkepov move are in the step ``n+1 -> n+2`` current). The steady node charge in units of the
    line charge per cell ``(I / v) dz`` is therefore

    ``c_k = dzeta * sum_{j >= 1} [W2(j dzeta - k) - s W2(-j dzeta - k)]``,

    with ``s = 0`` (masked: the ghost node ``k = -1`` is dropped, nodes ``k >= 1`` are exact by
    Esirkepov charge conservation) or ``s = 1`` (image path subtracted). Euler-Maclaurin on the
    half line gives ``c_1 = 1 - 1/48 - dzeta/16 + dzeta^2/24 + O(dzeta^3)`` (masked) and
    ``c_1 = 1 - 2/48 + O(dzeta^3)`` (image: the ``dzeta`` terms of the real and image sums
    cancel), ``c_k = 1`` up to sampling aliasing for ``k >= 2``.

    Returns
    -------
    numpy.ndarray
        ``c_k`` for ``k = 0 .. n_nodes - 1`` [-] (``c_0`` is the wall node, which carries the PEC
        surface charge and does not enter the interior field).
    """

    def w2(d):
        d = np.abs(d)
        return np.where(d < 0.5, 0.75 - d * d, np.where(d < 1.5, 0.5 * (1.5 - d) ** 2, 0.0))

    j = np.arange(1, int(np.ceil((n_nodes + 2.0) / dzeta)) + 1, dtype=np.float64)
    zj = j * dzeta
    sgn = {"masked": 0.0, "image": 1.0}[scheme]
    k = np.arange(n_nodes, dtype=np.float64)[:, None]
    return dzeta * np.sum(w2(zj[None, :] - k) - sgn * w2(-zj[None, :] - k), axis=1)


def wall_field_profiles(
    nx: int,
    nz: int,
    params: dict,
    n_edges: int = N_EDGES,
    deposit: str = "uniform",
    scheme: str = "image",
    dzeta: float | None = None,
    density: str = "both",
    n_iter: int = 30,
):
    """Return the model's per-line wall E_z edge profiles for the density cases.

    Parameters
    ----------
    deposit : {"uniform", "discrete"}
        ``"uniform"``: z-uniform node charge (the original model). ``"discrete"``: the near-wall
        node charge of :func:`deposit_node_profile` for ``scheme`` and ``dzeta`` (nodes 1-7;
        aliasing beyond is < 1e-4 and dropped).
    density : {"both", "selfconsistent"}
        ``"both"``: the z-uniform injected and depressed density cases. ``"selfconsistent"``:
        the line density follows ``rho_l u0 / v_l(z_k)`` with ``v_l`` from energy conservation
        in the line-gathered node potential (fixed-point iteration), which tends to the depressed
        case far from the wall.

    Returns
    -------
    list of (str, numpy.ndarray, numpy.ndarray)
        ``(case, ez_l, phi_l)``: line-gathered ``E_z`` on edges ``k = 0 .. n_edges - 1`` (edge
        ``k`` at ``z = (k + 1/2) dz``) [V/m] and line-gathered node potential on nodes
        ``0 .. n_edges`` [V].
    dict
        Model details of ``compute_discrete_dc_depression``.
    """
    det = s.compute_discrete_dc_depression(
        nx, params, weighting="charge", return_details=True, shape=s.PARTICLE_SHAPE
    )
    a, b = s.TUNNEL_HALF_WIDTH_M, s.BEAM_HALF_THICKNESS_M
    dx, dz = 2.0 * a / nx, s.TUNNEL_LENGTH_M / nz
    n = nx - 1
    lap = (
        np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    ) / dx**2
    kap2, vecs = np.linalg.eigh(-lap)
    mu = kap2 * dz**2
    lam = (2.0 + mu - np.sqrt(mu * (mu + 4.0))) / 2.0
    w_mat = det["w_mat"]
    n_pad = n_edges + 64  # nodes 1..n_pad solved; psi(n_pad + 1) = 0 (lambda^64 < 1e-9)
    if deposit == "uniform":
        c_k = np.ones(n_pad + 1)
    elif deposit == "discrete":
        c_k = np.ones(n_pad + 1)
        c_k[1:8] = deposit_node_profile(scheme, float(dzeta), 8)[1:8]
    else:
        raise ValueError(deposit)
    gam0 = float(params["gamma0"])
    qm_mc2 = -e_charge / (m_e * c_light**2)

    def _solve(rho_lk):
        """Node potential [V] on x-nodes 0..nx, z-nodes 0..n_pad for line densities rho_lk."""
        src = np.einsum("lk,lx->xk", rho_lk, w_mat) * (2.0 * b / dx)  # (nx+1, n_pad+1)
        s_int = src[1:-1, 1:]  # interior x, z-nodes 1..n_pad
        s_inf = s_int[:, -1]
        phi_inf = np.linalg.solve(lap, -float(params["rho0"]) * s_inf / epsilon_0)
        kk = np.arange(n_pad + 1)
        phi_m = (vecs.T @ phi_inf)[:, None] * (1.0 - lam[:, None] ** kk[None, :])
        rhs_m = vecs.T @ (-float(params["rho0"]) * (s_int - s_inf[:, None]) / epsilon_0)
        for m in range(n):
            ab = np.zeros((3, n_pad))
            ab[0, 1:] = 1.0
            ab[2, :-1] = 1.0
            ab[1, :] = -(2.0 + mu[m])
            phi_m[m, 1:] += solve_banded((1, 1), ab, dz**2 * rhs_m[m])
        phi_nodes = np.zeros((nx + 1, n_pad + 1))
        phi_nodes[1:-1] = vecs @ phi_m
        return phi_nodes

    rho_l = det["rho_l"]
    cases = []
    if density == "both":
        cases = [
            ("injected", rho_l[:, None] * c_k[None, :]),
            ("depressed", (rho_l * params["u0"] / det["u_l"])[:, None] * c_k[None, :]),
        ]
    elif density == "selfconsistent":
        rho_lk = rho_l[:, None] * c_k[None, :]
        for _ in range(n_iter):
            phi_l_k = w_mat @ _solve(rho_lk)
            gam = gam0 + qm_mc2 * phi_l_k  # electrons: phi < 0 lowers gamma
            v = c_light * np.sqrt(1.0 - 1.0 / gam**2)
            rho_new = rho_l[:, None] * c_k[None, :] * float(params["u0"]) / v
            if np.max(np.abs(rho_new - rho_lk)) < 1e-15 * np.max(np.abs(rho_lk)):
                rho_lk = rho_new
                break
            rho_lk = rho_new
        cases = [("selfconsistent", rho_lk)]
    else:
        raise ValueError(density)
    out = []
    for case, rho_lk in cases:
        phi_nodes = _solve(rho_lk)[:, : n_edges + 1]
        ez = -(phi_nodes[:, 1:] - phi_nodes[:, :-1]) / dz
        out.append((case, w_mat @ ez, w_mat @ phi_nodes))
    return out, det


def _nodal_w2(d):
    """Quadratic nodal B-spline weights for offset ``d in [-1/2, 1/2)`` from the nearest node."""
    return 0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2


def _delta_shift(det: dict, params: dict, dgamma_l: np.ndarray) -> float:
    """``delta(gamma_l + dgamma_l) - delta(gamma_l)`` with the model's charge-weighted harmonic mean."""

    def _d(gam):
        u = c_light * np.sqrt(gam**2 - 1.0)
        return 1.0 - np.sum(det["rho_l"]) / (params["u0"] * np.sum(det["rho_l"] / u))

    return float(_d(det["gamma_l"] + dgamma_l) - _d(det["gamma_l"]))


def make_gather(ez_line: np.ndarray, phi_line: np.ndarray, scheme: str):
    """Return ``(E_g(zeta), Phi_g(zeta))`` for one line and wall scheme (order-2 Galerkin gather).

    ``E_z`` uses the order-1 (linear) staggered weights along z, the node potential the order-2
    nodal weights; ``E_g = -dPhi_g/dz`` holds identically for each scheme.
    """
    n_e = ez_line.size

    def edge(k):
        if k >= 0:
            return ez_line[k] if k < n_e else ez_line[-1]
        if scheme == "masked":
            return 0.0
        if scheme == "image":
            return ez_line[-k - 1]  # even reflection of the normal field about the wall node
        raise ValueError(scheme)

    def node(k):
        if k >= 0:
            return phi_line[k]
        if scheme == "masked":
            return 0.0
        return -phi_line[-k]  # odd reflection of the potential (image charge)

    def e_g(zeta):
        t = zeta - 0.5
        k = int(np.floor(t))
        f = t - k
        return (1.0 - f) * edge(k) + f * edge(k + 1)

    def phi_g(zeta):
        k = int(np.floor(zeta + 0.5))
        w = _nodal_w2(zeta - k)
        return w[0] * node(k - 1) + w[1] * node(k) + w[2] * node(k + 1)

    return e_g, phi_g


def track_line(e_g, phi_g, dz, dt, gamma0, qm, n_steps, scheme):
    """Leapfrog a wall-born particle through a static line field; return excess history.

    Parameters
    ----------
    qm : float
        Signed ``q / m`` [C/kg].

    Returns
    -------
    dict
        ``excess`` [-] (synchronised), ``zeta`` [-] per step after the push, and the crossing
        phase ``theta`` of the knot at ``zeta = 1/2``.
    """
    u0 = c_light * np.sqrt(gamma0**2 - 1.0)
    q_mc2 = qm / c_light**2
    gam = gamma0
    if scheme == "masked":
        gam = gamma0 - q_mc2 * phi_g(0.0) * 1.0  # D17 kick: dgamma = (q/mc^2) sum W Psi, Psi = -phi
    u = c_light * np.sqrt(gam**2 - 1.0)
    # back-push to t_{n-1/2}
    u = u - qm * e_g(0.0) * 0.5 * dt
    z = 0.0
    zetas, excess = [], []
    theta = None
    for _ in range(n_steps):
        z_old = z
        u = u + qm * e_g(z / dz) * dt
        z = z + u / np.sqrt(1.0 + (u / c_light) ** 2) * dt
        if theta is None and z / dz >= 0.5:
            theta = float((z / dz - 0.5) / ((z - z_old) / dz))
        u_sync = u + qm * e_g(z / dz) * 0.5 * dt
        g_sync = np.sqrt(1.0 + (u_sync / c_light) ** 2)
        zetas.append(z / dz)
        excess.append(g_sync - gamma0 + q_mc2 * phi_g(z / dz))
    del u0
    return {"zeta": np.array(zetas), "excess": np.array(excess), "theta": theta}


def kink_prediction(ez_line, dz, dzeta, theta, qm_abs_mc2, scheme):
    """Closed-form trapezoid error at the zeta = 1/2 knot [-]."""
    e0, e1 = ez_line[0], ez_line[1]
    jump = (e1 - 2.0 * e0) if scheme == "masked" else (e1 - e0)
    return qm_abs_mc2 * dz * 0.5 * theta * (1.0 - theta) * dzeta**2 * abs(jump), abs(jump)


def cell_profile(zeta: np.ndarray, excess: np.ndarray, n_cells: int) -> np.ndarray:
    """Per-cell remainder binned like ``path_identity.py`` ``RPROF`` [-].

    Step ``m >= 1`` adds ``excess[m] - excess[m - 1]`` to the cell ``floor(zeta[m])`` of its end
    position; the first record (``m = 0``) is not binned, as in ``RPROF``.
    """
    out = np.zeros(n_cells)
    k = np.floor(zeta[1:]).astype(int)
    d = np.diff(excess)
    ok = k < n_cells
    np.add.at(out, k[ok], d[ok])
    return out


VARIANTS = (
    ("uniform", "both"),  # original model: z-uniform charge, injected / depressed density
    ("discrete", "both"),  # derived near-wall node deficit (deposit_node_profile)
    ("discrete", "selfconsistent"),  # deficit + axial density rho u0 / v(z)
    ("uniform", "selfconsistent"),
)


def run_grid(
    label: str,
    nx: int,
    nz: int,
    zeta_eval: float = 6.0,
    zeta_far: float = 64.0,
    n_cells: int = 64,
    variants_wanted=VARIANTS,
) -> list[dict]:
    params = s.compute_analytic_parameters(qm_scale=1.0)
    dt, _ = s._cached_integer_period_dt(int(nx), int(nz))
    dz = s.TUNNEL_LENGTH_M / nz
    gamma0 = float(params["gamma0"])
    qm = -e_charge / m_e
    q_abs_mc2 = e_charge / (m_e * c_light**2)
    v0 = float(params["u0"])  # injection speed [m/s] (params["u0"] is v0, not gamma0 v0)
    dzeta = v0 * dt / dz
    n_steps = int(np.ceil((max(zeta_eval, zeta_far) + 1.0) / dzeta))
    variants = []
    for scheme in ("masked", "image"):
        for deposit, density in variants_wanted:
            profiles, det = wall_field_profiles(
                nx, nz, params, deposit=deposit, scheme=scheme, dzeta=dzeta, density=density
            )
            for case, ez_l, phi_l in profiles:
                variants.append((scheme, deposit, case, ez_l, phi_l))
    recs = []
    for scheme, deposit, case, ez_l, phi_l in variants:
        ex_eval, ex_cell0, ex_cell01, kinks, thetas, jumps, e0s = [], [], [], [], [], [], []
        prof, ex_far = [], []
        for li in range(ez_l.shape[0]):
            e_g, phi_g = make_gather(ez_l[li], phi_l[li], scheme)
            tr = track_line(e_g, phi_g, dz, dt, gamma0, qm, n_steps, scheme)
            zt, exc = tr["zeta"], tr["excess"]
            i_eval = int(np.searchsorted(zt, zeta_eval))
            ex_eval.append(float(exc[i_eval]))
            ex_far.append(float(exc[int(np.searchsorted(zt, zeta_far))]))
            prof.append(cell_profile(zt, exc, n_cells))
            # remainder accrued while the particle is in cell 0 / cells 0-1 (by end position)
            i1 = int(np.searchsorted(zt, 1.0))
            i2 = int(np.searchsorted(zt, 2.0))
            ex_cell0.append(float(exc[i1 - 1]))
            ex_cell01.append(float(exc[i2 - 1]))
            kp, jp = kink_prediction(ez_l[li], dz, dzeta, tr["theta"], q_abs_mc2, scheme)
            kinks.append(float(kp))
            jumps.append(float(jp))
            thetas.append(tr["theta"])
            e0s.append(float(abs(ez_l[li][0])))
        rec = {
            "grid": label,
            "case": case,
            "scheme": scheme,
            "deposit": deposit,
            "dzeta": float(dzeta),
            "theta_mean": float(np.mean(thetas)),
            "theta_min": float(np.min(thetas)),
            "theta_max": float(np.max(thetas)),
            "excess_cell0_mean": float(np.mean(ex_cell0)),
            "excess_cells01_mean": float(np.mean(ex_cell01)),
            "excess_at_zeta_eval_mean": float(np.mean(ex_eval)),
            "zeta_eval": zeta_eval,
            "kink_pred_mean": float(np.mean(kinks)),
            "jump_mean_V_per_m": float(np.mean(jumps)),
            "e0_mean_V_per_m": float(np.mean(e0s)),
            "excess_at_zeta_eval_per_line": ex_eval,
            "zeta_far": zeta_far,
            "excess_far_mean": float(np.mean(ex_far)),
            # charge-weighted harmonic-mean delta shift of a per-line gamma excess [-]
            "delta_shift_far": _delta_shift(det, params, np.asarray(ex_far)),
            "rprof_cells_0_11_mean": np.mean(prof, axis=0)[:12].tolist(),
            "rprof_cum_far_mean": float(np.mean(prof, axis=0).sum()),
        }
        recs.append(rec)
        print("WGM " + json.dumps(rec), flush=True)
    return recs


def main() -> int:
    for label, (nx, _ny, nz) in (("coarse", s.COARSE_GRID), ("fine", s.FINE_GRID)):
        params = s.compute_analytic_parameters(qm_scale=1.0)
        dt, _ = s._cached_integer_period_dt(int(nx), int(nz))
        dzeta = float(params["u0"]) * dt / (s.TUNNEL_LENGTH_M / nz)
        for scheme in ("masked", "image"):
            c = deposit_node_profile(scheme, dzeta, 4)
            print(
                f"WGM_DEPOSIT grid={label} scheme={scheme} dzeta={dzeta:.6f} "
                f"c1={c[1]:.6f} c2={c[2]:.6f} c3={c[3]:.6f} deficit1={1.0 - c[1]:.6f}",
                flush=True,
            )
        for r in run_grid(label, nx, nz):
            cells = " ".join(f"{1e9 * v:+.3f}" for v in r["rprof_cells_0_11_mean"][:9])
            print(
                f"WGM_SUMMARY grid={label} scheme={r['scheme']} deposit={r['deposit']} "
                f"case={r['case']} cells0-8[1e-9]= {cells} cum={1e9 * r['rprof_cum_far_mean']:+.3f} "
                f"ddelta_far={r['delta_shift_far']:+.4e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
