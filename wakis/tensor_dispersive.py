# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Subpixel (Kottke) averaging at interfaces with dispersive materials.

An E edge whose averaging pixel holds the materials m with fill fractions
f_m (background included) and the interface normal n sees, along its
component c, the diagonal entry of the averaged inverse-permittivity tensor
[Kottke, Farjadpour & Johnson 2008] at every frequency:

    T_cc(w) = n_c^2 sum_m f_m / eps_m(w) + (1 - n_c^2) / sum_m f_m eps_m(w).

For dispersive eps_m(w) the first (normal, series) term is a sum of the
materials' own elastances and the second (tangential, parallel) term is the
elastance of a mixed pole model. Both are realized exactly as capacitors in
series sharing the edge's displacement D ("sub-capacitors"): each has a
voltage u_k, a static permittivity eps_inf_k and the polarizations of its
poles, driven by u_k with the ADE of `wakis.dispersive`:

    D / eps_0 = eps_inf_k u_k + sum_p mu_kp p_p,      E_c = sum_k c_k u_k
    series of material m:  c = n_c^2 f_m,  eps_inf_k = eps_inf_m,  mu = 1
    tangential:            c = 1 - n_c^2,  eps_inf_k = sum_m f_m eps_inf_m,
                           mu = f_m for the poles of material m

The instantaneous part sum_k c_k / eps_inf_k is the Kottke average of the
eps_inf values, which `wakis.conformal` already puts in `ieps`; the network
adds the dispersive remainder as a local, linear correction after the base
step (and after the off-diagonal tensor correction, which excludes these
edges):

    E^{n+1} = E* + sum_k c_k (u_k^{n+1} - u_k^n - dQ_k / eps_inf_k),
    dQ_k = sum_e b_ke (D^{n+1}_e - D^n_e) / eps_0,

with D^{n+1} - D^n recovered from the base step as (E* - E^n) / ieps. The
stored energy is a sum of positive quadratic forms (one per sub-capacitor
and pole), so the scheme is passive and stable under the bound of
`stability_bound`; see "Dispersive subpixel averaging" in the physics guide.
"""

import types

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import coo_matrix

from .dispersive import DispersiveMaterial, _ade_coefficients


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


class DispersiveNetwork:
    def __init__(self, solver, rows, b_map, c_map, eps_inf, materials, members):
        """
        Sub-capacitor network on a set of E edges (dispersive subpixel
        averaging).

        Parameters
        ----------
        solver : SolverFIT3D
            Solver whose `ieps` holds the static (eps_inf) averaged values.
        rows : ndarray of int
            Sorted flat E indices touched by the network (R).
        b_map : scipy.sparse matrix, shape (K, len(rows))
            Charge map: dQ_k = sum_e b_ke dD_e / eps_0 [dimensionless].
        c_map : scipy.sparse matrix, shape (len(rows), K)
            Voltage map: E_e += sum_k c_ek u_k [dimensionless].
        eps_inf : ndarray, shape (K,)
            Static relative permittivity of each sub-capacitor.
        materials : list of DispersiveMaterial
            Materials whose poles drive the sub-capacitors.
        members : list of (ndarray of int, ndarray of float)
            Per material: the sub-capacitors containing its poles and the
            pole-strength multipliers mu (1 for a series sub-capacitor, the
            fill fraction for a tangential mixture).
        """
        self.rows = np.asarray(rows, dtype=np.int64)
        self.b_host = b_map.tocsr()
        self.c_host = c_map.tocsr()
        self.eps_inf_host = np.asarray(eps_inf, dtype=float)
        self.materials = list(materials)
        self.members = [(np.asarray(s, np.int64), np.asarray(m, float))
                        for s, m in members]  # fmt: skip
        self.n_caps = self.eps_inf_host.size
        self.dt = float(solver.dt)
        ieps = _host(solver.ieps.array)[self.rows].astype(float)
        if np.any(ieps == 0):
            raise ValueError("Dispersive network on edges with ieps = 0 (PEC).")
        self.epsr_host = 1.0 / (eps_0 * ieps)  # static relative permittivity
        self._build_coefficients()
        self._to_backend(solver)
        self.reset()
        self.installed = False
        self.record = False  # store the last step's terms (design gradients)
        self.last = None

    # ---------------------------------------------------------- coefficients
    def _build_coefficients(self):
        """Per-channel ADE coefficients and the a1, den of every capacitor."""
        loss = np.zeros(self.n_caps)  # sum of mu beta + mu sigma dt / (2 eps_0)
        self.lorentz_host, self.debye_host = [], []
        for mat, (sel, mu) in zip(self.materials, self.members):
            lorentz, debye = _ade_coefficients(mat, self.dt)
            for alpha, xi, zeta in lorentz:
                self.lorentz_host.append((alpha, xi, zeta, sel, mu))
            for kappa, beta in debye:
                self.debye_host.append((kappa, beta, sel, mu))
                np.add.at(loss, sel, mu * beta)
            if mat.sigma:
                np.add.at(loss, sel, mu * mat.sigma * self.dt / eps_0 / 2)
        self.a1_host = self.eps_inf_host - loss
        self.den_host = self.eps_inf_host + loss

    def _to_backend(self, solver):
        self.use_gpu = solver.use_gpu
        dtype = solver.E.array.dtype
        if solver.use_gpu:
            import cupy as xp
            from cupyx.scipy.sparse import csr_matrix as gpu_csr

            self.B = gpu_csr(self.b_host.astype(dtype))
            self.Cm = gpu_csr(self.c_host.astype(dtype))
            self.BT = gpu_csr(self.b_host.T.tocsr().astype(dtype))
            self.CT = gpu_csr(self.c_host.T.tocsr().astype(dtype))
        else:
            xp = np
            self.B = self.b_host.astype(dtype)
            self.Cm = self.c_host.astype(dtype)
            self.BT = self.b_host.T.tocsr().astype(dtype)
            self.CT = self.c_host.T.tocsr().astype(dtype)
        self.xp = xp
        self.dtype = dtype
        self.rows_dev = xp.asarray(self.rows)
        self.epsr = xp.asarray(self.epsr_host, dtype=dtype)
        self.eps_inf = xp.asarray(self.eps_inf_host, dtype=dtype)
        self.a1 = xp.asarray(self.a1_host, dtype=dtype)
        self.den = xp.asarray(self.den_host, dtype=dtype)
        self.lorentz = [
            (a, x, z, xp.asarray(s), xp.asarray(m, dtype=dtype))
            for a, x, z, s, m in self.lorentz_host
        ]
        self.debye = [
            (k, b, xp.asarray(s), xp.asarray(m, dtype=dtype))
            for k, b, s, m in self.debye_host
        ]

    def reset(self):
        """Zero the sub-capacitor voltages and polarizations."""
        xp = self.xp
        self.u = xp.zeros(self.n_caps, dtype=self.dtype)
        self.pL = [xp.zeros(s.size, dtype=self.dtype) for *_, s, _ in self.lorentz]
        self.pL_old = [xp.zeros_like(p) for p in self.pL]
        self.pD = [xp.zeros(s.size, dtype=self.dtype) for *_, s, _ in self.debye]

    # ------------------------------------------------------------------ step
    def correct(self, solver, e_old):
        """
        Dispersive correction after the base (and tensor) step.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver whose E holds E* on the network edges.
        e_old : ndarray
            E^n on the network edges.
        """
        E = solver.E.array
        r = self.rows_dev
        dQ = self.B @ ((E[r] - e_old) * self.epsr)
        u = self.u
        acc = self.a1 * u + dQ
        if self.record:  # pole terms per unit multiplier, for design gradients
            lsum = self.xp.zeros_like(u)
            dsum = self.xp.zeros_like(u)
        for k, (alpha, xi, zeta, sel, mu) in enumerate(self.lorentz):
            p_new = alpha * self.pL[k] + xi * self.pL_old[k] + zeta * u[sel]
            acc[sel] -= mu * (p_new - self.pL[k])
            if self.record:
                lsum[sel] += p_new - self.pL[k]
            self.pL_old[k], self.pL[k] = self.pL[k], p_new
        for k, (kappa, _, sel, mu) in enumerate(self.debye):
            acc[sel] -= mu * (kappa - 1) * self.pD[k]
            if self.record:
                dsum[sel] += (kappa - 1) * self.pD[k]
        u_new = acc / self.den
        if self.record:
            self.last = (dQ, u, u_new, lsum, dsum)
        for k, (kappa, beta, sel, _) in enumerate(self.debye):
            self.pD[k] = kappa * self.pD[k] + beta * (u_new[sel] + u[sel])
        E[r] += self.Cm @ (u_new - u - dQ / self.eps_inf)
        self.u = u_new

    # ------------------------------------------------------------ stability
    def stability_bound(self, dt_static):
        """
        Sufficient condition for stability of the coupled leapfrog/ADE step.

        With r = dt / dt_static (dt_static the stable limit of the solver
        operators with the static, eps_inf-averaged permittivity), every
        sub-capacitor contains poles of total strength
        Omega^2 = sum_p mu_p S_p / eps_inf_k <= max_m Omega_m^2 (the bulk
        plasma frequency sum_p S_p / eps_inf of a material, with S_p =
        delta_eps w_0^2 or w_p^2) and resonances w_0 <= max_m w_0. The step is
        stable if

            r^2 + (dt Omega / 2)^2 / (1 - (dt w_0 / 2)^2) <= 1,

        for a lossless material this is the exact bulk condition of
        `DispersiveMaterial.is_stable` for a single Drude pole. Debye poles
        and conductivity are integrated with the trapezoidal rule and only
        add dissipation.

        Returns
        -------
        float
            The left-hand side (stable if <= 1).
        """
        r2 = (self.dt / dt_static) ** 2
        lhs = r2
        for mat in self.materials:
            poles = mat.lorentz_poles
            if not poles:
                continue
            omega2 = sum(p.strength for p in poles) / mat.eps_inf
            w0 = max(p.omega_0 for p in poles)
            z = (self.dt * w0 / 2) ** 2
            if z >= 1:
                return np.inf
            lhs = max(lhs, r2 + (self.dt / 2) ** 2 * omega2 / (1 - z))
        return lhs

    # --------------------------------------------------------------- adjoint
    def state_keys(self):
        keys = ["net_u"]
        keys += [f"net_pL{k}" for k in range(len(self.pL))]
        keys += [f"net_pLold{k}" for k in range(len(self.pL))]
        keys += [f"net_pD{k}" for k in range(len(self.pD))]
        return keys

    def get_state(self, state):
        state["net_u"] = self.u.copy()
        for k in range(len(self.pL)):
            state[f"net_pL{k}"] = self.pL[k].copy()
            state[f"net_pLold{k}"] = self.pL_old[k].copy()
        for k in range(len(self.pD)):
            state[f"net_pD{k}"] = self.pD[k].copy()

    def set_state(self, state, xp):
        self.u = xp.array(state["net_u"])
        self.pL = [xp.array(state[f"net_pL{k}"]) for k in range(len(self.pL))]
        self.pL_old = [xp.array(state[f"net_pLold{k}"]) for k in range(len(self.pL))]
        self.pD = [xp.array(state[f"net_pD{k}"]) for k in range(len(self.pD))]


class NetworkTranspose:
    """
    Transposed network correction for `wakis.inverse.AdjointStepper`.

    Forward (on the network edges R, with g = epsr (E*_R - E^n_R))::

        dQ = B g,  p'_L = alpha p_L + xi p_L,old + zeta u[s],  p'_L,old = p_L
        acc = a1 u + dQ - sum_L mu (p'_L - p_L) - sum_D mu (kappa - 1) p_D
        u' = acc / den,   p'_D = kappa p_D + beta (u'[s] + u[s])
        E'_R = E*_R + C (u' - u - dQ / eps_inf)
    """

    group = None  # not a `_dispersive_groups` entry

    def __init__(self, net):
        self.net = net
        self.idx = net.rows_dev
        self.idx_host = net.rows
        self.keys = net.state_keys()
        self.lam_unew = None

    def get_state(self, state):
        self.net.get_state(state)

    def set_state(self, state, xp):
        self.net.set_state(state, xp)

    def transpose_post(self, lam, lam_star, extra, out):
        net = self.net
        lam_E = lam["E"][self.idx]
        ct = net.CT @ lam_E  # adjoint of (u' - u - dQ / eps_inf)
        lam_unew = lam["net_u"] + ct
        for k, (kappa, beta, sel, _) in enumerate(net.debye):
            lam_unew[sel] += beta * lam[f"net_pD{k}"]
        self.lam_unew = lam_unew  # total adjoint of u' (design gradients)
        lam_acc = lam_unew / net.den
        lam_u = net.a1 * lam_acc - ct
        for k, (kappa, beta, sel, mu) in enumerate(net.debye):
            lpd = lam[f"net_pD{k}"]
            lam_u[sel] += beta * lpd
            out[f"net_pD{k}"] = kappa * lpd - mu * (kappa - 1) * lam_acc[sel]
        for k, (alpha, xi, zeta, sel, mu) in enumerate(net.lorentz):
            lam_pnew = lam[f"net_pL{k}"] - mu * lam_acc[sel]
            out[f"net_pL{k}"] = (
                alpha * lam_pnew + lam[f"net_pLold{k}"] + mu * lam_acc[sel]
            )
            out[f"net_pLold{k}"] = xi * lam_pnew
            lam_u[sel] += zeta * lam_pnew
        out["net_u"] = lam_u
        lam_dQ = lam_acc - ct / net.eps_inf
        lam_g = net.epsr * (net.BT @ lam_dQ)
        lam_star[self.idx] = lam_E + lam_g
        extra[self.idx] -= lam_g


def one_step_network(solver):
    """Solver step followed by the dispersive network correction."""
    net = solver.dispersive_network
    e_old = solver.E.array[net.rows_dev].copy()
    solver._one_step_unnetworked()
    net.correct(solver, e_old)


def install(solver, net):
    """Attach `net` to `solver` (after any tensor correction)."""
    solver.dispersive_network = net
    if net.installed:
        return
    solver._one_step_unnetworked = solver.one_step
    solver.one_step = types.MethodType(one_step_network, solver)
    net.installed = True


# ------------------------------------------------------------- construction
def diagonal_caps(f_disp, materials, p, f_static, eps_static, factor=None):
    """
    Sub-capacitors of the dispersive diagonal Kottke average of single edges.

    Parameters
    ----------
    f_disp : ndarray, shape (n_mat, n_edges)
        Fill fractions of the dispersive materials.
    materials : list of DispersiveMaterial
    p : ndarray, shape (n_edges,)
        n_c^2, the squared normal component along the edge (0 inside a
        material, where only the tangential capacitor remains).
    f_static, eps_static : ndarray, shape (n_static, n_edges) and (n_static,)
        Fractions and relative permittivities of the non-dispersive
        materials (background included); they enter only the tangential
        eps_inf (their series elastance is static and stays in `ieps`).
    factor : ndarray, optional
        Extra factor of each edge's elastance (sheet-edge correction).

    Returns
    -------
    edge, c, eps_inf : ndarray, shape (K,)
        Edge position (index into the edge arrays), weight c and eps_inf of
        each sub-capacitor.
    members : list of (ndarray, ndarray)
        Per material, the capacitors with its poles and the multipliers mu.
    kind : ndarray of int, shape (K,)
        -1 for a tangential capacitor, m for the series capacitor of
        material m.
    """
    n_mat, n_e = f_disp.shape
    factor = np.ones(n_e) if factor is None else factor
    eps_t = np.asarray(f_static).T @ np.asarray(eps_static) if len(eps_static) else 0
    eps_t = eps_t + np.array([m.eps_inf for m in materials]) @ f_disp
    edge, c, einf, kind = [], [], [], []
    members = [([], []) for _ in materials]
    count = 0
    # Tangential (parallel) capacitor of every edge with a dispersive share
    tang = np.flatnonzero((1 - p > 0) & (f_disp.sum(0) > 0))
    edge.append(tang)
    c.append(factor[tang] * (1 - p[tang]))
    einf.append(eps_t[tang])
    kind.append(np.full(tang.size, -1))
    for m in range(n_mat):
        has = f_disp[m, tang] > 0
        members[m][0].append(count + np.flatnonzero(has))
        members[m][1].append(f_disp[m, tang][has])
    count += tang.size
    # Series capacitor of each dispersive material
    for m, mat in enumerate(materials):
        ser = np.flatnonzero((p > 0) & (f_disp[m] > 0))
        edge.append(ser)
        c.append(factor[ser] * p[ser] * f_disp[m, ser])
        einf.append(np.full(ser.size, mat.eps_inf))
        kind.append(np.full(ser.size, m))
        members[m][0].append(count + np.arange(ser.size))
        members[m][1].append(np.ones(ser.size))
        count += ser.size
    members = [(np.concatenate(s), np.concatenate(mu)) for s, mu in members]
    out = (np.concatenate(edge), np.concatenate(c), np.concatenate(einf))
    return (*out, members, np.concatenate(kind))


def network_from_edges(solver, flat, edge, c, eps_inf, materials, members):
    """Build a `DispersiveNetwork` of single-edge sub-capacitors."""
    rows, pos = np.unique(flat[edge], return_inverse=True)
    K = edge.size
    b = coo_matrix((np.ones(K), (np.arange(K), pos)), shape=(K, rows.size))
    cm = coo_matrix((c, (pos, np.arange(K))), shape=(rows.size, K))
    net = DispersiveNetwork(solver, rows, b, cm, eps_inf, materials, members)
    net.cap_row = pos  # row (position in `rows`) of every sub-capacitor
    return net


def conformal_caps(solver):
    """
    Diagonal sub-capacitors of the STL solids with a `DispersiveMaterial` in
    a conformal ('all' or 'tensor') solver, from the per-edge fractions and
    normals of `wakis.conformal`.

    Returns
    -------
    dict or None
        sel (flat E indices with a dispersive share), f_disp (fractions on
        sel), materials, and the output of `diagonal_caps` (edge, c, eps_inf,
        members, kind); None without dispersive solids.
    """
    mix = getattr(solver, "_conformal_mixture", None)
    if mix is None:
        return None
    materials_all = solver.grid.stl_materials or {}
    keys = mix["keys"]
    disp = [
        i
        for i, k in enumerate(keys)
        if isinstance(materials_all[k], DispersiveMaterial)
        and (materials_all[k].poles or materials_all[k].sigma)
    ]
    if not disp:
        return None
    static = [i for i in range(len(keys)) if i not in disp]

    def flat(x):
        return np.concatenate([np.ravel(a, order="F") for a in x])

    fs = [np.concatenate([np.ravel(mix["fs"][c][i], order="F") for c in range(3)])
          for i in range(len(keys))]  # fmt: skip
    f_bg = flat(mix["f_bg"])
    p = flat(mix["p"])
    eps_bg = mix["eps_bg"] / eps_0
    f_disp = np.array([fs[i] for i in disp])
    ieps = _host(solver.ieps.toarray())
    factor = np.asarray(getattr(solver, "conformal_ieps_factor", np.ones(ieps.size)))
    sel = np.flatnonzero((f_disp.sum(0) > 0) & (ieps != 0))
    materials = [materials_all[keys[i]] for i in disp]
    f_static = np.array([f_bg[sel]] + [fs[i][sel] for i in static])
    eps_static = np.array([eps_bg] + [materials_all[keys[i]][0] for i in static])
    caps = diagonal_caps(
        f_disp[:, sel], materials, p[sel], f_static, eps_static, factor[sel]
    )
    out = dict(zip(("edge", "c", "eps_inf", "members", "kind"), caps))
    out.update(sel=sel, f_disp=f_disp[:, sel], p=p[sel], materials=materials)
    return out


def network_from_conformal(solver):
    """
    Dispersive subpixel averaging network of the conformal STL solids
    (see `conformal_caps`), or None.
    """
    caps = conformal_caps(solver)
    if caps is None:
        return None
    return network_from_edges(
        solver, caps["sel"], caps["edge"], caps["c"], caps["eps_inf"],
        caps["materials"], caps["members"],
    )  # fmt: skip


def apply_conformal_dispersive(solver):
    """
    Install the dispersive subpixel network of the conformal STL solids.

    Called by `SolverFIT3D` (through `_apply_stl_dispersive`) for
    ``conformal='all'`` or ``'tensor'``. Raises ValueError if the time step
    violates `DispersiveNetwork.stability_bound`.

    Returns
    -------
    bool
        True if a network was installed.
    """
    net = network_from_conformal(solver)
    if net is None:
        return False
    dt_static = getattr(solver, "conformal_dt_max", None)
    if dt_static:
        lhs = net.stability_bound(dt_static)
        if lhs > 1:
            raise ValueError(
                f"dt={solver.dt:.3e} s violates the stability bound of the "
                f"dispersive subpixel averaging ({lhs:.3f} > 1); pass a smaller "
                "`cfln` or `dt`."
            )
    install(solver, net)
    return True
