# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Device (CuPy) time stepping of `UnstructuredFIT2D` and `UnstructuredFIT3D`.

`DeviceStepper(solver)` copies the operators and the state of a fully set up
solver (materials, sources, probes, DFTs, PML, LTS) to the GPU and advances
it there::

    s = UnstructuredFIT3D(mesh, materials, bc="abc")
    s.add_point_dipole(...); pr = s.add_probe(...); s.add_dft(f)
    g = DeviceStepper(s)          # cupy / cupyx.scipy.sparse
    g.run(n_steps)                # no host synchronisation inside a step
    g.to_host()                   # fields, probe samples and DFTs -> s

Each step is the arithmetic of the solver's `one_step` (same operators, same
order of the elementwise updates) as sparse mat-vecs (cuSPARSE) and
elementwise kernels:

* 3D: ``b += -dt C_eff e``, ``r = C_f^T (M_nu b) - j``, capacitive update.
  The massless-edge elimination ``e_s = -K_ss^-1 K_sf e`` is folded into
  ``C_eff = C_f - C_s K_ss^-1 K_sf`` once at setup: ``K_ss`` is block
  diagonal with tiny blocks (massless edges are isolated), so ``K_ss^-1`` is
  computed block by block and stays sparse; the step is two SpMVs.
* 2D TM / TE: ``G`` / ``G^T`` SpMVs, conductivity / Silver-Mueller /
  absorber losses, the radial CFS-PML filters (gather / scatter on the
  spoke, chord and layer-node index sets), the two-level LTS correction
  (p - 1 SpMVs on the fine rows) and the scattered-field plane wave (one
  SpMV with a zero-copy reversed slice of the source table).
* Dispersive ADE groups: gather ``u[idx]``, per-pole elementwise updates,
  scatter of ``-c corr`` (the indices of a group are unique).
* Sources: the amplitudes ``f(t)`` are evaluated on the host (Python
  scalars passed to the kernels: no transfer); probes are SpMVs written into
  a device buffer; DFTs are accumulated on the device.

The same code runs with ``xp=numpy`` (host reference of the device code
path) and with ``absolute=True``, the *absolute-value twin* of the step
(every coefficient and operator replaced by its absolute value, every
subtraction by an addition) used by `agreement_bound` to bound the
floating-point difference between two implementations (see the physics
guide section ``docs/physicsguide_sections/unstructured_gpu.md``).

Not supported on the device (``NotImplementedError``): ``hodge="whitney"``
(3D; a sparse solve per step), plane waves with dispersive materials.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse.csgraph import connected_components

#: unit round-off of IEEE double precision
UNIT_ROUNDOFF = 2.0**-53
#: largest K_ss block inverted densely for the massless-edge elimination
MAX_MASSLESS_BLOCK = 64


def gamma(m):
    """Higham's gamma_m = m u / (1 - m u) for double precision."""
    mu = m * UNIT_ROUNDOFF
    return mu / (1 - mu)


def _max_row_nnz(A):
    A = sp.csr_matrix(A)
    return int(np.diff(A.indptr).max(initial=0))


def massless_schur_operator(solver):
    """
    ``S = K_ss^-1 K_sf`` (sparse) of a Voronoi `UnstructuredFIT3D` solver.

    ``K_ss = C_s^T M_nu C_s`` couples two massless edges only when they share
    a face; its connected components (blocks) are inverted densely.

    Returns
    -------
    Kinv : csr_matrix (n_s, n_s)
        Block-diagonal ``K_ss^-1``.
    S : csr_matrix (n_s, n_free)
        ``K_ss^-1 K_sf``.
    kappa_max : float
        Largest 2-norm condition number of a block (enters the round-off
        bound of the elimination).
    max_block : int
        Largest block size.
    """
    ns = solver.massless.size
    Kss = (solver.Cs.T @ solver.Mnu @ solver.Cs).tocsr()
    n_comp, lab = connected_components(Kss != 0, directed=False)
    sizes = np.bincount(lab, minlength=n_comp)
    if sizes.max(initial=0) > MAX_MASSLESS_BLOCK:
        raise NotImplementedError(
            f"massless block of {sizes.max()} edges (> {MAX_MASSLESS_BLOCK}): "
            "the dense block inverse would fill in; lower massless_tol"
        )
    rows, cols, vals = [], [], []
    kappa_max = 1.0
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)]
    for k in np.unique(sizes):
        comps = np.flatnonzero(sizes == k)
        idx = np.stack([order[starts[c] : starts[c] + k] for c in comps])  # (nb, k)
        blocks = Kss[idx.ravel()][:, idx.ravel()].toarray()
        nb = comps.size
        B = np.stack([blocks[i * k : (i + 1) * k, i * k : (i + 1) * k] for i in range(nb)])
        inv = np.linalg.inv(B)
        kappa_max = max(kappa_max, float(np.linalg.cond(B).max()))
        rows.append(np.repeat(idx, k, axis=1).ravel())
        cols.append(np.tile(idx, (1, k)).ravel())
        vals.append(inv.ravel())
    Kinv = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ns, ns),
    )
    S = (Kinv @ solver._Ksf).tocsr()
    return Kinv, S, kappa_max, int(sizes.max(initial=0))


class DeviceStepper:
    """
    Advance an unstructured FIT solver on the GPU (or with another array module).

    Parameters
    ----------
    solver : UnstructuredFIT2D or UnstructuredFIT3D
        Fully configured solver (materials, sources, probes, DFTs, PML, LTS).
        Its current state (fields, ADE / PML auxiliaries, step counter) is
        copied; later changes to the solver are not seen.
    xp : module, optional
        ``cupy`` (default) or ``numpy`` (host reference of the device code).
    absolute : bool, optional
        Build the absolute-value twin of the step (for `agreement_bound`):
        no probes / DFTs are recorded.
    fuse_massless : bool, optional
        3D: fold the massless-edge elimination into ``C_eff`` (default);
        False keeps the three-SpMV form ``C_f e + C_s (-K_ss^-1 K_sf e)``.
    sources : bool, optional
        False drops the sources (source-free propagation).

    Attributes
    ----------
    S : dict of arrays
        The evolving state (``u``; ``v`` in 2D or ``b`` in 3D; ADE and PML
        auxiliaries), on the device.
    n : int
        Step counter (as `solver.n`).
    depth : int
        Upper bound on the number of floating-point operations that feed any
        state component in one step (the m of gamma_m).
    """

    def __init__(self, solver, xp=None, absolute=False, fuse_massless=True, sources=True):
        if xp is None:
            import cupy as xp
        self.xp = xp
        self.solver = solver
        self.absolute = bool(absolute)
        self.m = 1.0 if absolute else -1.0  # "minus": a + m * b is a - b
        self.dt = float(solver.dt)
        self.n = int(solver.n)
        self.S = {}
        self.depth = 12
        self._use_sources = sources
        self._gpu = xp.__name__ == "cupy"
        if self._gpu:
            import cupyx.scipy.sparse as csp

            self._csp = csp
        if solver.mesh.dim == 3:
            self._build_3d(solver, fuse_massless)
        else:
            self._build_2d(solver)
        self._build_sources(solver)
        self._build_recorder(solver)
        self._kernels()

    # ------------------------------------------------------------ helpers
    def _f(self, x):
        """Coefficient array (|x| in the absolute twin)."""
        x = np.asarray(x, dtype=float)
        return self.xp.asarray(np.abs(x) if self.absolute else x)

    def _c(self, x):
        """Scalar coefficient."""
        return abs(float(x)) if self.absolute else float(x)

    def _i(self, i):
        return self.xp.asarray(np.asarray(i, dtype=np.int64))

    def _mat(self, A):
        A = sp.csr_matrix(A, dtype=float, copy=True)
        A.sort_indices()
        if self.absolute:
            A.data = np.abs(A.data)
        if self._gpu:
            return self._csp.csr_matrix(A)
        return A

    def _state(self, name, x):
        self.S[name] = self.xp.array(np.asarray(x, dtype=float), copy=True)

    def _kernels(self):
        """Fused elementwise kernels of the hot path (CuPy, normal mode)."""
        self._k_axpy = None
        if not self._gpu or self.absolute:
            return
        import cupy as cp

        # y <- y + c x  (in place)
        self._k_axpy = cp.ElementwiseKernel(
            "float64 x, float64 c", "float64 y", "y = y + c * x", "uns_axpy"
        )
        # u <- (a1 u + dt r) / den  (in place; the plain capacitive update)
        self._k_cap = cp.ElementwiseKernel(
            "float64 a1, float64 r, float64 dt, float64 den",
            "float64 u",
            "u = (a1 * u + dt * r) / den",
            "uns_cap",
        )
        # data += ph * u * dt  (DFT accumulation, broadcast (nf,1) x (1,n))
        self._k_dft = cp.ElementwiseKernel(
            "complex128 ph, float64 u, float64 dt",
            "complex128 data",
            "data = data + ph * u * dt",
            "uns_dft",
        )

    def _axpy(self, y, c, x):
        """y += c x in place."""
        if self._k_axpy is not None:
            self._k_axpy(x, c, y)
        else:
            y += c * x

    # ------------------------------------------------------------ build
    def _build_capacitive(self, up, prefix=""):
        """Device copy of a `CapacitiveUpdate` (a1, den, ADE groups)."""
        self.a1, self.den = self._f(up.a1), self._f(up.den)
        self.groups = []
        for gi, g in enumerate(up.groups):
            dg = {"idx": self._i(g["idx"]), "c": self._f(g["c"]), "lorentz": [], "debye": []}
            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                dg["lorentz"].append(
                    dict(alpha=self._c(alpha), xi=self._c(xi), zeta=self._c(zeta),
                         p=f"g{gi}_L{k}_p", q=f"g{gi}_L{k}_q")  # fmt: skip
                )
                self._state(f"g{gi}_L{k}_p", p)
                self._state(f"g{gi}_L{k}_q", p_old)
            for k, (kappa, beta, p) in enumerate(g["debye"]):
                dg["debye"].append(
                    dict(kappa=self._c(kappa), beta=self._c(beta),
                         km1=self._c(kappa - 1), p=f"g{gi}_D{k}_p")  # fmt: skip
                )
                self._state(f"g{gi}_D{k}_p", p)
            self.groups.append(dg)
            self.depth += 4 * (len(g["lorentz"]) + len(g["debye"])) + 4

    def _build_3d(self, s, fuse_massless):
        if s.hodge != "voronoi":
            raise NotImplementedError("DeviceStepper: hodge='whitney' (sparse solve per step)")
        self.kind = "3d"
        Cs, Ceff = s.Cs, s.Cf
        self.kappa_massless, self.massless_block = 1.0, 0
        self._fused = True
        if s.massless.size:
            Kinv, Sop, kap, nb = massless_schur_operator(s)
            self.kappa_massless, self.massless_block = kap, nb
            if self.absolute:
                # |C_f| + (1 + kappa) |C_s| |K_ss^-1| |K_sf|: rounding of the
                # product and of the host LU solve (forward error <= kappa u)
                A = lambda M: abs(sp.csr_matrix(M))  # noqa: E731
                Ceff = A(s.Cf) + (1 + kap) * (A(Cs) @ A(Kinv) @ A(s._Ksf))
                self.depth += _max_row_nnz(Kinv) + _max_row_nnz(s._Ksf) + _max_row_nnz(Cs)
            elif fuse_massless:
                Ceff = (s.Cf - Cs @ Sop).tocsr()
            else:
                self._fused = False
                self.Cs, self.Kinv, self.Ksf = self._mat(Cs), self._mat(Kinv), self._mat(s._Ksf)
            self.depth += 4 * nb  # block inverse vs LU
        self.Ceff = self._mat(Ceff if self._fused or self.absolute else s.Cf)
        self.CfT = self._mat(s.Cf.T)
        self.Mnu = self._f(s.Mnu_diag)
        self.depth += _max_row_nnz(Ceff) + _max_row_nnz(s.Cf.T)
        self._build_capacitive(s._update)
        self._state("u", s.u)
        self._state("b", s.b)
        self._w = {"u": s.cap, "b": s.Mnu_diag}
        for gi, g in enumerate(s._update.groups):
            for k in range(len(g["lorentz"])):
                self._w[f"g{gi}_L{k}_p"] = self._w[f"g{gi}_L{k}_q"] = g["c"]
            for k in range(len(g["debye"])):
                self._w[f"g{gi}_D{k}_p"] = g["c"]

    def _build_2d(self, s):
        self.kind = "2d"
        self.pol = s.pol
        self.G = self._mat(s.G)
        self.GT = self._mat(s.G.T)
        self.stiff = self._f(s.stiff)
        self.depth += 2 * _max_row_nnz(s.G) + _max_row_nnz(s.G.T)
        self.absorb = s._absorber is not None
        if self.absorb:
            ce = 0.5 * self.dt * s.edge_rate
            self.e_num, self.e_den = self._f(1 - ce), self._f(1 + ce)
        self.lts = None
        if s.lts is not None:
            L = s.lts
            self.lts = dict(
                p=L.p, F=self._i(L.F), C=self._i(L.C), AFF=self._mat(L.A_FF),
                AFC=self._mat(L.A_FC), GF=self._mat(s._G_F), dt2=L.dt**2,
                delta2=self._c(2 * L.delta), beta2=self._c(2 * L.beta),
                T2=[self._c(2 * t) for t in L.T], kappa=[self._c(k) for k in L.kappa],
                out=self._c(L.out),
            )  # fmt: skip
            self.depth += (L.p - 1) * (_max_row_nnz(L.A_FF) + 8) + _max_row_nnz(L.A_FC)
            self.depth += _max_row_nnz(s._G_F) + 4
        self._state("u", s.u)
        self._state("v", s.v)
        self._w = {"u": s.mass, "v": s.stiff}
        if s.pol == "TM":
            self._build_capacitive(s._update)
            for gi, g in enumerate(s._update.groups):
                for k in range(len(g["lorentz"])):
                    self._w[f"g{gi}_L{k}_p"] = self._w[f"g{gi}_L{k}_q"] = g["c"]
                for k in range(len(g["debye"])):
                    self._w[f"g{gi}_D{k}_p"] = g["c"]
        else:
            dt = self.dt
            self.a_te = self._f(s.mass - s.loss * dt / 2)
            self.b_te = self._f(s.mass + s.loss * dt / 2)
            self.te_groups = []
            n_e = s.mesh.n_edges
            for gi, g in enumerate(s._te_groups):
                nseg = g["e"].size
                H = sp.csr_matrix((g["h"], (g["e"], np.arange(nseg))), shape=(n_e, nseg))
                dg = dict(e=self._i(g["e"]), H=self._mat(H), Le0=self._f(g["L"] * eps_0),
                          den=self._c(g["eps_inf"] + g["beta"]), E=f"te{gi}_E",
                          lorentz=[], debye=[])  # fmt: skip
                self._state(f"te{gi}_E", g["E"])
                wseg = eps_0 * g["h"] * g["L"]
                self._w[f"te{gi}_E"] = g["eps_inf"] * wseg
                for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                    dg["lorentz"].append(
                        dict(alpha=self._c(alpha), xi=self._c(xi), zeta=self._c(zeta),
                             p=f"te{gi}_L{k}_p", q=f"te{gi}_L{k}_q")  # fmt: skip
                    )
                    self._state(f"te{gi}_L{k}_p", p)
                    self._state(f"te{gi}_L{k}_q", p_old)
                    self._w[f"te{gi}_L{k}_p"] = self._w[f"te{gi}_L{k}_q"] = wseg
                for k, (kappa, beta, p) in enumerate(g["debye"]):
                    dg["debye"].append(
                        dict(kappa=self._c(kappa), beta=self._c(beta), p=f"te{gi}_D{k}_p")
                    )
                    self._state(f"te{gi}_D{k}_p", p)
                    self._w[f"te{gi}_D{k}_p"] = wseg
                self.te_groups.append(dg)
                self.depth += _max_row_nnz(H) + 4 * (len(g["lorentz"]) + len(g["debye"])) + 6
        # plane wave (scattered field)
        self.pw = None
        if s.plane_wave is not None:
            if s.dispersive:
                raise NotImplementedError("DeviceStepper: plane wave with dispersive media")
            self.pw = dict(W=self._mat(s._pw_W), sub=s._pw_sub, nj=s._pw_W.shape[1],
                           s=s.plane_wave["s"], S=None, top=-1)  # fmt: skip
            self.depth += 2 * _max_row_nnz(s._pw_W) + 4
        # radial PML
        self.pml = None
        if s._pml is not None:
            p = s._pml
            self.pml = dict(nodes=self._i(p.nodes), sp=self._i(p.spokes), ch=self._i(p.chords),
                            massN=self._f(s.mass[p.nodes]), st_sp=self._f(s.stiff[p.spokes]),
                            st_ch=self._f(s.stiff[p.chords]))  # fmt: skip
            self._state("pml_q", p.q)
            self._w["pml_q"] = 1 / s.mass[p.nodes]
            w_nodes = s.mass[p.nodes]
            for name, f, w in (("fn1", p.fn1, w_nodes), ("fn2", p.fn2, w_nodes),
                               ("fs", p.fs, s.stiff[p.spokes]), ("fc", p.fc, s.stiff[p.chords])):  # fmt: skip
                self.pml[name] = dict(
                    c1=self._f(f.c1), c2=self._f(f.c2), kn=self._f(f.kn), sn=self._f(f.sn),
                    sd=self._f(f.sd), den=self._f(f.kd + f.sd * f.c2), key=f"pml_{name}",
                )  # fmt: skip
                # P = int x dt enters y as sigma P: weight w sigma^2 (zero where
                # the filter is the identity); x0, y0 are copies of x, y
                sig2 = np.broadcast_to(f.sn**2 + f.sd**2, np.shape(w))
                for v in ("Px", "Py", "x0", "y0"):
                    self._state(f"pml_{name}_{v}", getattr(f, v))
                    self._w[f"pml_{name}_{v}"] = w * sig2 if v in ("Px", "Py") else w
            self.depth += 30

    def _build_sources(self, s):
        self.sources = []
        if not self._use_sources:
            return
        for src in s.sources:
            if len(src) == 3:
                kind, w, f = src
            else:
                (w, f), kind = src, "node"
            w = np.asarray(w, dtype=float)
            nz = np.flatnonzero(w)
            dense = nz.size > 0.25 * w.size
            self.sources.append(
                dict(kind=kind, f=f, dense=dense,
                     w=self._f(w) if dense else self._f(w[nz]), idx=None if dense else self._i(nz))  # fmt: skip
            )
            self.depth += 2

    def _build_recorder(self, s):
        self.probes, self.dfts = [], []
        if self.absolute:
            return
        for rec in s.recorder.probes:
            P = self._mat(rec["P"])
            self.probes.append(dict(P=P, buf=self.xp.zeros((256, rec["P"].shape[0])), k=0))
        for rec in s.recorder.dfts:
            self.dfts.append(dict(freqs=self.xp.asarray(rec["freqs"]),
                                  data=self.xp.asarray(rec["data"]).astype(complex)))  # fmt: skip

    # ------------------------------------------------------------ pieces
    def _amp(self, f, t):
        a = float(f(t))
        return abs(a) if self.absolute else a

    def _add_sources(self, y, kind, t, c):
        """y += c * sum_k w_k f_k(t) over the sources of `kind` (in place)."""
        for src in self.sources:
            if src["kind"] != kind:
                continue
            a = c * self._amp(src["f"], t)
            if src["dense"]:
                self._axpy(y, a, src["w"])
            else:
                idx = src["idx"]
                y[idx] = y[idx] + a * src["w"]

    def _cap_step(self, r):
        """`CapacitiveUpdate.step` in place on S['u'] (r: curl - source)."""
        S, m, dt = self.S, self.m, self.dt
        u = S["u"]
        if not self.groups:
            if self._k_axpy is not None:
                self._k_cap(self.a1, r, dt, self.den, u)
            else:
                u[...] = (self.a1 * u + dt * r) / self.den
            return
        acc = self.a1 * u + dt * r
        olds = []
        for g in self.groups:
            idx, c = g["idx"], g["c"]
            un = u[idx]
            olds.append(un)
            corr = self.xp.zeros(un.shape)
            for L in g["lorentz"]:
                p, q = S[L["p"]], S[L["q"]]
                p_new = L["alpha"] * p + L["xi"] * q + L["zeta"] * un
                corr = corr + (p_new + m * p)
                S[L["q"]], S[L["p"]] = p, p_new
            for D in g["debye"]:
                corr = corr + D["km1"] * S[D["p"]]
            acc[idx] = acc[idx] + m * (c * corr)
        u[...] = acc / self.den
        for g, un in zip(self.groups, olds):
            for D in g["debye"]:
                S[D["p"]] = D["kappa"] * S[D["p"]] + D["beta"] * (u[g["idx"]] + un)

    def _grad(self, u):
        """G r(X) u (LTS correction on the fine nodes)."""
        g = self.G @ u
        L = self.lts
        if L is None or L["p"] == 1 or L["F"].size == 0:
            return g
        m = self.m
        yF = u[L["F"]]
        gC = L["AFC"] @ u[L["C"]]
        d_old, d = self.xp.zeros(yF.shape), yF
        for k in range(1, L["p"]):
            Xd = L["dt2"] * (L["AFF"] @ d + L["kappa"][k] * gC)
            d_new = L["delta2"] * d + m * (L["beta2"] * Xd) + L["T2"][k] * yF + m * d_old
            d_old, d = d, d_new
        corr = L["out"] * d + m * yF
        return g + L["GF"] @ corr

    def _pw_values(self, n):
        """W @ [S[sub n - j]]_j (zero-copy reversed slice of the padded table)."""
        pw = self.pw
        top = pw["sub"] * n
        if pw["S"] is None or pw["top"] < top:
            self._pw_table(top)
        g = pw["S"][top + 1 : top + pw["nj"] + 1][::-1]
        return pw["W"] @ self.xp.ascontiguousarray(g)

    def _pw_table(self, top):
        """(Re)build the padded source table up to index `top` (host -> device)."""
        pw = self.pw
        n_max = max(2 * (pw["top"] + 1), top + 1 + 4096)
        s = np.asarray(pw["s"](np.arange(n_max) * self.dt / pw["sub"]), dtype=float)
        if self.absolute:
            s = np.abs(s)
        pw["S"] = self.xp.asarray(np.r_[np.zeros(pw["nj"]), s])
        pw["top"] = n_max - 1

    def prepare(self, n_steps):
        """Precompute host-side tables for the next `n_steps` (plane wave)."""
        if getattr(self, "pw", None) is not None:
            top = self.pw["sub"] * (self.n + n_steps + 1)
            if self.pw["S"] is None or self.pw["top"] < top:
                self._pw_table(top)

    def _cfs(self, F, x):
        """`_CFSRatio.__call__` on the device state of filter F."""
        S, k, m = self.S, F["key"], self.m
        Px = F["c1"] * S[k + "_Px"] + F["c2"] * (x + S[k + "_x0"])
        y = (F["kn"] * x + F["sn"] * Px + m * (F["sd"] * (F["c1"] * S[k + "_Py"] + F["c2"] * S[k + "_y0"]))) / F["den"]
        S[k + "_Py"] = F["c1"] * S[k + "_Py"] + F["c2"] * (y + S[k + "_y0"])
        S[k + "_Px"], S[k + "_x0"], S[k + "_y0"] = Px, x, y
        return y

    def _pml_nodes(self, u1):
        P, S = self.pml, self.S
        u1[P["nodes"]] = self._cfs(P["fn2"], self._cfs(P["fn1"], S["pml_q"] / P["massN"]))

    def _te_voltage(self, v):
        vol = self.stiff * v
        for g in self.te_groups:
            vol = vol + g["H"] @ self.S[g["E"]]
        return vol

    def _te_ade_step(self, v):
        S, m = self.S, self.m
        for g in self.te_groups:
            En = S[g["E"]]
            Dn = v[g["e"]] / g["Le0"]
            psum = self.xp.zeros(En.shape)
            for L in g["lorentz"]:
                p, q = S[L["p"]], S[L["q"]]
                p_new = L["alpha"] * p + L["xi"] * q + L["zeta"] * En
                psum = psum + p_new
                S[L["q"]], S[L["p"]] = p, p_new
            for D in g["debye"]:
                psum = psum + (D["kappa"] * S[D["p"]] + D["beta"] * En)
            E_new = (Dn + m * psum) / g["den"]
            for D in g["debye"]:
                S[D["p"]] = D["kappa"] * S[D["p"]] + D["beta"] * (E_new + En)
            S[g["E"]] = E_new

    # ------------------------------------------------------------ steps
    def _step_3d(self, t_half):
        S, m, dt = self.S, self.m, self.dt
        u, b = S["u"], S["b"]
        if self._fused:
            self._axpy(b, m * dt, self.Ceff @ u)
        else:
            e_s = m * (self.Kinv @ (self.Ksf @ u))
            self._axpy(b, m * dt, self.Ceff @ u)
            self._axpy(b, m * dt, self.Cs @ e_s)
        r = self.CfT @ (self.Mnu * b)
        self._add_sources(r, "node", t_half, m)
        self._cap_step(r)

    def _step_tm(self, t_half):
        S, m, dt = self.S, self.m, self.dt
        u, v = S["u"], S["v"]
        g = self._grad(u)
        if self.absorb:
            v[...] = (self.e_num * v + m * dt * g) / self.e_den
        else:
            self._axpy(v, m * dt, g)
        if self.pml is None:
            r = self.GT @ (self.stiff * v)
        else:
            P = self.pml
            h = self.stiff * v
            h[P["sp"]] = P["st_sp"] * self._cfs(P["fs"], v[P["sp"]])
            h[P["ch"]] = P["st_ch"] * self._cfs(P["fc"], v[P["ch"]])
            r = self.GT @ h
        self._add_sources(r, "node", t_half, m)
        if self.pw is not None:
            r += m * (self._pw_values(self.n + 1) + m * self._pw_values(self.n))
        self._cap_step(r)
        if self.pml is not None:
            S["pml_q"] = S["pml_q"] + dt * r[self.pml["nodes"]]
            self._pml_nodes(u)

    def _step_te(self, t_half):
        S, m, dt = self.S, self.m, self.dt
        u, v = S["u"], S["v"]
        vol = self._te_voltage(v)
        P = self.pml
        if P is not None:
            vol[P["sp"]] = P["st_sp"] * self._cfs(P["fs"], v[P["sp"]])
            vol[P["ch"]] = P["st_ch"] * self._cfs(P["fc"], v[P["ch"]])
        if self.pw is not None:
            vol = vol + m * self._pw_values(self.n)
        y = self.GT @ vol
        self._add_sources(y, "node", t_half, 1.0)
        if P is None:
            u[...] = (self.a_te * u + m * dt * y) / self.b_te
        else:
            r = m * y
            u[...] = (self.a_te * u + dt * r) / self.b_te
            S["pml_q"] = S["pml_q"] + dt * r[P["nodes"]]
            self._pml_nodes(u)
        z = self._grad(u)
        self._add_sources(z, "edge", t_half, m)
        if self.absorb:
            v[...] = (self.e_num * v + dt * z) / self.e_den
        else:
            self._axpy(v, dt, z)
        self._te_ade_step(v)

    def step(self):
        """One leapfrog step (the solver's `one_step`)."""
        t_half = (self.n + 0.5) * self.dt
        if self.kind == "3d":
            self._step_3d(t_half)
        elif self.pol == "TM":
            self._step_tm(t_half)
        else:
            self._step_te(t_half)
        self.n += 1
        self._record(self.n * self.dt)

    one_step = step

    def run(self, n_steps):
        """Advance `n_steps` steps (no host synchronisation)."""
        n_steps = int(n_steps)
        self.prepare(n_steps)
        for pr in self.probes:
            self._reserve(pr, pr["k"] + n_steps)
        for _ in range(n_steps):
            self.step()

    # ------------------------------------------------------------ monitors
    def _reserve(self, pr, n):
        if pr["buf"].shape[0] < n:
            new = self.xp.zeros((max(n, 2 * pr["buf"].shape[0]), pr["buf"].shape[1]))
            new[: pr["k"]] = pr["buf"][: pr["k"]]
            pr["buf"] = new

    def _record(self, t):
        if self.absolute:
            return
        u = self.S["u"]
        for pr in self.probes:
            self._reserve(pr, pr["k"] + 1)
            pr["buf"][pr["k"]] = pr["P"] @ u
            pr["k"] += 1
        for d in self.dfts:
            ph = self.xp.exp(-2j * np.pi * d["freqs"] * t)
            if self._gpu:
                self._k_dft(ph[:, None], u[None], self.dt, d["data"])
            else:
                d["data"] += ph[:, None] * u[None] * self.dt

    def probe_signal(self, i=0):
        """Samples of probe `i` recorded on the device, (n_steps, n_points), host."""
        pr = self.probes[i]
        return self._host(pr["buf"][: pr["k"]])

    def dft_data(self, i=0):
        """Accumulated DFT `i`, (n_freqs, n_dofs), host."""
        return self._host(self.dfts[i]["data"])

    # ------------------------------------------------------------ transfer
    def _host(self, x):
        return x.get() if hasattr(x, "get") else np.array(x, copy=True)

    def synchronize(self):
        if self._gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def get_state(self):
        """Host copy of the state dict."""
        return {k: self._host(v) for k, v in self.S.items()}

    def set_state(self, state):
        for k, v in state.items():
            self.S[k] = self.xp.array(np.asarray(v, dtype=float), copy=True)

    def weights(self):
        """Diagonal norm weights W of each state entry (energy-type, host)."""
        return {k: np.asarray(v, dtype=float) for k, v in self._w.items()}

    def to_host(self):
        """Copy fields, step counter, probe samples and DFTs back into the solver."""
        s = self.solver
        st = self.get_state()
        s.u = st["u"]
        if self.kind == "3d":
            s.b = st["b"]
        else:
            s.v = st["v"]
        s.n = self.n
        for rec, pr in zip(s.recorder.probes, self.probes):
            rec["samples"] = list(self._host(pr["buf"][: pr["k"]]))
        for rec, d in zip(s.recorder.dfts, self.dfts):
            rec["data"] = self._host(d["data"])
        return s

    # ------------------------------------------------------------ memory
    def device_bytes(self):
        """Bytes of the arrays and sparse operators held by this stepper (the
        array module's memory: device for CuPy)."""
        skip = {"solver", "_w", "xp", "_csp", "S"}
        tot = sum(_nbytes(v, self.xp) for k, v in self.__dict__.items() if k not in skip)
        return tot + sum(_nbytes(v, self.xp) for v in self.S.values())


def _nbytes(v, xp):
    if hasattr(v, "data") and hasattr(v, "indptr"):
        return v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
    if isinstance(v, xp.ndarray):
        return int(v.nbytes)
    if isinstance(v, dict):
        return sum(_nbytes(x, xp) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(_nbytes(x, xp) for x in v)
    return 0


# ---------------------------------------------------------------- round-off
def _wnorm(state, weights, keys=None):
    keys = weights.keys() if keys is None else keys
    return float(np.sqrt(sum(np.sum(weights[k] * state[k] ** 2) for k in keys)))


def theory_amplification(solver):
    """
    sup_m ||A^m||_W of the lossless / passive leapfrog in the energy norm:
    sqrt((1 + rho) / (1 - rho)), rho = dt / 2 sqrt(lambda_max) (the Courant
    number; the LTS stability ratio with LTS).
    """
    rho = solver.lts_info["rho"] if getattr(solver, "lts", None) is not None else solver.courant
    return float(np.sqrt((1 + rho) / (1 - rho)))


def empirical_amplification(solver, n_steps, n_seeds=4, keys=None):
    """
    max over seeds and m <= n_steps of ||A^m x0||_W / ||x0||_W for random x0
    (all state entries, scaled by 1/sqrt(W)), source-free, host reference
    of the device code. A lower estimate of sup ||A^m||_W.
    """
    best = 1.0
    for seed in range(n_seeds):
        st = DeviceStepper(solver, xp=np, sources=False)
        st._record = lambda t: None
        W = st.weights()
        rng = np.random.default_rng(seed)
        x0 = {k: rng.standard_normal(W[k].shape) / np.sqrt(np.where(W[k] > 0, W[k], np.inf))
              for k in W}  # fmt: skip
        # zero-weight entries carry no energy; keep them zero
        st.set_state({**st.get_state(), **x0})
        n0 = _wnorm(st.get_state(), W, keys)
        for _ in range(int(n_steps)):
            st.step()
            best = max(best, _wnorm(st.get_state(), W, keys) / n0)
    return best


def agreement_bound(solver, n_steps, K=None, probes=True):
    """
    Round-off bound on the difference of two implementations of the solver step.

    Both implementations (host SciPy `one_step`, device `DeviceStepper`)
    evaluate the same linear recurrence x_{n+1} = A x_n + f_n with sums in
    different orders (cuSPARSE vs SciPy SpMV, FMA contraction). Each local
    error satisfies |delta_n| <= gamma_m |A|(|x_n|) componentwise (Higham
    2002, eq. 3.4 and Lemma 3.3, any summation order), where |A|(.) is the
    absolute-value twin of the step and m = `DeviceStepper.depth`. The
    difference d_n of the two runs obeys d_{n+1} = A d_n + (delta_n - delta'_n),
    so

        ||d_N||_W <= 2 K gamma_m sum_{n<N} || |A|(|x_n|) ||_W =: eps_N,

    with K = sup_m ||A^m||_W. For the lossless / passive leapfrog in the
    energy norm K = sqrt((1 + rho) / (1 - rho)) (`theory_amplification`);
    with the PML or LTS the larger of that and `empirical_amplification`.

    Parameters
    ----------
    solver : configured solver at its initial state (not advanced).
    n_steps : int
    K : float, optional
        Amplification constant (default as above).

    Returns
    -------
    dict
        ``eps`` (n_steps,) bound on ||d_n||_W after each step, ``K``,
        ``depth``, ``gamma``, ``norm`` (n_steps,) ||x_n||_W of the reference,
        ``probe`` list of (n_steps, n_points) componentwise probe bounds,
        ``dft`` list of (n_freqs,) bounds on ||Delta DFT_k||_{W_u},
        ``weights``.
    """
    ref = DeviceStepper(solver, xp=np)
    twin = DeviceStepper(solver, xp=np, absolute=True)
    W = ref.weights()
    gm = gamma(ref.depth)
    if K is None:
        K = theory_amplification(solver)
        if getattr(solver, "_pml", None) is not None or getattr(solver, "lts", None) is not None:
            K = max(K, empirical_amplification(solver, n_steps))
    ref.prepare(n_steps)
    twin.prepare(n_steps)
    eps, norm, unorm = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    acc = 0.0
    Pabs = [abs(sp.csr_matrix(r["P"])) for r in solver.recorder.probes] if probes else []
    Pw = [P @ (1 / np.sqrt(W["u"])) for P in Pabs]
    probe_b = [np.zeros((n_steps, P.shape[0])) for P in Pabs]
    dft_b = []
    for n in range(n_steps):
        x = ref.get_state()
        twin.set_state({k: np.abs(v) for k, v in x.items()})
        twin.n = ref.n
        twin.step()
        acc += _wnorm(twin.get_state(), W)
        ref.step()
        xn = ref.get_state()
        eps[n] = 2 * K * gm * acc
        norm[n] = _wnorm(xn, W)
        unorm[n] = _wnorm(xn, W, ["u"])
        for P, Pv, out in zip(Pabs, Pw, probe_b):
            # |P| |d| <= eps_n |P| W^-1/2, plus both probe SpMVs (gamma_k)
            out[n] = eps[n] * Pv + 2 * gamma(_max_row_nnz(P)) * (P @ np.abs(xn["u"]))
    dt = solver.dt
    for rec in solver.recorder.dfts:
        # sum_n dt [ ||d_n|| + (|delta phi| + gamma_3) ||u_n|| ] + 2 gamma_N sum |terms|
        # |delta phi| <= 8 u: complex exp of the same rounded argument, CUDA
        # sin/cos/exp within 2 ulp each (CUDA C Programming Guide, App. "Mathematical
        # Functions"), the host libm within 1 ulp
        c_phi = 8 * UNIT_ROUNDOFF + gamma(3)
        b = dt * (eps.sum() + c_phi * unorm.sum()) + 2 * gamma(n_steps) * dt * unorm.sum()
        dft_b.append(np.full(np.size(rec["freqs"]), b))
    return dict(eps=eps, K=K, depth=ref.depth, gamma=gm, norm=norm, probe=probe_b,
                dft=dft_b, weights=W)  # fmt: skip
