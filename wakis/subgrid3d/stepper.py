# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
`SubgridLTSFIT3D`: `SubgridFIT3D` with local time stepping, CPML and
GPU block kernels.

    z         = R_0 e^n                         (LTS, `lts.SubgridLTS`; R_0 = I without)
    psi_d     = bh_d psi_d + ah_d K^(d) z       (CPML faces, `pml.SubgridCPML`)
    b^{n+1/2} = b^{n-1/2} - dt (K z + sum_d psi_d)
    phi_d     = be_d phi_d + ae_d K^(d)H W b^{n+1/2}
    e^{n+1}   = e^n + dt D^-1 (K^H W b^{n+1/2} + sum_d phi_d - P_e^H i^{n+1/2})

``K``/``K^H`` are either the global CSR matrices (``kernels='csr'``, ``b`` on
the faces with nonzero Hodge) or the dense per-block curls plus the sparse
trace (``kernels='block'``, `gpu.BlockOps`, ``b`` on all faces); either on
NumPy/SciPy or CuPy (``backend='cupy'``).
"""

import numpy as np

from .gpu import BlockOps, get_xp
from .lts import SubgridLTS, lts_dt_max, stability_ratio
from .pml import SubgridCPML
from .solver import SubgridFIT3D

__all__ = ["SubgridLTSFIT3D"]


class SubgridLTSFIT3D(SubgridFIT3D):
    """
    3D subgrid FIT with multi-level LTS, CPML and block/GPU kernels.

    Parameters (beyond `SubgridFIT3D`)
    ----------------------------------
    lts : bool
        Local time stepping per refinement level (substeps = patch ratio).
    nu : float
        Chebyshev damping of the LTS polynomial.
    lts_kw : dict, optional
        Extra keywords of `SubgridLTS` (``ratios``, ``interface_parents``,
        ``grow``, ``edge_level``).
    dt : float, optional
        Coarse step [s]. Default ``cfl * dt_max`` with ``dt_max`` the global
        Lanczos limit (no LTS) or the LTS limit found by bisection on the
        Lanczos stability ratio (`lts.lts_dt_max`).
    pml : dict, optional
        Keywords of `SubgridCPML` (``thickness``, ``R0``, ``order``,
        ``alpha_hz``); the walls behind the layer keep their PEC/PMC type.
    kernels : {'csr', 'block'}
    backend : {'numpy', 'cupy'}
    dt_rtol : float
        Bracket tolerance of the LTS ``dt_max`` bisection.
    """

    def __init__(self, mesh, eps_r=None, mu_r=None, pec=None, pmc=None, dt=None, cfl=0.95,
                 lts=True, nu=0.05, lts_kw=None, pml=None, kernels="csr", backend="numpy",
                 dt_rtol=2e-3, **kw):  # fmt: skip
        super().__init__(mesh, eps_r=eps_r, mu_r=mu_r, pec=pec, pmc=pmc, dt=dt, cfl=cfl, **kw)
        self.cfl = cfl
        self.lts = None
        self.lts_info = {}
        lts_kw = dict(lts_kw or {})
        if lts and max(self.mesh.level) > 0:
            from .lts import is_stable

            if dt is None:
                probe = SubgridLTS(self, self.dt_max, nu=nu, **lts_kw)
                P = float(np.prod(probe.ratios))
                # bracket around P x the global limit; fall back to the global limit
                lo = 0.85 * P * self.dt_max
                if not is_stable(SubgridLTS(self, lo, nu=nu, **lts_kw))[0]:
                    lo = self.dt_max
                dt_lts, hist = lts_dt_max(self, 1.05 * P * self.dt_max, dt_lo=lo,
                                          rtol=dt_rtol, nu=nu, **lts_kw)  # fmt: skip
                self.lts_info.update(dt_max=dt_lts, bisection=hist)
                dt = cfl * dt_lts
            self.dt = float(dt)
            self.lts = SubgridLTS(self, self.dt, nu=nu, **lts_kw)
            ok, rho, marg = is_stable(self.lts, tol=1e-10)
            if not ok:
                raise ValueError(
                    f"LTS step dt={self.dt:.4e} s is unstable "
                    f"(rho={rho:.6f}, margins={marg})"
                )
            self.lts_info.update(self.lts.summary())
        self.cpml = None
        if pml:
            self.cpml = SubgridCPML(self, self.dt, **pml)
            if self.lts is not None:
                self._check_lts_pml()
        self.kernels = kernels
        self.backend = backend
        self._setup_backend()
        self.reset()

    # -------------------------------------------------------------- setup
    def _check_lts_pml(self):
        d = self.lts.levels[1]
        near = np.zeros(self.n_free, bool)
        near[d.F] = near[d.C] = True
        if np.any(near[self.cpml.edges]):
            raise ValueError("LTS fine edges or their buffer reach the CPML layer")
        touched = np.zeros(self.n_faces, bool)
        touched[d.Phi] = True
        if np.any(touched[self.cpml.face_set]):
            raise ValueError("LTS fine faces reach the CPML layer")

    def _setup_backend(self):
        xp, xsp = get_xp(self.backend)
        self.xp = xp
        if self.kernels == "block":
            self.ops = BlockOps(self, self.backend)
            self._Wb = self.ops.W_all
            self._nb = self.mesh.n_faces
            self._bmap = self.ops.faces  # state face -> index in b
        elif self.kernels == "csr":
            self.ops = None
            self._Kx = xsp.csr_matrix(self.K) if xp is not np else self.K
            self._KHx = xsp.csr_matrix(self.KH) if xp is not np else self.KH
            self._Wb = xp.asarray(self.W)
            self._nb = self.n_faces
            self._bmap = None
        else:
            raise ValueError("kernels must be 'csr' or 'block'")
        self._Dinv = xp.asarray(self.Dinv)
        if self.lts is not None:
            self.lts.to_backend(self.backend)
        if self.cpml is not None:
            c = self.cpml
            self._pml = []
            for d in range(3):
                pf = c.pf[d] if self._bmap is None else self.faces[c.pf[d]]
                Ke = c.Ke[d]
                if self._bmap is not None:  # columns -> all-face indices
                    Ke = Ke.tocoo()
                    import scipy.sparse as sp

                    Ke = sp.csr_matrix((Ke.data, (Ke.row, self.faces[Ke.col])),
                                       shape=(Ke.shape[0], self.mesh.n_faces))  # fmt: skip
                conv = (lambda m: xsp.csr_matrix(m)) if xp is not np else (lambda m: m)
                self._pml.append(dict(
                    pf=xp.asarray(pf), Kf=conv(c.Kf[d]), bh=xp.asarray(c.bh[d]),
                    ah=xp.asarray(c.ah[d]), pe=xp.asarray(c.pe[d]), Ke=conv(Ke),
                    be=xp.asarray(c.be[d]), ae=xp.asarray(c.ae[d])))  # fmt: skip

    # ---------------------------------------------------------- operators
    def _R(self, e):
        return e if self.lts is None else self.lts.apply(e)

    def _Kz(self, z):
        if self.ops is not None:
            return self.ops.K(z, self.xp.empty(self._nb))
        return self._Kx @ z

    def _KHy(self, y):
        if self.ops is not None:
            return self.ops.KH(y)
        return self._KHx @ y

    # -------------------------------------------------------------- fields
    def reset(self):
        if not hasattr(self, "xp"):
            return super().reset()
        xp = self.xp
        self.e = xp.zeros(self.n_free, self.dtype)
        self.b = xp.zeros(self._nb, self.dtype)
        self.t = 0.0
        self.n = 0
        if self.cpml is not None:
            self.psi = [xp.zeros(p["pf"].size, self.dtype) for p in self._pml]
            self.phi = [xp.zeros(p["pe"].size, self.dtype) for p in self._pml]
        for p in self.probes:
            p["values"] = []
            if p["dft"] is not None:
                p["dft"][:] = 0

    def b_faces(self):
        """``b`` on the solver faces (``self.faces`` order), as NumPy."""
        b = self.b if self._bmap is None else self.b[self._bmap]
        return b.get() if hasattr(b, "get") else np.asarray(b)

    def e_host(self):
        return self.e.get() if hasattr(self.e, "get") else np.asarray(self.e)

    # ------------------------------------------------------------ stepping
    def one_step(self):
        dt, xp = self.dt, self.xp
        z = self._R(self.e)
        Kz = self._Kz(z)
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                self.psi[d] = p["bh"] * self.psi[d] + p["ah"] * (p["Kf"] @ z)
                Kz[p["pf"]] += self.psi[d]
        self.b -= dt * Kz
        y = self._Wb * self.b
        rhs = self._KHy(y)
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                self.phi[d] = p["be"] * self.phi[d] + p["ae"] * (p["Ke"] @ y)
                rhs[p["pe"]] += self.phi[d]
        s = self._source_vector(self.t + 0.5 * dt)
        if s is not None:
            rhs -= xp.asarray(s)
        self.e += dt * self._Dinv * rhs
        self.n += 1
        self.t = self.n * dt

    def _record(self):
        for p in self.probes:
            row = p["row"]
            v = (row.data * self.e[row.indices]).sum()
            v = complex(v) if np.iscomplexobj(row.data) or self.dtype is complex else float(v)
            v = v / p["L"]
            p["values"].append(v)
            if p["dft"] is not None:
                p["dft"] += v * np.exp(-1j * 2 * np.pi * p["freqs"] * self.t) * self.dt

    def run(self, n_steps, energy=False):
        """Advance `n_steps`; returns the modified-energy history if `energy`
        (lossless only)."""
        hist = []
        if self.n == 0:
            self._record()
        for _ in range(int(n_steps)):
            e_old = self.e.copy() if energy else None
            self.one_step()
            self._record()
            if energy:
                hist.append(self.modified_energy(e_old, self.e))
        return np.array(hist) if energy else None

    def synchronize(self):
        if self.xp is not np:
            self.xp.cuda.Device().synchronize()

    # ------------------------------------------------------------ energy
    def S0(self, v):
        """``S_0 v = K^H W K R_0 v`` (Hermitian LTS stiffness)."""
        return self._KHy(self._Wb * self._Kz(self._R(v)))

    def modified_energy(self, e0, e1):
        """
        Conserved quadratic form of the (LTS) leapfrog, lossless and
        source-free: ``1/2 [w^H (D - dt^2/4 S_0) w + ybar^H S_0 ybar]``,
        ``w = (e1 - e0)/dt``, ``ybar = (e1 + e0)/2`` [J/s^2].
        """
        xp, dt = self.xp, self.dt
        w = (e1 - e0) / dt
        yb = 0.5 * (e1 + e0)
        D = 1.0 / self._Dinv
        val = xp.vdot(w, D * w) - dt**2 / 4 * xp.vdot(w, self.S0(w)) + xp.vdot(yb, self.S0(yb))
        val = val.item() if hasattr(val, "item") else val
        return float(0.5 * np.real(val))

    def stability_ratio(self, tol=1e-6, **kw):
        """Lanczos ``rho = dt/2 sqrt(lambda_max(D^-1 S_0))`` (stable iff < 1)."""
        if self.lts is None:
            return 0.5 * self.dt * np.sqrt(self.lambda_max)
        self.lts.to_backend("numpy")
        try:
            return stability_ratio(self.lts, tol=tol, **kw)
        finally:
            self.lts.to_backend(self.backend)

    # ------------------------------------------------ linear step (CPU)
    def _layout(self):
        sizes = [self._nb, self.n_free]
        if self.cpml is not None:
            sizes += [p["pf"].size for p in self._pml] + [p["pe"].size for p in self._pml]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    @property
    def n_state(self):
        return int(self._layout()[-1])

    def _split(self, x):
        o = self._layout()
        parts = [x[o[i] : o[i + 1]] for i in range(len(o) - 1)]
        b, e = parts[0], parts[1]
        psi = parts[2:5] if self.cpml is not None else []
        phi = parts[5:8] if self.cpml is not None else []
        return b, e, psi, phi

    def step_vec(self, x, i_all=None):
        """
        One step as a linear map of the packed state
        ``x = [b^{n-1/2}, e^n, psi_0..2, phi_0..2]`` and the source
        ``i^{n+1/2}`` on all edges (NumPy, either kernels).
        """
        if self.xp is not np:
            raise NotImplementedError("step_vec: NumPy backend only")
        dt = self.dt
        b, e, psi, phi = self._split(x)
        z = self._R(e)
        Kz = self._Kz(z).copy()
        psi1 = []
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                psi1.append(p["bh"] * psi[d] + p["ah"] * (p["Kf"] @ z))
                Kz[p["pf"]] += psi1[d]
        b1 = b - dt * Kz
        y = self._Wb * b1
        rhs = self._KHy(y).copy()
        phi1 = []
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                phi1.append(p["be"] * phi[d] + p["ae"] * (p["Ke"] @ y))
                rhs[p["pe"]] += phi1[d]
        if i_all is not None:
            rhs = rhs - self.P.conj().T @ i_all
        e1 = e + dt * self._Dinv * rhs
        return np.concatenate([b1, e1] + psi1 + phi1)

    def step_vec_T(self, y):
        """
        Exact (bilinear) transpose of `step_vec`: returns ``(xbar, ibar_all)``
        with ``<step(x, i), y> = <x, xbar> + <i, ibar>``.
        """
        if self.xp is not np:
            raise NotImplementedError("step_vec_T: NumPy backend only")
        dt = self.dt
        bb1, eb1, psib1, phib1 = self._split(y)
        ge = dt * self._Dinv * eb1
        # e1 = e + dt Dinv (KH Wb1 + sum T_d phi1_d - P^H i)
        ybar = self._KH_T(ge)
        phib = []
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                tot = phib1[d] + ge[p["pe"]]
                phib.append(p["be"] * tot)
                ybar = ybar + p["Ke"].T @ (p["ae"] * tot)
        b1bar = bb1 + self._Wb * ybar
        psib = []
        zbar = -dt * self._K_T(b1bar)
        if self.cpml is not None:
            for d, p in enumerate(self._pml):
                tot = psib1[d] - dt * b1bar[p["pf"]]
                psib.append(p["bh"] * tot)
                zbar = zbar + p["Kf"].T @ (p["ah"] * tot)
        ebar = eb1 + (zbar if self.lts is None else self.lts.apply_T(zbar))
        ibar = -(self.P.conj() @ ge)
        return np.concatenate([b1bar, ebar] + psib + phib), ibar

    def _K_T(self, y):
        """``K^T y`` (bilinear transpose of `_Kz`)."""
        if self.ops is not None:
            return self.ops.KH(y)  # real: K^T = K^H
        return self.K.T @ y

    def _KH_T(self, g):
        """``(K^H)^T g`` (bilinear transpose of `_KHy`)."""
        if self.ops is not None:
            return self.ops.K(g, np.empty(self._nb)).copy()
        return self.KH.T @ g

    # ------------------------------------------------------ eigen (small)
    def lts_eigenfrequencies(self, k, f_guess):
        """
        The `k` lowest time-discrete resonances [Hz] of the (LTS) leapfrog,
        ``sin(w dt / 2) = dt/2 sqrt(lambda(D^-1 S_0))`` (small meshes:
        builds ``S_0`` as a sparse matrix).
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla

        if self.lts is None:
            A, D = self.stiffness()
        else:
            self.lts.to_backend("numpy")
            A, D = self.lts.stiffness()
            self.lts.to_backend(self.backend)
        A = (0.5 * (A + A.conj().T)).tocsc()
        sigma = (2 * np.pi * 0.5 * f_guess) ** 2
        kk = min(A.shape[0] - 2, k + 8)
        lam = sla.eigsh(A, k=kk, M=sp.diags(D).tocsc(), sigma=sigma, which="LA",
                        ncv=min(A.shape[0] - 1, max(4 * kk, 40)), tol=1e-12,
                        return_eigenvectors=False)  # fmt: skip
        lam = np.sort(np.abs(lam))[:k]
        w = 2 / self.dt * np.arcsin(np.clip(0.5 * self.dt * np.sqrt(lam), 0, 1))
        return w / (2 * np.pi)
