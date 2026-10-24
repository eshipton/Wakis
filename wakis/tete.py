# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
"""
Explicit TE/TM time stepping without numerical dispersion along z.

Opt-in alternative to the Yee leapfrog of `SolverFIT3D`, selected with
``SolverFIT3D(..., scheme="tete")``. It is the explicit TE/TM scheme of
M. Dohlus and I. Zagorodnov, J. Comput. Phys. 228, 2822 (2009)
(arXiv:0901.1053), the explicit successor of the implicit TE/TM scheme of
I. Zagorodnov and T. Weiland, J. Comput. Phys. 207, 69 (2005) and
Phys. Rev. ST Accel. Beams 8, 042001 (2005).

Field components are split into the subsystems (relative to z)

* TE: ex, ey, hz, stored at half-integer times (n + 1/2) dt,
* TM: hx, hy, ez, stored at integer times n dt (Ez is sampled at the same
  times as in the Yee scheme, so the wake post-processing is unchanged).

With A = ``tDsiDmuiDaC`` and B = ``itDaiDepsDstC`` split into their
longitudinal (Pz) and transverse (Px, Py) couplings, one step is::

    l_E  = B_L h_xy^n                                (+ z-CPML psi, TFSF)
    e_xy += dt/2 (B_T hz + l_E - ieps (J + sigma e))  # TE half kick
    hz   -= dt A_Z e_xy                               # TE kick
    e_xy += dt/2 (B_T hz + l_E - ieps (J + sigma e))  # TE half kick
    l_H  = A_L e_xy^{n+1/2}                           (+ z-CPML psi, TFSF)
    h_xy -= dt/2 (A_T ez + l_H)                       # TM half kick
    ez   += dt (B_Z h_xy - ieps (J + sigma ez))       # TM kick
    h_xy -= dt/2 (A_T ez + l_H)                       # TM half kick

i.e. a leapfrog between TE and TM in which each subsystem's transverse
dynamics is integrated by a symmetric (Stormer-Verlet) splitting. In vacuum
on a uniform grid its dispersion relation is (Dohlus & Zagorodnov, eq. 27)

    sin^2(w dt/2)/(c dt)^2 = sin^2(Kz)/dz^2
        + (sin^2(Kx)/dx^2 + sin^2(Ky)/dy^2) (1 - (c dt/dz)^2 sin^2(Kz)),

K_i = k_i d_i / 2: at the "magic" step c dt = dz waves along z travel at
exactly c for all kz. The scheme is stable if (eq. 25)

    c dt <= dz   and   c dt <= (dx^-2 + dy^-2)^(-1/2),

which for general (conformal, material) operators reads
dt <= 2 / sqrt(lambda_max(K)) for the three operators K = B_L A_L,
B_Z A_T and A_Z B_T; the solver certifies dt with Gershgorin row-sum
bounds of these operators (`gershgorin_limits`).

Every sub-update is linear with a transposable sparse operator, so the step
has an exact Hermitian transpose (`TeteTranspose`, used by
`wakis.inverse.AdjointStepper`).
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigs

SCHEMES = ("yee", "tete")

# Relative tolerance for "uniform" z spacing (magic time step well defined)
_UNIFORM_RTOL = 1e-9


def check_supported(solver):
    """
    Raise NotImplementedError for solver features the TE/TM step lacks.

    Called by `SolverFIT3D` after the boundary conditions and materials are
    set, before the time step is chosen.
    """
    reasons = []
    for d, (lo, hi) in enumerate(zip(solver.bc_low, solver.bc_high)):
        for bc in (lo.lower(), hi.lower()):
            if bc == "cpml" and d < 2:
                reasons.append(
                    "CPML on x/y faces (the psi recursion is not defined for the "
                    "split transverse kicks; use PEC/PMC/periodic walls)"
                )
            if bc in ("pml", "abc"):
                reasons.append(f"'{bc}' boundaries (use 'cpml' on the z faces)")
    if solver.use_sibc:
        reasons.append("SIBC")
    if solver.source_type == "tfsf" and not solver.activate_cpml:
        reasons.append("plane-wave TFSF injection without CPML")
    if reasons:
        raise NotImplementedError(
            "scheme='tete' does not support: " + "; ".join(sorted(set(reasons)))
        )


def check_supported_late(solver):
    """Checks that need the fully initialized solver (end of __init__)."""
    reasons = []
    if getattr(solver, "tensor_operator", None) is not None:
        reasons.append("tensor averaging (conformal='tensor')")
    if (
        getattr(solver, "_dispersive_groups", None)
        or getattr(solver, "_dispersive_h_groups", None)
        or getattr(solver, "dispersive_network", None)
    ):
        reasons.append("dispersive (ADE) materials")
    if getattr(solver, "_gain_groups", None):
        reasons.append("gain (ADE) materials")
    if reasons:
        raise NotImplementedError(
            "scheme='tete' does not support: " + "; ".join(reasons)
        )


def magic_dt(solver):
    """
    Magic time step dt = dz / c [s] (smallest dz on non-uniform z meshes).

    Prints a warning when dz is not uniform: the dispersion-free property
    then holds only in the cells with the smallest dz.
    """
    dz = np.asarray(solver.grid.dz, dtype=float)
    if solver.verbose and np.ptp(dz) > _UNIFORM_RTOL * dz.max():
        print(
            "[!] scheme='tete': non-uniform dz; dt = min(dz)/c is dispersion-free "
            "only in the cells with dz = min(dz)."
        )
    return float(dz.min() / c_light)


def _host(a):
    return a.get() if hasattr(a, "get") else a


def _lam_max(B, A, tol=1e-8):
    """Largest eigenvalue of B A (real, >= 0: similar to a PSD matrix)."""
    n = A.shape[1]
    if n == 0 or A.nnz == 0 or B.nnz == 0:
        return 0.0
    op = LinearOperator((n, n), matvec=lambda x: B @ (A @ x), dtype=A.dtype)
    try:
        lam = eigs(op, k=1, which="LM", tol=tol, return_eigenvectors=False,
                   maxiter=20 * n)[0]  # fmt: skip
    except Exception:  # pragma: no cover - ARPACK failure: power iteration
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        lam = 0.0
        for _ in range(2000):
            y = op.matvec(x)
            lam_new = np.vdot(x, y).real / np.vdot(x, x).real
            x = y / np.linalg.norm(y)
            if abs(lam_new - lam) < tol * abs(lam_new):
                break
            lam = lam_new
    return float(abs(lam))


def stability_limits(solver, tol=1e-8):
    """
    Largest stable dt [s] of each sub-operator, from Arnoldi eigenvalues.

    Diagnostic (tests, small grids): the solver certifies its time step with
    the cheaper upper bounds of `gershgorin_limits`.

    Returns
    -------
    dict
        ``{'longitudinal': ..., 'tm': ..., 'te': ...}`` with
        2 / sqrt(lambda_max(K)) for K = B_L A_L, B_Z A_T and B_T A_Z
        (host copies of the operators of ``solver._tete``).
    """
    op = solver._tete
    h = {k: _host_sparse(v) for k, v in op.items() if k in _OP_NAMES}
    lim = {}
    for name, (b, a) in {
        "longitudinal": ("LE", "LH"),
        "tm": ("ZE", "TH"),
        "te": ("TE", "ZH"),
    }.items():
        lam = _lam_max(h[b], h[a], tol)
        lim[name] = np.inf if lam == 0 else 2.0 / np.sqrt(lam * (1 + 10 * tol))
    return lim


_OP_NAMES = ("LH", "TH", "ZH", "LE", "TE", "ZE")

# (name, rows of B or A, other factor) of the three sub-operators
_PAIRS = {"longitudinal": ("LE", "LH"), "tm": ("ZE", "TH"), "te": ("ZH", "TE")}


def gershgorin_limits(ops):
    """
    Certified stable dt [s] of each sub-operator from Gershgorin row sums.

    lambda_max(K) <= max_i (|X| |Y| 1)_i for K = X Y (any induced norm
    bounds the spectral radius), so dt <= 2 / sqrt(max row sum) is stable.
    The bound is tight on uniform staircase grids (it reproduces eq. (25) of
    Dohlus & Zagorodnov) and for the longitudinal operator with the
    trapezoidal conformal face areas (rows sum to 4 c^2/dz^2).
    """
    lim = {}
    for name, (x, y) in _PAIRS.items():
        X, Y = abs(_host_sparse(ops[x])), abs(_host_sparse(ops[y]))
        r = X @ (Y @ np.ones(Y.shape[1]))
        rmax = float(r.max()) if r.size else 0.0
        lim[name] = np.inf if rmax == 0 else 2.0 / np.sqrt(rmax)
    return lim


def _host_sparse(m):
    if hasattr(m, "get"):
        return m.get()
    return m


def build_operators(solver):
    """
    Split the curl operators into the six TE/TM blocks (scipy CSR, host).

    Convention: H -= dt (A e), E += dt (B h), blocks on component slices
    xy = [0, 2N), z = [2N, 3N):

    =====  ===================  ====================================
    name   maps                 coupling
    =====  ===================  ====================================
    LH     e_xy -> h_xy         longitudinal (Pz), A[xy, xy]
    TH     e_z  -> h_xy         transverse, A[xy, z]
    ZH     e_xy -> h_z          transverse, A[z, xy]
    LE     h_xy -> e_xy         longitudinal (Pz^H), B[xy, xy]
    TE     h_z  -> e_xy         transverse, B[xy, z]
    ZE     h_xy -> e_z          transverse, B[z, xy]
    =====  ===================  ====================================

    Without CPML the blocks are cut from ``tDsiDmuiDaC`` (A) and
    ``itDaiDepsDstC`` (B); with CPML they are assembled from the CPML
    derivative operators (``dxy`` ...) with imu and ieps folded in. The
    blocks A[z, z] and B[z, z] must vanish (component-block-diagonal
    material and face operators); otherwise a ValueError is raised.
    """
    N = solver.N
    xy, z = slice(0, 2 * N), slice(2 * N, 3 * N)
    if solver.activate_cpml:
        imu = [diags(np.asarray(_host(getattr(solver.imu, f"field_{d}"))))
               for d in "xyz"]  # fmt: skip
        ieps = [diags(np.asarray(_host(getattr(solver.ieps, f"field_{d}"))))
                for d in "xyz"]  # fmt: skip
        s = solver
        A = bmat(
            [
                [None, -imu[0] @ s.dxz, imu[0] @ s.dxy],
                [imu[1] @ s.dyz, None, -imu[1] @ s.dyx],
                [-imu[2] @ s.dzy, imu[2] @ s.dzx, None],
            ],
            format="csr",
        )
        B = bmat(
            [
                [None, -ieps[0] @ s.dtxz, ieps[0] @ s.dtxy],
                [ieps[1] @ s.dtyz, None, -ieps[1] @ s.dtyx],
                [-ieps[2] @ s.dtzy, ieps[2] @ s.dtzx, None],
            ],
            format="csr",
        )
    else:
        A = csr_matrix(solver.tDsiDmuiDaC)
        B = csr_matrix(solver.itDaiDepsDstC)
    A = A.astype(solver.dtype)
    B = B.astype(solver.dtype)
    for name, M in (("A", A), ("B", B)):
        zz = M[z, z]
        zz.eliminate_zeros()
        if zz.nnz:
            raise ValueError(
                f"scheme='tete': operator {name} couples z to z components "
                "(non block-diagonal face/material operator); not supported."
            )
    ops = {
        "LH": A[xy, xy],
        "TH": A[xy, z],
        "ZH": A[z, xy],
        "LE": B[xy, xy],
        "TE": B[xy, z],
        "ZE": B[z, xy],
    }
    for k in ops:
        ops[k] = csr_matrix(ops[k])
        ops[k].eliminate_zeros()
    return ops


def stabilize_transverse(solver, ops, dt, which=("ez", "hz"), margin=1e-3):
    """
    Bound the transverse TE/TM operators by Gershgorin row sums (conformal).

    Small open faces normal to x and y (not redistributed with
    scheme='tete') make the TM operator K = B_Z A_T large on the e_z edges
    next to them, and small z faces do the same for the TE operator
    K = A_Z B_T on h_z. The rows of B_Z (A_Z) are scaled by
    s = min(1, (2/dt)^2 (1 - margin) / r), with r the absolute row sum
    bound |B_Z| |A_T| 1 (|A_Z| |B_T| 1), together with ieps_z (imu_z) of
    those edges (faces). This raises the "mass" of e_z edges and h_z faces
    only: both vanish in the self-field of a v = c bunch in a uniform pipe
    (TEM, e_z = h_z = 0, A_Z e_xy = 0), which therefore stays an exact
    discrete solution, and the longitudinal operator is not touched.
    Gershgorin's theorem then gives lambda_max <= (2/dt)^2 (1 - margin).
    `which` selects the e_z rows ('ez') and/or h_z rows ('hz'); the solver
    scales 'hz' only when the TE limit is below dt (the z faces keep the
    redistribution of the Yee path, whose absolute row sums overestimate
    the eigenvalue).

    Returns
    -------
    dict
        Number of scaled e_z edges and h_z faces and the smallest factors.
    """
    budget = (2.0 / dt) ** 2 * (1.0 - margin)
    N = solver.N
    info = {}
    for rows, other, field, label in (("ZE", "TH", solver.ieps, "ez"),
                                      ("ZH", "TE", solver.imu, "hz")):  # fmt: skip
        if label not in which:
            continue
        r = abs(ops[rows]) @ (abs(ops[other]) @ np.ones(ops[other].shape[1]))
        scale = np.ones(N)
        hot = r > budget
        scale[hot] = budget / r[hot]
        if hot.any():
            ops[rows] = diags(scale) @ ops[rows]
            ops[rows] = csr_matrix(ops[rows])
            field.field_z = np.asarray(_host(field.field_z)) * scale
        info[f"{label}_scaled"] = int(hot.sum())
        info[f"{label}_min_factor"] = float(scale.min())
    return info


def _cpml_faces(solver):
    """z-CPML faces present on this solver (rank): (side, idx, bH, cH, bE, cE)."""
    faces = []
    for side, bcs in (("low", solver.bc_low), ("high", solver.bc_high)):
        if bcs[2].lower() != "cpml":
            continue
        idx = np.asarray(_host(getattr(solver, f"idx_z_{side}")))
        faces.append(
            (
                side,
                np.concatenate([idx, idx + solver.N]),
                np.tile(np.asarray(_host(getattr(solver, f"pml_b_H_z_{side}"))), 2),
                np.tile(np.asarray(_host(getattr(solver, f"pml_c_H_z_{side}"))), 2),
                np.tile(np.asarray(_host(getattr(solver, f"pml_b_E_z_{side}"))), 2),
                np.tile(np.asarray(_host(getattr(solver, f"pml_c_E_z_{side}"))), 2),
            )
        )
    return faces


def install(solver, ops):
    """
    Store the TE/TM operators (moved to the solver's device) and the z-CPML
    auxiliary arrays on the solver.

    The psi arrays are ``solver.tete_psiH_<side>`` and
    ``solver.tete_psiE_<side>`` (x and y parts concatenated). They hold the
    imu- (ieps-) scaled convolution of the signed longitudinal curl terms,
    i.e. -imu_x psiHb_x and imu_y psiHa_y of the Yee CPML step.
    """
    xp = solver.E.xp
    if solver.use_gpu:
        from cupyx.scipy.sparse import csr_matrix as gpu_csr

        dev = {k: gpu_csr(v) for k, v in ops.items()}
    else:
        dev = dict(ops)
    faces = []
    for side, idx2, bH, cH, bE, cE in _cpml_faces(solver):
        faces.append(
            (side, xp.asarray(idx2), xp.asarray(bH), xp.asarray(cH),
             xp.asarray(bE), xp.asarray(cE))  # fmt: skip
        )
        if not hasattr(solver, f"tete_psiH_{side}"):
            setattr(solver, f"tete_psiH_{side}", xp.zeros(idx2.size, solver.dtype))
            setattr(solver, f"tete_psiE_{side}", xp.zeros(idx2.size, solver.dtype))
    dev["faces"] = faces
    solver._tete = dev
    solver._tete_host = ops  # host blocks (stability analysis, transposes)


def psi_names(solver):
    """Names of the TE/TM CPML auxiliary arrays, in step order."""
    names = []
    for side, *_ in solver._tete["faces"]:
        names += [f"tete_psiE_{side}", f"tete_psiH_{side}"]
    return names


def _sigma_parts(solver):
    """(sigma*ieps) on xy and z, or None without conductivity."""
    if not solver.use_conductivity:
        return None, None
    N = solver.N
    s = solver.sigma.toarray() * solver.ieps.toarray()
    return s[: 2 * N], s[2 * N :]


def one_step(solver):
    """One TE/TM time step (see module docstring), in place."""
    s = solver
    if s.step_0:
        s._set_ghosts_to_0()
        s.step_0 = False
        s._attrcleanup()
        s._tete_sigma = _sigma_parts(s)
    if s.M is not None:
        raise NotImplementedError("scheme='tete' does not support magnetic currents.")
    op = s._tete
    N = s.N
    dt = s.dt
    half = 0.5 * dt
    E, H, J = s.E.array, s.H.array, s.J.array
    exy, ez = E[: 2 * N], E[2 * N :]
    hxy, hz = H[: 2 * N], H[2 * N :]
    ieps = s.ieps.toarray()
    ieps_xy, ieps_z = ieps[: 2 * N], ieps[2 * N :]
    sig_xy, sig_z = s._tete_sigma
    tfsf = s.source_type == "tfsf" and not s.injection_done

    # ---- TE subsystem (ex, ey, hz): n-1/2 -> n+1/2 with h_xy at n --------
    l_e = op["LE"] @ hxy
    if tfsf:
        l_e[:N] -= s.ieps.field_x * (s.tf_dtxz @ s.H_trans.field_y)
        l_e[N:] += s.ieps.field_y * (s.tf_dtyz @ s.H_trans.field_x)
    for side, idx2, _, _, bE, cE in op["faces"]:
        name = f"tete_psiE_{side}"
        psi = bE * getattr(s, name) + cE * l_e[idx2]
        setattr(s, name, psi)
        l_e[idx2] += psi
    jxy = J[: 2 * N]
    for _ in range(2):
        inc = op["TE"] @ hz
        inc += l_e
        inc -= ieps_xy * jxy
        if sig_xy is not None:
            inc -= sig_xy * exy
        inc *= half
        exy += inc
        if _ == 0:
            hz -= dt * (op["ZH"] @ exy)
    if s.use_mpi:
        s._mpi_communicate(s.E)

    # ---- TM subsystem (hx, hy, ez): n -> n+1 with e_xy at n+1/2 ----------
    l_h = op["LH"] @ exy
    if tfsf:
        l_h[:N] += s.imu.field_x * (s.tf_dxz @ s.E_trans.field_y)
        l_h[N:] -= s.imu.field_y * (s.tf_dyz @ s.E_trans.field_x)
    for side, idx2, bH, cH, _, _ in op["faces"]:
        name = f"tete_psiH_{side}"
        psi = bH * getattr(s, name) + cH * l_h[idx2]
        setattr(s, name, psi)
        l_h[idx2] += psi
    hxy -= half * (op["TH"] @ ez + l_h)
    inc = op["ZE"] @ hxy
    inc -= ieps_z * J[2 * N :]
    if sig_z is not None:
        inc -= sig_z * ez
    inc *= dt
    ez += inc
    hxy -= half * (op["TH"] @ ez + l_h)
    if s.use_mpi:
        s._mpi_communicate(s.H)


class TeteTranspose:
    """
    Exact Hermitian transpose of the source-free TE/TM step.

    State x = (H, E, psi) with the z-CPML auxiliaries of `psi_names`. The
    forward step is the composition of the linear maps listed in `one_step`;
    `step` applies their Hermitian transposes in reverse order.
    """

    def __init__(self, solver):
        if solver.use_mpi and solver.grid.use_mpi:
            raise NotImplementedError(
                "The TE/TM adjoint step is not implemented with MPI."
            )
        from .inverse.adjoint import _hermitian

        self.solver = solver
        self.xp = xp = solver.E.xp
        self.N = solver.N
        self.dt = solver.dt
        ops = solver._tete_host
        self.t = {k: _hermitian(v, solver) for k, v in ops.items()}
        self.faces = solver._tete["faces"]
        N = self.N
        if solver.use_conductivity:
            sig = xp.asarray(solver.sigma.toarray()) * xp.asarray(solver.ieps.toarray())
            self.loss_xy = 1.0 - 0.5 * self.dt * sig[: 2 * N]
            self.loss_z = 1.0 - self.dt * sig[2 * N :]
        else:
            self.loss_xy = self.loss_z = None
        self.psi_names = psi_names(solver)

    def _mv(self, name, x):
        op = self.t[name]
        if self.solver.imported_mkl and not self.solver.use_gpu:
            from sparse_dot_mkl import dot_product_mkl

            return dot_product_mkl(op, x)
        return op @ x

    def step(self, lam):
        """Adjoint variables before the step from those after it (dict)."""
        xp, N, dt = self.xp, self.N, self.dt
        half = 0.5 * dt
        lH = lam["H"].copy()
        lE = lam["E"].copy()
        out = {}
        lhxy, lhz = lH[: 2 * N], lH[2 * N :]
        lexy, lez = lE[: 2 * N], lE[2 * N :]

        # TM block, reversed: M3, M2, M1, then the l_H / psiH map
        l_lh = -half * lhxy  # from M3
        lez += self._mv("TH", -half * lhxy)
        lhxy += dt * self._mv("ZE", lez)  # M2: ez' = loss ez + dt ZE hxy
        if self.loss_z is not None:
            lez *= self.loss_z
        l_lh += -half * lhxy  # M1
        lez += self._mv("TH", -half * lhxy)
        for side, idx2, bH, cH, _, _ in self.faces:
            name = f"tete_psiH_{side}"
            lpsi = lam[name] + l_lh[idx2]
            l_lh[idx2] += xp.conj(cH) * lpsi
            out[name] = xp.conj(bH) * lpsi
        lexy += self._mv("LH", l_lh)

        # TE block, reversed: T3, T2, T1, then the l_E / psiE map
        l_le = half * lexy  # T3
        lhz += self._mv("TE", half * lexy)
        if self.loss_xy is not None:
            lexy *= self.loss_xy
        lexy -= dt * self._mv("ZH", lhz)  # T2: hz' = hz - dt ZH exy
        l_le += half * lexy  # T1
        lhz += self._mv("TE", half * lexy)
        if self.loss_xy is not None:
            lexy *= self.loss_xy
        for side, idx2, _, _, bE, cE in self.faces:
            name = f"tete_psiE_{side}"
            lpsi = lam[name] + l_le[idx2]
            l_le[idx2] += xp.conj(cE) * lpsi
            out[name] = xp.conj(bE) * lpsi
        lhxy += self._mv("LE", l_le)
        out["H"] = lH
        out["E"] = lE
        return out
