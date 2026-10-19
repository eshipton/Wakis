# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
"""
Plane-wave total-field/scattered-field (TF/SF) box source.

The incident plane wave is computed on a one-dimensional auxiliary FIT line
that uses the 3D grid's primary and dual lengths along the propagation axis,
the same time step and the same time levels. The 3D update is corrected only
on the samples that couple across the box surface, with the discrete
equivalence-principle (commutator) terms

    H += -dt (M_h A - A M_e) e_inc,     E += dt (M_e B - B M_h) h_inc,

where A and B are the solver's curl operators (H -= dt A E, E += dt B H) and
M_e, M_h are 0/1 masks of the total-field samples. Because the auxiliary line
solves exactly the discrete equations the 3D grid solves for a field that is
uniform transversally, the incident field cancels to round-off outside the
box. See the physics guide, section "Total-field/scattered-field plane-wave
source".
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse import csr_matrix

_AXES = "xyz"


def _levi_civita(i, j, k):
    """epsilon_ijk for axis indices 0, 1, 2."""
    return int(np.sign((j - i) * (k - i) * (k - j)))


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class TFSFSource:
    def __init__(
        self,
        src_time,
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        direction="+z",
        polarization="x",
        amplitude=1.0,
        background_eps_r=1.0,
    ):
        """
        Plane wave injected on the surface of a total-field/scattered-field box.

        Inside the box the solver holds the total field (incident plus
        scattered), outside only the scattered field. The incident field is

            E_inc(r, t) = amplitude * s(t - (k.r - k.r0) / v) p,
            H_inc = k x E_inc / eta,

        with k the unit propagation vector, p the polarization axis, r0 the
        box face where the wave enters, v = c / sqrt(eps_r) and
        eta = sqrt(mu_0 / (eps_0 eps_r)). It is computed on an auxiliary 1D
        FIT line with the grid's lengths along the propagation axis, so the
        incident field outside the box cancels to round-off (exact discrete
        cancellation), also on grids graded along that axis.

        Parameters
        ----------
        src_time : GaussianSource, ContinuousSource or CustomSource
            Time profile s(t) [dimensionless].
        center : tuple of float
            Centre of the total-field box (x, y, z) [m].
        size : tuple of float or None
            Extent of the box [m]. Its faces snap to the nearest grid nodes
            (see `box_bounds`). The box surface must lie inside the non-PML
            region, in the homogeneous background. Along a periodic axis
            transverse to the propagation, None (or a size covering the whole
            domain) makes the box span the period, with no faces on that axis:
            e.g. 2D problems in a thin domain periodic along z, or normal
            incidence on a periodic structure.
        direction : str
            Propagation direction: '+x', '-x', '+y', '-y', '+z' or '-z'.
        polarization : str
            Axis of E_inc ('x', 'y' or 'z'), transverse to `direction`.
        amplitude : float
            Peak incident electric field for |s| = 1 [V/m].
        background_eps_r : float
            Relative permittivity of the (lossless, non-magnetic) background
            at the box surface. Default is 1 (vacuum).

        Notes
        -----
        Time levels follow the solver: after the step starting at t, E is at
        t + dt and H at t + dt/2. The H correction is added before the step
        (it is independent of the fields) and the E correction enters as an
        equivalent electric current in `solver.J` at t + dt/2, like
        `CurrentSource`. Axis-aligned incidence only; no MPI.
        """
        direction = str(direction).lower()
        if len(direction) != 2 or direction[0] not in "+-" or direction[1] not in _AXES:
            raise ValueError(
                f"direction must be one of '+x' ... '-z', got {direction!r}."
            )
        polarization = str(polarization).lower()[-1]
        if polarization not in _AXES:
            raise ValueError(
                f"polarization must be 'x', 'y' or 'z', got {polarization!r}."
            )
        if polarization == direction[1]:
            raise ValueError("polarization must be transverse to direction.")
        if background_eps_r <= 0:
            raise ValueError("background_eps_r must be > 0.")
        self.src_time = src_time
        self.center = np.asarray(center, dtype=float)
        # None (or inf) along an axis: span the whole (periodic) axis
        self.size = np.array([np.inf if v is None else v for v in size], dtype=float)
        if np.any(self.size <= 0):
            raise ValueError("size must be > 0 along every axis.")
        self.direction = direction
        self.polarization = polarization
        self.amplitude = float(amplitude)
        self.background_eps_r = float(background_eps_r)

        self.axis = _AXES.index(direction[1])  # a: propagation axis
        self.sign = 1.0 if direction[0] == "+" else -1.0  # along +a or -a
        self.p = _AXES.index(polarization)  # E_inc component
        self.q = 3 - self.axis - self.p  # H_inc component
        # Faraday on the line: mu dH_q/dt = -s dE_p/da, s = eps_{q a p}
        self.s = _levi_civita(self.q, self.axis, self.p)
        self.v = c_light / np.sqrt(self.background_eps_r)
        self.eta = np.sqrt(mu_0 / (eps_0 * self.background_eps_r))
        # H_q = sign * s * E_p / eta for a wave along sign * a
        self.h_factor = self.sign * self.s / self.eta
        self.end_time = getattr(src_time, "end_time", np.inf)

        self.n_abs = 96  # absorbing cells at each end of the auxiliary line
        self.n_pad = 6  # plain cells between the box and the absorbers
        self.is_first_update = True
        # Test hooks (rule 1: the leakage test must be able to fail)
        self._mutation_time_shift = 0  # use e_inc^(n+k) instead of e_inc^n
        self._mutation_index_offset = 0  # shift the E_inc columns only
        self._mutation_index_shift_both = 0  # shift E_inc and H_inc columns
        self._apply_corrections = True  # False: only advance the incident line

    # ------------------------------------------------------------------ setup
    def _snap_box(self, solver):
        grid = solver.grid
        nodes = (grid.x, grid.y, grid.z)
        n_samples = (solver.Nx, solver.Ny, solver.Nz)
        lo_req = self.center - self.size / 2
        hi_req = self.center + self.size / 2
        self._idx = []  # (i_lo, i_hi) node indices, None for a full periodic axis
        bounds = []
        for d in range(3):
            pos = nodes[d][: n_samples[d]]  # sample nodes (no ghost)
            periodic = solver.bc_low[d].lower() in ("periodic", "bloch")
            tol = 1e-9 * (nodes[d][-1] - nodes[d][0])
            spans = lo_req[d] <= nodes[d][0] + tol and hi_req[d] >= nodes[d][-1] - tol
            if periodic and d != self.axis and spans:
                # The box spans the period: no faces, no corrections on this axis
                self._idx.append(None)
                bounds.append((nodes[d][0], nodes[d][-1]))
                continue
            if not np.isfinite(self.size[d]):
                raise ValueError(
                    f"size along {_AXES[d]} is None/inf: only allowed for a periodic "
                    "axis transverse to the propagation."
                )
            if periodic and d == self.axis:
                raise NotImplementedError("The propagation axis cannot be periodic.")
            i_lo = int(np.argmin(np.abs(pos - lo_req[d])))
            i_hi = int(np.argmin(np.abs(pos - hi_req[d])))
            if i_hi <= i_lo:
                raise ValueError(f"TF/SF box spans no cell along {_AXES[d]}.")
            margin_lo, margin_hi = 1, 1
            for side, bc in (("lo", solver.bc_low[d]), ("hi", solver.bc_high[d])):
                if bc.lower() in ("pml", "cpml"):
                    n = int(getattr(solver, "n_pml", 10)) + 1
                    if side == "lo":
                        margin_lo = n
                    else:
                        margin_hi = n
            if i_lo < margin_lo or i_hi > n_samples[d] - 1 - margin_hi:
                raise ValueError(
                    f"TF/SF box along {_AXES[d]} (nodes {i_lo}..{i_hi}) must lie inside "
                    f"the non-PML region (nodes {margin_lo}..{n_samples[d] - 1 - margin_hi})."
                )
            self._idx.append((i_lo, i_hi))
            bounds.append((pos[i_lo], pos[i_hi]))
        self._bounds = tuple(bounds)

        # Membership of node and mid positions along each axis
        self._in_node, self._in_mid = [], []
        for d in range(3):
            n = n_samples[d]
            if self._idx[d] is None:
                self._in_node.append(np.ones(n, bool))
                self._in_mid.append(np.ones(n, bool))
                continue
            i_lo, i_hi = self._idx[d]
            idx = np.arange(n)
            self._in_node.append((idx >= i_lo) & (idx <= i_hi))
            self._in_mid.append((idx >= i_lo) & (idx < i_hi))  # mid of [i, i+1]

    def _build_line(self, solver):
        """Auxiliary 1D line: coordinates, lengths and update coefficients."""
        grid = solver.grid
        a = self.axis
        nodes = (grid.x, grid.y, grid.z)[a]
        n_a = (solver.Nx, solver.Ny, solver.Nz)[a]
        i_lo, i_hi = self._idx[a]
        c0 = max(0, i_lo - 3)
        c1 = min(n_a - 1, i_hi + 3)
        core = np.asarray(nodes[c0 : c1 + 1], dtype=float)
        dl_core = np.diff(core)  # bit-identical to the 3D primary lengths
        h_lo, h_hi = dl_core[0], dl_core[-1]
        n_ext = self.n_abs + self.n_pad
        dl = np.concatenate([np.full(n_ext, h_lo), dl_core, np.full(n_ext, h_hi)])
        coords = core[0] - h_lo * np.arange(n_ext, 0, -1)
        coords = np.concatenate(
            [coords, core, core[-1] + h_hi * np.arange(1, n_ext + 1)]
        )
        self._offset = n_ext - c0  # line node m = 3D node i + offset
        n_line = coords.size
        # Dual lengths centred on the nodes, as GridFIT3D._compute_dual_lengths
        tl = 0.5 * (np.concatenate([[dl[0]], dl]) + np.concatenate([dl, [dl[-1]]]))

        # Graded matched absorber (sigma_m / mu = sigma_e / eps) at both ends
        eps = eps_0 * self.background_eps_r
        L_abs_lo, L_abs_hi = self.n_abs * h_lo, self.n_abs * h_hi
        order, R = 3, 1e-12
        x_lo_in = coords[self.n_abs]  # inner edge of the low absorber
        x_hi_in = coords[n_line - 1 - self.n_abs]

        def sigma_e(x):
            depth = (
                np.maximum(x_lo_in - x, 0) / L_abs_lo
                + np.maximum(x - x_hi_in, 0) / L_abs_hi
            )
            L = np.where(x < x_lo_in, L_abs_lo, L_abs_hi)
            smax = -(order + 1) * np.log(R) / (2 * self.eta * L)
            return smax * np.clip(depth, 0, 1) ** order

        mid = 0.5 * (coords[1:] + coords[:-1])
        se = sigma_e(coords)
        sm = sigma_e(mid) * mu_0 / eps
        dt = float(solver.dt)
        ieps, imu = 1.0 / eps, 1.0 / mu_0
        ae = se * dt / (2 * eps)
        am = sm * dt / (2 * mu_0)
        self._line = {
            "coords": coords,
            "dl": dl,
            "tl": tl,
            "ce": (1 - ae) / (1 + ae),
            "de": dt * ieps * self.s / (1 + ae) / tl,
            "ch": (1 - am) / (1 + am),
            "dh": dt * imu * self.s / (1 + am) / dl,
        }
        # In the plain region the coefficients are those of the solver step
        plain_e = se == 0
        plain_h = sm == 0
        self._line["de"][plain_e] = dt * ieps * self.s / tl[plain_e]
        self._line["dh"][plain_h] = dt * imu * self.s / dl[plain_h]

        # Entry plane and injection node. The injection boundary of the line
        # lies between nodes m_inj -/+ 1 and m_inj: one cell upstream of the
        # entry plane is the closest position at which H_inc half a cell
        # outside the box (needed by the E correction) is still a plain
        # (total-field) sample of the line.
        if self.sign > 0:
            self._m_entry = i_lo + self._offset
            self._m_inj = self._m_entry - 1
        else:
            self._m_entry = i_hi + self._offset
            self._m_inj = self._m_entry + 1
        self._xi0 = coords[self._m_entry]
        self._n_line = n_line

    def _couplings(self, solver):
        """
        Correction matrices from the commutators of the curl with the masks.

        Returns K_H (H rows x line E nodes) with c_H = K_H e_line, and K_J
        (E rows x line H samples) with J_corr = K_J h_line.
        """
        grid = solver.grid
        N = (solver.Nx, solver.Ny, solver.Nz)
        Ntot = N[0] * N[1] * N[2]
        a, p, q = self.axis, self.p, self.q
        In, Im = self._in_node, self._in_mid
        L = _host(grid.L.toarray()).reshape(3, -1)
        iA = _host(grid.iA.toarray()).reshape(3, -1)
        tL = _host(grid.tL.toarray()).reshape(3, -1)
        itA = _host(grid.itA.toarray()).reshape(3, -1)
        imu = _host(solver.imu.toarray()).reshape(3, -1)
        ieps = _host(solver.ieps.toarray()).reshape(3, -1)
        sigma = _host(solver.sigma.toarray()).reshape(3, -1)
        dt = float(solver.dt)
        off = self._offset
        off_e = self._offset + self._mutation_index_offset  # test hook
        off_h = self._offset + self._mutation_index_shift_both

        def flat(I, J, K):
            return I + N[0] * J + N[0] * N[1] * K

        def grid3(arrs):
            return np.meshgrid(*arrs, indexing="ij")

        # ---- H corrections: rows H_r (r != p), columns E_p, d = third axis
        rows, cols, vals = [], [], []
        for r in range(3):
            if r == p:
                continue
            d = 3 - r - p
            s = _levi_civita(r, d, p)
            for delta, csign in ((0, -s), (1, +s)):
                n_d = N[d] - delta
                # H mid along d vs E node along d at n_d + delta
                diff_d = Im[d][:n_d].astype(int) - In[d][delta : delta + n_d].astype(
                    int
                )
                cand_d = np.flatnonzero(diff_d)
                if cand_d.size == 0:
                    continue
                cand_r = np.flatnonzero(In[r])
                cand_p = np.flatnonzero(Im[p])
                axes = [None, None, None]
                axes[r], axes[p], axes[d] = cand_r, cand_p, cand_d
                I = grid3(axes)
                Dd = diff_d[I[d]]  # M_h - M_e
                h_flat = r * Ntot + flat(*I)
                e_idx = [I[0].copy(), I[1].copy(), I[2].copy()]
                e_idx[d] = e_idx[d] + delta
                e_flat = p * Ntot + flat(*e_idx)
                coef = (
                    -dt
                    * Dd
                    * iA[r][flat(*I)]
                    * imu[r][flat(*I)]
                    * csign
                    * L[p][flat(*e_idx)]
                )
                rows.append(h_flat.ravel())
                cols.append((e_idx[a] + off_e + off_h - off).ravel())
                vals.append(coef.ravel())
                # Material check on both sides of the surface
                self._check_bg(
                    ieps[p][flat(*e_idx)], imu[r][flat(*I)], sigma[p][flat(*e_idx)]
                )
        K_H = self._assemble(rows, cols, vals, self._n_line)

        # ---- E corrections (as J): rows E_R (R != q), columns H_q
        rows, cols, vals = [], [], []
        for R in range(3):
            if R == q:
                continue
            d = 3 - R - q
            s = _levi_civita(q, d, R)
            # E_R at m couples to H_q at m (coef -s) and at m - e_d (coef +s)
            for delta, csign in ((0, -s), (1, +s)):
                n_d = N[d] - delta
                # E node along d at m_d (m_d >= delta) vs H mid at m_d - delta
                diff_d = In[d][delta : delta + n_d].astype(int) - Im[d][:n_d].astype(
                    int
                )
                cand_d = np.flatnonzero(diff_d) + delta
                if cand_d.size == 0:
                    continue
                axes = [None, None, None]
                axes[R], axes[q], axes[d] = (
                    np.flatnonzero(Im[R]),
                    np.flatnonzero(In[q]),
                    cand_d,
                )
                I = grid3(axes)
                Dd = diff_d[I[d] - delta]  # M_e - M_h
                e_flat = R * Ntot + flat(*I)
                h_idx = [I[0].copy(), I[1].copy(), I[2].copy()]
                h_idx[d] = h_idx[d] - delta
                # J = -(M_e - M_h) itA C^T tL h_inc
                coef = -Dd * itA[R][flat(*I)] * csign * tL[q][flat(*h_idx)]
                rows.append(e_flat.ravel())
                cols.append((h_idx[a] + off_h).ravel())
                vals.append(coef.ravel())
                self._check_bg(
                    ieps[R][flat(*I)], imu[q][flat(*h_idx)], sigma[R][flat(*I)]
                )
        K_J = self._assemble(rows, cols, vals, self._n_line - 1)
        return K_H, K_J

    def _check_bg(self, ieps, imu, sigma):
        ieps0 = 1.0 / (eps_0 * self.background_eps_r)
        if ieps.size and (
            np.max(np.abs(ieps / ieps0 - 1)) > 1e-6
            or np.max(np.abs(imu * mu_0 - 1)) > 1e-6
            or np.max(np.abs(sigma)) > 0
        ):
            raise ValueError(
                "The TF/SF box surface must lie in the homogeneous lossless "
                f"background (eps_r={self.background_eps_r}, mu_r=1, sigma=0); "
                "move or enlarge the box, or set background_eps_r."
            )

    @staticmethod
    def _assemble(rows, cols, vals, n_cols):
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        cols = np.concatenate(cols) if cols else np.zeros(0, int)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        uniq, inv = np.unique(rows, return_inverse=True)
        K = csr_matrix((vals, (inv, cols)), shape=(uniq.size, n_cols))
        K.sum_duplicates()
        return uniq, K

    def _initialize(self, solver):
        if solver.use_mpi:
            raise NotImplementedError("TFSFSource does not support MPI yet.")
        if np.any(getattr(solver, "k_bloch", np.zeros(3)) != 0):
            raise NotImplementedError("TFSFSource does not support Bloch phases.")
        self._snap_box(solver)
        self._build_line(solver)
        (h_rows, K_H), (j_rows, K_J) = self._couplings(solver)
        xp = solver.E.xp
        self.xp = xp
        real = np.finfo(solver.E.array.dtype).dtype.type
        self._real = real
        L = self._line
        self._dev = {k: xp.asarray(L[k], dtype=real) for k in ("ce", "de", "ch", "dh")}
        self._e = xp.zeros(self._n_line, dtype=real)
        self._h = xp.zeros(self._n_line - 1, dtype=real)
        if solver.use_gpu:
            import cupyx.scipy.sparse as cusp

            self._K_H = cusp.csr_matrix(K_H.astype(real))
            self._K_J = cusp.csr_matrix(K_J.astype(real))
        else:
            self._K_H = K_H.astype(real)
            self._K_J = K_J.astype(real)
        self._h_rows = xp.asarray(h_rows)
        self._j_rows = xp.asarray(j_rows)
        self._J_old = xp.zeros(j_rows.size, dtype=solver.J.array.dtype)
        self._dt = float(solver.dt)
        # Entry-plane time series: E at t + dt, H (averaged onto the node) at t + dt/2
        self._series = xp.zeros((2, 1024), dtype=real)
        self._n_steps = 0
        self._t_last = -np.inf
        # Injection coefficients of the 1D TF/SF boundary
        m = self._m_inj
        if self.sign > 0:  # TF: nodes >= m; boundary between m-1 and m
            self._inj = {
                "h_idx": m - 1,
                "e_idx": m,
                "h_pos": 0.5 * (L["coords"][m - 1] + L["coords"][m]),
            }
            self._inj["h_coef"] = +self._dt / mu_0 * self.s / L["dl"][m - 1]
            self._inj["e_coef"] = (
                +self._dt / (eps_0 * self.background_eps_r) * self.s / L["tl"][m]
            )
        else:  # TF: nodes <= m; boundary between m and m+1
            self._inj = {
                "h_idx": m,
                "e_idx": m,
                "h_pos": 0.5 * (L["coords"][m] + L["coords"][m + 1]),
            }
            self._inj["h_coef"] = -self._dt / mu_0 * self.s / L["dl"][m]
            self._inj["e_coef"] = (
                -self._dt / (eps_0 * self.background_eps_r) * self.s / L["tl"][m]
            )
        self._inj["e_pos"] = L["coords"][m]

    # ------------------------------------------------------------- time step
    def _e_analytic(self, x, t):
        return self.amplitude * float(
            self.src_time(t - self.sign * (x - self._xi0) / self.v)
        )

    def _advance_h(self, t):
        """H^(n-1/2) -> H^(n+1/2) on the line, with E^n at t = n dt."""
        e, h, D = self._e, self._h, self._dev
        h *= D["ch"]
        h -= D["dh"] * (e[1:] - e[:-1])
        inj = self._inj
        h[inj["h_idx"]] += inj["h_coef"] * self._e_analytic(inj["e_pos"], t)

    def _advance_e(self, t):
        """E^n -> E^(n+1) on the line, with H^(n+1/2)."""
        e, h, D = self._e, self._h, self._dev
        curl = self.xp.zeros_like(e)
        curl[1:-1] = h[1:] - h[:-1]
        e *= D["ce"]
        e -= D["de"] * curl
        e[0] = 0.0
        e[-1] = 0.0
        inj = self._inj
        h_inc = self.h_factor * self._e_analytic(inj["h_pos"], t + self._dt / 2)
        e[inj["e_idx"]] += inj["e_coef"] * h_inc

    def update(self, solver, t):
        """
        Advance the incident field and add the TF/SF corrections for the step
        that starts at time t.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver object.
        t : float
            Time at the start of the step [s].
        """
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
        if (
            getattr(solver, "fused", False)
            and not self._mutation_time_shift
            and t >= self._t_last
        ):
            self._fused_update(solver, t)
            return
        if t < self._t_last:  # a new run with the same source object
            self._e[:] = 0.0
            self._h[:] = 0.0
            self._J_old[:] = 0.0
            self._n_steps = 0
        self._t_last = t

        # H correction from e_inc^n (independent of the 3D fields: added
        # before the step, equivalent to -dt imu M with M = -n x E_inc)
        if self._mutation_time_shift:
            e_save = self._e.copy()
            h_save = self._h.copy()
            self._advance_h(t)
            self._advance_e(t)
            e_used = self._e
            self._e, self._h = e_save, h_save
        else:
            e_used = self._e
        if self._apply_corrections:
            solver.H.array[self._h_rows] += self._K_H @ e_used

        self._advance_h(t)  # h_inc^(n+1/2)
        # E correction as the equivalent electric current J = n x H_inc at t + dt/2
        if self._apply_corrections:
            J_new = self._K_J @ self._h
            solver.J.array[self._j_rows] += J_new - self._J_old
            self._J_old = J_new
        self._advance_e(t)  # e_inc^(n+1)

        # Entry-plane record
        n = self._n_steps
        if n >= self._series.shape[1]:
            grown = self.xp.zeros(
                (2, 2 * self._series.shape[1]), dtype=self._series.dtype
            )
            grown[:, :n] = self._series
            self._series = grown
        m = self._m_entry
        self._series[0, n] = self._e[m]
        self._series[1, n] = 0.5 * (self._h[m - 1] + self._h[m])
        self._n_steps = n + 1

    # ------------------------------------------------- fused GPU time step
    def _fused_scalars(self, t):
        """Injection terms of the line at the step started at t (see update)."""
        inj = self._inj
        s_h = inj["h_coef"] * self._e_analytic(inj["e_pos"], t)
        h_inc = self.h_factor * self._e_analytic(inj["h_pos"], t + self._dt / 2)
        return s_h, inj["e_coef"] * h_inc

    def _fused_prepare(self, solver):
        from .gpu_fused import idx32

        if getattr(self, "_fdev", None) is None:
            self._fdev = {
                "h_rows": idx32(self._h_rows),
                "j_rows": idx32(self._j_rows),
                "KH": (idx32(self._K_H.indptr), idx32(self._K_H.indices),
                       self._K_H.data.astype(np.float64)),
                "KJ": (idx32(self._K_J.indptr), idx32(self._K_J.indices),
                       self._K_J.data.astype(np.float64)),
            }  # fmt: skip
        return self._fdev

    def _fused_launch(self, solver, s_h=0.0, s_e=0.0, n=0, tables=None, step=None):
        """The update as five kernels (host scalars, or tables + device step)."""
        from .gpu_fused import launch, tname

        d = self._fused_prepare(solver)
        e, h, D = self._e, self._h, self._dev
        T = tname(solver.H.array.dtype)
        th, te = (None, None) if tables is None else tables
        zero = np.int64(0)
        if self._apply_corrections:
            ptr, col, val = d["KH"]
            nr = d["h_rows"].size
            launch(f"csr_add<{T}>", nr, np.int32(nr), d["h_rows"], ptr, col, val,
                   e, solver.H.array)  # fmt: skip
        launch("line_h", h.size, np.int32(h.size), h, e, D["ch"], D["dh"],
               np.int32(self._inj["h_idx"]), np.float64(s_h), th, step,
               zero)  # fmt: skip
        if self._apply_corrections:
            ptr, col, val = d["KJ"]
            nr = d["j_rows"].size
            launch(f"csr_delta<{T}>", nr, np.int32(nr), d["j_rows"], ptr, col,
                   val, h, self._J_old, solver.J.array)  # fmt: skip
        launch("line_e", e.size, np.int32(e.size), e, h, D["ce"], D["de"],
               np.int32(self._inj["e_idx"]), np.float64(s_e), te, step,
               zero)  # fmt: skip
        launch(
            "line_series",
            1,
            e,
            h,
            np.int32(self._m_entry),
            self._series,
            np.int64(self._series.shape[1]),
            np.int64(n),
            step,
            zero,
        )

    def _fused_update(self, solver, t):
        """`update` with fused kernels (fused GPU solvers)."""
        self._t_last = t
        n = self._n_steps
        if n >= self._series.shape[1]:
            self._grow_series(2 * self._series.shape[1])
        s_h, s_e = self._fused_scalars(t)
        self._fused_launch(solver, s_h, s_e, n)
        self._n_steps = n + 1

    def _grow_series(self, cap):
        grown = self.xp.zeros((2, cap), dtype=self._series.dtype)
        grown[:, : self._n_steps] = self._series[:, : self._n_steps]
        self._series = grown

    def fused_loop_op(self, solver, Nt):
        """
        (launch(step), finish(n1)) for `wakis.gpu_fused.FusedLoop`: the update
        of every step n >= current, with the injection terms tabulated.
        """
        if self._mutation_time_shift or not self.xp.__name__.startswith("cupy"):
            return None
        if self._series.shape[1] < Nt:
            self._grow_series(int(Nt))
        sc = np.array([self._fused_scalars(n * solver.dt) for n in range(Nt)])
        tables = (self.xp.asarray(sc[:, 0]), self.xp.asarray(sc[:, 1]))

        def run(step):
            self._fused_launch(solver, tables=tables, step=step)

        def finish(n1):
            self._n_steps = n1
            self._t_last = (n1 - 1) * solver.dt

        return run, finish

    # --------------------------------------------------------------- outputs
    def box_bounds(self):
        """
        Box actually used on the grid.

        Returns
        -------
        tuple
            ((xmin, xmax), (ymin, ymax), (zmin, zmax)) [m], on grid nodes.
            Available after the first update, or after `prepare(solver)`.
        """
        if not hasattr(self, "_bounds"):
            raise RuntimeError("Call prepare(solver) or run the solver first.")
        return self._bounds

    def prepare(self, solver):
        """Set up the source on `solver` without stepping (e.g. to place monitors)."""
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
        return self

    def total_field_mask(self, solver, field="E", flat=False):
        """
        Boolean mask of the total-field samples.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver the source is (or will be) attached to.
        field : str
            'E' or 'H' ('J' is colocated with 'E').
        flat : bool
            False: shape (3, Nx, Ny, Nz). True: the layout of `Field.array`
            (components stacked, Fortran order), to index `solver.E.array`.

        Returns
        -------
        ndarray of bool
        """
        self.prepare(solver)
        In, Im = self._in_node, self._in_mid
        mask = np.zeros((3, solver.Nx, solver.Ny, solver.Nz), dtype=bool)
        for c in range(3):
            parts = []
            for d in range(3):
                mid = (d == c) if field.upper() in ("E", "J") else (d != c)
                parts.append(Im[d] if mid else In[d])
            mask[c] = (
                parts[0][:, None, None]
                & parts[1][None, :, None]
                & parts[2][None, None, :]
            )
        if flat:
            return np.concatenate([mask[c].ravel(order="F") for c in range(3)])
        return mask

    def _times(self):
        n = np.arange(self._n_steps)
        return (n + 1) * self._dt, (n + 0.5) * self._dt

    def incident_fields(self, frequencies):
        """
        Spectra of the incident field at the entry plane.

        Same convention as `DFTMonitor`/`FluxMonitor`:
        F(f) = sum_n F(t_n) exp(-2j pi f t_n) dt, with E at (n+1) dt and H at
        (n+1/2) dt.

        Parameters
        ----------
        frequencies : array_like
            Frequencies [Hz].

        Returns
        -------
        E_inc, H_inc : ndarray of complex
            Component of E along `polarization` [V/m s] and component of H
            along the third axis (k x p up to sign, see `h_component`) [A/m s]
            at the entry plane (H averaged from the two adjacent dual
            samples, as `FluxMonitor` does).
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        series = _host(self._series[:, : self._n_steps]).astype(float)
        tE, tH = self._times()
        kE = self._dt * np.exp(-2j * np.pi * f[:, None] * tE[None, :])
        kH = self._dt * np.exp(-2j * np.pi * f[:, None] * tH[None, :])
        return kE @ series[0], kH @ series[1]

    @property
    def h_component(self):
        """Axis ('x', 'y' or 'z') of the incident H field."""
        return _AXES[self.q]

    def incident_flux_density(self, frequencies):
        """
        Incident flux spectrum per unit area, Phi_inc(f) = Re(E_inc H_inc^*).k

        With the `FluxMonitor` convention, sigma_sca(f) =
        flux_scat.get_flux() / incident_flux_density(f) [m^2].

        Parameters
        ----------
        frequencies : array_like
            Frequencies [Hz].

        Returns
        -------
        ndarray
            [J/(Hz^2 m^2)].
        """
        E, H = self.incident_fields(frequencies)
        # S_a = eps_{p q a} E_p H_q = s E_p H_q; along k: sign * s
        return self.sign * self.s * np.real(E * np.conj(H))
