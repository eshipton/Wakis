"""
Independent reference for the ring showcase: a body-of-revolution (BOR)
finite-difference eigensolver in (r, z) for fields ~ exp(i m phi).

The ring's cross-section (a 500 x 220 nm rectangle at R = 1.5 um) is aligned
with the (r, z) Yee grid, so every material interface lies on grid lines and
the arithmetic (tangential) averaging at the interface nodes is the exact
second-order treatment; there is no staircasing. Richardson extrapolation of
three grids gives the reference resonance of each azimuthal order m.

Grid (cylindrical Yee): E_r(i+1/2, j), E_phi(i, j), E_z(i, j+1/2),
H_r(i, j+1/2), H_phi(i+1/2, j+1/2), H_z(i+1/2, j) on r_i = r0 + i d,
z_j = j d. Curl with d/dphi -> i m:

    (curl E)_r   = i m E_z / r - dE_phi/dz
    (curl E)_phi = dE_r/dz - dE_z/dr
    (curl E)_z   = [d(r E_phi)/dr - i m E_r] / r

Boundaries: PEC on r = r0 (the fields of m ~ 14 vanish like (r/R)^m there),
r = r_max and z = z_max; at z = 0 PMC (even TE-like modes) or PEC. The
generalized problem curl mu^-1 curl E = (w/c)^2 eps E is solved by
shift-invert ARPACK on the (non-Hermitian) sparse operator.

Usage::

    python bor_reference.py validate          # coaxial cavity vs Bessel roots
    python bor_reference.py ring --d 5 4 2.5  # ring resonances, m = 13..16
"""

import argparse
import json
import sys

import numpy as np
import scipy.sparse as sp
from scipy.constants import c
from scipy.sparse.linalg import eigs  # noqa: F401


def _d1(n):
    """(n-1) x n forward difference."""
    return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))


class BOR:
    def __init__(self, r0, r1, z1, d, m, eps_func, z0_bc="pmc"):
        self.r0, self.d, self.m = r0, d, m
        Nr = int(round((r1 - r0) / d))
        Nz = int(round(z1 / d))
        self.Nr, self.Nz = Nr, Nz
        ri = r0 + d * np.arange(Nr + 1)  # integer r
        rh = r0 + d * (np.arange(Nr) + 0.5)  # half r
        zi = d * np.arange(Nz + 1)
        zh = d * (np.arange(Nz) + 0.5)
        I = sp.identity
        im = 1j * m
        # E: Er (Nr, Nz+1), Ep (Nr+1, Nz+1), Ez (Nr+1, Nz) -- index (i, j) C order
        # H: Hr (Nr+1, Nz), Hp (Nr, Nz), Hz (Nr, Nz+1)
        Dz_n = _d1(Nz + 1) / d  # (Nz x Nz+1): nodes -> half
        Dr_n = _d1(Nr + 1) / d  # (Nr x Nr+1)
        # curl E -> H positions
        Hr_Ez = sp.kron(sp.diags(im / ri), I(Nz))
        Hr_Ep = -sp.kron(I(Nr + 1), Dz_n)
        Hp_Er = sp.kron(I(Nr), Dz_n)
        Hp_Ez = -sp.kron(Dr_n, I(Nz))
        Hz_Ep = sp.kron(sp.diags(1 / rh) @ Dr_n @ sp.diags(ri), I(Nz + 1))
        Hz_Er = sp.kron(sp.diags(-im / rh), I(Nz + 1))
        Z = sp.csr_matrix
        nEr, nEp, nEz = Nr * (Nz + 1), (Nr + 1) * (Nz + 1), (Nr + 1) * Nz
        nHr, nHp, nHz = (Nr + 1) * Nz, Nr * Nz, Nr * (Nz + 1)
        CE = sp.bmat(
            [
                [Z((nHr, nEr)), Hr_Ep, Hr_Ez],
                [Hp_Er, Z((nHp, nEp)), Hp_Ez],
                [Hz_Er, Hz_Ep, Z((nHz, nEz))],
            ]
        ).tocsr()
        # curl H -> E positions; tangential H outside z in [0, z1] is the
        # mirror image: PMC at z = 0 (odd H_t: H(-1/2) = -H(1/2)), and at a
        # PEC plane the tangential E is fixed anyway
        Dz_h = sp.lil_matrix((Nz + 1, Nz))  # half -> nodes
        for j in range(Nz + 1):
            if j < Nz:
                Dz_h[j, j] += 1 / d
            if j > 0:
                Dz_h[j, j - 1] -= 1 / d
        if z0_bc == "pmc":
            Dz_h[0, 0] = 2 / d
        Dz_h = Dz_h.tocsr()
        Dr_h = sp.lil_matrix((Nr + 1, Nr))  # half -> integer r
        for i in range(Nr + 1):
            if i < Nr:
                Dr_h[i, i] += 1 / d
            if i > 0:
                Dr_h[i, i - 1] -= 1 / d
        Dr_h = Dr_h.tocsr()
        Er_Hz = sp.kron(sp.diags(im / rh), I(Nz + 1))
        Er_Hp = -sp.kron(I(Nr), Dz_h)
        Ep_Hr = sp.kron(I(Nr + 1), Dz_h)
        Ep_Hz = -sp.kron(Dr_h, I(Nz + 1))
        Ez_Hp = sp.kron(sp.diags(1 / np.maximum(ri, 1e-300)) @ Dr_h @ sp.diags(rh), I(Nz))
        Ez_Hr = sp.kron(sp.diags(-im / ri), I(Nz))
        CH = sp.bmat(
            [
                [Z((nEr, nHr)), Er_Hp, Er_Hz],
                [Ep_Hr, Z((nEp, nHp)), Ep_Hz],
                [Ez_Hr, Ez_Hp, Z((nEz, nHz))],
            ]
        ).tocsr()
        # permittivity at the E nodes: average of the (up to 4) cells
        # around the node in the (r, z) plane (exact tangential averaging for
        # grid-aligned interfaces; normal components sit inside one cell)
        rc = r0 + d * (np.arange(Nr) + 0.5)
        zc = d * (np.arange(Nz) + 0.5)
        Ecell = eps_func(rc[:, None], zc[None, :])  # (Nr, Nz)
        Epad = np.pad(Ecell, ((1, 1), (1, 1)), mode="edge")
        if z0_bc == "pmc":
            Epad[:, 0] = Epad[:, 1]  # mirror at z = 0
        # Er at (i+1/2, j): cells (i, j-1), (i, j)
        eEr = 0.5 * (Epad[1:-1, :-1] + Epad[1:-1, 1:])
        # Ep at (i, j): cells (i-1..i, j-1..j)
        eEp = 0.25 * (Epad[:-1, :-1] + Epad[1:, :-1] + Epad[:-1, 1:] + Epad[1:, 1:])
        # Ez at (i, j+1/2): cells (i-1, j), (i, j)
        eEz = 0.5 * (Epad[:-1, 1:-1] + Epad[1:, 1:-1])
        eps = np.r_[eEr.ravel(), eEp.ravel(), eEz.ravel()]
        # PEC: tangential E on r = r0, r1 (Ep, Ez at i = 0, Nr) and z = z1
        # (Er, Ep at j = Nz), and z = 0 if PEC
        fixed = np.zeros(eps.size, bool)
        Ep_ix = np.arange(nEp).reshape(Nr + 1, Nz + 1) + nEr
        Ez_ix = np.arange(nEz).reshape(Nr + 1, Nz) + nEr + nEp
        Er_ix = np.arange(nEr).reshape(Nr, Nz + 1)
        for ix in (Ep_ix[0], Ep_ix[-1], Ez_ix[0], Ez_ix[-1], Er_ix[:, -1], Ep_ix[:, -1]):
            fixed[np.ravel(ix)] = True
        if z0_bc == "pec":
            fixed[Er_ix[:, 0]] = True
            fixed[Ep_ix[:, 0]] = True
        free = np.flatnonzero(~fixed)
        A = sp.diags(1 / eps) @ CH @ CE
        self.A = A[free][:, free].tocsc()
        self.n = free.size

    def modes(self, k_target, n=6):
        """Wavenumbers k = w / c nearest to k_target [1/m] (real parts)."""
        from scipy.sparse.linalg import LinearOperator, splu

        sig = k_target**2
        lu = splu((self.A - sig * sp.identity(self.n, format="csc")).tocsc())
        op = LinearOperator((self.n, self.n), matvec=lu.solve, dtype=complex)
        mu = eigs(op, k=n, which="LM", return_eigenvectors=False, tol=1e-12)
        k = np.sqrt((sig + 1 / mu).astype(complex))
        return np.sort(k.real[np.abs(k.imag) < 1e-6 * np.abs(k)])


def validate():
    """Coaxial PEC cavity (r0, r1, PEC z ends): TM_mnp, TE_mnp vs Bessel roots."""
    from scipy.optimize import brentq
    from scipy.special import jv, jvp, yv, yvp

    r0, r1, h, m = 0.5, 1.0, 0.6, 3
    out = {}
    for d in (0.02, 0.01):
        b = BOR(r0, r1, h, d, m, lambda r, z: np.ones(np.broadcast(r, z).shape), z0_bc="pec")
        out[d] = b.modes(9.0, 8)

    def roots(f):
        ks = np.linspace(0.5, 40, 20000)
        v = f(ks)
        return [brentq(f, ks[i], ks[i + 1]) for i in np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:]))]

    tm = roots(lambda k: jv(m, k * r0) * yv(m, k * r1) - jv(m, k * r1) * yv(m, k * r0))
    te = roots(lambda k: jvp(m, k * r0) * yvp(m, k * r1) - jvp(m, k * r1) * yvp(m, k * r0))
    ex = sorted([np.hypot(kt, p * np.pi / h) for kt in tm for p in range(0, 4)]
                + [np.hypot(kt, p * np.pi / h) for kt in te for p in range(1, 4)])
    ex = np.array(ex)
    for d, k in out.items():
        err = [np.min(np.abs(ex - kk)) / kk for kk in k]
        print(f"d = {d}: k = {np.round(k, 4)}  rel. err = {np.array2string(np.array(err), precision=1)}")
    return out, ex


def ring(ds_nm, ms=(13, 14, 15, 16)):
    import ring_study as R

    def eps(r, z):
        core = (r > R.R_IN) & (r < R.R_OUT) & (np.abs(z) < R.T / 2)
        return np.where(core, R.EPS_SI, R.EPS_OX)

    res = {}
    for m in ms:
        res[m] = {}
        for d in ds_nm:
            b = BOR(0.5e-6, 3.0e-6, 1.0e-6, d * 1e-9, m, eps)
            # TE-like fundamental near 1.55 um: n_eff ~ 2.4 -> f ~ m c / (2 pi R n_eff)
            k0 = 2 * np.pi * R.F0 / c
            k = b.modes(k0, 8)
            res[m][d] = (k * c / (2 * np.pi)).tolist()
            print(m, d, b.n, np.round(np.array(res[m][d]) / 1e12, 4), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "ring"])
    ap.add_argument("--d", type=float, nargs="+", default=[10.0, 5.0, 2.5])
    ap.add_argument("--out", default="data/bor_reference.json")
    args = ap.parse_args(argv)
    if args.cmd == "validate":
        validate()
    else:
        res = ring(args.d)
        with open(args.out, "w") as fh:
            json.dump({str(m): {str(d): v for d, v in r.items()} for m, r in res.items()}, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
