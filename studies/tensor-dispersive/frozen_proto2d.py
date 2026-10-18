"""
Prototype: 2D TE Cartesian Yee/FIT eigen-solver for a lossless Drude rod in a
square PEC box [-1,1]^2, comparing averaging schemes at interface pixels.
k^2 D = C^T C N(k) D, E = N(k) D, N real symmetric at real k (lossless).
Reference: unstructured FIT (conforming triangles) + Richardson.
"""

import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

CENTER = np.array([0.031, -0.017])
B = 0.5


def eps_drude(k, kp):
    return 1 - (kp / k) ** 2


def box_fraction(x0, x1, y0, y1, b=B, n_gauss=48):
    """Exact-ish area fraction of circle in boxes (vectorized, Gauss in x with
    splitting at the circle's x-extent)."""
    x0, x1, y0, y1 = map(np.atleast_1d, (x0, x1, y0, y1))
    cx, cy = CENTER
    a0 = np.clip(x0, cx - b, cx + b)
    a1 = np.clip(x1, cx - b, cx + b)
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    out = np.zeros(x0.shape)
    # split interval further at points where cy +- s(x) crosses y0, y1
    pts = [a0, a1]
    for yy in (y0, y1):
        d = yy - cy
        s = np.sqrt(np.clip(b * b - d * d, 0, None))
        for sgn in (-1, 1):
            xc = cx + sgn * s
            pts.append(np.clip(xc, a0, a1))
    P = np.sort(np.stack(pts, 1), 1)
    for k in range(P.shape[1] - 1):
        u0, u1 = P[:, k], P[:, k + 1]
        xs = 0.5 * (u1 - u0)[:, None] * (xg[None] + 1) + u0[:, None]
        s = np.sqrt(np.clip(b * b - (xs - cx) ** 2, 0, None))
        length = np.clip(
            np.minimum(y1[:, None], cy + s) - np.maximum(y0[:, None], cy - s), 0, None
        )
        out += (length * wg[None]).sum(1) * 0.5 * (u1 - u0)
    return out / ((x1 - x0) * (y1 - y0))


def normal_at(px, py):
    c = np.stack([px - CENTER[0], py - CENTER[1]], 1)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


class Grid:
    def __init__(self, n):
        self.n = n
        h = self.h = 2.0 / n
        x = self.x = np.linspace(-1, 1, n + 1)
        # x-edges (i, j): i = 0..n-1, j = 1..n-1 ; y-edges (i, j): i = 1..n-1, j=0..n-1
        I, J = np.meshgrid(np.arange(n), np.arange(1, n), indexing="ij")
        self.ex = np.stack([I.ravel(), J.ravel()], 1)
        I, J = np.meshgrid(np.arange(1, n), np.arange(n), indexing="ij")
        self.ey = np.stack([I.ravel(), J.ravel()], 1)
        self.nx_e = len(self.ex)
        self.ne = self.nx_e + len(self.ey)
        self.idx_x = -np.ones((n, n + 1), int)
        self.idx_x[self.ex[:, 0], self.ex[:, 1]] = np.arange(self.nx_e)
        self.idx_y = -np.ones((n + 1, n), int)
        self.idx_y[self.ey[:, 0], self.ey[:, 1]] = self.nx_e + np.arange(len(self.ey))
        # curl: cell (i, j): (Ey(i+1,j) - Ey(i,j) - Ex(i,j+1) + Ex(i,j)) / h
        rows, cols, vals = [], [], []
        ci, cj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ci, cj = ci.ravel(), cj.ravel()
        cell = ci * n + cj
        for e, s in (
            (self.idx_y[ci + 1, cj], 1),
            (self.idx_y[ci, cj], -1),
            (self.idx_x[ci, cj + 1], -1),
            (self.idx_x[ci, cj], 1),
        ):
            ok = e >= 0
            rows.append(cell[ok])
            cols.append(e[ok])
            vals.append(np.full(ok.sum(), s / h))
        self.C = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n * n, self.ne),
        )
        # edge pixels (dual cells)
        xe = x[self.ex[:, 0]]
        ye = x[self.ex[:, 1]]
        fx = box_fraction(xe, xe + h, ye - h / 2, ye + h / 2)
        nxv = normal_at(xe + h / 2, ye)
        xe = x[self.ey[:, 0]]
        ye = x[self.ey[:, 1]]
        fy = box_fraction(xe - h / 2, xe + h / 2, ye, ye + h)
        nyv = normal_at(xe, ye + h / 2)
        self.f_e = np.concatenate([fx, fy])
        self.n_e = np.concatenate([nxv, nyv])
        self.comp = np.r_[np.zeros(self.nx_e, int), np.ones(self.ne - self.nx_e, int)]
        mid = np.concatenate(
            [
                np.stack([x[self.ex[:, 0]] + h / 2, x[self.ex[:, 1]]], 1),
                np.stack([x[self.ey[:, 0]], x[self.ey[:, 1]] + h / 2], 1),
            ]
        )
        self.inside_mid = np.hypot(*(mid - CENTER).T) < B
        # quadrants: cell (i,j), corner (a,b): x-edge (i, j+b), y-edge (i+a, j)
        q = []
        for a in (0, 1):
            for bb in (0, 1):
                exq = self.idx_x[ci, cj + bb]
                eyq = self.idx_y[ci + a, cj]
                xc, yc = x[ci + a], x[cj + bb]
                xm, ym = x[ci] + h / 2, x[cj] + h / 2
                fq = box_fraction(
                    np.minimum(xc, xm),
                    np.maximum(xc, xm),
                    np.minimum(yc, ym),
                    np.maximum(yc, ym),
                )
                nq = normal_at((xc + xm) / 2, (yc + ym) / 2)
                # node-centred box (z-edge pixel) at the corner
                fz = box_fraction(xc - h / 2, xc + h / 2, yc - h / 2, yc + h / 2)
                q.append((exq, eyq, fq, nq, fz))
        self.quads = q


def tensors(f, n, z1, z2, e1, e2):
    zn = (1 - f) * z1 + f * z2
    et = (1 - f) * e1 + f * e2
    zt = 1 / et
    T = zt[:, None, None] * (np.eye(2)[None] - np.einsum("ci,cj->cij", n, n))
    T += zn[:, None, None] * np.einsum("ci,cj->cij", n, n)
    return T


def build_N(g, k, kp, scheme):
    import os

    e1, e2 = (
        1.0,
        (float(os.environ["EPS2"]) if "EPS2" in os.environ else eps_drude(k, kp)),
    )
    z1, z2 = 1 / e1, 1 / e2
    ne = g.ne
    if scheme == "stair":
        return sp.diags(np.where(g.inside_mid, z2, z1))
    Te = tensors(g.f_e, g.n_e, z1, z2, e1, e2)
    diag = Te[np.arange(ne), g.comp, g.comp]
    if scheme == "diag":
        return sp.diags(diag)
    rows, cols, vals = [np.arange(ne)], [np.arange(ne)], []
    if scheme == "edge_tensor":
        vals.append(diag)
        for exq, eyq, fq, nq, fz in g.quads:
            ok = (exq >= 0) & (eyq >= 0)
            a, b = exq[ok], eyq[ok]
            t = 0.5 * (Te[a, 0, 1] + Te[b, 0, 1]) / 4
            rows += [a, b]
            cols += [b, a]
            vals += [t, t]
    else:  # quadrant-pixel tensors
        vals.append(np.zeros(ne))
        for exq, eyq, fq, nq, fz in g.quads:
            if scheme == "quad_exact":
                ff, nn = fq, nq
            elif scheme == "quad_mean2":
                ff, nn = (
                    0.5 * (g.f_e[np.maximum(exq, 0)] + g.f_e[np.maximum(eyq, 0)]),
                    nq,
                )
                ff = np.where(exq < 0, g.f_e[np.maximum(eyq, 0)], ff)
                ff = np.where(eyq < 0, g.f_e[np.maximum(exq, 0)], ff)
            elif scheme == "quad_Tavg":
                ff, nn = fq, nq
            elif scheme == "quad_mean3":
                fa = g.f_e[np.maximum(exq, 0)]
                fb = g.f_e[np.maximum(eyq, 0)]
                ff = (fa + fb + fz) / 3
                nn = nq
            if scheme == "quad_Tavg":
                a_, b_ = np.maximum(exq, 0), np.maximum(eyq, 0)
                T = 0.5 * (Te[a_] + Te[b_]) / 4
                T = np.where(
                    ((exq >= 0) & (eyq >= 0))[:, None, None],
                    T,
                    np.where((exq >= 0)[:, None, None], Te[a_], Te[b_]) / 4,
                )
            else:
                T = tensors(ff, nn, z1, z2, e1, e2) / 4
            for p, ep in ((0, exq), (1, eyq)):
                for qq, eq in ((0, exq), (1, eyq)):
                    ok = (ep >= 0) & (eq >= 0)
                    rows.append(ep[ok])
                    cols.append(eq[ok])
                    vals.append(T[ok, p, qq])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ne, ne),
    )


def solve(g, kp, k_guess, scheme, n_ev=6, width=0.04):
    from fixedpoint import fixed_points

    A0 = (g.C.T @ g.C).tocsr()

    def modes_at(k):
        A = (A0 @ build_N(g, k, kp, scheme)).tocsc()
        w = eigs(
            A, k=n_ev, sigma=k * k, which="LM", tol=1e-13, return_eigenvectors=False
        )
        w = w.real
        return np.sqrt(np.abs(w)) * np.sign(w)

    fp = fixed_points(modes_at, k_guess - width, k_guess + width, n=17)
    if not fp:
        return np.nan
    return fp[np.argmin(abs(np.array(fp) - k_guess))]


def eigpairs(g, kp, k, scheme, n_ev):
    A = (g.C.T @ g.C @ build_N(g, k, kp, scheme)).tocsc()
    w, v = eigs(A, k=n_ev, sigma=k * k, which="LM", tol=1e-13)
    return w.real, v.real if np.allclose(v.imag, 0) else v


def solve_tracked(g, kp, k_guess, scheme, ref_vec, n_ev=12, verbose=False):
    """Mode with maximal overlap with ref_vec (H-field curl of D is the
    natural comparison; we use the H_z = C E vector)."""
    from scipy.optimize import brentq

    info = {}

    def sel(k):
        w, v = eigpairs(g, kp, k, scheme, n_ev)
        N = build_N(g, k, kp, scheme)
        H = g.C @ (N @ v)
        H /= np.linalg.norm(H, axis=0)
        ov = np.abs(H.conj().T @ ref_vec)
        i = np.argmax(ov)
        info["ov"] = ov[i]
        info["near"] = np.sort(np.sqrt(np.abs(w)))
        return np.sqrt(abs(w[i])) - k

    ks = np.linspace(k_guess - 0.03, k_guess + 0.03, 7)
    vals = [sel(k) for k in ks]
    if verbose:
        print(scheme, np.round(vals, 5), "ov", info["ov"])
    for i in range(len(ks) - 1):
        if vals[i] * vals[i + 1] < 0:
            r = brentq(sel, ks[i], ks[i + 1], xtol=1e-12)
            return r, info["ov"]
    return np.nan, info.get("ov")


def solve_old(g, kp, k_guess, scheme, n_ev=6, tol=1e-11):
    A0 = (g.C.T @ g.C).tocsr()
    prev = None

    def lam(k):
        nonlocal prev
        A = (A0 @ build_N(g, k, kp, scheme)).tocsc()
        w, v = eigs(A, k=n_ev, sigma=k * k, which="LM", tol=1e-12)
        w = w.real
        if prev is None:
            i = np.argmin(abs(w - k * k))
        else:
            ov = np.abs(v.conj().T @ prev) / np.linalg.norm(v, axis=0)
            i = np.argmax(ov)
        prev = v[:, i] / np.linalg.norm(v[:, i])
        return w[i]

    k0, k1 = k_guess, k_guess * (1 + 1e-4)
    g0 = lam(k0) - k0 * k0
    for _ in range(40):
        g1 = lam(k1) - k1 * k1
        if abs(g1) < tol * k1 * k1:
            return k1
        k0, k1, g0 = k1, k1 - g1 * (k1 - k0) / (g1 - g0), g1
    return k1


if __name__ == "__main__":
    kp = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    kref = float(sys.argv[2])
    ns = [int(a) for a in sys.argv[3].split(",")]
    schemes = sys.argv[4].split(",")
    for n in ns:
        g = Grid(n)
        ks = solve(g, kp, kref, "stair")
        w, v = eigpairs(g, kp, ks, "stair", 6)
        i = np.argmin(abs(np.sqrt(abs(w)) - ks))
        ref = g.C @ (build_N(g, ks, kp, "stair") @ v[:, i])
        ref /= np.linalg.norm(ref)
        out = [f"stair={ks / kref - 1:+.3e}"]
        for s in schemes:
            k, ov = solve_tracked(g, kp, ks, s, ref, verbose="-v" in sys.argv)
            out.append(f"{s}={k / kref - 1:+.3e}(ov {ov:.2f})")
        print(n, " ".join(out), flush=True)
