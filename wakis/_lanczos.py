# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Largest eigenvalue of the leapfrog update operator, for the stable time step
of the conformal (and tensor-averaged) FIT operators.

The source-free update H' = H - dt A E, E' = E + dt B H' is stable for
dt < 2 / sqrt(lambda_max(B A)). With the material operator N on the E edges
(diagonal ieps, or ieps plus the tensor couplings) and the diagonal metric
matrices,

    B A = N itDa C^H tDs G C Ds,

where G = iDa imu (diagonal, or block-symmetric for merged faces). The
nonzero eigenvalues of B A equal those of P K (cyclic permutation), with

    P = Ds N itDa = itDa (W ieps + S) itDa,    K = C^H tDs G C,

W = diag(L tA) the dual volumes and S the symmetric tensor couplings, so that
P and K are Hermitian positive semi-definite. P K is self-adjoint in the
P-semi-inner product, and the Lanczos recurrence for P^{1/2} K P^{1/2}
needs only products with P and K [e.g. Saad, *Numerical Methods for Large
Eigenvalue Problems*, 2nd ed., SIAM 2011, ch. 6 and 9 (generalized
Lanczos)]:

    beta_{j+1} r_{j+1} = K z_j - alpha_j r_j - beta_j r_{j-1},
    z_j = P r_j,   alpha_j = z_j^H K z_j,   beta_{j+1}^2 = r~^H P r~.

The Ritz values (eigenvalues of the tridiagonal matrix of the alpha, beta)
converge to lambda_max from below; iterations stop when the residual
estimate beta_{m+1} |s_m| is below tol * theta (the ARPACK criterion). Only
a few vectors are stored (no Krylov basis) and the sparse products run
multi-threaded on row blocks (SciPy releases the GIL in them), with the
same per-row summation order as the single-threaded product.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import eigh_tridiagonal


def default_threads():
    """Threads for the sparse products: OMP_NUM_THREADS, else the CPU count (max 8)."""
    env = os.environ.get("OMP_NUM_THREADS", "")
    n = int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)
    return max(1, min(n, 8))


_POOL = []


def _pool():
    """Shared worker threads (created on first use)."""
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(8, thread_name_prefix="wakis-spmv"))
    return _POOL[0]


# A forked child (e.g. a multiprocessing sweep) has no copies of the
# parent's worker threads: it creates its own pool
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOL.clear)


try:  # in-place CSR product (no output allocation); private SciPy API
    from scipy.sparse._sparsetools import csr_matvec as _csr_matvec
except ImportError:  # pragma: no cover
    _csr_matvec = None


class ParallelCSR:
    """
    CSR matrix multiplied with a vector on threads, by row blocks.

    Each row is summed in the same order as by ``csr @ x``, so the result is
    bitwise identical to the single-threaded product.

    Parameters
    ----------
    matrix : scipy.sparse matrix
    n_threads : int, optional
        Row blocks (threads); default `default_threads`.
    dtype : dtype, optional
        Data (and vector) dtype.
    """

    def __init__(self, matrix, n_threads=None, dtype=None):
        matrix = matrix.tocsr()
        if dtype is not None and matrix.dtype != dtype:
            matrix = matrix.astype(dtype)
        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        n = default_threads() if n_threads is None else max(1, int(n_threads))
        rows = matrix.shape[0]
        if n == 1 or rows < 200_000:
            cuts = np.array([0, rows])
        else:
            # Split by nonzeros so that the blocks cost about the same
            cuts = np.searchsorted(matrix.indptr, np.linspace(0, matrix.nnz, n + 1))
            cuts = np.unique(np.clip(cuts, 0, rows))
            cuts[0], cuts[-1] = 0, rows
        self.cuts = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

    def matvec(self, x, out=None):
        """out = A x (out is overwritten; allocated if None)."""
        m = self.matrix
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if out is None:
            out = np.empty(self.shape[0], dtype=self.dtype)

        def run(cut):
            a, b = cut
            if _csr_matvec is None:
                out[a:b] = m[a:b] @ x
                return
            out[a:b] = 0
            _csr_matvec(
                b - a, self.shape[1], m.indptr[a : b + 1], m.indices, m.data, x,
                out[a:b],
            )  # fmt: skip

        if len(self.cuts) == 1:
            run(self.cuts[0])
        else:
            list(_pool().map(run, self.cuts))
        return out

    def __matmul__(self, x):
        return self.matvec(x)


# Chunks of the element-wise products (run on threads)
_CHUNKS = 16


def _chunked(n_items, fn, n_threads):
    """Call fn(c) for every chunk index, on threads when n_threads > 1."""
    if n_threads > 1:
        return list(_pool().map(fn, range(n_items)))
    return [fn(c) for c in range(n_items)]


def lanczos_max(
    apply_K,
    p,
    corr=None,
    dtype=float,
    tol=1e-3,
    maxiter=1000,
    seed=0,
    n_threads=None,
    start=None,
):
    """
    Largest eigenvalue of P K, P and K Hermitian positive semi-definite,
    P = diag(p) + E with a sparse correction E.

    Parameters
    ----------
    apply_K : callable
        Product (x, out) -> K x, written to and returning `out`.
    p : ndarray
        Diagonal of P (>= 0).
    corr : callable, optional
        x -> (pos, values) with (E x)[pos] = values and zero elsewhere.
    dtype : dtype
        Vector dtype (complex for Bloch operators).
    tol : float
        Relative residual tolerance of the largest Ritz pair.
    maxiter : int
        Maximum number of iterations.
    seed : int
        Seed of the random start vector (the result is reproducible).
    n_threads : int, optional
        Threads for the vector operations (default `default_threads`).
    start : ndarray, optional
        Added to the random start vector.

    Returns
    -------
    float
        Largest Ritz value (a lower bound converging to lambda_max).
    int
        Number of iterations (products with K).
    """
    from scipy.linalg import get_blas_funcs

    n = p.size
    threads = default_threads() if n_threads is None else max(1, int(n_threads))
    bounds = np.linspace(0, n, _CHUNKS + 1).astype(np.int64)
    sl = [slice(bounds[c], bounds[c + 1]) for c in range(_CHUNKS)]
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n).astype(dtype)
    if np.iscomplexobj(r):
        r += 1j * rng.standard_normal(n)
    if start is not None:
        r += start
    # In-place level-1 BLAS (multi-threaded, no temporaries)
    axpy, scal = get_blas_funcs(("axpy", "scal"), (r,))
    dot = get_blas_funcs("dotc" if np.iscomplexobj(r) else "dot", (r,))

    def apply_P(x, out):
        """out = P x; returns x^H P x."""

        def run(c):
            np.multiply(p[sl[c]], x[sl[c]], out=out[sl[c]])

        _chunked(_CHUNKS, run, threads)
        if corr is not None:
            pos, val = corr(x)
            out[pos] += val
        return dot(x, out).real

    z = np.empty_like(r)
    beta = np.sqrt(max(apply_P(r, z), 0.0))
    if beta == 0:
        return 0.0, 0
    scal(1.0 / beta, r)
    scal(1.0 / beta, z)
    r_prev, beta_prev = None, 0.0
    z_new = np.empty_like(r)
    spare = np.empty_like(r)
    alphas, betas = [], []
    theta = 0.0
    for it in range(1, maxiter + 1):
        w = apply_K(z, spare)
        alpha = dot(z, w).real
        axpy(r, w, a=-alpha)
        if r_prev is not None:
            axpy(r_prev, w, a=-beta_prev)
        beta = np.sqrt(max(apply_P(w, z_new), 0.0))
        alphas.append(alpha)
        if len(alphas) == 1:
            theta, s_last = alphas[0], 1.0
        else:
            vals, vecs = eigh_tridiagonal(
                np.array(alphas),
                np.array(betas),
                select="i",
                select_range=(len(alphas) - 1, len(alphas) - 1),
            )
            theta, s_last = vals[0], vecs[-1, 0]
        if beta * abs(s_last) <= tol * abs(theta) or beta == 0:
            return float(theta), it
        betas.append(beta)
        scal(1.0 / beta, w)
        scal(1.0 / beta, z_new)
        free = r_prev
        r_prev, r = r, w
        spare = np.empty_like(r) if free is None else free
        z, z_new = z_new, z
        beta_prev = beta
    return float(theta), maxiter


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


def stable_dt(
    solver, op=None, tol=1e-3, n_threads=None, device="auto", return_info=False
):
    """
    Largest stable leapfrog time step 2 / sqrt(lambda_max(B A)) of a solver
    before its first step (conformal metrics, optional `TensorOperator`).

    The Lanczos vectors live on the E edges with P_ee > 0 only (edges in PEC,
    closed edges and ghost edges carry no field), and the curl and face
    operators are restricted to the faces those edges touch, which does not
    change P K on that subspace.

    Parameters
    ----------
    solver : SolverFIT3D
    op : TensorOperator, optional
        Assembled tensor couplings (conformal='tensor').
    tol : float
        Relative residual tolerance; the returned step includes a margin of
        10 tol on lambda_max (as for the ARPACK estimate it replaces).
    n_threads : int, optional
        Threads of the sparse products (default `default_threads`, 1 with MPI).
    device : {'auto', 'cpu', 'gpu'}
        'auto': on the GPU (CuPy) when ``solver.use_gpu``.

    Returns
    -------
    float
        dt_max [s] (and a dict with iterations and sizes if `return_info`).
    """
    import time

    from scipy.sparse import diags

    t0 = time.perf_counter()
    if n_threads is None and getattr(solver, "use_mpi", False):
        n_threads = 1
    ieps = _host(solver.ieps.toarray()).astype(float)
    itda = np.asarray(solver.itDa.diagonal()).real
    tds = np.asarray(solver.tDs.diagonal()).real
    ds = np.asarray(solver.Ds.diagonal()).real
    imu = _host(solver.imu.toarray()).astype(float)
    p = ds * ieps * itda
    active = np.flatnonzero(p > 0)
    dtype = np.result_type(solver.dtype, solver.C.dtype, float)
    if op is not None and op.rows is not None and op.rows.size:
        dtype = np.result_type(dtype, op.Kp_host.dtype)
    if active.size < 0.8 * p.size:
        # Lanczos vectors on the active edges only (e.g. PEC background):
        # curl without the inactive edges and the faces they alone touch
        C = solver.C.tocsc()[:, active].tocsr()
        faces = np.flatnonzero(np.diff(C.indptr))
        C = C[faces]
    else:
        # Few inactive edges (P = 0 there): keep the full operators
        active = np.arange(p.size)
        C = solver.C.tocsr()
        faces = active
    # tDs G imu on those faces, folded into the matrix entries
    G = solver.iDa.tocsr()
    G = G[faces][:, faces] if faces.size < G.shape[0] else G.copy()
    G.data = G.data * tds[faces][np.repeat(np.arange(faces.size), np.diff(G.indptr))]
    G.data *= imu[faces][G.indices]
    CH = C.conj().T.tocsr()
    p_a = p[active]
    info = dict(edges=int(active.size), faces=int(faces.size))
    Kp = pos = None
    if op is not None and op.rows is not None and op.rows.size:
        rows = op.rows
        pos = np.searchsorted(active, rows)
        if np.any(pos >= active.size) or np.any(
            active[np.minimum(pos, active.size - 1)] != rows
        ):
            raise ValueError("Tensor-coupled edges must have a positive P diagonal.")
        # Scaled couplings diag(ds) Kp diag(ieps itda) on the coupled rows
        Kp = diags(ds[rows]) @ op.Kp_host @ diags(ieps[rows] * itda[rows])
        info["coupled"] = int(rows.size)
    # Start vector: random plus the grid's highest-frequency (Nyquist) mode,
    # E_c ~ m_c (-1)^(i+j+k) with m transverse to (1, 1, 1), the top
    # eigenvector of the uniform curl-curl operator: the top of the spectrum
    # is reached in fewer iterations (the random part keeps an overlap with
    # every eigenvector, e.g. modes localized at small cells)
    ijk = [np.arange(m) for m in (solver.Nx, solver.Ny, solver.Nz)]
    sign = 1.0 - 2.0 * ((ijk[0][:, None, None] + ijk[1][:, None] + ijk[2]) % 2)
    sign = sign.ravel(order="F")
    start = np.concatenate([m * sign for m in (1.0, -1.0, 0.3)])
    start = start if active.size == start.size else start[active]
    del sign
    if device == "auto":
        device = "gpu" if getattr(solver, "use_gpu", False) else "cpu"
    info["device"] = device
    if device == "gpu":
        lam, iters = _stable_lambda_gpu(
            C, G, CH, p_a, Kp, pos, dtype, tol, info, t0, start
        )
    else:
        Cop = ParallelCSR(C, n_threads, dtype)
        Gop = ParallelCSR(G, n_threads, dtype)
        CHop = ParallelCSR(CH, n_threads, dtype)
        del C, G, CH
        f1 = np.empty(faces.size, dtype)
        f2 = np.empty(faces.size, dtype)

        def apply_K(x, out):
            Cop.matvec(x, f1)
            Gop.matvec(f1, f2)
            return CHop.matvec(f2, out)

        corr = None
        if Kp is not None:
            Kop = ParallelCSR(Kp, n_threads, dtype)

            def corr(x):
                return pos, Kop.matvec(x[pos])

        info["setup_s"] = time.perf_counter() - t0
        lam, iters = lanczos_max(
            apply_K, p_a, corr, dtype=dtype, tol=tol, n_threads=n_threads, start=start
        )
    dt = 2.0 / np.sqrt(abs(lam) * (1.0 + 10 * tol))
    if return_info:
        info["iterations"] = iters
        return dt, info
    return dt


def _stable_lambda_gpu(C, G, CH, p, Kp, pos, dtype, tol, info, t0, start, maxiter=1000):
    """`lanczos_max` with CuPy arrays (same recurrence and start vector)."""
    import time

    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix

    Cg, Gg, CHg = (csr_matrix(m.astype(dtype)) for m in (C, G, CH))
    pg = cp.asarray(p)
    Kg = csr_matrix(Kp.astype(dtype)) if Kp is not None else None
    posg = cp.asarray(pos) if Kp is not None else None
    info["setup_s"] = time.perf_counter() - t0
    n = p.size
    rng = np.random.default_rng(0)
    r = rng.standard_normal(n).astype(dtype)
    if np.iscomplexobj(r):
        r += 1j * rng.standard_normal(n)
    r = cp.asarray(r + start)

    def apply_P(x):
        y = (pg * x).astype(dtype, copy=False)
        if Kg is not None:
            y[posg] += Kg @ x[posg]
        return y, float(cp.vdot(x, y).real)

    z, bb = apply_P(r)
    beta = np.sqrt(max(bb, 0.0))
    if beta == 0:
        return 0.0, 0
    r /= beta
    z /= beta
    r_prev, beta_prev = None, 0.0
    alphas, betas = [], []
    theta = 0.0
    for it in range(1, maxiter + 1):
        w = CHg @ (Gg @ (Cg @ z))
        alpha = float(cp.vdot(z, w).real)
        w -= alpha * r
        if r_prev is not None:
            w -= beta_prev * r_prev
        z_new, bb = apply_P(w)
        beta = np.sqrt(max(bb, 0.0))
        alphas.append(alpha)
        if len(alphas) == 1:
            theta, s_last = alphas[0], 1.0
        else:
            vals, vecs = eigh_tridiagonal(
                np.array(alphas),
                np.array(betas),
                select="i",
                select_range=(len(alphas) - 1, len(alphas) - 1),
            )
            theta, s_last = vals[0], vecs[-1, 0]
        if beta * abs(s_last) <= tol * abs(theta) or beta == 0:
            return float(theta), it
        betas.append(beta)
        r_prev, r = r, w / beta
        z = z_new / beta
        beta_prev = beta
    return float(theta), maxiter
