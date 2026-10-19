# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""
Fused GPU time stepping (``SolverFIT3D(..., use_gpu=True, fused=True)``).

The CuPy path of `SolverFIT3D` issues one kernel per array operation: the CPML
step alone launches ~330 kernels per time step (6 + 6 sparse curls, gathers and
scatters of the 24 auxiliary-variable faces, temporaries), and a DFT monitor
~5 per component. For grids below a few million cells the GPU then waits on
the host (Python + launch latency, ~10-20 us per operation), not on memory.

This module performs the same arithmetic in a handful of hand-written kernels:

* `FusedStep`: one kernel per half step. Each thread owns one edge: it forms
  the curl from a merged CSR row (the two derivative terms of the component
  kept apart for the CPML), updates the CPML auxiliary variables psi owned by
  that edge, and updates the field (plus magnetic current M, conduction
  current J = sigma E). Without CPML the solver's own curl operators are used.
* `source_delta`: impressed-current sources (`CurrentSource`,
  `MagneticCurrentSource`, `ModeSource`) add w (s^n - s^{n-1}) in one kernel.
* `DFTBatch`: running DFTs of any number of `DFTMonitor`, `FluxMonitor` and
  `ModeMonitor` terms, all frequencies, in one kernel per step; the
  accumulators of the monitors become views into one buffer.
* `AdjointKernels` (used by `wakis.inverse.adjoint`): the transposed step in
  four kernels, the adjoint DFT sources and the gradient accumulation.
* `GraphRunner`: replays a CUDA graph of K captured steps (no Python, no
  launch overhead) when every per-step operation is device resident.

The arithmetic is that of the unfused path up to the order of floating-point
operations (round-off); sources and DFT accumulations reproduce it bit for
bit (the kernels are compiled without FMA contraction). All buffers are
allocated once, so the step is safe to capture in a CUDA graph.
"""

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None
    _HAS_CUPY = False

_THREADS = 256

# Same term tables as `wakis.inverse.adjoint` (target T, derivative axis d,
# source S, sign s, operator, psi prefix); a: s = +1, b: s = -1
_H_TERMS = (
    ("x", "y", "z", 1.0, "dxy", "psiHa_x"),
    ("x", "z", "y", -1.0, "dxz", "psiHb_x"),
    ("y", "z", "x", 1.0, "dyz", "psiHa_y"),
    ("y", "x", "z", -1.0, "dyx", "psiHb_y"),
    ("z", "x", "y", 1.0, "dzx", "psiHa_z"),
    ("z", "y", "x", -1.0, "dzy", "psiHb_z"),
)
_E_TERMS = (
    ("x", "y", "z", 1.0, "dtxy", "psiEa_x"),
    ("x", "z", "y", -1.0, "dtxz", "psiEb_x"),
    ("y", "z", "x", 1.0, "dtyz", "psiEa_y"),
    ("y", "x", "z", -1.0, "dtyx", "psiEb_y"),
    ("z", "x", "y", 1.0, "dtzx", "psiEa_z"),
    ("z", "y", "x", -1.0, "dtzy", "psiEb_z"),
)

_SOURCE = r"""
#include <cupy/complex.cuh>
typedef complex<double> cplx;

__device__ __forceinline__ double re_(double x) { return x; }
__device__ __forceinline__ double re_(cplx x) { return x.real(); }
__device__ __forceinline__ double conj_(double x) { return x; }
__device__ __forceinline__ cplx conj_(cplx x) { return conj(x); }

// Time of step n: the device step counter (graph replay) or the host value
__device__ __forceinline__ double step_time(const long long* step, long long add,
                                            double t_host, double dt) {
    return step ? (double)(step[0] + add) * dt : t_host;
}

// ------------------------------------------------------------ curl rows
// The two derivative terms p_a, p_b of row i: CSR (entries [ptr, split) and
// [split, ptr+1)) or ELL (wa + wb padded entries per row, column-major).
// Without CPML all entries belong to p_a.
template<typename T, typename V, bool ELL>
__device__ __forceinline__ void row_terms(const int i, const int n,
        const int* __restrict__ ptr, const int* __restrict__ split,
        const int* __restrict__ col, const V* __restrict__ val, const int wa,
        const int wb, const T* __restrict__ x, T& pa, T& pb) {
    pa = T(0);
    pb = T(0);
    if (ELL) {
        for (int j = 0; j < wa; ++j) {
            long long k = (long long)j * n + i;
            pa += val[k] * x[col[k]];
        }
        for (int j = wa; j < wa + wb; ++j) {
            long long k = (long long)j * n + i;
            pb += val[k] * x[col[k]];
        }
    } else {
        int j0 = ptr[i], j1 = ptr[i + 1];
        int js = split ? split[i] : j1;
        for (int j = j0; j < js; ++j) pa += val[j] * x[col[j]];
        for (int j = js; j < j1; ++j) pb += val[j] * x[col[j]];
    }
}

// ---------------------------------------------------------------- H half step
// CPML: H_i -= dt imu_i (p_a - p_b + psi_a - psi_b), psi = b psi + c p;
// else: H_i -= dt p_a (imu inside the curl operator); then -= dt imu_i M_i.
template<typename T, typename V, bool ELL>
__global__ void fused_h(const int n, const int cpml, const int* __restrict__ ptr,
        const int* __restrict__ split, const int* __restrict__ col,
        const V* __restrict__ val, const int wa, const int wb,
        const T* __restrict__ E, T* __restrict__ H, const double dt,
        const int* __restrict__ slot_a, const int* __restrict__ slot_b,
        T* __restrict__ psi, const T* __restrict__ pb, const T* __restrict__ pc,
        const T* __restrict__ M, const double* __restrict__ imu) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    T pa, pbv;
    row_terms<T, V, ELL>(i, n, ptr, split, col, val, wa, wb, E, pa, pbv);
    T h = H[i];
    if (cpml) {
        double f = dt * imu[i];
        h -= f * (pa - pbv);
        int sa = slot_a[i];
        if (sa >= 0) {
            T p = pb[sa] * psi[sa] + pc[sa] * pa;
            psi[sa] = p;
            h -= f * p;
        }
        int sb = slot_b[i];
        if (sb >= 0) {
            T p = pb[sb] * psi[sb] + pc[sb] * pbv;
            psi[sb] = p;
            h -= f * (-p);
        }
    } else {
        h -= dt * pa;
    }
    if (M) h = h - (dt * imu[i]) * M[i];
    H[i] = h;
}

// ---------------------------------------------------------------- E half step
// J += sigma E - J_old (conduction current); then
// CPML: E += dt ieps (q_a - q_b) - dt ieps J, E += dt ieps (psi_a - psi_b)
// else: E += dt (q - ieps J)   (ieps inside the curl operator)
template<typename T, typename V, bool ELL>
__global__ void fused_e(const int n, const int cpml, const int* __restrict__ ptr,
        const int* __restrict__ split, const int* __restrict__ col,
        const V* __restrict__ val, const int wa, const int wb,
        const T* __restrict__ H, T* __restrict__ E,
        const double* __restrict__ ieps, const double dt,
        const int* __restrict__ slot_a, const int* __restrict__ slot_b,
        T* __restrict__ psi, const T* __restrict__ pb, const T* __restrict__ pc,
        T* __restrict__ J, T* __restrict__ J_old, const double* __restrict__ sigma) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    T e = E[i];
    T j = J[i];
    if (sigma) {
        T jt = sigma[i] * e;
        j = j + (jt - J_old[i]);
        J[i] = j;
        J_old[i] = jt;
    }
    T qa, qb;
    row_terms<T, V, ELL>(i, n, ptr, split, col, val, wa, wb, H, qa, qb);
    if (cpml) {
        double f = dt * ieps[i];
        e += f * (qa - qb) - f * j;
        int sa = slot_a[i];
        if (sa >= 0) {
            T p = pb[sa] * psi[sa] + pc[sa] * qa;
            psi[sa] = p;
            e += f * p;
        }
        int sb = slot_b[i];
        if (sb >= 0) {
            T p = pb[sb] * psi[sb] + pc[sb] * qb;
            psi[sb] = p;
            e += f * (-p);
        }
    } else {
        e = e + dt * (qa - ieps[i] * j);
    }
    E[i] = e;
}

// ------------------------------------------------------------------ sources
// arr[idx_k] += w_k a - old_k; old_k = w_k a   (a = s(t) on the host, or
// amp[step] from a device table)
template<typename T, typename W>
__global__ void source_delta(const int n, const int* __restrict__ idx,
        const W* __restrict__ w, const double a_host,
        const double* __restrict__ amp, const long long* __restrict__ step,
        const long long add, W* __restrict__ old, T* __restrict__ arr) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    double a = amp ? amp[step[0] + add] : a_host;
    W nw = w[k] * a;
    arr[idx[k]] += (T)(nw - old[k]);
    old[k] = nw;
}

// ------------------------------------------------------------- running DFT
// Phasors ph[p] = dt exp(c1 f t_s) of each (segment s, frequency f) pair,
// t_s = t + toff_s: same complex operations as the CuPy expressions of the
// monitors, dt * exp(-2j * pi * freqs * t).
__global__ void dft_phase(const int nph, const int* __restrict__ ph_seg,
        const int* __restrict__ ph_f, const double* __restrict__ seg_toff,
        const double* __restrict__ freqs, const double c1_re, const double c1_im,
        const double dt, const double t_host, const long long* __restrict__ step,
        const long long add, cplx* __restrict__ ph) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nph) return;
    double t = step_time(step, add, t_host, dt) + seg_toff[ph_seg[p]];
    cplx z = cplx(c1_re, c1_im) * cplx(freqs[ph_f[p]], 0.0);
    z = z * cplx(t, 0.0);
    ph[p] = cplx(dt, 0.0) * exp(z);
}

// acc[s, f, k] += ph[s, f] v_k for every sample k of segment (term) s and
// its frequencies f (one thread per sample, looping over f: the sample is
// gathered once); v = F[a_k] or 0.5 (F[a_k] + F[b_k])
template<typename T>
__global__ void dft_acc(const long long total, const int nseg,
        const long long* __restrict__ seg_start, const int* __restrict__ seg_m,
        const int* __restrict__ seg_soff, const int* __restrict__ seg_poff,
        const int* __restrict__ seg_nf, const int* __restrict__ seg_field,
        const int* __restrict__ seg_avg, const int* __restrict__ ia,
        const int* __restrict__ ib, const cplx* __restrict__ ph,
        const T* __restrict__ E, const T* __restrict__ H, const T* __restrict__ J,
        cplx* __restrict__ acc) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= total) return;  // total = number of samples
    int s = 0;
    while (s + 1 < nseg && g >= seg_soff[s + 1]) ++s;
    int k = (int)(g - seg_soff[s]);
    int m = seg_m[s];
    const T* F = seg_field[s] == 0 ? E : (seg_field[s] == 1 ? H : J);
    T v = F[ia[g]];
    if (seg_avg[s]) v = 0.5 * (v + F[ib[g]]);
    cplx cv = cplx(v);
    cplx* a = acc + seg_start[s] + k;
    const cplx* p = ph + seg_poff[s];
    int nf = seg_nf[s];
    for (int f = 0; f < nf; ++f) {
        long long o = (long long)f * m;
        a[o] = a[o] + p[f] * cv;
    }
}

// ------------------------------------------------------------- record rows
// Adjoint forward: row_k = (E[idx_k] - e_old_k) / ieps_k into rec[step]
template<typename T>
__global__ void gather(const int n, const int* __restrict__ idx,
                       const T* __restrict__ src, T* __restrict__ out) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) out[k] = src[idx[k]];
}

template<typename T, typename R>
__global__ void record_row(const int n, const int* __restrict__ idx,
        const T* __restrict__ E, const T* __restrict__ e_old,
        const double* __restrict__ ieps_d, R* __restrict__ rec,
        const long long* __restrict__ step, const long long add,
        const long long row_host) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    long long r = step ? step[0] + add : row_host;
    rec[r * n + k] = (R)((E[idx[k]] - e_old[k]) / ieps_d[k]);
}


// ---------------------------------------------------- TF/SF auxiliary line
// F[rows_r] += sum_j K[r, j] x_j  (H correction of the TF/SF boundary)
template<typename T>
__global__ void csr_add(const int nr, const int* __restrict__ rows,
        const int* __restrict__ ptr, const int* __restrict__ col,
        const double* __restrict__ val, const double* __restrict__ x,
        T* __restrict__ F) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= nr) return;
    double s = 0.0;
    for (int j = ptr[r]; j < ptr[r + 1]; ++j) s += val[j] * x[col[j]];
    F[rows[r]] += s;
}

// jn = sum_j K[r, j] x_j; F[rows_r] += jn - old_r; old_r = jn
template<typename T>
__global__ void csr_delta(const int nr, const int* __restrict__ rows,
        const int* __restrict__ ptr, const int* __restrict__ col,
        const double* __restrict__ val, const double* __restrict__ x,
        T* __restrict__ old, T* __restrict__ F) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= nr) return;
    double s = 0.0;
    for (int j = ptr[r]; j < ptr[r + 1]; ++j) s += val[j] * x[col[j]];
    T jn = T(s);
    F[rows[r]] += jn - old[r];
    old[r] = jn;
}

// h = h ch - dh (e[i+1] - e[i]); h[hidx] += s (s = coefficient x incident value)
__global__ void line_h(const int nh, double* __restrict__ h,
        const double* __restrict__ e, const double* __restrict__ ch,
        const double* __restrict__ dh, const int hidx, const double s_host,
        const double* __restrict__ table, const long long* __restrict__ step,
        const long long add) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nh) return;
    double v = h[i] * ch[i];
    v = v - dh[i] * (e[i + 1] - e[i]);
    if (i == hidx) v = v + (table ? table[step[0] + add] : s_host);
    h[i] = v;
}

// e = e ce - de curl (curl = h[i] - h[i-1] inside); e[0] = e[-1] = 0;
// e[eidx] += s
__global__ void line_e(const int ne, double* __restrict__ e,
        const double* __restrict__ h, const double* __restrict__ ce,
        const double* __restrict__ de, const int eidx, const double s_host,
        const double* __restrict__ table, const long long* __restrict__ step,
        const long long add) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) return;
    double curl = (i >= 1 && i <= ne - 2) ? h[i] - h[i - 1] : 0.0;
    double v = e[i] * ce[i];
    v = v - de[i] * curl;
    if (i == 0 || i == ne - 1) v = 0.0;
    if (i == eidx) v = v + (table ? table[step[0] + add] : s_host);
    e[i] = v;
}

// Entry-plane record: series[0, n] = e[m], series[1, n] = (h[m-1] + h[m]) / 2
__global__ void line_series(const double* __restrict__ e,
        const double* __restrict__ h, const int m, double* __restrict__ series,
        const long long cap, const long long n_host,
        const long long* __restrict__ step, const long long add) {
    if (blockDim.x * blockIdx.x + threadIdx.x != 0) return;
    long long n = step ? step[0] + add : n_host;
    series[n] = e[m];
    series[cap + n] = 0.5 * (h[m - 1] + h[m]);
}


// Tensor averaging: delta = E[r] - e_old (kept as the step increment), then
// E[r_i] += sum_j K'_ij delta_j
template<typename T>
__global__ void tensor_delta(const int n, const int* __restrict__ rows,
        const T* __restrict__ E, const T* __restrict__ e_old,
        T* __restrict__ delta) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) delta[k] = E[rows[k]] - e_old[k];
}

template<typename T, typename V>
__global__ void rows_spmv_add(const int nr, const int* __restrict__ rows,
        const int* __restrict__ ptr, const int* __restrict__ col,
        const V* __restrict__ val, const T* __restrict__ x, T* __restrict__ F) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= nr) return;
    T s = T(0);
    for (int j = ptr[r]; j < ptr[r + 1]; ++j) s += val[j] * x[col[j]];
    F[rows[r]] += s;
}


// ADE correction of one dispersive group (see DispersiveMixin._one_step_dispersive)
__global__ void ade_update(const int n, const int* __restrict__ idx,
        double* __restrict__ E, const double* __restrict__ En,
        const double* __restrict__ eps_s, const double* __restrict__ a1,
        const double* __restrict__ den, const double* __restrict__ w,
        const int nL, const double* __restrict__ cL, double* __restrict__ P,
        double* __restrict__ Pold, const int nD, const double* __restrict__ cD,
        double* __restrict__ D) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    int e = idx[k];
    double en = En[k];
    double acc = (E[e] - en) * eps_s[k] + a1[k] * en;
    for (int l = 0; l < nL; ++l) {
        long long o = (long long)l * n + k;
        double p = P[o];
        double pnew = cL[3 * l] * p + cL[3 * l + 1] * Pold[o] + cL[3 * l + 2] * en;
        acc -= w[k] * (pnew - p);
        Pold[o] = p;
        P[o] = pnew;
    }
    for (int d = 0; d < nD; ++d)
        acc -= (w[k] * (cD[2 * d] - 1.0)) * D[(long long)d * n + k];
    double enew = acc / den[k];
    for (int d = 0; d < nD; ++d) {
        long long o = (long long)d * n + k;
        D[o] = cD[2 * d] * D[o] + cD[2 * d + 1] * (enew + en);
    }
    E[e] = enew;
}

__global__ void step_add(long long* step, const long long k) { step[0] += k; }

// ----------------------------------------------------- transposed CPML step
// Adjoint of the E half step, part 1 (per E row i, nu = dt ieps lam_E):
// beta_a = nu + c_a mu_a, beta_b = -nu + c_b mu_b, out_psi = b mu,
// mu_a = lam_psi_a + nu, mu_b = lam_psi_b - nu. Without CPML beta_a = nu.
template<typename T>
__global__ void adj_beta(const int n, const T* __restrict__ lam,
        const double* __restrict__ coef, const double dt,
        const int* __restrict__ slot_a, const int* __restrict__ slot_b,
        const T* __restrict__ lpsi, T* __restrict__ opsi,
        const T* __restrict__ pb, const T* __restrict__ pc,
        T* __restrict__ beta) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    T nu = (dt * coef[i]) * lam[i];
    T ba = nu, bb = -nu;
    if (slot_a) {
        int sa = slot_a[i];
        if (sa >= 0) {
            T mu = lpsi[sa] + nu;
            ba += conj_(pc[sa]) * mu;
            opsi[sa] = conj_(pb[sa]) * mu;
        }
        int sb = slot_b[i];
        if (sb >= 0) {
            T mu = lpsi[sb] - nu;
            bb += conj_(pc[sb]) * mu;
            opsi[sb] = conj_(pb[sb]) * mu;
        }
        beta[n + i] = bb;
    }
    beta[i] = ba;
}

// out_i = d_i x_i + sum_j A^H_ij beta_j  (merged transposed CSR, 3N x 6N or 3N x 3N)
template<typename T, typename V>
__global__ void adj_spmv(const int n, const int* __restrict__ ptr,
        const int* __restrict__ col, const V* __restrict__ val,
        const T* __restrict__ beta, const T* __restrict__ x,
        const double* __restrict__ d, T* __restrict__ out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    T s = T(0);
    for (int j = ptr[i]; j < ptr[i + 1]; ++j) s += val[j] * beta[col[j]];
    T base = d ? d[i] * x[i] : x[i];
    out[i] = base + s;
}

// grad_k += Re(conj(lam[idx_k]) * rec[step, k])
template<typename T, typename R>
__global__ void grad_acc(const int n, const int* __restrict__ idx,
        const T* __restrict__ lam, const R* __restrict__ rec,
        const long long* __restrict__ step, const long long add,
        const long long row_host, double* __restrict__ grad) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    long long r = step ? step[0] + add : row_host;
    grad[k] += re_(conj_(lam[idx[k]]) * (T)rec[r * n + k]);
}

// Adjoint DFT source, part 1: v_k = sum_f (dt exp(+2 pi i f t)) G[f, k]
__global__ void tap_v(const int m, const int nf, const double* __restrict__ freqs,
        const cplx* __restrict__ G, const double c1_im, const double dt,
        const double toff, const double t_host,
        const long long* __restrict__ step, const long long add,
        cplx* __restrict__ v) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= m) return;
    double t = step_time(step, add, t_host, dt) + toff;
    cplx s = cplx(0.0, 0.0);
    for (int f = 0; f < nf; ++f) {
        cplx z = cplx(0.0, c1_im) * cplx(freqs[f], 0.0);
        z = z * cplx(t, 0.0);
        s += (cplx(dt, 0.0) * exp(z)) * G[(long long)f * m + k];
    }
    v[k] = s;
}

// part 2: lam[rows_r] += (Re) sum_j WH[r, j] v_j
__device__ __forceinline__ void add_to(double& a, cplx s) { a += s.real(); }
__device__ __forceinline__ void add_to(cplx& a, cplx s) { a += s; }

template<typename T>
__global__ void tap_add(const int nr, const int* __restrict__ rows,
        const int* __restrict__ ptr, const int* __restrict__ col,
        const cplx* __restrict__ val, const cplx* __restrict__ v,
        T* __restrict__ lam) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= nr) return;
    cplx s = cplx(0.0, 0.0);
    for (int j = ptr[r]; j < ptr[r + 1]; ++j) s += val[j] * v[col[j]];
    add_to(lam[rows[r]], s);
}
"""

_TYPES = {np.dtype(np.float64): "double", np.dtype(np.complex128): "cplx"}
_REC_TYPES = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "float",
    np.dtype(np.complex128): "cplx",
    np.dtype(np.complex64): "complex<float>",
}
_MODULE = None


def _names():
    names = []
    pairs = (("double", "double"), ("cplx", "double"), ("cplx", "cplx"))
    for T, V in pairs:
        for ell in ("true", "false"):
            names += [f"fused_h<{T},{V},{ell}>", f"fused_e<{T},{V},{ell}>"]
        names += [f"source_delta<{T},{V}>", f"adj_spmv<{T},{V}>"]
        names += [f"rows_spmv_add<{T},{V}>"]
    recs = {"double": ("double", "float"), "cplx": ("cplx", "complex<float>")}
    for T in ("double", "cplx"):
        names += [f"dft_acc<{T}>", f"adj_beta<{T}>", f"tap_add<{T}>", f"gather<{T}>"]
        names += [f"csr_add<{T}>", f"csr_delta<{T}>", f"tensor_delta<{T}>"]
        for R in recs[T]:
            names += [f"record_row<{T},{R}>", f"grad_acc<{T},{R}>"]
    return names + ["step_add", "tap_v", "dft_phase", "line_h", "line_e", "line_series",
                    "ade_update"]


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(
            code=_SOURCE,
            options=("--std=c++17", "--fmad=false"),
            name_expressions=_names(),
        )
    return _MODULE


_KERNELS = {}


def kernel(name):
    """Compiled kernel by its template name, e.g. 'fused_h<double,double,true>'."""
    k = _KERNELS.get(name)
    if k is None:
        k = _KERNELS[name] = _module().get_function(name)
    return k


def tname(dtype):
    """Kernel type name of a NumPy/CuPy dtype (float64 or complex128)."""
    return _TYPES[np.dtype(dtype)]


def rname(dtype):
    """Kernel type name of a stored-increment dtype."""
    return _REC_TYPES[np.dtype(dtype)]


def launch(name, n, *args):
    """Launch kernel `name` over n threads (no-op for n == 0)."""
    n = int(n)
    if n <= 0:
        return
    kernel(name)(((n + _THREADS - 1) // _THREADS,), (_THREADS,), args)


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _host_csr(op):
    """SciPy CSR copy of a (CuPy or SciPy) sparse operator."""
    import scipy.sparse as sp

    if hasattr(op, "get"):
        op = op.get()
    return sp.csr_matrix(op)


def merge_rows(A, B):
    """
    CSR (ptr, split, col, val) with row i = row i of A, then row i of B.

    Entries [ptr[i], split[i]) belong to A, [split[i], ptr[i+1]) to B.
    """
    A, B = A.tocsr(), B.tocsr()
    na, nb = np.diff(A.indptr), np.diff(B.indptr)
    n = A.shape[0]
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(na + nb, out=ptr[1:])
    split = ptr[:-1] + na
    dtype = np.result_type(A.dtype, B.dtype)
    col = np.empty(ptr[-1], dtype=np.int64)
    val = np.empty(ptr[-1], dtype=dtype)
    rows_a = np.repeat(np.arange(n), na)
    pos_a = ptr[rows_a] + (np.arange(A.nnz) - A.indptr[rows_a])
    rows_b = np.repeat(np.arange(n), nb)
    pos_b = split[rows_b] + (np.arange(B.nnz) - B.indptr[rows_b])
    col[pos_a], val[pos_a] = A.indices, A.data
    col[pos_b], val[pos_b] = B.indices, B.data
    if ptr[-1] >= 2**31:
        raise ValueError("Operator too large for 32-bit fused kernels.")
    return ptr.astype(np.int32), split.astype(np.int32), col.astype(np.int32), val


class RowOp:
    """
    Curl rows of a fused half step on the device: ELL (column-major, padded
    with zero entries, when every row has at most `max_width` entries and
    the padding stays below `max_pad` times the stored entries) or CSR.
    ELL loads are coalesced (~6 % faster CPML steps on an L4); the
    summation order within a row is the same in both formats.
    """

    def __init__(self, ptr, split, col, val, max_width=8, max_pad=1.6):
        n = ptr.size - 1
        split = ptr[1:] if split is None else split
        na, nb = split - ptr[:-1], ptr[1:] - split
        wa, wb = int(na.max(initial=0)), int(nb.max(initial=0))
        self.vdtype = val.dtype
        self.ell = 0 < wa + wb <= max_width and (wa + wb) * n <= max_pad * max(
            col.size, 1
        )
        if not self.ell:
            self.ptr, self.split = cp.asarray(ptr), cp.asarray(split)
            self.col, self.val = cp.asarray(col), cp.asarray(val)
            self.wa = self.wb = 0
            return
        rows = np.arange(n)
        c = np.tile(rows, (wa + wb, 1)).astype(np.int32)  # padding: own index
        v = np.zeros((wa + wb, n), dtype=val.dtype)
        for j in range(wa):
            r = rows[na > j]
            c[j, r], v[j, r] = col[ptr[r] + j], val[ptr[r] + j]
        for j in range(wb):
            r = rows[nb > j]
            c[wa + j, r], v[wa + j, r] = col[split[r] + j], val[split[r] + j]
        self.ptr = self.split = None
        self.col, self.val = cp.asarray(c.ravel()), cp.asarray(v.ravel())
        self.wa, self.wb = wa, wb

    @classmethod
    def device_csr(cls, op):
        """CSR rows of a CuPy CSR operator, without a copy."""
        self = cls.__new__(cls)
        self.vdtype, self.ell = op.data.dtype, False
        self.ptr, self.split, self.col, self.val = op.indptr, None, op.indices, op.data
        self.wa = self.wb = 0
        return self

    def args(self):
        return (self.ptr, self.split, self.col, self.val, np.int32(self.wa),
                np.int32(self.wb))  # fmt: skip

    def name(self, kernel, T):
        ell = "true" if self.ell else "false"
        return f"{kernel}<{T},{tname(self.vdtype)},{ell}>"


def _block(ops, terms, sign):
    """3N x 3N block matrix of the `sign` terms: block (T, S) = op."""
    import scipy.sparse as sp

    blocks = [[None] * 3 for _ in range(3)]
    for T, d, S, s, op, prefix in terms:
        if s == sign:
            blocks["xyz".index(T)]["xyz".index(S)] = ops[op]
    return sp.bmat(blocks, format="csr")


class _Half:
    """Device data of one CPML half step (merged curl rows, psi slots)."""

    def __init__(self, solver, terms, field):
        import scipy.sparse as sp

        N = solver.N
        ops = {t[4]: _host_csr(getattr(solver, t[4])) for t in terms}
        A = _block(ops, terms, 1.0)
        B = _block(ops, terms, -1.0)
        self.rows = RowOp(*merge_rows(A, B))
        # Transposed rows for the adjoint: [A^H B^H] acting on [beta_a; beta_b]
        AT = sp.hstack([A.conj().T, B.conj().T], format="csr")
        AT.sort_indices()
        self.t_ptr = cp.asarray(AT.indptr.astype(np.int32))
        self.t_col = cp.asarray(AT.indices.astype(np.int32))
        self.t_val = cp.asarray(AT.data)
        self.t_vdtype = AT.data.dtype
        # psi slots of each face, in a fixed order
        slot_a = np.full(3 * N, -1, dtype=np.int64)
        slot_b = np.full(3 * N, -1, dtype=np.int64)
        names, sizes, bs, cs = [], [], [], []
        off = 0
        for T, d, S, s, op, prefix in terms:
            for side, bcs in (("low", solver.bc_low), ("high", solver.bc_high)):
                if bcs["xyz".index(d)].lower() != "cpml":
                    continue
                idx = _host(getattr(solver, f"idx_{d}_{side}")).astype(np.int64)
                rows = "xyz".index(T) * N + idx
                slots = slot_a if s > 0 else slot_b
                if np.any(slots[rows] >= 0) or np.unique(rows).size != rows.size:
                    raise NotImplementedError(
                        "Overlapping CPML faces (domain thinner than 2 n_pml)."
                    )
                slots[rows] = off + np.arange(idx.size)
                names.append(f"{prefix}_{side}")
                sizes.append(idx.size)
                bs.append(_host(getattr(solver, f"pml_b_{field}_{d}_{side}")))
                cs.append(_host(getattr(solver, f"pml_c_{field}_{d}_{side}")))
                off += idx.size
        dtype = solver.E.array.dtype
        self.names = names
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.slot_a = cp.asarray(slot_a.astype(np.int32))
        self.slot_b = cp.asarray(slot_b.astype(np.int32))
        self.pb = cp.asarray(np.concatenate(bs) if bs else np.zeros(1), dtype=dtype)
        self.pc = cp.asarray(np.concatenate(cs) if cs else np.zeros(1), dtype=dtype)
        self.n_psi = int(off)


class FusedStep:
    """
    Fused GPU time step of a `SolverFIT3D` (standard or CPML update).

    Installed as ``solver.one_step`` by ``SolverFIT3D(..., fused=True)``;
    wrappers (tensor averaging, ADE, ABC) call it as their base step.

    The CPML auxiliary variables of the solver (``psiHa_x_low``, ...) become
    views into one buffer per half step owned by this object; arrays assigned
    to those attributes from outside are copied into the buffer at the next
    step. Fields, materials and currents are read from the solver at every
    step, so in-place changes (e.g. ``solver.ieps`` set by a design) take
    effect.
    """

    def __init__(self, solver):
        self.solver = solver
        self.ready = False
        self.cpml = solver.activate_cpml
        self.views = {}
        self.pre, self.post = [], []  # device-only hooks around the step

    def _build(self):
        s = self.solver
        self.T = tname(s.E.array.dtype)
        if self.cpml:
            self.h = _Half(s, _H_TERMS, "H")
            self.e = _Half(s, _E_TERMS, "E")
            dtype = s.E.array.dtype
            self.psi_h = cp.zeros(max(self.h.n_psi, 1), dtype=dtype)
            self.psi_e = cp.zeros(max(self.e.n_psi, 1), dtype=dtype)
            for half, buf in ((self.h, self.psi_h), (self.e, self.psi_e)):
                o = half.offsets
                for k, name in enumerate(half.names):
                    view = buf[o[k] : o[k + 1]]
                    view[...] = cp.asarray(getattr(s, name))
                    setattr(s, name, view)
                    self.views[name] = view
        self.ready = True

    def _sync_psi(self):
        """Copy psi arrays rebound from outside into the buffer views."""
        s = self.solver
        for name, view in self.views.items():
            cur = getattr(s, name)
            if cur is not view:
                view[...] = cp.asarray(cur)
                setattr(s, name, view)

    def __call__(self):
        s = self.solver
        if s.source_type == "tfsf" and not s.injection_done:
            # Legacy TF/SF injection inside the step: unfused path
            for hook in self.pre:
                hook()
            (s._one_step_cpml if self.cpml else s._one_step)()
            for hook in self.post:
                hook()
            return
        if s.step_0:
            s._set_ghosts_to_0()
            s.step_0 = False
            s._attrcleanup()
            if s.source_type == "direct" or s.use_conductivity:
                s.J_old = cp.zeros_like(s.J.array)
        if not self.ready:
            self._build()
        if self.cpml:
            self._sync_psi()
        self.launch_step()

    def launch_step(self):
        """Launch the step kernels and hooks (no host synchronization)."""
        for hook in self.pre:
            hook()
        self._launch_halves()
        for hook in self.post:
            hook()

    def _launch_halves(self):
        """Launch the two half-step kernels."""
        s = self.solver
        n = 3 * s.N
        E, H, J = s.E.array, s.H.array, s.J.array
        M = None if s.M is None else s.M.array
        dt = np.float64(s.dt)
        T = self.T
        if s.use_conductivity:
            J_old, sigma = s.J_old, s.sigma.array
        else:
            J_old = sigma = None
        if self.cpml:
            h, e = self.h.rows, self.e.rows
            launch(h.name("fused_h", T), n, np.int32(n), np.int32(1), *h.args(),
                   E, H, dt, self.h.slot_a, self.h.slot_b, self.psi_h, self.h.pb,
                   self.h.pc, M, s.imu.array)  # fmt: skip
            launch(e.name("fused_e", T), n, np.int32(n), np.int32(1), *e.args(),
                   H, E, s.ieps.array, dt, self.e.slot_a, self.e.slot_b,
                   self.psi_e, self.e.pb, self.e.pc, J, J_old, sigma)  # fmt: skip
            return
        A, B = self._rows("tDsiDmuiDaC"), self._rows("itDaiDepsDstC")
        launch(A.name("fused_h", T), n, np.int32(n), np.int32(0), *A.args(), E, H,
               dt, None, None, None, None, None, M, s.imu.array)  # fmt: skip
        launch(B.name("fused_e", T), n, np.int32(n), np.int32(0), *B.args(), H, E,
               s.ieps.array, dt, None, None, None, None, None, J, J_old,
               sigma)  # fmt: skip

    def _rows(self, name):
        """RowOp of a solver operator, rebuilt when the operator changes."""
        op = self._dev_op(name)
        key = (id(op), op.data.data.ptr)
        cache = self.__dict__.setdefault("_row_cache", {})
        if cache.get(name, (None,))[0] != key:
            cache[name] = (key, RowOp.device_csr(op), op)
        return cache[name][1]

    def _dev_op(self, name):
        """Solver operator as CuPy CSR (`update_tensors` rebuilds SciPy ones)."""
        op = getattr(self.solver, name)
        if not type(op).__module__.startswith("cupyx"):
            from cupyx.scipy.sparse import csr_matrix

            op = csr_matrix(_host_csr(op))
            setattr(self.solver, name, op)
        return op

    def arrays(self):
        """Device arrays the kernels point to (CUDA graph validity check)."""
        s = self.solver
        out = [s.E.array, s.H.array, s.J.array, s.ieps.array, s.imu.array]
        out.append(None if s.M is None else s.M.array)
        if s.use_conductivity:
            out += [s.J_old, s.sigma.array]
        if not self.cpml:
            out += [self._rows(k).val for k in ("tDsiDmuiDaC", "itDaiDepsDstC")]
        return out


def eligible(solver):
    """True if `solver` can use the fused step (GPU, no MPI, float64/complex128)."""
    return (
        _HAS_CUPY
        and solver.use_gpu
        and not (solver.use_mpi and getattr(solver.grid, "use_mpi", False))
        and np.dtype(solver.E.array.dtype) in _TYPES
    )


def is_fused(solver):
    """True if `solver` time-steps with `FusedStep`."""
    return getattr(solver, "_fused_step", None) is not None


# ------------------------------------------------------------------ sources


def idx32(a):
    """Contiguous int32 device copy of an index array."""
    return cp.ascontiguousarray(cp.asarray(a), dtype=np.int32)


def source_update(arr, idx, weights, a, old, step=None, amp=None, add=0):
    """arr[idx] += weights * a - old; old[:] = weights * a (one kernel)."""
    launch(
        f"source_delta<{tname(arr.dtype)},{tname(weights.dtype)}>", idx.size,
        np.int32(idx.size), idx, weights, np.float64(a), amp, step,
        np.int64(add), old, arr,
    )  # fmt: skip


# ------------------------------------------------------------- running DFT

_FIELD = {"E": 0, "H": 1, "J": 2}
_C1 = -2j * np.pi  # the factor of the monitors' exp(-2j * pi * f * t)


def monitor_terms(monitor, solver):
    """
    DFT terms of an initialized monitor, or None if it is not supported:
    list of (field, idx_a, idx_b or None, time offset [s], (container, key, d)).

    Each term accumulates acc[f, k] += dt exp(-2 pi i f (t + toff)) v_k with
    v = F[idx_a] (or 0.5 (F[idx_a] + F[idx_b])); its accumulator is
    container[key] (or container[key][d]).
    """
    from .field_monitors import DFTMonitor, FluxMonitor
    from .mode_solver import ModeMonitor

    dt = solver.dt
    out = []
    if type(monitor) is DFTMonitor:
        times = {"E": dt, "J": dt, "H": dt / 2}
        for comp in monitor.components:
            out.append(
                (comp[0], monitor._flat[comp], None, times[comp[0]],
                 (monitor._acc, comp, None))
            )  # fmt: skip
    elif type(monitor) is FluxMonitor:
        for term in monitor._terms:
            out.append(("E", term["E"], None, dt, (term, "E_acc", None)))
            out.append(
                ("H", term["H_above"], term["H_below"], dt / 2, (term, "H_acc", None))
            )
    elif isinstance(monitor, ModeMonitor):
        n = monitor.plane.n_b * monitor.plane.n_c
        for f, toff in (("E", dt), ("H", dt / 2)):
            for d in range(2):
                idx = monitor._idx[f][d]
                if f == "H":
                    # values = 0.5 * (H[below] + H[above])
                    out.append((f, idx[:n], idx[n:], toff, (monitor._acc, f, d)))
                else:
                    out.append((f, idx, None, toff, (monitor._acc, f, d)))
    else:
        return None
    return out


def monitor_supported(monitor):
    """True for monitor types `monitor_terms` handles (initialized or not)."""
    from .field_monitors import DFTMonitor, FluxMonitor
    from .mode_solver import ModeMonitor

    return type(monitor) in (DFTMonitor, FluxMonitor) or isinstance(
        monitor, ModeMonitor
    )


class DFTBatch:
    """
    One-kernel running DFT of several monitors (all terms, all frequencies).

    The accumulators of the monitors are replaced by views into one buffer
    (current values are copied), so `get_field`, `get_flux`, the mode
    amplitudes and the adjoint readouts work unchanged.

    Parameters
    ----------
    monitors : list
        Initialized `DFTMonitor`, `FluxMonitor` or `ModeMonitor` objects.
    solver : SolverFIT3D
        GPU solver.
    """

    def __init__(self, monitors, solver):
        self.monitors = list(monitors)
        terms, freqs = [], []
        f_off = 0
        for mon in self.monitors:
            mt = monitor_terms(mon, solver)
            if mt is None:
                raise TypeError(f"Unsupported monitor {type(mon).__name__}")
            terms += [(*t, f_off, mon.frequencies.size) for t in mt]
            freqs.append(np.asarray(mon.frequencies, dtype=float))
            f_off += mon.frequencies.size
        self.freqs = cp.asarray(np.concatenate(freqs) if freqs else np.zeros(1))
        m = [int(t[1].size) for t in terms]
        start = np.concatenate([[0], np.cumsum([mi * t[6] for mi, t in zip(m, terms)])])
        start = start.astype(np.int64)
        self.total = int(start[-1])
        self.buf = cp.zeros(max(self.total, 1), dtype=np.complex128)
        ia, ib = [], []
        for k, t in enumerate(terms):
            field, a, b, toff, (cont, key, d), fo, nf = t
            view = self.buf[start[k] : start[k + 1]].reshape(nf, m[k])
            if d is None:
                view[...] = cont[key]
                cont[key] = view
            else:
                view[...] = cont[key][d]
            ia.append(idx32(a))
            ib.append(idx32(a if b is None else b))
        # (2, n_f, n) mode accumulators: one view over their two halves
        for k, t in enumerate(terms):
            field, a, b, toff, (cont, key, d), fo, nf = t
            if d == 0:
                k1 = k + 1
                u = terms[k1][4]
                if not (u[0] is cont and u[1] == key and u[2] == 1):
                    raise RuntimeError("Mode accumulator halves not adjacent.")
                cont[key] = self.buf[start[k] : start[k1 + 1]].reshape(2, nf, m[k])
        self.ia = cp.concatenate(ia) if ia else cp.zeros(1, np.int32)
        self.ib = cp.concatenate(ib) if ib else cp.zeros(1, np.int32)
        soff = np.concatenate([[0], np.cumsum(m)])[:-1] if m else np.zeros(0)
        nfs = [t[6] for t in terms]
        poff = np.concatenate([[0], np.cumsum(nfs)])
        self.nph = int(poff[-1])
        self.ph = cp.zeros(max(self.nph, 1), dtype=np.complex128)
        self.ph_seg = cp.asarray(np.repeat(np.arange(len(terms)), nfs).astype(np.int32))
        self.ph_f = cp.asarray(
            np.concatenate([t[5] + np.arange(t[6]) for t in terms]).astype(np.int32)
            if terms
            else np.zeros(1, np.int32)
        )
        self.nseg = len(terms)
        self.seg_start = cp.asarray(start)
        self.seg_m = cp.asarray(np.asarray(m, dtype=np.int32))
        self.seg_soff = cp.asarray(np.asarray(soff, dtype=np.int32))
        self.seg_poff = cp.asarray(poff[:-1].astype(np.int32))
        self.seg_field = cp.asarray(
            np.asarray([_FIELD[t[0]] for t in terms], dtype=np.int32)
        )
        self.seg_avg = cp.asarray(
            np.asarray([t[2] is not None for t in terms], dtype=np.int32)
        )
        self.seg_toff = cp.asarray(np.asarray([t[3] for t in terms], dtype=float))
        self.seg_nf = cp.asarray(np.asarray(nfs, dtype=np.int32))
        self.n_samples = int(sum(m))
        self.T = tname(solver.E.array.dtype)
        for mon in self.monitors:
            mon._dft_batch = self

    def update(self, solver, t, step=None, add=0):
        """Accumulate the fields after the step started at time t."""
        if self.total == 0:
            return
        launch(
            "dft_phase", self.nph, np.int32(self.nph), self.ph_seg, self.ph_f,
            self.seg_toff, self.freqs, np.float64(_C1.real), np.float64(_C1.imag),
            np.float64(solver.dt), np.float64(t), step, np.int64(add), self.ph,
        )  # fmt: skip
        launch(
            f"dft_acc<{self.T}>", self.n_samples, np.int64(self.n_samples),
            np.int32(self.nseg), self.seg_start, self.seg_m, self.seg_soff,
            self.seg_poff, self.seg_nf, self.seg_field, self.seg_avg, self.ia,
            self.ib, self.ph, solver.E.array, solver.H.array, solver.J.array,
            self.buf,
        )  # fmt: skip


def monitor_update(monitor, solver, t):
    """
    Fused `update` of a DFT, flux or mode monitor on a fused GPU solver.

    Returns False if the monitor type is not supported (the caller then runs
    its unfused update). A monitor that belongs to a multi-monitor batch is
    updated by that batch's owner, not here.
    """
    batch = getattr(monitor, "_dft_batch", None)
    if batch is None:
        if monitor_terms(monitor, solver) is None:
            return False
        batch = DFTBatch([monitor], solver)
    if len(batch.monitors) != 1:
        raise RuntimeError("Monitor is part of a DFTBatch; update the batch.")
    batch.update(solver, t)
    return True


# ------------------------------------------------------------- CUDA graphs


class GraphRunner:
    """
    Run a device-only step function many times through a CUDA graph.

    `step_fn()` must only launch kernels on the current stream (no
    allocations, host synchronizations or host-dependent arguments that
    change between calls): it is captured `chunk` times into one graph,
    which is then replayed. Python state changed by `step_fn` during the
    capture (e.g. ping-pong buffer choices) must return to its initial value
    after `chunk` calls (use an even `chunk` for two-buffer schemes), since
    the capture does not execute anything.

    Parameters
    ----------
    step_fn : callable
        Launches the kernels of one step.
    chunk : int
        Steps per graph.
    """

    def __init__(self, step_fn, chunk=64):
        self.step_fn = step_fn
        self.chunk = int(chunk)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None

    def _enter(self):
        ev = cp.cuda.Event(disable_timing=True)
        ev.record(cp.cuda.get_current_stream())
        self.stream.wait_event(ev)

    def _exit(self):
        ev = cp.cuda.Event(disable_timing=True)
        ev.record(self.stream)
        cp.cuda.get_current_stream().wait_event(ev)

    def run(self, n_steps):
        """Run `n_steps` steps (graph replays plus direct launches)."""
        n_steps = int(n_steps)
        self._enter()
        with self.stream:
            if n_steps >= self.chunk:
                if self.graph is None:
                    self.stream.begin_capture()
                    try:
                        for _ in range(self.chunk):
                            self.step_fn()
                    finally:
                        self.graph = self.stream.end_capture()
                for _ in range(n_steps // self.chunk):
                    self.graph.launch(self.stream)
            for _ in range(n_steps % self.chunk):
                self.step_fn()
        self._exit()


def _source_terms(src, solver, Nt):
    """
    (array, idx, weights, old, amplitude table) of an initialized source, or
    None if the source is not supported by `FusedLoop`.
    """
    from .mode_solver import ModeSource
    from .sources import CurrentSource, MagneticCurrentSource

    dt = solver.dt
    times = np.arange(Nt) * dt  # n * dt, as in the time loops
    if type(src) in (CurrentSource, MagneticCurrentSource):
        if type(src) is CurrentSource:
            arr, amp = solver.J.array, [float(src.src_time(t + dt / 2)) for t in times]
        else:
            arr, amp = solver.M.array, [float(src.src_time(t)) for t in times]
        if getattr(src, "_idx32", None) is None:
            return None
        return [(arr, src._idx32, src.weights, src.J_old, cp.asarray(amp))]
    if isinstance(src, ModeSource):
        out = []
        for term in src.terms:
            if "idx32" not in term:
                return None
            if term["field"] == "J":
                amp = [float(src.src_time(t + dt / 2 + src.h_advance)) for t in times]
            else:
                amp = [float(src.src_time(t)) for t in times]
            arr = getattr(solver, term["field"]).array
            out.append((arr, term["idx32"], term["weights"], term["old"], cp.asarray(amp)))
        return out
    return None


class FusedLoop:
    """
    Time loop of a fused solver with impressed sources and DFT monitors,
    replayed as CUDA graphs: sources, step, monitors (and optionally the
    adjoint design increments) of K steps per graph launch.

    Call `supported(...)` first; step 0 must have run through the regular
    loop (it initializes the sources, monitors and the fused step).

    Parameters
    ----------
    solver : SolverFIT3D
        Fused GPU solver whose `one_step` is the bare `FusedStep`.
    sources : list
        `CurrentSource`, `MagneticCurrentSource` or `ModeSource` objects.
    monitors : list
        `DFTMonitor`, `FluxMonitor` or `ModeMonitor` objects.
    Nt : int
        Total number of steps of the run (size of the amplitude tables).
    record : tuple, optional
        (idx, ieps_d, e_old, increments): record row n of the adjoint design
        increments (E^{n+1} - E^n)[idx] / ieps_d.
    chunk : int, optional
        Steps per CUDA graph (default 64).
    """

    def __init__(self, solver, sources, monitors, Nt, record=None, chunk=64):
        self.solver = solver
        self.terms, self.ops, self.finish = [], [], []
        for src in sources:
            if hasattr(src, "fused_loop_op"):
                op = src.fused_loop_op(solver, Nt)
                if op is None:
                    raise ValueError(f"{type(src).__name__} cannot run in a FusedLoop.")
                self.ops.append(op[0])
                self.finish.append(op[1])
                continue
            for term in _source_terms(src, solver, Nt):
                self.terms.append(term)
                self.ops.append(self._source_op(term))
        self.monitors = list(monitors)
        self.batch = DFTBatch(self.monitors, solver) if self.monitors else None
        self.record = record
        self.T = tname(solver.E.array.dtype)
        self.counter = cp.zeros(1, dtype=np.int64)
        self.runner = GraphRunner(self._one, chunk)
        self.step = solver._fused_step
        self._ptrs = self._pointers()

    @staticmethod
    def supported(solver, sources, monitors, elements=()):
        """True if the loop can run as CUDA graphs (after step 0)."""
        s = solver
        if not is_fused(s) or s.one_step is not s._fused_step or elements:
            return False
        if s.source_type == "tfsf" and not s.injection_done:
            return False
        if any(_source_terms_ok(src) is False for src in sources):
            return False
        return all(monitor_supported(m) for m in monitors)

    def _pointers(self):
        arrs = self.step.arrays() + [t[0] for t in self.terms]
        return [None if a is None else a.data.ptr for a in arrs]

    @staticmethod
    def _source_op(term):
        arr, idx, w, old, amp = term

        def op(c):
            source_update(arr, idx, w, 0.0, old, step=c, amp=amp)

        return op

    def _one(self):
        s = self.solver
        c = self.counter
        for op in self.ops:
            op(c)
        if self.record is not None:
            idx, ieps_d, e_old, inc = self.record
            m = np.int32(idx.size)
            launch(f"gather<{self.T}>", m, m, idx, s.E.array, e_old)
        self.step.launch_step()
        if self.record is not None:
            launch(f"record_row<{self.T},{rname(inc.dtype)}>", m, m, idx,
                   s.E.array, e_old, ieps_d, inc, c, np.int64(0),
                   np.int64(0))  # fmt: skip
        if self.batch is not None:
            self.batch.update(s, 0.0, step=c)
        launch("step_add", 1, c, np.int64(1))

    def run(self, n0, n1):
        """Run steps n0 .. n1 - 1 (n0 >= 1)."""
        if n1 <= n0:
            return
        if self._pointers() != self._ptrs:
            raise RuntimeError("Solver arrays were reallocated during a fused loop.")
        self.counter.fill(n0)
        self.runner.run(n1 - n0)
        for finish in self.finish:
            finish(n1)

    def close(self):
        """Hand the monitors back to their own (per-monitor) fused updates."""
        for m in self.monitors:
            m._dft_batch = None


def _source_terms_ok(src):
    from .mode_solver import ModeSource
    from .sources import CurrentSource, MagneticCurrentSource

    if hasattr(src, "fused_loop_op"):  # TFSFSource
        return not src._mutation_time_shift
    return type(src) in (CurrentSource, MagneticCurrentSource) or isinstance(
        src, ModeSource
    )


# ------------------------------------------------------- tensor averaging


class TensorHooks:
    """
    Fused correction of `wakis.tensor_averaging.TensorOperator` (conformal
    'tensor'): E <- E* + K'(E* - E) on the coupled edges, as three kernels
    around the fused step (gather of E, increment, CSR product), with fixed
    buffers so that the step stays capturable in a CUDA graph.
    """

    def __init__(self, op, solver):
        self.op, self.solver = op, solver
        self.key = None

    def _buffers(self):
        op = self.op
        key = (id(op.Kp), int(op.rows.size))
        if key != self.key:
            self.rows = idx32(op.rows_dev)
            self.e_old = cp.zeros(op.rows.size, dtype=self.solver.E.array.dtype)
            self.delta = cp.zeros_like(self.e_old)
            K = op.Kp
            self.ptr, self.col = idx32(K.indptr), idx32(K.indices)
            self.val = K.data
            self.key = key

    def pre(self):
        if self.op.rows.size == 0:
            return
        self._buffers()
        n = np.int32(self.rows.size)
        launch(f"gather<{tname(self.e_old.dtype)}>", n, n, self.rows,
               self.solver.E.array, self.e_old)  # fmt: skip

    def post(self):
        if self.op.rows.size == 0:
            return
        E = self.solver.E.array
        T = tname(E.dtype)
        n = np.int32(self.rows.size)
        launch(f"tensor_delta<{T}>", n, n, self.rows, E, self.e_old, self.delta)
        launch(f"rows_spmv_add<{T},{tname(self.val.dtype)}>", n, n, self.rows,
               self.ptr, self.col, self.val, self.delta, E)  # fmt: skip
        self.op.last_increment = self.delta


# ------------------------------------------------------- dispersive (ADE)


class ADEHooks:
    """
    Fused ADE correction of the dispersive groups of a real-field solver
    (`DispersiveMixin._one_step_dispersive`): a gather of E^n per group before
    the step and one kernel per group after it, bit for bit the CuPy
    arithmetic. The pole polarizations of each group live in fixed buffers;
    the arrays in ``group['lorentz']`` / ``group['debye']`` are views into
    them, and arrays assigned there from outside are copied in before the
    next step.
    """

    def __init__(self, solver):
        self.solver = solver
        self.state = {}

    def _group_state(self, g):
        st = self.state.get(id(g))
        n = int(g["idx"].size)
        nL, nD = len(g["lorentz"]), len(g["debye"])
        if st is None or st["n"] != n or st["g"] is not g:
            st = {
                "g": g,
                "n": n,
                "idx": idx32(g["idx"]),
                "En": cp.zeros(n),
                "P": cp.zeros((max(nL, 1), n)),
                "Pold": cp.zeros((max(nL, 1), n)),
                "D": cp.zeros((max(nD, 1), n)),
                "cL": cp.asarray(np.ravel([cf[:3] for cf in g["lorentz"]] or [0.0])),
                "cD": cp.asarray(np.ravel([cf[:2] for cf in g["debye"]] or [0.0])),
            }
            for l, cf in enumerate(g["lorentz"]):
                st["P"][l] = cf[3]
                st["Pold"][l] = cf[4]
            for d, cf in enumerate(g["debye"]):
                st["D"][d] = cf[2]
            g["lorentz"] = [(*cf[:3], st["P"][l], st["Pold"][l])
                            for l, cf in enumerate(g["lorentz"])]  # fmt: skip
            g["debye"] = [(*cf[:2], st["D"][d]) for d, cf in enumerate(g["debye"])]
            st["views"] = ([(cf[3], cf[4]) for cf in g["lorentz"]],
                           [cf[2] for cf in g["debye"]])  # fmt: skip
            self.state[id(g)] = st
            return st
        # Arrays rebound from outside (adjoint set_state, reset): copy them in
        lv, dv = st["views"]
        for l, cf in enumerate(g["lorentz"]):
            if cf[3] is not lv[l][0] or cf[4] is not lv[l][1]:
                lv[l][0][...] = cf[3]
                lv[l][1][...] = cf[4]
                g["lorentz"][l] = (*cf[:3], lv[l][0], lv[l][1])
        for d, cf in enumerate(g["debye"]):
            if cf[2] is not dv[d]:
                dv[d][...] = cf[2]
                g["debye"][d] = (*cf[:2], dv[d])
        return st

    def pre(self):
        E = self.solver.E.array
        for g in self.solver._dispersive_groups:
            st = self._group_state(g)
            n = np.int32(st["n"])
            launch("gather<double>", n, n, st["idx"], E, st["En"])

    def post(self):
        E = self.solver.E.array
        for g in self.solver._dispersive_groups:
            st = self.state[id(g)]
            n = np.int32(st["n"])
            launch(
                "ade_update", n, n, st["idx"], E, st["En"], g["eps_s"], g["a1"],
                g["den"], g["weight"], np.int32(len(g["lorentz"])), st["cL"],
                st["P"], st["Pold"], np.int32(len(g["debye"])), st["cD"], st["D"],
            )  # fmt: skip
