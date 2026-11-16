# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Fused GPU time stepping for 3D Cartesian subgridding (`SubgridFIT3D` and
`SubgridLTSFIT3D`).

The CuPy `BlockOps` path (`wakis/subgrid3d/gpu.py`) issues separate slice
kernels per block and axis in `curl_all` and `curl_T_all`, separate sparse
products for `Ph`, `PhT`, the LTS level operators and the CPML rows, and
allocates temporaries per step. On an NVIDIA L4 this makes the subgrid step
launch-latency bound (0.8201-0.8249 ms/step on uniform grids and
1.8215-2.8940 ms/step with a patch and LTS).

This module fuses the subgrid step into a handful of zero-allocation device
kernels compiled with ``--fmad=false``:

* `FusedTables`: host gather tables folding the trace prolongation `P_e` into
  an edge-source table (<= 2 parent free edges per all-edge) and the
  restriction `P_e^T` into a CSR transpose table over hanging edges, with a
  NumPy emulation (`apply_K`, `apply_KH`) that reproduces `BlockOps.K` and
  `BlockOps.KH` bit for bit.
* `sg_faraday`: one thread per face of all blocks. Decodes `(block, a, i, j, k)`,
  evaluates the four edges of the face through the edge-source table, forms
  the curl `(e_b0 + e_c1) - (e_b1 + e_c0)` and its two derivative-axis splits
  `e_c1 - e_c0` and `e_b0 - e_b1` where CPML `psi` rows exist, updates `psi`
  and updates `b`.
* `sg_ampere`: one thread per free edge. Evaluates the <= 4 faces of the edge
  in its own block in the pass order of `BlockOps.curl_T_all`, adds the faces
  of its hanging children through `P_h^T`, updates the CPML `phi` rows,
  applies the precomputed current-source samples, and scales by `dt * D^-1`.
* `sg_lts_face` and `sg_lts_edge`: per LTS level and Chebyshev substep, one
  face kernel on `Phi_l` (with `K_F` and `K_C` merged via
  `wakis.gpu_fused.merge_rows`) and one edge kernel on `F_l` (`KH_F`, `D^-1`,
  three-term Chebyshev recurrence), replicating `SubgridLTS._corr` and `_R`
  without intermediate gather/scatter kernels.
* `sg_probe_dft`: records all point probes and accumulates their running DFTs
  on the device.
* `FusedSubgridStepper`: wraps a constructed `SubgridFIT3D` or
  `SubgridLTSFIT3D`, preallocates all device buffers once, and replays chunks
  of `graph_chunk` steps through `wakis.gpu_fused.GraphRunner`.
"""

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None
    _HAS_CUPY = False

from ..gpu_fused import GraphRunner, RowOp, idx32, merge_rows

__all__ = ["FusedSubgridStepper", "FusedTables"]

_THREADS = 256
_VALID_MUTATIONS = (None, "no_hanging", "face_offset", "lts_kappa")

_SOURCE = r"""
#include <cupy/complex.cuh>
typedef complex<double> cplx;

extern "C" {

__device__ __forceinline__ double eval_free(
        const int f_idx, const double* __restrict__ z,
        const double* __restrict__ z_f1, const int* __restrict__ f1_pos) {
    if (z_f1) {
        int p = f1_pos[f_idx];
        if (p >= 0) return z_f1[p];
    }
    return z[f_idx];
}

__device__ __forceinline__ double eval_edge(
        const int e_all, const int* __restrict__ src_i0,
        const int* __restrict__ src_i1, const double* __restrict__ src_w0,
        const double* __restrict__ src_w1, const double* __restrict__ z,
        const double* __restrict__ z_f1, const int* __restrict__ f1_pos) {
    int i0 = src_i0[e_all];
    if (i0 < 0) return 0.0;
    double v = src_w0[e_all] * eval_free(i0, z, z_f1, f1_pos);
    int i1 = src_i1[e_all];
    if (i1 >= 0) v += src_w1[e_all] * eval_free(i1, z, z_f1, f1_pos);
    return v;
}

// ------------------------------------------------------------------ Faraday
// One thread per face f in 0 .. nF - 1 across all blocks.
// blk_meta layout per block (stride 32 ints):
//   [0..3]   eo[0..3] (global all-edge offsets)
//   [4..7]   fo[0..3] (global all-face offsets)
//   [8..16]  es[3][3] (edge shapes)
//   [17..25] fs[3][3] (face shapes)
//   [26..28] nc[3]    (cell counts)
__global__ void sg_faraday(
        const int nF, const int n_blocks, const int* __restrict__ blk_meta,
        const int face_shift, const int* __restrict__ src_i0,
        const int* __restrict__ src_i1, const double* __restrict__ src_w0,
        const double* __restrict__ src_w1, const double* __restrict__ z,
        const double* __restrict__ z_f1, const int* __restrict__ f1_pos,
        const double dt, const int only_K, const int* __restrict__ psi_slot0,
        const int* __restrict__ psi_slot1, double* __restrict__ psi,
        const double* __restrict__ bh, const double* __restrict__ ah,
        double* __restrict__ b) {
    int f = blockDim.x * blockIdx.x + threadIdx.x;
    if (f >= nF) return;
    int bi = 0;
    while (bi + 1 < n_blocks && f >= blk_meta[(bi + 1) * 32 + 4]) ++bi;
    const int* bm = blk_meta + bi * 32;
    int a = (f >= bm[5]) + (f >= bm[6]);
    int f_start = bm[4 + a], f_end = bm[5 + a];
    int n_comp = f_end - f_start;
    int loc = f - f_start;
    if (face_shift) loc = (loc + face_shift) % n_comp;
    int fs1 = bm[17 + 3 * a + 1], fs2 = bm[17 + 3 * a + 2];
    int k = loc % fs2, tmp = loc / fs2;
    int j = tmp % fs1, i = tmp / fs1;
    int q[3] = {i, j, k};

    int eb = (a + 1) % 3, ec = (a + 2) % 3;
    int esb1 = bm[8 + 3 * eb + 1], esb2 = bm[8 + 3 * eb + 2];
    int e_b0 = bm[eb] + (q[0] * esb1 + q[1]) * esb2 + q[2];
    int stride_b_c = (ec == 0) ? (esb1 * esb2) : ((ec == 1) ? esb2 : 1);
    int e_b1 = e_b0 + stride_b_c;

    int esc1 = bm[8 + 3 * ec + 1], esc2 = bm[8 + 3 * ec + 2];
    int e_c0 = bm[ec] + (q[0] * esc1 + q[1]) * esc2 + q[2];
    int stride_c_b = (eb == 0) ? (esc1 * esc2) : ((eb == 1) ? esc2 : 1);
    int e_c1 = e_c0 + stride_c_b;

    double vb0 = eval_edge(e_b0, src_i0, src_i1, src_w0, src_w1, z, z_f1, f1_pos);
    double vc1 = eval_edge(e_c1, src_i0, src_i1, src_w0, src_w1, z, z_f1, f1_pos);
    double vb1 = eval_edge(e_b1, src_i0, src_i1, src_w0, src_w1, z, z_f1, f1_pos);
    double vc0 = eval_edge(e_c0, src_i0, src_i1, src_w0, src_w1, z, z_f1, f1_pos);

    // Same parenthesization as BlockOps.curl_all: (vb0 + vc1) - (vb1 + vc0)
    double kz = (vb0 + vc1) - (vb1 + vc0);
    if (only_K) {
        b[f] = kz;
        return;
    }
    if (psi_slot0) {
        double c_lo = (eb < ec) ? (vc1 - vc0) : (vb0 - vb1);
        double c_hi = (eb < ec) ? (vb0 - vb1) : (vc1 - vc0);
        int s0 = psi_slot0[f];
        if (s0 >= 0) {
            double p = bh[s0] * psi[s0] + ah[s0] * c_lo;
            psi[s0] = p;
            kz += p;
        }
        int s1 = psi_slot1[f];
        if (s1 >= 0) {
            double p = bh[s1] * psi[s1] + ah[s1] * c_hi;
            psi[s1] = p;
            kz += p;
        }
    }
    b[f] -= dt * kz;
}

// ------------------------------------------------------------------- Ampere
// Evaluate C^T (W_all * b) on one all-edge e_all in the pass order of
// BlockOps.curl_T_all (a = 0, 1, 2; within each a: += then -=), and split
// into the two derivative axes d_lo < d_hi for CPML phi.
__device__ __forceinline__ double edge_curl_T(
        const int e_all, const int n_blocks, const int* __restrict__ blk_meta,
        const int face_shift, const double* __restrict__ b,
        const double* __restrict__ W_all, double& q_lo, double& q_hi) {
    int bi = 0;
    while (bi + 1 < n_blocks && e_all >= blk_meta[(bi + 1) * 32]) ++bi;
    const int* bm = blk_meta + bi * 32;
    int dir_e = (e_all >= bm[1]) + (e_all >= bm[2]);
    int loc = e_all - bm[dir_e];
    int es1 = bm[8 + 3 * dir_e + 1], es2 = bm[8 + 3 * dir_e + 2];
    int k = loc % es2, tmp = loc / es2;
    int j = tmp % es1, i = tmp / es1;
    int q[3] = {i, j, k};
    double g = 0.0;
    q_lo = 0.0;
    q_hi = 0.0;
    for (int a = 0; a < 3; ++a) {
        if (a == dir_e) continue;
        int d = 3 - a - dir_e;
        int f_start = bm[4 + a], f_end = bm[5 + a];
        int n_comp = f_end - f_start;
        int fs1 = bm[17 + 3 * a + 1], fs2 = bm[17 + 3 * a + 2];
        int nc_d = bm[26 + d];
        int base = (q[0] * fs1 + q[1]) * fs2 + q[2];
        int stride_d = (d == 0) ? (fs1 * fs2) : ((d == 1) ? fs2 : 1);
        int f_curr = -1, f_prev = -1;
        if (q[d] < nc_d) {
            int lc = face_shift ? ((base + face_shift) % n_comp) : base;
            f_curr = f_start + lc;
        }
        if (q[d] >= 1) {
            int lp = base - stride_d;
            if (face_shift) lp = (lp + face_shift) % n_comp;
            f_prev = f_start + lp;
        }
        int f_plus  = (dir_e == (a + 1) % 3) ? f_curr : f_prev;
        int f_minus = (dir_e == (a + 1) % 3) ? f_prev : f_curr;
        double q_d = 0.0;
        if (f_plus >= 0) {
            double yp = W_all ? (W_all[f_plus] * b[f_plus]) : b[f_plus];
            g += yp;
            if (!W_all || W_all[f_plus] != 0.0) q_d += yp;
        }
        if (f_minus >= 0) {
            double ym = W_all ? (W_all[f_minus] * b[f_minus]) : b[f_minus];
            g -= ym;
            if (!W_all || W_all[f_minus] != 0.0) q_d -= ym;
        }
        if (d < a) q_lo = q_d;
        else       q_hi = q_d;
    }
    return g;
}

__global__ void sg_ampere(
        const int n_free, const int n_blocks, const int* __restrict__ blk_meta,
        const int face_shift, const int* __restrict__ free_edges,
        const int* __restrict__ hang_edges, const int* __restrict__ pht_ptr,
        const int* __restrict__ pht_col, const double* __restrict__ pht_val,
        const double* __restrict__ b, const double* __restrict__ W_all,
        const double* __restrict__ Dinv, const double dt, const int only_KH,
        const int* __restrict__ phi_slot0, const int* __restrict__ phi_slot1,
        double* __restrict__ phi, const double* __restrict__ be,
        const double* __restrict__ ae, const int n_src_edges,
        const int* __restrict__ src_slot, const double* __restrict__ src_table,
        const long long* __restrict__ step, const long long step_host,
        double* __restrict__ e) {
    int f_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (f_idx >= n_free) return;
    double q_lo, q_hi;
    double rhs = edge_curl_T(free_edges[f_idx], n_blocks, blk_meta, face_shift,
                             b, W_all, q_lo, q_hi);
    if (pht_ptr) {
        int j0 = pht_ptr[f_idx], j1 = pht_ptr[f_idx + 1];
        double s_hang = 0.0;
        for (int j = j0; j < j1; ++j) {
            double cq_lo, cq_hi;
            int e_child = hang_edges[pht_col[j]];
            double gc = edge_curl_T(e_child, n_blocks, blk_meta, face_shift,
                                    b, W_all, cq_lo, cq_hi);
            double w = pht_val[j];
            s_hang += w * gc;
            q_lo += w * cq_lo;
            q_hi += w * cq_hi;
        }
        rhs += s_hang;
    }
    if (only_KH) {
        e[f_idx] = rhs;
        return;
    }
    if (phi_slot0) {
        int s0 = phi_slot0[f_idx];
        if (s0 >= 0) {
            double p = be[s0] * phi[s0] + ae[s0] * q_lo;
            phi[s0] = p;
            rhs += p;
        }
        int s1 = phi_slot1[f_idx];
        if (s1 >= 0) {
            double p = be[s1] * phi[s1] + ae[s1] * q_hi;
            phi[s1] = p;
            rhs += p;
        }
    }
    if (src_slot) {
        int sl = src_slot[f_idx];
        if (sl >= 0) {
            long long st = step ? step[0] : step_host;
            rhs -= src_table[st * n_src_edges + sl];
        }
    }
    e[f_idx] += (dt * Dinv[f_idx]) * rhs;
}

// ---------------------------------------------------------------------- LTS
// Per level m and Chebyshev substep k:
//   sg_lts_face: w_phi[i] = W[i] * ((K_F z)[i] + kappa_k * (K_C yC)[i])
//                on Phi_m, using the merged CSR rows of K_F | K_C.
//   sg_lts_edge: Xd = (tau2 * Dinv) * (KH_F w_phi),
//                d_new = ((2 delta dk - 2 beta Xd) + 2 T_k yF) - d_old,
//                and on the final substep k == p - 1: uF = yF + (out * d_new - yF).
__global__ void sg_lts_face(
        const int n_phi, const int* __restrict__ ptr,
        const int* __restrict__ split, const int* __restrict__ col,
        const double* __restrict__ val, const double* __restrict__ z_base,
        const int* __restrict__ z_map, const double* __restrict__ z_sub,
        const int* __restrict__ sub_pos, const double* __restrict__ v_parent,
        const int* __restrict__ c_map, const double* __restrict__ W,
        const double kappa_k, double* __restrict__ w_phi) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_phi) return;
    double kf = 0.0;
    int j0 = ptr[i], js = split[i], j1 = ptr[i + 1];
    for (int j = j0; j < js; ++j) {
        int c = col[j];
        double zc;
        if (z_sub && sub_pos[c] >= 0) {
            zc = z_sub[sub_pos[c]];
        } else if (z_map) {
            zc = z_base[z_map[c]];
        } else {
            zc = z_base[c];
        }
        kf += val[j] * zc;
    }
    double g = 0.0;
    for (int j = js; j < j1; ++j) {
        int c = col[j];
        double yc = c_map ? v_parent[c_map[c]] : v_parent[c];
        g += val[j] * yc;
    }
    w_phi[i] = W[i] * (kf + kappa_k * g);
}

__global__ void sg_lts_edge(
        const int nF, const int* __restrict__ kh_ptr,
        const int* __restrict__ kh_col, const double* __restrict__ kh_val,
        const double* __restrict__ w_phi, const double* __restrict__ Dinv,
        const double tau2, const double delta, const double beta,
        const double Tk, const double out_scale, const int k_sub,
        const int is_last, const double* __restrict__ v_parent,
        const int* __restrict__ yf_map, const double* __restrict__ dk_in,
        const double* __restrict__ dold_in, double* __restrict__ out_vec,
        const int* __restrict__ out_map) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nF) return;
    double kh = 0.0;
    for (int j = kh_ptr[i]; j < kh_ptr[i + 1]; ++j) {
        kh += kh_val[j] * w_phi[kh_col[j]];
    }
    double yf = yf_map ? v_parent[yf_map[i]] : v_parent[i];
    double dk = (k_sub == 1) ? yf : dk_in[i];
    double dold = (k_sub == 1) ? 0.0 : ((k_sub == 2) ? yf : dold_in[i]);
    double xd = (tau2 * Dinv[i]) * kh;
    double d_new = (((2.0 * delta) * dk - (2.0 * beta) * xd) + (2.0 * Tk) * yf) - dold;
    if (is_last) {
        double u = yf + (out_scale * d_new - yf);
        if (out_map) out_vec[out_map[i]] = u;
        else         out_vec[i] = u;
    } else {
        out_vec[i] = d_new;
    }
}

// ----------------------------------------------------------- Probes and DFT
__global__ void sg_probe_dft(
        const int n_probes, const int max_samples, const int total_freqs,
        const int* __restrict__ p_i0, const int* __restrict__ p_i1,
        const double* __restrict__ p_w0, const double* __restrict__ p_w1,
        const double* __restrict__ p_L, const int* __restrict__ p_foff,
        const int* __restrict__ p_nf, const double* __restrict__ e,
        const cplx* __restrict__ phasors, const double dt,
        const long long* __restrict__ step, const long long step_host,
        double* __restrict__ probe_buf, cplx* __restrict__ dft_buf) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= n_probes) return;
    long long st = step ? step[0] : step_host;
    int i0 = p_i0[p];
    double s = (i0 >= 0) ? (p_w0[p] * e[i0]) : 0.0;
    int i1 = p_i1[p];
    if (i1 >= 0) s += p_w1[p] * e[i1];
    double v = s / p_L[p];
    probe_buf[(long long)p * max_samples + st] = v;
    int nf = p_nf[p];
    if (nf > 0 && phasors) {
        int foff = p_foff[p];
        const cplx* ph = phasors + st * total_freqs + foff;
        cplx* acc = dft_buf + foff;
        cplx cv = cplx(v, 0.0);
        cplx cdt = cplx(dt, 0.0);
        for (int f = 0; f < nf; ++f) {
            acc[f] = acc[f] + (cv * ph[f]) * cdt;
        }
    }
}

__global__ void sg_step_add(long long* step, const long long k) {
    step[0] += k;
}

}  // extern "C"
"""

_MODULE = None
_KERNELS = {}


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(
            code=_SOURCE,
            options=("--std=c++17", "--fmad=false"),
        )
    return _MODULE


def _kernel(name):
    k = _KERNELS.get(name)
    if k is None:
        k = _KERNELS[name] = _module().get_function(name)
    return k


_NULL = np.intp(0)


def _launch(name, n, *args):
    n = int(n)
    if n <= 0:
        return
    kargs = tuple(_NULL if a is None else a for a in args)
    _kernel(name)(((n + _THREADS - 1) // _THREADS,), (_THREADS,), kargs)


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _host_csr(op):
    if hasattr(op, "get"):
        op = op.get()
    return sp.csr_matrix(op)


def _validate_solver(solver, mutate=None):
    """Validate `solver` and `mutate` against the fused subgrid contract."""
    if mutate not in _VALID_MUTATIONS:
        raise ValueError(
            f"unknown mutation {mutate!r}; expected one of {_VALID_MUTATIONS}"
        )
    M = solver.mesh
    if any(any(b.periodic) for b in M.blocks):
        raise NotImplementedError(
            "fused subgrid kernels: Bloch-periodic blocks use the CSR path"
        )
    if solver.dtype is not float and np.dtype(solver.dtype).kind == "c":
        raise NotImplementedError("fused subgrid kernels: real fields only")
    if getattr(solver, "dispersive_groups", None) or getattr(
        solver, "_dispersive_groups", None
    ):
        raise NotImplementedError(
            "fused subgrid kernels: dispersive media not supported"
        )
    cpml = getattr(solver, "cpml", None)
    n_aux = int(cpml.n_aux) if cpml is not None else 0
    expected_n_state = len(solver.b) + len(solver.e) + n_aux
    if hasattr(solver, "n_state") and int(solver.n_state) != expected_n_state:
        raise NotImplementedError(
            f"fused subgrid kernels: unsupported n_state={solver.n_state} "
            f"(expected {expected_n_state})"
        )


class FusedTables:
    """
    Host gather/transpose tables for fused 3D subgrid kernels and their NumPy
    emulation (`apply_K`, `apply_KH`).

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D
        Constructed 3D subgrid solver.
    mutate : {None, 'no_hanging', 'face_offset', 'lts_kappa'}, optional
        Verification-only mutation hook:
        * ``'no_hanging'``: drop the hanging-edge trace sources in `P_h` and
          `P_h^T`.
        * ``'face_offset'``: shift the per-block face offset by one when
          decoding face indices.
        * ``'lts_kappa'``: drop the Chebyshev buffer term ``kappa_k g`` in LTS.
    """

    def __init__(self, solver, mutate=None):
        _validate_solver(solver, mutate=mutate)
        self.solver = solver
        self.mutate = mutate
        M = solver.mesh
        self.n_blocks = int(M.n_blocks)
        self.nE = int(M.n_edges)
        self.nF = int(M.n_faces)
        self.n_free = int(solver.n_free)
        self.face_shift = 1 if mutate == "face_offset" else 0

        # Block layout metadata: 32 int32s per block
        blk_meta = np.zeros((self.n_blocks, 32), dtype=np.int32)
        self.blocks = []
        for bi, blk in enumerate(M.blocks):
            e0, f0 = int(M.edge_off[bi]), int(M.face_off[bi])
            eo = [e0 + int(blk.edge_off[a]) for a in range(4)]
            fo = [f0 + int(blk.face_off[a]) for a in range(4)]
            es = [tuple(int(x) for x in blk.edge_shape[a]) for a in range(3)]
            fs = [tuple(int(x) for x in blk.face_shape[a]) for a in range(3)]
            nc = tuple(int(x) for x in blk.nc)
            blk_meta[bi, 0:4] = eo
            blk_meta[bi, 4:8] = fo
            blk_meta[bi, 8:17] = np.asarray(es, dtype=np.int32).ravel()
            blk_meta[bi, 17:26] = np.asarray(fs, dtype=np.int32).ravel()
            blk_meta[bi, 26:29] = nc
            self.blocks.append((eo, fo, es, fs, nc))
        self.blk_meta = blk_meta

        # Edge-source table folding P_e: (src_i0, src_i1, src_w0, src_w1)
        src_i0 = np.full(self.nE, -1, dtype=np.int32)
        src_i1 = np.full(self.nE, -1, dtype=np.int32)
        src_w0 = np.zeros(self.nE, dtype=np.float64)
        src_w1 = np.zeros(self.nE, dtype=np.float64)

        free_edges = _host(solver.free_edges).astype(np.int32)
        src_i0[free_edges] = np.arange(self.n_free, dtype=np.int32)
        src_w0[free_edges] = 1.0

        hang_edges = np.flatnonzero(_host(solver.hanging)).astype(np.int32)
        P_csr = _host_csr(solver.P)
        Ph = P_csr[hang_edges].real.tocsr()
        if mutate != "no_hanging" and hang_edges.size > 0:
            for r, e_all in enumerate(hang_edges):
                p0, p1 = int(Ph.indptr[r]), int(Ph.indptr[r + 1])
                if p1 > p0:
                    src_i0[e_all] = int(Ph.indices[p0])
                    src_w0[e_all] = float(Ph.data[p0])
                if p1 > p0 + 1:
                    src_i1[e_all] = int(Ph.indices[p0 + 1])
                    src_w1[e_all] = float(Ph.data[p0 + 1])

        self.free_edges = free_edges
        self.hang_edges = (
            hang_edges if mutate != "no_hanging" else np.zeros(0, dtype=np.int32)
        )
        self.src_i0 = src_i0
        self.src_i1 = src_i1
        self.src_w0 = src_w0
        self.src_w1 = src_w1

        PhT = Ph.T.tocsr()
        if mutate == "no_hanging" or hang_edges.size == 0:
            self.pht_ptr = np.zeros(self.n_free + 1, dtype=np.int32)
            self.pht_col = np.zeros(0, dtype=np.int32)
            self.pht_val = np.zeros(0, dtype=np.float64)
        else:
            self.pht_ptr = PhT.indptr.astype(np.int32).copy()
            self.pht_col = PhT.indices.astype(np.int32).copy()
            self.pht_val = PhT.data.astype(np.float64).copy()

    def _eval_edges_cpu(self, e_idx, e):
        """Evaluate all-edge voltages at `e_idx` from free-edge voltages `e`."""
        i0 = self.src_i0[e_idx]
        i1 = self.src_i1[e_idx]
        v = np.where(i0 >= 0, self.src_w0[e_idx] * e[np.maximum(i0, 0)], 0.0)
        has1 = i1 >= 0
        if np.any(has1):
            v = np.where(has1, v + self.src_w1[e_idx] * e[np.maximum(i1, 0)], v)
        return v

    def apply_K(self, e, out=None):
        """
        NumPy emulation of the fused Faraday curl + trace ``K e = C P_e e``
        on all faces (`nF`), using the exact operation order of `sg_faraday`
        and `BlockOps.K`.
        """
        e = np.asarray(e, dtype=np.float64)
        if out is None:
            out = np.empty(self.nF, dtype=np.float64)
        for eo, fo, es, fs, _nc in self.blocks:
            for a in range(3):
                f_start, f_end = fo[a], fo[a + 1]
                n_comp = f_end - f_start
                loc = np.arange(n_comp, dtype=np.int32)
                if self.face_shift:
                    loc = (loc + self.face_shift) % n_comp
                fs1, fs2 = fs[a][1], fs[a][2]
                k = loc % fs2
                tmp = loc // fs2
                j = tmp % fs1
                i = tmp // fs1
                q = (i, j, k)

                eb, ec = (a + 1) % 3, (a + 2) % 3
                esb1, esb2 = es[eb][1], es[eb][2]
                e_b0 = eo[eb] + (q[0] * esb1 + q[1]) * esb2 + q[2]
                stride_b_c = (esb1 * esb2) if ec == 0 else (esb2 if ec == 1 else 1)
                e_b1 = e_b0 + stride_b_c

                esc1, esc2 = es[ec][1], es[ec][2]
                e_c0 = eo[ec] + (q[0] * esc1 + q[1]) * esc2 + q[2]
                stride_c_b = (esc1 * esc2) if eb == 0 else (esc2 if eb == 1 else 1)
                e_c1 = e_c0 + stride_c_b

                vb0 = self._eval_edges_cpu(e_b0, e)
                vc1 = self._eval_edges_cpu(e_c1, e)
                vb1 = self._eval_edges_cpu(e_b1, e)
                vc0 = self._eval_edges_cpu(e_c0, e)
                out[f_start:f_end] = (vb0 + vc1) - (vb1 + vc0)
        return out

    def _edge_curl_T_cpu(self, edge_indices, y_all):
        """NumPy emulation of `edge_curl_T` for an array of all-edge indices."""
        n = edge_indices.size
        g = np.zeros(n, dtype=np.float64)
        if n == 0:
            return g
        for bi, (eo, fo, es, fs, nc) in enumerate(self.blocks):
            in_blk = (edge_indices >= eo[0]) & (edge_indices < eo[3])
            if not np.any(in_blk):
                continue
            for dir_e in range(3):
                sel = np.flatnonzero(
                    in_blk
                    & (edge_indices >= eo[dir_e])
                    & (edge_indices < eo[dir_e + 1])
                )
                if sel.size == 0:
                    continue
                loc = edge_indices[sel] - eo[dir_e]
                es1, es2 = es[dir_e][1], es[dir_e][2]
                k = loc % es2
                tmp = loc // es2
                j = tmp % es1
                i = tmp // es1
                q = (i, j, k)
                g_sub = np.zeros(sel.size, dtype=np.float64)
                for a in range(3):
                    if a == dir_e:
                        continue
                    d = 3 - a - dir_e
                    f_start, f_end = fo[a], fo[a + 1]
                    n_comp = f_end - f_start
                    fs1, fs2 = fs[a][1], fs[a][2]
                    nc_d = nc[d]
                    base = (q[0] * fs1 + q[1]) * fs2 + q[2]
                    stride_d = (fs1 * fs2) if d == 0 else (fs2 if d == 1 else 1)

                    has_curr = q[d] < nc_d
                    lc = (base + self.face_shift) % n_comp if self.face_shift else base
                    f_curr = np.where(has_curr, f_start + lc, -1)

                    has_prev = q[d] >= 1
                    lp = base - stride_d
                    if self.face_shift:
                        lp = (lp + self.face_shift) % n_comp
                    f_prev = np.where(has_prev, f_start + lp, -1)

                    if dir_e == (a + 1) % 3:
                        f_plus, f_minus = f_curr, f_prev
                    else:
                        f_plus, f_minus = f_prev, f_curr

                    mp = f_plus >= 0
                    if np.any(mp):
                        g_sub[mp] += y_all[f_plus[mp]]
                    mm = f_minus >= 0
                    if np.any(mm):
                        g_sub[mm] -= y_all[f_minus[mm]]
                g[sel] = g_sub
        return g

    def apply_KH(self, y_all, out=None):
        """
        NumPy emulation of the fused Ampere transposed curl + restriction
        ``K^H y_all = P_e^T C^T y_all`` on the free edges (`n_free`), using
        the exact operation order of `sg_ampere` and `BlockOps.KH`.
        """
        y_all = np.asarray(y_all, dtype=np.float64)
        r = self._edge_curl_T_cpu(self.free_edges, y_all)
        if self.hang_edges.size > 0:
            g_hang = self._edge_curl_T_cpu(self.hang_edges, y_all)
            PhT = sp.csr_matrix(
                (self.pht_val, self.pht_col, self.pht_ptr),
                shape=(self.n_free, self.hang_edges.size),
            )
            r += PhT @ g_hang
        if out is not None:
            out[...] = r
            return out
        return r


class FusedSubgridStepper:
    """
    Fused GPU time stepper for `SubgridFIT3D` and `SubgridLTSFIT3D`.

    All device buffers are allocated once at initialization; time stepping
    launches a fixed sequence of fused kernels (2 per global-`dt` step, or
    `2 P` per coarse LTS step with total refinement ratio product `P`, plus 1
    when probes are attached) and replays chunks of `graph_chunk` steps via
    `wakis.gpu_fused.GraphRunner`.

    Parameters
    ----------
    solver : SubgridFIT3D or SubgridLTSFIT3D
        Constructed 3D subgrid solver.
    graph_chunk : int, optional
        Steps captured per CUDA graph (default 64; 0 or 1 disables graph
        capture and uses direct kernel launches).
    mutate : {None, 'no_hanging', 'face_offset', 'lts_kappa'}, optional
        Verification-only mutation hook.
    """

    def __init__(self, solver, graph_chunk=64, mutate=None):
        if not _HAS_CUPY:
            raise RuntimeError("FusedSubgridStepper requires CuPy and a CUDA device")
        self.tables = FusedTables(solver, mutate=mutate)
        self.solver = solver
        self.mutate = mutate
        self.graph_chunk = int(graph_chunk)
        self.dt = float(solver.dt)
        self.n_free = self.tables.n_free
        self.nF = self.tables.nF
        self.n_blocks = self.tables.n_blocks
        self.face_shift = np.int32(self.tables.face_shift)

        # Device gather and transpose tables
        t = self.tables
        self.blk_meta = idx32(t.blk_meta.ravel())
        self.free_edges = idx32(t.free_edges)
        self.src_i0 = idx32(t.src_i0)
        self.src_i1 = idx32(t.src_i1)
        self.src_w0 = cp.asarray(t.src_w0, dtype=np.float64)
        self.src_w1 = cp.asarray(t.src_w1, dtype=np.float64)
        if t.hang_edges.size > 0:
            self.hang_edges = idx32(t.hang_edges)
            self.pht_ptr = idx32(t.pht_ptr)
            self.pht_col = idx32(t.pht_col)
            self.pht_val = cp.asarray(t.pht_val, dtype=np.float64)
        else:
            self.hang_edges = None
            self.pht_ptr = None
            self.pht_col = None
            self.pht_val = None

        # Diagonal Hodges and masses
        self.faces_map = idx32(_host(solver.faces))
        self.W_all = cp.asarray(np.real(_host(solver.W_all)).astype(np.float64))
        self.Dinv = cp.asarray(np.real(_host(solver.Dinv)).astype(np.float64))
        self.D = 1.0 / self.Dinv

        # State vectors on device (b on all nF faces, e on n_free free edges)
        self.e = cp.zeros(self.n_free, dtype=np.float64)
        self.b = cp.zeros(self.nF, dtype=np.float64)
        self._tmp_f = cp.zeros(self.nF, dtype=np.float64)
        self._tmp_e = cp.zeros(self.n_free, dtype=np.float64)
        self._e_old = cp.zeros(self.n_free, dtype=np.float64)
        self._step_dev = cp.zeros(1, dtype=np.int64)
        self.n = int(solver.n)
        self.t = float(solver.t)

        # Build CPML, LTS, sources and probes on the device
        self._build_cpml()
        self._build_lts()
        self._cap = 4096
        self._src_sig = None
        self._prb_sig = None
        self._build_sources(self._cap)
        self._build_probes(self._cap)
        self._sync_from_solver()

        self.runner = (
            GraphRunner(self._graph_step, chunk=self.graph_chunk)
            if self.graph_chunk > 1
            else None
        )

    def _sync_from_solver(self):
        """Copy initial fields and CPML auxiliaries from `solver`."""
        s = self.solver
        e_h = _host(s.e).astype(np.float64)
        self.e[...] = cp.asarray(e_h)
        self._init_e_host = e_h.copy()
        b_h = _host(s.b).astype(np.float64)
        self.b.fill(0.0)
        if b_h.size == self.nF:
            self.b[...] = cp.asarray(b_h)
        else:
            self.b[self.faces_map] = cp.asarray(b_h)
        self._init_b_host = b_h.copy()
        if self.has_cpml and hasattr(s, "psi") and hasattr(s, "phi"):
            psi_parts = [_host(x).astype(np.float64) for x in s.psi]
            phi_parts = [_host(x).astype(np.float64) for x in s.phi]
            if any(p.size > 0 for p in psi_parts):
                self.psi[...] = cp.asarray(np.concatenate(psi_parts))
            if any(p.size > 0 for p in phi_parts):
                self.phi[...] = cp.asarray(np.concatenate(phi_parts))
        self.n = int(s.n)
        self.t = float(s.t)
        self._step_dev.fill(self.n)

    def _check_solver_resync(self):
        """If the caller mutated `solver.e` or `solver.b` after `__init__` at step 0,
        copy the updated initial state to the device before stepping."""
        if self.n != 0:
            return
        s = self.solver
        e_h = _host(s.e).astype(np.float64)
        if not np.array_equal(e_h, self._init_e_host):
            self.e[...] = cp.asarray(e_h)
            self._init_e_host = e_h.copy()
        b_h = _host(s.b).astype(np.float64)
        if not np.array_equal(b_h, self._init_b_host):
            self.b.fill(0.0)
            if b_h.size == self.nF:
                self.b[...] = cp.asarray(b_h)
            else:
                self.b[self.faces_map] = cp.asarray(b_h)
            self._init_b_host = b_h.copy()

    def reset(self):
        """Reset fields, CPML auxiliaries, step counter and probe buffers."""
        self.e.fill(0.0)
        self.b.fill(0.0)
        if self.has_cpml:
            self.psi.fill(0.0)
            self.phi.fill(0.0)
        if self.n_probes > 0:
            self.probe_buf.fill(0.0)
            if self.total_freqs > 0:
                self.dft_buf.fill(0.0)
        self.n = 0
        self.t = 0.0
        self._step_dev.fill(0)
        self._recorded_zero = False
        self._init_e_host = np.zeros(self.n_free, dtype=np.float64)
        self._init_b_host = np.zeros_like(_host(self.solver.b), dtype=np.float64)

    # ------------------------------------------------------------------ CPML
    def _build_cpml(self):
        c = getattr(self.solver, "cpml", None)
        if c is None:
            self.has_cpml = False
            self._n_psi = 0
            self._n_phi = 0
            self.psi_slot0 = self.psi_slot1 = None
            self.psi = self.bh = self.ah = None
            self.phi_slot0 = self.phi_slot1 = None
            self.phi = self.be = self.ae = None
            return
        self.has_cpml = True
        faces_host = _host(self.solver.faces).astype(np.int32)
        face_norm = _host(self.solver.face_normal).astype(np.int32)
        edge_dir = _host(self.solver.edge_dir[_host(self.solver.free_edges)]).astype(
            np.int32
        )

        # Psi slots on all-faces (2 derivative axes per face normal a: d_lo < d_hi)
        psi_slot0 = np.full(self.nF, -1, dtype=np.int32)
        psi_slot1 = np.full(self.nF, -1, dtype=np.int32)
        bh_list, ah_list = [], []
        off_f = 0
        for d in range(3):
            pf_all = faces_host[_host(c.pf[d]).astype(np.int32)]
            n_pf = pf_all.size
            if n_pf == 0:
                continue
            a_of = face_norm[pf_all]
            eb = (a_of + 1) % 3
            ec = (a_of + 2) % 3
            d_lo = np.minimum(eb, ec)
            slots = off_f + np.arange(n_pf, dtype=np.int32)
            is_lo = d == d_lo
            psi_slot0[pf_all[is_lo]] = slots[is_lo]
            psi_slot1[pf_all[~is_lo]] = slots[~is_lo]
            bh_list.append(_host(c.bh[d]).astype(np.float64))
            ah_list.append(_host(c.ah[d]).astype(np.float64))
            off_f += n_pf

        # Phi slots on free edges (2 derivative axes per edge direction dir_e: d_lo < d_hi)
        phi_slot0 = np.full(self.n_free, -1, dtype=np.int32)
        phi_slot1 = np.full(self.n_free, -1, dtype=np.int32)
        be_list, ae_list = [], []
        off_e = 0
        for d in range(3):
            pe = _host(c.pe[d]).astype(np.int32)
            n_pe = pe.size
            if n_pe == 0:
                continue
            de = edge_dir[pe]
            d_other = 3 - de - d
            is_lo = d < d_other
            slots = off_e + np.arange(n_pe, dtype=np.int32)
            phi_slot0[pe[is_lo]] = slots[is_lo]
            phi_slot1[pe[~is_lo]] = slots[~is_lo]
            be_list.append(_host(c.be[d]).astype(np.float64))
            ae_list.append(_host(c.ae[d]).astype(np.float64))
            off_e += n_pe

        self._n_psi = int(off_f)
        self._n_phi = int(off_e)
        self.psi_slot0 = idx32(psi_slot0)
        self.psi_slot1 = idx32(psi_slot1)
        self.psi = cp.zeros(max(off_f, 1), dtype=np.float64)
        self.bh = cp.asarray(
            np.concatenate(bh_list) if bh_list else np.zeros(1), dtype=np.float64
        )
        self.ah = cp.asarray(
            np.concatenate(ah_list) if ah_list else np.zeros(1), dtype=np.float64
        )

        self.phi_slot0 = idx32(phi_slot0)
        self.phi_slot1 = idx32(phi_slot1)
        self.phi = cp.zeros(max(off_e, 1), dtype=np.float64)
        self.be = cp.asarray(
            np.concatenate(be_list) if be_list else np.zeros(1), dtype=np.float64
        )
        self.ae = cp.asarray(
            np.concatenate(ae_list) if ae_list else np.zeros(1), dtype=np.float64
        )

    # ------------------------------------------------------------------- LTS
    def _build_lts(self):
        lts = getattr(self.solver, "lts", None)
        if lts is None or lts.L == 0:
            self.has_lts = False
            self.L = 0
            self.lts_levels = [None]
            self.f1_pos = None
            return
        self.has_lts = True
        self.L = int(lts.L)
        self.lts_levels = [None]

        # Map from global free edges 0..n_free-1 into F_1 (or -1 outside F_1)
        d1 = lts.levels[1]
        f1_pos = np.full(self.n_free, -1, dtype=np.int32)
        f1_host = _host(d1.F).astype(np.int32)
        f1_pos[f1_host] = np.arange(f1_host.size, dtype=np.int32)
        self.f1_pos = idx32(f1_pos)

        for lv in range(1, self.L + 1):
            d = lts.levels[lv]
            nF = int(d.F.size)
            nPhi = int(d.Phi.size)
            K_F = _host_csr(d.K_F)
            K_C = _host_csr(d.K_C)
            KH_F = _host_csr(d.KH_F)
            ptr, split, col, val = merge_rows(K_F, K_C)
            fc_op = RowOp(ptr, split, col, val, max_width=0)
            kh_op = RowOp(
                KH_F.indptr.astype(np.int32),
                None,
                KH_F.indices.astype(np.int32),
                KH_F.data.astype(np.float64),
                max_width=0,
            )
            iF_h = _host(d.iF).astype(np.int32)
            iC_h = _host(d.iC).astype(np.int32)
            F_h = _host(d.F).astype(np.int32)
            C_h = _host(d.C).astype(np.int32)

            if lv < self.L:
                d_next = lts.levels[lv + 1]
                sub_pos_h = np.full(nF, -1, dtype=np.int32)
                if_next = _host(d_next.iF).astype(np.int32)
                sub_pos_h[if_next] = np.arange(if_next.size, dtype=np.int32)
                sub_pos = idx32(sub_pos_h)
            else:
                sub_pos = None

            kappa = (
                np.zeros_like(d.kappa)
                if self.mutate == "lts_kappa"
                else np.asarray(d.kappa, float)
            )
            info = {
                "p": int(d.p),
                "nF": nF,
                "nPhi": nPhi,
                "tau2": float(d.tau2),
                "delta": float(d.delta),
                "beta": float(d.beta),
                "T": np.asarray(d.T, dtype=np.float64),
                "kappa": kappa,
                "out": float(d.out),
                "fc_op": fc_op,
                "kh_op": kh_op,
                "W": cp.asarray(_host(d.W), dtype=np.float64),
                "Dinv": cp.asarray(_host(d.Dinv), dtype=np.float64),
                "iF": idx32(iF_h),
                "iC": idx32(iC_h),
                "F_root": idx32(F_h),
                "C_root": idx32(C_h),
                "sub_pos": sub_pos,
                "w_phi": cp.zeros(max(nPhi, 1), dtype=np.float64),
                "dk_a": cp.zeros(max(nF, 1), dtype=np.float64),
                "dk_b": cp.zeros(max(nF, 1), dtype=np.float64),
                "uF": cp.zeros(max(nF, 1), dtype=np.float64),
            }
            self.lts_levels.append(info)

    def _launch_corr(self, m, v_parent, v_is_root, out_vec, out_map):
        """Launch the Chebyshev substep kernels of `SubgridLTS._corr` at level `m`."""
        lv = self.lts_levels[m]
        p = lv["p"]
        nF, nPhi = lv["nF"], lv["nPhi"]
        if p <= 1 or nF == 0:
            return
        yf_map = lv["F_root"] if v_is_root else lv["iF"]
        c_map = lv["C_root"] if v_is_root else lv["iC"]
        fc = lv["fc_op"]
        kh = lv["kh_op"]
        w_phi = lv["w_phi"]
        dk_cur, dk_prev = lv["dk_a"], lv["dk_b"]

        for k in range(1, p):
            is_last = 1 if (k == p - 1) else 0
            if k == 1:
                z_base = v_parent
                z_map = yf_map
            else:
                z_base = dk_cur
                z_map = None

            # Recursive evaluation z = _R(m, dk) on finer levels m + 1 .. L
            if m < self.L:
                if k == 1:
                    self._launch_R_into_uF(m + 1, v_parent, v_is_root)
                else:
                    self._launch_R_into_uF(m + 1, dk_cur, False)
                z_sub = self.lts_levels[m + 1]["uF"]
                sub_pos = lv["sub_pos"]
            else:
                z_sub = None
                sub_pos = None

            _launch(
                "sg_lts_face", nPhi, np.int32(nPhi), fc.ptr, fc.split, fc.col,
                fc.val, z_base, z_map, z_sub, sub_pos, v_parent, c_map, lv["W"],
                np.float64(lv["kappa"][k]), w_phi,
            )  # fmt: skip

            target = out_vec if is_last else (dk_cur if k == 1 else dk_prev)
            t_map = out_map if is_last else None
            dold_in = dk_prev if k >= 3 else None

            _launch(
                "sg_lts_edge", nF, np.int32(nF), kh.ptr, kh.col, kh.val, w_phi,
                lv["Dinv"], np.float64(lv["tau2"]), np.float64(lv["delta"]),
                np.float64(lv["beta"]), np.float64(lv["T"][k]),
                np.float64(lv["out"]), np.int32(k), np.int32(is_last),
                v_parent, yf_map, dk_cur if k >= 2 else None, dold_in,
                target, t_map,
            )  # fmt: skip

            if not is_last and k >= 2:
                dk_cur, dk_prev = dk_prev, dk_cur

    def _launch_R_into_uF(self, m, v_parent, v_is_root):
        """Compute `_R(m - 1, v_parent)` restricted to `F_m` into `lts_levels[m]['uF']`."""
        lv = self.lts_levels[m]
        uF = lv["uF"]
        self._launch_corr(m, v_parent, v_is_root, uF, None)
        if m < self.L:
            lv_next = self.lts_levels[m + 1]
            if m + 1 == self.L:
                self._launch_corr(m + 1, uF, False, uF, lv_next["iF"])
            else:
                self._launch_R_into_uF(m + 1, uF, False)
                # Copy nested uF into uF[iF]; L <= 2 in standard subgrid meshes
                uF[lv_next["iF"]] = lv_next["uF"]

    # --------------------------------------------------------------- Sources
    def _build_sources(self, cap):
        sources = self.solver.sources
        sig = (len(sources), tuple(int(s["edge"]) for s in sources))
        if not sources:
            self.n_src_edges = 0
            self.src_slot = None
            self.src_table = None
            self._src_filled = 0
            self._src_sig = sig
            return
        rows = np.unique(
            np.concatenate([np.asarray(s["row"], dtype=np.int32) for s in sources])
        )
        self.n_src_edges = int(rows.size)
        self._src_rows = rows
        slot = np.full(self.n_free, -1, dtype=np.int32)
        slot[rows] = np.arange(self.n_src_edges, dtype=np.int32)
        self.src_slot = idx32(slot)
        self.src_table = cp.zeros((cap, self.n_src_edges), dtype=np.float64)
        self._src_filled = 0
        self._src_sig = sig
        self._fill_sources(0, cap)

    def _fill_sources(self, n0, n1):
        if self.n_src_edges == 0 or n1 <= n0:
            return
        sources = self.solver.sources
        dt = self.dt
        pos = {int(r): i for i, r in enumerate(self._src_rows)}
        buf = np.zeros((n1 - n0, self.n_src_edges), dtype=np.float64)
        for st in range(n0, n1):
            t_half = (st * dt) + 0.5 * dt
            row_buf = buf[st - n0]
            for src in sources:
                val = src["amp"] * float(src["f"](t_half)) * np.real(src["w"])
                for r, wv in zip(src["row"], val):
                    row_buf[pos[int(r)]] += wv
        self.src_table[n0:n1] = cp.asarray(buf)
        self._src_filled = max(self._src_filled, n1)

    # --------------------------------------------------------- Probes & DFT
    def _build_probes(self, cap):
        probes = self.solver.probes
        sig = (
            len(probes),
            tuple(
                (int(p["edge"]), 0 if p["freqs"] is None else len(p["freqs"]))
                for p in probes
            ),
        )
        self.n_probes = len(probes)
        self._prb_sig = sig
        self._recorded_zero = False
        if self.n_probes == 0:
            self.total_freqs = 0
            self.probe_buf = None
            self.dft_buf = None
            self.dft_phasors = None
            self._prb_filled = 0
            return
        p_i0 = np.full(self.n_probes, -1, dtype=np.int32)
        p_i1 = np.full(self.n_probes, -1, dtype=np.int32)
        p_w0 = np.zeros(self.n_probes, dtype=np.float64)
        p_w1 = np.zeros(self.n_probes, dtype=np.float64)
        p_L = np.zeros(self.n_probes, dtype=np.float64)
        p_nf = np.zeros(self.n_probes, dtype=np.int32)
        freqs_all = []
        for idx, p in enumerate(probes):
            row = _host_csr(p["row"])
            if row.nnz >= 1:
                p_i0[idx] = int(row.indices[0])
                p_w0[idx] = float(np.real(row.data[0]))
            if row.nnz >= 2:
                p_i1[idx] = int(row.indices[1])
                p_w1[idx] = float(np.real(row.data[1]))
            p_L[idx] = float(p["L"])
            if p["freqs"] is not None and len(p["freqs"]) > 0:
                f = np.asarray(p["freqs"], dtype=np.float64).ravel()
                p_nf[idx] = f.size
                freqs_all.append(f)
        p_foff = np.zeros(self.n_probes, dtype=np.int32)
        if self.n_probes > 1:
            np.cumsum(p_nf[:-1], out=p_foff[1:])
        self.total_freqs = int(p_nf.sum())
        self._freqs_concat = (
            np.concatenate(freqs_all) if freqs_all else np.zeros(0, dtype=np.float64)
        )
        self._p_foff_host = p_foff.copy()
        self._p_nf_host = p_nf.copy()

        self.p_i0 = idx32(p_i0)
        self.p_i1 = idx32(p_i1)
        self.p_w0 = cp.asarray(p_w0, dtype=np.float64)
        self.p_w1 = cp.asarray(p_w1, dtype=np.float64)
        self.p_L = cp.asarray(p_L, dtype=np.float64)
        self.p_foff = idx32(p_foff)
        self.p_nf = idx32(p_nf)

        self.probe_buf = cp.zeros((self.n_probes, cap), dtype=np.float64)
        if self.total_freqs > 0:
            self.dft_buf = cp.zeros(self.total_freqs, dtype=np.complex128)
            self.dft_phasors = cp.zeros((cap, self.total_freqs), dtype=np.complex128)
        else:
            self.dft_buf = None
            self.dft_phasors = None
        self._prb_filled = 0
        self._fill_phasors(0, cap)

    def _fill_phasors(self, n0, n1):
        if self.n_probes == 0 or self.total_freqs == 0 or n1 <= n0:
            self._prb_filled = max(getattr(self, "_prb_filled", 0), n1)
            return
        t = np.arange(n0, n1, dtype=np.float64)[:, None] * self.dt
        ph = np.exp((-1j * 2 * np.pi * self._freqs_concat[None, :]) * t)
        self.dft_phasors[n0:n1] = cp.asarray(ph, dtype=np.complex128)
        self._prb_filled = max(self._prb_filled, n1)

    def _ensure_capacity(self, needed_step):
        """Ensure preallocated source/probe tables cover `needed_step`."""
        sources = self.solver.sources
        src_sig = (len(sources), tuple(int(s["edge"]) for s in sources))
        probes = self.solver.probes
        prb_sig = (
            len(probes),
            tuple(
                (int(p["edge"]), 0 if p["freqs"] is None else len(p["freqs"]))
                for p in probes
            ),
        )
        rebuilt = False
        if needed_step > self._cap:
            while self._cap < needed_step:
                self._cap *= 2
            self._build_sources(self._cap)
            old_probe = self.probe_buf
            old_dft = self.dft_buf
            old_rec = getattr(self, "_recorded_zero", False)
            self._build_probes(self._cap)
            if old_probe is not None and self.probe_buf is not None:
                n_copy = min(old_probe.shape[1], self.n + 1)
                self.probe_buf[:, :n_copy] = old_probe[:, :n_copy]
            if old_dft is not None and self.dft_buf is not None:
                self.dft_buf[...] = old_dft
            self._recorded_zero = old_rec
            rebuilt = True
        else:
            if src_sig != self._src_sig:
                self._build_sources(self._cap)
                rebuilt = True
            if prb_sig != self._prb_sig:
                self._build_probes(self._cap)
                rebuilt = True
        if rebuilt and self.runner is not None:
            self.runner.graph = None

    # ------------------------------------------------------------- Stepping
    def _launch_step_kernels(self):
        """Launch the LTS, Faraday and Ampere kernels of one coarse step."""
        if self.has_lts:
            self._launch_R_into_uF(1, self.e, True)
            z_f1 = self.lts_levels[1]["uF"]
            f1_pos = self.f1_pos
        else:
            z_f1 = None
            f1_pos = None

        _launch(
            "sg_faraday", self.nF, np.int32(self.nF), np.int32(self.n_blocks),
            self.blk_meta, self.face_shift, self.src_i0, self.src_i1,
            self.src_w0, self.src_w1, self.e, z_f1, f1_pos,
            np.float64(self.dt), np.int32(0), self.psi_slot0, self.psi_slot1,
            self.psi, self.bh, self.ah, self.b,
        )  # fmt: skip

        _launch(
            "sg_ampere", self.n_free, np.int32(self.n_free),
            np.int32(self.n_blocks), self.blk_meta, self.face_shift,
            self.free_edges, self.hang_edges, self.pht_ptr, self.pht_col,
            self.pht_val, self.b, self.W_all, self.Dinv, np.float64(self.dt),
            np.int32(0), self.phi_slot0, self.phi_slot1, self.phi, self.be,
            self.ae, np.int32(self.n_src_edges), self.src_slot, self.src_table,
            self._step_dev, np.int64(0), self.e,
        )  # fmt: skip

    def _launch_record(self):
        if self.n_probes == 0:
            return
        _launch(
            "sg_probe_dft", self.n_probes, np.int32(self.n_probes),
            np.int32(self._cap), np.int32(self.total_freqs), self.p_i0,
            self.p_i1, self.p_w0, self.p_w1, self.p_L, self.p_foff, self.p_nf,
            self.e, self.dft_phasors, np.float64(self.dt), self._step_dev,
            np.int64(0), self.probe_buf, self.dft_buf,
        )  # fmt: skip

    def _graph_step(self):
        """One device step + counter advance + probe/DFT record (graph-safe)."""
        self._launch_step_kernels()
        _launch("sg_step_add", 1, self._step_dev, np.int64(1))
        self._launch_record()

    def one_step(self):
        """Advance one coarse time step on the device."""
        self._check_solver_resync()
        self._ensure_capacity(self.n + 2)
        self._launch_step_kernels()
        _launch("sg_step_add", 1, self._step_dev, np.int64(1))
        self.n += 1
        self.t = self.n * self.dt

    def run(self, n_steps, energy=False):
        """
        Advance `n_steps` coarse steps on the device.

        Uses `GraphRunner` when `graph_chunk > 1` and `energy=False`. Passing
        `energy=True` bypasses the CUDA-graph replay path and runs the
        per-step Python loop to evaluate `modified_energy(e^n, e^{n+1})` after
        each step, returning the history as a 1D NumPy array [J/s^2].
        """
        n_steps = int(n_steps)
        self._check_solver_resync()
        self._ensure_capacity(self.n + n_steps + 2)
        if self.n == 0 and not self._recorded_zero:
            self._launch_record()
            self._recorded_zero = True
        if not energy and self.runner is not None:
            self.runner.run(n_steps)
            self.n += n_steps
            self.t = self.n * self.dt
            return None
        hist = []
        for _ in range(n_steps):
            if energy:
                self._e_old[...] = self.e
            self._graph_step()
            self.n += 1
            self.t = self.n * self.dt
            if energy:
                hist.append(self.modified_energy(self._e_old, self.e))
        return np.asarray(hist, dtype=np.float64) if energy else None

    # ---------------------------------------------------- Device operators
    def apply_K(self, e, out=None):
        """Device application of ``K e = C P_e e`` on all `nF` faces."""
        e_dev = cp.asarray(e, dtype=np.float64)
        out_dev = self._tmp_f if out is None else out
        _launch(
            "sg_faraday", self.nF, np.int32(self.nF), np.int32(self.n_blocks),
            self.blk_meta, self.face_shift, self.src_i0, self.src_i1,
            self.src_w0, self.src_w1, e_dev, None, None, np.float64(0.0),
            np.int32(1), None, None, None, None, None, out_dev,
        )  # fmt: skip
        return out_dev

    def apply_KH(self, y_all, out=None):
        """Device application of ``K^H y_all = P_e^T C^T y_all`` on all `n_free` free edges."""
        y_dev = cp.asarray(y_all, dtype=np.float64)
        out_dev = self._tmp_e if out is None else out
        _launch(
            "sg_ampere", self.n_free, np.int32(self.n_free),
            np.int32(self.n_blocks), self.blk_meta, self.face_shift,
            self.free_edges, self.hang_edges, self.pht_ptr, self.pht_col,
            self.pht_val, y_dev, None, None, np.float64(0.0), np.int32(1),
            None, None, None, None, None, np.int32(0), None, None, None,
            np.int64(0), out_dev,
        )  # fmt: skip
        return out_dev

    def S0(self, v):
        """Device application of the Hermitian LTS stiffness ``S_0 v = K^H W K R_0 v``."""
        v_dev = cp.asarray(v, dtype=np.float64)
        if self.has_lts:
            self._launch_R_into_uF(1, v_dev, True)
            z_f1 = self.lts_levels[1]["uF"]
            f1_pos = self.f1_pos
        else:
            z_f1 = None
            f1_pos = None
        _launch(
            "sg_faraday", self.nF, np.int32(self.nF), np.int32(self.n_blocks),
            self.blk_meta, self.face_shift, self.src_i0, self.src_i1,
            self.src_w0, self.src_w1, v_dev, z_f1, f1_pos, np.float64(0.0),
            np.int32(1), None, None, None, None, None, self._tmp_f,
        )  # fmt: skip
        y = self.W_all * self._tmp_f
        out = cp.empty(self.n_free, dtype=np.float64)
        return self.apply_KH(y, out=out)

    def modified_energy(self, e0, e1):
        """
        Conserved quadratic form of the (LTS) leapfrog on the device:
        ``1/2 [w^H (D - dt^2/4 S_0) w + ybar^H S_0 ybar]`` [J/s^2].
        """
        e0_d = cp.asarray(e0, dtype=np.float64)
        e1_d = cp.asarray(e1, dtype=np.float64)
        dt = self.dt
        w = (e1_d - e0_d) / dt
        yb = 0.5 * (e1_d + e0_d)
        val = (
            cp.vdot(w, self.D * w)
            - (dt**2 / 4.0) * cp.vdot(w, self.S0(w))
            + cp.vdot(yb, self.S0(yb))
        )
        return float(0.5 * np.real(val.item()))

    # ------------------------------------------------------------ Readouts
    def kernels_per_step(self):
        """
        Number of field-update kernels launched per coarse step (`2 * prod p_l`:
        2 for global `dt`, 4 for 2:1 LTS, 6 for 3:1 LTS, 12 for nested 2:1+3:1).
        """
        if not self.has_lts:
            return 2
        p_prod = 1
        for lv in self.lts_levels[1:]:
            p_prod *= int(lv["p"])
        return 2 * p_prod

    def e_host(self):
        """Free-edge voltages `e` [V] as a 1D NumPy array."""
        return self.e.get()

    def b_faces(self):
        """Face fluxes `b` [Wb] on `solver.faces` as a 1D NumPy array."""
        return self.b[self.faces_map].get()

    def state_host(self):
        """
        Full packed state ``[b, e, psi_0..2, phi_0..2]`` as a 1D NumPy array,
        matching `solver._layout()`.
        """
        nb = getattr(self.solver, "_nb", self.solver.n_faces)
        b_part = self.b.get() if nb == self.nF else self.b_faces()
        parts = [b_part, self.e.get()]
        if self.has_cpml:
            if self._n_psi > 0:
                parts.append(self.psi[: self._n_psi].get())
            if self._n_phi > 0:
                parts.append(self.phi[: self._n_phi].get())
        return np.concatenate(parts)

    def step_vec(self, x, i_all=None):
        """
        One coarse step as a linear map of the packed state
        ``x = [b^{n-1/2}, e^n, psi_0..2, phi_0..2]`` and optional all-edge
        current ``i_all``, evaluated via the fused device operators.
        """
        s = self.solver
        b, e, psi, phi = s._split(np.asarray(x, dtype=np.float64))
        b_full = np.zeros(self.nF, dtype=np.float64)
        if b.size == self.nF:
            b_full[...] = b
        else:
            b_full[_host(s.faces)] = b
        b_dev = cp.asarray(b_full)
        e_dev = cp.asarray(e, dtype=np.float64)

        if self.has_lts:
            self._launch_R_into_uF(1, e_dev, True)
            z_f1 = self.lts_levels[1]["uF"]
            f1_pos = self.f1_pos
        else:
            z_f1 = None
            f1_pos = None

        if self.has_cpml:
            psi_cat = (
                np.concatenate(psi) if any(p.size > 0 for p in psi) else np.zeros(1)
            )
            phi_cat = (
                np.concatenate(phi) if any(p.size > 0 for p in phi) else np.zeros(1)
            )
            psi_dev = cp.asarray(psi_cat, dtype=np.float64)
            phi_dev = cp.asarray(phi_cat, dtype=np.float64)
        else:
            psi_dev = None
            phi_dev = None

        _launch(
            "sg_faraday", self.nF, np.int32(self.nF), np.int32(self.n_blocks),
            self.blk_meta, self.face_shift, self.src_i0, self.src_i1,
            self.src_w0, self.src_w1, e_dev, z_f1, f1_pos,
            np.float64(self.dt), np.int32(0), self.psi_slot0, self.psi_slot1,
            psi_dev, self.bh, self.ah, b_dev,
        )  # fmt: skip

        if i_all is not None:
            i_free = np.real(
                _host_csr(s.P).conj().T @ np.asarray(i_all, dtype=np.float64)
            )
            src_slot = idx32(np.arange(self.n_free, dtype=np.int32))
            src_table = cp.asarray(i_free[None, :], dtype=np.float64)
            n_src = self.n_free
        else:
            src_slot = None
            src_table = None
            n_src = 0

        _launch(
            "sg_ampere", self.n_free, np.int32(self.n_free),
            np.int32(self.n_blocks), self.blk_meta, self.face_shift,
            self.free_edges, self.hang_edges, self.pht_ptr, self.pht_col,
            self.pht_val, b_dev, self.W_all, self.Dinv, np.float64(self.dt),
            np.int32(0), self.phi_slot0, self.phi_slot1, phi_dev, self.be,
            self.ae, np.int32(n_src), src_slot, src_table, None,
            np.int64(0), e_dev,
        )  # fmt: skip

        b1_all = b_dev.get()
        b1 = b1_all if b.size == self.nF else b1_all[_host(s.faces)]
        parts = [b1, e_dev.get()]
        if self.has_cpml:
            if self._n_psi > 0:
                parts.append(psi_dev[: self._n_psi].get())
            if self._n_phi > 0:
                parts.append(phi_dev[: self._n_phi].get())
        return np.concatenate(parts)

    def step_vec_T(self, y):
        """
        Exact bilinear transpose of `step_vec` using the fused device operators
        `apply_K` (for ``(K^H)^T``) and `apply_KH` (for ``K^T``):
        ``<step_vec(x, i), y> = <x, xbar> + <i, ibar>``.
        """
        s = self.solver
        dt = self.dt
        bb1, eb1, psib1, phib1 = s._split(np.asarray(y, dtype=np.float64))
        dinv_h = self.Dinv.get()
        w_all_h = self.W_all.get()
        faces_h = _host(s.faces).astype(np.int32)
        ge = (dt * dinv_h) * eb1
        ybar_all = self.apply_K(ge).get().copy()
        ybar = ybar_all if bb1.size == self.nF else ybar_all[faces_h].copy()
        phib = []
        if self.has_cpml:
            for d, p in enumerate(s._pml):
                pe_h = _host(p["pe"]).astype(np.int32)
                be_h = _host(p["be"]).astype(np.float64)
                ae_h = _host(p["ae"]).astype(np.float64)
                ke_csr = _host_csr(p["Ke"])
                tot = phib1[d] + ge[pe_h]
                phib.append(be_h * tot)
                ybar = ybar + ke_csr.T @ (ae_h * tot)
        wb_h = w_all_h if bb1.size == self.nF else w_all_h[faces_h]
        b1bar = bb1 + wb_h * ybar
        b1bar_all = np.zeros(self.nF, dtype=np.float64)
        if bb1.size == self.nF:
            b1bar_all[...] = b1bar
        else:
            b1bar_all[faces_h] = b1bar
        zbar = -dt * self.apply_KH(b1bar_all).get().copy()
        psib = []
        if self.has_cpml:
            for d, p in enumerate(s._pml):
                pf_h = _host(p["pf"]).astype(np.int32)
                bh_h = _host(p["bh"]).astype(np.float64)
                ah_h = _host(p["ah"]).astype(np.float64)
                kf_csr = _host_csr(p["Kf"])
                tot = psib1[d] - dt * b1bar[pf_h]
                psib.append(bh_h * tot)
                zbar = zbar + kf_csr.T @ (ah_h * tot)
        if self.has_lts:
            s.lts.to_backend("numpy")
            try:
                zbar_r = s.lts.apply_T(zbar)
            finally:
                s.lts.to_backend(s.backend)
        else:
            zbar_r = zbar
        ebar = eb1 + zbar_r
        ibar = -np.real(_host_csr(s.P).conj() @ ge)
        return np.concatenate([b1bar, ebar] + psib + phib), ibar

    def probe_signal(self, i):
        """Recorded time series [V/m] of probe `i` as a 1D NumPy array."""
        if not (0 <= i < self.n_probes):
            raise IndexError(f"probe index {i} out of range (n_probes={self.n_probes})")
        n_samp = self.n + 1 if self._recorded_zero else 0
        return self.probe_buf[i, :n_samp].get()

    def dft_data(self, i):
        """Running DFT accumulator of probe `i` as a 1D complex128 NumPy array."""
        if not (0 <= i < self.n_probes):
            raise IndexError(f"probe index {i} out of range (n_probes={self.n_probes})")
        nf = int(self._p_nf_host[i])
        if nf == 0 or self.dft_buf is None:
            return None
        foff = int(self._p_foff_host[i])
        return self.dft_buf[foff : foff + nf].get()

    def synchronize(self):
        """Synchronize the CUDA device."""
        cp.cuda.Device().synchronize()
