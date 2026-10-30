# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""GPU-resident and MPI-aware particle diagnostics for :mod:`wakis.particles`.

On the GPU path (``sp.on_gpu == True``), all diagnostic monitors accumulate on
the active CUDA device during time stepping without any device-to-host
synchronization or per-step dynamic allocation. On the CPU path, monitors
accumulate into preallocated host buffers using vectorised NumPy operations
(which allocate small temporary arrays during per-step reductions/binning).
Host readout and cross-rank MPI reductions occur exclusively when
:meth:`result` (or its alias :meth:`to_numpy`) is called collectively.

Time levels at the end of each particle step ``t_n -> t_{n+1}``
---------------------------------------------------------------
When :meth:`ParticleSolver._run_diagnostics` is invoked at the end of a step:

* Particle positions ``(x, y, z)`` are at integer time level ``t_{n+1} = t_n + dt``
  (already wrapped on periodic axes and migrated to their owning MPI z-slab).
* Particle proper velocities ``(ux, uy, uz) = gamma * v`` are at the leapfrog
  half-step time level ``t_{n+1/2} = t_n + 0.5 * dt``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge

from .species import Species

_AXIS_CODES: dict[str, int] = {
    "x": 0,
    "y": 1,
    "z": 2,
    "ux": 3,
    "uy": 4,
    "uz": 5,
    "vx": 6,
    "vy": 7,
    "vz": 8,
    "xp": 9,
    "x'": 9,
    "yp": 10,
    "y'": 10,
    "gamma": 11,
    "ek_ev": 12,
    "energy_ev": 12,
}

_BLOCK_SIZE = 256
_SMEM_REDUCTION_SIZE = _BLOCK_SIZE
_EMIT_N_STATS = 20
assert (
    isinstance(_BLOCK_SIZE, int)
    and _BLOCK_SIZE > 0
    and (_BLOCK_SIZE & (_BLOCK_SIZE - 1)) == 0
    and _SMEM_REDUCTION_SIZE == _BLOCK_SIZE
), (
    f"_BLOCK_SIZE must be a positive power of two matching shared-memory "
    f"reduction array size, got {_BLOCK_SIZE} vs {_SMEM_REDUCTION_SIZE}."
)

_DIAG_CUDA_SRC = (
    f"#ifndef DIAG_BLOCK_SIZE\n#define DIAG_BLOCK_SIZE {_SMEM_REDUCTION_SIZE}\n#endif\n"
    + r"""
static_assert(
    DIAG_BLOCK_SIZE > 0 && ((DIAG_BLOCK_SIZE & (DIAG_BLOCK_SIZE - 1)) == 0),
    "DIAG_BLOCK_SIZE must be a power of two"
);

__device__ __forceinline__ int find_bin_dev(
    double val,
    double first_edge,
    double last_edge,
    int n_bins,
    const double* __restrict__ edges
) {
    if (!(val >= first_edge && val <= last_edge)) {
        return -1;
    }
    double denom = last_edge - first_edge;
    double f_idx = ((val - first_edge) / denom) * (double)n_bins;
    int idx = (int)f_idx;
    if (idx >= n_bins) idx = n_bins - 1;
    if (idx < 0) idx = 0;
    while (idx > 0 && val < edges[idx]) {
        idx -= 1;
    }
    while (idx < n_bins - 1 && val >= edges[idx + 1]) {
        idx += 1;
    }
    return idx;
}

__device__ __forceinline__ double eval_axis_dev(
    int code,
    double x, double y, double z,
    double ux, double uy, double uz,
    double mass, double inv_c2, double inv_c, double inv_e
) {
    if (code == 0) return x;
    if (code == 1) return y;
    if (code == 2) return z;
    if (code == 3) return ux;
    if (code == 4) return uy;
    if (code == 5) return uz;
    double u2 = ux * ux + uy * uy + uz * uz;
    double gamma = sqrt(1.0 + u2 * inv_c2);
    if (code == 6) return ux / gamma;
    if (code == 7) return uy / gamma;
    if (code == 8) return uz / gamma;
    if (code == 9) return (uz != 0.0) ? (ux / uz) : __longlong_as_double(0x7ff8000000000000ULL);
    if (code == 10) return (uz != 0.0) ? (uy / uz) : __longlong_as_double(0x7ff8000000000000ULL);
    if (code == 11) return gamma;
    return (mass * inv_e) * (u2 / (gamma + 1.0));
}

extern "C" __global__ void zero_f64_kernel(
    double* __restrict__ buf,
    int offset,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        buf[offset + tid] = 0.0;
    }
}

extern "C" __global__ void zero_hist_frame_kernel(
    double* __restrict__ frames,
    int frame_offset,
    int frame_len,
    double* __restrict__ frame_ovf,
    unsigned long long* __restrict__ frame_ovf_cnt,
    int slot
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frame_len) {
        frames[frame_offset + tid] = 0.0;
    }
    if (tid == 0) {
        frame_ovf[slot] = 0.0;
        frame_ovf_cnt[slot] = 0ULL;
    }
}

extern "C" __global__ void zero_probe_slot_kernel(
    double* __restrict__ d_charge,
    long long* __restrict__ d_counts,
    int slot
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_charge[slot] = 0.0;
        d_counts[slot] = 0LL;
    }
}

extern "C" __global__ void fill_nan_f64_kernel(
    double* __restrict__ buf,
    int offset,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        buf[offset + tid] = __longlong_as_double(0x7ff8000000000000ULL);
    }
}

extern "C" __global__ void reset_track_frame_kernel(
    double* __restrict__ tracks,
    int frame_offset,
    int n_frame_elems,
    int* __restrict__ match_counts,
    int match_offset,
    int n_targets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_frame_elems) {
        tracks[frame_offset + tid] = __longlong_as_double(0x7ff8000000000000ULL);
    }
    if (tid < n_targets) {
        match_counts[match_offset + tid] = 0;
    }
}

extern "C" __global__ void phase_space_hist2d_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    double mass,
    double inv_c2,
    double inv_c,
    double inv_e,
    int n_host,
    const int* __restrict__ d_n_particles,
    int ax0_code,
    int ax1_code,
    double min0,
    double max0,
    int nbins0,
    const double* __restrict__ edges0,
    double min1,
    double max1,
    int nbins1,
    const double* __restrict__ edges1,
    int weighted,
    double* __restrict__ hist_accum,
    double* __restrict__ ovf_accum,
    unsigned long long* __restrict__ ovf_cnt_accum,
    double* __restrict__ frames,
    int frame_offset,
    double* __restrict__ frame_ovf,
    unsigned long long* __restrict__ frame_ovf_cnt,
    int slot
) {
    // Shared-memory privatised histogram for <= 3072 bins (24 KiB per block);
    // falls back to global atomicAdd if nbins0 * nbins1 > 3072.
    __shared__ double s_hist[3072];
    __shared__ double s_ovf;
    __shared__ unsigned long long s_ovf_cnt;

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;
    int nb_tot = nbins0 * nbins1;
    int use_smem = (nb_tot <= 3072);

    if (use_smem) {
        for (int i = tx; i < nb_tot; i += blockDim.x) {
            s_hist[i] = 0.0;
        }
        if (tx == 0) {
            s_ovf = 0.0;
            s_ovf_cnt = 0ULL;
        }
        __syncthreads();
    }

    if (tid < n_act) {
        double v0 = eval_axis_dev(
            ax0_code, x[tid], y[tid], z[tid],
            ux[tid], uy[tid], uz[tid], mass, inv_c2, inv_c, inv_e
        );
        double v1 = eval_axis_dev(
            ax1_code, x[tid], y[tid], z[tid],
            ux[tid], uy[tid], uz[tid], mass, inv_c2, inv_c, inv_e
        );

        int b0 = find_bin_dev(v0, min0, max0, nbins0, edges0);
        int b1 = find_bin_dev(v1, min1, max1, nbins1, edges1);
        double wt = weighted ? w[tid] : 1.0;

        if (b0 >= 0 && b1 >= 0) {
            int flat = b0 * nbins1 + b1;
            if (use_smem) {
                atomicAdd(&s_hist[flat], wt);
            } else {
                atomicAdd(&hist_accum[flat], wt);
                if (frames) {
                    atomicAdd(&frames[frame_offset + flat], wt);
                }
            }
        } else {
            if (use_smem) {
                atomicAdd(&s_ovf, wt);
                atomicAdd(&s_ovf_cnt, 1ULL);
            } else {
                atomicAdd(ovf_accum, wt);
                atomicAdd(ovf_cnt_accum, 1ULL);
                if (frame_ovf) {
                    atomicAdd(&frame_ovf[slot], wt);
                    atomicAdd(&frame_ovf_cnt[slot], 1ULL);
                }
            }
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tx; i < nb_tot; i += blockDim.x) {
            double val = s_hist[i];
            if (val != 0.0) {
                atomicAdd(&hist_accum[i], val);
                if (frames) {
                    atomicAdd(&frames[frame_offset + i], val);
                }
            }
        }
        if (tx == 0 && s_ovf_cnt > 0ULL) {
            atomicAdd(ovf_accum, s_ovf);
            atomicAdd(ovf_cnt_accum, s_ovf_cnt);
            if (frame_ovf) {
                atomicAdd(&frame_ovf[slot], s_ovf);
                atomicAdd(&frame_ovf_cnt[slot], s_ovf_cnt);
            }
        }
    }
}

extern "C" __global__ void energy_spectrum_hist1d_kernel(
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    double mass_over_e,
    double inv_c2,
    int n_host,
    const int* __restrict__ d_n_particles,
    double min_eV,
    double max_eV,
    int nbins,
    const double* __restrict__ edges,
    int weighted,
    double* __restrict__ hist_accum,
    double* __restrict__ ovf_accum,
    unsigned long long* __restrict__ ovf_cnt_accum,
    double* __restrict__ frames,
    int frame_offset,
    double* __restrict__ frame_ovf,
    unsigned long long* __restrict__ frame_ovf_cnt,
    int slot
) {
    __shared__ double s_hist[3072];
    __shared__ double s_ovf;
    __shared__ unsigned long long s_ovf_cnt;

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;
    int use_smem = (nbins <= 3072);

    if (use_smem) {
        for (int i = tx; i < nbins; i += blockDim.x) {
            s_hist[i] = 0.0;
        }
        if (tx == 0) {
            s_ovf = 0.0;
            s_ovf_cnt = 0ULL;
        }
        __syncthreads();
    }

    if (tid < n_act) {
        double uxi = ux[tid];
        double uyi = uy[tid];
        double uzi = uz[tid];
        double u2 = uxi * uxi + uyi * uyi + uzi * uzi;
        double gamma = sqrt(1.0 + u2 * inv_c2);
        double ek_eV = mass_over_e * (u2 / (gamma + 1.0));

        int b = find_bin_dev(ek_eV, min_eV, max_eV, nbins, edges);
        double wt = weighted ? w[tid] : 1.0;

        if (b >= 0) {
            if (use_smem) {
                atomicAdd(&s_hist[b], wt);
            } else {
                atomicAdd(&hist_accum[b], wt);
                if (frames) {
                    atomicAdd(&frames[frame_offset + b], wt);
                }
            }
        } else {
            if (use_smem) {
                atomicAdd(&s_ovf, wt);
                atomicAdd(&s_ovf_cnt, 1ULL);
            } else {
                atomicAdd(ovf_accum, wt);
                atomicAdd(ovf_cnt_accum, 1ULL);
                if (frame_ovf) {
                    atomicAdd(&frame_ovf[slot], wt);
                    atomicAdd(&frame_ovf_cnt[slot], 1ULL);
                }
            }
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tx; i < nbins; i += blockDim.x) {
            double val = s_hist[i];
            if (val != 0.0) {
                atomicAdd(&hist_accum[i], val);
                if (frames) {
                    atomicAdd(&frames[frame_offset + i], val);
                }
            }
        }
        if (tx == 0 && s_ovf_cnt > 0ULL) {
            atomicAdd(ovf_accum, s_ovf);
            atomicAdd(ovf_cnt_accum, s_ovf_cnt);
            if (frame_ovf) {
                atomicAdd(&frame_ovf[slot], s_ovf);
                atomicAdd(&frame_ovf_cnt[slot], s_ovf_cnt);
            }
        }
    }
}

extern "C" __global__ void plane_current_probe_kernel(
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    double charge,
    double dt,
    double z_plane,
    double inv_c2,
    int periodic_z,
    double z_min,
    double z_max,
    int n_host,
    const int* __restrict__ d_n_particles,
    double* __restrict__ d_charge,
    long long* __restrict__ d_counts,
    double* __restrict__ d_total_charge,
    long long* __restrict__ d_total_counts,
    int slot
) {
    __shared__ double s_q[DIAG_BLOCK_SIZE];
    __shared__ long long s_c[DIAG_BLOCK_SIZE];
    __shared__ int s_any;
    static_assert(
        sizeof(s_q) / sizeof(double) == DIAG_BLOCK_SIZE
        && sizeof(s_c) / sizeof(long long) == DIAG_BLOCK_SIZE,
        "Shared-memory reduction array size must match DIAG_BLOCK_SIZE"
    );

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;
    if (tx == 0) {
        s_any = 0;
    }
    __syncthreads();

    double q_p = 0.0;
    long long c_p = 0LL;

    if (tid < n_act) {
        double uxi = ux[tid];
        double uyi = uy[tid];
        double uzi = uz[tid];
        // Same formula as the pusher; agrees to a few ulp(z)
        double inv_gamma = 1.0 / sqrt(1.0 + (uxi * uxi + uyi * uyi + uzi * uzi) * inv_c2);
        double dt_ig = dt * inv_gamma;
        double dz = dt_ig * uzi;
        double z_new = z[tid];
        double z_raw = z_new - dz;

        int cross = 0;
        if (!periodic_z) {
            if (z_raw < z_plane && z_plane <= z_new) {
                cross = 1;
            } else if (z_raw >= z_plane && z_plane > z_new) {
                cross = -1;
            }
        } else {
            double Lz = z_max - z_min;
            if (dz > 0.0) {
                if (z_raw < z_min) {
                    double z_old = z_raw + Lz;
                    if (z_old < z_plane || z_plane <= z_new) {
                        cross = 1;
                    }
                } else if (z_raw < z_plane && z_plane <= z_new) {
                    cross = 1;
                }
            } else if (dz < 0.0) {
                if (z_raw >= z_max) {
                    double z_old = z_raw - Lz;
                    if (z_old >= z_plane || z_plane > z_new) {
                        cross = -1;
                    }
                } else if (z_raw >= z_plane && z_plane > z_new) {
                    cross = -1;
                }
            }
        }

        if (cross != 0) {
            q_p = charge * w[tid] * (double)cross;
            c_p = (long long)cross;
            s_any = 1;
        }
    }

    s_q[tx] = q_p;
    s_c[tx] = c_p;
    __syncthreads();

    if (!s_any) return;

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s_q[tx] += s_q[tx + stride];
            s_c[tx] += s_c[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(&d_charge[slot], s_q[0]);
        atomicAdd(d_total_charge, s_q[0]);
        atomicAdd(
            (unsigned long long*)&d_counts[slot],
            (unsigned long long)s_c[0]
        );
        atomicAdd(
            (unsigned long long*)d_total_counts,
            (unsigned long long)s_c[0]
        );
    }
}

extern "C" __global__ void emittance_pass1_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    double z_lo,
    double z_hi,
    double inv_c,
    int n_host,
    const int* __restrict__ d_n_particles,
    double* __restrict__ stats,
    int base_offset
) {
    __shared__ double sdata[10][DIAG_BLOCK_SIZE];
    static_assert(
        sizeof(sdata[0]) / sizeof(double) == DIAG_BLOCK_SIZE,
        "Shared-memory reduction array size must match DIAG_BLOCK_SIZE"
    );

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;

    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
    double v5 = 0.0, v6 = 0.0, v7 = 0.0, v8 = 0.0, v9 = 0.0;

    if (tid < n_act) {
        double zi = z[tid];
        double uzi = uz[tid];
        if (zi >= z_lo && zi <= z_hi && uzi != 0.0) {
            double xi = x[tid];
            double yi = y[tid];
            double uxi = ux[tid];
            double uyi = uy[tid];
            double wt = w[tid];

            double xpi = uxi / uzi;
            double ypi = uyi / uzi;
            double u_mag = sqrt(uxi * uxi + uyi * uyi + uzi * uzi);

            v0 = wt;
            v1 = wt * xi;
            v2 = wt * xpi;
            v3 = wt * yi;
            v4 = wt * ypi;
            v5 = wt * (u_mag * inv_c);
            v6 = wt * (uzi * inv_c);
            v7 = 1.0;
            v8 = wt * uxi;
            v9 = wt * uyi;
        }
    }

    sdata[0][tx] = v0;
    sdata[1][tx] = v1;
    sdata[2][tx] = v2;
    sdata[3][tx] = v3;
    sdata[4][tx] = v4;
    sdata[5][tx] = v5;
    sdata[6][tx] = v6;
    sdata[7][tx] = v7;
    sdata[8][tx] = v8;
    sdata[9][tx] = v9;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            #pragma unroll
            for (int m = 0; m < 10; ++m) {
                sdata[m][tx] += sdata[m][tx + stride];
            }
        }
        __syncthreads();
    }

    if (tx == 0 && sdata[7][0] > 0.0) {
        atomicAdd(&stats[base_offset + 0], sdata[0][0]);
        atomicAdd(&stats[base_offset + 1], sdata[1][0]);
        atomicAdd(&stats[base_offset + 2], sdata[2][0]);
        atomicAdd(&stats[base_offset + 3], sdata[3][0]);
        atomicAdd(&stats[base_offset + 4], sdata[4][0]);
        atomicAdd(&stats[base_offset + 5], sdata[5][0]);
        atomicAdd(&stats[base_offset + 6], sdata[6][0]);
        atomicAdd(&stats[base_offset + 13], sdata[7][0]);
        atomicAdd(&stats[base_offset + 14], sdata[8][0]);
        atomicAdd(&stats[base_offset + 15], sdata[9][0]);
    }
}

extern "C" __global__ void emittance_pass2_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    double z_lo,
    double z_hi,
    int n_host,
    const int* __restrict__ d_n_particles,
    double* __restrict__ stats,
    int base_offset
) {
    __shared__ double sdata[10][DIAG_BLOCK_SIZE];
    static_assert(
        sizeof(sdata[0]) / sizeof(double) == DIAG_BLOCK_SIZE,
        "Shared-memory reduction array size must match DIAG_BLOCK_SIZE"
    );

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;

    double W = stats[base_offset + 0];
    if (W <= 0.0) return;

    double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
    double m5 = 0.0, m6 = 0.0, m7 = 0.0, m8 = 0.0, m9 = 0.0;

    if (tid < n_act) {
        double zi = z[tid];
        double uzi = uz[tid];
        if (zi >= z_lo && zi <= z_hi && uzi != 0.0) {
            double inv_W = 1.0 / W;
            double x_bar = stats[base_offset + 1] * inv_W;
            double xp_bar = stats[base_offset + 2] * inv_W;
            double y_bar = stats[base_offset + 3] * inv_W;
            double yp_bar = stats[base_offset + 4] * inv_W;
            double ux_bar = stats[base_offset + 14] * inv_W;
            double uy_bar = stats[base_offset + 15] * inv_W;

            double uxi = ux[tid];
            double uyi = uy[tid];
            double dx = x[tid] - x_bar;
            double dxp = (uxi / uzi) - xp_bar;
            double dy = y[tid] - y_bar;
            double dyp = (uyi / uzi) - yp_bar;
            double dux = uxi - ux_bar;
            double duy = uyi - uy_bar;
            double wt = w[tid];

            m0 = wt * dx * dx;
            m1 = wt * dxp * dxp;
            m2 = wt * dx * dxp;
            m3 = wt * dy * dy;
            m4 = wt * dyp * dyp;
            m5 = wt * dy * dyp;
            m6 = wt * dux * dux;
            m7 = wt * dx * dux;
            m8 = wt * duy * duy;
            m9 = wt * dy * duy;
        }
    }

    sdata[0][tx] = m0;
    sdata[1][tx] = m1;
    sdata[2][tx] = m2;
    sdata[3][tx] = m3;
    sdata[4][tx] = m4;
    sdata[5][tx] = m5;
    sdata[6][tx] = m6;
    sdata[7][tx] = m7;
    sdata[8][tx] = m8;
    sdata[9][tx] = m9;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            #pragma unroll
            for (int m = 0; m < 10; ++m) {
                sdata[m][tx] += sdata[m][tx + stride];
            }
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(&stats[base_offset + 7], sdata[0][0]);
        atomicAdd(&stats[base_offset + 8], sdata[1][0]);
        atomicAdd(&stats[base_offset + 9], sdata[2][0]);
        atomicAdd(&stats[base_offset + 10], sdata[3][0]);
        atomicAdd(&stats[base_offset + 11], sdata[4][0]);
        atomicAdd(&stats[base_offset + 12], sdata[5][0]);
        atomicAdd(&stats[base_offset + 16], sdata[6][0]);
        atomicAdd(&stats[base_offset + 17], sdata[7][0]);
        atomicAdd(&stats[base_offset + 18], sdata[8][0]);
        atomicAdd(&stats[base_offset + 19], sdata[9][0]);
    }
}

extern "C" __global__ void track_recorder_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const long long* __restrict__ sp_ids,
    int n_host,
    const int* __restrict__ d_n_particles,
    const long long* __restrict__ target_ids,
    int n_targets,
    double* __restrict__ tracks,
    int frame_offset,
    int* __restrict__ match_counts,
    int match_offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_act = d_n_particles ? d_n_particles[0] : n_host;
    if (tid >= n_act) return;

    long long pid = sp_ids[tid];
    for (int k = 0; k < n_targets; ++k) {
        if (pid == target_ids[k]) {
            atomicAdd(&match_counts[match_offset + k], 1);
            int base = frame_offset + 6 * k;
            tracks[base + 0] = x[tid];
            tracks[base + 1] = y[tid];
            tracks[base + 2] = z[tid];
            tracks[base + 3] = ux[tid];
            tracks[base + 4] = uy[tid];
            tracks[base + 5] = uz[tid];
        }
    }
}
"""
)

_DIAG_MODULE = None
_DIAG_KERNELS: dict[str, object] = {}


def _get_diag_kernel(name: str):
    """Compile and cache CUDA diagnostic kernels on first GPU use."""
    global _DIAG_MODULE
    assert (
        isinstance(_BLOCK_SIZE, int)
        and _BLOCK_SIZE > 0
        and (_BLOCK_SIZE & (_BLOCK_SIZE - 1)) == 0
        and _SMEM_REDUCTION_SIZE == _BLOCK_SIZE
    ), (
        f"_BLOCK_SIZE must be a positive power of two matching shared-memory "
        f"reduction array size, got {_BLOCK_SIZE} vs {_SMEM_REDUCTION_SIZE}."
    )
    if _DIAG_MODULE is None:
        import cupy as cp

        _DIAG_MODULE = cp.RawModule(
            code=_DIAG_CUDA_SRC,
            options=("-std=c++14", f"-DDIAG_BLOCK_SIZE={_BLOCK_SIZE}"),
        )
        for kname in (
            "zero_f64_kernel",
            "zero_hist_frame_kernel",
            "zero_probe_slot_kernel",
            "fill_nan_f64_kernel",
            "reset_track_frame_kernel",
            "phase_space_hist2d_kernel",
            "energy_spectrum_hist1d_kernel",
            "plane_current_probe_kernel",
            "emittance_pass1_kernel",
            "emittance_pass2_kernel",
            "track_recorder_kernel",
        ):
            _DIAG_KERNELS[kname] = _DIAG_MODULE.get_function(kname)
    return _DIAG_KERNELS[name]


def _grid_1d(n: int) -> tuple[tuple[int], tuple[int]]:
    """Return 1D CUDA ``(grid, block)`` tuples for ``n`` threads."""
    n_pos = max(int(n), 1)
    return (((n_pos + _BLOCK_SIZE - 1) // _BLOCK_SIZE,), (_BLOCK_SIZE,))


class DiagnosticResult(dict):
    """Dictionary of host NumPy diagnostic outputs supporting attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _find_bins_cpu(
    vals: np.ndarray,
    first_edge: float,
    last_edge: float,
    n_bins: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 0-based bin indices and in-range mask matching ``numpy.histogram``."""
    in_range = (vals >= first_edge) & (vals <= last_edge)
    if not bool(np.any(in_range)):
        return np.empty(0, dtype=np.intp), in_range

    v_in = vals[in_range]
    f_idx = ((v_in - first_edge) / (last_edge - first_edge)) * n_bins
    idx = f_idx.astype(np.intp)
    idx[idx >= n_bins] = n_bins - 1
    idx[idx < 0] = 0
    dec = (v_in < edges[idx]) & (idx > 0)
    idx[dec] -= 1
    inc = (v_in >= edges[idx + 1]) & (idx < n_bins - 1)
    idx[inc] += 1
    return idx, in_range


def _eval_axis_cpu(code: int, sp: Species) -> np.ndarray:
    """Evaluate phase-space axis ``code`` on CPU :class:`Species` ``sp``."""
    n = sp.n_particles
    if code == 0:
        return sp._x[:n]
    if code == 1:
        return sp._y[:n]
    if code == 2:
        return sp._z[:n]
    if code == 3:
        return sp._ux[:n]
    if code == 4:
        return sp._uy[:n]
    if code == 5:
        return sp._uz[:n]
    ux = sp._ux[:n]
    uy = sp._uy[:n]
    uz = sp._uz[:n]
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (c_light * c_light))
    if code == 6:
        return ux / gamma
    if code == 7:
        return uy / gamma
    if code == 8:
        return uz / gamma
    if code == 9:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(uz != 0.0, ux / uz, np.nan)
    if code == 10:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(uz != 0.0, uy / uz, np.nan)
    if code == 11:
        return gamma
    return (sp.mass / e_charge) * (u2 / (gamma + 1.0))


class _BaseDiagnostic:
    """Shared base class for GPU-resident, MPI-aware particle diagnostics."""

    def __init__(
        self,
        every: int = 1,
        species: Species | str | int | None = None,
    ) -> None:
        if int(every) < 1:
            raise ValueError(f"every must be >= 1, got {every}.")
        self.every = int(every)
        self.species = species
        self._solver_ref = None
        self._on_gpu = False
        self._bound = False

    def _select_species(self, source) -> list[Species]:
        """Return the list of active :class:`Species` objects to sample."""
        if isinstance(source, Species):
            return [source]
        sp_list = list(source.species)
        if self.species is None:
            return sp_list
        if isinstance(self.species, int):
            return [sp_list[self.species]]
        if isinstance(self.species, str):
            matched = [sp for sp in sp_list if sp.name == self.species]
            if not matched:
                raise ValueError(
                    f"No Species named {self.species!r} found in ParticleSolver."
                )
            return matched
        if isinstance(self.species, Species):
            matched = [
                sp
                for sp in sp_list
                if sp is self.species or sp.name == self.species.name
            ]
            return matched if matched else [self.species]
        raise TypeError(f"Unsupported species selector {self.species!r}.")

    def _get_comm(self):
        """Return the active MPI communicator if attached to a multi-rank solver."""
        ps = self._solver_ref
        if ps is None:
            return None
        s = getattr(ps, "solver", None)
        if (
            getattr(ps, "use_mpi", False)
            and s is not None
            and int(getattr(s, "size", 1)) > 1
        ):
            return s.comm
        return None

    def sample(
        self,
        source,
        t: float = 0.0,
        dt: float = 0.0,
        step: int | None = None,
    ) -> None:
        """Record one diagnostic sample from a :class:`Species` or :class:`ParticleSolver`."""
        if not self._bound:
            self._bind(source)
        st = 0 if step is None else int(step)
        self._record(source, float(t), float(dt), st)

    def to_numpy(self) -> DiagnosticResult:
        """Alias for :meth:`result` returning host NumPy arrays."""
        return self.result()


class PhaseSpaceMonitor(_BaseDiagnostic):
    """Two-dimensional phase-space histogram monitor (accumulated and/or ring buffer).

    Parameters
    ----------
    axes : tuple of 2 str, optional
        Pair of phase-space coordinate names chosen from ``("x", "y", "z",
        "ux", "uy", "uz", "vx", "vy", "vz", "xp", "yp", "gamma", "ek_eV")``.
        Default is ``("z", "uz")``.
    bins : int or tuple of 2 int, optional
        Number of uniform histogram bins along ``axes[0]`` and ``axes[1]``.
        Default is ``64``.
    ranges : tuple of 2 (float, float), optional
        Closed outer bin limits ``((min0, max0), (min1, max1))`` in the physical
        units of ``axes``. Default is ``((-1.0, 1.0), (-1.0, 1.0))``.
    weighted : bool, optional
        If True (default), accumulate macroparticle weights ``w_p``; if False,
        accumulate unweighted macroparticle counts ``1``.
    every : int, optional
        Sampling interval in particle steps. Default is ``1``.
    species : Species, str, int, or None, optional
        Restrict monitor to a specific species (or ``None`` for all species).
    ring_size : int or None, optional
        If provided, also retain the most recent ``ring_size`` individual
        snapshot frames in a preallocated device ring buffer.

    Notes
    -----
    * **Time levels:** At the end-of-step diagnostic hook, spatial coordinates
      ``(x, y, z)`` are at ``t_{n+1}`` while proper velocities ``(ux, uy, uz)``
      and derived slopes/energies are at ``t_{n+1/2}``.
    * **Binning convention:** Matches :func:`numpy.histogram2d`—bins
      ``0 .. B - 2`` are half-open ``[edge[b], edge[b+1])`` and the last bin
      ``B - 1`` is closed ``[edge[B-1], edge[B]]``. Out-of-range particles are
      accumulated in ``overflow`` (weight/count) and ``overflow_count``.
    * **GPU shared-memory privatisation:** When the total number of bins
      ``nb0 * nb1 <= 3072`` (24 KB of ``double`` shared memory per thread
      block), each CUDA thread block accumulates into a block-private
      shared-memory histogram and flushes non-zero bins to global memory once
      per block. Larger histograms fall back to direct global ``atomicAdd``.
    * **MPI semantics:** Each rank accumulates locally on its device without
      communication. Calling :meth:`result` (or :meth:`to_numpy`) performs a
      collective ``MPI.SUM`` reduction across all ranks.
    """

    def __init__(
        self,
        axes: tuple[str, str] = ("z", "uz"),
        bins: int | tuple[int, int] = 64,
        ranges: tuple[tuple[float, float], tuple[float, float]] = (
            (-1.0, 1.0),
            (-1.0, 1.0),
        ),
        weighted: bool = True,
        *,
        every: int = 1,
        species: Species | str | int | None = None,
        ring_size: int | None = None,
    ) -> None:
        super().__init__(every=every, species=species)
        if len(axes) != 2:
            raise ValueError(f"axes must be a 2-tuple, got {axes!r}.")
        ax0 = str(axes[0]).strip().lower()
        ax1 = str(axes[1]).strip().lower()
        if ax0 not in _AXIS_CODES or ax1 not in _AXIS_CODES:
            raise ValueError(
                f"Unsupported axes={axes!r}; valid axes are {sorted(_AXIS_CODES)}."
            )
        self.axes = (str(axes[0]), str(axes[1]))
        self._ax0_code = _AXIS_CODES[ax0]
        self._ax1_code = _AXIS_CODES[ax1]

        if isinstance(bins, (int, np.integer)):
            nb0 = nb1 = int(bins)
        else:
            if len(bins) != 2:
                raise ValueError(f"bins must be int or 2-tuple, got {bins!r}.")
            nb0, nb1 = int(bins[0]), int(bins[1])
        if nb0 < 1 or nb1 < 1:
            raise ValueError(f"bins must be >= 1, got {(nb0, nb1)}.")
        self.bins = (nb0, nb1)

        if len(ranges) != 2 or len(ranges[0]) != 2 or len(ranges[1]) != 2:
            raise ValueError(
                f"ranges must be ((min0, max0), (min1, max1)), got {ranges!r}."
            )
        min0, max0 = float(ranges[0][0]), float(ranges[0][1])
        min1, max1 = float(ranges[1][0]), float(ranges[1][1])
        if not (max0 > min0 and max1 > min1):
            raise ValueError(
                f"ranges must have max > min on both axes, got {ranges!r}."
            )
        self.ranges = ((min0, max0), (min1, max1))
        self.weighted = bool(weighted)

        if ring_size is not None and int(ring_size) < 1:
            raise ValueError(f"ring_size must be >= 1, got {ring_size}.")
        self.ring_size = None if ring_size is None else int(ring_size)

        self.x_edges = np.linspace(min0, max0, nb0 + 1, dtype=np.float64)
        self.y_edges = np.linspace(min1, max1, nb1 + 1, dtype=np.float64)

        self._hist_cpu = np.zeros((nb0, nb1), dtype=np.float64)
        self._ovf_cpu = 0.0
        self._ovf_cnt_cpu = 0
        self._frames_cpu = (
            np.zeros((self.ring_size, nb0, nb1), dtype=np.float64)
            if self.ring_size is not None
            else None
        )
        self._frame_ovf_cpu = (
            np.zeros(self.ring_size, dtype=np.float64)
            if self.ring_size is not None
            else None
        )
        self._frame_ovf_cnt_cpu = (
            np.zeros(self.ring_size, dtype=np.int64)
            if self.ring_size is not None
            else None
        )
        cap_t = self.ring_size if self.ring_size is not None else 16384
        self._times = np.zeros(cap_t, dtype=np.float64)
        self._steps = np.zeros(cap_t, dtype=np.int64)
        self._frame_count = 0

        self._d_edges0 = None
        self._d_edges1 = None
        self._d_hist = None
        self._d_ovf = None
        self._d_ovf_cnt = None
        self._d_frames = None
        self._d_frame_ovf = None
        self._d_frame_ovf_cnt = None

    def _bind(self, source) -> None:
        if not isinstance(source, Species):
            self._solver_ref = source
        on_gpu = bool(getattr(source, "on_gpu", False))
        self._on_gpu = on_gpu
        if on_gpu and self._d_hist is None:
            import cupy as cp

            _get_diag_kernel("phase_space_hist2d_kernel")
            nb0, nb1 = self.bins
            self._d_edges0 = cp.asarray(self.x_edges)
            self._d_edges1 = cp.asarray(self.y_edges)
            self._d_hist = cp.zeros(nb0 * nb1, dtype=np.float64)
            self._d_ovf = cp.zeros(1, dtype=np.float64)
            self._d_ovf_cnt = cp.zeros(1, dtype=np.uint64)
            if self.ring_size is not None:
                self._d_frames = cp.zeros(
                    self.ring_size * nb0 * nb1, dtype=np.float64
                )
                self._d_frame_ovf = cp.zeros(self.ring_size, dtype=np.float64)
                self._d_frame_ovf_cnt = cp.zeros(
                    self.ring_size, dtype=np.uint64
                )
        self._bound = True

    def _record(self, source, t: float, dt: float, step: int) -> None:
        if not self._bound or bool(getattr(source, "on_gpu", False)) != self._on_gpu:
            self._bind(source)
        sp_list = self._select_species(source)
        nb0, nb1 = self.bins
        nb_tot = nb0 * nb1
        (min0, max0), (min1, max1) = self.ranges

        idx = self._frame_count
        slot = idx % self.ring_size if self.ring_size is not None else 0
        if self.ring_size is not None:
            self._times[slot] = t
            self._steps[slot] = step
        elif idx < self._times.size:
            self._times[idx] = t
            self._steps[idx] = step
        self._frame_count = idx + 1

        if self._on_gpu:
            frame_offset = slot * nb_tot
            if self.ring_size is not None:
                z_kern = _get_diag_kernel("zero_hist_frame_kernel")
                g_z, b_z = _grid_1d(nb_tot)
                z_kern(
                    g_z,
                    b_z,
                    (
                        self._d_frames,
                        np.int32(frame_offset),
                        np.int32(nb_tot),
                        self._d_frame_ovf,
                        self._d_frame_ovf_cnt,
                        np.int32(slot),
                    ),
                )

            kern = _get_diag_kernel("phase_space_hist2d_kernel")
            inv_c2 = np.float64(1.0 / (c_light * c_light))
            inv_c = np.float64(1.0 / c_light)
            inv_e = np.float64(1.0 / e_charge)
            d_frames_ptr = 0 if self._d_frames is None else self._d_frames
            d_fovf_ptr = 0 if self._d_frame_ovf is None else self._d_frame_ovf
            d_fcnt_ptr = (
                0 if self._d_frame_ovf_cnt is None else self._d_frame_ovf_cnt
            )

            for sp in sp_list:
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                grid, block = _grid_1d(n_l)
                kern(
                    grid,
                    block,
                    (
                        sp._x,
                        sp._y,
                        sp._z,
                        sp._ux,
                        sp._uy,
                        sp._uz,
                        sp._w,
                        np.float64(sp.mass),
                        inv_c2,
                        inv_c,
                        inv_e,
                        np.int32(n_l),
                        d_n_ptr,
                        np.int32(self._ax0_code),
                        np.int32(self._ax1_code),
                        np.float64(min0),
                        np.float64(max0),
                        np.int32(nb0),
                        self._d_edges0,
                        np.float64(min1),
                        np.float64(max1),
                        np.int32(nb1),
                        self._d_edges1,
                        np.int32(1 if self.weighted else 0),
                        self._d_hist,
                        self._d_ovf,
                        self._d_ovf_cnt,
                        d_frames_ptr,
                        np.int32(frame_offset),
                        d_fovf_ptr,
                        d_fcnt_ptr,
                        np.int32(slot),
                    ),
                )
            return

        if self.ring_size is not None:
            self._frames_cpu[slot].fill(0.0)
            self._frame_ovf_cpu[slot] = 0.0
            self._frame_ovf_cnt_cpu[slot] = 0

        for sp in sp_list:
            n_p = sp.n_particles
            if n_p == 0:
                continue
            v0 = _eval_axis_cpu(self._ax0_code, sp)
            v1 = _eval_axis_cpu(self._ax1_code, sp)
            w_arr = sp._w[:n_p] if self.weighted else None

            _, in0 = _find_bins_cpu(v0, min0, max0, nb0, self.x_edges)
            _, in1 = _find_bins_cpu(v1, min1, max1, nb1, self.y_edges)
            in_both = in0 & in1
            n_in = int(np.sum(in_both))
            n_out = n_p - n_in

            if n_out > 0:
                ovf_val = (
                    float(np.sum(w_arr[~in_both]))
                    if w_arr is not None
                    else float(n_out)
                )
                self._ovf_cpu += ovf_val
                self._ovf_cnt_cpu += n_out
                if self.ring_size is not None:
                    self._frame_ovf_cpu[slot] += ovf_val
                    self._frame_ovf_cnt_cpu[slot] += n_out

            if n_in > 0:
                b0, _ = _find_bins_cpu(
                    v0[in_both], min0, max0, nb0, self.x_edges
                )
                b1, _ = _find_bins_cpu(
                    v1[in_both], min1, max1, nb1, self.y_edges
                )
                flat = b0 * nb1 + b1
                w_in = w_arr[in_both] if w_arr is not None else None
                h_step = np.bincount(
                    flat, weights=w_in, minlength=nb_tot
                ).astype(np.float64).reshape((nb0, nb1))
                self._hist_cpu += h_step
                if self.ring_size is not None:
                    self._frames_cpu[slot] += h_step

    def result(self) -> DiagnosticResult:
        """Return the host 2D phase-space histogram (collective across MPI ranks)."""
        nb0, nb1 = self.bins
        if self._on_gpu and self._d_hist is not None:
            hist = self._d_hist.get().reshape((nb0, nb1)).copy()
            ovf = float(self._d_ovf.get()[0])
            ovf_cnt = int(self._d_ovf_cnt.get()[0])
            if self.ring_size is not None:
                frames_raw = self._d_frames.get().reshape(
                    (self.ring_size, nb0, nb1)
                )
                fovf_raw = self._d_frame_ovf.get().copy()
                fcnt_raw = self._d_frame_ovf_cnt.get().astype(np.int64)
        else:
            hist = self._hist_cpu.copy()
            ovf = float(self._ovf_cpu)
            ovf_cnt = int(self._ovf_cnt_cpu)
            if self.ring_size is not None:
                frames_raw = self._frames_cpu.copy()
                fovf_raw = self._frame_ovf_cpu.copy()
                fcnt_raw = self._frame_ovf_cnt_cpu.copy()

        comm = self._get_comm()
        if comm is not None:
            from mpi4py import MPI

            comm.Allreduce(MPI.IN_PLACE, hist, op=MPI.SUM)
            ovf = float(comm.allreduce(ovf, op=MPI.SUM))
            ovf_cnt = int(comm.allreduce(ovf_cnt, op=MPI.SUM))
            if self.ring_size is not None:
                comm.Allreduce(MPI.IN_PLACE, frames_raw, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, fovf_raw, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, fcnt_raw, op=MPI.SUM)

        out = DiagnosticResult(
            hist=hist,
            x_edges=self.x_edges.copy(),
            y_edges=self.y_edges.copy(),
            edges=(self.x_edges.copy(), self.y_edges.copy()),
            axes=self.axes,
            overflow=ovf,
            overflow_count=ovf_cnt,
            n_samples=self._frame_count,
        )
        if self.ring_size is not None:
            n_stored = min(self._frame_count, self.ring_size)
            if self._frame_count <= self.ring_size:
                frames = frames_raw[:n_stored].copy()
                fovf = fovf_raw[:n_stored].copy()
                fcnt = fcnt_raw[:n_stored].copy()
                times = self._times[:n_stored].copy()
                steps = self._steps[:n_stored].copy()
            else:
                shift = -(self._frame_count % self.ring_size)
                frames = np.roll(frames_raw, shift, axis=0)
                fovf = np.roll(fovf_raw, shift)
                fcnt = np.roll(fcnt_raw, shift)
                times = np.roll(self._times, shift)
                steps = np.roll(self._steps, shift)
            out["frames"] = frames
            out["snapshots"] = frames
            out["frame_overflow"] = fovf
            out["frame_overflow_count"] = fcnt
            out["times"] = times
            out["t"] = times
            out["steps"] = steps
        return out


class EnergySpectrumMonitor(_BaseDiagnostic):
    """One-dimensional kinetic-energy histogram monitor in electron-volts [eV].

    Parameters
    ----------
    bins : int, optional
        Number of uniform energy bins across ``range_eV``. Default is ``64``.
    range_eV : tuple of (float, float), optional
        Closed energy interval ``(E_min_eV, E_max_eV)`` [eV]. Default is
        ``(0.0, 1.0e6)``.
    weighted : bool, optional
        If True (default), accumulate physical macroparticle weights ``w_p``;
        if False, accumulate unweighted macroparticle counts ``1``.
    every : int, optional
        Sampling interval in particle steps. Default is ``1``.
    species : Species, str, int, or None, optional
        Restrict monitor to a specific species (or ``None`` for all species).
    ring_size : int or None, optional
        If provided, also retain the most recent ``ring_size`` individual
        snapshot spectra in a preallocated device ring buffer.

    Notes
    -----
    * **Time level:** Evaluates the single-particle relativistic kinetic energy
      ``E_k = (m / e) * |u^{n+1/2}|^2 / (gamma^{n+1/2} + 1)`` [eV] directly from
      the leapfrog half-step proper velocity ``u^{n+1/2}`` at the end of the
      step. This avoids catastrophic cancellation at non-relativistic speeds
      ``|u| << c`` without requiring an extra field gather for velocity
      synchronization.
    * **GPU shared-memory privatisation:** When ``bins <= 3072`` (24 KB of
      ``double`` shared memory per thread block), each CUDA thread block
      accumulates into a block-private shared-memory histogram and flushes
      non-zero bins to global memory once per block.
    * **MPI semantics:** Calling :meth:`result` performs a collective
      ``MPI.SUM`` reduction across all ranks.
    """

    def __init__(
        self,
        bins: int = 64,
        range_eV: tuple[float, float] = (0.0, 1.0e6),
        weighted: bool = True,
        *,
        every: int = 1,
        species: Species | str | int | None = None,
        ring_size: int | None = None,
    ) -> None:
        super().__init__(every=every, species=species)
        nb = int(bins)
        if nb < 1:
            raise ValueError(f"bins must be >= 1, got {bins}.")
        if len(range_eV) != 2:
            raise ValueError(
                f"range_eV must be (min_eV, max_eV), got {range_eV!r}."
            )
        min_eV, max_eV = float(range_eV[0]), float(range_eV[1])
        if not (max_eV > min_eV):
            raise ValueError(
                f"range_eV must have max_eV > min_eV, got {range_eV!r}."
            )
        self.bins = nb
        self.range_eV = (min_eV, max_eV)
        self.weighted = bool(weighted)
        if ring_size is not None and int(ring_size) < 1:
            raise ValueError(f"ring_size must be >= 1, got {ring_size}.")
        self.ring_size = None if ring_size is None else int(ring_size)

        self.edges = np.linspace(min_eV, max_eV, nb + 1, dtype=np.float64)
        self.centers_eV = 0.5 * (self.edges[:-1] + self.edges[1:])

        self._hist_cpu = np.zeros(nb, dtype=np.float64)
        self._ovf_cpu = 0.0
        self._ovf_cnt_cpu = 0
        self._frames_cpu = (
            np.zeros((self.ring_size, nb), dtype=np.float64)
            if self.ring_size is not None
            else None
        )
        self._frame_ovf_cpu = (
            np.zeros(self.ring_size, dtype=np.float64)
            if self.ring_size is not None
            else None
        )
        self._frame_ovf_cnt_cpu = (
            np.zeros(self.ring_size, dtype=np.int64)
            if self.ring_size is not None
            else None
        )
        cap_t = self.ring_size if self.ring_size is not None else 16384
        self._times = np.zeros(cap_t, dtype=np.float64)
        self._steps = np.zeros(cap_t, dtype=np.int64)
        self._frame_count = 0

        self._d_edges = None
        self._d_hist = None
        self._d_ovf = None
        self._d_ovf_cnt = None
        self._d_frames = None
        self._d_frame_ovf = None
        self._d_frame_ovf_cnt = None

    def _bind(self, source) -> None:
        if not isinstance(source, Species):
            self._solver_ref = source
        on_gpu = bool(getattr(source, "on_gpu", False))
        self._on_gpu = on_gpu
        if on_gpu and self._d_hist is None:
            import cupy as cp

            _get_diag_kernel("energy_spectrum_hist1d_kernel")
            nb = self.bins
            self._d_edges = cp.asarray(self.edges)
            self._d_hist = cp.zeros(nb, dtype=np.float64)
            self._d_ovf = cp.zeros(1, dtype=np.float64)
            self._d_ovf_cnt = cp.zeros(1, dtype=np.uint64)
            if self.ring_size is not None:
                self._d_frames = cp.zeros(
                    self.ring_size * nb, dtype=np.float64
                )
                self._d_frame_ovf = cp.zeros(self.ring_size, dtype=np.float64)
                self._d_frame_ovf_cnt = cp.zeros(
                    self.ring_size, dtype=np.uint64
                )
        self._bound = True

    def _record(self, source, t: float, dt: float, step: int) -> None:
        if not self._bound or bool(getattr(source, "on_gpu", False)) != self._on_gpu:
            self._bind(source)
        sp_list = self._select_species(source)
        nb = self.bins
        min_eV, max_eV = self.range_eV

        idx = self._frame_count
        slot = idx % self.ring_size if self.ring_size is not None else 0
        if self.ring_size is not None:
            self._times[slot] = t
            self._steps[slot] = step
        elif idx < self._times.size:
            self._times[idx] = t
            self._steps[idx] = step
        self._frame_count = idx + 1

        if self._on_gpu:
            frame_offset = slot * nb
            if self.ring_size is not None:
                z_kern = _get_diag_kernel("zero_hist_frame_kernel")
                g_z, b_z = _grid_1d(nb)
                z_kern(
                    g_z,
                    b_z,
                    (
                        self._d_frames,
                        np.int32(frame_offset),
                        np.int32(nb),
                        self._d_frame_ovf,
                        self._d_frame_ovf_cnt,
                        np.int32(slot),
                    ),
                )

            kern = _get_diag_kernel("energy_spectrum_hist1d_kernel")
            inv_c2 = np.float64(1.0 / (c_light * c_light))
            d_frames_ptr = 0 if self._d_frames is None else self._d_frames
            d_fovf_ptr = 0 if self._d_frame_ovf is None else self._d_frame_ovf
            d_fcnt_ptr = (
                0 if self._d_frame_ovf_cnt is None else self._d_frame_ovf_cnt
            )

            for sp in sp_list:
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                grid, block = _grid_1d(n_l)
                kern(
                    grid,
                    block,
                    (
                        sp._ux,
                        sp._uy,
                        sp._uz,
                        sp._w,
                        np.float64(sp.mass / e_charge),
                        inv_c2,
                        np.int32(n_l),
                        d_n_ptr,
                        np.float64(min_eV),
                        np.float64(max_eV),
                        np.int32(nb),
                        self._d_edges,
                        np.int32(1 if self.weighted else 0),
                        self._d_hist,
                        self._d_ovf,
                        self._d_ovf_cnt,
                        d_frames_ptr,
                        np.int32(frame_offset),
                        d_fovf_ptr,
                        d_fcnt_ptr,
                        np.int32(slot),
                    ),
                )
            return

        if self.ring_size is not None:
            self._frames_cpu[slot].fill(0.0)
            self._frame_ovf_cpu[slot] = 0.0
            self._frame_ovf_cnt_cpu[slot] = 0

        for sp in sp_list:
            n_p = sp.n_particles
            if n_p == 0:
                continue
            ek_eV = _eval_axis_cpu(12, sp)
            w_arr = sp._w[:n_p] if self.weighted else None
            b_idx, in_range = _find_bins_cpu(
                ek_eV, min_eV, max_eV, nb, self.edges
            )
            n_in = int(b_idx.size)
            n_out = n_p - n_in

            if n_out > 0:
                ovf_val = (
                    float(np.sum(w_arr[~in_range]))
                    if w_arr is not None
                    else float(n_out)
                )
                self._ovf_cpu += ovf_val
                self._ovf_cnt_cpu += n_out
                if self.ring_size is not None:
                    self._frame_ovf_cpu[slot] += ovf_val
                    self._frame_ovf_cnt_cpu[slot] += n_out

            if n_in > 0:
                w_in = w_arr[in_range] if w_arr is not None else None
                h_step = np.bincount(
                    b_idx, weights=w_in, minlength=nb
                ).astype(np.float64)
                self._hist_cpu += h_step
                if self.ring_size is not None:
                    self._frames_cpu[slot] += h_step

    def result(self) -> DiagnosticResult:
        """Return the host 1D kinetic-energy spectrum (collective across MPI ranks)."""
        if self._on_gpu and self._d_hist is not None:
            hist = self._d_hist.get().copy()
            ovf = float(self._d_ovf.get()[0])
            ovf_cnt = int(self._d_ovf_cnt.get()[0])
            if self.ring_size is not None:
                frames_raw = self._d_frames.get().reshape(
                    (self.ring_size, self.bins)
                )
                fovf_raw = self._d_frame_ovf.get().copy()
                fcnt_raw = self._d_frame_ovf_cnt.get().astype(np.int64)
        else:
            hist = self._hist_cpu.copy()
            ovf = float(self._ovf_cpu)
            ovf_cnt = int(self._ovf_cnt_cpu)
            if self.ring_size is not None:
                frames_raw = self._frames_cpu.copy()
                fovf_raw = self._frame_ovf_cpu.copy()
                fcnt_raw = self._frame_ovf_cnt_cpu.copy()

        comm = self._get_comm()
        if comm is not None:
            from mpi4py import MPI

            comm.Allreduce(MPI.IN_PLACE, hist, op=MPI.SUM)
            ovf = float(comm.allreduce(ovf, op=MPI.SUM))
            ovf_cnt = int(comm.allreduce(ovf_cnt, op=MPI.SUM))
            if self.ring_size is not None:
                comm.Allreduce(MPI.IN_PLACE, frames_raw, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, fovf_raw, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, fcnt_raw, op=MPI.SUM)

        out = DiagnosticResult(
            hist=hist,
            edges=self.edges.copy(),
            bin_edges=self.edges.copy(),
            centers_eV=self.centers_eV.copy(),
            energy_eV=self.centers_eV.copy(),
            range_eV=self.range_eV,
            overflow=ovf,
            overflow_count=ovf_cnt,
            n_samples=self._frame_count,
        )
        if self.ring_size is not None:
            n_stored = min(self._frame_count, self.ring_size)
            if self._frame_count <= self.ring_size:
                frames = frames_raw[:n_stored].copy()
                fovf = fovf_raw[:n_stored].copy()
                fcnt = fcnt_raw[:n_stored].copy()
                times = self._times[:n_stored].copy()
                steps = self._steps[:n_stored].copy()
            else:
                shift = -(self._frame_count % self.ring_size)
                frames = np.roll(frames_raw, shift, axis=0)
                fovf = np.roll(fovf_raw, shift)
                fcnt = np.roll(fcnt_raw, shift)
                times = np.roll(self._times, shift)
                steps = np.roll(self._steps, shift)
            out["frames"] = frames
            out["snapshots"] = frames
            out["frame_overflow"] = fovf
            out["frame_overflow_count"] = fcnt
            out["times"] = times
            out["t"] = times
            out["steps"] = steps
        return out


class PlaneCurrentProbe(_BaseDiagnostic):
    """Longitudinal plane current probe recording net charge crossing ``z = z_plane``.

    Parameters
    ----------
    z_plane : float
        Longitudinal position ``z`` [m] of the measurement plane.
    every : int, optional
        Sampling interval in particle steps. Crossing detection runs on every
        step and accumulates into the active slot, which advances every
        ``every`` steps so ``I = Q_slot / (every * dt)``. Default is ``1``.
    species : Species, str, int, or None, optional
        Restrict probe to a specific species (or ``None`` for all species).
    capacity : int, optional
        Preallocated number of time samples in the device ring buffer.
        Default is ``16384``.

    Notes
    -----
    * **Leapfrog trajectory reconstruction:** During step ``t_n -> t_{n+1}``,
      the particle pusher updates proper velocity ``u^{n-1/2} -> u^{n+1/2}`` and
      immediately advances position via ``z^{n+1} = z^n + dz`` using
      ``inv_gamma = 1 / sqrt(1 + |u^{n+1/2}|^2 / c^2)`` and
      ``dz = (dt * inv_gamma) * u_z^{n+1/2}`` (or ``(dt / gamma) * u_z`` on the
      chunked CPU path for ``n > 65536``). Reconstructing ``z_old = z^{n+1} - dz``
      uses the same formula as the pusher; agrees to a few ``ulp(z)`` of IEEE-754
      floating-point round-off error. Known differences between the diagnostic
      reconstruction and the pusher position advance:
      (i) the fused CPU path for fully periodic domains
      (``solver.py`` ~l. 2831-2838) always uses ``fac = dt/sqrt(...)``;
      (ii) the ``n_p > 65536`` branch test uses the particle count after
      compaction; and
      (iii) ``nvcc`` FMA contraction.
    * **Placement near injectors and collectors:** A :class:`UserWarning` is
      emitted if ``z_plane`` equals an injector plane ``z0`` or lies within
      ``c * dt`` of an absorbing collector boundary, because particles absorbed
      by a collector during step ``t_n -> t_{n+1}`` are removed before the
      end-of-step diagnostic hook and are not counted.
    * **Running totals:** ``Q_total`` and ``Q_cumulative`` include charge from
      the entire run via a device-side running total, even when the number of
      recorded samples exceeds ``capacity``.
    * **Signed crossing rule:** A particle contributes ``+q_s * w_p`` when
      ``z_old < z_plane <= z_new`` (forward crossing) and ``-q_s * w_p`` when
      ``z_old >= z_plane > z_new`` (backward crossing). On periodic ``z``
      domains ``[z_min, z_max)``, particles that wrapped across ``z_max <-> z_min``
      during the current step are unwrapped before testing the crossing.
    * **MPI semantics:** Because :meth:`ParticleSolver._run_diagnostics` runs
      after MPI particle migration, every active particle is owned by exactly
      one rank at the end of the step. Each rank accumulates crossings among
      its post-migration particles, and :meth:`result` performs a collective
      ``MPI.SUM`` reduction across all ranks.
    """

    def __init__(
        self,
        z_plane: float,
        *,
        every: int = 1,
        species: Species | str | int | None = None,
        capacity: int = 16384,
    ) -> None:
        super().__init__(every=every, species=species)
        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.z_plane = float(z_plane)
        self.capacity = int(capacity)

        self._charge_cpu = np.zeros(self.capacity, dtype=np.float64)
        self._counts_cpu = np.zeros(self.capacity, dtype=np.int64)
        self._total_charge_cpu = 0.0
        self._total_counts_cpu = 0
        self._times = np.zeros(self.capacity, dtype=np.float64)
        self._dts = np.zeros(self.capacity, dtype=np.float64)
        self._steps = np.zeros(self.capacity, dtype=np.int64)
        self._sample_count = 0
        self._step_count = 0
        self._always_record = True
        self._warned_objects: set[int] = set()
        self._placement_checked = False

        self._d_charge = None
        self._d_counts = None
        self._d_total_charge = None
        self._d_total_counts = None

    def _check_placement_warnings(self, dt: float | None) -> None:
        """Emit a non-repeating :class:`UserWarning` if ``z_plane`` is at an injector or within ``c*dt`` of a collector."""
        ps = self._solver_ref
        if ps is None:
            self._placement_checked = True
            return
        for inj in getattr(ps, "injectors", ()):
            obj_id = id(inj)
            if obj_id in self._warned_objects:
                continue
            if hasattr(inj, "z0") and self.z_plane == float(inj.z0):
                self._warned_objects.add(obj_id)
                warnings.warn(
                    f"PlaneCurrentProbe z_plane={self.z_plane:g} m equals injector "
                    f"z0={float(inj.z0):g} m; reconstructed z_old may differ from z0 "
                    "by a few ulp(z).",
                    UserWarning,
                    stacklevel=3,
                )

        if dt is None or dt <= 0.0:
            return
        bounds_mgr = getattr(ps, "boundaries", None)
        if bounds_mgr is not None:
            c_dt = c_light * float(dt)
            for col in getattr(bounds_mgr, "collectors", ()):
                obj_id = id(col)
                if obj_id in self._warned_objects:
                    continue
                near = False
                if getattr(col, "plane", None) is not None:
                    axis = col.plane[0]
                    pos = float(col.plane[1])
                    is_z = (axis == 2) or (str(axis).strip().lower() == "z")
                    if is_z and abs(self.z_plane - pos) <= c_dt:
                        near = True
                elif getattr(col, "bounds", None) is not None:
                    z_lo = float(col.bounds[4])
                    z_hi = float(col.bounds[5])
                    if (z_lo - c_dt) <= self.z_plane <= (z_hi + c_dt):
                        near = True
                if near:
                    self._warned_objects.add(obj_id)
                    warnings.warn(
                        f"PlaneCurrentProbe z_plane={self.z_plane:g} m lies within one "
                        f"c*dt ({c_dt:g} m) of collector {col.name!r}; crossings absorbed "
                        "by a collector in the same step are not counted.",
                        UserWarning,
                        stacklevel=3,
                    )
        self._placement_checked = True

    def _bind(self, source) -> None:
        if not isinstance(source, Species):
            self._solver_ref = source
        on_gpu = bool(getattr(source, "on_gpu", False))
        self._on_gpu = on_gpu
        if on_gpu and self._d_charge is None:
            import cupy as cp

            _get_diag_kernel("plane_current_probe_kernel")
            self._d_charge = cp.zeros(self.capacity, dtype=np.float64)
            self._d_counts = cp.zeros(self.capacity, dtype=np.int64)
            self._d_total_charge = cp.zeros(1, dtype=np.float64)
            self._d_total_counts = cp.zeros(1, dtype=np.int64)
        self._placement_checked = False
        em_solver = getattr(self._solver_ref, "solver", None)
        dt_bind = getattr(em_solver, "dt", None)
        self._check_placement_warnings(
            float(dt_bind) if dt_bind is not None else None
        )
        self._bound = True

    def _compute_z_old_cpu(
        self, sp: Species, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(z_new, dz)`` on CPU using the same formula as the pusher; agrees to a few ``ulp(z)``."""
        n_p = sp.n_particles
        ux = sp._ux[:n_p]
        uy = sp._uy[:n_p]
        uz = sp._uz[:n_p]
        inv_c2 = 1.0 / (c_light * c_light)
        z_new = sp._z[:n_p]
        if n_p > 65536:
            dz = np.empty(n_p, dtype=np.float64)
            for s in range(0, n_p, 65536):
                e = min(s + 65536, n_p)
                ux_c = ux[s:e]
                uy_c = uy[s:e]
                uz_c = uz[s:e]
                fac = dt / np.sqrt(
                    1.0 + (ux_c * ux_c + uy_c * uy_c + uz_c * uz_c) * inv_c2
                )
                dz[s:e] = fac * uz_c
        else:
            inv_gamma = 1.0 / np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2)
            dz = (dt * inv_gamma) * uz
        return z_new, dz

    def _compute_z_old_vel_gpu(self, sp: Species, dt: float):
        """Return ``(ux, uy, uz)`` device arrays used by :func:`plane_current_probe_kernel`."""
        return sp._ux, sp._uy, sp._uz

    def _record(self, source, t: float, dt: float, step: int) -> None:
        if not self._bound or bool(getattr(source, "on_gpu", False)) != self._on_gpu:
            self._bind(source)
        if not self._placement_checked and dt > 0.0:
            self._check_placement_warnings(dt)
        sp_list = self._select_species(source)

        periodic_z = False
        z_min = 0.0
        z_max = 1.0
        ps = self._solver_ref
        if ps is not None and hasattr(ps, "boundaries") and hasattr(ps, "grid_map"):
            periodic_z = bool(ps.boundaries.periodic[2])
            z_min = float(ps.grid_map.zmin)
            z_max = float(ps.grid_map.zmax)

        sub = (self._step_count % self.every) if self._always_record else 0
        self._step_count += 1
        if sub == 0:
            slot = self._sample_count % self.capacity
            self._sample_count += 1
        else:
            slot = (self._sample_count - 1) % self.capacity

        self._times[slot] = t
        self._dts[slot] = (sub + 1) * dt
        self._steps[slot] = step

        if self._on_gpu:
            if sub == 0:
                z_kern = _get_diag_kernel("zero_probe_slot_kernel")
                z_kern(
                    (1,),
                    (1,),
                    (self._d_charge, self._d_counts, np.int32(slot)),
                )

            kern = _get_diag_kernel("plane_current_probe_kernel")
            inv_c2 = np.float64(1.0 / (c_light * c_light))
            for sp in sp_list:
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                ux_d, uy_d, uz_d = self._compute_z_old_vel_gpu(sp, dt)
                grid, block = _grid_1d(n_l)
                kern(
                    grid,
                    block,
                    (
                        sp._z,
                        ux_d,
                        uy_d,
                        uz_d,
                        sp._w,
                        np.float64(sp.charge),
                        np.float64(dt),
                        np.float64(self.z_plane),
                        inv_c2,
                        np.int32(1 if periodic_z else 0),
                        np.float64(z_min),
                        np.float64(z_max),
                        np.int32(n_l),
                        d_n_ptr,
                        self._d_charge,
                        self._d_counts,
                        self._d_total_charge,
                        self._d_total_counts,
                        np.int32(slot),
                    ),
                )
            return

        if sub == 0:
            self._charge_cpu[slot] = 0.0
            self._counts_cpu[slot] = 0
        zp = self.z_plane
        Lz = z_max - z_min

        for sp in sp_list:
            n_p = sp.n_particles
            if n_p == 0:
                continue
            z_new, dz = self._compute_z_old_cpu(sp, dt)
            z_raw = z_new - dz
            w_arr = sp._w[:n_p]

            if not periodic_z:
                pos_mask = (z_raw < zp) & (zp <= z_new)
                neg_mask = (z_raw >= zp) & (zp > z_new)
            else:
                pos_no_wrap = (
                    (dz > 0.0) & (z_raw >= z_min) & (z_raw < zp) & (zp <= z_new)
                )
                pos_wrap = (
                    (dz > 0.0)
                    & (z_raw < z_min)
                    & (((z_raw + Lz) < zp) | (zp <= z_new))
                )
                pos_mask = pos_no_wrap | pos_wrap

                neg_no_wrap = (
                    (dz < 0.0) & (z_raw < z_max) & (z_raw >= zp) & (zp > z_new)
                )
                neg_wrap = (
                    (dz < 0.0)
                    & (z_raw >= z_max)
                    & (((z_raw - Lz) >= zp) | (zp > z_new))
                )
                neg_mask = neg_no_wrap | neg_wrap

            n_pos = int(np.sum(pos_mask))
            n_neg = int(np.sum(neg_mask))
            q_step = 0.0
            if n_pos > 0:
                q_step += float(sp.charge * np.sum(w_arr[pos_mask]))
            if n_neg > 0:
                q_step -= float(sp.charge * np.sum(w_arr[neg_mask]))
            self._charge_cpu[slot] += q_step
            self._counts_cpu[slot] += n_pos - n_neg
            self._total_charge_cpu += q_step
            self._total_counts_cpu += n_pos - n_neg

    def result(self) -> DiagnosticResult:
        """Return the host time series of crossing charge and current (collective across MPI ranks)."""
        n_stored = min(self._sample_count, self.capacity)
        if self._on_gpu and self._d_charge is not None:
            q_raw = self._d_charge.get().copy()
            c_raw = self._d_counts.get().copy()
            q_tot_arr = self._d_total_charge.get().copy()
            c_tot_arr = self._d_total_counts.get().copy()
        else:
            q_raw = self._charge_cpu.copy()
            c_raw = self._counts_cpu.copy()
            q_tot_arr = np.array([self._total_charge_cpu], dtype=np.float64)
            c_tot_arr = np.array([self._total_counts_cpu], dtype=np.int64)

        if self._sample_count <= self.capacity:
            dQ = q_raw[:n_stored].copy()
            counts = c_raw[:n_stored].copy()
            times = self._times[:n_stored].copy()
            dts = self._dts[:n_stored].copy()
            steps = self._steps[:n_stored].copy()
        else:
            shift = -(self._sample_count % self.capacity)
            dQ = np.roll(q_raw, shift)
            counts = np.roll(c_raw, shift)
            times = np.roll(self._times, shift)
            dts = np.roll(self._dts, shift)
            steps = np.roll(self._steps, shift)

        comm = self._get_comm()
        if comm is not None:
            from mpi4py import MPI

            comm.Allreduce(MPI.IN_PLACE, dQ, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, q_tot_arr, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, c_tot_arr, op=MPI.SUM)

        with np.errstate(divide="ignore", invalid="ignore"):
            current = np.where(dts > 0.0, dQ / dts, 0.0)
        q_tot = float(q_tot_arr[0])
        c_tot = int(c_tot_arr[0])
        if self._sample_count <= self.capacity:
            q_cum = np.cumsum(dQ)
        else:
            q_prior = q_tot - float(np.sum(dQ))
            q_cum = q_prior + np.cumsum(dQ)

        return DiagnosticResult(
            t=times,
            times=times,
            dt=dts,
            steps=steps,
            dQ=dQ,
            charge_per_step=dQ,
            I=current,
            current=current,
            Q_total=q_tot,
            integrated_charge=q_tot,
            Q_cumulative=q_cum,
            cumulative_charge=q_cum,
            counts=counts,
            total_counts=c_tot,
            z_plane=self.z_plane,
        )


class EmittanceProbe(_BaseDiagnostic):
    """Transverse RMS trace-space and normalized emittance probe in a ``z`` window.

    Parameters
    ----------
    z_window : tuple of (float, float) or None, optional
        Longitudinal window ``(z_min, z_max)`` [m] selecting active particles
        with ``z_min <= z <= z_max`` (or ``None`` for the entire domain).
    every : int, optional
        Sampling interval in particle steps. Default is ``1``.
    species : Species, str, int, or None, optional
        Restrict probe to a specific species (or ``None`` for all species).
    capacity : int, optional
        Preallocated number of time samples on the device. Default is ``16384``.

    Notes
    -----
    * **Definitions:** With trace-space slopes ``x' = u_x / u_z`` and
      ``y' = u_y / u_z`` [rad], weighted central second moments
      ``sigma_x^2 = <(x - <x>)^2>_w``, ``sigma_{x'}^2 = <(x' - <x'>)^2>_w``, and
      ``sigma_{xx'} = <(x - <x>)(x' - <x'>)>_w`` yield the geometric RMS
      trace-space emittance and its trace-space normalized counterpart:

      .. math::

          \\varepsilon_x = \\sqrt{\\sigma_x^2 \\sigma_{x'}^2 - \\sigma_{xx'}^2},
          \\qquad
          \\varepsilon_{n,x} = \\langle \\beta \\gamma \\rangle_w \\, \\varepsilon_x,

      where ``<beta * gamma>_w = <|u| / c>_w``. In addition, the probe reports
      the true momentum-space normalized RMS emittance:

      .. math::

          \\varepsilon_{n,u,x} = \\frac{1}{c}
          \\sqrt{\\sigma_x^2 \\sigma_{u_x}^2 - \\sigma_{x u_x}^2},

      with ``sigma_{u_x}^2 = <(u_x - <u_x>)^2>_w`` and
      ``sigma_{x u_x} = <(x - <x>)(u_x - <u_x>)>_w``. Note that
      ``<|u|/c>_w * varepsilon_x`` equals ``varepsilon_{n,u,x}`` only for a
      monoenergetic beam in the paraxial limit (zero longitudinal momentum
      spread).
    * **Mixed leapfrog time levels:** Because diagnostics execute at the end of
      the particle step, transverse positions ``(x, y)`` are evaluated at
      integer time ``t_{n+1}`` whereas proper velocities ``(u_x, u_y, u_z)`` and
      slopes ``(x', y')`` are at half-step time ``t_{n+1/2}``.
    * **Two-pass central moments on the device (round-off justification):**
      Textbook one-pass ``<x^2> - <x>^2`` suffers catastrophic cancellation
      proportional to ``kappa^2 = (1 + <x>^2 / sigma_x^2)`` when the beam has a
      non-zero centroid offset ``|<x>| >> sigma_x``, while sequential Welford
      updates serialize across threads on a GPU. This monitor uses a two-pass
      device algorithm with shared-memory block reductions followed by one
      atomic update per thread block: Pass 1 accumulates ``W = sum w_p`` and
      weighted first moments; Pass 2 reads the means directly on the device
      without host synchronization and accumulates central second moments. By
      Chan, Golub & LeVeque (1983), two-pass central summation bounds the
      relative round-off error at ``lesssim N * u + N^2 * u^2 * kappa^2``.
    * **MPI semantics:** Each rank records its local first- and second-pass
      sufficient statistics on the device. In :meth:`result`, ranks are
      combined via the parallel variance/covariance update formula of
      Chan, Golub & LeVeque (1983), Eq. (2.1).
    """

    def __init__(
        self,
        z_window: tuple[float, float] | None = None,
        *,
        every: int = 1,
        species: Species | str | int | None = None,
        capacity: int = 16384,
    ) -> None:
        super().__init__(every=every, species=species)
        if z_window is None:
            self.z_window = (-np.inf, np.inf)
        else:
            if len(z_window) != 2:
                raise ValueError(
                    f"z_window must be (z_min, z_max), got {z_window!r}."
                )
            z_lo, z_hi = float(z_window[0]), float(z_window[1])
            if z_hi < z_lo:
                raise ValueError(
                    f"z_window must have z_max >= z_min, got {z_window!r}."
                )
            self.z_window = (z_lo, z_hi)

        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.capacity = int(capacity)

        # 20 statistics per sample:
        # [0]: W, [1]: Sx, [2]: Sxp, [3]: Sy, [4]: Syp, [5]: S_bg, [6]: S_bgz,
        # [7]: Mxx, [8]: Mxpxp, [9]: Mxxp, [10]: Myy, [11]: Mypyp, [12]: Myyp, [13]: N_cnt,
        # [14]: Sux, [15]: Suy, [16]: Muxux, [17]: Mxux, [18]: Muyuy, [19]: Myuy
        self._stats_cpu = np.zeros((self.capacity, _EMIT_N_STATS), dtype=np.float64)
        self._times = np.zeros(self.capacity, dtype=np.float64)
        self._steps = np.zeros(self.capacity, dtype=np.int64)
        self._sample_count = 0
        self._drop_cov_term = False

        self._d_stats = None

    def _bind(self, source) -> None:
        if not isinstance(source, Species):
            self._solver_ref = source
        on_gpu = bool(getattr(source, "on_gpu", False))
        self._on_gpu = on_gpu
        if on_gpu and self._d_stats is None:
            import cupy as cp

            _get_diag_kernel("emittance_pass1_kernel")
            self._d_stats = cp.zeros(
                self.capacity * _EMIT_N_STATS, dtype=np.float64
            )
        self._bound = True

    def _record(self, source, t: float, dt: float, step: int) -> None:
        if not self._bound or bool(getattr(source, "on_gpu", False)) != self._on_gpu:
            self._bind(source)
        sp_list = self._select_species(source)
        z_lo, z_hi = self.z_window

        slot = self._sample_count % self.capacity
        self._times[slot] = t
        self._steps[slot] = step
        self._sample_count += 1

        if self._on_gpu:
            base_offset = slot * _EMIT_N_STATS
            z_kern = _get_diag_kernel("zero_f64_kernel")
            z_kern(
                (1,),
                (32,),
                (self._d_stats, np.int32(base_offset), np.int32(_EMIT_N_STATS)),
            )

            k_p1 = _get_diag_kernel("emittance_pass1_kernel")
            k_p2 = _get_diag_kernel("emittance_pass2_kernel")
            inv_c = np.float64(1.0 / c_light)

            for sp in sp_list:
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                grid, block = _grid_1d(n_l)
                k_p1(
                    grid,
                    block,
                    (
                        sp._x,
                        sp._y,
                        sp._z,
                        sp._ux,
                        sp._uy,
                        sp._uz,
                        sp._w,
                        np.float64(z_lo),
                        np.float64(z_hi),
                        inv_c,
                        np.int32(n_l),
                        d_n_ptr,
                        self._d_stats,
                        np.int32(base_offset),
                    ),
                )

            for sp in sp_list:
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                grid, block = _grid_1d(n_l)
                k_p2(
                    grid,
                    block,
                    (
                        sp._x,
                        sp._y,
                        sp._z,
                        sp._ux,
                        sp._uy,
                        sp._uz,
                        sp._w,
                        np.float64(z_lo),
                        np.float64(z_hi),
                        np.int32(n_l),
                        d_n_ptr,
                        self._d_stats,
                        np.int32(base_offset),
                    ),
                )
            return

        row = self._stats_cpu[slot]
        row.fill(0.0)

        # Pass 1: weighted sums for centroids
        for sp in sp_list:
            n_p = sp.n_particles
            if n_p == 0:
                continue
            z = sp._z[:n_p]
            uz = sp._uz[:n_p]
            mask = (z >= z_lo) & (z <= z_hi) & (uz != 0.0)
            if not bool(np.any(mask)):
                continue
            x_m = sp._x[:n_p][mask]
            y_m = sp._y[:n_p][mask]
            ux_m = sp._ux[:n_p][mask]
            uy_m = sp._uy[:n_p][mask]
            uz_m = uz[mask]
            w_m = sp._w[:n_p][mask]

            xp_m = ux_m / uz_m
            yp_m = uy_m / uz_m
            u_mag = np.sqrt(ux_m * ux_m + uy_m * uy_m + uz_m * uz_m)

            row[0] += float(np.sum(w_m))
            row[1] += float(np.sum(w_m * x_m))
            row[2] += float(np.sum(w_m * xp_m))
            row[3] += float(np.sum(w_m * y_m))
            row[4] += float(np.sum(w_m * yp_m))
            row[5] += float(np.sum(w_m * (u_mag / c_light)))
            row[6] += float(np.sum(w_m * (uz_m / c_light)))
            row[13] += float(x_m.size)
            row[14] += float(np.sum(w_m * ux_m))
            row[15] += float(np.sum(w_m * uy_m))

        W = row[0]
        if W <= 0.0:
            return
        x_bar = row[1] / W
        xp_bar = row[2] / W
        y_bar = row[3] / W
        yp_bar = row[4] / W
        ux_bar = row[14] / W
        uy_bar = row[15] / W

        # Pass 2: weighted central second moments
        for sp in sp_list:
            n_p = sp.n_particles
            if n_p == 0:
                continue
            z = sp._z[:n_p]
            uz = sp._uz[:n_p]
            mask = (z >= z_lo) & (z <= z_hi) & (uz != 0.0)
            if not bool(np.any(mask)):
                continue
            ux_m = sp._ux[:n_p][mask]
            uy_m = sp._uy[:n_p][mask]
            uz_m = uz[mask]
            dx = sp._x[:n_p][mask] - x_bar
            dxp = (ux_m / uz_m) - xp_bar
            dy = sp._y[:n_p][mask] - y_bar
            dyp = (uy_m / uz_m) - yp_bar
            dux = ux_m - ux_bar
            duy = uy_m - uy_bar
            w_m = sp._w[:n_p][mask]

            row[7] += float(np.sum(w_m * dx * dx))
            row[8] += float(np.sum(w_m * dxp * dxp))
            row[9] += float(np.sum(w_m * dx * dxp))
            row[10] += float(np.sum(w_m * dy * dy))
            row[11] += float(np.sum(w_m * dyp * dyp))
            row[12] += float(np.sum(w_m * dy * dyp))
            row[16] += float(np.sum(w_m * dux * dux))
            row[17] += float(np.sum(w_m * dx * dux))
            row[18] += float(np.sum(w_m * duy * duy))
            row[19] += float(np.sum(w_m * dy * duy))

    @staticmethod
    def _combine_parallel_stats(stats_a: np.ndarray, stats_b: np.ndarray) -> np.ndarray:
        """Combine two ``(N_samples, 20)`` central-moment arrays via Chan-Golub-LeVeque (1983)."""
        out = stats_a.copy()
        wa = stats_a[:, 0]
        wb = stats_b[:, 0]
        w_tot = wa + wb

        has_b = wb > 0.0
        only_b = (wa <= 0.0) & has_b
        both = (wa > 0.0) & has_b

        if bool(np.any(only_b)):
            out[only_b] = stats_b[only_b]

        if bool(np.any(both)):
            wa_m = wa[both]
            wb_m = wb[both]
            wt_m = w_tot[both]
            factor = (wa_m * wb_m) / wt_m

            xa = stats_a[both, 1] / wa_m
            xpa = stats_a[both, 2] / wa_m
            ya = stats_a[both, 3] / wa_m
            ypa = stats_a[both, 4] / wa_m
            uxa = stats_a[both, 14] / wa_m
            uya = stats_a[both, 15] / wa_m

            xb = stats_b[both, 1] / wb_m
            xpb = stats_b[both, 2] / wb_m
            yb = stats_b[both, 3] / wb_m
            ypb = stats_b[both, 4] / wb_m
            uxb = stats_b[both, 14] / wb_m
            uyb = stats_b[both, 15] / wb_m

            dx = xb - xa
            dxp = xpb - xpa
            dy = yb - ya
            dyp = ypb - ypa
            dux = uxb - uxa
            duy = uyb - uya

            out[both, 0] = wt_m
            out[both, 1:7] = stats_a[both, 1:7] + stats_b[both, 1:7]
            out[both, 7] = stats_a[both, 7] + stats_b[both, 7] + factor * dx * dx
            out[both, 8] = stats_a[both, 8] + stats_b[both, 8] + factor * dxp * dxp
            out[both, 9] = stats_a[both, 9] + stats_b[both, 9] + factor * dx * dxp
            out[both, 10] = stats_a[both, 10] + stats_b[both, 10] + factor * dy * dy
            out[both, 11] = stats_a[both, 11] + stats_b[both, 11] + factor * dyp * dyp
            out[both, 12] = stats_a[both, 12] + stats_b[both, 12] + factor * dy * dyp
            out[both, 13] = stats_a[both, 13] + stats_b[both, 13]
            out[both, 14:16] = stats_a[both, 14:16] + stats_b[both, 14:16]
            out[both, 16] = stats_a[both, 16] + stats_b[both, 16] + factor * dux * dux
            out[both, 17] = stats_a[both, 17] + stats_b[both, 17] + factor * dx * dux
            out[both, 18] = stats_a[both, 18] + stats_b[both, 18] + factor * duy * duy
            out[both, 19] = stats_a[both, 19] + stats_b[both, 19] + factor * dy * duy
        return out

    def result(self) -> DiagnosticResult:
        """Return host RMS geometric and normalized emittance series (collective across MPI ranks)."""
        n_stored = min(self._sample_count, self.capacity)
        if self._on_gpu and self._d_stats is not None:
            s_raw = (
                self._d_stats.get()
                .reshape((self.capacity, _EMIT_N_STATS))
                .copy()
            )
        else:
            s_raw = self._stats_cpu.copy()

        if self._sample_count <= self.capacity:
            stats = s_raw[:n_stored].copy()
            times = self._times[:n_stored].copy()
            steps = self._steps[:n_stored].copy()
        else:
            shift = -(self._sample_count % self.capacity)
            stats = np.roll(s_raw, shift, axis=0)
            times = np.roll(self._times, shift)
            steps = np.roll(self._steps, shift)

        comm = self._get_comm()
        if comm is not None:
            all_stats = comm.allgather(stats)
            combined = all_stats[0].copy()
            for r_stats in all_stats[1:]:
                combined = self._combine_parallel_stats(combined, r_stats)
            stats = combined

        W = stats[:, 0]
        valid = W > 0.0
        inv_W = np.where(valid, 1.0 / np.maximum(W, 1e-300), 0.0)

        mean_x = stats[:, 1] * inv_W
        mean_xp = stats[:, 2] * inv_W
        mean_y = stats[:, 3] * inv_W
        mean_yp = stats[:, 4] * inv_W
        mean_bg = stats[:, 5] * inv_W
        mean_bg_z = stats[:, 6] * inv_W
        mean_ux = stats[:, 14] * inv_W
        mean_uy = stats[:, 15] * inv_W

        var_x = stats[:, 7] * inv_W
        var_xp = stats[:, 8] * inv_W
        cov_xxp = stats[:, 9] * inv_W
        var_y = stats[:, 10] * inv_W
        var_yp = stats[:, 11] * inv_W
        cov_yyp = stats[:, 12] * inv_W
        counts = np.rint(stats[:, 13]).astype(np.int64)
        var_ux = stats[:, 16] * inv_W
        cov_xux = stats[:, 17] * inv_W
        var_uy = stats[:, 18] * inv_W
        cov_yuy = stats[:, 19] * inv_W

        if self._drop_cov_term:
            det_x = var_x * var_xp
            det_y = var_y * var_yp
            det_ux = var_x * var_ux
            det_uy = var_y * var_uy
        else:
            det_x = var_x * var_xp - cov_xxp * cov_xxp
            det_y = var_y * var_yp - cov_yyp * cov_yyp
            det_ux = var_x * var_ux - cov_xux * cov_xux
            det_uy = var_y * var_uy - cov_yuy * cov_yuy

        emit_x = np.sqrt(np.maximum(det_x, 0.0))
        emit_y = np.sqrt(np.maximum(det_y, 0.0))
        norm_emit_x = mean_bg * emit_x
        norm_emit_y = mean_bg * emit_y
        norm_emit_u_x = np.sqrt(np.maximum(det_ux, 0.0)) / c_light
        norm_emit_u_y = np.sqrt(np.maximum(det_uy, 0.0)) / c_light

        return DiagnosticResult(
            t=times,
            times=times,
            steps=steps,
            emittance_x=emit_x,
            emit_x=emit_x,
            emittance_y=emit_y,
            emit_y=emit_y,
            normalized_emittance_x=norm_emit_x,
            norm_emit_x=norm_emit_x,
            normalized_emittance_y=norm_emit_y,
            norm_emit_y=norm_emit_y,
            momentum_emittance_x=norm_emit_u_x,
            momentum_emittance_y=norm_emit_u_y,
            normalized_emittance_u_x=norm_emit_u_x,
            normalized_emittance_u_y=norm_emit_u_y,
            norm_emit_u_x=norm_emit_u_x,
            norm_emit_u_y=norm_emit_u_y,
            epsilon_n_u_x=norm_emit_u_x,
            epsilon_n_u_y=norm_emit_u_y,
            beta_gamma=mean_bg,
            beta_z_gamma=mean_bg_z,
            mean_x=mean_x,
            mean_xp=mean_xp,
            mean_y=mean_y,
            mean_yp=mean_yp,
            mean_ux=mean_ux,
            mean_uy=mean_uy,
            var_x=var_x,
            var_xp=var_xp,
            cov_xxp=cov_xxp,
            var_y=var_y,
            var_yp=var_yp,
            cov_yyp=cov_yyp,
            var_ux=var_ux,
            cov_xux=cov_xux,
            var_uy=var_uy,
            cov_yuy=cov_yuy,
            weight_sum=W,
            counts=counts,
            z_window=self.z_window,
        )


class TrackRecorder(_BaseDiagnostic):
    """Device ring-buffer trajectory recorder for selected macroparticle ``ids``.

    Parameters
    ----------
    ids : sequence of int or 1D ndarray
        Macroparticle integer ``id`` values to track.
    every : int, optional
        Sampling interval in particle steps. Default is ``1``.
    species : Species, str, int, or None, optional
        Target species (or ``None`` when the solver holds exactly one species).
        :class:`TrackRecorder` requires a single resolved species and raises
        :class:`ValueError` if multiple species match.
    ring_size : int, optional
        Preallocated number of snapshot frames in the device ring buffer.
        Default is ``4096``.

    Notes
    -----
    * **Time levels:** At each recorded step, stored positions ``(x, y, z)``
      are at ``t_{n+1}`` [m] and stored proper velocities ``(ux, uy, uz)`` are
      at ``t_{n+1/2}`` [m/s].
    * **Single-species and ID uniqueness:** :class:`TrackRecorder` requires a
      single species because each :class:`Species` instance maintains its own
      ``_next_id`` counter starting at ``0``. In :meth:`result`, if any
      ``(id, sample)`` entry matched more than one particle (``valid_sum > 1``)
      on either CPU or GPU (locally or across MPI ranks), a :class:`ValueError`
      is raised.
    * **MPI ID uniqueness:** In :mod:`wakis.particles.species`, each rank
      initializes ``_next_id = 0`` independently and increments it locally in
      :meth:`Species.add_particles`. Consequently, particle ``id`` values are
      globally unique across MPI ranks only if the user guarantees it (for
      example, by seeding tracked particles on a single rank or assigning
      disjoint ``_next_id`` offsets per rank).
    * **Absorbed and migrated particles:** Each frame slot is initialized to
      ``NaN`` on the device before scanning active particles by ``id``. Any
      tracked particle that has been absorbed by a boundary/collector remains
      ``NaN``. Under MPI domain decomposition, each active particle resides on
      at most one rank; :meth:`result` combines the per-rank buffers
      collectively so particles migrating across MPI z-slabs produce seamless
      trajectories.
    * **Computational cost:** Matching ``K = len(ids)`` tracked IDs against
      ``N`` active particles costs ``O(K * N)`` comparisons per recorded sample,
      and the CPU path uses a Python loop over ``ids``.
    """

    def __init__(
        self,
        ids: Sequence[int] | np.ndarray,
        every: int = 1,
        *,
        species: Species | str | int | None = None,
        ring_size: int = 4096,
    ) -> None:
        super().__init__(every=every, species=species)
        id_arr = np.asarray(ids, dtype=np.int64).ravel()
        if id_arr.size == 0:
            raise ValueError("ids must contain at least one particle id.")
        if int(ring_size) < 1:
            raise ValueError(f"ring_size must be >= 1, got {ring_size}.")
        self.ids = id_arr.copy()
        self.ring_size = int(ring_size)

        k_ids = self.ids.size
        self._tracks_cpu = np.full(
            (self.ring_size, k_ids, 6), np.nan, dtype=np.float64
        )
        self._match_counts_cpu = np.zeros(
            (self.ring_size, k_ids), dtype=np.int32
        )
        self._times = np.zeros(self.ring_size, dtype=np.float64)
        self._steps = np.zeros(self.ring_size, dtype=np.int64)
        self._frame_count = 0

        self._d_target_ids = None
        self._d_tracks = None
        self._d_match_counts = None

    def _select_species(self, source) -> list[Species]:
        sp_list = super()._select_species(source)
        if len(sp_list) != 1:
            raise ValueError(
                f"TrackRecorder requires a single species, got {len(sp_list)} species."
            )
        return sp_list

    def _bind(self, source) -> None:
        if not isinstance(source, Species):
            self._solver_ref = source
            if len(getattr(source, "species", ())) > 1 and self.species is None:
                raise ValueError(
                    f"TrackRecorder requires a single species, got {len(source.species)} species."
                )
        on_gpu = bool(getattr(source, "on_gpu", False))
        self._on_gpu = on_gpu
        if on_gpu and self._d_tracks is None:
            import cupy as cp

            _get_diag_kernel("track_recorder_kernel")
            k_ids = self.ids.size
            self._d_target_ids = cp.asarray(self.ids, dtype=np.int64)
            self._d_tracks = cp.full(
                self.ring_size * k_ids * 6, np.nan, dtype=np.float64
            )
            self._d_match_counts = cp.zeros(
                self.ring_size * k_ids, dtype=np.int32
            )
        self._bound = True

    def _record(self, source, t: float, dt: float, step: int) -> None:
        if not self._bound or bool(getattr(source, "on_gpu", False)) != self._on_gpu:
            self._bind(source)
        sp_list = self._select_species(source)
        k_ids = self.ids.size

        slot = self._frame_count % self.ring_size
        self._times[slot] = t
        self._steps[slot] = step
        self._frame_count += 1

        if self._on_gpu:
            frame_offset = slot * k_ids * 6
            match_offset = slot * k_ids
            n_frame_elems = k_ids * 6
            rst_kern = _get_diag_kernel("reset_track_frame_kernel")
            g_rst, b_rst = _grid_1d(n_frame_elems)
            rst_kern(
                g_rst,
                b_rst,
                (
                    self._d_tracks,
                    np.int32(frame_offset),
                    np.int32(n_frame_elems),
                    self._d_match_counts,
                    np.int32(match_offset),
                    np.int32(k_ids),
                ),
            )

            kern = _get_diag_kernel("track_recorder_kernel")
            for sp in sp_list:
                if not sp.track_ids:
                    continue
                n_l = sp._n_launch_bound
                if n_l == 0:
                    continue
                d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else 0
                grid, block = _grid_1d(n_l)
                kern(
                    grid,
                    block,
                    (
                        sp._x,
                        sp._y,
                        sp._z,
                        sp._ux,
                        sp._uy,
                        sp._uz,
                        sp._id,
                        np.int32(n_l),
                        d_n_ptr,
                        self._d_target_ids,
                        np.int32(k_ids),
                        self._d_tracks,
                        np.int32(frame_offset),
                        self._d_match_counts,
                        np.int32(match_offset),
                    ),
                )
            return

        frame = self._tracks_cpu[slot]
        frame.fill(np.nan)
        m_row = self._match_counts_cpu[slot]
        m_row.fill(0)
        for sp in sp_list:
            if not sp.track_ids:
                continue
            n_p = sp.n_particles
            if n_p == 0:
                continue
            sp_ids = sp._id[:n_p]
            for k, target_id in enumerate(self.ids):
                matches = np.flatnonzero(sp_ids == target_id)
                m_row[k] += int(matches.size)
                if matches.size > 0:
                    m = int(matches[0])
                    frame[k, 0] = sp._x[m]
                    frame[k, 1] = sp._y[m]
                    frame[k, 2] = sp._z[m]
                    frame[k, 3] = sp._ux[m]
                    frame[k, 4] = sp._uy[m]
                    frame[k, 5] = sp._uz[m]

    def result(self) -> DiagnosticResult:
        """Return host particle trajectories ``(n_frames, n_ids, 6)`` (collective across MPI ranks)."""
        k_ids = self.ids.size
        n_stored = min(self._frame_count, self.ring_size)
        if self._on_gpu and self._d_tracks is not None:
            t_raw = self._d_tracks.get().reshape(
                (self.ring_size, k_ids, 6)
            ).copy()
            m_raw = self._d_match_counts.get().reshape(
                (self.ring_size, k_ids)
            ).copy()
        else:
            t_raw = self._tracks_cpu.copy()
            m_raw = self._match_counts_cpu.copy()

        if self._frame_count <= self.ring_size:
            tracks = t_raw[:n_stored].copy()
            valid_local = m_raw[:n_stored].copy()
            times = self._times[:n_stored].copy()
            steps = self._steps[:n_stored].copy()
        else:
            shift = -(self._frame_count % self.ring_size)
            tracks = np.roll(t_raw, shift, axis=0)
            valid_local = np.roll(m_raw, shift, axis=0)
            times = np.roll(self._times, shift)
            steps = np.roll(self._steps, shift)

        comm = self._get_comm()
        if comm is not None and n_stored > 0:
            from mpi4py import MPI

            clean_local = np.where(np.isnan(tracks), 0.0, tracks)
            valid_sum = np.zeros_like(valid_local)
            clean_sum = np.zeros_like(clean_local)
            comm.Allreduce(valid_local, valid_sum, op=MPI.SUM)
            comm.Allreduce(clean_local, clean_sum, op=MPI.SUM)
            if bool(np.any(valid_sum > 1)):
                dup_frames, dup_k = np.nonzero(valid_sum > 1)
                dup_id = int(self.ids[int(dup_k[0])])
                dup_step = int(steps[int(dup_frames[0])])
                dup_cnt = int(valid_sum[int(dup_frames[0]), int(dup_k[0])])
                raise ValueError(
                    f"TrackRecorder detected duplicate particle id={dup_id} "
                    f"at step {dup_step} across MPI ranks (valid_sum={dup_cnt} > 1)."
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                tracks = np.where(
                    valid_sum[:, :, None] > 0,
                    clean_sum / np.maximum(valid_sum[:, :, None], 1),
                    np.nan,
                )
        elif n_stored > 0:
            valid_sum = valid_local
            if bool(np.any(valid_sum > 1)):
                dup_frames, dup_k = np.nonzero(valid_sum > 1)
                dup_id = int(self.ids[int(dup_k[0])])
                dup_step = int(steps[int(dup_frames[0])])
                dup_cnt = int(valid_sum[int(dup_frames[0]), int(dup_k[0])])
                raise ValueError(
                    f"TrackRecorder detected duplicate particle id={dup_id} "
                    f"at step {dup_step} (valid_sum={dup_cnt} > 1)."
                )

        return DiagnosticResult(
            t=times,
            times=times,
            steps=steps,
            ids=self.ids.copy(),
            tracks=tracks,
            x=tracks[:, :, 0],
            y=tracks[:, :, 1],
            z=tracks[:, :, 2],
            ux=tracks[:, :, 3],
            uy=tracks[:, :, 4],
            uz=tracks[:, :, 5],
        )


__all__ = [
    "DiagnosticResult",
    "EmittanceProbe",
    "EnergySpectrumMonitor",
    "PhaseSpaceMonitor",
    "PlaneCurrentProbe",
    "TrackRecorder",
]
