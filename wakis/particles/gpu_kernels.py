# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""CUDA RawKernels (via CuPy) for `wakis.particles`.

Provides asynchronous GPU kernels for the coupled particle-in-cell fast path
(no device-to-host synchronization on the standard Esirkepov path, and zero
memory-pool sub-allocations on non-sorting steps; periodic cell sorting via
``cupy.argsort`` reuses cached blocks in the CuPy memory pool without growing
``pool.total_bytes()``):
- Fused field gathering and relativistic particle pushing (Boris, Vay,
  Higuera-Cary; Galerkin and momentum-conserving gather; B-spline orders
  ``p = 1, 2, 3``; uniform and graded tensor-product meshes).
- 3D Esirkepov current deposition on the minimal ``(p + 2)``-node window
  using ``float64`` ``atomicAdd`` into the FIT dual-face flux buffers, with an
  optional device CFL error counter ``d_cfl_error`` recording any single-step
  displacement violation ``|b1 - b0| > 1`` for host-side diagnostics.
- Nodal charge deposition using ``float64`` ``atomicAdd``.
- Boundary trajectory truncation, absorbed-charge deposition, and periodic
  coordinate wrapping.
- Three-pass shared-memory prefix-sum stream compaction for absorbed particles
  without device-to-host synchronization.
- Linear cell-index evaluation and in-place SoA permutation for periodic
  cache-locality sorting.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None
    _HAS_CUPY = False


_BLOCK_SIZE = 256
_INV_C2 = 1.0 / (c_light * c_light)

PUSHER_IDS = {
    "boris": 0,
    "vay": 1,
    "hc": 2,
    "higuera_cary": 2,
    "higuera-cary": 2,
}


_SOURCE = r"""
extern "C" {

__device__ __forceinline__ double to_logical_1d_dev(
    double pos, int is_uni, double p_min, double L_ax, double dx0,
    const double* __restrict__ nodes, const double* __restrict__ deltas, int n_cells
) {
    if (is_uni) {
        return (pos - p_min) / dx0;
    }
    double n_wrap = (L_ax > 0.0) ? floor((pos - p_min) / L_ax) : 0.0;
    double pos_mod = (L_ax > 0.0) ? (p_min + (pos - p_min) - n_wrap * L_ax) : pos;
    int lo = 0;
    int hi = n_cells - 1;
    if (pos_mod < nodes[1]) {
        lo = 0;
    } else if (pos_mod >= nodes[n_cells - 1]) {
        lo = n_cells - 1;
    } else {
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (nodes[mid] <= pos_mod) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
    }
    double xi_mod = (double)lo + (pos_mod - nodes[lo]) / deltas[lo];
    return n_wrap * (double)n_cells + xi_mod;
}

__device__ __forceinline__ int bspline_1d_dev(
    double xi, int order, int staggered, double w[4]
) {
    double u = staggered ? (xi - 0.5) : xi;
    if (order == 0) {
        long long i0 = (long long)floor(u + 0.5);
        w[0] = 1.0;
        return (int)i0;
    }
    if (order == 1) {
        long long i0 = (long long)floor(u);
        double d = u - (double)i0;
        w[0] = 1.0 - d;
        w[1] = d;
        return (int)i0;
    }
    if (order == 2) {
        long long ic = (long long)floor(u + 0.5);
        double d = u - (double)ic;
        w[0] = 0.5 * (0.5 - d) * (0.5 - d);
        w[1] = 0.75 - d * d;
        w[2] = 0.5 * (0.5 + d) * (0.5 + d);
        return (int)(ic - 1);
    }
    long long ifloor = (long long)floor(u);
    double d = u - (double)ifloor;
    double omd = 1.0 - d;
    double d2 = d * d;
    double d3 = d2 * d;
    w[0] = (1.0 / 6.0) * omd * omd * omd;
    w[1] = (1.0 / 6.0) * (4.0 - 6.0 * d2 + 3.0 * d3);
    w[2] = (1.0 / 6.0) * (1.0 + 3.0 * d + 3.0 * d2 - 3.0 * d3);
    w[3] = (1.0 / 6.0) * d3;
    return (int)(ifloor - 1);
}

__device__ __forceinline__ int wrap_or_mask_dev(
    int idx, int n_cells, int is_per, int &valid
) {
    if (is_per) {
        int r = idx % n_cells;
        if (r < 0) r += n_cells;
        valid = 1;
        return r;
    }
    if (idx >= 0 && idx < n_cells) {
        valid = 1;
        return idx;
    }
    valid = 0;
    return (idx < 0) ? 0 : (n_cells - 1);
}

__device__ __forceinline__ double gather_comp_dev(
    const double* __restrict__ field_flat,
    int Nx, int Ny, int Nz,
    const int* ix_list, const int* vx_list, const double* wx, int nx,
    const int* iy_list, const int* vy_list, const double* wy, int ny,
    const int* iz_list, const int* vz_list, const double* wz, int nz
) {
    double out = 0.0;
    int stride_y = Nx;
    int stride_z = Nx * Ny;
    for (int oz = 0; oz < nz; ++oz) {
        int vz_m = vz_list[oz];
        int iz_off = iz_list[oz] * stride_z;
        double wz_m = wz[oz];
        for (int oy = 0; oy < ny; ++oy) {
            int vyz_m = vz_m & vy_list[oy];
            int iyz_off = iy_list[oy] * stride_y + iz_off;
            double wyz = wy[oy] * wz_m;
            for (int ox = 0; ox < nx; ++ox) {
                int v = vyz_m & vx_list[ox];
                double w = v ? (wx[ox] * wyz) : 0.0;
                out += w * field_flat[ix_list[ox] + iyz_off];
            }
        }
    }
    return out;
}

__device__ __forceinline__ void gather_eb_point_dev(
    double xi, double eta, double zeta,
    const double* __restrict__ Ex_f,
    const double* __restrict__ Ey_f,
    const double* __restrict__ Ez_f,
    const double* __restrict__ Bx_f,
    const double* __restrict__ By_f,
    const double* __restrict__ Bz_f,
    int Nx, int Ny, int Nz,
    int order, int p_stag,
    int per_x, int per_y, int per_z,
    double &Ex, double &Ey, double &Ez,
    double &Bx, double &By, double &Bz
) {
    const int stag_flag = 1;
    const int nod_flag = 0;

    double wx_nod[4], wy_nod[4], wz_nod[4];
    double wx_stg[4], wy_stg[4], wz_stg[4];

    int ix_nod0 = bspline_1d_dev(xi, order, nod_flag, wx_nod);
    int iy_nod0 = bspline_1d_dev(eta, order, nod_flag, wy_nod);
    int iz_nod0 = bspline_1d_dev(zeta, order, nod_flag, wz_nod);

    int ix_stg0 = bspline_1d_dev(xi, p_stag, stag_flag, wx_stg);
    int iy_stg0 = bspline_1d_dev(eta, p_stag, stag_flag, wy_stg);
    int iz_stg0 = bspline_1d_dev(zeta, p_stag, stag_flag, wz_stg);

    int n_nod = order + 1;
    int n_stg = p_stag + 1;

    int ix_nod[4], vx_nod[4], iy_nod[4], vy_nod[4], iz_nod[4], vz_nod[4];
    int ix_stg[4], vx_stg[4], iy_stg[4], vy_stg[4], iz_stg[4], vz_stg[4];

    for (int m = 0; m < n_nod; ++m) {
        ix_nod[m] = wrap_or_mask_dev(ix_nod0 + m, Nx, per_x, vx_nod[m]);
        iy_nod[m] = wrap_or_mask_dev(iy_nod0 + m, Ny, per_y, vy_nod[m]);
        iz_nod[m] = wrap_or_mask_dev(iz_nod0 + m, Nz, per_z, vz_nod[m]);
    }
    for (int m = 0; m < n_stg; ++m) {
        ix_stg[m] = wrap_or_mask_dev(ix_stg0 + m, Nx, per_x, vx_stg[m]);
        iy_stg[m] = wrap_or_mask_dev(iy_stg0 + m, Ny, per_y, vy_stg[m]);
        iz_stg[m] = wrap_or_mask_dev(iz_stg0 + m, Nz, per_z, vz_stg[m]);
    }

    Ex = gather_comp_dev(
        Ex_f, Nx, Ny, Nz,
        ix_stg, vx_stg, wx_stg, n_stg,
        iy_nod, vy_nod, wy_nod, n_nod,
        iz_nod, vz_nod, wz_nod, n_nod
    );
    Ey = gather_comp_dev(
        Ey_f, Nx, Ny, Nz,
        ix_nod, vx_nod, wx_nod, n_nod,
        iy_stg, vy_stg, wy_stg, n_stg,
        iz_nod, vz_nod, wz_nod, n_nod
    );
    Ez = gather_comp_dev(
        Ez_f, Nx, Ny, Nz,
        ix_nod, vx_nod, wx_nod, n_nod,
        iy_nod, vy_nod, wy_nod, n_nod,
        iz_stg, vz_stg, wz_stg, n_stg
    );

    Bx = gather_comp_dev(
        Bx_f, Nx, Ny, Nz,
        ix_nod, vx_nod, wx_nod, n_nod,
        iy_stg, vy_stg, wy_stg, n_stg,
        iz_stg, vz_stg, wz_stg, n_stg
    );
    By = gather_comp_dev(
        By_f, Nx, Ny, Nz,
        ix_stg, vx_stg, wx_stg, n_stg,
        iy_nod, vy_nod, wy_nod, n_nod,
        iz_stg, vz_stg, wz_stg, n_stg
    );
    Bz = gather_comp_dev(
        Bz_f, Nx, Ny, Nz,
        ix_stg, vx_stg, wx_stg, n_stg,
        iy_stg, vy_stg, wy_stg, n_stg,
        iz_nod, vz_nod, wz_nod, n_nod
    );
}

__device__ __forceinline__ void push_momentum_dev(
    double &ux, double &uy, double &uz,
    double Ex, double Ey, double Ez,
    double Bx, double By, double Bz,
    double q, double m, double dt,
    int pusher_id, double c_val, double inv_c2
) {
    double alpha = 0.5 * (q / m) * dt;
    if (pusher_id == 0) {
        // Boris (1970)
        double umx = ux + alpha * Ex;
        double umy = uy + alpha * Ey;
        double umz = uz + alpha * Ez;

        double inv_gamma_m = 1.0 / sqrt(1.0 + (umx * umx + umy * umy + umz * umz) * inv_c2);
        double tx = (alpha * Bx) * inv_gamma_m;
        double ty = (alpha * By) * inv_gamma_m;
        double tz = (alpha * Bz) * inv_gamma_m;

        double t_mag2 = tx * tx + ty * ty + tz * tz;
        double s_fac = 2.0 / (1.0 + t_mag2);
        double sx = tx * s_fac;
        double sy = ty * s_fac;
        double sz = tz * s_fac;

        double upx = umx + (umy * tz - umz * ty);
        double upy = umy + (umz * tx - umx * tz);
        double upz = umz + (umx * ty - umy * tx);

        ux = umx + (upy * sz - upz * sy) + alpha * Ex;
        uy = umy + (upz * sx - upx * sz) + alpha * Ey;
        uz = umz + (upx * sy - upy * sx) + alpha * Ez;
    } else if (pusher_id == 1) {
        // Vay (2008)
        double inv_gamma_old = 1.0 / sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * inv_c2);
        double vx = ux * inv_gamma_old;
        double vy = uy * inv_gamma_old;
        double vz = uz * inv_gamma_old;

        double taux = alpha * Bx;
        double tauy = alpha * By;
        double tauz = alpha * Bz;
        double tau2 = taux * taux + tauy * tauy + tauz * tauz;

        double upx = ux + 2.0 * alpha * Ex + (vy * tauz - vz * tauy);
        double upy = uy + 2.0 * alpha * Ey + (vz * taux - vx * tauz);
        double upz = uz + 2.0 * alpha * Ez + (vx * tauy - vy * taux);

        double up2_c2 = (upx * upx + upy * upy + upz * upz) * inv_c2;
        double ustar_c = (upx * taux + upy * tauy + upz * tauz) / c_val;
        double sigma = 1.0 + up2_c2 - tau2;

        double gamma_new = sqrt(
            0.5 * (sigma + sqrt(sigma * sigma + 4.0 * (tau2 + ustar_c * ustar_c)))
        );
        double inv_gamma_new = 1.0 / gamma_new;

        double tx = taux * inv_gamma_new;
        double ty = tauy * inv_gamma_new;
        double tz = tauz * inv_gamma_new;
        double s = 1.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        double up_dot_t = upx * tx + upy * ty + upz * tz;

        ux = s * (upx + up_dot_t * tx + (upy * tz - upz * ty));
        uy = s * (upy + up_dot_t * ty + (upz * tx - upx * tz));
        uz = s * (upz + up_dot_t * tz + (upx * ty - upy * tx));
    } else {
        // Higuera-Cary (2017)
        double umx = ux + alpha * Ex;
        double umy = uy + alpha * Ey;
        double umz = uz + alpha * Ez;

        double taux = alpha * Bx;
        double tauy = alpha * By;
        double tauz = alpha * Bz;
        double tau2 = taux * taux + tauy * tauy + tauz * tauz;

        double gamma_m2 = 1.0 + (umx * umx + umy * umy + umz * umz) * inv_c2;
        double ustar_c = (umx * taux + umy * tauy + umz * tauz) / c_val;
        double sigma = gamma_m2 - tau2;

        double gamma_hc = sqrt(
            0.5 * (sigma + sqrt(sigma * sigma + 4.0 * (tau2 + ustar_c * ustar_c)))
        );
        double inv_gamma_hc = 1.0 / gamma_hc;

        double tx = taux * inv_gamma_hc;
        double ty = tauy * inv_gamma_hc;
        double tz = tauz * inv_gamma_hc;

        double t_mag2 = tx * tx + ty * ty + tz * tz;
        double s_fac = 2.0 / (1.0 + t_mag2);
        double sx = tx * s_fac;
        double sy = ty * s_fac;
        double sz = tz * s_fac;

        double upx = umx + (umy * tz - umz * ty);
        double upy = umy + (umz * tx - umx * tz);
        double upz = umz + (umx * ty - umy * tx);

        ux = umx + (upy * sz - upz * sy) + alpha * Ex;
        uy = umy + (upz * sx - upx * sz) + alpha * Ey;
        uz = umz + (upx * sy - upy * sx) + alpha * Ez;
    }
}

__global__ void gather_fields_kernel(
    const double* __restrict__ xi,
    const double* __restrict__ eta,
    const double* __restrict__ zeta,
    const double* __restrict__ Ex_f,
    const double* __restrict__ Ey_f,
    const double* __restrict__ Ez_f,
    const double* __restrict__ Bx_f,
    const double* __restrict__ By_f,
    const double* __restrict__ Bz_f,
    double* __restrict__ Ex_out,
    double* __restrict__ Ey_out,
    double* __restrict__ Ez_out,
    double* __restrict__ Bx_out,
    double* __restrict__ By_out,
    double* __restrict__ Bz_out,
    int Nx, int Ny, int Nz,
    int order, int p_stag,
    int per_x, int per_y, int per_z,
    int n_p
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_p) return;

    double Ex, Ey, Ez, Bx, By, Bz;
    gather_eb_point_dev(
        xi[tid], eta[tid], zeta[tid],
        Ex_f, Ey_f, Ez_f, Bx_f, By_f, Bz_f,
        Nx, Ny, Nz, order, p_stag, per_x, per_y, per_z,
        Ex, Ey, Ez, Bx, By, Bz
    );
    Ex_out[tid] = Ex;
    Ey_out[tid] = Ey;
    Ez_out[tid] = Ez;
    Bx_out[tid] = Bx;
    By_out[tid] = By;
    Bz_out[tid] = Bz;
}

__global__ void gather_and_push_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uz,
    double* __restrict__ x0_out,
    double* __restrict__ y0_out,
    double* __restrict__ z0_out,
    double* __restrict__ xi0_out,
    double* __restrict__ eta0_out,
    double* __restrict__ zeta0_out,
    double* __restrict__ xi1_out,
    double* __restrict__ eta1_out,
    double* __restrict__ zeta1_out,
    const double* __restrict__ E_flat,
    const double* __restrict__ B_flat,
    const double* __restrict__ Ex_ext_arr,
    const double* __restrict__ Ey_ext_arr,
    const double* __restrict__ Ez_ext_arr,
    const double* __restrict__ Bx_ext_arr,
    const double* __restrict__ By_ext_arr,
    const double* __restrict__ Bz_ext_arr,
    double Ex_ext0, double Ey_ext0, double Ez_ext0,
    double Bx_ext0, double By_ext0, double Bz_ext0,
    int Nx, int Ny, int Nz,
    int uni_x, int uni_y, int uni_z,
    double xmin, double ymin, double zmin,
    double Lx, double Ly, double Lz,
    double dx0, double dy0, double dz0,
    const double* __restrict__ x_nodes, const double* __restrict__ dx_arr,
    const double* __restrict__ y_nodes, const double* __restrict__ dy_arr,
    const double* __restrict__ z_nodes, const double* __restrict__ dz_arr,
    int order, int p_stag,
    int per_x, int per_y, int per_z,
    double q, double m, double dt,
    int pusher_id, int advance_pos,
    double c_val, double inv_c2,
    const int* __restrict__ d_n_particles,
    int n_host
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_active = d_n_particles ? d_n_particles[0] : n_host;
    if (tid >= n_active) return;

    int N = Nx * Ny * Nz;
    const double* Ex_f = E_flat;
    const double* Ey_f = E_flat + N;
    const double* Ez_f = E_flat + 2 * N;
    const double* Bx_f = B_flat;
    const double* By_f = B_flat + N;
    const double* Bz_f = B_flat + 2 * N;

    double xp0 = x[tid];
    double yp0 = y[tid];
    double zp0 = z[tid];

    double xi0 = to_logical_1d_dev(xp0, uni_x, xmin, Lx, dx0, x_nodes, dx_arr, Nx);
    double eta0 = to_logical_1d_dev(yp0, uni_y, ymin, Ly, dy0, y_nodes, dy_arr, Ny);
    double zeta0 = to_logical_1d_dev(zp0, uni_z, zmin, Lz, dz0, z_nodes, dz_arr, Nz);

    double Ex, Ey, Ez, Bx, By, Bz;
    gather_eb_point_dev(
        xi0, eta0, zeta0,
        Ex_f, Ey_f, Ez_f, Bx_f, By_f, Bz_f,
        Nx, Ny, Nz, order, p_stag, per_x, per_y, per_z,
        Ex, Ey, Ez, Bx, By, Bz
    );

    Ex += Ex_ext0 + (Ex_ext_arr ? Ex_ext_arr[tid] : 0.0);
    Ey += Ey_ext0 + (Ey_ext_arr ? Ey_ext_arr[tid] : 0.0);
    Ez += Ez_ext0 + (Ez_ext_arr ? Ez_ext_arr[tid] : 0.0);
    Bx += Bx_ext0 + (Bx_ext_arr ? Bx_ext_arr[tid] : 0.0);
    By += By_ext0 + (By_ext_arr ? By_ext_arr[tid] : 0.0);
    Bz += Bz_ext0 + (Bz_ext_arr ? Bz_ext_arr[tid] : 0.0);

    double uxp = ux[tid];
    double uyp = uy[tid];
    double uzp = uz[tid];

    push_momentum_dev(
        uxp, uyp, uzp,
        Ex, Ey, Ez, Bx, By, Bz,
        q, m, dt, pusher_id, c_val, inv_c2
    );

    ux[tid] = uxp;
    uy[tid] = uyp;
    uz[tid] = uzp;

    if (advance_pos) {
        x0_out[tid] = xp0;
        y0_out[tid] = yp0;
        z0_out[tid] = zp0;
        xi0_out[tid] = xi0;
        eta0_out[tid] = eta0;
        zeta0_out[tid] = zeta0;

        double inv_gamma = 1.0 / sqrt(1.0 + (uxp * uxp + uyp * uyp + uzp * uzp) * inv_c2);
        double dt_ig = dt * inv_gamma;
        double xp1 = xp0 + dt_ig * uxp;
        double yp1 = yp0 + dt_ig * uyp;
        double zp1 = zp0 + dt_ig * uzp;

        x[tid] = xp1;
        y[tid] = yp1;
        z[tid] = zp1;

        xi1_out[tid] = to_logical_1d_dev(xp1, uni_x, xmin, Lx, dx0, x_nodes, dx_arr, Nx);
        eta1_out[tid] = to_logical_1d_dev(yp1, uni_y, ymin, Ly, dy0, y_nodes, dy_arr, Ny);
        zeta1_out[tid] = to_logical_1d_dev(zp1, uni_z, zmin, Lz, dz0, z_nodes, dz_arr, Nz);
    }
}

__device__ __forceinline__ void esirkepov_1d_window_dev(
    double xi0, double xi1, int order,
    int &i_base, double S0[5], double S1[5], double dS[5], double F[4],
    int* d_cfl_error
) {
    double w0[4], w1[4];
    int b0 = bspline_1d_dev(xi0, order, 0, w0);
    int b1 = bspline_1d_dev(xi1, order, 0, w1);
    int db = b1 - b0;
    if ((db < -1 || db > 1) && d_cfl_error != nullptr) {
        atomicAdd(d_cfl_error, 1);
    }
    i_base = (b0 < b1) ? b0 : b1;
    int m_win = order + 2;

    for (int m = 0; m < m_win; ++m) {
        if (db == -1) {
            S0[m] = (m == 0) ? 0.0 : w0[m - 1];
            S1[m] = (m == m_win - 1) ? 0.0 : w1[m];
        } else if (db == 1) {
            S0[m] = (m == m_win - 1) ? 0.0 : w0[m];
            S1[m] = (m == 0) ? 0.0 : w1[m - 1];
        } else {
            S0[m] = (m == m_win - 1) ? 0.0 : w0[m];
            S1[m] = (m == m_win - 1) ? 0.0 : w1[m];
        }
        dS[m] = S1[m] - S0[m];
    }

    double acc = -dS[0];
    F[0] = acc;
    for (int m = 1; m < order; ++m) {
        acc -= dS[m];
        F[m] = acc;
    }
    if (order >= 1) {
        F[order] = dS[order + 1];
    }
}

__global__ void deposit_esirkepov_kernel(
    const double* __restrict__ xi0,
    const double* __restrict__ eta0,
    const double* __restrict__ zeta0,
    const double* __restrict__ xi1,
    const double* __restrict__ eta1,
    const double* __restrict__ zeta1,
    const double* __restrict__ w_arr,
    double q_scale,
    double* __restrict__ Wx,
    double* __restrict__ Wy,
    double* __restrict__ Wz,
    int Nx, int Ny, int Nz,
    int order,
    int per_x, int per_y, int per_z,
    const int* __restrict__ d_n_particles,
    int n_host,
    int* __restrict__ d_cfl_error
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_active = d_n_particles ? d_n_particles[0] : n_host;
    if (tid >= n_active) return;

    double qw = q_scale * w_arr[tid];
    int m_win = order + 2;
    int m_flux = order + 1;

    int ix_base, iy_base, iz_base;
    double Sx0[5], Sx1[5], dSx[5], Fx[4];
    double Sy0[5], Sy1[5], dSy[5], Fy[4];
    double Sz0[5], Sz1[5], dSz[5], Fz[4];

    esirkepov_1d_window_dev(xi0[tid], xi1[tid], order, ix_base, Sx0, Sx1, dSx, Fx, d_cfl_error);
    esirkepov_1d_window_dev(eta0[tid], eta1[tid], order, iy_base, Sy0, Sy1, dSy, Fy, d_cfl_error);
    esirkepov_1d_window_dev(zeta0[tid], zeta1[tid], order, iz_base, Sz0, Sz1, dSz, Fz, d_cfl_error);

    int stride_y = Nx;
    int stride_z = Nx * Ny;

    int ix_list[5], vx_list[5];
    int iy_list[5], vy_list[5];
    int iz_list[5], vz_list[5];

    for (int m = 0; m < m_win; ++m) {
        ix_list[m] = wrap_or_mask_dev(ix_base + m, Nx, per_x, vx_list[m]);
        iy_list[m] = wrap_or_mask_dev(iy_base + m, Ny, per_y, vy_list[m]) * stride_y;
        iz_list[m] = wrap_or_mask_dev(iz_base + m, Nz, per_z, vz_list[m]) * stride_z;
    }

    // Esirkepov (2001) transverse weight decomposition:
    // W_yz = S_y0 * S_z0 + 0.5 * (S_y0 * dS_z + dS_y * S_z0) + (1/3) * dS_y * dS_z
    const double esirkepov_dd_coeff = 1.0 / 3.0;

    double qFx[4], qFy[4], qFz[4];
    for (int m = 0; m < m_flux; ++m) {
        qFx[m] = qw * Fx[m];
        qFy[m] = qw * Fy[m];
        qFz[m] = qw * Fz[m];
    }

    // 1. Deposit Wx
    for (int mz = 0; mz < m_win; ++mz) {
        int iz_off = iz_list[mz];
        int vz_m = vz_list[mz];
        double sz0_m = Sz0[mz];
        double dsz_m = dSz[mz];
        for (int my = 0; my < m_win; ++my) {
            int vyz = vz_m & vy_list[my];
            if (!vyz) continue;
            double sy0_m = Sy0[my];
            double dsy_m = dSy[my];
            double wyz = (
                sy0_m * sz0_m
                + 0.5 * (sy0_m * dsz_m + dsy_m * sz0_m)
                + esirkepov_dd_coeff * dsy_m * dsz_m
            );
            int iyz_off = iy_list[my] + iz_off;
            for (int mx = 0; mx < m_flux; ++mx) {
                if (vx_list[mx]) {
                    double val = qFx[mx] * wyz;
                    if (val != 0.0) {
                        atomicAdd(&Wx[ix_list[mx] + iyz_off], val);
                    }
                }
            }
        }
    }

    // 2. Deposit Wy
    for (int mz = 0; mz < m_win; ++mz) {
        int iz_off = iz_list[mz];
        int vz_m = vz_list[mz];
        double sz0_m = Sz0[mz];
        double dsz_m = dSz[mz];
        for (int mx = 0; mx < m_win; ++mx) {
            int vxz = vz_m & vx_list[mx];
            if (!vxz) continue;
            double sx0_m = Sx0[mx];
            double dsx_m = dSx[mx];
            double wxz = (
                sx0_m * sz0_m
                + 0.5 * (sx0_m * dsz_m + dsx_m * sz0_m)
                + esirkepov_dd_coeff * dsx_m * dsz_m
            );
            int ixz_off = ix_list[mx] + iz_off;
            for (int my = 0; my < m_flux; ++my) {
                if (vy_list[my]) {
                    double val = qFy[my] * wxz;
                    if (val != 0.0) {
                        atomicAdd(&Wy[ixz_off + iy_list[my]], val);
                    }
                }
            }
        }
    }

    // 3. Deposit Wz
    for (int my = 0; my < m_win; ++my) {
        int iy_off = iy_list[my];
        int vy_m = vy_list[my];
        double sy0_m = Sy0[my];
        double dsy_m = dSy[my];
        for (int mx = 0; mx < m_win; ++mx) {
            int vxy = vy_m & vx_list[mx];
            if (!vxy) continue;
            double sx0_m = Sx0[mx];
            double dsx_m = dSx[mx];
            double wxy = (
                sx0_m * sy0_m
                + 0.5 * (sx0_m * dsy_m + dsx_m * sy0_m)
                + esirkepov_dd_coeff * dsx_m * dsy_m
            );
            int ixy_off = ix_list[mx] + iy_off;
            for (int mz = 0; mz < m_flux; ++mz) {
                if (vz_list[mz]) {
                    double val = qFz[mz] * wxy;
                    if (val != 0.0) {
                        atomicAdd(&Wz[ixy_off + iz_list[mz]], val);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void deposit_charge_point_dev(
    double xi, double eta, double zeta, double qw,
    double* __restrict__ q_flat,
    int Nx, int Ny, int Nz,
    int order, int per_x, int per_y, int per_z
) {
    double wx[4], wy[4], wz[4];
    int ix0 = bspline_1d_dev(xi, order, 0, wx);
    int iy0 = bspline_1d_dev(eta, order, 0, wy);
    int iz0 = bspline_1d_dev(zeta, order, 0, wz);

    int n_nod = order + 1;
    int stride_y = Nx;
    int stride_z = Nx * Ny;

    int ix_list[4], vx_list[4];
    int iy_list[4], vy_list[4];
    int iz_list[4], vz_list[4];

    for (int m = 0; m < n_nod; ++m) {
        ix_list[m] = wrap_or_mask_dev(ix0 + m, Nx, per_x, vx_list[m]);
        iy_list[m] = wrap_or_mask_dev(iy0 + m, Ny, per_y, vy_list[m]) * stride_y;
        iz_list[m] = wrap_or_mask_dev(iz0 + m, Nz, per_z, vz_list[m]) * stride_z;
    }

    for (int oz = 0; oz < n_nod; ++oz) {
        if (!vz_list[oz]) continue;
        int iz_off = iz_list[oz];
        double wz_m = wz[oz];
        for (int oy = 0; oy < n_nod; ++oy) {
            if (!vy_list[oy]) continue;
            int iyz_off = iy_list[oy] + iz_off;
            double q_yz = qw * (wy[oy] * wz_m);
            for (int ox = 0; ox < n_nod; ++ox) {
                if (vx_list[ox]) {
                    double val = wx[ox] * q_yz;
                    if (val != 0.0) {
                        atomicAdd(&q_flat[ix_list[ox] + iyz_off], val);
                    }
                }
            }
        }
    }
}

__global__ void deposit_charge_kernel(
    const double* __restrict__ xi,
    const double* __restrict__ eta,
    const double* __restrict__ zeta,
    const double* __restrict__ w_arr,
    double q_scale,
    double* __restrict__ q_flat,
    int Nx, int Ny, int Nz,
    int order,
    int per_x, int per_y, int per_z,
    const int* __restrict__ d_n_particles,
    int n_host
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_active = d_n_particles ? d_n_particles[0] : n_host;
    if (tid >= n_active) return;

    double qw = q_scale * w_arr[tid];
    deposit_charge_point_dev(
        xi[tid], eta[tid], zeta[tid], qw,
        q_flat, Nx, Ny, Nz, order, per_x, per_y, per_z
    );
}

__global__ void boundary_prepare_paths_kernel(
    const double* __restrict__ x0,
    const double* __restrict__ y0,
    const double* __restrict__ z0,
    const double* __restrict__ x1,
    const double* __restrict__ y1,
    const double* __restrict__ z1,
    const double* __restrict__ xi0,
    const double* __restrict__ eta0,
    const double* __restrict__ zeta0,
    const double* xi1,
    const double* eta1,
    const double* zeta1,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w_arr,
    double q_charge, double mass, double inv_c2,
    double* xi1_dep,
    double* eta1_dep,
    double* zeta1_dep,
    int* __restrict__ keep_mask,
    int Nx, int Ny, int Nz,
    int per_x, int per_y, int per_z,
    double xi_min_x, double xi_max_x,
    double xi_min_y, double xi_max_y,
    double xi_min_z, double xi_max_z,
    const unsigned char* __restrict__ solid_mask,
    int n_collectors,
    const int* __restrict__ col_types,
    const double* __restrict__ col_params,
    int order,
    double* __restrict__ q_bg_flat,
    double* __restrict__ d_absorbed_charge,
    double* __restrict__ d_absorbed_energy,
    unsigned long long* __restrict__ d_absorbed_count,
    const int* __restrict__ d_n_particles,
    int n_launch
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;
    int n_active = d_n_particles ? d_n_particles[0] : n_launch;
    if (tid >= n_active) {
        keep_mask[tid] = 0;
        return;
    }

    double c0_arr[3] = {xi0[tid], eta0[tid], zeta0[tid]};
    double c1_arr[3] = {xi1[tid], eta1[tid], zeta1[tid]};
    double cmin_arr[3] = {xi_min_x, xi_min_y, xi_min_z};
    double cmax_arr[3] = {xi_max_x, xi_max_y, xi_max_z};
    int per_arr[3] = {per_x, per_y, per_z};

    double alpha_min = 1.0;
    int hit_code = -1;

    // 1. Domain / CPML inner-face crossings on non-periodic axes
    for (int a = 0; a < 3; ++a) {
        if (per_arr[a]) continue;
        double c0 = c0_arr[a];
        double c1 = c1_arr[a];
        double c_min = cmin_arr[a];
        double c_max = cmax_arr[a];

        if (c1 < c_min) {
            double dc = fmin(c1 - c0, -1e-30);
            double a_low = (c0 > c_min) ? fmin(fmax((c_min - c0) / dc, 0.0), 1.0) : 0.0;
            if (a_low <= alpha_min) {
                alpha_min = a_low;
                hit_code = 2 * a;
            }
        }
        if (c1 > c_max) {
            double dc = fmax(c1 - c0, 1e-30);
            double a_high = (c0 < c_max) ? fmin(fmax((c_max - c0) / dc, 0.0), 1.0) : 0.0;
            if (a_high <= alpha_min) {
                alpha_min = a_high;
                hit_code = 2 * a + 1;
            }
        }
    }

    // 2. Solid / PEC cell mask crossings
    if (solid_mask != nullptr && hit_code < 0) {
        int ix1 = min(max((int)floor(c1_arr[0]), 0), Nx - 1);
        int iy1 = min(max((int)floor(c1_arr[1]), 0), Ny - 1);
        int iz1 = min(max((int)floor(c1_arr[2]), 0), Nz - 1);
        // solid_mask is stored in C-order (Nx, Ny, Nz) or F-order; pass F-order flat
        int sm_idx = ix1 + Nx * (iy1 + Ny * iz1);
        if (solid_mask[sm_idx]) {
            int ix0 = min(max((int)floor(c0_arr[0]), 0), Nx - 1);
            int iy0 = min(max((int)floor(c0_arr[1]), 0), Ny - 1);
            int iz0 = min(max((int)floor(c0_arr[2]), 0), Nz - 1);
            int i0_arr[3] = {ix0, iy0, iz0};
            int i1_arr[3] = {ix1, iy1, iz1};

            double a_pec = 1.0;
            for (int a = 0; a < 3; ++a) {
                int di = i1_arr[a] - i0_arr[a];
                double dc = c1_arr[a] - c0_arr[a];
                if (di != 0 && fabs(dc) > 1e-30) {
                    double face = (di > 0) ? (double)i1_arr[a] : (double)i0_arr[a];
                    double a_ax = fmin(fmax((face - c0_arr[a]) / dc, 0.0), 1.0);
                    a_pec = fmin(a_pec, a_ax);
                }
            }
            if (ix0 == ix1 && iy0 == iy1 && iz0 == iz1) {
                a_pec = 0.0;
            }
            if (a_pec <= alpha_min) {
                alpha_min = a_pec;
                hit_code = 6;
            }
        }
    }

    // 3. User-defined plane or bounding-box collectors
    if (n_collectors > 0) {
        double p0_arr[3] = {x0[tid], y0[tid], z0[tid]};
        double p1_arr[3] = {x1[tid], y1[tid], z1[tid]};
        for (int c_idx = 0; c_idx < n_collectors; ++c_idx) {
            int ctype = col_types[c_idx];
            const double* cp = col_params + 6 * c_idx;
            int c_hit = 0;
            double c_alpha = 1.0;
            if (ctype == 0) {
                // Plane collector: cp[0]=axis, cp[1]=pos, cp[2]=direction
                int axis = (int)cp[0];
                double pos = cp[1];
                double dir = cp[2];
                double pc0 = p0_arr[axis];
                double pc1 = p1_arr[axis];
                if (dir > 0.0) {
                    if (pc1 >= pos) {
                        c_hit = 1;
                        double dc = fmax(pc1 - pc0, 1e-30);
                        double a_cross = fmin(fmax((pos - pc0) / dc, 0.0), 1.0);
                        c_alpha = (pc0 < pos) ? a_cross : 0.0;
                    }
                } else {
                    if (pc1 <= pos) {
                        c_hit = 1;
                        double dc = fmin(pc1 - pc0, -1e-30);
                        double a_cross = fmin(fmax((pos - pc0) / dc, 0.0), 1.0);
                        c_alpha = (pc0 > pos) ? a_cross : 0.0;
                    }
                }
            } else if (ctype == 1) {
                // Bounding-box collector: cp[0..5] = xmin, xmax, ymin, ymax, zmin, zmax
                if (p1_arr[0] >= cp[0] && p1_arr[0] <= cp[1] &&
                    p1_arr[1] >= cp[2] && p1_arr[1] <= cp[3] &&
                    p1_arr[2] >= cp[4] && p1_arr[2] <= cp[5]) {
                    c_hit = 1;
                    double t_enter = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        double bmin = cp[2 * a];
                        double bmax = cp[2 * a + 1];
                        double dc = p1_arr[a] - p0_arr[a];
                        if (p0_arr[a] < bmin && p1_arr[a] >= bmin) {
                            t_enter = fmax(t_enter, (bmin - p0_arr[a]) / dc);
                        }
                        if (p0_arr[a] > bmax && p1_arr[a] <= bmax) {
                            t_enter = fmax(t_enter, (bmax - p0_arr[a]) / dc);
                        }
                    }
                    c_alpha = fmin(fmax(t_enter, 0.0), 1.0);
                }
            }
            if (c_hit && c_alpha <= alpha_min) {
                alpha_min = c_alpha;
                hit_code = 7 + c_idx;
            }
        }
    }

    if (hit_code < 0) {
        xi1_dep[tid] = c1_arr[0];
        eta1_dep[tid] = c1_arr[1];
        zeta1_dep[tid] = c1_arr[2];
        keep_mask[tid] = 1;
        return;
    }

    double dep_x = c0_arr[0] + alpha_min * (c1_arr[0] - c0_arr[0]);
    double dep_y = c0_arr[1] + alpha_min * (c1_arr[1] - c0_arr[1]);
    double dep_z = c0_arr[2] + alpha_min * (c1_arr[2] - c0_arr[2]);

    if (!per_x) dep_x = fmin(fmax(dep_x, xi_min_x), xi_max_x);
    if (!per_y) dep_y = fmin(fmax(dep_y, xi_min_y), xi_max_y);
    if (!per_z) dep_z = fmin(fmax(dep_z, xi_min_z), xi_max_z);

    xi1_dep[tid] = dep_x;
    eta1_dep[tid] = dep_y;
    zeta1_dep[tid] = dep_z;
    keep_mask[tid] = 0;

    double w_p = w_arr[tid];
    double qw = q_charge * w_p;
    double uxp = ux[tid];
    double uyp = uy[tid];
    double uzp = uz[tid];
    double u2 = uxp * uxp + uyp * uyp + uzp * uzp;
    double g = sqrt(1.0 + u2 * inv_c2);
    double ke = w_p * mass * (u2 / (g + 1.0));

    if (d_absorbed_charge != nullptr) {
        atomicAdd(&d_absorbed_charge[hit_code], qw);
        atomicAdd(&d_absorbed_energy[hit_code], ke);
        atomicAdd(&d_absorbed_count[hit_code], 1ULL);
    }

    if (q_bg_flat != nullptr) {
        deposit_charge_point_dev(
            dep_x, dep_y, dep_z, qw,
            q_bg_flat, Nx, Ny, Nz, order, per_x, per_y, per_z
        );
    }
}

__global__ void periodic_wrap_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    int per_x, int per_y, int per_z,
    double xmin, double ymin, double zmin,
    double Lx, double Ly, double Lz,
    const int* __restrict__ d_n_particles,
    int n_host
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_active = d_n_particles ? d_n_particles[0] : n_host;
    if (tid >= n_active) return;

    if (per_x) {
        double r = fmod(x[tid] - xmin, Lx);
        if (r < 0.0) r += Lx;
        x[tid] = xmin + r;
    }
    if (per_y) {
        double r = fmod(y[tid] - ymin, Ly);
        if (r < 0.0) r += Ly;
        y[tid] = ymin + r;
    }
    if (per_z) {
        double r = fmod(z[tid] - zmin, Lz);
        if (r < 0.0) r += Lz;
        z[tid] = zmin + r;
    }
}

__global__ void compact_scan_blocks_kernel(
    const int* __restrict__ keep_mask,
    int* __restrict__ dst_idx,
    int* __restrict__ block_sums,
    const int* __restrict__ d_n_particles,
    int n_launch
) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int n_active = d_n_particles ? d_n_particles[0] : n_launch;

    int val = (gid < n_active) ? (keep_mask[gid] != 0 ? 1 : 0) : 0;
    sdata[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < 256; offset <<= 1) {
        int temp = (tid >= offset) ? sdata[tid - offset] : 0;
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    if (gid < n_launch) {
        dst_idx[gid] = sdata[tid] - val;
    }
    if (tid == 255) {
        block_sums[blockIdx.x] = sdata[255];
    }
}

__global__ void compact_scan_totals_kernel(
    const int* __restrict__ block_sums,
    int* __restrict__ block_offsets,
    int* __restrict__ d_n_keep,
    int n_blocks
) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int running_total = 0;

    for (int base = 0; base < n_blocks; base += 256) {
        int b = base + tid;
        int val = (b < n_blocks) ? block_sums[b] : 0;
        sdata[tid] = val;
        __syncthreads();

        for (int offset = 1; offset < 256; offset <<= 1) {
            int temp = (tid >= offset) ? sdata[tid - offset] : 0;
            __syncthreads();
            sdata[tid] += temp;
            __syncthreads();
        }

        if (b < n_blocks) {
            block_offsets[b] = running_total + sdata[tid] - val;
        }
        int chunk_sum = sdata[255];
        __syncthreads();
        running_total += chunk_sum;
    }

    if (tid == 0) {
        d_n_keep[0] = running_total;
    }
}

__global__ void compact_scatter_kernel(
    const int* __restrict__ keep_mask,
    const int* __restrict__ dst_idx,
    const int* __restrict__ block_offsets,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const double* __restrict__ z_in,
    const double* __restrict__ ux_in,
    const double* __restrict__ uy_in,
    const double* __restrict__ uz_in,
    const double* __restrict__ w_in,
    const long long* __restrict__ id_in,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double* __restrict__ z_out,
    double* __restrict__ ux_out,
    double* __restrict__ uy_out,
    double* __restrict__ uz_out,
    double* __restrict__ w_out,
    long long* __restrict__ id_out,
    int has_id,
    const int* __restrict__ d_n_particles,
    int n_launch
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_active = d_n_particles ? d_n_particles[0] : n_launch;
    if (gid >= n_active) return;

    if (keep_mask[gid] != 0) {
        int dst = dst_idx[gid] + block_offsets[blockIdx.x];
        x_out[dst] = x_in[gid];
        y_out[dst] = y_in[gid];
        z_out[dst] = z_in[gid];
        ux_out[dst] = ux_in[gid];
        uy_out[dst] = uy_in[gid];
        uz_out[dst] = uz_in[gid];
        w_out[dst] = w_in[gid];
        if (has_id && id_in != nullptr && id_out != nullptr) {
            id_out[dst] = id_in[gid];
        }
    }
}

__global__ void compute_cell_index_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    long long* __restrict__ cell_idx,
    int Nx, int Ny, int Nz,
    int uni_x, int uni_y, int uni_z,
    double xmin, double ymin, double zmin,
    double Lx, double Ly, double Lz,
    double dx0, double dy0, double dz0,
    const double* __restrict__ x_nodes, const double* __restrict__ dx_arr,
    const double* __restrict__ y_nodes, const double* __restrict__ dy_arr,
    const double* __restrict__ z_nodes, const double* __restrict__ dz_arr,
    const int* __restrict__ d_n_particles,
    int n_launch
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;
    int n_active = d_n_particles ? d_n_particles[0] : n_launch;
    if (tid >= n_active) {
        cell_idx[tid] = (long long)(Nx * Ny * Nz) + (long long)tid;
        return;
    }

    double xi = to_logical_1d_dev(x[tid], uni_x, xmin, Lx, dx0, x_nodes, dx_arr, Nx);
    double eta = to_logical_1d_dev(y[tid], uni_y, ymin, Ly, dy0, y_nodes, dy_arr, Ny);
    double zeta = to_logical_1d_dev(z[tid], uni_z, zmin, Lz, dz0, z_nodes, dz_arr, Nz);

    long long ix = min(max((long long)floor(xi), 0LL), (long long)(Nx - 1));
    long long iy = min(max((long long)floor(eta), 0LL), (long long)(Ny - 1));
    long long iz = min(max((long long)floor(zeta), 0LL), (long long)(Nz - 1));

    cell_idx[tid] = ix + (long long)Nx * (iy + (long long)Ny * iz);
}

__global__ void permute_particles_kernel(
    const long long* __restrict__ perm,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const double* __restrict__ z_in,
    const double* __restrict__ ux_in,
    const double* __restrict__ uy_in,
    const double* __restrict__ uz_in,
    const double* __restrict__ w_in,
    const long long* __restrict__ id_in,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double* __restrict__ z_out,
    double* __restrict__ ux_out,
    double* __restrict__ uy_out,
    double* __restrict__ uz_out,
    double* __restrict__ w_out,
    long long* __restrict__ id_out,
    int has_id,
    int n_launch
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;
    long long src = perm[tid];
    x_out[tid] = x_in[src];
    y_out[tid] = y_in[src];
    z_out[tid] = z_in[src];
    ux_out[tid] = ux_in[src];
    uy_out[tid] = uy_in[src];
    uz_out[tid] = uz_in[src];
    w_out[tid] = w_in[src];
    if (has_id && id_in != nullptr && id_out != nullptr) {
        id_out[tid] = id_in[src];
    }
}

__global__ void compute_centred_B_kernel(
    const double* __restrict__ H_prev,
    const double* __restrict__ H_curr,
    double* __restrict__ B_out,
    double half_mu0,
    int n_total3
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total3) return;
    B_out[tid] = half_mu0 * (H_prev[tid] + H_curr[tid]);
}

__global__ void finalize_j_delta_kernel(
    const double* Wx,
    const double* Wy,
    const double* Wz,
    const double* __restrict__ itA,
    double* J_particles,
    double* __restrict__ J_prev,
    double* __restrict__ J_solver,
    double inv_dt,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int i0 = tid;
    int i1 = tid + N;
    int i2 = tid + 2 * N;

    double jx = Wx[tid] * itA[i0] * inv_dt;
    double jy = Wy[tid] * itA[i1] * inv_dt;
    double jz = Wz[tid] * itA[i2] * inv_dt;

    J_solver[i0] += jx - J_prev[i0];
    J_solver[i1] += jy - J_prev[i1];
    J_solver[i2] += jz - J_prev[i2];

    J_particles[i0] = jx;
    J_particles[i1] = jy;
    J_particles[i2] = jz;

    J_prev[i0] = jx;
    J_prev[i1] = jy;
    J_prev[i2] = jz;
}

__global__ void classify_and_pack_migration_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ ux,
    const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double* __restrict__ w,
    const long long* __restrict__ pid,
    int has_ids,
    int has_lo,
    int has_hi,
    double z_own_min,
    double z_own_max,
    double z_ghost_lo,
    double z_ghost_hi,
    int* __restrict__ keep_mask,
    int* __restrict__ counts,
    double* __restrict__ send_lo,
    double* __restrict__ send_hi,
    int pack_cap,
    const int* __restrict__ d_n_particles,
    int n_launch
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_launch) return;
    int n_active = (d_n_particles != nullptr) ? d_n_particles[0] : n_launch;
    if (tid >= n_active) {
        keep_mask[tid] = 0;
        return;
    }

    double zp = z[tid];
    if (has_lo && zp < z_own_min) {
        keep_mask[tid] = 0;
        if (zp < z_ghost_lo) {
            atomicAdd(&counts[3], 1);
        }
        int dst = atomicAdd(&counts[0], 1);
        if (dst < pack_cap) {
            int base = dst * 8;
            send_lo[base + 0] = x[tid];
            send_lo[base + 1] = y[tid];
            send_lo[base + 2] = zp;
            send_lo[base + 3] = ux[tid];
            send_lo[base + 4] = uy[tid];
            send_lo[base + 5] = uz[tid];
            send_lo[base + 6] = w[tid];
            send_lo[base + 7] = has_ids ? __longlong_as_double(pid[tid]) : 0.0;
        }
    } else if (has_hi && zp >= z_own_max) {
        keep_mask[tid] = 0;
        if (zp >= z_ghost_hi) {
            atomicAdd(&counts[3], 1);
        }
        int dst = atomicAdd(&counts[1], 1);
        if (dst < pack_cap) {
            int base = dst * 8;
            send_hi[base + 0] = x[tid];
            send_hi[base + 1] = y[tid];
            send_hi[base + 2] = zp;
            send_hi[base + 3] = ux[tid];
            send_hi[base + 4] = uy[tid];
            send_hi[base + 5] = uz[tid];
            send_hi[base + 6] = w[tid];
            send_hi[base + 7] = has_ids ? __longlong_as_double(pid[tid]) : 0.0;
        }
    } else {
        keep_mask[tid] = 1;
        atomicAdd(&counts[2], 1);
    }
}

__global__ void unpack_migration_kernel(
    const double* __restrict__ recv_buf,
    int n_recv,
    int dst_offset,
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double* __restrict__ uz,
    double* __restrict__ w,
    long long* __restrict__ pid,
    int has_ids
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_recv) return;
    int dst = dst_offset + tid;
    int base = tid * 8;
    x[dst] = recv_buf[base + 0];
    y[dst] = recv_buf[base + 1];
    z[dst] = recv_buf[base + 2];
    ux[dst] = recv_buf[base + 3];
    uy[dst] = recv_buf[base + 4];
    uz[dst] = recv_buf[base + 5];
    w[dst] = recv_buf[base + 6];
    if (has_ids) {
        pid[dst] = __double_as_longlong(recv_buf[base + 7]);
    }
}

} // extern "C"
"""

_MODULE_CACHE: dict[str, "cp.RawModule"] = {}


def _get_module() -> "cp.RawModule":
    """Compile and cache the CUDA RawModule keyed by ``_SOURCE``."""
    mod = _MODULE_CACHE.get(_SOURCE)
    if mod is None:
        mod = cp.RawModule(
            code=_SOURCE,
            options=("--std=c++17", "--fmad=false"),
        )
        _MODULE_CACHE[_SOURCE] = mod
    return mod


def get_kernel(name: str):
    """Retrieve a compiled ``cp.RawKernel`` from the current ``_SOURCE`` module."""
    return _get_module().get_function(name)


def _grid_1d(n: int) -> tuple[tuple[int], tuple[int]]:
    blocks = max(1, (int(n) + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
    return (blocks,), (_BLOCK_SIZE,)


def gather_fields_gpu(
    xi,
    eta,
    zeta,
    Ex_f,
    Ey_f,
    Ez_f,
    Bx_f,
    By_f,
    Bz_f,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int,
    p_stag: int,
    periodic: tuple[bool, bool, bool],
):
    """Launch ``gather_fields_kernel`` on the GPU."""
    n_p = int(xi.size)
    Ex_out = cp.empty(n_p, dtype=np.float64)
    Ey_out = cp.empty(n_p, dtype=np.float64)
    Ez_out = cp.empty(n_p, dtype=np.float64)
    Bx_out = cp.empty(n_p, dtype=np.float64)
    By_out = cp.empty(n_p, dtype=np.float64)
    Bz_out = cp.empty(n_p, dtype=np.float64)
    if n_p == 0:
        return Ex_out, Ey_out, Ez_out, Bx_out, By_out, Bz_out

    kern = get_kernel("gather_fields_kernel")
    grid, block = _grid_1d(n_p)
    kern(
        grid,
        block,
        (
            xi,
            eta,
            zeta,
            Ex_f,
            Ey_f,
            Ez_f,
            Bx_f,
            By_f,
            Bz_f,
            Ex_out,
            Ey_out,
            Ez_out,
            Bx_out,
            By_out,
            Bz_out,
            np.int32(Nx),
            np.int32(Ny),
            np.int32(Nz),
            np.int32(order),
            np.int32(p_stag),
            np.int32(1 if periodic[0] else 0),
            np.int32(1 if periodic[1] else 0),
            np.int32(1 if periodic[2] else 0),
            np.int32(n_p),
        ),
    )
    return Ex_out, Ey_out, Ez_out, Bx_out, By_out, Bz_out


def launch_gather_and_push_gpu(
    species,
    grid_map,
    E_flat,
    B_flat,
    order: int,
    p_stag: int,
    periodic: tuple[bool, bool, bool],
    dt: float,
    pusher: str,
    advance_pos: bool = True,
    E_ext_uniform: tuple[float, float, float] = (0.0, 0.0, 0.0),
    B_ext_uniform: tuple[float, float, float] = (0.0, 0.0, 0.0),
    E_ext_arrays=None,
    B_ext_arrays=None,
) -> None:
    """Launch ``gather_and_push_kernel`` for ``species`` without allocations or host sync."""
    n_launch = species._n_launch_bound
    if n_launch == 0:
        return
    if species._gpu_scratch_cap < species._capacity:
        species._ensure_gpu_scratch()

    pusher_id = PUSHER_IDS[str(pusher).strip().lower()]
    if E_ext_arrays is not None:
        Ex_ea, Ey_ea, Ez_ea = E_ext_arrays
    else:
        Ex_ea = Ey_ea = Ez_ea = 0
    if B_ext_arrays is not None:
        Bx_ea, By_ea, Bz_ea = B_ext_arrays
    else:
        Bx_ea = By_ea = Bz_ea = 0

    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0
    kern = get_kernel("gather_and_push_kernel")
    grid, block = _grid_1d(n_launch)
    kern(
        grid,
        block,
        (
            species._x,
            species._y,
            species._z,
            species._ux,
            species._uy,
            species._uz,
            species._x0_scratch if advance_pos else 0,
            species._y0_scratch if advance_pos else 0,
            species._z0_scratch if advance_pos else 0,
            species._xi0_scratch if advance_pos else 0,
            species._eta0_scratch if advance_pos else 0,
            species._zeta0_scratch if advance_pos else 0,
            species._xi1_scratch if advance_pos else 0,
            species._eta1_scratch if advance_pos else 0,
            species._zeta1_scratch if advance_pos else 0,
            E_flat,
            B_flat,
            Ex_ea,
            Ey_ea,
            Ez_ea,
            Bx_ea,
            By_ea,
            Bz_ea,
            np.float64(E_ext_uniform[0]),
            np.float64(E_ext_uniform[1]),
            np.float64(E_ext_uniform[2]),
            np.float64(B_ext_uniform[0]),
            np.float64(B_ext_uniform[1]),
            np.float64(B_ext_uniform[2]),
            np.int32(grid_map.Nx),
            np.int32(grid_map.Ny),
            np.int32(grid_map.Nz),
            np.int32(1 if grid_map._uniform[0] else 0),
            np.int32(1 if grid_map._uniform[1] else 0),
            np.int32(1 if grid_map._uniform[2] else 0),
            np.float64(grid_map._mins[0]),
            np.float64(grid_map._mins[1]),
            np.float64(grid_map._mins[2]),
            np.float64(grid_map._spans[0]),
            np.float64(grid_map._spans[1]),
            np.float64(grid_map._spans[2]),
            np.float64(grid_map._dx0[0]),
            np.float64(grid_map._dx0[1]),
            np.float64(grid_map._dx0[2]),
            grid_map.x_nodes,
            grid_map.dx,
            grid_map.y_nodes,
            grid_map.dy,
            grid_map.z_nodes,
            grid_map.dz,
            np.int32(order),
            np.int32(p_stag),
            np.int32(1 if periodic[0] else 0),
            np.int32(1 if periodic[1] else 0),
            np.int32(1 if periodic[2] else 0),
            np.float64(species.charge),
            np.float64(species.mass),
            np.float64(dt),
            np.int32(pusher_id),
            np.int32(1 if advance_pos else 0),
            np.float64(c_light),
            np.float64(_INV_C2),
            d_n_ptr,
            np.int32(n_launch),
        ),
    )


def deposit_current_esirkepov_gpu(
    xi0,
    eta0,
    zeta0,
    xi1,
    eta1,
    zeta1,
    w_arr,
    q_scale: float,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int,
    periodic: tuple[bool, bool, bool],
    Wx_out=None,
    Wy_out=None,
    Wz_out=None,
    d_n_particles=None,
    n_launch: int | None = None,
    d_cfl_error=None,
):
    """Launch ``deposit_esirkepov_kernel`` on the GPU using ``float64`` ``atomicAdd``."""
    n_total = Nx * Ny * Nz
    Wx = cp.zeros(n_total, dtype=np.float64) if Wx_out is None else Wx_out
    Wy = cp.zeros(n_total, dtype=np.float64) if Wy_out is None else Wy_out
    Wz = cp.zeros(n_total, dtype=np.float64) if Wz_out is None else Wz_out

    n_p = int(xi0.size) if n_launch is None else int(n_launch)
    if n_p == 0:
        return Wx, Wy, Wz

    d_n_ptr = 0 if d_n_particles is None else d_n_particles
    d_cfl_ptr = 0 if d_cfl_error is None else d_cfl_error
    kern = get_kernel("deposit_esirkepov_kernel")
    grid, block = _grid_1d(n_p)
    kern(
        grid,
        block,
        (
            xi0,
            eta0,
            zeta0,
            xi1,
            eta1,
            zeta1,
            w_arr,
            np.float64(q_scale),
            Wx,
            Wy,
            Wz,
            np.int32(Nx),
            np.int32(Ny),
            np.int32(Nz),
            np.int32(order),
            np.int32(1 if periodic[0] else 0),
            np.int32(1 if periodic[1] else 0),
            np.int32(1 if periodic[2] else 0),
            d_n_ptr,
            np.int32(n_p),
            d_cfl_ptr,
        ),
    )
    return Wx, Wy, Wz


def deposit_charge_gpu(
    xi,
    eta,
    zeta,
    w_arr,
    q_scale: float,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int,
    periodic: tuple[bool, bool, bool],
    q_nodes=None,
    d_n_particles=None,
    n_launch: int | None = None,
):
    """Launch ``deposit_charge_kernel`` on the GPU using ``float64`` ``atomicAdd``."""
    n_total = Nx * Ny * Nz
    if q_nodes is None:
        q_flat = cp.zeros(n_total, dtype=np.float64)
    elif q_nodes.flags.f_contiguous:
        q_flat = q_nodes.reshape(-1, order="F")
    else:
        q_flat = q_nodes.reshape(-1, order="F").copy()

    n_p = int(xi.size) if n_launch is None else int(n_launch)
    if n_p > 0:
        d_n_ptr = 0 if d_n_particles is None else d_n_particles
        kern = get_kernel("deposit_charge_kernel")
        grid, block = _grid_1d(n_p)
        kern(
            grid,
            block,
            (
                xi,
                eta,
                zeta,
                w_arr,
                np.float64(q_scale),
                q_flat,
                np.int32(Nx),
                np.int32(Ny),
                np.int32(Nz),
                np.int32(order),
                np.int32(1 if periodic[0] else 0),
                np.int32(1 if periodic[1] else 0),
                np.int32(1 if periodic[2] else 0),
                d_n_ptr,
                np.int32(n_p),
            ),
        )

    out = q_flat.reshape((Nx, Ny, Nz), order="F")
    if q_nodes is not None and not q_nodes.flags.f_contiguous:
        q_nodes[...] = out
        return q_nodes
    return q_nodes if q_nodes is not None else out


def launch_boundary_prepare_paths_gpu(
    boundaries,
    species,
    order: int,
    q_bg_nodes=None,
):
    """Launch ``boundary_prepare_paths_kernel`` on the GPU without host sync."""
    n_launch = species._n_launch_bound
    if n_launch == 0:
        return (
            species._xi1_scratch,
            species._eta1_scratch,
            species._zeta1_scratch,
            species._keep_mask_scratch,
        )

    if species._gpu_scratch_cap < species._capacity or species._xi1_dep_scratch is None:
        species._ensure_gpu_scratch(need_boundary=True)
    if boundaries._col_types_gpu is None and boundaries.collectors:
        boundaries._ensure_gpu_buffers()

    if q_bg_nodes is not None and q_bg_nodes.flags.f_contiguous:
        q_bg_ptr = q_bg_nodes.reshape(-1, order="F")
    else:
        q_bg_ptr = 0

    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0
    gmap = boundaries.grid_map
    kern = get_kernel("boundary_prepare_paths_kernel")
    grid, block = _grid_1d(n_launch)
    kern(
        grid,
        block,
        (
            species._x0_scratch,
            species._y0_scratch,
            species._z0_scratch,
            species._x,
            species._y,
            species._z,
            species._xi0_scratch,
            species._eta0_scratch,
            species._zeta0_scratch,
            species._xi1_scratch,
            species._eta1_scratch,
            species._zeta1_scratch,
            species._ux,
            species._uy,
            species._uz,
            species._w,
            np.float64(species.charge),
            np.float64(species.mass),
            np.float64(_INV_C2),
            species._xi1_dep_scratch,
            species._eta1_dep_scratch,
            species._zeta1_dep_scratch,
            species._keep_mask_scratch,
            np.int32(gmap.Nx),
            np.int32(gmap.Ny),
            np.int32(gmap.Nz),
            np.int32(1 if boundaries.periodic[0] else 0),
            np.int32(1 if boundaries.periodic[1] else 0),
            np.int32(1 if boundaries.periodic[2] else 0),
            np.float64(boundaries.xi_min[0]),
            np.float64(boundaries.xi_max[0]),
            np.float64(boundaries.xi_min[1]),
            np.float64(boundaries.xi_max[1]),
            np.float64(boundaries.xi_min[2]),
            np.float64(boundaries.xi_max[2]),
            boundaries._solid_mask_f_u8 if boundaries.solid_mask is not None else 0,
            np.int32(len(boundaries.collectors)),
            boundaries._col_types_gpu if boundaries.collectors else 0,
            boundaries._col_params_gpu if boundaries.collectors else 0,
            np.int32(order),
            q_bg_ptr,
            boundaries._d_absorbed_charge,
            boundaries._d_absorbed_energy,
            boundaries._d_absorbed_count,
            d_n_ptr,
            np.int32(n_launch),
        ),
    )
    boundaries._gpu_stats_dirty = True
    return (
        species._xi1_dep_scratch,
        species._eta1_dep_scratch,
        species._zeta1_dep_scratch,
        species._keep_mask_scratch,
    )


def launch_periodic_wrap_gpu(boundaries, species) -> None:
    """Launch ``periodic_wrap_kernel`` on the GPU without host sync."""
    if not any(boundaries.periodic):
        return
    n_launch = species._n_launch_bound
    if n_launch == 0:
        return
    gmap = boundaries.grid_map
    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0
    kern = get_kernel("periodic_wrap_kernel")
    grid, block = _grid_1d(n_launch)
    kern(
        grid,
        block,
        (
            species._x,
            species._y,
            species._z,
            np.int32(1 if boundaries.periodic[0] else 0),
            np.int32(1 if boundaries.periodic[1] else 0),
            np.int32(1 if boundaries.periodic[2] else 0),
            np.float64(gmap._mins[0]),
            np.float64(gmap._mins[1]),
            np.float64(gmap._mins[2]),
            np.float64(gmap._spans[0]),
            np.float64(gmap._spans[1]),
            np.float64(gmap._spans[2]),
            d_n_ptr,
            np.int32(n_launch),
        ),
    )


def compact_species_gpu(species, keep_mask_i32) -> None:
    """Three-pass GPU stream compaction without device-to-host synchronization."""
    n_launch = species._n_launch_bound
    if n_launch == 0:
        return
    if (
        species._gpu_scratch_cap < species._capacity
        or getattr(species, "_x_tmp", None) is None
    ):
        species._ensure_gpu_scratch(need_tmp=True)

    n_blocks = max(1, (n_launch + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0

    k_blocks = get_kernel("compact_scan_blocks_kernel")
    k_totals = get_kernel("compact_scan_totals_kernel")
    k_scatter = get_kernel("compact_scatter_kernel")

    k_blocks(
        (n_blocks,),
        (_BLOCK_SIZE,),
        (
            keep_mask_i32,
            species._scan_dst_idx,
            species._scan_block_sums,
            d_n_ptr,
            np.int32(n_launch),
        ),
    )
    k_totals(
        (1,),
        (_BLOCK_SIZE,),
        (
            species._scan_block_sums,
            species._scan_block_offsets,
            species._d_n_next,
            np.int32(n_blocks),
        ),
    )
    k_scatter(
        (n_blocks,),
        (_BLOCK_SIZE,),
        (
            keep_mask_i32,
            species._scan_dst_idx,
            species._scan_block_offsets,
            species._x,
            species._y,
            species._z,
            species._ux,
            species._uy,
            species._uz,
            species._w,
            species._id if species.track_ids else 0,
            species._x_tmp,
            species._y_tmp,
            species._z_tmp,
            species._ux_tmp,
            species._uy_tmp,
            species._uz_tmp,
            species._w_tmp,
            species._id_tmp if species.track_ids else 0,
            np.int32(1 if species.track_ids else 0),
            d_n_ptr,
            np.int32(n_launch),
        ),
    )

    species._x, species._x_tmp = species._x_tmp, species._x
    species._y, species._y_tmp = species._y_tmp, species._y
    species._z, species._z_tmp = species._z_tmp, species._z
    species._ux, species._ux_tmp = species._ux_tmp, species._ux
    species._uy, species._uy_tmp = species._uy_tmp, species._uy
    species._uz, species._uz_tmp = species._uz_tmp, species._uz
    species._w, species._w_tmp = species._w_tmp, species._w
    if species.track_ids:
        species._id, species._id_tmp = species._id_tmp, species._id

    species._d_n_particles, species._d_n_next = (
        species._d_n_next,
        species._d_n_particles,
    )
    species._n_dirty_gpu = True


def sort_species_by_cell_gpu(species, grid_map) -> None:
    """Sort active particles on the GPU by linear 3D cell index without host sync.

    Notes
    -----
    ``cp.argsort(..., kind="stable")`` executes on the device stream without
    synchronizing the host, and sub-allocates temporary permutation and
    radix-sort workspace buffers from the CuPy memory pool cache (pre-warmed
    in ``Species._ensure_gpu_scratch()`` so ``pool.total_bytes()`` does not
    grow).
    """
    n_launch = species._n_launch_bound
    if n_launch <= 1:
        return
    if (
        species._gpu_scratch_cap < species._capacity
        or getattr(species, "_x_tmp", None) is None
        or species._cell_idx_scratch is None
    ):
        species._ensure_gpu_scratch(need_tmp=True, need_sort=True)

    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0
    k_cell = get_kernel("compute_cell_index_kernel")
    k_perm = get_kernel("permute_particles_kernel")
    grid, block = _grid_1d(n_launch)

    k_cell(
        grid,
        block,
        (
            species._x,
            species._y,
            species._z,
            species._cell_idx_scratch,
            np.int32(grid_map.Nx),
            np.int32(grid_map.Ny),
            np.int32(grid_map.Nz),
            np.int32(1 if grid_map._uniform[0] else 0),
            np.int32(1 if grid_map._uniform[1] else 0),
            np.int32(1 if grid_map._uniform[2] else 0),
            np.float64(grid_map._mins[0]),
            np.float64(grid_map._mins[1]),
            np.float64(grid_map._mins[2]),
            np.float64(grid_map._spans[0]),
            np.float64(grid_map._spans[1]),
            np.float64(grid_map._spans[2]),
            np.float64(grid_map._dx0[0]),
            np.float64(grid_map._dx0[1]),
            np.float64(grid_map._dx0[2]),
            grid_map.x_nodes,
            grid_map.dx,
            grid_map.y_nodes,
            grid_map.dy,
            grid_map.z_nodes,
            grid_map.dz,
            d_n_ptr,
            np.int32(n_launch),
        ),
    )

    perm = cp.argsort(species._cell_idx_scratch[:n_launch], kind="stable")
    k_perm(
        grid,
        block,
        (
            perm,
            species._x,
            species._y,
            species._z,
            species._ux,
            species._uy,
            species._uz,
            species._w,
            species._id if species.track_ids else 0,
            species._x_tmp,
            species._y_tmp,
            species._z_tmp,
            species._ux_tmp,
            species._uy_tmp,
            species._uz_tmp,
            species._w_tmp,
            species._id_tmp if species.track_ids else 0,
            np.int32(1 if species.track_ids else 0),
            np.int32(n_launch),
        ),
    )

    species._x, species._x_tmp = species._x_tmp, species._x
    species._y, species._y_tmp = species._y_tmp, species._y
    species._z, species._z_tmp = species._z_tmp, species._z
    species._ux, species._ux_tmp = species._ux_tmp, species._ux
    species._uy, species._uy_tmp = species._uy_tmp, species._uy
    species._uz, species._uz_tmp = species._uz_tmp, species._uz
    species._w, species._w_tmp = species._w_tmp, species._w
    if species.track_ids:
        species._id, species._id_tmp = species._id_tmp, species._id


def compute_centred_B_gpu(H_prev, H_curr, B_out) -> None:
    """Launch ``compute_centred_B_kernel`` on the GPU.

    Evaluates ``B_out = 0.5 * mu_0 * (H_prev + H_curr)`` [T], assuming vacuum
    permeability ``mu = mu_0`` in the active particle domain (same as the CPU
    path).
    """
    n3 = int(B_out.size)
    kern = get_kernel("compute_centred_B_kernel")
    grid, block = _grid_1d(n3)
    kern(
        grid,
        block,
        (H_prev, H_curr, B_out, np.float64(0.5 * mu_0), np.int32(n3)),
    )


def finalize_j_delta_gpu(
    Wx, Wy, Wz, itA, J_particles, J_prev, J_solver, inv_dt: float, N: int
) -> None:
    """Launch ``finalize_j_delta_kernel`` on the GPU."""
    kern = get_kernel("finalize_j_delta_kernel")
    grid, block = _grid_1d(N)
    kern(
        grid,
        block,
        (
            Wx,
            Wy,
            Wz,
            itA,
            J_particles,
            J_prev,
            J_solver,
            np.float64(inv_dt),
            np.int32(N),
        ),
    )


def classify_and_pack_migration_gpu(
    species,
    has_lo: bool,
    has_hi: bool,
    z_own_min: float,
    z_own_max: float,
    z_ghost_lo: float,
    z_ghost_hi: float,
    d_counts,
    d_send_lo,
    d_send_hi,
    pack_cap: int,
) -> None:
    """Classify active GPU particles and pack leavers into ``d_send_lo`` and ``d_send_hi``."""
    n_launch = species._n_launch_bound
    d_counts.fill(0)
    if n_launch == 0:
        return
    if species._gpu_scratch_cap < species._capacity:
        species._ensure_gpu_scratch()
    d_n_ptr = species._d_n_particles if species._n_dirty_gpu else 0
    kern = get_kernel("classify_and_pack_migration_kernel")
    grid, block = _grid_1d(n_launch)
    kern(
        grid,
        block,
        (
            species._x,
            species._y,
            species._z,
            species._ux,
            species._uy,
            species._uz,
            species._w,
            species._id if species.track_ids else 0,
            np.int32(1 if species.track_ids else 0),
            np.int32(1 if has_lo else 0),
            np.int32(1 if has_hi else 0),
            np.float64(z_own_min),
            np.float64(z_own_max),
            np.float64(z_ghost_lo),
            np.float64(z_ghost_hi),
            species._keep_mask_scratch,
            d_counts,
            d_send_lo,
            d_send_hi,
            np.int32(pack_cap),
            d_n_ptr,
            np.int32(n_launch),
        ),
    )


def unpack_migration_gpu(
    species,
    d_recv_buf,
    n_recv: int,
    dst_offset: int,
) -> None:
    """Unpack ``n_recv`` migrated 8-word particle records into ``species`` at ``dst_offset``."""
    if n_recv <= 0:
        return
    kern = get_kernel("unpack_migration_kernel")
    grid, block = _grid_1d(n_recv)
    kern(
        grid,
        block,
        (
            d_recv_buf,
            np.int32(n_recv),
            np.int32(dst_offset),
            species._x,
            species._y,
            species._z,
            species._ux,
            species._uy,
            species._uz,
            species._w,
            species._id if species.track_ids else 0,
            np.int32(1 if species.track_ids else 0),
        ),
    )
