# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification problems for `HybridFIT3D` (frequency domain, semi-discrete).

All three use the semi-discrete operator ``K^T W K - w^2 M`` of the leapfrog
(``dt -> 0``) with the exp(-i w t) convention; complex coordinate stretching
``s = 1 + i sigma0 (d / T)^3`` in the Cartesian cells only.

* `waveguide_reflection`: TE10 mode of a PEC rectangular guide crossing a
  tetrahedral box patch (vacuum). |R| of the patch from a two-wave fit with
  the exact discrete propagation constant of the Cartesian guide.
* `cavity_modes`: resonances of a PEC box cavity with a vacuum patch against
  the analytic ``c/2 sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``.
* `pec_sphere_scattering`: scattered-field formulation for a PEC sphere
  (Dirichlet data minus the incident edge voltages on the sphere edges),
  relative error of the scattered edge voltages against the Mie series.
"""

import numpy as np
from scipy.constants import c as C0

from .mesh import delaunay_box_mesh, kuhn_box_mesh
from .mie_pec import pec_sphere_field, plane_wave_line_integrals, segment_line_integrals
from .solver import HybridFIT3D

__all__ = [
    "waveguide_reflection",
    "cavity_modes",
    "cavity_analytic",
    "pec_sphere_scattering",
    "stretch_functions",
]


def stretch_functions(lo_phys, hi_phys, T, sigma0=8.0, axes=(0, 1, 2)):
    """
    Complex stretching factors (exp(-i w t)): ``s_a = 1 + i sigma0 (d/T)^3``
    outside ``[lo_phys[a], hi_phys[a]]`` on each axis in `axes`.

    Returns (edge_scale, face_scale) for `HybridFIT3D`: an edge of
    direction a gets ``s_b s_c / s_a``, a face of normal a ``s_a / (s_b s_c)``.
    """
    lo_phys, hi_phys = np.asarray(lo_phys, float), np.asarray(hi_phys, float)

    def S(X):
        out = np.ones(X.shape, complex)
        for a in axes:
            d = np.maximum(lo_phys[a] - X[:, a], 0) + np.maximum(X[:, a] - hi_phys[a], 0)
            out[:, a] = 1 + 1j * sigma0 * (d / T) ** 3
        return out

    def edge_scale(mid, direction):
        s = S(mid)
        return s.prod(1) / s[np.arange(len(direction)), direction] ** 2

    def face_scale(cen, normal):
        s = S(cen)
        return s[np.arange(len(normal)), normal] ** 2 / s.prod(1)

    return edge_scale, face_scale


def _tet_patch(kind, xs, ys, zs, h, r=1, **kw):
    if kind == "kuhn":
        return kuhn_box_mesh(xs, ys, zs, r=r)
    if kind == "delaunay":
        return delaunay_box_mesh(xs, ys, zs, h=h, r=r, **kw)
    raise ValueError("kind must be 'kuhn' or 'delaunay'")


def waveguide_reflection(cells_per_a, patch="kuhn", r=1, h_tet=None, wavelength_a=1.4,
                         mutate=None, return_solver=False):  # fmt: skip
    """
    |R| of the TE10 mode at a vacuum tetrahedral patch in a PEC guide.

    Guide: y in [0, a], z in [0, a/2], a = 1 m, along x in [0, 9a] with
    stretched absorbers of length 2a at both ends; source sheet J_z ~
    sin(pi y / a) at x = 2.25a; fit window [2.5a, 4a]; patch box
    x in [5.5a, 6a], one cell from the y and z walls.

    Parameters
    ----------
    cells_per_a : int
        Cartesian cells across the broad wall (multiple of 4).
    patch : {'kuhn', 'delaunay', None}
        Tetrahedral patch type; None = pure Cartesian (fit/absorber floor).
    r : int
        Refinement of the patch surface (Kuhn: of the whole patch).
    h_tet : float, optional
        Delaunay target edge length [m] (default: the Cartesian h).
    wavelength_a : float
        Free-space wavelength in units of a (TE10 only: between 1 and 2).
    mutate : passed to `HybridFIT3D`.

    Returns
    -------
    float (|R|), or (|R|, solver)
    """
    N = int(cells_per_a)
    if N % 4:
        raise ValueError("cells_per_a must be a multiple of 4")
    A = 1.0
    h = A / N
    xn = h * np.arange(9 * N + 1)
    yn = h * np.arange(N + 1)
    zn = h * np.arange(N // 2 + 1)
    T = 2.0 * A
    es, fs = stretch_functions([T, -1, -1], [xn[-1] - T, 9, 9], T, sigma0=6.0, axes=(0,))
    box = None
    tm = None
    if patch is not None:
        i0, i1 = int(round(5.5 * N)), int(round(6.0 * N))
        box = ((i0, i1), (1, N - 1), (1, N // 2 - 1))
        tm = _tet_patch(patch, xn[i0 : i1 + 1], yn[1:N], zn[1 : N // 2], h_tet or h, r=r)
    s = HybridFIT3D(xn, yn, zn, box=box, tet_mesh=tm, edge_scale=es, face_scale=fs,
                    mutate=mutate)  # fmt: skip
    lam = wavelength_a * A
    k = 2 * np.pi / lam
    w = k * C0
    blk = s.block
    # source sheet on z-edges at x node i_s
    i_s = int(round(2.25 * N))
    ijk = blk.edge_grid(2)
    zsel = (ijk[0] == i_s).ravel()
    yj = yn[ijk[1].ravel()]
    i_all = np.zeros(s.nEc + s.nEt, complex)
    idx = blk.edge_off[2] + np.flatnonzero(zsel)
    i_all[idx] = np.sin(np.pi * yj[zsel] / A)
    e = s.solve_frequency(w, rhs=1j * w * (s.P.T @ i_all))
    ea = (s.P @ e)[: s.nEc]
    ez = ea[blk.edge_off[2] : blk.edge_off[3]].reshape(blk.edge_shape[2])
    prof = np.sin(np.pi * yn / A)
    amp = np.einsum("ijk,j->i", ez, prof)
    xi = xn
    win = (xi >= 2.5 * A - 1e-9) & (xi <= 4.0 * A + 1e-9)
    ky = 2 / h * np.sin(np.pi * h / (2 * A))
    kx = 2 / h * np.arcsin(np.sqrt(k**2 - ky**2) * h / 2)
    X = xi[win]
    Mfit = np.stack([np.exp(1j * kx * X), np.exp(-1j * kx * X)], 1)
    (a_inc, b_ref), *_ = np.linalg.lstsq(Mfit, amp[win], rcond=None)
    R = float(abs(b_ref / a_inc))
    return (R, s) if return_solver else R


def cavity_analytic(dims, n_modes):
    """Lowest `n_modes` resonances [Hz] of a PEC box cavity (with multiplicity)."""
    a, b, d = dims
    f = []
    for m in range(8):
        for n in range(8):
            for p in range(8):
                nz = (m > 0) + (n > 0) + (p > 0)
                if nz >= 2:  # TE and TM when all three indices are non-zero
                    fr = C0 / 2 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
                    f += [fr] * (nz - 1)
    return np.sort(f)[:n_modes]


def cavity_modes(cells_per_m, patch="kuhn", r=1, n_modes=6, dims=(1.0, 0.75, 0.5),
                 mutate=None, h_tet=None, seed=0):  # fmt: skip
    """
    Resonances [Hz] of a PEC box cavity with a vacuum tetrahedral box patch
    (the middle half of each side, rounded to whole cells), and the analytic
    values. ``patch=None``: pure Cartesian.
    """
    h = 1.0 / cells_per_m
    lines = [h * np.arange(int(round(L / h)) + 1) for L in dims]
    box, tm = None, None
    if patch is not None:
        bx = []
        for v in lines:
            n = len(v) - 1
            i0, i1 = max(1, int(round(n / 4))), min(n - 1, int(round(3 * n / 4)))
            bx.append((i0, i1))
        box = tuple(bx)
        tm = _tet_patch(patch, *[v[i0 : i1 + 1] for v, (i0, i1) in zip(lines, bx)],
                        h_tet or h, r=r, **({"seed": seed} if patch == "delaunay" else {}))  # fmt: skip
    s = HybridFIT3D(*lines, box=box, tet_mesh=tm, mutate=mutate, dt=1.0)
    f_ex = cavity_analytic(dims, n_modes)
    f = s.eigenfrequencies(n_modes, 0.9 * f_ex[0], 1.15 * f_ex[-1])
    return f, f_ex, s


def pec_sphere_scattering(cells_per_wavelength, method="hybrid", ka=2.0, h_tet=None,
                          r=1, gap=0.25, T=0.6, sigma0=8.0, box_margin=0.12,
                          return_solver=False):  # fmt: skip
    """
    Plane wave ``x_hat exp(i k z)`` on a PEC sphere (vacuum, wavelength 1 m).

    Parameters
    ----------
    cells_per_wavelength : int
        Cartesian cells per wavelength N (h = 1/N m).
    method : {'hybrid', 'staircase', 'tet'}
        'hybrid': tetrahedral box ``|x|_inf <= B`` (B = a + box_margin rounded
        up to cells) around the sphere; 'staircase': pure Cartesian with PEC
        cells whose centre is inside the sphere; 'tet': tetrahedra in the
        whole physical region (Cartesian cells only in the absorber).
    ka : float
        Size parameter.
    h_tet : float, optional
        Tetrahedral target edge [m] (default h).
    gap, T : float
        Physical margin beyond the box and absorber thickness [wavelengths].

    Returns
    -------
    dict
        ``err_near``: relative L2 error of the scattered edge voltages on all
        free edges with ``a + h < |r|`` and ``|x|_inf < B``;
        ``err_far``: same on edges with ``B + h/4 < |x|_inf < B + gap - h/4``;
        ``n_free``, ``n_tets``.
    """
    N = int(cells_per_wavelength)
    h = 1.0 / N
    k = 2 * np.pi
    a = ka / k
    ht = h_tet or h
    nb = int(np.ceil((a + max(box_margin, 2.2 * ht)) / h))
    ng = int(np.ceil(gap / h))
    nt = int(np.ceil(T / h))
    nh = nb + ng + nt
    v = h * np.arange(-nh, nh + 1)
    Lp = (nb + ng) * h
    es, fs = stretch_functions([-Lp] * 3, [Lp] * 3, nt * h, sigma0=sigma0)
    box, tm, pec = None, None, None
    if method == "hybrid":
        i0, i1 = nh - nb, nh + nb
    elif method == "tet":
        i0, i1 = nt, 2 * nh - nt
    elif method == "staircase":
        i0 = None

        def pec(X, Y, Z):
            return X**2 + Y**2 + Z**2 < a**2

    else:
        raise ValueError("method must be 'hybrid', 'staircase' or 'tet'")
    if i0 is not None:
        box = ((i0, i1),) * 3
        tm = delaunay_box_mesh(v[i0 : i1 + 1], v[i0 : i1 + 1], v[i0 : i1 + 1], h=ht,
                               r=r, sphere=((0.0, 0.0, 0.0), a), hole=True)  # fmt: skip
    s = HybridFIT3D(v, v, v, box=box, tet_mesh=tm, pec=pec, edge_scale=es,
                    face_scale=fs)  # fmt: skip
    # all-edge end points
    mid, L, d = s.edge_mid, s.edge_len, s.edge_dir
    ar = np.arange(len(d))
    pc, qc = mid.copy(), mid.copy()
    pc[ar, d] -= L / 2
    qc[ar, d] += L / 2
    if tm is not None:
        pt, qt = tm.nodes[tm.edges[:, 0]], tm.nodes[tm.edges[:, 1]]
        p_all, q_all = np.vstack([pc, pt]), np.vstack([qc, qt])
    else:
        p_all, q_all = pc, qc
    if method == "staircase":
        fixed = np.flatnonzero(s.pec_edges_c)
    else:
        fixed = s.nEc + s.tet_fixed
    vinc = plane_wave_line_integrals(k, p_all[fixed], q_all[fixed])
    e = s.solve_frequency(k * C0, fixed=(fixed, -vinc))
    e_all = s.P @ e
    fe = s.free_edges
    m_all = 0.5 * (p_all + q_all)
    rr = np.linalg.norm(m_all[fe], axis=1)
    rinf = np.abs(m_all[fe]).max(1)
    B = nb * h
    near = (rr > a + h) & (rinf < B)
    far = (rinf > B + 0.25 * h) & (rinf < B + ng * h - 0.25 * h)
    out = {"n_free": int(s.n_free), "n_tets": int(tm.n_cells) if tm is not None else 0,
           "h": h, "a": a, "B": B}  # fmt: skip

    def field(X):
        return pec_sphere_field(X, k, a, "scattered")

    for name, sel in (("near", near), ("far", far)):
        idx = fe[sel]
        ref = segment_line_integrals(field, p_all[idx], q_all[idx])
        num = e_all[idx]
        out[f"err_{name}"] = float(np.linalg.norm(num - ref) / np.linalg.norm(ref))
        out[f"n_{name}"] = int(idx.size)
    if return_solver:
        out["solver"] = s
    return out
