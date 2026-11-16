"""
Brief W4-B Part 2: Perturbed-box ``TE101`` convergence order factorial study
reconciling the study order (``D = 1.795576``, quoted ``1.80``) and the legacy
two-level test order (``1.999214``).

Factorial axes:
- ``F1``: isotropic ``(n, n, n)``, ``n in {4, 6, 8, 12, 16, 24}`` (``--quick``: ``n <= 16``), ``amp = 0.3`` (cell mode)
- ``F2``: aspect-matched ``(4k, 3k, 4k)``, ``k in {1, 2, 3, 4, 6}`` (``--quick``: ``k <= 2``), ``amp = 0.3`` (cell mode)
- ``F3``: fixed physical map (displacement amplitude fixed in metres at the ``F1`` ``n = 6`` value), ``n in {6, 8, 12, 16, 24}`` (``--quick``: ``n <= 16``)
- ``amp = 0`` Cartesian controls for ``F1`` and ``F2`` and exact semi-discrete Yee dispersion comparison
- Error decomposition ``e(amp) = e_Yee + e_dist`` where ``e_Yee = e(amp=0)`` and ``e_dist = e(amp) - e(amp=0)``

Usage::

    python studies/curvilinear3d/perturbed_box_order.py [--quick] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.constants import c as C0
from scipy.optimize import minimize_scalar

from wakis.curvilinear import CurvilinearFIT3D, HexBlock3D
from wakis.curvilinear.mappings3d import box_te101_omega, perturbed_box_nodes

F1_NS_FULL = (4, 6, 8, 12, 16, 24)
F1_NS_QUICK = (4, 6, 8, 12, 16)
F1_STUDY_WINDOW = (6, 8, 12, 16)

F2_KS_FULL = (1, 2, 3, 4, 6)
F2_KS_QUICK = (1, 2)

F3_NS_FULL = (6, 8, 12, 16, 24)
F3_NS_QUICK = (6, 8, 12, 16)


def fit_order(ns, rel_errors):
    """
    Least-squares convergence order ``p`` in ``|e(n)| ~ C n^{-p}``.

    Parameters
    ----------
    ns : sequence of float
        Resolution parameter values ``n``.
    rel_errors : sequence of float
        Relative errors (signed or unsigned).

    Returns
    -------
    float
        Estimated convergence order ``-d(ln |e|) / d(ln n)``, or ``nan`` if
        fewer than two positive finite values are present.
    """
    ns_arr = np.asarray(ns, dtype=float)
    err_arr = np.abs(np.asarray(rel_errors, dtype=float))
    mask = np.isfinite(err_arr) & (err_arr > 0.0) & (ns_arr > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    log_n = np.log(ns_arr[mask])
    log_e = np.log(err_arr[mask])
    slope, _ = np.polyfit(log_n, log_e, 1)
    return float(-slope)


def local_orders(ns, rel_errors):
    """
    Two-point local convergence orders between consecutive resolutions.

    Parameters
    ----------
    ns : sequence of float
        Resolution parameter values ``n``.
    rel_errors : sequence of float
        Relative errors (signed or unsigned).

    Returns
    -------
    list of float
        Local orders ``-ln(|e_{k+1}| / |e_k|) / ln(n_{k+1} / n_k)``.
    """
    ns_arr = np.asarray(ns, dtype=float)
    err_arr = np.abs(np.asarray(rel_errors, dtype=float))
    out = []
    for i in range(len(ns_arr) - 1):
        e0, e1 = err_arr[i], err_arr[i + 1]
        n0, n1 = ns_arr[i], ns_arr[i + 1]
        if e0 > 0.0 and e1 > 0.0 and n0 > 0.0 and n1 > 0.0 and n1 != n0:
            out.append(float(-np.log(e1 / e0) / np.log(n1 / n0)))
        else:
            out.append(float("nan"))
    return out


def two_level_order(n1, e1, n2, e2):
    """
    Two-level convergence order ``ln(|e1| / |e2|) / ln(n2 / n1)``.
    """
    return float(np.log(abs(e1) / abs(e2)) / np.log(float(n2) / float(n1)))


def two_term_fit(ns, signed_errors, q_bounds=(2.05, 6.0)):
    """
    Fit the two-term asymptotic model ``e(N) = C2 * N^{-2} + Cq * N^{-q}``
    to signed relative errors ``signed_errors``.

    Parameters
    ----------
    ns : sequence of float
        Resolution parameter values ``N``.
    signed_errors : sequence of float
        Signed relative errors ``(omega_h - omega) / omega``.
    q_bounds : tuple of float, optional
        Search interval for the higher-order exponent ``q`` (default
        ``(2.05, 6.0)``).

    Returns
    -------
    dict
        Fitted parameters ``C2``, ``Cq``, ``q``, and ``rms_residual``.
    """
    ns_arr = np.asarray(ns, dtype=float)
    e_arr = np.asarray(signed_errors, dtype=float)
    if ns_arr.size < 3:
        return {
            "C2": float("nan"),
            "Cq": float("nan"),
            "q": float("nan"),
            "rms_residual": float("nan"),
        }

    def _solve_coeffs(q_val):
        design = np.column_stack([ns_arr ** (-2.0), ns_arr ** (-float(q_val))])
        coeffs, _, _, _ = np.linalg.lstsq(design, e_arr, rcond=None)
        res = e_arr - design @ coeffs
        rms = float(np.sqrt(np.mean(res * res)))
        return coeffs, rms

    res_opt = minimize_scalar(
        lambda q_val: _solve_coeffs(q_val)[1],
        bounds=q_bounds,
        method="bounded",
    )
    q_best = float(res_opt.x)
    coeffs_best, rms_best = _solve_coeffs(q_best)
    return {
        "C2": float(coeffs_best[0]),
        "Cq": float(coeffs_best[1]),
        "q": q_best,
        "rms_residual": rms_best,
    }


def yee_te101_omega(nx, nz, a=1.0, d=0.8):
    """
    Exact semi-discrete Yee ``TE101`` angular eigenfrequency [rad/s] on an
    unperturbed ``nx x ny x nz`` Cartesian grid over ``[0, a] x [0, b] x [0, d]``:

    .. math::

        \\omega_h^2 = c_0^2 \\left[
            \\left(\\frac{2}{h_x}\\sin\\frac{\\pi h_x}{2 a}\\right)^2
            + \\left(\\frac{2}{h_z}\\sin\\frac{\\pi h_z}{2 d}\\right)^2
        \\right].

    Parameters
    ----------
    nx, nz : int
        Cell counts along ``x`` and ``z``.
    a, d : float, optional
        Box lengths along ``x`` and ``z`` [m] (defaults 1.0, 0.8).

    Returns
    -------
    float
        Semi-discrete Yee ``TE101`` angular frequency ``omega_h`` [rad/s].
    """
    hx = float(a) / int(nx)
    hz = float(d) / int(nz)
    kx_h = (2.0 / hx) * np.sin(0.5 * np.pi * hx / float(a))
    kz_h = (2.0 / hz) * np.sin(0.5 * np.pi * hz / float(d))
    return float(C0 * np.hypot(kx_h, kz_h))


def _make_perturbed_box_nodes(
    nx,
    ny,
    nz,
    amp,
    amp_mode="cell",
    a=1.0,
    b=0.7,
    d=0.8,
    n_ref=6,
):
    """
    Construct perturbed-box node coordinates [m] in either ``"cell"`` mode
    (displacement proportional to local cell size ``h_i``) or ``"fixed"`` mode
    (physical displacement amplitude fixed in metres at the ``n = n_ref`` value).
    """
    if amp_mode == "cell":
        return perturbed_box_nodes(nx, ny, nz, a=a, b=b, d=d, amp=amp)
    if amp_mode == "fixed":
        X = perturbed_box_nodes(nx, ny, nz, a=a, b=b, d=d, amp=0.0)
        if float(amp) != 0.0:
            env = (
                np.sin(np.pi * X[..., 0] / float(a))
                * np.sin(np.pi * X[..., 1] / float(b))
                * np.sin(np.pi * X[..., 2] / float(d))
            )
            X[..., 0] += float(amp) * (float(a) / float(n_ref)) * env
            X[..., 1] += float(amp) * (float(b) / float(n_ref)) * env
            X[..., 2] += float(amp) * (float(d) / float(n_ref)) * env
        return X
    raise ValueError(f"amp_mode must be 'cell' or 'fixed', got {amp_mode!r}")


def te101_errors(
    counts,
    amp,
    amp_mode="cell",
    a=1.0,
    b=0.7,
    d=0.8,
    return_omegas=False,
    pec_reconstruction="constrained",
):
    """
    Compute signed relative ``TE101`` eigenfrequency errors
    ``(omega_h - omega_ana) / omega_ana`` across mesh cell counts ``counts``.

    Parameters
    ----------
    counts : sequence of int or tuple of (int, int, int)
        Mesh cell counts ``(nx, ny, nz)`` (an integer ``n`` is expanded to
        ``(n, n, n)``).
    amp : float
        Perturbation amplitude parameter (e.g. ``0.3`` or ``0.0``).
    amp_mode : {"cell", "fixed"}, optional
        In ``"cell"`` mode (default), node displacement scales with local cell
        spacing ``amp * h_i``. In ``"fixed"`` mode, physical displacement is
        fixed in metres at ``amp * L_i / 6``.
    a, b, d : float, optional
        Box side lengths [m] (defaults 1.0, 0.7, 0.8).
    return_omegas : bool, optional
        If True, return ``(signed_errors, omegas_h)``.
    pec_reconstruction : {"constrained", "drop_only"}, optional
        PEC boundary octant reconstruction rule passed to
        :class:`CurvilinearFIT3D` (default ``"constrained"``).

    Returns
    -------
    ndarray or tuple of ndarray
        Signed relative errors ``(omega_h - omega_ana) / omega_ana`` (and
        discrete ``omega_h`` [rad/s] if ``return_omegas=True``).
    """
    omega_ana = box_te101_omega(a=a, d=d)
    errs = []
    omegas_h = []
    for c_item in counts:
        if isinstance(c_item, (int, np.integer)):
            nx = ny = nz = int(c_item)
        else:
            nx, ny, nz = (int(x) for x in c_item)
        X = _make_perturbed_box_nodes(
            nx, ny, nz, amp=amp, amp_mode=amp_mode, a=a, b=b, d=d
        )
        sol = CurvilinearFIT3D(HexBlock3D(X), _pec_reconstruction=pec_reconstruction)
        w = sol.cavity_frequencies(k=2)
        w_h = float(w[0])
        omegas_h.append(w_h)
        errs.append((w_h - omega_ana) / omega_ana)

    err_arr = np.asarray(errs, dtype=float)
    if return_omegas:
        return err_arr, np.asarray(omegas_h, dtype=float)
    return err_arr


def compute_d_order(
    a=1.0,
    b=0.7,
    d=0.8,
    amp=0.3,
    pec_reconstruction="constrained",
):
    """
    Compute the canonical W4-B definition **D**: least-squares slope of
    ``ln |e|`` vs ``ln n`` on family ``F1`` ``(n, n, n)``,
    ``n in {6, 8, 12, 16}``, ``amp = 0.3`` (cell mode).

    Returns
    -------
    d_order : float
        Measured convergence order ``D``.
    ns : list of int
        Resolutions ``[6, 8, 12, 16]``.
    signed_errs : ndarray
        Signed relative errors at ``ns``.
    """
    ns = list(F1_STUDY_WINDOW)
    signed_errs = te101_errors(
        ns,
        amp=amp,
        amp_mode="cell",
        a=a,
        b=b,
        d=d,
        pec_reconstruction=pec_reconstruction,
    )
    d_order = fit_order(ns, signed_errs)
    return d_order, ns, signed_errs


def _summarize_family(
    name,
    ns,
    counts,
    e_pert,
    e_zero,
    omegas_zero=None,
    a=1.0,
    d=0.8,
):
    """
    Build summary dictionary for a factorial family including ``e(amp)``,
    ``e(amp=0)``, and ``e_dist = e(amp) - e(amp=0)``.
    """
    ns_list = [float(x) for x in ns]
    e_pert_arr = np.asarray(e_pert, dtype=float)
    e_zero_arr = np.asarray(e_zero, dtype=float)
    e_dist_arr = e_pert_arr - e_zero_arr

    yee_rel_mismatches = []
    omegas_num_list = []
    omegas_yee_list = []
    if omegas_zero is not None:
        for c_item, w_num in zip(counts, omegas_zero):
            nx = (
                int(c_item) if isinstance(c_item, (int, np.integer)) else int(c_item[0])
            )
            nz = (
                int(c_item) if isinstance(c_item, (int, np.integer)) else int(c_item[2])
            )
            w_yee = yee_te101_omega(nx, nz, a=a, d=d)
            omegas_num_list.append(float(w_num))
            omegas_yee_list.append(float(w_yee))
            yee_rel_mismatches.append(float(abs(w_num - w_yee) / w_yee))

    return {
        "family": name,
        "ns": [int(x) for x in ns],
        "counts": [
            [int(x), int(x), int(x)]
            if isinstance(x, (int, np.integer))
            else [int(c) for c in x]
            for x in counts
        ],
        "signed_errors_amp": [float(x) for x in e_pert_arr],
        "abs_errors_amp": [float(abs(x)) for x in e_pert_arr],
        "signed_errors_yee_amp0": [float(x) for x in e_zero_arr],
        "signed_errors_dist": [float(x) for x in e_dist_arr],
        "omegas_num_amp0": omegas_num_list,
        "omegas_yee_amp0": omegas_yee_list,
        "sign_yee": (
            "negative"
            if np.all(e_zero_arr < 0.0)
            else ("positive" if np.all(e_zero_arr > 0.0) else "mixed")
        ),
        "sign_dist": (
            "positive"
            if np.all(e_dist_arr > 0.0)
            else ("negative" if np.all(e_dist_arr < 0.0) else "mixed")
        ),
        "ls_order_amp": fit_order(ns_list, e_pert_arr),
        "local_orders_amp": local_orders(ns_list, e_pert_arr),
        "two_term_fit_amp": two_term_fit(ns_list, e_pert_arr),
        "ls_order_yee_amp0": fit_order(ns_list, e_zero_arr),
        "local_orders_yee_amp0": local_orders(ns_list, e_zero_arr),
        "ls_order_dist": fit_order(ns_list, e_dist_arr),
        "local_orders_dist": local_orders(ns_list, e_dist_arr),
        "two_term_fit_dist": two_term_fit(ns_list, e_dist_arr),
        "yee_analytic_rel_mismatch": yee_rel_mismatches,
        "max_yee_analytic_rel_mismatch": (
            float(max(yee_rel_mismatches)) if yee_rel_mismatches else None
        ),
    }


def run_perturbed_box_study(quick=False, a=1.0, b=0.7, d=0.8, amp=0.3):
    """
    Execute the W4-B Part 2 perturbed-box factorial convergence study.

    Parameters
    ----------
    quick : bool, optional
        If True, restrict ``F1`` and ``F3`` to ``n <= 16`` and ``F2`` to
        ``k <= 2``.

    Returns
    -------
    dict
        JSON-serialisable factorial results, decomposition, and headline
        estimators ``D`` and ``legacy_two_level_F2_k1_to_2``.
    """
    t0 = time.perf_counter()
    f1_ns = list(F1_NS_QUICK if quick else F1_NS_FULL)
    f2_ks = list(F2_KS_QUICK if quick else F2_KS_FULL)
    f3_ns = list(F3_NS_QUICK if quick else F3_NS_FULL)

    # Family F1: (n, n, n), cell mode, plus amp=0 control
    e_f1_pert = te101_errors(f1_ns, amp=amp, amp_mode="cell", a=a, b=b, d=d)
    e_f1_zero, w_f1_zero = te101_errors(
        f1_ns, amp=0.0, amp_mode="cell", a=a, b=b, d=d, return_omegas=True
    )
    f1_summary = _summarize_family(
        "F1_isotropic_cell",
        f1_ns,
        f1_ns,
        e_f1_pert,
        e_f1_zero,
        omegas_zero=w_f1_zero,
        a=a,
        d=d,
    )

    # Extract study window {6, 8, 12, 16} from F1
    idx_win = [f1_ns.index(n) for n in F1_STUDY_WINDOW]
    e_win_pert = e_f1_pert[idx_win]
    e_win_zero = e_f1_zero[idx_win]
    e_win_dist = e_win_pert - e_win_zero
    d_order = fit_order(F1_STUDY_WINDOW, e_win_pert)

    f1_window_summary = {
        "ns": list(F1_STUDY_WINDOW),
        "D_ls_order_amp": float(d_order),
        "ls_order_yee_amp0": fit_order(F1_STUDY_WINDOW, e_win_zero),
        "ls_order_dist": fit_order(F1_STUDY_WINDOW, e_win_dist),
        "ls_order_amp_6_8_12": fit_order(F1_STUDY_WINDOW[:3], e_win_pert[:3]),
        "ls_order_yee_amp0_6_8_12": fit_order(F1_STUDY_WINDOW[:3], e_win_zero[:3]),
        "ls_order_dist_6_8_12": fit_order(F1_STUDY_WINDOW[:3], e_win_dist[:3]),
        "two_level_6_to_12_amp": two_level_order(6, e_win_pert[0], 12, e_win_pert[2]),
        "two_level_8_to_16_amp": two_level_order(8, e_win_pert[1], 16, e_win_pert[3]),
        "local_orders_amp": local_orders(F1_STUDY_WINDOW, e_win_pert),
        "local_orders_dist": local_orders(F1_STUDY_WINDOW, e_win_dist),
        "signed_errors_amp": [float(x) for x in e_win_pert],
        "signed_errors_yee_amp0": [float(x) for x in e_win_zero],
        "signed_errors_dist": [float(x) for x in e_win_dist],
    }

    # Family F2: (4k, 3k, 4k), cell mode, plus amp=0 control
    f2_counts = [(4 * k, 3 * k, 4 * k) for k in f2_ks]
    f2_n_eff = [4 * k for k in f2_ks]
    e_f2_pert = te101_errors(f2_counts, amp=amp, amp_mode="cell", a=a, b=b, d=d)
    e_f2_zero, w_f2_zero = te101_errors(
        f2_counts,
        amp=0.0,
        amp_mode="cell",
        a=a,
        b=b,
        d=d,
        return_omegas=True,
    )
    f2_summary = _summarize_family(
        "F2_aspect_4k_3k_4k_cell",
        f2_n_eff,
        f2_counts,
        e_f2_pert,
        e_f2_zero,
        omegas_zero=w_f2_zero,
        a=a,
        d=d,
    )
    legacy_f2_order = two_level_order(
        f2_n_eff[0], e_f2_pert[0], f2_n_eff[1], e_f2_pert[1]
    )

    # Family F3: (n, n, n), fixed physical displacement amplitude = F1 n=6 value
    idx_f3_in_f1 = [f1_ns.index(n) for n in f3_ns]
    e_f3_zero = e_f1_zero[idx_f3_in_f1]
    e_f3_pert = np.empty(len(f3_ns), dtype=float)
    for i, n_val in enumerate(f3_ns):
        if n_val == 6:
            e_f3_pert[i] = e_f1_pert[f1_ns.index(6)]
        else:
            e_f3_pert[i] = te101_errors(
                [n_val], amp=amp, amp_mode="fixed", a=a, b=b, d=d
            )[0]
    f3_summary = _summarize_family(
        "F3_isotropic_fixed_map",
        f3_ns,
        f3_ns,
        e_f3_pert,
        e_f3_zero,
        omegas_zero=None,
        a=a,
        d=d,
    )
    f3_win_mask = [i for i, n_val in enumerate(f3_ns) if n_val in F1_STUDY_WINDOW]
    f3_summary["ls_order_amp_window_6_to_16"] = fit_order(
        [f3_ns[i] for i in f3_win_mask], e_f3_pert[f3_win_mask]
    )
    f3_summary["ls_order_dist_window_6_to_16"] = fit_order(
        [f3_ns[i] for i in f3_win_mask],
        (e_f3_pert - e_f3_zero)[f3_win_mask],
    )

    # Compare F1 two-level at n=4->8 vs F2 two-level at k=1->2 (nx=4->8)
    f1_two_level_4_to_8 = two_level_order(4, e_f1_pert[0], 8, e_f1_pert[2])

    # Also evaluate _pec_reconstruction="drop_only" on F1 window {6, 8, 12, 16}
    # and F2 k in {1, 2} to reproduce the pre-PEC-fix study JSON value 1.795576
    e_win_drop = te101_errors(
        F1_STUDY_WINDOW,
        amp=amp,
        amp_mode="cell",
        a=a,
        b=b,
        d=d,
        pec_reconstruction="drop_only",
    )
    e_win_drop_dist = e_win_drop - e_win_zero
    d_drop_only = fit_order(F1_STUDY_WINDOW, e_win_drop)

    e_f2_drop_k12 = te101_errors(
        [(4, 3, 4), (8, 6, 8)],
        amp=amp,
        amp_mode="cell",
        a=a,
        b=b,
        d=d,
        pec_reconstruction="drop_only",
    )
    f2_drop_two_level = two_level_order(4, e_f2_drop_k12[0], 8, e_f2_drop_k12[1])

    return {
        "quick": bool(quick),
        "wall_seconds": float(time.perf_counter() - t0),
        "omega_analytic_rad_s": float(box_te101_omega(a=a, d=d)),
        "D": float(d_order),
        "D_drop_only": float(d_drop_only),
        "legacy_two_level_F2_k1_to_2": float(legacy_f2_order),
        "legacy_two_level_F2_k1_to_2_drop_only": float(f2_drop_two_level),
        "F1_two_level_n4_to_8": float(f1_two_level_4_to_8),
        "F1_study_window_6_8_12_16": f1_window_summary,
        "F1_study_window_drop_only": {
            "ns": list(F1_STUDY_WINDOW),
            "D_drop_only_ls_order_amp": float(d_drop_only),
            "ls_order_dist": fit_order(F1_STUDY_WINDOW, e_win_drop_dist),
            "ls_order_amp_6_8_12": fit_order(F1_STUDY_WINDOW[:3], e_win_drop[:3]),
            "ls_order_dist_6_8_12": fit_order(F1_STUDY_WINDOW[:3], e_win_drop_dist[:3]),
            "n6_e_dist_ratio_vs_constrained": float(e_win_drop_dist[0] / e_win_dist[0]),
            "local_orders_amp": local_orders(F1_STUDY_WINDOW, e_win_drop),
            "local_orders_dist": local_orders(F1_STUDY_WINDOW, e_win_drop_dist),
            "signed_errors_amp": [float(x) for x in e_win_drop],
            "signed_errors_dist": [float(x) for x in e_win_drop_dist],
        },
        "F1": f1_summary,
        "F2": f2_summary,
        "F3": f3_summary,
    }


def _print_summary(data, stream=sys.stdout):
    print(
        f"=== W4-B Part 2: Perturbed-Box Order Study (quick={data['quick']}) ===",
        file=stream,
    )
    print(
        f"omega_analytic = {data['omega_analytic_rad_s']:.8e} rad/s",
        file=stream,
    )
    d_val = data["D"]
    d_drop = data["D_drop_only"]
    leg_val = data["legacy_two_level_F2_k1_to_2"]
    print(
        f"D (LS slope on F1 n in {{6, 8, 12, 16}}, amp=0.3 cell mode) = {d_val:.6f}",
        file=stream,
    )
    print(
        f"D_drop_only (pre-PEC-fix study JSON definition) = {d_drop:.6f}",
        file=stream,
    )
    print(
        f"legacy two-level F2 k=1->2 = {leg_val:.6f} "
        f"(drop_only: {data['legacy_two_level_F2_k1_to_2_drop_only']:.6f})",
        file=stream,
    )
    print(
        f"F1 two-level n=4->8 = {data['F1_two_level_n4_to_8']:.6f}",
        file=stream,
    )
    print(
        f"Reconciliation breakdown: legacy - D_drop_only = {leg_val - d_drop:.6f} "
        f"(PEC fix D - D_drop_only = {d_val - d_drop:.6f}, "
        f"family/window legacy - D = {leg_val - d_val:.6f})",
        file=stream,
    )
    w = data["F1_study_window_6_8_12_16"]
    wd = data["F1_study_window_drop_only"]
    print(
        f"F1 window {{6,8,12,16}} (constrained): order(e)={w['D_ls_order_amp']:.6f}, "
        f"order(e_Yee)={w['ls_order_yee_amp0']:.6f}, "
        f"order(e_dist)={w['ls_order_dist']:.6f} "
        f"(on {{6,8,12}}: order(e)={w['ls_order_amp_6_8_12']:.6f}, "
        f"order(e_Yee)={w['ls_order_yee_amp0_6_8_12']:.6f}, "
        f"order(e_dist)={w['ls_order_dist_6_8_12']:.6f}), "
        f"local_orders(e)={[round(x, 4) for x in w['local_orders_amp']]}",
        file=stream,
    )
    print(
        f"F1 window {{6,8,12,16}} (drop_only)  : order(e)={wd['D_drop_only_ls_order_amp']:.6f}, "
        f"order(e_dist)={wd['ls_order_dist']:.6f} "
        f"(on {{6,8,12}}: order(e)={wd['ls_order_amp_6_8_12']:.6f}, "
        f"order(e_dist)={wd['ls_order_dist_6_8_12']:.6f}), "
        f"local_orders(e)={[round(x, 4) for x in wd['local_orders_amp']]}",
        file=stream,
    )
    for n_val, ep_c, ed_c, ep_d, ed_d in zip(
        w["ns"],
        w["signed_errors_amp"],
        w["signed_errors_dist"],
        wd["signed_errors_amp"],
        wd["signed_errors_dist"],
    ):
        ratio = ed_d / ed_c
        print(
            f"  F1 window N={n_val:2d}: constrained e={ep_c:+.6e} e_dist={ed_c:+.6e} | "
            f"drop_only e={ep_d:+.6e} e_dist={ed_d:+.6e} (ratio={ratio:.2f}x)",
            file=stream,
        )
    for key in ("F1", "F2", "F3"):
        sec = data[key]
        print(
            f"\n[{sec['family']}] ns={sec['ns']}: "
            f"LS(e)={sec['ls_order_amp']:.6f}, "
            f"LS(e_Yee)={sec['ls_order_yee_amp0']:.6f} ({sec['sign_yee']}), "
            f"LS(e_dist)={sec['ls_order_dist']:.6f} ({sec['sign_dist']})",
            file=stream,
        )
        print(
            f"  local_orders(e)    = {[round(x, 6) for x in sec['local_orders_amp']]}",
            file=stream,
        )
        print(
            f"  local_orders(dist) = {[round(x, 6) for x in sec['local_orders_dist']]}",
            file=stream,
        )
        has_yee = bool(sec.get("omegas_num_amp0"))
        for idx, (n_val, ep, ey, ed) in enumerate(
            zip(
                sec["ns"],
                sec["signed_errors_amp"],
                sec["signed_errors_yee_amp0"],
                sec["signed_errors_dist"],
            )
        ):
            extra = ""
            if has_yee:
                wn = sec["omegas_num_amp0"][idx]
                wy = sec["omegas_yee_amp0"][idx]
                rm = sec["yee_analytic_rel_mismatch"][idx]
                extra = f"  w_num(0)={wn:.12e}  w_yee(0)={wy:.12e}  yee_rel_mismatch={rm:.3e}"
            print(
                f"  N={n_val:2d}: e(amp)={ep:+.6e}  e_Yee(0)={ey:+.6e}  e_dist={ed:+.6e}{extra}",
                file=stream,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="W4-B Part 2: perturbed-box TE101 convergence order study."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick resolution set (F1/F3 n <= 16, F2 k <= 2).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    args = parser.parse_args(argv)

    data = run_perturbed_box_study(quick=args.quick)
    _print_summary(data, stream=sys.stdout)
    if args.out:
        payload = json.dumps(data, indent=2)
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
