# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Reproducible study: 2D EIM silicon Bragg grating (Tutorial 6.1) on the subgridded
quadtree mesh (`SubgridPICSimulation`) vs the graded tensor-product Cartesian
mesh (`PICSimulation`) — computational cost at equal accuracy.

Usage
-----
Run quick study locally (or a single task ``I``):
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --quick
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --quick --task 0
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --plot
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --list-tasks
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --list-tasks --full
    python studies/subgrid-vs-graded-bragg/bragg_subgrid_study.py --preflight
"""

from __future__ import annotations

import os

# Pin every task to 1 CPU core before importing NumPy/SciPy/Numba so all wall
# times are uncontended single-core measurements on the same machine type.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import shlex  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.constants import c as c_light, epsilon_0  # noqa: E402
from scipy.sparse import diags as sp_diags  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
T18_DIR = REPO_ROOT / "tutorials" / "06_periodic_structures" / "06-01_bragg_grating"

for _p in (str(REPO_ROOT), str(T18_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _helpers import (  # noqa: E402
    FastPolySlab,
    bragg_grating,
    build_bragg_mesh_overrides,
    extract_stopband_metrics,
)
from wakis.subgrid import SubgridFIT2D, SubgridPICSimulation  # noqa: E402
from wakis.subgrid.monitors import SubgridDFTMonitor  # noqa: E402
from wakis.templates.photonics import (  # noqa: E402
    PICSimulation,
    Port,
    n_to_eps,
    slab_neff,
)
import wakis.unstructured.lts as _lts_mod  # noqa: E402

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# ===========================================================================
# Physical & geometric constants (identical to Tutorial T18, Wang et al. 2014)
# ===========================================================================
N_SI = 3.47
N_SIO2 = 1.44
N_AIR = 1.00
H_SI = 220e-9

W_CORE = 0.45e-6  # 450 nm inner core width
DW_CORRUG = 0.05e-6  # 50 nm tooth depth per side (w_avg = 500 nm, w_max = 550 nm)
PITCH = 0.324e-6  # 324 nm grating period Lambda
DUTY_CYCLE = 0.50  # 50% duty cycle (l_tooth = 162 nm)
N_PERIODS = 100  # 100 periods -> L_wg = 32.40 um

WL_MIN = 1500e-9
WL_MAX = 1620e-9
WL_REF = 1532e-9
N_FREQS = 81
WAVELENGTHS = np.linspace(WL_MIN, WL_MAX, N_FREQS)

DOMAIN_100 = (-17.0e-6, 17.0e-6, -0.72e-6, 0.72e-6)
PORT_X_IN = -16.6e-6
PORT_X_OUT = 16.6e-6
PORT_WIDTH = 1.10e-6
PORT_LENGTH = 3.0e-6

N_EFF_SLAB = float(
    slab_neff(N_SI, N_AIR, H_SI, WL_REF, n_sub=N_SIO2, polarization="TE")
)
EPS_CORE = float(n_to_eps(N_EFF_SLAB))
EPS_CLAD = float(n_to_eps(N_AIR))


def _install_fast_lanczos_patch(tol: float = 1e-4) -> None:
    """Use ``tol=1e-4`` in the CFL spectral-radius Lanczos estimates.

    Parameters
    ----------
    tol : float, optional
        Relative eigenvalue convergence tolerance passed to ``eigsh``
        (dimensionless). On a 100-period grating ($34\\;\\mu\\text{m}$ long,
        $\\sim 10^3$ longitudinal cells), adjacent longitudinal Nyquist modes
        are separated by $\\Delta\\lambda/\\lambda \\sim (\\pi/N_x)^2 \\sim 10^{-7}$.
        Relaxing ``tol`` from ``1e-6``/``1e-8`` to ``1e-4`` accelerates setup by
        $50\\text{--}100\\times$ while changing ``dt`` and ``rho`` by $< 4\\times 10^{-6}$
        (well inside the $10\\text{--}15\\%$ CFL safety margin).
    """
    _orig_lambda_max = SubgridFIT2D._lambda_max

    def _fast_lambda_max(self, tol_val=tol):
        return _orig_lambda_max(self, tol=tol_val)

    SubgridFIT2D._lambda_max = _fast_lambda_max

    _orig_stability_ratio = _lts_mod.stability_ratio

    def _fast_stability_ratio(K, mass, lts, tol=tol, return_mode=False):
        return _orig_stability_ratio(K, mass, lts, tol=tol, return_mode=return_mode)

    _lts_mod.stability_ratio = _fast_stability_ratio


_install_fast_lanczos_patch(tol=1e-4)


def _current_rss_mb() -> float:
    """Return current process resident set size (RSS) in MiB."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return float(pages * os.sysconf("SC_PAGESIZE") / (1024.0 * 1024.0))
    except Exception:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _peak_rss_mb() -> float:
    """Return peak resident set size (ru_maxrss) in MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def build_bragg_structure(
    n_periods: int = N_PERIODS,
) -> tuple[list[FastPolySlab], list[Port], tuple[float, float, float, float], tuple[float, float]]:
    """Construct the 3D/2D-EIM aligned Bragg grating polygon, ports, domain, and grating x-span.

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods $N$ (dimensionless).

    Returns
    -------
    structures : list of FastPolySlab
        Single-polygon corrugated silicon waveguide (`z in [0, H_SI]`).
    ports : list of Port
        Input and output mode ports (`in`, `out`).
    domain : tuple of float
        Simulation domain ``(xmin, xmax, ymin, ymax)`` in meters [m].
    x_span : tuple of float
        Start and end longitudinal coordinates ``(x0, x1)`` of the corrugated
        section in meters [m].
    """
    length = float(n_periods) * PITCH
    x0 = -0.5 * length
    x1 = +0.5 * length
    verts = bragg_grating(
        n_periods=n_periods,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=DUTY_CYCLE,
        shift=0.0,
        apodization="uniform",
        y0=0.0,
        pad_x=PORT_LENGTH,
    )
    slab = FastPolySlab(
        verts,
        z_min=0.0,
        z_max=H_SI,
        material=n_to_eps(N_SI),
    )
    x_port = 0.5 * length + 0.40e-6
    domain_x = 0.5 * length + 0.80e-6
    domain_y = 0.72e-6
    ports = [
        Port(
            "in",
            center=(-x_port, 0.0, 0.5 * H_SI),
            size=(0.0, PORT_WIDTH, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out",
            center=(+x_port, 0.0, 0.5 * H_SI),
            size=(0.0, PORT_WIDTH, 0.8e-6),
            direction=+1,
        ),
    ]
    domain = (-domain_x, +domain_x, -domain_y, +domain_y)
    return [slab], ports, domain, (x0, x1)


def build_subgrid_patches(
    x_span: tuple[float, float],
    ratio: int = 2,
    max_level: int = 2,
    patch_mode: str = "nested",
) -> list[tuple[float, float, float, float, int]]:
    """Build quadtree refinement box tuples around the corrugated Bragg region.

    Parameters
    ----------
    x_span : tuple of float
        Corrugated grating longitudinal extent ``(x0, x1)`` in meters [m].
    ratio : int, optional
        Isotropic split factor $r \\in \\{2, 3\\}$ per refinement level.
    max_level : int, optional
        Finest refinement level $L$ on the $50\\;\\text{nm}$ sidewall teeth.
    patch_mode : {"nested", "tooth_only", "uniform"}, optional
        - ``"nested"``: levels $1..L-1$ refine the corrugated waveguide envelope
          ``y in [-300, +300] nm`` ($x \\in [x_0, x_1]$), and level $L$ refines
          only the two sidewall tooth strips ``y in [-300, -200] nm`` and
          ``[+200, +300] nm``.
        - ``"tooth_only"``: levels $1..L$ refine only the two sidewall tooth
          strips ``|y| in [200, 300] nm``, leaving the waveguide core center at
          level 0.
        - ``"uniform"``: no patches (uniform grid at ``max_level=0``).

    Returns
    -------
    boxes : list of tuple
        List of ``(xmin, xmax, ymin, ymax, level)`` refinement boxes for
        ``SubgridPICSimulation(..., refine=boxes)``.
    """
    if max_level <= 0 or patch_mode == "uniform":
        return []
    x0, x1 = x_span
    y_core_inner = 0.5 * W_CORE - 0.5 * DW_CORRUG  # 200 nm
    y_tooth_outer = 0.5 * W_CORE + 1.5 * DW_CORRUG  # 300 nm
    boxes: list[tuple[float, float, float, float, int]] = []

    if patch_mode == "nested":
        for lev in range(1, max_level):
            boxes.append((x0, x1, -y_tooth_outer, +y_tooth_outer, lev))
        boxes.append((x0, x1, -y_tooth_outer, -y_core_inner, max_level))
        boxes.append((x0, x1, +y_core_inner, +y_tooth_outer, max_level))
    elif patch_mode == "tooth_only":
        for lev in range(1, max_level + 1):
            boxes.append((x0, x1, -y_tooth_outer, -y_core_inner, lev))
            boxes.append((x0, x1, +y_core_inner, +y_tooth_outer, lev))
    else:
        raise ValueError(f"Unknown patch_mode={patch_mode!r}")
    return boxes


def build_tuned_bragg_mesh_overrides(
    n_periods: int = N_PERIODS,
    pitch: float = PITCH,
    w_core: float = W_CORE,
    dw: float = DW_CORRUG,
    shift: float = 0.0,
    y0: float = 0.0,
    n_feature: float = 4.0,
    dx_background: float | None = None,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    """Construct tuned tensor-product ``mesh_overrides`` whose tooth and core
    strips use the same cell sizes as the subgrid arm at the same ``N_tooth``
    ladder.

    Parameters
    ----------
    n_periods : int, optional
        Number of grating periods $N_{\\text{per}}$ (dimensionless).
    pitch : float, optional
        Grating period $\\Lambda$ in meters [m].
    w_core : float, optional
        Inner uncorrugated core width $w_{\\text{core}}$ in meters [m].
    dw : float, optional
        Sidewall tooth depth $\\Delta w$ in meters [m].
    shift : float, optional
        Longitudinal shift in meters [m].
    y0 : float, optional
        Waveguide center y-coordinate in meters [m].
    n_feature : float, optional
        Number of cells across the $50\\;\\text{nm}$ tooth depth $N_{\\text{tooth}}$.
    dx_background : float or None, optional
        Background slab cell size in meters [m].

    Returns
    -------
    overrides : list of tuple
        Per-axis override boxes ``(lo, hi, (hx, hy, hz))`` in meters [m]:
        - Tooth strips (``|y - y0| in [200, 300] nm``): ``dy_tooth = dw / N``
        - Grating longitudinal extent (``x in [-L/2, +L/2]``): ``dx_tooth = dy_tooth``
        - Core strip (``|y - y0| <= 200 nm``): ``dy_core = 2 * dy_tooth`` (matching
          the subgrid arm's intermediate $L=1$ / $L-1$ core patch size).
    """
    if n_feature is None or float(n_feature) <= 0:
        return []
    n_feat = float(n_feature)
    dw = float(dw)
    w_core = float(w_core)
    pitch = float(pitch)
    length = float(n_periods) * pitch

    hy_tooth = dw / n_feat
    hx_tooth = hy_tooth
    hy_core = 2.0 * hy_tooth
    if dx_background is not None:
        hy_tooth = min(float(dx_background), hy_tooth)
        hx_tooth = min(float(dx_background), hx_tooth)
        hy_core = min(float(dx_background), hy_core)

    y_in = 0.5 * w_core - 0.5 * dw  # 200 nm
    y_out = 0.5 * w_core + 1.5 * dw  # 300 nm
    x_lo = -0.5 * length
    x_hi = +0.5 * length + max(0.0, float(shift))

    return [
        ((-np.inf, y0 - y_out, -np.inf), (np.inf, y0 - y_in, np.inf), (1.0, hy_tooth, 1.0)),
        ((-np.inf, y0 + y_in, -np.inf), (np.inf, y0 + y_out, np.inf), (1.0, hy_tooth, 1.0)),
        ((-np.inf, y0 - y_in, -np.inf), (np.inf, y0 + y_in, np.inf), (1.0, hy_core, 1.0)),
        ((x_lo, -np.inf, -np.inf), (x_hi, np.inf, np.inf), (hx_tooth, 1.0, 1.0)),
    ]


# ===========================================================================
# Task table definitions (--quick and --full)
# ===========================================================================
def get_task_table(quick: bool = True) -> list[dict]:
    """Return the benchmark task table for ``--quick`` or ``--full``.

    Parameters
    ----------
    quick : bool, optional
        If True, return the ``--quick`` task table preserving indices ``0..19``
        for existing runs and appending ``20..24`` for the tuned graded arms.
        If False, return the 18-task 100-period ``--full`` table for Cloud Batch
        (dropping diagnostic/redundant runs that no longer serve the comparison).

    Returns
    -------
    tasks : list of dict
        Task specification dictionaries with 0-based ``index``.
    """
    n_pml = 6 if quick else 10
    energy_decay = 2e-2 if quick else 1e-4
    max_time = 0.95e-12 if quick else 1.80e-12

    common = {
        "mode": "quick" if quick else "full",
        "n_periods": N_PERIODS,
        "n_pml": n_pml,
        "energy_decay": energy_decay,
        "max_time": max_time,
    }

    tuned_graded_tasks = [
        # --- Tuned Graded Cartesian (isotropic teeth dx=dy + L=1 core strip) ---
        {
            **common,
            "name": "graded_tuned_L1",
            "arm": "graded_tuned",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "tuned",
            "cpw": 10.0,
            "n_tooth": 4,
            "description": "Tuned Graded Kottke L1 (cpw=10, N_tooth=4, dx=dy=12.50 nm, dy_core=25.0 nm)",
        },
        {
            **common,
            "name": "graded_tuned_L2",
            "arm": "graded_tuned",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "tuned",
            "cpw": 14.0,
            "n_tooth": 6,
            "description": "Tuned Graded Kottke L2 (cpw=14, N_tooth=6, dx=dy=8.33 nm, dy_core=16.7 nm)",
        },
        {
            **common,
            "name": "graded_tuned_L3",
            "arm": "graded_tuned",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "tuned",
            "cpw": 20.0,
            "n_tooth": 8,
            "description": "Tuned Graded Kottke L3 (cpw=20, N_tooth=8, dx=dy=6.25 nm, dy_core=12.5 nm)",
        },
        # --- Tuned Graded Staircase variant (same tuned mesh, staircase sampling) ---
        {
            **common,
            "name": "graded_tuned_staircase_L1",
            "arm": "graded_tuned_staircase",
            "backend": "cartesian",
            "averaging": "staircase",
            "mesh_style": "tuned",
            "cpw": 10.0,
            "n_tooth": 4,
            "description": "Tuned Graded Staircase L1 (cpw=10, N_tooth=4, dx=dy=12.50 nm, dy_core=25.0 nm)",
        },
        {
            **common,
            "name": "graded_tuned_staircase_L2",
            "arm": "graded_tuned_staircase",
            "backend": "cartesian",
            "averaging": "staircase",
            "mesh_style": "tuned",
            "cpw": 14.0,
            "n_tooth": 6,
            "description": "Tuned Graded Staircase L2 (cpw=14, N_tooth=6, dx=dy=8.33 nm, dy_core=16.7 nm)",
        },
    ]

    t18_graded_kottke_tasks = [
        # --- T18-style Graded Cartesian with Kottke smoothing (anisotropic teeth, unrefined core) ---
        {
            **common,
            "name": "graded_t18_L1",
            "arm": "graded_t18",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "t18",
            "cpw": 10.0,
            "n_tooth": 4,
            "description": "T18 Graded Kottke L1 (cpw=10, N_tooth=4, dy=12.50 nm, dx=40.50 nm, dy_core=52.7 nm)",
        },
        {
            **common,
            "name": "graded_t18_L2",
            "arm": "graded_t18",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "t18",
            "cpw": 14.0,
            "n_tooth": 6,
            "description": "T18 Graded Kottke L2 (cpw=14, N_tooth=6, dy=8.33 nm, dx=27.00 nm, dy_core=35.9 nm)",
        },
        {
            **common,
            "name": "graded_t18_L3",
            "arm": "graded_t18",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "t18",
            "cpw": 20.0,
            "n_tooth": 8,
            "description": "T18 Graded Kottke L3 (cpw=20, N_tooth=8, dy=6.25 nm, dx=20.25 nm, dy_core=25.8 nm)",
        },
        {
            **common,
            "name": "graded_t18_R",
            "arm": "graded_t18",
            "backend": "cartesian",
            "averaging": "kottke",
            "mesh_style": "t18",
            "cpw": 24.0 if quick else 40.0,
            "n_tooth": 10 if quick else 16,
            "description": (
                "T18 Graded Kottke R_quick (cpw=24, N_tooth=10, dy=5.00 nm)"
                if quick
                else "T18 Graded Kottke R_full (cpw=40, N_tooth=16, dy=3.125 nm)"
            ),
        },
    ]

    t18_graded_staircase_tasks = [
        {
            **common,
            "name": "graded_t18_staircase_L1",
            "arm": "graded_t18_staircase",
            "backend": "cartesian",
            "averaging": "staircase",
            "mesh_style": "t18",
            "cpw": 10.0,
            "n_tooth": 4,
            "description": "T18 Graded Staircase L1 (cpw=10, N_tooth=4, dy=12.50 nm)",
        },
        {
            **common,
            "name": "graded_t18_staircase_L2",
            "arm": "graded_t18_staircase",
            "backend": "cartesian",
            "averaging": "staircase",
            "mesh_style": "t18",
            "cpw": 14.0,
            "n_tooth": 6,
            "description": "T18 Graded Staircase L2 (cpw=14, N_tooth=6, dy=8.33 nm)",
        },
        {
            **common,
            "name": "graded_t18_staircase_L3",
            "arm": "graded_t18_staircase",
            "backend": "cartesian",
            "averaging": "staircase",
            "mesh_style": "t18",
            "cpw": 20.0,
            "n_tooth": 8,
            "description": "T18 Graded Staircase L3 (cpw=20, N_tooth=8, dy=6.25 nm)",
        },
    ]

    subgrid_main_tasks = [
        {
            **common,
            "name": "subgrid_cpw10_N4",
            "arm": "subgrid",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 10.6,
            "ratio": 2,
            "max_level": 2,
            "lts": 1,
            "n_tooth": 4,
            "description": "Subgrid nested (cpw=10.6, r=2, L=2, N_tooth=4.0, dy=12.41 nm)",
        },
        {
            **common,
            "name": "subgrid_cpw14_N6",
            "arm": "subgrid",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 14.0,
            "ratio": 2,
            "max_level": 2,
            "lts": 1,
            "n_tooth": 6,
            "description": "Subgrid nested (cpw=14.0, r=2, L=2, N_tooth=5.4, dy=9.23 nm)",
        },
        {
            **common,
            "name": "subgrid_cpw10_N6",
            "arm": "subgrid",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 15.6,
            "ratio": 2,
            "max_level": 2,
            "lts": 1,
            "n_tooth": 6,
            "description": "Subgrid nested (cpw=15.6, r=2, L=2, N_tooth=6.0, dy=8.37 nm)",
        },
        {
            **common,
            "name": "subgrid_cpw10_N8",
            "arm": "subgrid",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 10.6,
            "ratio": 2,
            "max_level": 3,
            "lts": 1,
            "n_tooth": 8,
            "description": "Subgrid nested (cpw=10.6, r=2, L=3, N_tooth=8.1, dy=6.21 nm)",
        },
    ]

    subgrid_lts3_tasks = [
        {
            **common,
            "name": "subgrid_lts3_cpw10_N4",
            "arm": "subgrid_lts3",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 10.6,
            "ratio": 2,
            "max_level": 2,
            "lts": 3,
            "n_tooth": 4,
            "description": "Subgrid+LTS3 nested (cpw=10.6, r=2, L=2, lts=3, N_tooth=4.0)",
        },
        {
            **common,
            "name": "subgrid_lts3_cpw14_N6",
            "arm": "subgrid_lts3",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 14.0,
            "ratio": 2,
            "max_level": 2,
            "lts": 3,
            "n_tooth": 6,
            "description": "Subgrid+LTS3 nested (cpw=14.0, r=2, L=2, lts=3, N_tooth=5.4)",
        },
        {
            **common,
            "name": "subgrid_lts3_cpw10_N8",
            "arm": "subgrid_lts3",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 10.6,
            "ratio": 2,
            "max_level": 3,
            "lts": 3,
            "n_tooth": 8,
            "description": "Subgrid+LTS3 nested (cpw=10.6, r=2, L=3, lts=3, N_tooth=8.1)",
        },
        {
            **common,
            "name": "subgrid_lts3_ref",
            "arm": "subgrid_lts3",
            "backend": "subgrid",
            "patch_mode": "nested",
            "cpw": 14.0 if quick else 20.0,
            "ratio": 2,
            "max_level": 3,
            "lts": 3,
            "n_tooth": 11 if quick else 16,
            "description": (
                "Subgrid+LTS3 finest ref (cpw=14.0, r=2, L=3, lts=3, N_tooth=10.8, dy=4.62 nm)"
                if quick
                else "Subgrid+LTS3 finest ref (cpw=20.0, r=2, L=3, lts=3, N_tooth=15.5, dy=3.23 nm)"
            ),
        },
    ]

    uniform_task = {
        **common,
        "name": "uniform_N4",
        "arm": "uniform",
        "backend": "subgrid",
        "patch_mode": "uniform",
        "cpw": 42.8,
        "ratio": 2,
        "max_level": 0,
        "lts": 1,
        "n_tooth": 4,
        "description": "Uniform mesh at N_tooth=4 resolution (cpw=42.8, L=0, dy=12.31 nm)",
    }

    if quick:
        # Preserve indices 0..19 for already-completed quick data, and append 20..24 for tuned graded arms
        raw_tasks = [
            *t18_graded_kottke_tasks,  # 0..3
            *t18_graded_staircase_tasks,  # 4..6
            subgrid_main_tasks[0],  # 7: subgrid_cpw10_N4
            {
                **common,
                "name": "subgrid_cpw14_N4",
                "arm": "subgrid",
                "backend": "subgrid",
                "patch_mode": "nested",
                "cpw": 14.0,
                "ratio": 3,
                "max_level": 1,
                "lts": 1,
                "n_tooth": 4,
                "description": "Subgrid single-level r=3 (cpw=14.0, r=3, L=1, N_tooth=4.1, dy=12.31 nm)",
            },  # 8
            subgrid_main_tasks[1],  # 9: subgrid_cpw14_N6
            subgrid_main_tasks[2],  # 10: subgrid_cpw10_N6
            subgrid_main_tasks[3],  # 11: subgrid_cpw10_N8
            {
                **common,
                "name": "subgrid_toothonly_cpw10_N4",
                "arm": "subgrid_toothonly",
                "backend": "subgrid",
                "patch_mode": "tooth_only",
                "cpw": 10.6,
                "ratio": 2,
                "max_level": 2,
                "lts": 1,
                "n_tooth": 4,
                "description": "Subgrid tooth-only (cpw=10.6, r=2, L=2, core unrefined at L=0)",
            },  # 12
            {
                **common,
                "name": "subgrid_lts2_cpw10_N4",
                "arm": "subgrid_lts2",
                "backend": "subgrid",
                "patch_mode": "nested",
                "cpw": 10.6,
                "ratio": 2,
                "max_level": 2,
                "lts": 2,
                "n_tooth": 4,
                "description": "Subgrid+LTS2 nested (cpw=10.6, r=2, L=2, lts=2, N_tooth=4.0)",
            },  # 13
            *subgrid_lts3_tasks,  # 14..17
            uniform_task,  # 18
            {
                **common,
                "name": "convergence_20per_and_fields",
                "arm": "diagnostics",
                "backend": "diagnostics",
                "cpw": 20.0,
                "n_tooth": 16,
                "description": "20-period continuum convergence ladder, step-rate microbench & 2D DFT field map",
            },  # 19
            *tuned_graded_tasks,  # 20..24
        ]
    else:
        # Clean 18-task 100-period --full table (dropped non-comparison diagnostic/redundant runs)
        raw_tasks = [
            *tuned_graded_tasks,  # 0..4: graded_tuned (L1..L3) + graded_tuned_staircase (L1..L2)
            *t18_graded_kottke_tasks,  # 5..8: graded_t18 (L1..R at 40 cpw, N=16)
            *subgrid_main_tasks,  # 9..12: subgrid (N=4, 5.4, 6.0, 8.1)
            *subgrid_lts3_tasks,  # 13..16: subgrid_lts3 (N=4, 5.4, 8.1, ref at 20 cpw, N=16)
            uniform_task,  # 17: uniform_N4
            # 18..20: uniform ladder (cpw scaled from uniform_N4: dy = 12.31 nm * 42.8 / cpw)
            *[
                {
                    **uniform_task,
                    "name": f"uniform_{dy_nm}nm",
                    "cpw": round(42.8 * 12.3077 / dy_nm, 3),
                    "n_tooth": n_t,
                    "description": f"Uniform mesh dy~{dy_nm} nm (cpw={42.8 * 12.3077 / dy_nm:.2f}, L=0)",
                }
                for dy_nm, n_t in ((15, 3), (20, 2), (25, 2))
            ],
        ]

    for idx, t in enumerate(raw_tasks):
        t["index"] = idx
    return raw_tasks


def compute_preflight_estimates(quick: bool = False) -> list[dict]:
    """Compute preflight cell, unknown, timestep, work, and runtime estimates.

    Parameters
    ----------
    quick : bool, optional
        Whether to estimate for ``--quick`` (`max_time=0.95 ps`) or ``--full``
        (`max_time=1.80 ps`).

    Returns
    -------
    estimates : list of dict
        Per-task preflight estimates calibrated against measured single-core
        step rates on this VM (`~100-115 ns/cell-step` for Cartesian `SolverFIT3D`
        on 1 core, `~48 ns/cell-step` for `SubgridFIT2D`, and `~67 ns/cell-step`
        per coarse macrostep with `lts=3`).
    """
    tasks = get_task_table(quick=quick)
    scale_t = 1.0 if quick else (1.80 / 0.95)
    # Base measured/analytical mesh geometry at 100 periods (calibrated to 1-core runs)
    base_specs = {
        "graded_tuned_L1": (174174, 522522, 12.50, 12.50, 24.95, 38433, 869.7),
        "graded_tuned_L2": (347336, 1042008, 8.33, 8.33, 16.64, 57549, 2494.9),
        "graded_tuned_L3": (599754, 1799262, 6.25, 6.25, 12.49, 76866, 5750.0),
        "graded_tuned_staircase_L1": (174174, 522522, 12.50, 12.50, 24.95, 38433, 863.2),
        "graded_tuned_staircase_L2": (347336, 1042008, 8.33, 8.33, 16.64, 57549, 2428.2),
        "graded_t18_L1": (50700, 152100, 12.50, 40.50, 52.72, 28758, 158.4),
        "graded_t18_L2": (98046, 294138, 8.33, 27.00, 35.91, 43002, 484.1),
        "graded_t18_L3": (167500, 502500, 6.25, 20.25, 25.79, 57516, 1179.2),
        "graded_t18_R": (
            (244179, 732537, 5.00, 16.20, 21.75, 71758, 2169.4)
            if quick
            else (621612, 1864836, 3.125, 10.125, 12.90, 218351, 15880.0)
        ),
        "graded_t18_staircase_L1": (50700, 152100, 12.50, 40.50, 52.72, 28758, 156.3),
        "graded_t18_staircase_L2": (98046, 294138, 8.33, 27.00, 35.91, 43002, 515.7),
        "graded_t18_staircase_L3": (167500, 502500, 6.25, 20.25, 25.79, 57516, 1112.7),
        "subgrid_cpw10_N4": (96594, 297799, 12.41, 12.48, 24.83, 35491, 174.6),
        "subgrid_cpw14_N4": (101424, 309647, 12.31, 12.59, 12.31, 35441, 177.2),
        "subgrid_cpw14_N6": (167538, 513195, 9.23, 9.44, 18.46, 47546, 391.4),
        "subgrid_cpw10_N6": (190482, 583203, 8.37, 8.48, 16.74, 52670, 486.1),
        "subgrid_cpw10_N8": (299196, 913465, 6.21, 6.24, 12.41, 71640, 1020.3),
        "subgrid_toothonly_cpw10_N4": (82965, 258196, 12.41, 12.48, 49.66, 35491, 151.7),
        "subgrid_lts2_cpw10_N4": (96594, 297799, 12.41, 12.48, 24.83, 19964, 132.6),
        "subgrid_lts3_cpw10_N4": (96594, 297799, 12.41, 12.48, 24.83, 13309, 93.4),
        "subgrid_lts3_cpw14_N6": (167538, 513195, 9.23, 9.44, 18.46, 17830, 213.6),
        "subgrid_lts3_cpw10_N8": (299196, 913465, 6.21, 6.24, 12.41, 26865, 566.3),
        "subgrid_lts3_ref": (
            (517752, 1574225, 4.62, 4.72, 9.23, 35878, 1261.8)
            if quick
            else (1081500, 3275000, 3.23, 3.23, 6.45, 97200, 6800.0)
        ),
        "uniform_N4": (356169, 1074288, 12.31, 12.37, 12.31, 36288, 678.6),
        # uniform ladder: scaled from uniform_N4 (cells ~ h^-2, steps ~ h^-1)
        **{
            f"uniform_{d}nm": (
                int(356169 * (12.31 / d) ** 2),
                int(1074288 * (12.31 / d) ** 2),
                float(d),
                float(d),
                float(d),
                int(36288 * 12.31 / d),
                678.6 * (12.31 / d) ** 3,
            )
            for d in (15, 20, 25)
        },
        "convergence_20per_and_fields": (96594, 297799, 2.34, 2.34, 4.69, 15000, 1950.0),
    }

    out = []
    for t in tasks:
        name = t["name"]
        n_cells, n_unk, dy_t, dx_t, dy_c, n_steps_q, wall_q = base_specs[name]
        if name.endswith("_R") and not quick:
            n_steps = n_steps_q
            wall_s = wall_q
        elif name == "subgrid_lts3_ref" and not quick:
            n_steps = n_steps_q
            wall_s = wall_q
        elif name == "convergence_20per_and_fields":
            n_steps = n_steps_q
            wall_s = wall_q
        else:
            n_steps = int(round(n_steps_q * scale_t))
            wall_s = float(wall_q * scale_t)
        work_raw = float(n_unk) * float(n_steps)
        est_rss_mb = round(140.0 + 0.00055 * n_unk, 1)
        out.append(
            {
                "index": t["index"],
                "name": name,
                "arm": t["arm"],
                "mode": t["mode"],
                "n_cells": int(n_cells),
                "n_unknowns": int(n_unk),
                "dy_tooth_nm": float(dy_t),
                "dx_tooth_nm": float(dx_t),
                "dy_core_nm": float(dy_c),
                "n_steps_est": int(n_steps),
                "work_raw_est": float(work_raw),
                "wall_time_1core_est_s": round(wall_s, 1),
                "peak_rss_est_mb": est_rss_mb,
                "description": t["description"],
            }
        )
    return out


# ===========================================================================
# Single-task execution
# ===========================================================================
def _run_cartesian_task(task: dict, out_dir: Path) -> dict:
    """Execute a graded Cartesian (`PICSimulation`) Bragg grating task."""
    n_periods = int(task["n_periods"])
    length = float(n_periods) * PITCH
    structures, ports, domain, (x0, x1) = build_bragg_structure(n_periods=n_periods)

    cpw = float(task["cpw"])
    n_tooth = int(task["n_tooth"])
    averaging = str(task.get("averaging", "kottke"))
    mesh_style = str(task.get("mesh_style", "t18"))
    dx_bg = WL_MIN / (N_EFF_SLAB * cpw)

    if mesh_style == "tuned":
        overrides = build_tuned_bragg_mesh_overrides(
            n_periods=n_periods,
            pitch=PITCH,
            w_core=W_CORE,
            dw=DW_CORRUG,
            shift=0.0,
            n_feature=n_tooth,
            dx_background=dx_bg,
        )
        patch_mode_str = "tuned_isotropic_core"
    else:
        overrides = build_bragg_mesh_overrides(
            n_periods=n_periods,
            pitch=PITCH,
            w_core=W_CORE,
            dw=DW_CORRUG,
            duty_cycle=DUTY_CYCLE,
            shift=0.0,
            n_feature=n_tooth,
            dx_background=dx_bg,
        )
        patch_mode_str = "anisotropic_strips"

    t0 = time.perf_counter()
    sim = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=averaging,
        cells_per_wavelength=cpw,
        domain=domain,
        n_pml=int(task["n_pml"]),
        cfln=0.85,
        mesh_overrides=overrides,
        use_gpu=False,
    )
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    s_mat = sim.run_s_parameters(
        drive=["in"],
        max_time=float(task["max_time"]),
        energy_decay=float(task["energy_decay"]),
        collect_fields=False,
    )
    t_solve = time.perf_counter() - t1
    t_total = t_setup + t_solve

    R = np.abs(s_mat[("in", "in")]) ** 2
    T = np.abs(s_mat[("out", "in")]) ** 2
    wl_m = np.asarray(sim.wavelengths, dtype=float)
    wl_nm = wl_m * 1e9
    metrics = extract_stopband_metrics(wl_m, R, T, length)

    nx, ny = int(sim.grid.Nx), int(sim.grid.Ny)
    n_cells = int(nx * ny)
    n_unknowns = int(3 * n_cells)  # Ex, Ey, Hz per 2D cell
    dt_s = float(sim.solver.dt)
    n_steps = int(
        getattr(
            sim.solver,
            "Nt",
            int(np.ceil(float(task["max_time"]) / dt_s)),
        )
    )
    if n_steps <= 0:
        n_steps = int(np.ceil(float(task["max_time"]) / dt_s))
    work_raw = float(n_unknowns) * float(n_steps)
    work_cell_steps = float(n_cells) * float(n_steps)

    # Measure actual grid spacings in tooth strip and at core center y=0
    dy_arr = np.diff(sim.grid.y)
    yc_arr = 0.5 * (sim.grid.y[:-1] + sim.grid.y[1:])
    dx_arr = np.diff(sim.grid.x)
    xc_arr = 0.5 * (sim.grid.x[:-1] + sim.grid.x[1:])

    tooth_y_mask = (np.abs(yc_arr) >= 0.225e-6) & (np.abs(yc_arr) <= 0.275e-6)
    grating_x_mask = (xc_arr >= x0) & (xc_arr <= x1)
    dy_tooth_nm = float(np.median(dy_arr[tooth_y_mask]) * 1e9)
    dx_tooth_nm = float(np.median(dx_arr[grating_x_mask]) * 1e9)
    idx_y0 = int(np.argmin(np.abs(yc_arr)))
    dy_core_nm = float(dy_arr[idx_y0] * 1e9)

    result = {
        "index": int(task["index"]),
        "name": str(task["name"]),
        "arm": str(task["arm"]),
        "backend": "cartesian",
        "mode": str(task["mode"]),
        "description": str(task["description"]),
        "averaging": averaging,
        "patch_mode": patch_mode_str,
        "cpw": cpw,
        "ratio": 1,
        "max_level": 0,
        "lts": 1,
        "n_tooth_target": n_tooth,
        "n_tooth_actual": round(50.0 / dy_tooth_nm, 3),
        "dy_tooth_nm": round(dy_tooth_nm, 4),
        "dx_tooth_nm": round(dx_tooth_nm, 4),
        "dy_core_nm": round(dy_core_nm, 4),
        "dx_core_nm": round(dx_tooth_nm, 4),
        "nx": nx,
        "ny": ny,
        "n_cells": n_cells,
        "n_dof": n_cells,
        "n_edges": 2 * n_cells,
        "n_unknowns": n_unknowns,
        "n_hanging": 0,
        "n_cut_cells": 0,
        "lts_fine_fraction": 1.0,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "t_end_ps": round(n_steps * dt_s * 1e12, 4),
        "work_raw": work_raw,
        "work_eff": work_raw,
        "work_cell_steps": work_cell_steps,
        "setup_time_s": round(t_setup, 3),
        "solve_time_s": round(t_solve, 3),
        "total_wall_time_s": round(t_total, 3),
        "ns_per_cell_step": round((t_solve * 1e9) / max(work_cell_steps, 1.0), 3),
        "ns_per_unknown_step": round((t_solve * 1e9) / max(work_raw, 1.0), 3),
        "rss_mb": round(_current_rss_mb(), 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
        "lambda_B_nm": round(float(metrics["lambda_bragg"] * 1e9), 4),
        "bandwidth_3db_nm": round(float(metrics["fwhm_bandwidth"] * 1e9), 4),
        "R_max": round(float(metrics["R_max"]), 6),
        "T_min": round(float(metrics["T_min"]), 6),
        "extinction_db": round(float(metrics["extinction_db"]), 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    with open(out_dir / f"task_{idx}.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    np.savez_compressed(
        out_dir / f"task_{idx}.npz",
        wavelengths_nm=wl_nm,
        R=R,
        T=T,
        S11=s_mat[("in", "in")],
        S21=s_mat[("out", "in")],
    )
    return result


def _run_subgrid_task(task: dict, out_dir: Path) -> dict:
    """Execute a subgridded quadtree (`SubgridPICSimulation`) Bragg grating task."""
    n_periods = int(task["n_periods"])
    length = float(n_periods) * PITCH
    structures, ports, domain, (x0, x1) = build_bragg_structure(n_periods=n_periods)

    cpw = float(task["cpw"])
    ratio = int(task.get("ratio", 2))
    max_level = int(task.get("max_level", 2))
    lts = int(task.get("lts", 1))
    patch_mode = str(task.get("patch_mode", "nested"))
    n_tooth = int(task["n_tooth"])

    boxes = build_subgrid_patches(
        x_span=(x0, x1),
        ratio=ratio,
        max_level=max_level,
        patch_mode=patch_mode,
    )

    t0 = time.perf_counter()
    sim = SubgridPICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        cells_per_wavelength=cpw,
        domain=domain,
        n_pml=int(task["n_pml"]),
        refine=boxes if max_level > 0 else None,
        ratio=ratio,
        max_level=max_level,
        lts=None if lts <= 1 else lts,
    )
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    s_mat = sim.run_s_parameters(
        drive=["in"],
        max_time=float(task["max_time"]),
        energy_decay=float(task["energy_decay"]),
        collect_fields=False,
    )
    t_solve = time.perf_counter() - t1
    t_total = t_setup + t_solve

    R = np.abs(s_mat[("in", "in")]) ** 2
    T = np.abs(s_mat[("out", "in")]) ** 2
    wl_m = np.asarray(sim.wavelengths, dtype=float)
    wl_nm = wl_m * 1e9
    metrics = extract_stopband_metrics(wl_m, R, T, length)

    summary = sim.summary()
    n_cells = int(summary["n_cells"])
    n_dof = int(summary["n_dof"])
    n_edges = int(summary["n_edges"])
    n_unknowns = int(n_dof + n_edges)
    n_hanging = int(np.sum(sim.mesh.hanging))
    c_dict = sim.mesh.cells
    xm_c = 0.5 * (c_dict["x0"] + c_dict["x1"])
    ym_c = 0.5 * (c_dict["y0"] + c_dict["y1"])
    rad_c = 0.5 * np.hypot(c_dict["x1"] - c_dict["x0"], c_dict["y1"] - c_dict["y0"])
    n_cut = int(np.sum(sim.geometry.cut(xm_c, ym_c, rad_c)))
    n_steps = int(summary["n_steps"])
    dt_s = float(summary["dt"])

    H_y_nm = float(np.median(sim.grid.dy) * 1e9)
    H_x_nm = float(np.median(sim.grid.dx) * 1e9)
    dy_tooth_nm = H_y_nm / (ratio**max_level)
    dx_tooth_nm = H_x_nm / (ratio**max_level)
    if patch_mode == "nested" and max_level >= 1:
        dy_core_nm = H_y_nm / (ratio ** max(1, max_level - 1))
        dx_core_nm = H_x_nm / (ratio ** max(1, max_level - 1))
    elif patch_mode == "tooth_only":
        dy_core_nm = H_y_nm
        dx_core_nm = H_x_nm
    else:
        dy_core_nm = dy_tooth_nm
        dx_core_nm = dx_tooth_nm

    lts_fine_frac = 1.0
    if sim.solver.lts is not None:
        lts_fine_frac = float(sim.solver.lts.fine_fraction)

    work_raw = float(n_unknowns) * float(n_steps)
    work_cell_steps = float(n_cells) * float(n_steps)
    # In TwoLevelLTS, solver.dt is the coarse macrostep and n_steps is the
    # number of coarse macrosteps. Each macrostep updates all n_unknowns once
    # plus (p - 1) Chebyshev A_FF scalar node passes on the fine nodes F.
    if lts > 1 and sim.solver.lts is not None:
        work_eff = work_raw * (
            1.0 + float(lts - 1) * lts_fine_frac * (float(n_dof) / float(n_unknowns))
        )
    else:
        work_eff = work_raw

    result = {
        "index": int(task["index"]),
        "name": str(task["name"]),
        "arm": str(task["arm"]),
        "backend": "subgrid",
        "mode": str(task["mode"]),
        "description": str(task["description"]),
        "averaging": "cut_cell_subgrid",
        "patch_mode": patch_mode,
        "cpw": cpw,
        "ratio": ratio,
        "max_level": max_level,
        "lts": lts,
        "n_tooth_target": n_tooth,
        "n_tooth_actual": round(50.0 / dy_tooth_nm, 3),
        "dy_tooth_nm": round(dy_tooth_nm, 4),
        "dx_tooth_nm": round(dx_tooth_nm, 4),
        "dy_core_nm": round(dy_core_nm, 4),
        "dx_core_nm": round(dx_core_nm, 4),
        "nx": int(sim.grid.Nx),
        "ny": int(sim.grid.Ny),
        "n_cells": n_cells,
        "n_dof": n_dof,
        "n_edges": n_edges,
        "n_unknowns": n_unknowns,
        "n_hanging": n_hanging,
        "n_cut_cells": n_cut,
        "lts_fine_fraction": round(lts_fine_frac, 4),
        "dt_s": dt_s,
        "n_steps": n_steps,
        "t_end_ps": round(n_steps * dt_s * 1e12, 4),
        "work_raw": work_raw,
        "work_eff": work_eff,
        "work_cell_steps": work_cell_steps,
        "setup_time_s": round(t_setup, 3),
        "solve_time_s": round(t_solve, 3),
        "total_wall_time_s": round(t_total, 3),
        "ns_per_cell_step": round((t_solve * 1e9) / max(work_cell_steps, 1.0), 3),
        "ns_per_unknown_step": round((t_solve * 1e9) / max(work_raw, 1.0), 3),
        "rss_mb": round(_current_rss_mb(), 2),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
        "lambda_B_nm": round(float(metrics["lambda_bragg"] * 1e9), 4),
        "bandwidth_3db_nm": round(float(metrics["fwhm_bandwidth"] * 1e9), 4),
        "R_max": round(float(metrics["R_max"]), 6),
        "T_min": round(float(metrics["T_min"]), 6),
        "extinction_db": round(float(metrics["extinction_db"]), 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    with open(out_dir / f"task_{idx}.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    np.savez_compressed(
        out_dir / f"task_{idx}.npz",
        wavelengths_nm=wl_nm,
        R=R,
        T=T,
        S11=s_mat[("in", "in")],
        S21=s_mat[("out", "in")],
    )
    return result


def _run_diagnostics_task(task: dict, out_dir: Path) -> dict:
    """Run 20-period exact convergence ladder, per-cell-step microbenchmark,
    and capture 2D DFT field maps at the Bragg stopband and passband.
    """
    t_start = time.perf_counter()

    # 1. Per-cell-step single-core microbenchmark (SubgridFIT2D vs SolverFIT3D)
    structs_20, ports_20, dom_20, (x0_20, x1_20) = build_bragg_structure(n_periods=20)
    l_wg_20 = 20.0 * PITCH

    sim_cart = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=21,
        structures=structs_20,
        ports=ports_20,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20.0,
        domain=dom_20,
        n_pml=8,
        cfln=0.85,
        use_gpu=False,
    )
    for _ in range(15):
        sim_cart.solver.one_step()
    t0 = time.perf_counter()
    n_bench_steps = 500
    for _ in range(n_bench_steps):
        sim_cart.solver.one_step()
    dt_cart = time.perf_counter() - t0
    cells_cart = int(sim_cart.grid.Nx * sim_cart.grid.Ny)
    ns_cell_cart = float(dt_cart * 1e9 / (cells_cart * n_bench_steps))
    ns_unk_cart = float(dt_cart * 1e9 / (3 * cells_cart * n_bench_steps))

    sim_sg0 = SubgridPICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=21,
        structures=structs_20,
        ports=ports_20,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        cells_per_wavelength=20.0,
        domain=dom_20,
        n_pml=8,
        ratio=2,
        max_level=0,
        lts=None,
    )
    for _ in range(15):
        sim_sg0.solver.one_step()
    t0 = time.perf_counter()
    for _ in range(n_bench_steps):
        sim_sg0.solver.one_step()
    dt_sg0 = time.perf_counter() - t0
    cells_sg0 = int(sim_sg0.mesh.n_cells)
    unk_sg0 = int(sim_sg0.solver.n_dof + sim_sg0.solver.n_edges)
    ns_cell_sg0 = float(dt_sg0 * 1e9 / (cells_sg0 * n_bench_steps))
    ns_unk_sg0 = float(dt_sg0 * 1e9 / (unk_sg0 * n_bench_steps))

    # 2. 20-period exact convergence ladder (up to (40, 16) and dy=2.34 nm)
    conv_graded_kottke = []
    conv_graded_stair = []
    for cpw_val, nt_val in [(10, 4), (14, 6), (20, 8), (28, 12), (40, 16)]:
        dx_bg_20 = WL_MIN / (N_EFF_SLAB * float(cpw_val))
        ov = build_bragg_mesh_overrides(
            n_periods=20,
            pitch=PITCH,
            w_core=W_CORE,
            dw=DW_CORRUG,
            duty_cycle=DUTY_CYCLE,
            shift=0.0,
            n_feature=nt_val,
            dx_background=dx_bg_20,
        )
        s_k = PICSimulation(
            wavelength_min=1480e-9,
            wavelength_max=1640e-9,
            n_freqs=161,
            structures=structs_20,
            ports=ports_20,
            n_core=N_SI,
            n_clad=N_AIR,
            n_sub=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=float(cpw_val),
            domain=dom_20,
            n_pml=6,
            cfln=0.85,
            mesh_overrides=ov,
            use_gpu=False,
        )
        sm_k = s_k.run_s_parameters(
            drive=["in"], max_time=0.34e-12, energy_decay=1e-3, collect_fields=False
        )
        Rk = np.abs(sm_k[("in", "in")]) ** 2
        Tk = np.abs(sm_k[("out", "in")]) ** 2
        mk = extract_stopband_metrics(s_k.wavelengths, Rk, Tk, l_wg_20)
        conv_graded_kottke.append(
            {
                "cpw": cpw_val,
                "n_tooth": nt_val,
                "dy_tooth_nm": round(50.0 / nt_val, 4),
                "n_cells": int(s_k.grid.Nx * s_k.grid.Ny),
                "lambda_B_nm": round(float(mk["lambda_bragg"] * 1e9), 4),
                "bandwidth_3db_nm": round(float(mk["fwhm_bandwidth"] * 1e9), 4),
                "R_max": round(float(mk["R_max"]), 6),
            }
        )
        if nt_val <= 12:
            s_st = PICSimulation(
                wavelength_min=1480e-9,
                wavelength_max=1640e-9,
                n_freqs=161,
                structures=structs_20,
                ports=ports_20,
                n_core=N_SI,
                n_clad=N_AIR,
                n_sub=N_SIO2,
                wavelength_ref=WL_REF,
                two_d=True,
                averaging="staircase",
                cells_per_wavelength=float(cpw_val),
                domain=dom_20,
                n_pml=6,
                cfln=0.85,
                mesh_overrides=ov,
                use_gpu=False,
            )
            sm_st = s_st.run_s_parameters(
                drive=["in"], max_time=0.34e-12, energy_decay=1e-3, collect_fields=False
            )
            Rst = np.abs(sm_st[("in", "in")]) ** 2
            Tst = np.abs(sm_st[("out", "in")]) ** 2
            mst = extract_stopband_metrics(s_st.wavelengths, Rst, Tst, l_wg_20)
            conv_graded_stair.append(
                {
                    "cpw": cpw_val,
                    "n_tooth": nt_val,
                    "dy_tooth_nm": round(50.0 / nt_val, 4),
                    "n_cells": int(s_st.grid.Nx * s_st.grid.Ny),
                    "lambda_B_nm": round(float(mst["lambda_bragg"] * 1e9), 4),
                    "bandwidth_3db_nm": round(float(mst["fwhm_bandwidth"] * 1e9), 4),
                    "R_max": round(float(mst["R_max"]), 6),
                }
            )

    conv_subgrid = []
    for cpw_val, r_val, lev_val in [
        (10.6, 2, 2),
        (14.0, 2, 2),
        (10.6, 2, 3),
        (14.0, 2, 3),
        (20.0, 2, 3),
        (28.0, 2, 3),
    ]:
        p_20 = build_subgrid_patches((x0_20, x1_20), ratio=r_val, max_level=lev_val, patch_mode="nested")
        s_sg = SubgridPICSimulation(
            wavelength_min=1480e-9,
            wavelength_max=1640e-9,
            n_freqs=161,
            structures=structs_20,
            ports=ports_20,
            n_core=N_SI,
            n_clad=N_AIR,
            n_sub=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            cells_per_wavelength=float(cpw_val),
            domain=dom_20,
            n_pml=6,
            refine=p_20,
            ratio=r_val,
            max_level=lev_val,
            lts=3 if cpw_val >= 20.0 else None,
        )
        sm_sg = s_sg.run_s_parameters(
            drive=["in"], max_time=0.34e-12, energy_decay=1e-3, collect_fields=False
        )
        Rsg = np.abs(sm_sg[("in", "in")]) ** 2
        Tsg = np.abs(sm_sg[("out", "in")]) ** 2
        msg = extract_stopband_metrics(s_sg.wavelengths, Rsg, Tsg, l_wg_20)
        hy_nm = float(np.median(s_sg.grid.dy) * 1e9 / (r_val**lev_val))
        conv_subgrid.append(
            {
                "cpw": cpw_val,
                "ratio": r_val,
                "max_level": lev_val,
                "dy_tooth_nm": round(hy_nm, 4),
                "n_tooth": round(50.0 / hy_nm, 2),
                "n_cells": int(s_sg.mesh.n_cells),
                "lambda_B_nm": round(float(msg["lambda_bragg"] * 1e9), 4),
                "bandwidth_3db_nm": round(float(msg["fwhm_bandwidth"] * 1e9), 4),
                "R_max": round(float(msg["R_max"]), 6),
            }
        )

    # 3. 2D DFT field map on the 100-period subgridded Bragg grating at lambda_B & passband
    wl_bragg_field = 1553.8e-9
    wl_pass_field = 1515.0e-9
    structs_100, ports_100, dom_100, (x0_100, x1_100) = build_bragg_structure(n_periods=N_PERIODS)
    patches_100 = build_subgrid_patches((x0_100, x1_100), ratio=2, max_level=2, patch_mode="nested")
    sim_fld = SubgridPICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structs_100,
        ports=ports_100,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        cells_per_wavelength=10.6,
        domain=dom_100,
        n_pml=6,
        refine=patches_100,
        ratio=2,
        max_level=2,
        lts=3,
        field_wavelengths=(wl_bragg_field, wl_pass_field),
    )
    # Also attach a high-resolution zoomed DFT monitor over the first 8 grating periods
    x_zoom = np.linspace(-16.45e-6, -13.60e-6, 360)
    y_zoom = np.linspace(-0.42e-6, 0.42e-6, 168)
    zoom_mon = SubgridDFTMonitor(
        [c_light / wl_bragg_field, c_light / wl_pass_field],
        points=(x_zoom, y_zoom),
        components=("Ex", "Ey"),
    )
    _orig_one_step = sim_fld.solver.one_step
    _step_counter = [0]
    _dt_fld = float(sim_fld.solver.dt)

    def _hooked_one_step():
        t_cur = _step_counter[0] * _dt_fld
        _orig_one_step()
        zoom_mon.update(sim_fld.solver, t_cur)
        _step_counter[0] += 1

    sim_fld.solver.one_step = _hooked_one_step
    sim_fld.run_s_parameters(
        drive=["in"],
        max_time=0.65e-12,
        energy_decay=2e-2,
        collect_fields=True,
    )

    fld_bragg = sim_fld.dft_fields[wl_bragg_field]
    fld_pass = sim_fld.dft_fields[wl_pass_field]
    ex_z_raw, _ = zoom_mon.get_field("Ex")
    ey_z_raw, _ = zoom_mon.get_field("Ey")

    t_total = time.perf_counter() - t_start
    result = {
        "index": int(task["index"]),
        "name": str(task["name"]),
        "arm": "diagnostics",
        "backend": "diagnostics",
        "mode": str(task["mode"]),
        "description": str(task["description"]),
        "microbenchmark": {
            "cartesian_ns_per_cell_step": round(ns_cell_cart, 3),
            "cartesian_ns_per_unknown_step": round(ns_unk_cart, 3),
            "subgrid_L0_ns_per_cell_step": round(ns_cell_sg0, 3),
            "subgrid_L0_ns_per_unknown_step": round(ns_unk_sg0, 3),
            "speed_ratio_cartesian_over_subgrid_cell_step": round(ns_cell_cart / ns_cell_sg0, 3),
            "speed_ratio_cartesian_over_subgrid_unknown_step": round(ns_unk_cart / ns_unk_sg0, 3),
            "reference_baseline_subgrid_ns_per_cell_step": 38.7,
            "reference_baseline_cartesian_ns_per_cell_step": 87.1,
            "reference_baseline_ratio": round(87.1 / 38.7, 3),
        },
        "convergence_20_periods": {
            "graded_kottke": conv_graded_kottke,
            "graded_staircase": conv_graded_stair,
            "subgrid_nested": conv_subgrid,
        },
        "total_wall_time_s": round(t_total, 3),
        "peak_rss_mb": round(_peak_rss_mb(), 2),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task["index"]
    with open(out_dir / f"task_{idx}.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    np.savez_compressed(
        out_dir / f"task_{idx}.npz",
        x_full=fld_bragg["x"],
        y_full=fld_bragg["y"],
        Ey_bragg=fld_bragg["Ey"],
        E2_bragg=fld_bragg["E2"],
        Ey_pass=fld_pass["Ey"],
        E2_pass=fld_pass["E2"],
        x_zoom=x_zoom.astype(np.float32),
        y_zoom=y_zoom.astype(np.float32),
        Ex_zoom_bragg=ex_z_raw[0].astype(np.complex64),
        Ey_zoom_bragg=ey_z_raw[0].astype(np.complex64),
        Ex_zoom_pass=ex_z_raw[1].astype(np.complex64),
        Ey_zoom_pass=ey_z_raw[1].astype(np.complex64),
    )
    return result


def run_single_task(task: dict, out_dir: Path = DATA_DIR) -> dict:
    """Dispatch and run a single benchmark task, saving JSON and NPZ outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = out_dir / f".lock_task_{task['index']}"
    if (
        os.environ.get("FORCE_RERUN") != "1"
        and lock_file.exists()
        and (time.time() - lock_file.stat().st_mtime < 2400)
    ):
        print(
            f"[Task {task['index']:02d}] Already running or completed ({task['name']}); skipping duplicate.",
            flush=True,
        )
        return {}
    lock_file.write_text(str(os.getpid()), encoding="utf-8")

    backend = task["backend"]
    print(f"[Task {task['index']:02d}] Starting {task['name']}: {task['description']}", flush=True)
    if backend == "cartesian":
        res = _run_cartesian_task(task, out_dir)
    elif backend == "subgrid":
        res = _run_subgrid_task(task, out_dir)
    elif backend == "diagnostics":
        res = _run_diagnostics_task(task, out_dir)
    else:
        raise ValueError(f"Unknown backend={backend!r}")
    print(
        f"[Task {task['index']:02d}] Finished {task['name']} in {res['total_wall_time_s']:.1f}s "
        f"(peak RSS {res['peak_rss_mb']:.0f} MB)",
        flush=True,
    )
    return res


# ===========================================================================
# Reference loading, error metrics, and cost-at-equal-accuracy interpolation
# ===========================================================================
def _interpolate_cost_for_target(
    errors: np.ndarray,
    costs: np.ndarray,
    target_err: float,
    labels: list[str] | None = None,
    unit: str = "nm",
) -> tuple[float | None, str]:
    """Interpolate cost at ``target_err`` in log-log space between bracketing points only.

    Parameters
    ----------
    errors : ndarray
        Positive error values ordered from coarse to fine resolution.
    costs : ndarray
        Corresponding cost values (work or wall time).
    target_err : float
        Target error threshold.
    labels : list of str or None, optional
        Optional human-readable point labels (e.g. ``["N=4.0", "N=6.0"]``).
    unit : str, optional
        Error unit string for status description (default ``"nm"``).

    Returns
    -------
    cost_at_target : float or None
        Log-log interpolated cost to reach ``target_err``, or ``None`` if not bracketed.
    status : str
        ``"interpolated (...)"`` if bracketed within the measured sweep, or
        ``"not reached"`` otherwise (no extrapolation).
    """
    valid = (np.isfinite(errors)) & (errors > 1e-9) & (np.isfinite(costs)) & (costs > 0)
    e = np.asarray(errors[valid], dtype=float)
    c = np.asarray(costs[valid], dtype=float)
    if e.size < 2:
        return None, "not reached"

    # Sort by increasing cost (coarse to fine)
    order = np.argsort(c)
    e = e[order]
    c = c[order]
    lbls = None
    if labels is not None:
        valid_idx = np.flatnonzero(valid)
        lbls = [labels[valid_idx[i]] for i in order]

    for k in range(len(e) - 1):
        if (e[k] >= target_err >= e[k + 1]) and (e[k] > e[k + 1] > 0):
            log_e0, log_e1 = np.log10(e[k]), np.log10(e[k + 1])
            log_c0, log_c1 = np.log10(c[k]), np.log10(c[k + 1])
            w = (np.log10(target_err) - log_e0) / (log_e1 - log_e0)
            val = float(10.0 ** (log_c0 + w * (log_c1 - log_c0)))
            if lbls is not None:
                return val, f"interpolated ({lbls[k]} {e[k]:.3f} {unit} .. {lbls[k + 1]} {e[k + 1]:.3f} {unit})"
            return val, "interpolated"

    return None, "not reached"


def _richardson_h2(h1: float, v1: float, h2: float, v2: float) -> float:
    """Two-point $O(h^2)$ Richardson continuum extrapolation at $h \\to 0$."""
    return float((v2 * (h1**2) - v1 * (h2**2)) / ((h1**2) - (h2**2)))


def compute_summary_and_errors(
    data_dir: Path = DATA_DIR,
    ref_override_nm: float | None = None,
    write_json: bool = True,
) -> dict:
    """Load the 18 ``--full`` task JSON/NPZ outputs, compute the three $O(h^2)$
    Richardson continuum limits (`graded_tuned` tasks 1,2; `graded_t18` tasks 7,8;
    `subgrid_lts3` tasks 15,16), the neutral reference (mean of the three) and
    uncertainty $\\sigma_{\\mathrm{ref}}$ (half the spread), per-task errors, and
    log-log interpolated ``work_eff`` (and ``wall_time_s``) to reach $1.0\\;\\mathrm{nm}$
    and $0.5\\;\\mathrm{nm}$ per arm without extrapolation. Writes ``data/analysis_full.json``
    and ``data/summary_metrics.json``.
    """
    records = {}
    spectra = {}
    for p in sorted(data_dir.glob("task_*.json")):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("mode") != "full":
            continue
        idx = int(d["index"])
        records[idx] = d
        npz_path = data_dir / f"task_{idx}.npz"
        if npz_path.exists():
            z = np.load(npz_path)
            spectra[idx] = {
                "wavelengths_nm": np.asarray(z["wavelengths_nm"], dtype=float),
                "R": np.asarray(z["R"], dtype=float),
                "T": np.asarray(z["T"], dtype=float),
            }

    by_name = {r["name"]: r for r in records.values()}
    spec_by_name = {records[k]["name"]: spectra[k] for k in spectra}

    # Three 100-period O(h^2) Richardson continuum limits (h = dy_tooth_nm)
    # 1. graded_tuned (tasks 1, 2: graded_tuned_L2, graded_tuned_L3)
    t_gt1, t_gt2 = records[1], records[2]
    lam_inf_graded_tuned = _richardson_h2(
        t_gt1["dy_tooth_nm"], t_gt1["lambda_B_nm"], t_gt2["dy_tooth_nm"], t_gt2["lambda_B_nm"]
    )
    bw_inf_graded_tuned = _richardson_h2(
        t_gt1["dy_tooth_nm"], t_gt1["bandwidth_3db_nm"], t_gt2["dy_tooth_nm"], t_gt2["bandwidth_3db_nm"]
    )
    rm_inf_graded_tuned = _richardson_h2(
        t_gt1["dy_tooth_nm"], t_gt1["R_max"], t_gt2["dy_tooth_nm"], t_gt2["R_max"]
    )

    # 2. graded_t18 (tasks 7, 8: graded_t18_L3, graded_t18_R)
    t_t18_1, t_t18_2 = records[7], records[8]
    lam_inf_graded_t18 = _richardson_h2(
        t_t18_1["dy_tooth_nm"], t_t18_1["lambda_B_nm"], t_t18_2["dy_tooth_nm"], t_t18_2["lambda_B_nm"]
    )
    bw_inf_graded_t18 = _richardson_h2(
        t_t18_1["dy_tooth_nm"], t_t18_1["bandwidth_3db_nm"], t_t18_2["dy_tooth_nm"], t_t18_2["bandwidth_3db_nm"]
    )
    rm_inf_graded_t18 = _richardson_h2(
        t_t18_1["dy_tooth_nm"], t_t18_1["R_max"], t_t18_2["dy_tooth_nm"], t_t18_2["R_max"]
    )

    # 3. subgrid_lts3 (tasks 15, 16: subgrid_lts3_cpw10_N8, subgrid_lts3_ref)
    t_sg1, t_sg2 = records[15], records[16]
    lam_inf_subgrid_lts3 = _richardson_h2(
        t_sg1["dy_tooth_nm"], t_sg1["lambda_B_nm"], t_sg2["dy_tooth_nm"], t_sg2["lambda_B_nm"]
    )
    bw_inf_subgrid_lts3 = _richardson_h2(
        t_sg1["dy_tooth_nm"], t_sg1["bandwidth_3db_nm"], t_sg2["dy_tooth_nm"], t_sg2["bandwidth_3db_nm"]
    )
    rm_inf_subgrid_lts3 = _richardson_h2(
        t_sg1["dy_tooth_nm"], t_sg1["R_max"], t_sg2["dy_tooth_nm"], t_sg2["R_max"]
    )

    lam_limits = [lam_inf_graded_tuned, lam_inf_graded_t18, lam_inf_subgrid_lts3]
    bw_limits = [bw_inf_graded_tuned, bw_inf_graded_t18, bw_inf_subgrid_lts3]
    rm_limits = [rm_inf_graded_tuned, rm_inf_graded_t18, rm_inf_subgrid_lts3]

    ref_neutral_wl = float(np.mean(lam_limits)) if ref_override_nm is None else float(ref_override_nm)
    ref_uncertainty_wl = 0.5 * float(max(lam_limits) - min(lam_limits))
    min_valid_threshold_wl = 3.0 * ref_uncertainty_wl

    ref_neutral_bw = float(np.mean(bw_limits))
    ref_uncertainty_bw = 0.5 * float(max(bw_limits) - min(bw_limits))
    ref_uncertainty_bw_pct = 100.0 * ref_uncertainty_bw / max(ref_neutral_bw, 1e-12)
    min_valid_threshold_bw_pct = 3.0 * ref_uncertainty_bw_pct

    ref_common_rmax = float(np.mean(rm_limits))

    # Neutral reference spectrum: mean of finest runs of the three converged arms
    ref_sg = by_name["subgrid_lts3_ref"]
    ref_gr = by_name["graded_t18_R"]
    ref_gt = by_name["graded_tuned_L3"]
    if all(n in spec_by_name for n in ("graded_tuned_L3", "graded_t18_R", "subgrid_lts3_ref")):
        ref_neutral_R_spec = (
            spec_by_name["graded_tuned_L3"]["R"]
            + spec_by_name["graded_t18_R"]["R"]
            + spec_by_name["subgrid_lts3_ref"]["R"]
        ) / 3.0
        ref_neutral_T_spec = (
            spec_by_name["graded_tuned_L3"]["T"]
            + spec_by_name["graded_t18_R"]["T"]
            + spec_by_name["subgrid_lts3_ref"]["T"]
        ) / 3.0
    else:
        ref_neutral_R_spec = None
        ref_neutral_T_spec = None

    enriched_tasks = []
    for idx in sorted(records):
        r = dict(records[idx])
        err_wl_exact = abs(float(r["lambda_B_nm"]) - ref_neutral_wl)
        rel_bw_exact = abs(float(r["bandwidth_3db_nm"]) - ref_neutral_bw) / ref_neutral_bw
        r["err_lambda_B_nm"] = round(err_wl_exact, 4)
        r["_err_lambda_B_exact_nm"] = err_wl_exact
        r["signed_dlambda_B_nm"] = round(float(r["lambda_B_nm"]) - ref_neutral_wl, 4)
        r["err_bw_nm"] = round(abs(float(r["bandwidth_3db_nm"]) - ref_neutral_bw), 4)
        r["rel_err_bw"] = round(rel_bw_exact, 6)
        r["err_bw_pct"] = round(100.0 * rel_bw_exact, 4)
        r["err_R_max"] = round(abs(float(r["R_max"]) - ref_common_rmax), 6)
        if idx in spectra and ref_neutral_R_spec is not None:
            sp = spectra[idx]
            r["err_spectrum_R_rms"] = round(float(np.sqrt(np.mean((sp["R"] - ref_neutral_R_spec) ** 2))), 5)
            r["err_spectrum_T_rms"] = round(float(np.sqrt(np.mean((sp["T"] - ref_neutral_T_spec) ** 2))), 5)
        else:
            r["err_spectrum_R_rms"] = 0.0
            r["err_spectrum_T_rms"] = 0.0

        # Errors vs each arm's Richardson limit separately
        r["err_lambda_B_vs_graded_tuned_limit_nm"] = round(
            abs(float(r["lambda_B_nm"]) - lam_inf_graded_tuned), 4
        )
        r["err_lambda_B_vs_graded_limit_nm"] = round(
            abs(float(r["lambda_B_nm"]) - lam_inf_graded_t18), 4
        )
        r["err_lambda_B_vs_subgrid_limit_nm"] = round(
            abs(float(r["lambda_B_nm"]) - lam_inf_subgrid_lts3), 4
        )
        r["err_bw_vs_graded_tuned_limit_pct"] = round(
            100.0 * abs(float(r["bandwidth_3db_nm"]) - bw_inf_graded_tuned) / bw_inf_graded_tuned, 4
        )
        r["err_bw_vs_graded_limit_pct"] = round(
            100.0 * abs(float(r["bandwidth_3db_nm"]) - bw_inf_graded_t18) / bw_inf_graded_t18, 4
        )
        r["err_bw_vs_subgrid_limit_pct"] = round(
            100.0 * abs(float(r["bandwidth_3db_nm"]) - bw_inf_subgrid_lts3) / bw_inf_subgrid_lts3, 4
        )
        enriched_tasks.append(r)

    arm_sequences = {
        "graded_tuned": ["graded_tuned_L1", "graded_tuned_L2", "graded_tuned_L3"],
        "graded_tuned_staircase": ["graded_tuned_staircase_L1", "graded_tuned_staircase_L2"],
        "graded_t18": ["graded_t18_L1", "graded_t18_L2", "graded_t18_L3", "graded_t18_R"],
        "subgrid": ["subgrid_cpw10_N4", "subgrid_cpw14_N6", "subgrid_cpw10_N6", "subgrid_cpw10_N8"],
        "subgrid_lts3": [
            "subgrid_lts3_cpw10_N4",
            "subgrid_lts3_cpw14_N6",
            "subgrid_lts3_cpw10_N8",
            "subgrid_lts3_ref",
        ],
        "uniform": ["uniform_25nm", "uniform_20nm", "uniform_15nm", "uniform_N4"],
    }

    enriched_by_name = {r["name"]: r for r in enriched_tasks}
    candidate_targets_wl = [1.0, 0.5]
    active_targets_wl = [tw for tw in candidate_targets_wl if tw >= min_valid_threshold_wl]
    dropped_targets_wl = [tw for tw in candidate_targets_wl if tw < min_valid_threshold_wl]

    def _build_cost_table(err_wl_key: str) -> dict:
        out_table = {}
        for arm_key, names in arm_sequences.items():
            pts = [enriched_by_name[n] for n in names if n in enriched_by_name]
            if not pts:
                continue
            errs_wl = np.array([p[err_wl_key] for p in pts], dtype=float)
            work_raw = np.array([p["work_raw"] for p in pts], dtype=float)
            work_eff = np.array([p["work_eff"] for p in pts], dtype=float)
            wall_s = np.array([p["total_wall_time_s"] for p in pts], dtype=float)
            lbls = [f"N={p['n_tooth_actual']:.1f}" for p in pts]

            arm_targets = {}
            for tw in candidate_targets_wl:
                w_eff_val, st_w = _interpolate_cost_for_target(
                    errs_wl, work_eff, tw, labels=lbls, unit="nm"
                )
                w_raw_val, _ = _interpolate_cost_for_target(
                    errs_wl, work_raw, tw, labels=lbls, unit="nm"
                )
                t_val, _ = _interpolate_cost_for_target(
                    errs_wl, wall_s, tw, labels=lbls, unit="nm"
                )
                arm_targets[f"dlambda_le_{tw:.1f}nm"] = {
                    "work_eff": w_eff_val if w_eff_val is not None else "not reached",
                    "work_raw": w_raw_val if w_raw_val is not None else "not reached",
                    "wall_time_s": t_val if t_val is not None else "not reached",
                    "status": st_w,
                    "dropped_below_3x_uncertainty": bool(tw < min_valid_threshold_wl),
                }
            out_table[arm_key] = arm_targets
        return out_table

    cost_to_target = _build_cost_table("_err_lambda_B_exact_nm")
    cost_to_target_vs_graded_tuned = _build_cost_table("err_lambda_B_vs_graded_tuned_limit_nm")
    cost_to_target_vs_graded_t18 = _build_cost_table("err_lambda_B_vs_graded_limit_nm")
    cost_to_target_vs_subgrid_lts3 = _build_cost_table("err_lambda_B_vs_subgrid_limit_nm")

    # Strip internal _err_lambda_B_exact_nm helper key before serializing tasks
    for r in enriched_tasks:
        r.pop("_err_lambda_B_exact_nm", None)

    work_eff_1nm = {
        arm: cost_to_target[arm]["dlambda_le_1.0nm"]["work_eff"]
        for arm in ("graded_tuned", "graded_tuned_staircase", "graded_t18", "subgrid", "subgrid_lts3")
        if arm in cost_to_target
    }
    work_eff_0p5nm = {
        arm: cost_to_target[arm]["dlambda_le_0.5nm"]["work_eff"]
        for arm in ("graded_tuned", "graded_tuned_staircase", "graded_t18", "subgrid", "subgrid_lts3")
        if arm in cost_to_target
    }
    wall_time_1nm = {
        arm: cost_to_target[arm]["dlambda_le_1.0nm"]["wall_time_s"]
        for arm in ("graded_tuned", "graded_tuned_staircase", "graded_t18", "subgrid", "subgrid_lts3")
        if arm in cost_to_target
    }
    wall_time_0p5nm = {
        arm: cost_to_target[arm]["dlambda_le_0.5nm"]["wall_time_s"]
        for arm in ("graded_tuned", "graded_tuned_staircase", "graded_t18", "subgrid", "subgrid_lts3")
        if arm in cost_to_target
    }

    u_rec = enriched_by_name["uniform_N4"]
    uniform_summary = {
        "task_index": int(u_rec["index"]),
        "name": u_rec["name"],
        "dy_tooth_nm": float(u_rec["dy_tooth_nm"]),
        "dx_tooth_nm": float(u_rec["dx_tooth_nm"]),
        "dy_core_nm": float(u_rec["dy_core_nm"]),
        "lambda_B_nm": float(u_rec["lambda_B_nm"]),
        "err_lambda_B_nm": float(u_rec["err_lambda_B_nm"]),
        "bandwidth_3db_nm": float(u_rec["bandwidth_3db_nm"]),
        "rel_err_bw": float(u_rec["rel_err_bw"]),
        "err_bw_pct": float(u_rec["err_bw_pct"]),
        "work_eff": float(u_rec["work_eff"]),
        "work_raw": float(u_rec["work_raw"]),
        "total_wall_time_s": float(u_rec["total_wall_time_s"]),
        "ns_per_cell_step": float(u_rec["ns_per_cell_step"]),
        "ns_per_unknown_step": float(u_rec["ns_per_unknown_step"]),
    }

    def _safe_ratio(num, den):
        if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den > 0:
            return round(float(num) / float(den), 4)
        return "not reached"

    verdict_ratios = {
        "work_eff_ratios_1p0nm": {
            "subgrid_lts3_vs_graded_tuned": _safe_ratio(work_eff_1nm.get("graded_tuned"), work_eff_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_graded_t18": _safe_ratio(work_eff_1nm.get("graded_t18"), work_eff_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_subgrid": _safe_ratio(work_eff_1nm.get("subgrid"), work_eff_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_uniform_N4": _safe_ratio(uniform_summary["work_eff"], work_eff_1nm.get("subgrid_lts3")),
            "subgrid_vs_graded_tuned": _safe_ratio(work_eff_1nm.get("graded_tuned"), work_eff_1nm.get("subgrid")),
            "subgrid_vs_graded_t18": _safe_ratio(work_eff_1nm.get("graded_t18"), work_eff_1nm.get("subgrid")),
            "subgrid_vs_uniform_N4": _safe_ratio(uniform_summary["work_eff"], work_eff_1nm.get("subgrid")),
            "graded_tuned_vs_graded_t18": _safe_ratio(work_eff_1nm.get("graded_t18"), work_eff_1nm.get("graded_tuned")),
        },
        "work_eff_ratios_0p5nm": {
            "subgrid_lts3_vs_graded_tuned": _safe_ratio(work_eff_0p5nm.get("graded_tuned"), work_eff_0p5nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_graded_t18": _safe_ratio(work_eff_0p5nm.get("graded_t18"), work_eff_0p5nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_subgrid": _safe_ratio(work_eff_0p5nm.get("subgrid"), work_eff_0p5nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_uniform_N4": _safe_ratio(uniform_summary["work_eff"], work_eff_0p5nm.get("subgrid_lts3")),
            "subgrid_vs_graded_tuned": _safe_ratio(work_eff_0p5nm.get("graded_tuned"), work_eff_0p5nm.get("subgrid")),
            "subgrid_vs_graded_t18": _safe_ratio(work_eff_0p5nm.get("graded_t18"), work_eff_0p5nm.get("subgrid")),
            "subgrid_vs_uniform_N4": _safe_ratio(uniform_summary["work_eff"], work_eff_0p5nm.get("subgrid")),
        },
        "wall_time_ratios_1p0nm": {
            "subgrid_lts3_vs_graded_tuned": _safe_ratio(wall_time_1nm.get("graded_tuned"), wall_time_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_graded_t18": _safe_ratio(wall_time_1nm.get("graded_t18"), wall_time_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_subgrid": _safe_ratio(wall_time_1nm.get("subgrid"), wall_time_1nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_uniform_N4": _safe_ratio(uniform_summary["total_wall_time_s"], wall_time_1nm.get("subgrid_lts3")),
            "subgrid_vs_graded_tuned": _safe_ratio(wall_time_1nm.get("graded_tuned"), wall_time_1nm.get("subgrid")),
            "subgrid_vs_graded_t18": _safe_ratio(wall_time_1nm.get("graded_t18"), wall_time_1nm.get("subgrid")),
            "subgrid_vs_uniform_N4": _safe_ratio(uniform_summary["total_wall_time_s"], wall_time_1nm.get("subgrid")),
            "graded_tuned_vs_graded_t18": _safe_ratio(wall_time_1nm.get("graded_t18"), wall_time_1nm.get("graded_tuned")),
        },
        "wall_time_ratios_0p5nm": {
            "subgrid_lts3_vs_graded_tuned": _safe_ratio(wall_time_0p5nm.get("graded_tuned"), wall_time_0p5nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_subgrid": _safe_ratio(wall_time_0p5nm.get("subgrid"), wall_time_0p5nm.get("subgrid_lts3")),
            "subgrid_lts3_vs_uniform_N4": _safe_ratio(uniform_summary["total_wall_time_s"], wall_time_0p5nm.get("subgrid_lts3")),
            "subgrid_vs_graded_tuned": _safe_ratio(wall_time_0p5nm.get("graded_tuned"), wall_time_0p5nm.get("subgrid")),
            "subgrid_vs_uniform_N4": _safe_ratio(uniform_summary["total_wall_time_s"], wall_time_0p5nm.get("subgrid")),
        },
    }

    ns_per_cell_step_by_arm = {}
    for arm_key, names in arm_sequences.items():
        pts = [enriched_by_name[n] for n in names if n in enriched_by_name]
        if not pts:
            continue
        cs_vals = [float(p["ns_per_cell_step"]) for p in pts]
        us_vals = [float(p["ns_per_unknown_step"]) for p in pts]
        ns_per_cell_step_by_arm[arm_key] = {
            "tasks": [p["index"] for p in pts],
            "ns_per_cell_step_values": cs_vals,
            "ns_per_cell_step_mean": round(float(np.mean(cs_vals)), 3),
            "ns_per_cell_step_min": round(float(min(cs_vals)), 3),
            "ns_per_cell_step_max": round(float(max(cs_vals)), 3),
            "ns_per_unknown_step_mean": round(float(np.mean(us_vals)), 3),
        }

    lts3_vs_no_lts_pairs = []
    for i_no, i_lts in [(9, 13), (10, 14), (12, 15)]:
        t_no, t_lts = enriched_by_name[records[i_no]["name"]], enriched_by_name[records[i_lts]["name"]]
        d_lam = float(t_lts["lambda_B_nm"]) - float(t_no["lambda_B_nm"])
        d_bw = float(t_lts["bandwidth_3db_nm"]) - float(t_no["bandwidth_3db_nm"])
        lts3_vs_no_lts_pairs.append(
            {
                "task_no_lts": i_no,
                "task_lts3": i_lts,
                "name_no_lts": t_no["name"],
                "name_lts3": t_lts["name"],
                "dy_tooth_nm": float(t_no["dy_tooth_nm"]),
                "lambda_B_no_lts_nm": float(t_no["lambda_B_nm"]),
                "lambda_B_lts3_nm": float(t_lts["lambda_B_nm"]),
                "delta_lambda_B_nm": round(d_lam, 4),
                "abs_delta_over_sigma_ref": round(abs(d_lam) / max(ref_uncertainty_wl, 1e-12), 4),
                "delta_bandwidth_3db_nm": round(d_bw, 4),
            }
        )

    t11 = enriched_by_name["subgrid_cpw10_N6"]
    t12 = enriched_by_name["subgrid_cpw10_N8"]
    subgrid_non_monotone_cpw = {
        "task_11": {
            "index": 11,
            "name": t11["name"],
            "cpw": float(t11["cpw"]),
            "max_level": int(t11["max_level"]),
            "dy_tooth_nm": float(t11["dy_tooth_nm"]),
            "dy_core_nm": float(t11["dy_core_nm"]),
            "dy_coarse_bg_nm": round(float(t11["dy_tooth_nm"]) * (int(t11["ratio"]) ** int(t11["max_level"])), 4),
            "err_lambda_B_nm": float(t11["err_lambda_B_nm"]),
            "rel_err_bw": float(t11["rel_err_bw"]),
            "work_eff": float(t11["work_eff"]),
        },
        "task_12": {
            "index": 12,
            "name": t12["name"],
            "cpw": float(t12["cpw"]),
            "max_level": int(t12["max_level"]),
            "dy_tooth_nm": float(t12["dy_tooth_nm"]),
            "dy_core_nm": float(t12["dy_core_nm"]),
            "dy_coarse_bg_nm": round(float(t12["dy_tooth_nm"]) * (int(t12["ratio"]) ** int(t12["max_level"])), 4),
            "err_lambda_B_nm": float(t12["err_lambda_B_nm"]),
            "rel_err_bw": float(t12["rel_err_bw"]),
            "work_eff": float(t12["work_eff"]),
        },
    }

    summary = {
        "mode": "full",
        "n_tasks": len(enriched_tasks),
        "richardson_limits_nm": {
            "graded_tuned": round(lam_inf_graded_tuned, 6),
            "graded_t18": round(lam_inf_graded_t18, 6),
            "subgrid_lts3": round(lam_inf_subgrid_lts3, 6),
            "pairs": {
                "graded_tuned": [1, 2],
                "graded_t18": [7, 8],
                "subgrid_lts3": [15, 16],
            },
        },
        "richardson_limits_bandwidth_nm": {
            "graded_tuned": round(bw_inf_graded_tuned, 6),
            "graded_t18": round(bw_inf_graded_t18, 6),
            "subgrid_lts3": round(bw_inf_subgrid_lts3, 6),
        },
        "reference": {
            "lambda_B_ref_nm": round(ref_neutral_wl, 6),
            "sigma_ref_nm": round(ref_uncertainty_wl, 6),
            "three_sigma_ref_nm": round(min_valid_threshold_wl, 6),
            "BW_ref_nm": round(ref_neutral_bw, 6),
            "sigma_BW_nm": round(ref_uncertainty_bw, 6),
            "sigma_BW_rel": round(ref_uncertainty_bw / max(ref_neutral_bw, 1e-12), 6),
        },
        "work_eff_to_1p0nm": work_eff_1nm,
        "work_eff_to_0p5nm": work_eff_0p5nm,
        "wall_time_s_to_1p0nm": wall_time_1nm,
        "wall_time_s_to_0p5nm": wall_time_0p5nm,
        "uniform_N4": uniform_summary,
        "verdict_ratios": verdict_ratios,
        "ns_per_cell_step_by_arm": ns_per_cell_step_by_arm,
        "lts3_vs_no_lts_equal_mesh": lts3_vs_no_lts_pairs,
        "subgrid_non_monotone_cpw": subgrid_non_monotone_cpw,
        "references": {
            "neutral_reference": {
                "lambda_B_ref_nm": round(ref_neutral_wl, 4),
                "lambda_B_uncertainty_nm": round(ref_uncertainty_wl, 4),
                "min_valid_dlambda_threshold_3x_sigma_nm": round(min_valid_threshold_wl, 4),
                "active_wavelength_thresholds_nm": active_targets_wl,
                "dropped_wavelength_thresholds_nm": dropped_targets_wl,
                "bandwidth_3db_ref_nm": round(ref_neutral_bw, 4),
                "bandwidth_3db_uncertainty_nm": round(ref_uncertainty_bw, 4),
                "bandwidth_3db_uncertainty_pct": round(ref_uncertainty_bw_pct, 4),
                "min_valid_dbw_threshold_3x_sigma_pct": round(min_valid_threshold_bw_pct, 4),
            },
            "continuum_limit_graded_tuned_nm": round(lam_inf_graded_tuned, 4),
            "continuum_limit_graded_kottke_nm": round(lam_inf_graded_t18, 4),
            "continuum_limit_subgrid_nested_nm": round(lam_inf_subgrid_lts3, 4),
            "continuum_limit_graded_tuned_bandwidth_nm": round(bw_inf_graded_tuned, 4),
            "continuum_limit_graded_bandwidth_nm": round(bw_inf_graded_t18, 4),
            "continuum_limit_subgrid_bandwidth_nm": round(bw_inf_subgrid_lts3, 4),
            "continuum_limit_bandwidth_nm": round(ref_neutral_bw, 4),
            "continuum_limit_R_max": round(ref_common_rmax, 6),
            "subgrid_finest_ref": {
                "name": ref_sg["name"],
                "lambda_B_nm": ref_sg["lambda_B_nm"],
                "bandwidth_3db_nm": ref_sg["bandwidth_3db_nm"],
                "R_max": ref_sg["R_max"],
                "dy_tooth_nm": ref_sg["dy_tooth_nm"],
                "n_cells": ref_sg["n_cells"],
                "n_unknowns": ref_sg["n_unknowns"],
            },
            "graded_finest_run": {
                "name": ref_gr["name"],
                "lambda_B_nm": ref_gr["lambda_B_nm"],
                "bandwidth_3db_nm": ref_gr["bandwidth_3db_nm"],
                "R_max": ref_gr["R_max"],
                "dy_tooth_nm": ref_gr["dy_tooth_nm"],
                "n_cells": ref_gr["n_cells"],
                "n_unknowns": ref_gr["n_unknowns"],
            },
            "graded_tuned_finest_run": {
                "name": ref_gt["name"],
                "lambda_B_nm": ref_gt["lambda_B_nm"],
                "bandwidth_3db_nm": ref_gt["bandwidth_3db_nm"],
                "R_max": ref_gt["R_max"],
                "dy_tooth_nm": ref_gt["dy_tooth_nm"],
                "n_cells": ref_gt["n_cells"],
                "n_unknowns": ref_gt["n_unknowns"],
            },
        },
        "tasks": enriched_tasks,
        "cost_to_target": cost_to_target,
        "cost_to_target_vs_graded_tuned_limit": cost_to_target_vs_graded_tuned,
        "cost_to_target_vs_graded_limit": cost_to_target_vs_graded_t18,
        "cost_to_target_vs_subgrid_limit": cost_to_target_vs_subgrid_lts3,
    }

    if write_json and ref_override_nm is None:
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "analysis_full.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        with open(data_dir / "summary_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


analyze_full_results = compute_summary_and_errors


# ===========================================================================
# Figure Suite Generation (`figures/01_*.png` .. `figures/05_*.png`)
# ===========================================================================
def _plot_01_mesh_zoom(fig_dir: Path) -> Path:
    """Figure 1: Zoomed meshes of each arm around two grating teeth (`x in [-16.25, -15.55] um`,
    `y in [-0.36, +0.36] um`), comparing (a) T18-style anisotropic graded mesh, (b) tuned
    graded mesh with isotropic tooth cells and refined core strip, (c) quadtree subgrid
    with nested core+tooth patches and LTS fine set F, and (d) uniform fine grid.
    """
    domain_small = (-17.0e-6, -14.0e-6, -0.72e-6, 0.72e-6)
    structs, _, _, _ = build_bragg_structure(n_periods=N_PERIODS)
    slab_verts = structs[0].vertices * 1e6  # in um

    ports_s = [
        Port("in", center=(-16.6e-6, 0.0, 0.5 * H_SI), size=(0.0, PORT_WIDTH, 0.8e-6), direction=+1),
        Port("out", center=(-14.4e-6, 0.0, 0.5 * H_SI), size=(0.0, PORT_WIDTH, 0.8e-6), direction=+1),
    ]

    # 1) Graded T18-style Cartesian L1 (cpw=10, N_tooth=4, anisotropic teeth + coarse core)
    dx_bg_10 = WL_MIN / (N_EFF_SLAB * 10.0)
    ov_t18_l1 = build_bragg_mesh_overrides(
        n_periods=N_PERIODS,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=DUTY_CYCLE,
        shift=0.0,
        n_feature=4,
        dx_background=dx_bg_10,
    )
    sim_gr_t18 = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=5,
        structures=structs,
        ports=ports_s,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10.0,
        domain=domain_small,
        n_pml=6,
        cfln=0.85,
        mesh_overrides=ov_t18_l1,
        use_gpu=False,
    )

    # 2) Graded Tuned Cartesian L1 (cpw=10.0, N_tooth=4, dx=dy=12.5 nm in teeth, dy=25 nm in core)
    ov_tuned_l1 = build_tuned_bragg_mesh_overrides(
        n_periods=N_PERIODS,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        shift=0.0,
        n_feature=4,
        dx_background=dx_bg_10,
    )
    sim_gr_tuned = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=5,
        structures=structs,
        ports=ports_s,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10.0,
        domain=domain_small,
        n_pml=6,
        cfln=0.85,
        mesh_overrides=ov_tuned_l1,
        use_gpu=False,
    )

    # 3) Subgrid nested + LTS=3 (cpw=10.6, r=2, L=2, lts=3)
    patches_ne = build_subgrid_patches((-16.2e-6, -14.2e-6), ratio=2, max_level=2, patch_mode="nested")
    sim_ne = SubgridPICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=5,
        structures=structs,
        ports=ports_s,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        cells_per_wavelength=10.6,
        domain=domain_small,
        n_pml=6,
        refine=patches_ne,
        ratio=2,
        max_level=2,
        lts=3,
    )

    # 4) Uniform at tooth resolution (cpw=42.8, L=0)
    sim_un = SubgridPICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=5,
        structures=structs,
        ports=ports_s,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        cells_per_wavelength=42.8,
        domain=domain_small,
        n_pml=6,
        ratio=2,
        max_level=0,
        lts=None,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True, sharey=True)
    x_win = (-16.26, -15.52)
    y_win = (-0.36, 0.36)

    def _draw_tensor_grid(ax_obj, sim_obj, line_col="#3b528b"):
        xg = sim_obj.grid.x * 1e6
        yg = sim_obj.grid.y * 1e6
        xg_sub = xg[(xg >= x_win[0] - 0.05) & (xg <= x_win[1] + 0.05)]
        yg_sub = yg[(yg >= y_win[0] - 0.05) & (yg <= y_win[1] + 0.05)]
        vlines = [[(x, y_win[0]), (x, y_win[1])] for x in xg_sub]
        hlines = [[(x_win[0], y), (x_win[1], y)] for y in yg_sub]
        ax_obj.add_collection(LineCollection(vlines + hlines, colors=line_col, linewidths=0.5, alpha=0.72))
        ax_obj.fill(slab_verts[:, 0], slab_verts[:, 1], color="#f59e0b", alpha=0.28, label="Si grating polygon")
        ax_obj.plot(slab_verts[:, 0], slab_verts[:, 1], color="#b45309", lw=1.8)

    # Panel (a): Graded T18-style tensor-product Cartesian mesh
    ax = axes[0, 0]
    _draw_tensor_grid(ax, sim_gr_t18, line_col="#1e3a8a")
    ax.set_title(
        "(a) Graded T18 Mesh (`graded_t18_L1`, cpw=10, N=4)\n"
        "Anisotropic teeth: $\\Delta x = 40.5\\;\\mathrm{nm},\\;\\Delta y = 12.5\\;\\mathrm{nm}$; "
        "coarse core $\\Delta y = 52.7\\;\\mathrm{nm}$ (58,004 cells)",
        fontsize=9.8,
        fontweight="bold",
    )

    # Panel (b): Graded Tuned tensor-product Cartesian mesh
    ax = axes[0, 1]
    _draw_tensor_grid(ax, sim_gr_tuned, line_col="#0284c7")
    ax.set_title(
        "(b) Tuned Graded Mesh (`graded_tuned_L1`, cpw=10, N=4)\n"
        "Isotropic teeth $\\Delta x = \\Delta y = 12.5\\;\\mathrm{nm}$ + core $\\Delta y = 25\\;\\mathrm{nm}$ "
        "(195,878 cells: $\\Delta x$ extends across all $y$)",
        fontsize=9.8,
        fontweight="bold",
    )

    # Helper to draw QuadMesh leaves in window
    def _draw_quadmesh(ax_obj, sim_obj, show_lts=False):
        mesh = sim_obj.mesh
        c = mesh.cells
        x0_arr = c["x0"] * 1e6
        x1_arr = c["x1"] * 1e6
        y0_arr = c["y0"] * 1e6
        y1_arr = c["y1"] * 1e6
        lev_arr = c["level"]
        level_colors = {0: "#e2e8f0", 1: "#bae6fd", 2: "#fde68a", 3: "#fecaca"}
        rects_by_lev = {0: [], 1: [], 2: [], 3: []}
        in_c = (
            (x1_arr >= x_win[0])
            & (x0_arr <= x_win[1])
            & (y1_arr >= y_win[0])
            & (y0_arr <= y_win[1])
        )
        for idx_c in np.flatnonzero(in_c):
            x0_u, y0_u = float(x0_arr[idx_c]), float(y0_arr[idx_c])
            hx_u = float(x1_arr[idx_c] - x0_arr[idx_c])
            hy_u = float(y1_arr[idx_c] - y0_arr[idx_c])
            lev = int(lev_arr[idx_c])
            rects_by_lev.setdefault(lev, []).append(Rectangle((x0_u, y0_u), hx_u, hy_u))
        for lev, rlist in sorted(rects_by_lev.items()):
            if rlist:
                pc = PatchCollection(
                    rlist,
                    facecolor=level_colors.get(lev, "#fecaca"),
                    edgecolor="#334155",
                    linewidth=0.45,
                    alpha=0.55,
                )
                ax_obj.add_collection(pc)
        ax_obj.fill(slab_verts[:, 0], slab_verts[:, 1], color="#f59e0b", alpha=0.22)
        ax_obj.plot(slab_verts[:, 0], slab_verts[:, 1], color="#b45309", lw=1.8, label="Si tooth boundary")

        if show_lts and sim_obj.solver.lts is not None:
            f_node_ids = sim_obj.solver.dof[sim_obj.solver.lts.F]
            xy_f = mesh.nodes[f_node_ids] * 1e6
            in_win = (
                (xy_f[:, 0] >= x_win[0])
                & (xy_f[:, 0] <= x_win[1])
                & (xy_f[:, 1] >= y_win[0])
                & (xy_f[:, 1] <= y_win[1])
            )
            f_pts = xy_f[in_win]
            ax_obj.scatter(
                f_pts[:, 0],
                f_pts[:, 1],
                s=3.5,
                c="#dc2626",
                alpha=0.75,
                label="LTS fine DOFs $F$ (air gaps, $p=3$ substeps)",
                zorder=5,
            )

    # Panel (c): Subgrid nested core+tooth + LTS=3
    ax = axes[1, 0]
    _draw_quadmesh(ax, sim_ne, show_lts=True)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax.set_title(
        "(c) Subgrid Nested + LTS (`subgrid_lts3_cpw10_N4`, r=2, L=2, lts=3)\n"
        "Core $L=1$ ($24.8\\;\\mathrm{nm}$), teeth $L=2$ ($12.4\\;\\mathrm{nm}$); "
        "cladding $L=0$ (102,530 cells, $1.91\\times$ fewer)",
        fontsize=9.2,
        fontweight="bold",
    )

    # Panel (d): Uniform fine mesh at tooth resolution
    ax = axes[1, 1]
    _draw_quadmesh(ax, sim_un, show_lts=False)
    ax.set_title(
        "(d) Uniform Fine Mesh (`uniform_N4`, cpw=42.8, L=0)\n"
        "Uniform $h = 12.3\\;\\mathrm{nm}$ everywhere (379,353 cells, "
        "$3.70\\times$ more than subgrid)",
        fontsize=9.2,
        fontweight="bold",
    )

    for r_i in range(2):
        for c_i in range(2):
            axes[r_i, c_i].set_xlim(*x_win)
            axes[r_i, c_i].set_ylim(*y_win)
            if r_i == 1:
                axes[r_i, c_i].set_xlabel("Longitudinal position $x$ [$\\mu$m]", fontsize=10.5)
            if c_i == 0:
                axes[r_i, c_i].set_ylabel("Transverse position $y$ [$\\mu$m]", fontsize=10.5)

    fig.suptitle(
        "Bragg Grating Mesh Topology Zoomed on First Two Corrugation Periods ($\\Lambda = 324\\;\\mathrm{nm}$, "
        "$\\Delta w = 50\\;\\mathrm{nm}$)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "01_mesh_zoom_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_02_spectra_and_convergence(
    summary: dict,
    data_dir: Path,
    fig_dir: Path,
) -> Path:
    """Figure 2: Reflection/transmission spectra across arms vs neutral reference, and
    Bragg wavelength convergence ($\\lambda_B$ vs tooth cell size $h_{\\text{tooth}}$)
    demonstrating how tuned graded, T18 graded, and quadtree subgrid converge to the neutral reference.
    """
    tasks_by_name = {t["name"]: t for t in summary["tasks"]}
    spectra_by_name = {}
    for t in summary["tasks"]:
        npz_p = data_dir / f"task_{t['index']}.npz"
        if npz_p.exists():
            z = np.load(npz_p)
            spectra_by_name[t["name"]] = {
                "wl": np.asarray(z["wavelengths_nm"], dtype=float),
                "R": np.asarray(z["R"], dtype=float),
                "T": np.asarray(z["T"], dtype=float),
            }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))

    # Panel (a): Reflection spectra R(lambda) [linear] comparing L1 arms vs references
    ax = axes[0, 0]
    plot_arms = [
        ("subgrid_lts3_ref", "Finest Subgrid+LTS3 N=15.3 (dy=3.27 nm)", "#111827", "-", 2.3),
        ("graded_t18_R", "Finest Graded T18 N=16 (dy=3.125 nm)", "#374151", "--", 1.9),
        ("graded_tuned_L1", "Graded Tuned L1 (N=4, dx=dy=12.5 nm)", "#0284c7", "-", 1.9),
        ("subgrid_lts3_cpw10_N4", "Subgrid+LTS3 (N=4, dx=dy=12.4 nm)", "#16a34a", "-.", 1.9),
        ("uniform_N4", "Uniform Fine N=4.1 (dy=12.3 nm)", "#7c3aed", "--", 1.6),
        ("graded_t18_L1", "Graded T18 L1 (N=4, dx=40.5 nm, coarse core)", "#1e3a8a", ":", 1.8),
        ("graded_tuned_staircase_L1", "Graded Tuned Staircase L1 (N=4)", "#9ca3af", ":", 1.4),
    ]
    for name, label, col, ls, lw in plot_arms:
        if name in spectra_by_name:
            sp = spectra_by_name[name]
            ax.plot(sp["wl"], sp["R"], color=col, linestyle=ls, linewidth=lw, label=label)
    ax.set_xlim(1510, 1605)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Wavelength $\\lambda$ [nm]", fontsize=10.5)
    ax.set_ylabel("Reflectance $R = |S_{11}|^2$", fontsize=10.5)
    ax.set_title("(a) Stopband Reflection Spectra $R(\\lambda)$ (100 Periods, `--full`)", fontsize=11.0, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    # Panel (b): Transmission spectra T(lambda) in dB
    ax = axes[0, 1]
    for name, label, col, ls, lw in plot_arms[:6]:
        if name in spectra_by_name:
            sp = spectra_by_name[name]
            t_db = 10.0 * np.log10(np.clip(sp["T"], 1e-6, None))
            ax.plot(sp["wl"], t_db, color=col, linestyle=ls, linewidth=lw, label=label)
    ax.axhline(-3.0, color="#6b7280", linestyle=":", linewidth=1.0, label="$-3\\;\\mathrm{dB}$ bandwidth level")
    ax.set_xlim(1510, 1605)
    ax.set_ylim(-48, 2)
    ax.set_xlabel("Wavelength $\\lambda$ [nm]", fontsize=10.5)
    ax.set_ylabel("Transmittance $10\\log_{10}|S_{21}|^2$ [dB]", fontsize=10.5)
    ax.set_title("(b) Stopband Transmission Extinction $T(\\lambda)$ [dB]", fontsize=11.0, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    # Panel (c): 100-period Bragg wavelength convergence vs tooth cell size dy_tooth
    ax = axes[1, 0]
    gt_names = ["graded_tuned_L1", "graded_tuned_L2", "graded_tuned_L3"]
    gts_names = ["graded_tuned_staircase_L1", "graded_tuned_staircase_L2"]
    gk_names = ["graded_t18_L1", "graded_t18_L2", "graded_t18_L3", "graded_t18_R"]
    sg_names = ["subgrid_cpw10_N4", "subgrid_cpw14_N6", "subgrid_cpw10_N6", "subgrid_cpw10_N8"]
    sgl_names = ["subgrid_lts3_cpw10_N4", "subgrid_lts3_cpw14_N6", "subgrid_lts3_cpw10_N8", "subgrid_lts3_ref"]

    def _extract_xy(nlist):
        xs = [tasks_by_name[n]["dy_tooth_nm"] for n in nlist if n in tasks_by_name]
        ys = [tasks_by_name[n]["lambda_B_nm"] for n in nlist if n in tasks_by_name]
        return np.array(xs), np.array(ys)

    x_gt, y_gt = _extract_xy(gt_names)
    x_gts, y_gts = _extract_xy(gts_names)
    x_gk, y_gk = _extract_xy(gk_names)
    x_sg, y_sg = _extract_xy(sg_names)
    x_sgl, y_sgl = _extract_xy(sgl_names)

    if x_gt.size:
        ax.plot(x_gt, y_gt, "o-", color="#0284c7", lw=2.2, ms=7.0, label="Graded Tuned Kottke ($h_x = h_y$, core $L=1$)")
    if x_gts.size:
        ax.plot(x_gts, y_gts, "s--", color="#38bdf8", lw=1.7, ms=6.0, label="Graded Tuned Staircase")
    if x_gk.size:
        ax.plot(x_gk, y_gk, "d-.", color="#1e3a8a", lw=1.8, ms=6.0, label="Graded T18 Kottke ($h_x = 3.24\\,h_y$, coarse core)")
    if x_sg.size:
        ax.plot(x_sg, y_sg, "^-", color="#dc2626", lw=2.0, ms=7.0, label="Subgrid Nested (no LTS)")
    if x_sgl.size:
        ax.plot(x_sgl, y_sgl, "v-", color="#16a34a", lw=2.0, ms=7.0, label="Subgrid Nested + LTS=3")
    if "uniform_N4" in tasks_by_name:
        t_un = tasks_by_name["uniform_N4"]
        ax.plot(
            [t_un["dy_tooth_nm"]],
            [t_un["lambda_B_nm"]],
            "P",
            color="#7c3aed",
            ms=9.0,
            label="Uniform Fine Mesh ($L=0$, $12.3\\;\\mathrm{nm}$)",
        )

    refs = summary["references"]
    nref = refs["neutral_reference"]
    lam_ref = nref["lambda_B_ref_nm"]
    sig_ref = nref["lambda_B_uncertainty_nm"]
    ax.axhline(
        lam_ref,
        color="#111827",
        linestyle="-",
        linewidth=1.4,
        label=f"Neutral ref $\\lambda_{{B,\\mathrm{{ref}}}} = {lam_ref:.3f} \\pm {sig_ref:.3f}\\;\\mathrm{{nm}}$",
    )
    ax.axhspan(lam_ref - sig_ref, lam_ref + sig_ref, color="#94a3b8", alpha=0.25)
    ax.set_xlabel("Transverse tooth cell size $\\Delta y_{\\mathrm{tooth}} = \\Delta w / N$ [nm]", fontsize=10.5)
    ax.set_ylabel("Bragg Wavelength $\\lambda_B$ [nm]", fontsize=10.5)
    ax.set_title(
        "(c) 100-Period Convergence: $\\lambda_B$ vs Tooth Cell Size $\\Delta y_{\\mathrm{tooth}}$",
        fontsize=11.0,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    # Panel (d): 100-period O(h^2) Richardson extrapolation vs (dy_tooth)^2
    ax = axes[1, 1]
    if x_gt.size:
        ax.plot(x_gt**2, y_gt, "o-", color="#0284c7", lw=2.2, ms=7.0, label="Graded Tuned Kottke (tasks 0..2)")
    if x_gk.size:
        ax.plot(x_gk**2, y_gk, "d-.", color="#1e3a8a", lw=1.7, ms=6.0, label="Graded T18 Kottke (tasks 5..8)")
    if x_sgl.size:
        ax.plot(x_sgl**2, y_sgl, "v-", color="#16a34a", lw=2.0, ms=7.0, label="Subgrid + LTS=3 (tasks 13..16)")

    lam_inf_gt = refs["continuum_limit_graded_tuned_nm"]
    lam_inf_g = refs["continuum_limit_graded_kottke_nm"]
    lam_inf_s = refs["continuum_limit_subgrid_nested_nm"]
    ax.axhline(
        lam_inf_gt,
        color="#0284c7",
        linestyle="--",
        lw=1.3,
        label=f"Graded Tuned limit $\\lambda_{{B,\\infty}}^{{\\mathrm{{tuned}}}} = {lam_inf_gt:.3f}\\;\\mathrm{{nm}}$",
    )
    ax.axhline(
        lam_inf_g,
        color="#1e3a8a",
        linestyle="--",
        lw=1.3,
        label=f"Graded T18 limit $\\lambda_{{B,\\infty}}^{{\\mathrm{{t18}}}} = {lam_inf_g:.3f}\\;\\mathrm{{nm}}$",
    )
    ax.axhline(
        lam_inf_s,
        color="#16a34a",
        linestyle="--",
        lw=1.3,
        label=f"Subgrid+LTS3 limit $\\lambda_{{B,\\infty}}^{{\\mathrm{{subgrid}}}} = {lam_inf_s:.3f}\\;\\mathrm{{nm}}$",
    )
    ax.axhline(
        lam_ref,
        color="#111827",
        linestyle=":",
        lw=1.6,
        label=f"Neutral reference $\\lambda_{{B,\\mathrm{{ref}}}} = {lam_ref:.3f} \\pm {sig_ref:.3f}\\;\\mathrm{{nm}}$",
    )
    ax.axhspan(lam_inf_s, lam_inf_g, color="#94a3b8", alpha=0.22)
    ax.set_xlabel("Squared tooth grid spacing $(\\Delta y_{\\mathrm{tooth}})^2$ [$\\mathrm{nm}^2$]", fontsize=10.5)
    ax.set_ylabel("Bragg Wavelength $\\lambda_B$ [nm] (100 periods)", fontsize=10.5)
    ax.set_title(
        "(d) 100-Period $O(h^2)$ Richardson Limits & Neutral Reference Band",
        fontsize=11.0,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    fig.suptitle(
        "Spectral Accuracy and Discretization Convergence: Tuned Graded vs T18 Graded vs Quadtree Subgrid (`--full`)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = fig_dir / "02_spectra_vs_reference.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_03_error_vs_work(summary: dict, fig_dir: Path) -> Path:
    """Figure 3: Log-log plots of accuracy metrics ($|\\lambda_B - \\lambda_{B,\\mathrm{ref}}|$,
    bandwidth error %, and spectrum RMS error) vs effective work ``work_eff``
    (one curve per arm, ``uniform_N4`` as a point, target lines at $1.0\\;\\mathrm{nm}$ and $0.5\\;\\mathrm{nm}$).
    """
    tasks_by_name = {t["name"]: t for t in summary["tasks"]}
    nref = summary["references"]["neutral_reference"]
    lam_ref = nref["lambda_B_ref_nm"]
    sig_ref = nref["lambda_B_uncertainty_nm"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5))

    series_cfg = [
        (
            ["graded_tuned_L1", "graded_tuned_L2", "graded_tuned_L3"],
            "Graded Tuned Kottke ($h_x=h_y$, core $L=1$)",
            "#0284c7",
            "o-",
            "work_eff",
        ),
        (
            ["graded_tuned_staircase_L1", "graded_tuned_staircase_L2"],
            "Graded Tuned Staircase",
            "#38bdf8",
            "s--",
            "work_eff",
        ),
        (
            ["graded_t18_L1", "graded_t18_L2", "graded_t18_L3", "graded_t18_R"],
            "Graded T18 Kottke (anisotropic)",
            "#1e3a8a",
            "d-.",
            "work_eff",
        ),
        (
            ["subgrid_cpw10_N4", "subgrid_cpw14_N6", "subgrid_cpw10_N6", "subgrid_cpw10_N8"],
            "Subgrid Nested (no LTS)",
            "#dc2626",
            "^-",
            "work_eff",
        ),
        (
            ["subgrid_lts3_cpw10_N4", "subgrid_lts3_cpw14_N6", "subgrid_lts3_cpw10_N8", "subgrid_lts3_ref"],
            "Subgrid + LTS=3 (`work_eff`)",
            "#16a34a",
            "v-",
            "work_eff",
        ),
        (
            ["uniform_N4"],
            "Uniform Fine ($12.3\\;\\mathrm{nm}$ everywhere)",
            "#7c3aed",
            "P",
            "work_eff",
        ),
    ]

    # Panel (a): |lambda_B - ref| vs Effective Work (log-log)
    ax = axes[0]
    for names, label, col, fmt, w_key in series_cfg:
        pts = [tasks_by_name[n] for n in names if n in tasks_by_name]
        if not pts:
            continue
        w = [p[w_key] for p in pts]
        e = [max(p["err_lambda_B_nm"], 0.01) for p in pts]
        ms_val = 9.5 if len(pts) == 1 else 6.8
        ax.loglog(w, e, fmt, color=col, lw=2.0, ms=ms_val, label=label)
    for thresh, tcol in [(1.0, "#334155"), (0.5, "#475569")]:
        ax.axhline(thresh, color=tcol, linestyle=":", lw=1.2)
        ax.text(8.5e9, thresh * 1.08, f"Target $|\\Delta\\lambda_B| = {thresh:.1f}\\;\\mathrm{{nm}}$", fontsize=8.0, color=tcol)
    ax.axhspan(0.01, sig_ref, color="#cbd5e1", alpha=0.35, label=f"$\\sigma_{{\\mathrm{{ref}}}} = {sig_ref:.3f}\\;\\mathrm{{nm}}$ band")
    ax.set_ylim(0.01, 20.0)
    ax.set_xlabel("Effective Work $\\mathrm{work\\_eff}$ [unknown-steps]", fontsize=10.0)
    ax.set_ylabel("Bragg Wavelength Error $|\\lambda_B - \\lambda_{B,\\mathrm{ref}}|$ [nm]", fontsize=10.0)
    ax.set_title("(a) $|\\Delta\\lambda_B|$ vs Effective Work (`work_eff`)", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    # Panel (b): -3 dB Bandwidth Error (%) vs Effective Work
    ax = axes[1]
    for names, label, col, fmt, w_key in series_cfg:
        pts = [tasks_by_name[n] for n in names if n in tasks_by_name]
        if not pts:
            continue
        w = [p[w_key] for p in pts]
        e = [max(p["err_bw_pct"], 0.05) for p in pts]
        ms_val = 9.5 if len(pts) == 1 else 6.8
        ax.loglog(w, e, fmt, color=col, lw=2.0, ms=ms_val, label=label)
    ax.axhline(1.0, color="#334155", linestyle=":", lw=1.2, label="Reference $|\\Delta\\mathrm{BW}| / \\mathrm{BW}_{\\mathrm{ref}} = 1\\%$")
    ax.set_xlabel("Effective Work $\\mathrm{work\\_eff}$ [unknown-steps]", fontsize=10.0)
    ax.set_ylabel("$-3\\;\\mathrm{dB}$ Bandwidth Error $|\\Delta\\mathrm{BW}| / \\mathrm{BW}_{\\mathrm{ref}}$ [%]", fontsize=10.0)
    ax.set_title("(b) Bandwidth Error [%] vs Effective Work", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    # Panel (c): Spectrum RMS Error vs Effective Work
    ax = axes[2]
    for names, label, col, fmt, w_key in series_cfg:
        pts = [tasks_by_name[n] for n in names if n in tasks_by_name]
        if not pts:
            continue
        w = [p[w_key] for p in pts]
        e = [max(p["err_spectrum_R_rms"], 1e-3) for p in pts]
        ms_val = 9.5 if len(pts) == 1 else 6.8
        ax.loglog(w, e, fmt, color=col, lw=2.0, ms=ms_val, label=label)
    ax.set_xlabel("Effective Work $\\mathrm{work\\_eff}$ [unknown-steps]", fontsize=10.0)
    ax.set_ylabel("Band-Averaged Spectrum RMS Error $\\mathrm{RMS}(R - R_{\\mathrm{ref}})$", fontsize=10.0)
    ax.set_title("(c) Full-Band Spectrum RMS Error vs Effective Work", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    fig.suptitle(
        f"Algorithmic Cost vs Accuracy (`--full`, Neutral Reference $\\lambda_{{B,\\mathrm{{ref}}}} = {lam_ref:.3f} \\pm {sig_ref:.3f}\\;\\mathrm{{nm}}$)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = fig_dir / "03_error_vs_work.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_04_error_vs_walltime(summary: dict, fig_dir: Path) -> Path:
    """Figure 4: Log-log plots of accuracy metrics vs single-core wall time [s]
    and cost-to-target comparison bar chart at $1.0\\;\\mathrm{nm}$ and $0.5\\;\\mathrm{nm}$.
    """
    tasks_by_name = {t["name"]: t for t in summary["tasks"]}
    nref = summary["references"]["neutral_reference"]
    sig_ref = nref["lambda_B_uncertainty_nm"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5))

    series_cfg = [
        (
            ["graded_tuned_L1", "graded_tuned_L2", "graded_tuned_L3"],
            "Graded Tuned Kottke ($h_x=h_y$, core $L=1$)",
            "#0284c7",
            "o-",
        ),
        (
            ["graded_tuned_staircase_L1", "graded_tuned_staircase_L2"],
            "Graded Tuned Staircase",
            "#38bdf8",
            "s--",
        ),
        (
            ["graded_t18_L1", "graded_t18_L2", "graded_t18_L3", "graded_t18_R"],
            "Graded T18 Kottke (anisotropic)",
            "#1e3a8a",
            "d-.",
        ),
        (
            ["subgrid_cpw10_N4", "subgrid_cpw14_N6", "subgrid_cpw10_N6", "subgrid_cpw10_N8"],
            "Subgrid Nested (no LTS)",
            "#dc2626",
            "^-",
        ),
        (
            ["subgrid_lts3_cpw10_N4", "subgrid_lts3_cpw14_N6", "subgrid_lts3_cpw10_N8", "subgrid_lts3_ref"],
            "Subgrid + LTS=3",
            "#16a34a",
            "v-",
        ),
        (
            ["uniform_N4"],
            "Uniform Fine ($12.3\\;\\mathrm{nm}$ everywhere)",
            "#7c3aed",
            "P",
        ),
    ]

    # Panel (a): |lambda_B - ref| vs 1-core Wall Time [s]
    ax = axes[0]
    for names, label, col, fmt in series_cfg:
        pts = [tasks_by_name[n] for n in names if n in tasks_by_name]
        if not pts:
            continue
        t_s = [p["total_wall_time_s"] for p in pts]
        e = [max(p["err_lambda_B_nm"], 0.01) for p in pts]
        ms_val = 9.5 if len(pts) == 1 else 6.8
        ax.loglog(t_s, e, fmt, color=col, lw=2.0, ms=ms_val, label=label)
    for thresh, tcol in [(1.0, "#334155"), (0.5, "#475569")]:
        ax.axhline(thresh, color=tcol, linestyle=":", lw=1.2)
        ax.text(135.0, thresh * 1.08, f"Target $|\\Delta\\lambda_B| = {thresh:.1f}\\;\\mathrm{{nm}}$", fontsize=8.0, color=tcol)
    ax.axhspan(0.01, sig_ref, color="#cbd5e1", alpha=0.35, label=f"$\\sigma_{{\\mathrm{{ref}}}} = {sig_ref:.3f}\\;\\mathrm{{nm}}$ band")
    ax.set_ylim(0.01, 20.0)
    ax.set_xlabel("Single-Core Wall Time [s] (`OMP_NUM_THREADS=1`)", fontsize=10.0)
    ax.set_ylabel("Bragg Wavelength Error $|\\lambda_B - \\lambda_{B,\\mathrm{ref}}|$ [nm]", fontsize=10.0)
    ax.set_title("(a) $|\\Delta\\lambda_B|$ vs Single-Core Wall Time [s]", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    # Panel (b): Spectrum RMS Error vs 1-core Wall Time [s]
    ax = axes[1]
    for names, label, col, fmt in series_cfg:
        pts = [tasks_by_name[n] for n in names if n in tasks_by_name]
        if not pts:
            continue
        t_s = [p["total_wall_time_s"] for p in pts]
        e = [max(p["err_spectrum_R_rms"], 1e-3) for p in pts]
        ms_val = 9.5 if len(pts) == 1 else 6.8
        ax.loglog(t_s, e, fmt, color=col, lw=2.0, ms=ms_val, label=label)
    ax.set_xlabel("Single-Core Wall Time [s] (`OMP_NUM_THREADS=1`)", fontsize=10.0)
    ax.set_ylabel("Spectrum RMS Error $\\mathrm{RMS}(R - R_{\\mathrm{ref}})$", fontsize=10.0)
    ax.set_title("(b) Spectrum RMS Error vs Single-Core Wall Time [s]", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    # Panel (c): Effective work to reach target accuracy thresholds (1.0 nm and 0.5 nm)
    ax = axes[2]
    ct = summary.get("cost_to_target", {})
    u_work = float(summary["uniform_N4"]["work_eff"])
    arms_bar = [
        ("graded_tuned", "Graded Tuned\nKottke", "#0284c7"),
        ("graded_t18", "Graded T18\nKottke", "#1e3a8a"),
        ("subgrid", "Subgrid\n(no LTS)", "#dc2626"),
        ("subgrid_lts3", "Subgrid\n+ LTS=3", "#16a34a"),
        ("uniform", "Uniform N=4\n(0.11 nm pt)", "#7c3aed"),
    ]
    t_keys = [
        ("dlambda_le_1.0nm", "$|\\Delta\\lambda_B| \\leq 1.0\\;\\mathrm{nm}$"),
        ("dlambda_le_0.5nm", "$|\\Delta\\lambda_B| \\leq 0.5\\;\\mathrm{nm}$"),
    ]
    x_pos = np.arange(len(t_keys))
    bar_w = 0.15
    offset_mid = 0.5 * (len(arms_bar) - 1)
    for i_a, (a_key, a_lbl, a_col) in enumerate(arms_bar):
        vals = []
        for tk, _ in t_keys:
            if a_key == "uniform":
                vals.append(u_work)
            else:
                v = ct.get(a_key, {}).get(tk, {}).get("work_eff")
                vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
        bars = ax.bar(x_pos + (i_a - offset_mid) * bar_w, vals, width=bar_w, color=a_col, alpha=0.88, label=a_lbl)
        for j_b, (b, v) in enumerate(zip(bars, vals)):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() * 0.5,
                    v * 1.06,
                    f"{v / 1e9:.1f}G",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    rotation=20,
                )
            else:
                ax.text(
                    x_pos[j_b] + (i_a - offset_mid) * bar_w,
                    1.25e10,
                    "not\nreached",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=a_col,
                    fontweight="bold",
                )
    ax.set_yscale("log")
    ax.set_ylim(1.0e10, 1.4e12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([lbl for _, lbl in t_keys], fontsize=9.5)
    ax.set_ylabel("Effective Work $\\mathrm{work\\_eff}$ to Target (log scale)", fontsize=10.0)
    ax.set_title("(c) Effective Work $\\mathrm{work\\_eff}$ to $1.0\\;\\mathrm{nm}$ & $0.5\\;\\mathrm{nm}$", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.92, ncol=2)

    fig.suptitle(
        "Wall-Clock Cost vs Accuracy on 1 CPU Core & Effective Work to Target Thresholds (full mode)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = fig_dir / "04_error_vs_walltime.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_05_field_map(data_dir: Path, fig_dir: Path) -> Path | None:
    """Figure 5: 2D DFT field maps on the subgridded Bragg grating at the stopband
    Bragg wavelength ($\\lambda_B \\approx 1553.8\\;\\mathrm{nm}$, exponential decay)
    and passband ($\\lambda = 1515\\;\\mathrm{nm}$, full transmission), plus zoomed
    high-resolution mode interference inside the first 8 grating periods.
    """
    diag_npz = data_dir / "task_19.npz"
    if not diag_npz.exists():
        return None
    z = np.load(diag_npz)
    x_full = np.asarray(z["x_full"], dtype=float) * 1e6
    y_full = np.asarray(z["y_full"], dtype=float) * 1e6
    E2_bragg = np.asarray(z["E2_bragg"], dtype=float)
    E2_pass = np.asarray(z["E2_pass"], dtype=float)

    x_zoom = np.asarray(z["x_zoom"], dtype=float) * 1e6
    y_zoom = np.asarray(z["y_zoom"], dtype=float) * 1e6
    Ey_z_bragg = np.asarray(z["Ey_zoom_bragg"], dtype= np.complex64)
    Ex_z_bragg = np.asarray(z["Ex_zoom_bragg"], dtype=np.complex64)
    E2_z_bragg = np.abs(Ex_z_bragg) ** 2 + np.abs(Ey_z_bragg) ** 2

    structs, _, _, _ = build_bragg_structure(n_periods=N_PERIODS)
    verts_um = structs[0].vertices * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.2))

    # Panel (a): Full-device |E|^2 at stopband center lambda_B = 1553.8 nm
    ax = axes[0]
    norm_b = max(float(np.percentile(E2_bragg, 99.5)), 1e-30)
    im0 = ax.pcolormesh(
        x_full,
        y_full,
        (E2_bragg / norm_b).T,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    ax.plot(verts_um[:, 0], verts_um[:, 1], color="#38bdf8", lw=0.6, alpha=0.75)
    ax.set_xlim(-16.8, 16.8)
    ax.set_ylim(-0.65, 0.65)
    ax.set_ylabel("$y$ [$\\mu$m]", fontsize=10.0)
    ax.set_title(
        "(a) Stopband Resonance ($\\lambda = \\lambda_B = 1553.8\\;\\mathrm{nm}$): "
        "Normalized $|\\mathbf{E}(x,y)|^2$ Evanescent Envelope Decay Along Grating",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.colorbar(im0, ax=ax, pad=0.012, label="$|\\mathbf{E}|^2 / |\\mathbf{E}|_{\\max}^2$")

    # Panel (b): Full-device |E|^2 at passband lambda = 1515.0 nm
    ax = axes[1]
    norm_p = max(float(np.percentile(E2_pass, 99.5)), 1e-30)
    im1 = ax.pcolormesh(
        x_full,
        y_full,
        (E2_pass / norm_p).T,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    ax.plot(verts_um[:, 0], verts_um[:, 1], color="#38bdf8", lw=0.6, alpha=0.75)
    ax.set_xlim(-16.8, 16.8)
    ax.set_ylim(-0.65, 0.65)
    ax.set_ylabel("$y$ [$\\mu$m]", fontsize=10.0)
    ax.set_title(
        "(b) Passband Transmission ($\\lambda = 1515.0\\;\\mathrm{nm}$): "
        "Normalized $|\\mathbf{E}(x,y)|^2$ Guided Wave Propagating to Output Port",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.colorbar(im1, ax=ax, pad=0.012, label="$|\\mathbf{E}|^2 / |\\mathbf{E}|_{\\max}^2$")

    # Panel (c): Zoomed Re(Ey) standing wave on the first 8 grating teeth at lambda_B
    ax = axes[2]
    re_ey = np.real(Ey_z_bragg)
    v_lim = max(float(np.percentile(np.abs(re_ey), 99.5)), 1e-30)
    im2 = ax.pcolormesh(
        x_zoom,
        y_zoom,
        (re_ey / v_lim).T,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    ax.plot(verts_um[:, 0], verts_um[:, 1], color="#111827", lw=1.4, label="Si corrugated boundary")
    ax.set_xlim(float(x_zoom[0]), float(x_zoom[-1]))
    ax.set_ylim(-0.40, 0.40)
    ax.set_xlabel("Longitudinal position $x$ [$\\mu$m]", fontsize=10.5)
    ax.set_ylabel("$y$ [$\\mu$m]", fontsize=10.0)
    ax.set_title(
        "(c) Zoomed Standing-Wave Field $\\mathrm{Re}(E_y)$ at $\\lambda_B = 1553.8\\;\\mathrm{nm}$ "
        "Across First 8 Corrugation Periods (`SubgridPICSimulation`, $N_{\\mathrm{tooth}}=4$, $\\mathrm{lts}=3$)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    fig.colorbar(im2, ax=ax, pad=0.012, label="$\\mathrm{Re}(E_y) / |E_y|_{\\max}$")

    fig.suptitle(
        "Subgrid Frequency-Domain (DFT) Field Maps on 100-Period Silicon Bragg Grating",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = fig_dir / "05_field_map_bragg.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> dict:
    """Compute summary metrics and generate all 5 study figures in ``fig_dir``."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_summary_and_errors(data_dir=data_dir)
    f1 = _plot_01_mesh_zoom(fig_dir=fig_dir)
    f2 = _plot_02_spectra_and_convergence(summary=summary, data_dir=data_dir, fig_dir=fig_dir)
    f3 = _plot_03_error_vs_work(summary=summary, fig_dir=fig_dir)
    f4 = _plot_04_error_vs_walltime(summary=summary, fig_dir=fig_dir)
    f5 = _plot_05_field_map(data_dir=data_dir, fig_dir=fig_dir)
    print(f"Generated figures in {fig_dir}:", [str(p.name) for p in (f1, f2, f3, f4, f5) if p is not None])
    return summary


def assert_full_verdict(summary: dict) -> None:
    """Assert the ``--full`` study conclusions from ``summary`` (or ``data/analysis_full.json``).

    Raises ``AssertionError`` if any Richardson limit, reference value, interpolated
    ``work_eff``, or named-baseline ratio deviates from the ``--full`` results.
    """
    lims = summary["richardson_limits_nm"]
    ref = summary["reference"]
    w1 = summary["work_eff_to_1p0nm"]
    w05 = summary["work_eff_to_0p5nm"]
    u4 = summary["uniform_N4"]

    assert np.isclose(lims["graded_tuned"], 1551.311, rtol=0, atol=2e-3)
    assert np.isclose(lims["graded_t18"], 1551.418, rtol=0, atol=2e-3)
    assert np.isclose(lims["subgrid_lts3"], 1551.197, rtol=0, atol=2e-3)
    assert np.isclose(ref["lambda_B_ref_nm"], 1551.308, rtol=0, atol=2e-3)
    assert np.isclose(ref["sigma_ref_nm"], 0.110, rtol=0, atol=2e-3)

    # work_eff to 1.0 nm
    assert np.isclose(w1["graded_tuned"], 9.31e10, rtol=5e-3)
    assert np.isclose(w1["graded_t18"], 3.83e11, rtol=5e-3)
    assert np.isclose(w1["subgrid"], 6.87e10, rtol=5e-3)
    assert np.isclose(w1["subgrid_lts3"], 2.58e10, rtol=5e-3)
    assert w1["graded_tuned_staircase"] == "not reached"

    # work_eff to 0.5 nm
    assert np.isclose(w05["graded_tuned"], 2.49e11, rtol=5e-3)
    assert w05["graded_t18"] == "not reached"
    assert np.isclose(w05["subgrid"], 1.02e11, rtol=5e-3)
    assert np.isclose(w05["subgrid_lts3"], 3.86e10, rtol=5e-3)
    assert w05["graded_tuned_staircase"] == "not reached"

    # uniform_N4 single point (12.3 nm everywhere)
    assert np.isclose(u4["dy_tooth_nm"], 12.31, atol=0.05)
    assert np.isclose(u4["err_lambda_B_nm"], 0.1145, atol=0.01)
    assert np.isclose(u4["work_eff"], 7.86e10, rtol=5e-3)

    # Subgrid without LTS requires more work to reach 0.5 nm than uniform_N4 (which already achieves 0.11 nm)
    assert w05["subgrid"] > u4["work_eff"]

    # LTS3 vs no-LTS wavelength shift at equal mesh (tasks 12 vs 15) is ~0.11 nm (~sigma_ref)
    pair_12_15 = next(
        p for p in summary["lts3_vs_no_lts_equal_mesh"] if p["task_no_lts"] == 12 and p["task_lts3"] == 15
    )
    assert np.isclose(abs(pair_12_15["delta_lambda_B_nm"]), 0.1051, atol=0.01)
    assert 0.5 <= pair_12_15["abs_delta_over_sigma_ref"] <= 1.5

    # Subgrid arm is non-monotone in cpw between tasks 11 (cpw=15.6) and 12 (cpw=10.6)
    nm_cpw = summary["subgrid_non_monotone_cpw"]
    assert nm_cpw["task_11"]["cpw"] > nm_cpw["task_12"]["cpw"]
    assert nm_cpw["task_12"]["rel_err_bw"] > nm_cpw["task_11"]["rel_err_bw"]


def run_self_tests(data_dir: Path = DATA_DIR) -> None:
    """Run unit and mutation checks on tuned mesh overrides, full-mode Richardson
    reference and verdict numbers, and log-log bracketing cost interpolation.
    """
    # 1. Verify build_tuned_bragg_mesh_overrides produces 4 overrides:
    #    two tooth y-strips (hy=12.5 nm), one core y-strip (hy=25.0 nm), and
    #    one grating x-span (hx=12.5 nm == hy_tooth)
    ov_tuned = build_tuned_bragg_mesh_overrides(
        n_periods=10,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        n_feature=4,
    )
    assert len(ov_tuned) == 4, f"Expected 4 overrides (2 tooth y-strips, 1 core y-strip, 1 x-span), got {len(ov_tuned)}"
    hy_tooth_lo = ov_tuned[0][2][1]
    hy_tooth_hi = ov_tuned[1][2][1]
    hy_core = ov_tuned[2][2][1]
    hx_tooth = ov_tuned[3][2][0]
    assert np.isclose(hy_tooth_lo, 12.5e-9, rtol=1e-12) and np.isclose(hy_tooth_hi, 12.5e-9, rtol=1e-12)
    assert np.isclose(hx_tooth, hy_tooth_lo, rtol=1e-12), f"Tuned dx_tooth ({hx_tooth}) must equal dy_tooth ({hy_tooth_lo})"
    assert np.isclose(hy_core, 25.0e-9, rtol=1e-12), f"Tuned dy_core ({hy_core}) must equal 2 * dy_tooth (25.0 nm)"

    # Mutation check 1: T18-style overrides have anisotropic dx_tooth != dy_tooth and no core strip override
    ov_t18 = build_bragg_mesh_overrides(
        n_periods=10,
        pitch=PITCH,
        w_core=W_CORE,
        dw=DW_CORRUG,
        duty_cycle=DUTY_CYCLE,
        n_feature=4,
    )
    mut1_caught = False
    try:
        t18_hy_tooth = ov_t18[0][2][1]
        t18_hx_tooth = ov_t18[2][2][0]
        assert len(ov_t18) == 4 and np.isclose(t18_hx_tooth, t18_hy_tooth, rtol=1e-6)
    except AssertionError:
        mut1_caught = True
    assert mut1_caught, "Mutation check 1 failed: anisotropic T18 override was not caught"

    # 2. Verify log-log interpolation on exact power-law curve E(W) = 2.0 * (W / 1e10)^(-1)
    #    and verify no extrapolation when target is not bracketed.
    w_interp, st_interp = _interpolate_cost_for_target(
        np.array([2.0, 0.5]),
        np.array([1.0e10, 4.0e10]),
        1.0,
        labels=["N=4.0", "N=8.0"],
        unit="nm",
    )
    assert w_interp is not None and st_interp.startswith("interpolated")
    assert np.isclose(w_interp, 2.0e10, rtol=1e-12), f"Expected 2.0e10, got {w_interp}"

    w_unreached, st_unreached = _interpolate_cost_for_target(
        np.array([2.0, 0.8]),
        np.array([1.0e10, 4.0e10]),
        0.5,
        labels=["N=4.0", "N=8.0"],
        unit="nm",
    )
    assert w_unreached is None and st_unreached == "not reached"

    # Mutation check 2: linear interpolation on (1e10, 2.0) -> (4e10, 0.5) at E=1.0 gives 3.0e10 != 2.0e10
    mut2_caught = False
    try:
        w_linear_mut = 1.0e10 + (2.0 - 1.0) / (2.0 - 0.5) * (4.0e10 - 1.0e10)
        assert np.isclose(w_linear_mut, w_interp, rtol=1e-3)
    except AssertionError:
        mut2_caught = True
    assert mut2_caught, "Mutation check 2 failed: linear vs log-log interpolation was not distinguished"

    # 3. Verify full-mode conclusions and mutation check (+1 nm reference perturbation fails)
    summary = compute_summary_and_errors(data_dir=data_dir)
    assert_full_verdict(summary)

    mut3_caught = False
    try:
        mut_summary = compute_summary_and_errors(
            data_dir=data_dir,
            ref_override_nm=summary["reference"]["lambda_B_ref_nm"] + 1.0,
            write_json=False,
        )
        assert_full_verdict(mut_summary)
    except AssertionError:
        mut3_caught = True
    assert mut3_caught, "Mutation check 3 failed: +1 nm reference mutation did not fail verdict assertions"

    ref = summary["reference"]
    w1 = summary["work_eff_to_1p0nm"]
    w05 = summary["work_eff_to_0p5nm"]
    print(
        "ALL SELF-TESTS & MUTATION CHECKS PASSED:",
        f"ref={ref['lambda_B_ref_nm']:.3f} +/- {ref['sigma_ref_nm']:.3f} nm,",
        f"work_eff(1nm)={w1},",
        f"work_eff(0.5nm)={w05}",
    )


# ===========================================================================
# CLI Entrypoint
# ===========================================================================
def parse_args(argv=None):
    """Parse CLI arguments for the subgrid vs graded Bragg grating study."""
    ap = argparse.ArgumentParser(
        description="Bragg grating (PIC T18) on the subgridded mesh vs the graded mesh — cost at equal accuracy"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick 2D benchmark study locally and render all figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Select full-resolution converged study task table (for Cloud Batch)",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task I (0-indexed) at --quick or --full resolution",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for --task JSON/NPZ files (default: data/)",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze --full task JSONs, write data/analysis_full.json, and print summary",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.json/npz and generate analysis_full.json + all figures",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the task table (with preflight comments) and exit",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="Print preflight resource table for both --quick and --full and exit",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run unit and mutation checks on study logic and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    return ap.parse_args(argv)


def main(argv=None):
    """Main CLI entrypoint."""
    args = parse_args(argv)

    if args.analyze:
        summary = compute_summary_and_errors(data_dir=args.out, write_json=True)
        lims = summary["richardson_limits_nm"]
        ref = summary["reference"]
        print(
            f"Richardson limits [nm]: graded_tuned={lims['graded_tuned']:.3f}, "
            f"graded_t18={lims['graded_t18']:.3f}, subgrid_lts3={lims['subgrid_lts3']:.3f}"
        )
        print(
            f"Neutral reference: lambda_B_ref = {ref['lambda_B_ref_nm']:.3f} +- {ref['sigma_ref_nm']:.3f} nm, "
            f"BW_ref = {ref['BW_ref_nm']:.3f} +- {ref['sigma_BW_nm']:.3f} nm"
        )
        print("work_eff to 1.0 nm:", summary["work_eff_to_1p0nm"])
        print("work_eff to 0.5 nm:", summary["work_eff_to_0p5nm"])
        print("uniform_N4:", summary["uniform_N4"])

    if args.self_test:
        run_self_tests(data_dir=args.out)

    if args.plot and not (args.quick or args.full):
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)

    if args.analyze or args.self_test or (args.plot and not (args.quick or args.full)):
        return

    if args.preflight:
        for mode_q, label in [(True, "QUICK"), (False, "FULL")]:
            est = compute_preflight_estimates(quick=mode_q)
            tot_cpu = sum(e["wall_time_1core_est_s"] for e in est)
            print(f"# === {label} PREFLIGHT ({len(est)} tasks, est 1-core sum = {tot_cpu:.0f}s = {tot_cpu/60:.1f} min) ===")
            print("# idx\tname\tn_cells\tn_unknowns\tdy_tooth_nm\tn_steps\twork_raw\twall_1core_s\trss_mb")
            for e in est:
                print(
                    f"# {e['index']:2d}\t{e['name']:28s}\t{e['n_cells']:8d}\t{e['n_unknowns']:9d}\t"
                    f"{e['dy_tooth_nm']:6.2f}\t{e['n_steps_est']:7d}\t{e['work_raw_est']:10.3e}\t"
                    f"{e['wall_time_1core_est_s']:7.1f}\t{e['peak_rss_est_mb']:6.1f}"
                )
        return

    if args.list_tasks:
        quick_mode = not args.full
        tasks = get_task_table(quick=quick_mode)
        est = compute_preflight_estimates(quick=quick_mode)
        tot_cpu = sum(e["wall_time_1core_est_s"] for e in est)
        print(f"# Total tasks: {len(tasks)} (mode={'quick' if quick_mode else 'full'}, est 1-core total={tot_cpu:.0f}s)")
        for t, e in zip(tasks, est):
            print(
                f"# preflight task {t['index']}: cells={e['n_cells']}, unknowns={e['n_unknowns']}, "
                f"steps~{e['n_steps_est']}, work~{e['work_raw_est']:.2e}, est_1core={e['wall_time_1core_est_s']:.0f}s"
            )
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        quick_mode = not args.full
        tasks = get_task_table(quick=quick_mode)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot and not (args.quick or args.full):
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    # Default when invoked with --quick or no flags: run all quick tasks and plot
    quick_mode = not args.full
    tasks = get_task_table(quick=quick_mode)
    for t in tasks:
        run_single_task(t, args.out)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
