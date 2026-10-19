# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Arm adapter interface and runner.

An *arm* is one mesh/solver combination (e.g. "Cartesian conformal, graded"
or "triangles, graded, Voronoi"). It plugs into the suite by subclassing
`Arm` and implementing

* ``supports(case) -> bool``: whether the arm can run this case;
* ``resolutions(case) -> list``: its resolution ladder (any hashable labels,
  coarse to fine);
* ``run(case, resolution) -> dict``: one simulation. For 2D scattering cases
  the dict holds either

  - ``samples`` (n_freq, n_radii, n_points): Wakis DFT phasors (kernel
    exp(-i w t), i.e. exp(+i w t) phasors) of the *scattered* E_z (TM) or
    H_z (TE) at ``wakis.cylinder_harmonics.circle_points(r, case.n_points)``
    for every r in ``radii``, plus ``u0`` (n_freq,): the incident phasor at
    the origin, in the same time levels as the samples; or
  - ``c_ext``, ``c_sca`` (n_freq,) widths [m] (flux-based arms);

  and, for cases with a ``gap_point``: ``e_gap`` (n_freq, 2) total in-plane
  E phasors there and ``e0_gap`` (n_freq,) the incident amplitude phasor.
  For the dipole case B4: ``z_in`` (n_freq,) input impedance [Ohm]
  (exp(+i w t): R + i X) and optionally ``directivity`` (scalar, broadside
  at the resonance) and ``pattern`` (theta, D(theta)).
  Always: ``cost``, a dict with ``unknowns`` (field degrees of freedom),
  ``steps``, ``dt`` [s], ``run_s`` (time-stepping wall time [s], one core,
  uncontended), ``setup_s`` and any arm-specific fields (cells, h_min...).

`run` executes one arm at one resolution and returns the metrics of
``case.evaluate`` together with the cost; `cost_to_target` turns a ladder of
such records into the unknowns / work / time needed to reach each target.
"""

import json
import os
import time
import traceback

import numpy as np

from . import metrics as M

__all__ = ["Arm", "run", "cost_to_target", "save_record", "load_records"]


class Arm:
    """Base class of a mesh/solver arm (see the module docstring)."""

    name = "arm"

    def supports(self, case):
        return False

    def resolutions(self, case):
        raise NotImplementedError

    def run(self, case, resolution):
        raise NotImplementedError


def run(case, arm, resolution, catch=True):
    """
    Run one arm at one resolution and evaluate it.

    Returns
    -------
    dict
        ``case``, ``arm``, ``resolution``, ``metrics`` (from
        ``case.evaluate``), ``cost`` and ``wall_s``; ``error`` and
        ``traceback`` instead of metrics if the run failed and `catch`.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    rec = dict(case=case.name, arm=arm.name, resolution=resolution,
               t_run=getattr(case, "t_run", None))  # fmt: skip
    try:
        result = arm.run(case, resolution)
        rec["cost"] = {k: _plain(v) for k, v in result.get("cost", {}).items()}
        rec["metrics"] = _plain(case.evaluate(result))
    except Exception as e:  # noqa: BLE001
        if not catch:
            raise
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
    rec["wall_s"] = time.time() - t0
    try:
        with open("/proc/cpuinfo") as f:
            rec["cpu"] = next(
                line.split(":")[1].strip() for line in f if "model name" in line
            )
    except OSError:
        import platform

        rec["cpu"] = platform.processor() or platform.machine()
    return rec


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, complex):
        return [x.real, x.imag]
    return x


def save_record(rec, path):
    with open(path, "w") as f:
        json.dump(rec, f)


def load_records(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def cost_to_target(records, metric, target, cost_keys=("unknowns", "work", "run_s")):
    """
    Interpolated cost to reach ``metric <= target`` on one arm's ladder.

    Parameters
    ----------
    records : list of dict
        Records of one arm and case (from `run`), any order.
    metric : str
        Key of ``record["metrics"]``.
    target : float
    cost_keys : tuple
        Cost fields to interpolate; ``work`` is unknowns x steps.

    Returns
    -------
    dict
        ``{cost_key: value}``: NaN if the target is not reached and kept on
        the ladder, negative (an upper bound: |value|) if the coarsest run
        already meets it. Run times are interpolated along the same ladder
        (ordered by unknowns) as the other costs.
    """
    good = [r for r in records if "metrics" in r and np.isfinite(r["metrics"][metric])]
    if not good:
        return {k: np.nan for k in cost_keys}
    out = {}
    err = [r["metrics"][metric] for r in good]
    for key in cost_keys:
        if key == "work":
            cost = [r["cost"]["unknowns"] * r["cost"]["steps"] for r in good]
        else:
            cost = [r["cost"][key] for r in good]
        # Errors need not be monotone: use the ladder in order of unknowns
        o = np.argsort([r["cost"]["unknowns"] for r in good])
        u = np.asarray([r["cost"]["unknowns"] for r in good], float)[o]
        v = M.interp_to_target(u, np.asarray(err)[o], target)
        if key == "unknowns" or not np.isfinite(v):
            out[key] = v
        else:  # same crossing, interpolated in log(cost) vs log(unknowns)
            c = np.asarray(cost, float)[o]
            out[key] = (-1 if v < 0 else 1) * float(
                np.exp(np.interp(np.log(abs(v)), np.log(u), np.log(c)))
            )
    return out
