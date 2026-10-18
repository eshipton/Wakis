# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Parameter sweeps and gradient-free optimization of template simulations.

A design is described by two functions: `build(**params) -> Simulation`
(usually a template) and `analyze(sim) -> result` (a number, array or dict).
Both must be module-level functions to run in parallel processes.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import optimize as scipy_optimize


def _run(build, params, run_kwargs):
    sim = build(**params)
    sim.run(**run_kwargs)
    return sim


def _evaluate(build, analyze, params, run_kwargs):
    return analyze(_run(build, params, run_kwargs))


def parameter_sweep(build, analyze, parameters, processes=None, **run_kwargs):
    """
    Run a simulation for every point of a parameter grid.

    Parameters
    ----------
    build : callable
        `build(**params)` returns a Simulation.
    analyze : callable
        `analyze(sim)` returns the result of a finished run.
    parameters : dict of str -> sequence, or list of dict
        Grid axes (the Cartesian product is swept) or an explicit list of
        parameter sets.
    processes : int, optional
        Number of worker processes. Default None: run sequentially in this
        process. Each process holds one simulation in memory.
    **run_kwargs
        Passed to `Simulation.run` (e.g. `energy_decay`).

    Returns
    -------
    list of dict
        One entry per parameter set: the parameters plus 'result'.
    """
    if isinstance(parameters, dict):
        names = list(parameters)
        points = [
            dict(zip(names, values))
            for values in itertools.product(*parameters.values())
        ]
    else:
        points = list(parameters)
    if processes is None:
        results = [_evaluate(build, analyze, p, run_kwargs) for p in points]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            futures = [
                pool.submit(_evaluate, build, analyze, p, run_kwargs) for p in points
            ]
            results = [future.result() for future in futures]
    return [dict(p, result=r) for p, r in zip(points, results)]


def optimize(
    build,
    objective,
    x0,
    bounds=None,
    method="Nelder-Mead",
    fixed=None,
    run_kwargs=None,
    **options,
):
    """
    Minimize a scalar figure of merit over a few design parameters with a
    gradient-free optimizer (every evaluation is a full simulation).

    Parameters
    ----------
    build : callable
        `build(**params)` returns a Simulation.
    objective : callable
        `objective(sim)` returns the float to minimize, e.g. |S11| in dB at
        the target frequency.
    x0 : dict of str -> float
        Initial values of the optimized parameters.
    bounds : dict of str -> (float, float), optional
        Bounds per parameter (required by 'differential_evolution' and
        'bounded').
    method : str, optional
        'Nelder-Mead' (default), 'Powell' (scipy.optimize.minimize),
        'bounded' (scipy.optimize.minimize_scalar, one parameter) or
        'differential_evolution'.
    fixed : dict, optional
        Further keyword arguments of `build` kept constant.
    run_kwargs : dict, optional
        Passed to `Simulation.run`.
    **options
        Passed to the scipy optimizer (e.g. `options={'xatol': 1e-4}` for
        Nelder-Mead, `maxiter`, `tol`, `xatol` for 'bounded').

    Returns
    -------
    scipy.optimize.OptimizeResult
        With the extra fields `params` (best parameters as a dict) and
        `history` (list of (params, value) of all evaluations).
    """
    names = list(x0)
    fixed = {} if fixed is None else dict(fixed)
    run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
    history = []

    def evaluate(x):
        params = dict(zip(names, np.atleast_1d(x).tolist()))
        value = float(objective(_run(build, {**fixed, **params}, run_kwargs)))
        history.append((params, value))
        return value

    start = np.array([x0[n] for n in names], dtype=float)
    box = None if bounds is None else [tuple(bounds[n]) for n in names]
    if method == "bounded":
        if len(names) != 1 or box is None:
            raise ValueError("method='bounded' needs one parameter with bounds.")
        result = scipy_optimize.minimize_scalar(
            evaluate, bounds=box[0], method="bounded", options=options or None
        )
        result.x = np.atleast_1d(result.x)
    elif method == "differential_evolution":
        if box is None:
            raise ValueError("differential_evolution needs bounds.")
        result = scipy_optimize.differential_evolution(
            evaluate, box, x0=start, **options
        )
    else:
        result = scipy_optimize.minimize(
            evaluate, start, method=method, bounds=box, **options
        )
    result.params = dict(zip(names, np.atleast_1d(result.x).tolist()))
    result.history = history
    return result
