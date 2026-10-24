# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""
Memory ladder for the exact discrete adjoint of `AdjointSimulation`.

The gradient needs, at every step n of the backward sweep, one *row*: the
design-edge quantity recorded by the forward step n (the solver-step
increment q^n / ieps, the ADE sensitivity, or the sub-capacitor terms of a
dispersive tensor design). The rungs differ only in how those rows are
obtained, in decreasing n:

``'store'``
    Keep all Nt rows (Nt x N_design values). No recomputation.
``'checkpoint'``
    Binomial checkpointing (Griewank & Walther 2000, "revolve") over blocks
    of steps: full solver snapshots at a few steps, the rows of one block
    held at a time, everything else recomputed from the snapshots. Replay
    is bitwise identical to the forward run, so the gradient is bitwise
    identical to ``'store'``. Valid for every physics the forward step
    supports.
``'reverse'``
    Time reversal of the lossless leapfrog. The interior fields are stepped
    backwards from the final state; CPML regions are not reversed, the
    fields on the interior/CPML interface layers are recorded during the
    forward run and re-injected. Exact in exact arithmetic; round-off makes
    it agree with ``'store'`` to ~1e-12 relative (measured, see the physics
    guide). Falls back to ``'checkpoint'`` with loss, dispersion, lumped
    elements, tensor averaging or MPI.
``'nyquist'``
    APPROXIMATE: rows stored every `decimation` steps and reconstructed by
    Lanczos-windowed sinc interpolation. Never the default; the gradient is
    exact only in the limit of a band-limited forward run sampled above
    Nyquist.
"""

import warnings
from math import comb

import numpy as np

# --------------------------------------------------------------------------
# Binomial (revolve) schedule


def beta(s, r):
    """Steps reversible with s snapshots and r repetitions: binom(s + r, s)."""
    return comb(s + r, s)


def _repetitions(l, s):
    r = 0
    while beta(s, r) < l:
        r += 1
    return r


def revolve_split(l, s):
    """
    Optimal first advance of a binomial reversal of l >= 2 steps.

    s >= 2 snapshot slots, counting the one that holds the start state.
    The rule (advance by beta(s, r - 2), clipped to the feasible range)
    reproduces the dynamic-programming optimum for all l < 150, s <= 7
    (`tests/test_045_adjoint_memory.py`).
    """
    r = _repetitions(l, s)
    lo = max(1, l - beta(s - 1, r))
    hi = min(l - 1, beta(s, r - 1))
    return max(lo, min(hi, beta(s, r - 2) if r >= 2 else 1))


def revolve_cost(l, s):
    """
    Forward step evaluations of the backward sweep (recording steps
    included) for an optimal reversal of l steps with s snapshot slots
    (start state included): r l - binom(s + r, s + 1) + l, with r the
    smallest integer with binom(s + r, s) >= l (revolve's `numforw`, plus
    the l recording steps). s = 1 gives l (l + 1) / 2.
    """
    if l <= 1:
        return l
    if s <= 1:
        return l * (l + 1) // 2
    r = _repetitions(l, s)
    return r * l - comb(s + r, s + 1) + l


def plan_checkpoints(Nt, state_bytes, row_bytes, budget_bytes, max_slots=256):
    """
    Choose the snapshot slots and block length that minimize recomputation
    within a memory budget.

    Memory: slots x state_bytes + (block + 3) x row_bytes (row 0 and the
    two working rows of a recording step). Recompute:
    ``revolve_cost(n_blocks - 1, slots) * block`` forward steps at most
    (the last block is recorded during the forward run).

    Returns
    -------
    dict
        mode ('store' or 'checkpoint'), slots, block, bytes, recompute_steps.
    """
    Nt = int(Nt)
    if Nt * row_bytes <= budget_bytes:
        return dict(mode="store", slots=0, block=Nt, bytes=Nt * row_bytes,
                    recompute_steps=0)  # fmt: skip
    best = None
    for s in range(1, max_slots + 1):
        free = budget_bytes - s * state_bytes - 3 * row_bytes
        max_block = min(int(free // row_bytes) if row_bytes else Nt, Nt - 1)
        if max_block < 1:
            break
        # The cost drops in steps where the number of blocks (minus the
        # one recorded by the forward run) reaches beta(s, r): try those
        # block lengths and the longest one that fits
        blocks = {max_block}
        for r in range(0, 64):
            b = -(-(Nt - 1) // (beta(s, r) + 1))
            if b <= max_block:
                blocks.add(b)
            if b <= 1:
                break
        for block in blocks:
            n_blocks = -(-(Nt - 1) // block)
            cost = revolve_cost(n_blocks - 1, s) * block if n_blocks > 1 else 0
            mem = s * state_bytes + (block + 3) * row_bytes
            key = (cost, mem)
            if best is None or key < best[0]:
                best = (key, s, block, mem, cost)
    if best is None:
        raise MemoryError(
            f"A budget of {budget_bytes / 1e9:.3g} GB holds less than one solver "
            f"snapshot ({state_bytes / 1e9:.3g} GB) and one block of rows."
        )
    _, s, block, mem, cost = best
    return dict(mode="checkpoint", slots=s, block=block, bytes=mem,
                recompute_steps=cost)  # fmt: skip


# --------------------------------------------------------------------------
# Solver snapshots


def _is_array(x):
    return isinstance(x, np.ndarray) or (
        type(x).__name__ == "ndarray" and type(x).__module__.startswith("cupy")
    )


def _snap(value):
    """Copy of the arrays inside a (nested) value; other objects by reference."""
    if _is_array(value):
        return value.copy()
    if isinstance(value, list):
        return [_snap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snap(v) for v in value)
    if isinstance(value, dict):
        return {k: _snap(v) for k, v in value.items()}
    return value


def _nbytes(value):
    if _is_array(value):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    return 0


class SolverState:
    """
    Capture and restore of everything a forward step reads and writes.

    Solver: E, H, J, M, the conduction bookkeeping J_old, the CPML psi
    arrays, the ADE polarizations of the dispersive groups and the state of
    a dispersive sub-capacitor network. Sources and lumped elements: every
    array, container and scalar attribute (incremental source currents
    J_old, TF/SF incident line, inductor currents, recorded port signals);
    other attributes (operators, time functions) are shared by reference.
    Restoring copies again, so a snapshot can be restored many times.
    """

    def __init__(self, solver, sources):
        self.solver = solver
        self.sources = list(sources)
        self.psi = sorted(k for k in vars(solver) if k.startswith("psi"))

    def capture(self):
        s = self.solver
        snap = {"E": s.E.array.copy(), "H": s.H.array.copy(), "J": s.J.array.copy()}
        if s.M is not None:
            snap["M"] = s.M.array.copy()
        if _is_array(getattr(s, "J_old", None)):
            snap["J_old"] = s.J_old.copy()
        for name in self.psi:
            snap[name] = getattr(s, name).copy()
        snap["groups"] = [
            (_snap(g["lorentz"]), _snap(g["debye"]))
            for g in getattr(s, "_dispersive_groups", [])
        ]
        snap["beams"] = [
            (b.v1.copy(), b.rho1.copy(), b.J_old.copy())
            for b in getattr(s, "_beams", [])
        ]
        snap["h_groups"] = [
            (_snap(g["lorentz"]), _snap(g["debye"]))
            for g in getattr(s, "_dispersive_h_groups", [])
        ]
        net = getattr(s, "dispersive_network", None)
        if net is not None:
            state = {}
            net.get_state(state)
            snap["net"] = state
        snap["sources"] = [
            {k: _snap(v) for k, v in vars(src).items()} for src in self.sources
        ]
        return snap

    def restore(self, snap):
        s = self.solver
        s.E.array[...] = snap["E"]
        s.H.array[...] = snap["H"]
        s.J.array[...] = snap["J"]
        if "M" in snap:
            s.M.array[...] = snap["M"]
        if "J_old" in snap:
            s.J_old = snap["J_old"].copy()
        for name in self.psi:
            cur = getattr(s, name)
            if getattr(cur, "shape", None) == snap[name].shape:
                cur[...] = snap[name]
            else:
                setattr(s, name, snap[name].copy())
        for g, (lorentz, debye) in zip(
            getattr(s, "_dispersive_groups", []), snap["groups"]
        ):
            g["lorentz"], g["debye"] = _snap(lorentz), _snap(debye)
        for b, (v1, rho1, J_old) in zip(
            getattr(s, "_beams", []), snap.get("beams", [])
        ):
            b.v1[...] = v1
            b.rho1[...] = rho1
            b.J_old[...] = J_old
        for g, (lorentz, debye) in zip(
            getattr(s, "_dispersive_h_groups", []), snap.get("h_groups", ())
        ):
            g["lorentz"], g["debye"] = _snap(lorentz), _snap(debye)
        if "net" in snap:
            s.dispersive_network.set_state(snap["net"], s.E.xp)
        for src, attrs in zip(self.sources, snap["sources"]):
            for k, v in attrs.items():
                setattr(src, k, _snap(v))

    @staticmethod
    def nbytes(snap):
        return _nbytes(snap)


# --------------------------------------------------------------------------
# Rungs. Each receives the forward run step by step (`wants_row`,
# `before_step`, `mid_step`, `after_step`) and then serves rows backwards.


class _Rung:
    exact = True
    name = ""

    def wants_row(self, n):
        return False

    def before_step(self, n):
        pass

    def mid_step(self, n):
        pass

    def after_step(self, n, row):
        pass

    def finish(self):
        pass

    def reader(self):
        raise NotImplementedError

    def stats(self):
        return {"mode": self.name, "peak_bytes": int(self.peak_bytes),
                "recompute_steps": int(self.recompute_steps)}  # fmt: skip


class NullRung(_Rung):
    """Objective-only forward run: nothing recorded (`forward(record=False)`)."""

    name = "none"
    peak_bytes = 0
    recompute_steps = 0

    def reader(self):
        raise RuntimeError("forward(record=False) kept nothing for a gradient.")


def available_host_bytes():
    """Available host memory [bytes], portably (psutil, sysconf, sysctl)."""
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    import os

    try:
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS: no SC_AVPHYS_PAGES; half the physical memory as available
        import subprocess

        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        return 0.5 * total
    except Exception:
        return 4e9  # conservative default


class StoreRung(_Rung):
    """All rows in one (Nt, n_row) array, on the device or on the host."""

    name = "store"

    def __init__(self, Nt, row_shape, dtype, xp, on_host):
        self.on_host = on_host
        self.rows = (np if on_host else xp).empty((Nt, *row_shape), dtype=dtype)
        self.xp = xp
        self.peak_bytes = self.rows.nbytes
        self.recompute_steps = 0

    def wants_row(self, n):
        return True

    def after_step(self, n, row):
        self.rows[n] = _host(row) if self.on_host else row

    def reader(self):
        rows, xp = self.rows, self.xp
        return lambda n: xp.asarray(rows[n])


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


class CheckpointRung(_Rung):
    """
    Binomial checkpointing over blocks of steps (exact replay).

    Step 0 initializes the sources, so its row is kept and the snapshot
    slots cover the states x^1 ... x^{Nt-1}: block k starts at step
    1 + k * block. `slots` counts all snapshots held at once, including
    the persistent one of x^1 (kept, so `vjp` can run more than once).
    The forward run lays down the first descent of the schedule and records
    the rows of the last block, which the backward sweep needs first.
    """

    name = "checkpoint"

    def __init__(self, Nt, slots, block, advance, io, dtype, xp, verify=False):
        if slots < 1 or block < 1:
            raise ValueError("checkpoint slots and block length must be >= 1.")
        self.Nt, self.slots, self.block = int(Nt), int(slots), int(block)
        self.advance, self.io, self.xp, self.dtype = advance, io, xp, dtype
        self.n_blocks = -(-(self.Nt - 1) // self.block) if self.Nt > 1 else 0
        self.snaps = {}
        self.row0 = None
        self.tail = None
        self.pos = None  # step of the solver's current state
        self.recompute_steps = 0
        self.peak_bytes = 0
        self._snap_bytes = 0
        # First descent of the schedule: snapshots laid down by the forward
        self.chain = set()
        a, c = 0, self.slots - 1
        while self.n_blocks - a > 1 and c > 0:
            a += revolve_split(self.n_blocks - a, c + 1)
            self.chain.add(a)
            c -= 1
        self.tail_start = self._start(self.n_blocks - 1) if self.n_blocks else self.Nt
        # verify: also keep every forward row and count replayed rows that
        # differ from it in any bit (testing aid; costs the store memory)
        self.verify = verify
        self.forward_rows = {} if verify else None
        self.mismatches = 0

    def _start(self, k):
        return 1 + k * self.block

    def _stop(self, k):
        return min(1 + (k + 1) * self.block, self.Nt)

    def _account(self, extra=0):
        rows = (0 if self.tail is None else len(self.tail)) + extra
        row_bytes = getattr(self, "_row_bytes", 0)
        # + row 0 and the two working rows of a recording step (E^n, increment)
        total = len(self.snaps) * self._snap_bytes + (rows + 3) * row_bytes
        self.peak_bytes = max(self.peak_bytes, total)

    # forward hooks
    def wants_row(self, n):
        return self.verify or n == 0 or n >= self.tail_start

    def before_step(self, n):
        if n >= 1 and (n - 1) % self.block == 0:
            k = (n - 1) // self.block
            if k == 0 or k in self.chain:
                self._save(k)

    def after_step(self, n, row):
        if self.verify:
            self.forward_rows[n] = row.copy()
            if not (n == 0 or n >= self.tail_start):
                return
        if n == 0:
            self.row0 = row.copy()
            self._row_bytes = _nbytes(row)
            self.tail = []
        elif n >= self.tail_start:
            self.tail.append(row.copy())
        self._account()

    def finish(self):
        self.pos = self.Nt

    def _save(self, k):
        self.snaps[k] = self.io.capture()
        if not self._snap_bytes:
            self._snap_bytes = SolverState.nbytes(self.snaps[k])
        self._account()

    # backward
    def _goto(self, step):
        """Bring the solver to x^step from the nearest snapshot at or below."""
        if self.pos != step:
            k = max(j for j in self.snaps if self._start(j) <= step)
            if self.pos is None or not (self._start(k) <= self.pos <= step):
                self.io.restore(self.snaps[k])
                self.pos = self._start(k)
        while self.pos < step:
            self.advance(self.pos, False)
            self.pos += 1
            self.recompute_steps += 1

    def _leaf(self, k):
        if k == self.n_blocks - 1 and self.tail is not None:
            rows, self.tail = self.tail, None
        else:
            self._goto(self._start(k))
            rows = []
            for n in range(self._start(k), self._stop(k)):
                rows.append(self.advance(n, True))
                self.recompute_steps += 1
                self._account(len(rows))
            self.pos = self._stop(k)
        for i in reversed(range(len(rows))):
            yield self._start(k) + i, rows[i]

    def _reverse(self, a, b, c):
        """Blocks [a, b) with a snapshot of block a held and c free slots."""
        if b - a == 1:
            yield from self._leaf(a)
            return
        if c == 0:
            for k in reversed(range(a, b)):
                yield from self._leaf(k)
            return
        m = a + revolve_split(b - a, c + 1)
        if m not in self.snaps:
            self._goto(self._start(m))
            self._save(m)
        yield from self._reverse(m, b, c - 1)
        if m != 0:
            del self.snaps[m]
        yield from self._reverse(a, m, c)

    def _rows(self):
        if self.n_blocks:
            if 0 not in self.snaps:
                raise RuntimeError("Run forward before vjp.")
            yield from self._reverse(0, self.n_blocks, self.slots - 1)
        yield 0, self.row0

    def reader(self):
        self.recompute_steps = 0  # per backward sweep
        gen = self._rows()

        def get(n):
            m, row = next(gen)
            if m != n:
                raise RuntimeError(f"Row {m} served for step {n}.")
            if self.verify and not bool(
                self.xp.array_equal(row, self.forward_rows[n])
            ):
                self.mismatches += 1
            return row

        return get


class NyquistRung(_Rung):
    """
    APPROXIMATE storage: every `decimation`-th row, Lanczos-sinc interpolated.

    q^n ~ sum_m q^{m s} L((n - m s) / s), L(x) = sinc(x) sinc(x / a) for
    |x| < a. Exact at stored steps; elsewhere exact only for rows
    band-limited below 1 / (2 s dt). Use `decimation` well inside the
    Nyquist limit of the source bandwidth.
    """

    name = "nyquist"
    exact = False

    def __init__(self, Nt, decimation, row_shape, dtype, xp, lobes=6):
        self.s, self.a = int(decimation), int(lobes)
        if self.s < 1:
            raise ValueError("decimation must be >= 1.")
        self.Nt = int(Nt)
        n_keep = (self.Nt - 1) // self.s + 1
        self.rows = xp.empty((n_keep, *row_shape), dtype=dtype)
        self.xp = xp
        self.peak_bytes = self.rows.nbytes
        self.recompute_steps = 0

    def wants_row(self, n):
        return n % self.s == 0

    def after_step(self, n, row):
        if n % self.s == 0:
            self.rows[n // self.s] = row

    def reader(self):
        s, a, rows = self.s, self.a, self.rows

        def get(n):
            if n % s == 0:
                return rows[n // s]
            x0 = n / s
            m = np.arange(max(0, int(np.floor(x0)) - a + 1),
                          min(rows.shape[0], int(np.floor(x0)) + a + 1))  # fmt: skip
            x = x0 - m
            w = np.sinc(x) * np.sinc(x / a)
            out = w[0] * rows[m[0]]
            for wi, mi in zip(w[1:], m[1:]):
                out = out + wi * rows[mi]
            return out

        return get


class ReverseRung(_Rung):
    """
    Time reversal of the lossless leapfrog with recorded interface layers.

    Forward step n (no loss, no dispersion), with I the interior edges (not
    in a CPML slab) and Hc the TF/SF H correction added by the sources::

        H^{n+1/2} = H^{n-1/2} + Hc^n - dt A E^n - dt imu M^n
        E^{n+1}   = E^n + dt (B H^{n+1/2} - ieps J^n)

    Backward, on I only::

        E^n_I       = E^{n+1}_I - dt (B H^{n+1/2} - ieps J^n)_I
        H^{n-1/2}_I = H^{n+1/2}_I + dt (A E^n)_I + dt imu M^n_I - Hc^n_I

    The rows of B on I read H on I and on the layer G_H just outside it; A
    reads E on I and G_E. H^{n+1/2}[G_H], E^n[G_E], J^n, M^n and Hc^n on
    the source supports are recorded each step (surfaces, not volumes).
    The design rows are (E^{n+1} - E^n) / ieps on the design edges.
    """

    name = "reverse"

    def __init__(self, Nt, solver, sources, design_idx, ieps_d, dtype):
        self.Nt = int(Nt)
        self.solver = solver
        self.sources = sources
        self.xp = xp = solver.E.xp
        self.idx = design_idx
        self.ieps_d = ieps_d
        self.dt = solver.dt
        self.recompute_steps = 0
        self.corrupt = None  # (step, key) to falsify the test: see tests
        n3 = 3 * solver.N
        A, B = _curl_operators(solver)
        interior = _interior_mask(solver)
        self.mask_E = self.mask_H = interior
        I = np.flatnonzero(interior)
        g_H = np.setdiff1d(np.unique(B[I].indices), I)
        g_E = np.setdiff1d(np.unique(A[I].indices), I)
        if not np.all(interior[_host(design_idx)]):
            raise ValueError("Design edges inside the CPML: time reversal needs "
                             "an interior design region.")  # fmt: skip
        self.ieps = xp.asarray(solver.ieps.toarray())
        self.imu = xp.asarray(solver.imu.toarray())
        if solver.use_gpu:
            from cupyx.scipy.sparse import csr_matrix as gcsr

            A, B = gcsr(A), gcsr(B)
            for m in (A, B):  # canonical once: else every matvec copies m
                m.sum_duplicates()
        self.A, self.B = A, B
        # The interior operators are held by the rung for the whole run
        self._op_bytes = sum(
            int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) for m in (A, B)
        )
        self.I = xp.asarray(I)
        self.g_H, self.g_E = xp.asarray(g_H), xp.asarray(g_E)
        self.n3 = n3
        self.rec = {}
        self.supports = None
        self.peak_bytes = 0

    def _supports(self):
        J, M, Hc = [], [], []
        for src in self.sources:
            name = type(src).__name__
            if name in ("CurrentSource", "MagneticCurrentSource"):
                (J if src._field == "J" else M).append(_host(src.flat_idx))
            elif name == "ModeSource":
                for term in src.terms:
                    (J if term["field"] == "J" else M).append(_host(term["idx"]))
            elif name == "TFSFSource":
                J.append(_host(src._j_rows))
                Hc.append(_host(src._h_rows))
            else:
                raise NotImplementedError(name)
        xp = self.xp

        def join(parts):
            return xp.asarray(np.unique(np.concatenate(parts)) if parts else
                              np.zeros(0, dtype=np.int64))  # fmt: skip

        return {"J": join(J), "M": join(M), "Hc": join(Hc)}

    def _alloc(self):
        xp, Nt, dtype = self.xp, self.Nt, self.solver.E.array.dtype
        sup = self.supports
        sizes = {"H_layer": self.g_H.size, "E_layer": self.g_E.size,
                 "J": sup["J"].size, "M": sup["M"].size, "Hc": sup["Hc"].size}  # fmt: skip
        self.rec = {k: xp.zeros((Nt, n), dtype=dtype) for k, n in sizes.items()}
        self.peak_bytes = sum(r.nbytes for r in self.rec.values()) + self._op_bytes

    def before_step(self, n):
        if n == 0:
            self._H_prev = self.solver.H.array.copy()

    def mid_step(self, n):
        s = self.solver
        if self.supports is None:
            self.supports = self._supports()
            self._alloc()
        sup, rec = self.supports, self.rec
        rec["E_layer"][n] = s.E.array[self.g_E]
        rec["J"][n] = s.J.array[sup["J"]]
        if s.M is not None:
            rec["M"][n] = s.M.array[sup["M"]]
        rec["Hc"][n] = s.H.array[sup["Hc"]] - self._H_prev[sup["Hc"]]

    def after_step(self, n, row):
        s = self.solver
        self.rec["H_layer"][n] = s.H.array[self.g_H]
        if self.supports["Hc"].size:
            self._H_prev = s.H.array.copy()
        else:
            self._H_prev = s.H.array[:0]

    def finish(self):
        self.final = (self.solver.E.array.copy(), self.solver.H.array.copy())
        state = 2 * self.final[0].nbytes
        self.peak_bytes += state + 2 * state  # final state + working copies
        self._H_prev = None

    def reader(self):
        xp, dt = self.xp, self.dt
        E, H = (a.copy() for a in self.final)
        not_I = xp.ones(self.n3, dtype=bool)
        not_I[self.I] = False
        E[not_I] = 0
        H[not_I] = 0
        sup, rec = self.supports, self.rec
        if self.corrupt is not None:
            step, key = self.corrupt
            rec = dict(rec)
            rec[key] = rec[key].copy()
            rec[key][step] *= 1.5
        state = {"E": E, "H": H, "next": self.Nt - 1}
        zeros = xp.zeros(self.n3, dtype=E.dtype)

        def get(n):
            if n != state["next"]:
                raise RuntimeError(f"Reverse rung expected step {state['next']}.")
            E, H = state["E"], state["H"]
            H[self.g_H] = rec["H_layer"][n]
            J = zeros.copy()
            J[sup["J"]] = rec["J"][n]
            dE = self.B @ H
            dE -= self.ieps * J
            dE *= dt
            e_new = E[self.idx]
            E -= dE
            E[not_I] = 0
            E[self.g_E] = rec["E_layer"][n]
            row = (e_new - E[self.idx]) / self.ieps_d
            dH = self.A @ E
            if sup["M"].size:
                M = zeros.copy()
                M[sup["M"]] = rec["M"][n]
                dH += self.imu * M
            dH *= dt
            H += dH
            H[sup["Hc"]] -= rec["Hc"][n]
            H[not_I] = 0
            state["next"] = n - 1
            return row

        return get


def _curl_operators(solver):
    """
    A and B of the step on the host as CSR, with imu and ieps included::

        H' = H - dt A E,   E' = E + dt B H'   (interior rows)

    Standard step: ``tDsiDmuiDaC`` and ``itDaiDepsDstC``. CPML step: the
    component operators of `_one_step_cpml` (rows without psi terms).
    """
    from scipy.sparse import bmat, csr_matrix, diags

    from .adjoint import _E_TERMS, _H_TERMS

    def host(op):
        if hasattr(op, "get"):
            op = op.get()
        return csr_matrix(op)

    if not solver.activate_cpml:
        return host(solver.tDsiDmuiDaC), host(solver.itDaiDepsDstC)
    N = solver.N
    imu, ieps = _host(solver.imu.toarray()), _host(solver.ieps.toarray())

    def assemble(terms, coef, sign):
        blocks = [[None] * 3 for _ in range(3)]
        for T, d, S, s, op, prefix in terms:
            t, c = "xyz".index(T), "xyz".index(S)
            blk = sign * s * host(getattr(solver, op))
            blocks[t][c] = blk if blocks[t][c] is None else blocks[t][c] + blk
        for t in range(3):
            for c in range(3):
                if blocks[t][c] is None:
                    blocks[t][c] = csr_matrix((N, N), dtype=solver.dtype)
        return (diags(coef) @ bmat(blocks)).tocsr()

    return assemble(_H_TERMS, imu, 1.0), assemble(_E_TERMS, ieps, 1.0)


def _interior_mask(solver):
    """Edges (E and H, all components) of cells outside every CPML slab."""
    N = solver.N
    inside = np.ones(N, dtype=bool)
    for d, (lo, hi) in enumerate(zip(solver.bc_low, solver.bc_high)):
        for side, bc in (("low", lo), ("high", hi)):
            if bc.lower() == "cpml":
                inside[_host(getattr(solver, f"idx_{'xyz'[d]}_{side}"))] = False
    return np.concatenate([inside] * 3)


def reverse_supported(solver, sources, design):
    """Reason why time reversal does not apply, or None."""
    if solver.use_mpi and solver.grid.use_mpi:
        return "MPI"
    if solver.use_conductivity:
        return "conductivity (loss)"
    if getattr(solver, "_dispersive_groups", []) or getattr(
        solver, "_dispersive_h_groups", []
    ):
        return "dispersive media"
    if getattr(solver, "_beams", []):
        return "linearized electron beam"
    if getattr(solver, "dispersive_network", None) is not None:
        return "dispersive tensor network"
    op = getattr(solver, "tensor_operator", None)
    if op is not None and op.rows.size:
        return "tensor averaging"
    for src in sources:
        if type(src).__name__ == "LinearizedBeam":
            return "linearized electron beam"
        if hasattr(src, "post_step"):
            return "lumped elements"
        if type(src).__name__ not in (
            "CurrentSource", "MagneticCurrentSource", "ModeSource", "TFSFSource",
        ):  # fmt: skip
            return f"source {type(src).__name__}"
    if design.material is not None or design.smoothing is not None:
        return "dispersive or tensor design"
    return None


def warn_fallback(reason):
    warnings.warn(
        f"Time reversal is not exact with {reason}; using binomial "
        "checkpointing instead.",
        stacklevel=3,
    )
