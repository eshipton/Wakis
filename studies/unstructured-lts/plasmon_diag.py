"""
Diagnose the time-domain plasmon test (TE Drude rod, kp a = 4, in a PEC
cavity, graded mesh h = 0.04): global vs LTS stepping on the same mesh,
harminv in several windows, FFT peak, and the dispersive eigenmode.
Run on Batch (slow).
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.constants import c

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_047_unstructured_lts import A_CAV, B_ROD, drude_solver, graded_rod_mesh  # noqa: E402

from wakis.resonances import harminv  # noqa: E402
from wakis.unstructured import analytic  # noqa: E402

m = graded_rod_mesh(0.04)
k_sp = analytic.drude_rod_in_cavity_te(A_CAV, B_ROD, 4.0, n=1)[1]
f_an = k_sp * c / (2 * np.pi)
for mode in ("global", "lts"):
    t0 = time.time()
    s = drude_solver("TE", m, 0.0, **({"p": 2} if mode == "lts" else {}))
    f_sp = s.dispersive_eigenmode(f_an)
    s0 = 1 / (2 * np.pi * 0.3 * f_sp)

    def pulse(t, s0=s0, f0=f_sp):
        return np.exp(-(((t - 5 * s0) / s0) ** 2) / 2) * np.sin(
            2 * np.pi * f0 * (t - 5 * s0)
        )

    s.add_point_source([0.6, 0.1], pulse, [1, 0.3])
    probe = s.add_probe([[0.55, -0.2]])
    n = int(60 / f_sp / s.dt)
    s.run(n)
    sig = s.probe_signal(probe)[:, 0]
    i_end = int(10 * s0 / s.dt) + 1  # source off
    print(f"{mode}: dt {s.dt:.3e} steps {n} f_an {f_an:.6e} f_eig {f_sp:.6e} "
          f"(eig/an-1 {f_sp / f_an - 1:+.2e}) max|sig| {np.abs(sig).max():.3e} "
          f"run {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    F = np.fft.rfftfreq(len(sig[i_end:]) * 8, s.dt)
    A = np.abs(np.fft.rfft(sig[i_end:], len(sig[i_end:]) * 8))
    band = (F > 0.5 * f_sp) & (F < 1.5 * f_sp)
    print(f"  FFT peak in 0.5-1.5 f_sp: {F[band][np.argmax(A[band])] / f_sp:.4f} f_sp")
    for i0, lo, hi in ((n // 5, 0.97, 1.03), (i_end, 0.97, 1.03), (i_end, 0.9, 1.1),
                       (i_end, 0.5, 1.5)):  # fmt: skip
        try:
            for mre in (1e-3, None):
                r = harminv(sig[i0:], s.dt, lo * f_sp, hi * f_sp, t0=(i0 + 1) * s.dt,
                            max_rel_error=mre)  # fmt: skip
                fr = np.atleast_1d(r.frequency)
                amp = np.atleast_1d(getattr(r, "amplitude", np.ones_like(fr)))
                err = np.atleast_1d(getattr(r, "error", np.zeros_like(fr)))
                txt = ", ".join(f"{np.real(f) / f_sp:.5f} (|a| {abs(a):.1e}, err "
                                f"{np.abs(e) / f_sp:.1e})" for f, a, e in zip(fr, amp, err))  # fmt: skip
                print(f"  harminv i0={i0} [{lo},{hi}] max_rel_error={mre}: {txt or 'none'}")
        except Exception as e:  # noqa: BLE001
            print(f"  harminv i0={i0} [{lo},{hi}]: ERROR {e!r}")
