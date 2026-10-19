import numpy as np
from scipy.optimize import brentq


def fixed_points(modes_at, k_lo, k_hi, n=41, xtol=1e-13):
    """Self-consistent modes k = mode(k): sign changes of nearest-mode(k) - k."""
    ks = np.linspace(k_lo, k_hi, n)

    def d(k):
        m = np.asarray(modes_at(k))
        return m[np.argmin(abs(m - k))] - k

    vals = np.array([d(k) for k in ks])
    out = []
    step = ks[1] - ks[0]
    for i in range(n - 1):
        if vals[i] * vals[i + 1] < 0 and max(abs(vals[i]), abs(vals[i + 1])) < 2 * step:
            try:
                r = brentq(d, ks[i], ks[i + 1], xtol=xtol)
            except ValueError:
                continue
            if abs(d(r)) < 1e-8:
                out.append(r)
    return out
