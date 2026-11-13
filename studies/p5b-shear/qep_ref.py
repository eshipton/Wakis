import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

import _space_charge_waves_common as s

nx, nz, p = int(sys.argv[1]), int(sys.argv[2]), 2
for qm in (1.0, 2.0):
    P = s.compute_analytic_parameters(qm)
    dz = s.TUNNEL_LENGTH_M / nz
    dt, _ = s._cached_integer_period_dt(nx, nz)
    dep = s.compute_discrete_dc_depression(nx, P, return_details=True, shape=p)
    u_l = dep["u_l"]
    g_l = dep["gamma_l"]
    u0 = P["u0"]
    sig = P["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
    Om = (2 / dt) * np.sin(0.5 * P["omega"] * dt)
    be = Om / u_l

    def Msym(kk, P=P, dz=dz, sig=sig):
        op = s.compute_transverse_modal_operator(
            "particles", nx, nz, P, k_val=kk, shape=p
        )
        d = op["d_vec"]
        sz = np.sinc(0.5 * kk * dz / np.pi) ** (2 * p)
        sl = np.sqrt(d / sig)
        sr = np.sqrt(sig / d)
        m = (sl[:, None] * (sz * sig)[:, None] * op["G"]) * sr[None, :]
        return 0.5 * (m + m.T)

    det = s.discrete_modal_prediction(
        "particles", nx, nz, P, return_details=True, shape=p
    )

    def F(kk, be=be, Msym=Msym):
        return np.linalg.slogdet(np.diag((kk - be) ** 2) - Msym(kk))[0]

    def root(k0, w, F=F):
        ks = np.linspace(k0 - w, k0 + w, 121)
        fs = np.array([F(x) for x in ks])
        idx = np.where(np.diff(fs) != 0)[0]
        return min(
            [brentq(F, ks[i], ks[i + 1], xtol=1e-13) for i in idx],
            key=lambda q: abs(q - k0),
        )

    c = det["c_modes"]
    bqm = det["bq_modes"].copy()
    kcm = det["kc_modes"].copy()
    bqx = bqm.copy()
    kcx = kcm.copy()
    for j in np.where(np.abs(c) > 1e-3)[0]:
        w = min(2.0, 0.3 * bqm[j])
        kp = root(kcm[j] + bqm[j], w)
        km = root(kcm[j] - bqm[j], w)
        bqx[j] = 0.5 * (kp - km)
        kcx[j] = 0.5 * (kp + km)
        print(
            f"   qm={qm} mode {j} c={c[j]:.3e} bq model {bqm[j]:.5f} "
            f"exact {bqx[j]:.5f} rel {bqx[j] / bqm[j] - 1:+.3e}"
        )
    z = det["z_coords"]
    zmin, zmax = s.fit_window(P)

    def fit(bq, kc, P=P, c=c, z=z, zmin=zmin, zmax=zmax):
        i1 = P["I1_0"] * np.sum(
            c[:, None] * np.cos(np.outer(bq, z)) * np.exp(-1j * np.outer(kc, z)),
            axis=0,
        )
        return s.fit_beta_q_from_envelope(z, i1, float(bq[0]), z_min=zmin, z_max=zmax)[
            "beta_q"
        ]

    f0 = fit(bqm, kcm)
    fx = fit(bqx, kcx)
    print(
        f"nx={nx} qm={qm}: fitted model {f0:.6f} (matches pred {det['beta_q']:.6f}); "
        f"fitted with exact-shear roots {fx:.6f}; rel shift {fx / f0 - 1:+.3e}"
    )
