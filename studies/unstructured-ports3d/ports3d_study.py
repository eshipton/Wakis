"""
Ports on tetrahedral FIT: convergence and validation study.

Cases (all on extruded tetrahedral meshes, `wakis.unstructured.ports3d`):

* ``wg``    WR-90 straight guide, TE10 modal ports: S11 floor, S21 vs the
            discrete ladder and vs continuum exp(-i beta L), refinement
            h = a/4 .. a/24 (D = h / 1.5);
* ``slab``  eps 2.2 slab (10 mm) in WR-90: two-run 2x2 S vs ladder and
            continuum (transfer-matrix) results, same refinement;
* ``coax``  50 Ohm coax with a 100 Ohm sheet load: input reflection vs
            ladder and continuum transmission-line theory;
* ``lumped`` radial edge-path lumped ports on the coax: time domain vs the
            frequency-domain solve of the same discrete equations, DC limit.

Usage: python ports3d_study.py [out_dir]   (parallel over cases; Cloud Batch)
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.pml3d import _snap_degenerate_duals
from wakis.unstructured.ports3d import (
    ExtrudedLine,
    LumpedPort,
    ModalPort,
    PortPlane,
    attach_ports,
    coax_section,
    edge_path,
    extrude_section,
    power_waves,
    rectangle_section,
    s_matrix,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D

ETA = np.sqrt(mu_0 / epsilon_0)
A, B = 22.86e-3, 10.16e-3
F_X = np.linspace(8e9, 12e9, 41)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")


def pulse(f0, bw):
    sig = 1 / (2 * np.pi * bw)
    t0 = 6 * sig
    return lambda t: np.exp(-(((t - t0) / sig) ** 2) / 2) * np.sin(2 * np.pi * f0 * (t - t0))


def wr90_mesh(h_div, length, slab=None):
    h = A / h_div
    nodes, tris, tags = rectangle_section(A, B, h, wall=h / 2, seed=1)
    nz = int(np.ceil(length / (h / 1.5)))
    z = np.linspace(0, length, nz + 1)
    m = extrude_section(nodes, tris, tags, z)
    mats, eps_l = {0: "pec"}, np.ones(nz)
    if slab is not None:
        z0, z1, eps = slab
        # snap the slab faces to mesh levels
        z0, z1 = z[np.argmin(abs(z - z0))], z[np.argmin(abs(z - z1))]
        zc = m.nodes[m.cells].mean(1)[:, 2]
        tg = m.cell_tags.copy()
        tg[(zc > z0) & (zc < z1) & (tg == 1)] = 2
        m = SimplexMesh(m.nodes, m.cells, tg)
        _snap_degenerate_duals(m)
        mats[2] = [eps, 1.0]
        mid = (z[:-1] + z[1:]) / 2
        eps_l[(mid > z0) & (mid < z1)] = eps
        slab = (z0, z1, eps)
    return m, mats, z, eps_l, slab


def run_modal(m, mats, z, eps_l, kind, excite, f0, bw, t_end, freqs, g_load=None):
    s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    planes = [PortPlane(s, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
    if kind == "te":
        kc = planes[0].te_modes(1)[0][0]
        phis = [p.te_modes(1)[1][:, 0] for p in planes]
        volts = [1.0, 1.0]
    else:
        kc = 0.0
        tm = [p.tem_mode() for p in planes]
        phis, volts = [t[0] for t in tm], [t[1] for t in tm]
    line = ExtrudedLine(z, eps_l, 1.0, kc, s.dt)

    def Y(f):
        return np.real(line.port_admittance(f) / line.cos_half(f))

    ports = []
    for k in range(2):
        g = g_load if (g_load is not None and k == 1) else None
        ports.append(ModalPort(planes[k], phis[k], Y, g=g, f0=f0,
                               i_src=pulse(f0, bw) if k == excite else None))
    attach_ports(s, ports)
    n = int(t_end / s.dt)
    e_max, t0 = 0.0, time.time()
    for _ in range(25):
        s.run(n // 25)
        e_max = max(e_max, s.energy())
    spec = [p.spectra(freqs) for p in ports]
    return dict(V=[x[0] for x in spec], I=[x[1] for x in spec], Y=Y(freqs), kc=kc,
                volts=volts, dt=s.dt, n_steps=n, n_tets=m.n_cells, n_free=s.free.size,
                energy_ratio=s.energy() / e_max, wall=time.time() - t0, line=line)


def case_wg(h_div):
    m, mats, z, eps_l, _ = wr90_mesh(h_div, 40e-3)
    r = run_modal(m, mats, z, eps_l, "te", 0, 10e9, 1.2e9, 10e-9, F_X)
    a, b = power_waves(np.array(r["V"]).T, np.array(r["I"]).T, r["Y"][:, None])
    S = np.array([np.linalg.solve([[a[k, 0], a[k, 1]], [a[k, 1], a[k, 0]]], b[k]) for k in range(F_X.size)])
    line = r["line"]
    beta = np.sqrt((2 * np.pi * F_X / c) ** 2 - (np.pi / A) ** 2)
    L = z[-1]
    S21d = np.exp(-1j * line.theta(F_X) * (z.size - 1))
    return dict(case="wg", h_div=h_div, h=A / h_div, dz=z[1] - z[0], n_tets=r["n_tets"],
                n_steps=r["n_steps"], wall_s=r["wall"], energy_ratio=r["energy_ratio"],
                kc_rel_err=float(r["kc"] / (np.pi / A) - 1),
                s11_floor=float(np.abs(S[:, 0]).max()),
                s21_vs_ladder=float(np.abs(S[:, 1] - S21d).max()),
                unitarity=float(np.abs(np.abs(S[:, 0]) ** 2 + np.abs(S[:, 1]) ** 2 - 1).max()),
                phase_err_cont=np.angle(S[:, 1] * np.exp(1j * beta * L)).tolist(),
                phase_err_pred=(-(np.real(line.beta(F_X)) - beta) * L).tolist())


def slab_continuum(f, z0, z1, eps, L):
    k0 = 2 * np.pi * f / c
    b1 = np.sqrt(k0**2 - (np.pi / A) ** 2)
    b2 = np.sqrt(eps * k0**2 - (np.pi / A) ** 2)
    d = z1 - z0
    G = (1 / b2 - 1 / b1) / (1 / b2 + 1 / b1)
    E = np.exp(-2j * b2 * d)
    S11 = G * (1 - E) / (1 - G**2 * E) * np.exp(-2j * b1 * z0)
    S21 = (1 - G**2) * np.exp(-1j * b2 * d) / (1 - G**2 * E) * np.exp(-1j * b1 * (L - d))
    return S11, S21


def case_slab(h_div):
    m, mats, z, eps_l, slab = wr90_mesh(h_div, 40e-3, slab=(15e-3, 25e-3, 2.2))
    runs = [run_modal(m, mats, z, eps_l, "te", k, 10e9, 1.2e9, 12e-9, F_X) for k in range(2)]
    V = np.transpose(np.array([r["V"] for r in runs]), (2, 0, 1))
    I = np.transpose(np.array([r["I"] for r in runs]), (2, 0, 1))
    Y = np.stack([runs[0]["Y"]] * 2, 1)
    S = s_matrix(V, I, Y)
    Sd = runs[0]["line"].s_matrix(F_X)
    S11c, S21c = slab_continuum(F_X, *slab, z[-1])
    U = np.einsum("fji,fjk->fik", S.conj(), S)
    return dict(case="slab", h_div=h_div, h=A / h_div, dz=z[1] - z[0], n_tets=runs[0]["n_tets"],
                n_steps=runs[0]["n_steps"], wall_s=sum(r["wall"] for r in runs),
                energy_ratio=max(r["energy_ratio"] for r in runs),
                s_vs_ladder=float(np.abs(S - Sd).max()),
                reciprocity=float(np.abs(S[:, 0, 1] - S[:, 1, 0]).max()),
                unitarity=float(np.abs(U - np.eye(2)).max()),
                s11_err_cont=float(np.abs(S[:, 0, 0] - S11c).max()),
                s21_err_cont=float(np.abs(S[:, 1, 0] - S21c).max()),
                S11=[S[:, 0, 0].real.tolist(), S[:, 0, 0].imag.tolist()],
                S21=[S[:, 1, 0].real.tolist(), S[:, 1, 0].imag.tolist()],
                S11d=[Sd[:, 0, 0].real.tolist(), Sd[:, 0, 0].imag.tolist()],
                S11c=[S11c.real.tolist(), S11c.imag.tolist()],
                S21c=[S21c.real.tolist(), S21c.imag.tolist()])


F_C = np.linspace(1e9, 8e9, 36)


def coax_mesh(h, length):
    nodes, tris, tags = coax_section(2e-3, 4.6e-3, h, wall=h)
    nz = int(np.ceil(length / h))
    z = np.linspace(0, length, nz + 1)
    return extrude_section(nodes, tris, tags, z), {0: "pec", 2: "pec"}, z


def case_coax(h):
    m, mats, z = coax_mesh(h, 20e-3)
    s0 = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    _, v = PortPlane(s0, (0, 0, z[-1]), (0, 0, 1)).tem_mode()
    RL = 100.0
    r = run_modal(m, mats, z, np.ones(z.size - 1), "tem", 0, 5e9, 1.5e9, 6e-9, F_C,
                  g_load=v**2 / RL)
    a, b = power_waves(r["V"][0], r["I"][0], r["Y"])
    G = b / a
    Gd = r["line"].input_reflection(F_C, v**2 / RL)
    Z0c = ETA * np.log(2.3) / (2 * np.pi)
    Gc = (RL - Z0c) / (RL + Z0c) * np.exp(-2j * 2 * np.pi * F_C / c * z[-1])
    return dict(case="coax", h=h, dz=z[1] - z[0], n_tets=r["n_tets"], n_steps=r["n_steps"],
                wall_s=r["wall"], energy_ratio=r["energy_ratio"],
                z0_rel_err=float(v**2 / (np.log(2.3) / (2 * np.pi)) - 1),
                g_vs_ladder=float(np.abs(G - Gd).max()),
                g_err_cont=float(np.abs(G - Gc).max()),
                G=[G.real.tolist(), G.imag.tolist()], Gc=[Gc.real.tolist(), Gc.imag.tolist()])


def case_lumped(h):
    m, mats, z = coax_mesh(h, 10e-3)
    s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    e1, s1 = edge_path(s, (2e-3, 0, 0), (4.6e-3, 0, 0))
    e2, sgn2 = edge_path(s, (0, 2e-3, z[-1]), (0, 4.6e-3, z[-1]))
    q = [LumpedPort(s, e1, s1, R=50.0, v_src=pulse(5e9, 1.5e9)), LumpedPort(s, e2, sgn2, R=50.0)]
    ps = attach_ports(s, q)
    out = dict(case="lumped", h=h, n_tets=m.n_cells, path_edges=[len(e1), len(e2)])
    errs = []
    for k in range(4):
        s.run(int(1e-9 / s.dt))
        V, I = q[0].spectra(F_C)
        a, b = power_waves(V, I, 1 / 50.0)
        errs.append(b / a)
    Vfd = ps.frequency_response(F_C, [1.0, 0.0])[:, 0]
    afd, bfd = power_waves(Vfd, 1.0 - Vfd / 50.0, 1 / 50.0)
    S_fd = bfd / afd
    out["td_minus_fd_vs_time_ns"] = [float(np.abs(e - S_fd).max()) for e in errs]
    out["S11_td"] = [errs[-1].real.tolist(), errs[-1].imag.tolist()]
    out["S11_fd"] = [S_fd.real.tolist(), S_fd.imag.tolist()]
    fdc = np.logspace(5, 9, 9)
    # Same paths with R_L = 75 Ohm for the DC limit Z_in -> R_L
    s2 = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    ps2 = attach_ports(s2, [LumpedPort(s2, e1, s1, R=50.0), LumpedPort(s2, e2, sgn2, R=75.0)])
    V = ps2.frequency_response(fdc, [1.0, 0.0])[:, 0]
    Zin = V / (1 - V / 50.0)
    out.update(f_dc=fdc.tolist(), zin_dc=[Zin.real.tolist(), Zin.imag.tolist()], dt=s.dt)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(case_wg, d) for d in (4, 6, 8, 12, 16, 24)]
    jobs += [(case_slab, d) for d in (4, 6, 8, 12, 16, 24)]
    jobs += [(case_coax, h) for h in (1.2e-3, 0.8e-3, 0.6e-3, 0.4e-3)]
    jobs += [(case_lumped, 1.0e-3)]
    # Largest first for load balance
    jobs = jobs[::-1]
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(_call, jobs)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res:
        print({k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)


def _call(job):
    fn, arg = job
    t0 = time.time()
    r = fn(arg)
    r["total_s"] = time.time() - t0
    return r


if __name__ == "__main__":
    main()
