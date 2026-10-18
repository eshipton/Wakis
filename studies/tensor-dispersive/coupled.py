"""
RESEARCH ONLY (not part of the Wakis API): the edge-pixel dispersive tensor,
i.e. the off-diagonal entries of the averaged tensor of dispersive pixels,
T_ij(e, w) = m_i m_j (z_n(w) - z_t(w)), added to the corner couplings of
`wakis.tensor_averaging` through the sub-capacitor engine of
`wakis.tensor_dispersive`. It is linear and exactly transposable and
reproduces the static full-tensor accuracy for a Lorentz dielectric, but it
is not a sum of positive quadratic forms: with metals (Drude, lossless
Lorentz) the time stepping is unstable. See the physics guide, section
"Dispersive subpixel averaging", and README.md in this folder.

Usage: solver = make_coupled(solver) on a solver built with
conformal='tensor' and dispersive STL solids (before the first step).
"""

import types

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import coo_matrix

from wakis.tensor_dispersive import DispersiveNetwork, conformal_caps

# Static anisotropy [m/F] registered for dispersive pixels whose eps_inf
# averages are isotropic (Drude with eps_inf = 1 in vacuum), so that the tensor
# operator provides their corner-coupling coefficients; negligible otherwise
TINY_ANISO = 1e-12 / eps_0


def coupled_network(solver, op, sel, f_disp, edge, kind, c, einf, materials,
                     members):  # fmt: skip
    """
    Diagonal sub-capacitors plus the dispersive off-diagonal tensor entries
    T_ij(e, w) = m_i m_j (z_n(w) - z_t(w)) of every dispersive pixel e, in
    the symmetric corner coupling S_ab = c_ab (T_ij(a) + T_ij(b)) of
    `wakis.tensor_averaging`:

    - E_b += c_eb T_ij(e, w) D_e / W_b reuses the pixel's own capacitors
      (u_T = D_e / eps_t, u_m = D_e / eps_m; z_n D - z_t D = sum f_m u_m - u_T
      up to static terms);
    - E_e += m_i (z_n - z_t)(w) Q_e, Q_e = sum_b c_eb m_j D_b / W_e, needs a
      second set of capacitors with the same models, driven by Q_e.

    The static (eps_inf) parts stay in the tensor operator; the network adds
    the dynamic remainder. Unlike the diagonal network this is not a sum of
    positive quadratic forms (see the physics guide; it is unstable for metals).
    """
    if np.iscomplexobj(op.pair_coef) and np.any(np.imag(op.pair_coef) != 0):
        raise NotImplementedError("Dispersive tensor couplings with Bloch phases.")
    n_mat = len(materials)
    flat_e = sel[edge]
    n_caps = edge.size
    pos_sel = {int(f): i for i, f in enumerate(sel)}
    # tangential and series capacitors of each edge (`diagonal_caps` order)
    t_cap, s_cap = {}, [dict() for _ in range(n_mat)]
    for k in range(n_caps):
        if kind[k] < 0:
            t_cap[int(flat_e[k])] = k
        else:
            s_cap[kind[k]][int(flat_e[k])] = k

    W = op.W
    ea, eb, coef = op.pair_a, op.pair_b, np.real(op.pair_coef)
    Bi, Bj, Bv = list(np.arange(n_caps)), list(flat_e), list(np.ones(n_caps))
    Ci, Cj, Cv = list(flat_e), list(np.arange(n_caps)), list(c)
    new_eps, new_members = [], [([], []) for _ in range(n_mat)]
    nb_caps = {}  # neighbour-charge capacitors of a pixel: (tangential, series)

    def add_nb(e):
        if e in nb_caps:
            return nb_caps[e]
        k0 = n_caps + len(new_eps)
        i_e = pos_sel[e]
        new_eps.append(einf[t_cap[e]])
        for m in range(n_mat):
            if f_disp[m, i_e] > 0:
                new_members[m][0].append(k0)
                new_members[m][1].append(f_disp[m, i_e])
        series = {}
        for m in range(n_mat):
            if e in s_cap[m]:
                series[m] = n_caps + len(new_eps)
                new_eps.append(materials[m].eps_inf)
                new_members[m][0].append(series[m])
                new_members[m][1].append(1.0)
        nb_caps[e] = (k0, series)
        return nb_caps[e]

    _, vec_a = op._lookup(ea)
    _, vec_b = op._lookup(eb)
    for a, b, cf, va, vb in zip(ea, eb, coef, vec_a, vec_b):
        for e, o, vec in ((int(a), int(b), va), (int(b), int(a), vb)):
            if e not in t_cap or not any(e in s_cap[m] for m in range(n_mat)):
                continue
            mi, mj = vec[e // op.N], vec[o // op.N]
            if mi * mj == 0:
                continue
            i_e = pos_sel[e]
            # E_o += cf m_i m_j (sum f_m u_m - u_T)(e) / W_o
            w_o = cf * mi * mj / W[o]
            Ci.append(o), Cj.append(t_cap[e]), Cv.append(-w_o)
            for m in range(n_mat):
                if e in s_cap[m]:
                    (
                        Ci.append(o),
                        Cj.append(s_cap[m][e]),
                        Cv.append(w_o * f_disp[m, i_e]),
                    )
            # E_e += m_i (sum f_m u'_m - u'_T), charge Q_e += cf m_j D_o / W_e
            kt, series = add_nb(e)
            Bi.append(kt), Bj.append(o), Bv.append(cf * mj / W[e])
            for m, ks in series.items():
                Bi.append(ks), Bj.append(o), Bv.append(cf * mj / W[e])
    for e, (kt, series) in nb_caps.items():
        _, vec = op._lookup(np.array([e]))
        mi = vec[0, e // op.N]
        Ci.append(e), Cj.append(kt), Cv.append(-mi)
        for m, ks in series.items():
            Ci.append(e), Cj.append(ks), Cv.append(mi * f_disp[m, pos_sel[e]])
    K = n_caps + len(new_eps)
    eps_inf = np.concatenate([einf, new_eps])
    members = [
        (np.concatenate([s, np.array(ns, int)]), np.concatenate([mu, np.array(nm)]))
        for (s, mu), (ns, nm) in zip(members, new_members)
    ]
    rows, inv = np.unique(
        np.concatenate([np.array(Bj), np.array(Ci)]), return_inverse=True
    )
    nB = len(Bj)
    bm = coo_matrix((Bv, (Bi, inv[:nB])), shape=(K, rows.size))
    cm = coo_matrix((Cv, (inv[nB:], Cj)), shape=(rows.size, K))
    net = DispersiveNetwork(solver, rows, bm, cm, eps_inf, materials, members)
    return net


def _one_step_coupled(solver):
    """Base step + tensor correction + network on the diagonal increment."""
    net = solver.dispersive_network
    op = solver.tensor_operator
    e_old = solver.E.array[net.rows_dev].copy()
    solver._one_step_unnetworked()
    pos_net, pos_op = net.tensor_map
    e_ref = e_old.copy()
    e_ref[pos_net] += (op.Kp @ op.last_increment)[pos_op]
    net.correct(solver, e_ref)


def make_coupled(solver):
    """Replace the diagonal dispersive network of `solver` by the coupled one."""
    op = solver.tensor_operator
    mix = solver._conformal_mixture
    caps = conformal_caps(solver)
    normal = np.concatenate([np.reshape(n, (-1, 3), order="F") for n in mix["normal"]])
    sel, p = caps["sel"], caps["p"]
    cut = sel[(p > 0) & (p < 1)]
    new = cut[op._lookup(cut)[0] == 0]
    if new.size:
        op.set_edges(new, np.full(new.size, TINY_ANISO), normal[new])
    op.assemble(solver)  # no exclusion: the network shares the coupled edges
    net = coupled_network(
        solver,
        op,
        sel,
        caps["f_disp"],
        caps["edge"],
        caps["kind"],
        caps["c"],
        caps["eps_inf"],
        caps["materials"],
        caps["members"],
    )
    _, pos_net, pos_op = np.intersect1d(net.rows, op.rows, return_indices=True)
    net.tensor_map = (pos_net, pos_op)
    solver.dispersive_network = net
    solver.one_step = types.MethodType(_one_step_coupled, solver)
    return solver
