"""
Mutation checks for tests/test_101_unstructured_ports3d.py.

Each mutation monkeypatches one ingredient of `wakis.unstructured.ports3d`
and runs the whole test file in its own process; a mutation is *caught* when
at least one test fails. Usage (Cloud Batch, from the worktree root):

    python studies/unstructured-ports3d/mutations.py [out.json]
"""

import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

TEST = "tests/test_101_unstructured_ports3d.py"
MUTATIONS = [
    "none",  # control: everything passes
    "monitor_V_at_n",  # record V^n instead of the trapezoidal V^{n+1/2}
    "explicit_conductance",  # drop the implicit (Woodbury) half of the port conductance
    "backward_euler_conductance",  # g fully implicit instead of trapezoidal
    "continuum_port_admittance",  # reference Y = beta / (w mu) without the stencil factors
    "transpose_no_massless",  # drop the massless (Schur) term of linear_step_T
    "tem_no_normalization",  # TEM mode not normalized (phi^T M phi != 1)
]


def apply(name):
    import wakis.unstructured.ports3d as P

    if name == "monitor_V_at_n":
        orig = P.PortSet.step

        def step(self, u, r):
            n0 = [len(p.V) for p in self.ports]
            out = orig(self, u, r)
            Vn = self.W.T @ u
            for k, p in enumerate(self.ports):
                p.V[n0[k]] = Vn[k]
            return out

        P.PortSet.step = step
    elif name == "explicit_conductance":
        def step(self, u, r):
            t = (self.solver.n + 0.5) * self.dt
            Is = np.array([p.source(t) for p in self.ports])
            Vn = self.W.T @ u
            u_new = self.base.step(u, r + self.W @ (Is - self.g * Vn))
            Vavg = 0.5 * (Vn + self.W.T @ u_new)
            for k, p in enumerate(self.ports):
                p.V.append(Vavg[k])
                p.Is.append(Is[k])
            return u_new

        P.PortSet.step = step
    elif name == "backward_euler_conductance":
        orig_init = P.PortSet.__init__

        def init(self, solver, ports):
            orig_init(self, solver, ports)
            self.S = np.eye(len(self.ports)) + self.g[:, None] * self.Q
            self._Sinv = np.linalg.inv(self.S)
            self._be = True

        def step(self, u, r):
            # implicit part g (not g/2), no explicit part
            t = (self.solver.n + 0.5) * self.dt
            Is = np.array([p.source(t) for p in self.ports])
            Vn = self.W.T @ u
            u_star = self.base.step(u, r + self.W @ Is)
            alpha = self._Sinv @ (self.g * (self.W.T @ u_star))
            u_new = u_star - self.dinv * (self.W @ alpha)
            Vavg = 0.5 * (Vn + self.W.T @ u_new)
            for k, p in enumerate(self.ports):
                p.V.append(Vavg[k])
                p.Is.append(Is[k])
            return u_new

        P.PortSet.__init__ = init
        P.PortSet.step = step
    elif name == "continuum_port_admittance":
        def port_admittance(self, f, side=0):
            layer = 0 if side == 0 else -1
            w = 2 * np.pi * np.asarray(f, float)
            k2 = w**2 * self.eps[layer] * self.mu[layer] - self.kc2
            return np.sqrt(k2.astype(complex)) / (w * self.mu[layer]) * self.cos_half(f)

        P.ExtrudedLine.port_admittance = port_admittance
    elif name == "transpose_no_massless":
        def linear_step_T(self, ub, bb):
            s = self.solver
            dt = self.dt
            z = self.W.T @ (self.dinv * ub)
            y = ub / self.base.den
            y = y - (self.W @ (0.5 * self.g * (self._Sinv.T @ z))) / self.base.den
            u_b = self.base.a1 * y - dt * (self.W @ (0.5 * self.g * (self.W.T @ y)))
            Is_b = dt * (self.W.T @ y)
            b_b = bb + s.Mnu @ (s.Cf @ (dt * y))
            return u_b - dt * (s.Cf.T @ b_b), b_b, Is_b

        P.PortSet.linear_step_T = linear_step_T
    elif name == "tem_no_normalization":
        orig = P.PortPlane.tem_mode

        def tem_mode(self):
            phi, v = orig(self)
            return phi * v, 1.0

        P.PortPlane.tem_mode = tem_mode
    elif name != "none":
        raise ValueError(name)


def run(name):
    code = (
        "import sys, pytest; sys.path.insert(0, 'studies/unstructured-ports3d'); "
        f"import mutations; mutations.apply({name!r}); "
        f"sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '-W', 'ignore', '{TEST}']))"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:]
    return dict(mutation=name, returncode=p.returncode, failed=lines, summary=summary)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "mutations.json"
    with Pool(min(len(MUTATIONS), os.cpu_count())) as pool:
        res = pool.map(run, MUTATIONS)
    for r in res:
        caught = "control" if r["mutation"] == "none" else ("CAUGHT" if r["returncode"] else "MISSED")
        print(f"{r['mutation']:30s} {caught:8s} {r['summary']}")
        for ln in r["failed"]:
            print("    ", ln)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
