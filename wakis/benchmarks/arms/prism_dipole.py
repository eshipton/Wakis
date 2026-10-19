# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Prism FIT arm for B4 (centre-fed thin-wire dipole): `PrismFIT3D` on a graded
triangle mesh extruded along the wire.

* cross-section: `graded_disk_mesh` with an interface ring on the wire
  surface r = a (n_ring >= 8 nodes, spacing h_t), a hexagonal lattice inside
  the wire, growth x1.2 per cell to lambda_min / 20; ODT smoothing; outer
  radius 0.21 lambda_max plus a matched lossy annulus of 0.3 lambda_max with
  a first-order ABC (the in-plane absorber of `PrismFIT3D`);
* z: uniform layers h_z = min(g, 4 h_t) along the wire (the gap is an integer
  number of layers), x1.2 growth to lambda_min / 20 up to 0.14 lambda_max
  beyond the wire ends, then a split-field z PML of 10 layers on each end;
* the wire: prisms of the wire disk with g/2 < |z| < L/2 are PEC (the wire
  surface is conforming: a polygon with the circle's area);
* the port: all vertical edges of the wire disk inside the gap (uniform
  generator field over the gap and the wire cross-section, the finite-gap
  model of the Hallen reference), R = Z0, `PrismFIT3D.add_lumped_port`.

The resolution label is h_t [m], the in-plane cell at the wire surface.
"""

import time

import numpy as np
from scipy.constants import c as c_light

from ..runner import Arm


def _z_levels(L, g, hz, h_out, z_end, n_pml, growth=1.2):
    """Symmetric z levels: the gap is m whole layers, uniform hz along the wire."""
    m = max(1, int(round(g / hz)))
    hz = g / m
    n = int(np.ceil((L / 2 + 5 * hz - g / 2) / hz))
    zs = list(g / 2 + hz * np.arange(n + 1))
    dz = hz
    while zs[-1] < z_end:
        dz = min(h_out, dz * growth)
        zs.append(zs[-1] + dz)
    zs = np.r_[zs, zs[-1] + dz * np.arange(1, n_pml + 1)]
    gap = -g / 2 + hz * np.arange(1, m)
    return np.r_[-zs[::-1], gap, zs], hz


class PrismDipoleArm(Arm):
    def __init__(self, ladder=None, n_pml=10, cfl=0.95, absorber=0.3, name=None):
        self.ladder = ladder
        self.n_pml = n_pml
        self.cfl = cfl
        self.absorber = absorber
        self.name = name or "prism-graded"

    def supports(self, case):
        return case.name == "B4"

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        return [case.radius / q for q in (1, 1.5, 2, 3, 4)]

    def build(self, case, ht):
        from ...unstructured import PrismFIT3D, PrismMesh, graded_disk_mesh
        from ...unstructured.smoothing import odt_smooth, radial_region

        L, a, g = case.length, case.radius, case.gap
        lam_min = c_light / case.freqs.max()
        lam_max = c_light / case.freqs.min()
        h_out = lam_min / 20
        r_int = 0.21 * lam_max
        d_abs = self.absorber * lam_max
        r_dom = r_int + d_abs
        h_ring = min(ht, 2 * np.pi * a / 8)

        def size(r):
            d = max(r - a, 0.0)
            return min(h_out, h_ring * 1.2 ** (d / h_ring))

        radii = [a, r_dom]
        m2 = graded_disk_mesh(radii, size)
        m2 = odt_smooth(m2, radial_region(radii), 20)
        hz = min(g, 4 * ht)
        z, hz = _z_levels(L, g, hz, h_out, L / 2 + 0.14 * lam_max, self.n_pml)

        def tags(tri, zm):
            wire = (tri == 0) & (np.abs(zm) > g / 2) & (np.abs(zm) < L / 2)
            return np.where(wire, 1, 0)

        pm = PrismMesh(m2, z, tags=tags, merge_tol=0.15)
        n_lo = self.n_pml
        s_max = 4 * c_light / (n_lo * (z[-1] - z[-2]))
        solver = PrismFIT3D(
            pm, {1: "pec"}, bc_xy="abc", bc_z="pec", cfl=self.cfl,
            absorber_xy=dict(r0=r_int, r1=r_dom, s_max=3 * c_light / d_abs, order=2),
            pml_z=dict(n_low=n_lo, n_high=n_lo, s_max=s_max, order=3),
        )  # fmt: skip
        # port: vertical edges of the wire-disk nodes in the gap layers
        c2 = pm.c2
        wire_cells = np.flatnonzero(pm.tags[:, np.argmin(np.abs(pm.z_mid))] == 0)
        wire_cells = wire_cells[np.hypot(*c2.cell_centroid[wire_cells].T) < a]
        nodes = np.unique(c2.cell_node_abs[wire_cells].indices)
        layers = np.flatnonzero(np.abs(pm.z_mid) < g / 2)
        edges = solver.vertical_edge_index(nodes, layers)
        edges = edges[~solver.fixed_mask[edges]]
        port = solver.add_lumped_port(edges, case.z0, case.source(), length=g)
        info = dict(hz=hz, h_out=h_out, h_ring=h_ring, nodes2d=int(c2.n_nodes),
                    layers=int(pm.K), port_edges=int(edges.size),
                    merged=int(c2.n_removed))  # fmt: skip
        return solver, port, info

    def run(self, case, ht):
        t0 = time.time()
        solver, port, info = self.build(case, ht)
        setup_s = time.time() - t0
        n_steps = int(case.t_run / solver.dt)
        t1 = time.time()
        solver.run(n_steps)
        run_s = time.time() - t1
        z = solver.port_impedance(port, case.freqs)
        out = dict(z_in=z)
        out["cost"] = dict(
            unknowns=int(solver.n_unknowns), cells=int(solver.mesh.n_cells),
            steps=n_steps, dt=solver.dt, run_s=run_s, setup_s=setup_s, h=ht, **info,
        )  # fmt: skip
        return out
