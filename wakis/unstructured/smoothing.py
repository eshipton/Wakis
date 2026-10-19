# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
2D mesh smoothing towards well-centred (acute) Delaunay triangulations.

The diagonal (Voronoi) Hodge of an edge-based FIT discretization needs
positive dual edge lengths h_e = (L_e / 2)(cot a_1 + cot a_2), with a_1, a_2
the angles opposite the edge. Right or obtuse angles give h_e <= 0, i.e.
massless edges that collapse the leapfrog timestep (prism FIT, 3D TE). The
optimal-Delaunay-triangulation (ODT) update of Chen & Xu (2004) moves every
free node to the area-weighted mean of the circumcentres of its triangles,
which drives the mesh towards equilateral triangles; nodes on material
interfaces and on the boundary stay fixed, so interfaces are preserved.

Reference: L. Chen and J. Xu, "Optimal Delaunay triangulations," J. Comput.
Math. 22(2), 299-308 (2004).
"""

import numpy as np
from scipy.spatial import Delaunay

from .mesh import SimplexMesh, _circumcenter_tri


def interface_nodes(mesh):
    """Boolean mask of nodes on material interfaces or on the boundary."""
    ce = mesh.cell_edges.ravel()
    tri = np.repeat(np.arange(mesh.n_cells), 3)
    tag_min = np.full(mesh.n_edges, np.iinfo(np.int64).max)
    tag_max = np.full(mesh.n_edges, np.iinfo(np.int64).min)
    np.minimum.at(tag_min, ce, mesh.cell_tags[tri])
    np.maximum.at(tag_max, ce, mesh.cell_tags[tri])
    iface = tag_min != tag_max
    fixed = np.zeros(mesh.n_nodes, bool)
    fixed[mesh.edges[iface].ravel()] = True
    fixed[mesh.boundary_nodes] = True
    return fixed


def odt_smooth(mesh, region, n_iter=30, fixed=None):
    """
    ODT smoothing of a 2D Delaunay mesh with fixed interface/boundary nodes.

    Parameters
    ----------
    mesh : SimplexMesh
        Triangular mesh; its tags must equal ``region(centroids)``.
    region : callable
        (n, 2) points [m] -> (n,) integer material tag. A node move that
        changes the region of the node is rejected, so free nodes never
        cross an interface.
    n_iter : int
        Number of sweeps.
    fixed : ndarray of bool, optional
        Nodes that do not move (default `interface_nodes`).

    Returns
    -------
    SimplexMesh
        Re-triangulated (Delaunay) mesh, tagged with `region` at the
        triangle centroids.
    """
    nodes = mesh.nodes.copy()
    fixed = interface_nodes(mesh) if fixed is None else np.asarray(fixed, bool)
    reg0 = region(nodes)
    free = ~fixed
    for _ in range(int(n_iter)):
        tri = Delaunay(nodes).simplices
        P = nodes[tri]
        u, v = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        A = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cc = _circumcenter_tri(P)
        num = np.zeros_like(nodes)
        den = np.zeros(len(nodes))
        for k in range(3):
            np.add.at(num, tri[:, k], A[:, None] * cc)
            np.add.at(den, tri[:, k], A)
        new = num / np.maximum(den, 1e-300)[:, None]
        ok = free & (region(new) == reg0)
        nodes[ok] = new[ok]
    tri = Delaunay(nodes).simplices
    P = nodes[tri]
    u, v = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    tri = tri[area > 1e-12 * area.max()]
    return SimplexMesh(nodes, tri, region(nodes[tri].mean(1)))


def radial_region(radii):
    """Region function of concentric disks: tag k for r_{k-1} <= r < r_k."""
    radii = np.sort(np.atleast_1d(np.asarray(radii, float)))[:-1]

    def region(pts):
        return np.searchsorted(radii, np.hypot(*np.asarray(pts).T), side="right")

    return region


def dual_quality(mesh):
    """
    Normalized dual edge lengths h_e / L_e of the interior edges (0.577 for
    equilateral triangles; <= 0 for non-Delaunay edges).
    """
    h = np.asarray(mesh.dual_edge_partial.sum(1)).ravel() / mesh.edge_length
    inner = np.ones(mesh.n_edges, bool)
    inner[mesh.boundary_edges] = False
    return h[inner]
