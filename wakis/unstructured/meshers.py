# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Mesh generators for unstructured FIT.

The pure-Python generators (`disk_mesh`, `ball_mesh`) build interface-
conforming Delaunay meshes of concentric disks / balls from a regular
interior lattice plus point rings (2D) or shells (3D) sampled on every
material interface and on the outer boundary, triangulated with
`scipy.spatial.Delaunay`. They need no external mesher and give meshes whose
interior dual measures are non-negative (Delaunay property), which the
diagonal Voronoi Hodge of `UnstructuredFIT2D`/`UnstructuredFIT3D` requires.

`gmsh_mesh` drives the optional external mesh generator gmsh (GPL, not a
Wakis dependency) for CAD-like geometries: it is only imported when called,
and the mesh is exchanged through a file read with meshio.
"""

import numpy as np
from scipy.spatial import Delaunay

from .mesh import SimplexMesh


def _ring(r, h, preserve_area=False):
    n = max(8, int(np.ceil(2 * np.pi * r / h)))
    th = 2 * np.pi * np.arange(n) / n
    if preserve_area:
        # Regular n-gon with the area of the circle: n R^2 sin(2 pi / n) / 2
        r = r * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    return r * np.stack([np.cos(th), np.sin(th)], 1)


def _fibonacci_sphere(r, h):
    """Nearly uniform points on a sphere with spacing ~h (hexagonal patches)."""
    n = max(12, int(round(4 * np.pi * r**2 / (h**2 * np.sqrt(3) / 2))))
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + np.sqrt(5)) * k
    s = np.sqrt(1 - z**2)
    return r * np.stack([s * np.cos(phi), s * np.sin(phi), z], 1)


def _tags_by_radius(centroid_r, radii):
    """Tag k for cells with r_{k-1} <= r < r_k (k = 0 innermost)."""
    return np.searchsorted(np.asarray(radii), centroid_r, side="right")


def disk_mesh(radii, h, clearance=0.5, preserve_area=True):
    """
    Delaunay triangulation of concentric disks with conforming interfaces.

    Nodes are placed on a ring at every radius in `radii` (spacing ~h) and on
    a hexagonal lattice (spacing h) elsewhere, keeping lattice nodes at least
    ``clearance * h`` away from every ring. Each triangle is one material.

    Parameters
    ----------
    radii : sequence of float
        Increasing radii [m]; the last one is the outer boundary.
    h : float
        Target edge length [m].
    clearance : float, optional
        Minimum lattice-to-ring distance in units of h.
    preserve_area : bool, optional
        Place each ring on a slightly larger radius so that the polygon
        encloses exactly the area of its circle (removes the leading
        geometric error of the inscribed polygon; still 2nd order).

    Returns
    -------
    SimplexMesh
        Triangles tagged 0 (r < radii[0]), 1 (radii[0] < r < radii[1]), ...
    """
    radii = np.sort(np.atleast_1d(np.asarray(radii, dtype=float)))
    a = radii[-1]
    xs = np.arange(-a, a + h, h)
    ys = np.arange(-a, a + h, h * np.sqrt(3) / 2)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X = X + (np.arange(len(ys)) % 2)[:, None] * h / 2
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    r = np.hypot(*pts.T)
    keep = r < a - clearance * h
    for rk in radii[:-1]:
        keep &= np.abs(r - rk) > clearance * h
    nodes = np.vstack([_ring(rk, h, preserve_area) for rk in radii[::-1]] + [pts[keep]])
    tri = Delaunay(nodes).simplices
    rc = np.hypot(*nodes[tri].mean(1).T)
    return SimplexMesh(nodes, tri, _tags_by_radius(rc, radii[:-1]))


def _odt_step(nodes, on_shell, radii, gap):
    """
    One optimal-Delaunay-triangulation smoothing step (Chen & Xu 2004).

    Every node moves to the volume-weighted mean of the circumcentres of its
    tetrahedra; shell nodes are projected back onto their sphere and interior
    nodes are kept at least `gap` away from every sphere.
    """
    tet = Delaunay(nodes).simplices
    P = nodes[tet]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    A = P[:, 1:] - P[:, :1]
    rhs = 0.5 * np.einsum("nij,nij->ni", A, A)
    ok = vol > 1e-12 * vol.max()
    cc = np.zeros((len(tet), 3))
    cc[ok] = P[ok, 0] + np.linalg.solve(A[ok], rhs[ok, :, None])[..., 0]
    num = np.zeros_like(nodes)
    den = np.zeros(len(nodes))
    for k in range(4):
        np.add.at(num, tet[ok, k], vol[ok, None] * cc[ok])
        np.add.at(den, tet[ok, k], vol[ok])
    new = num / np.maximum(den, 1e-300)[:, None]
    r_new = np.linalg.norm(new, axis=1)
    shell = on_shell > 0
    new[shell] *= (on_shell[shell] / r_new[shell])[:, None]
    # Interior nodes: stay inside their layer, `gap` away from the spheres
    inner = ~shell
    r_old = np.linalg.norm(nodes[inner], axis=1)
    layer = np.searchsorted(radii, r_old)
    lo = np.where(layer > 0, radii[np.maximum(layer - 1, 0)] + gap, 0.0)
    hi = radii[np.minimum(layer, len(radii) - 1)] - gap
    rn = r_new[inner]
    target = np.clip(rn, lo, hi)
    scale = np.where(rn > 0, target / np.maximum(rn, 1e-300), 1.0)
    new[inner] *= scale[:, None]
    return new


def ball_mesh(radii, h, clearance=0.55, smooth=8):
    """
    Delaunay tetrahedralization of concentric balls with conforming shells.

    Nodes are placed on a Fibonacci point set on every sphere in `radii`
    (spacing ~h) and on a body-centred cubic lattice (cube edge h) elsewhere,
    keeping lattice nodes at least ``clearance * h`` from every sphere. All
    nodes lie inside the outer sphere and its nodes are extreme points, so
    the Delaunay tetrahedra fill the polyhedral ball exactly. Each
    tetrahedron is tagged by the radius of its centroid.

    Parameters
    ----------
    radii : sequence of float
        Increasing radii [m]; the last one is the outer boundary.
    h : float
        Target edge length [m].
    clearance : float, optional
        Minimum lattice-to-shell distance in units of h.
    smooth : int, optional
        Number of optimal-Delaunay (ODT) smoothing sweeps of the node set
        (Chen & Xu 2004), which removes most slivers.

    Returns
    -------
    SimplexMesh
        Tetrahedra tagged 0 (r < radii[0]), 1, ... by shell.
    """
    radii = np.sort(np.atleast_1d(np.asarray(radii, dtype=float)))
    a = radii[-1]
    g = np.arange(-a, a + h, h)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    cube = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = np.vstack([cube, cube + h / 2])
    r = np.linalg.norm(pts, axis=1)
    keep = r < a - clearance * h
    for rk in radii[:-1]:
        keep &= np.abs(r - rk) > clearance * h
    shells = [_fibonacci_sphere(rk, h) for rk in radii[::-1]]
    on_shell = np.concatenate([np.full(len(s), r) for s, r in zip(shells, radii[::-1])])
    nodes = np.vstack(shells + [pts[keep]])
    on_shell = np.r_[on_shell, np.zeros(keep.sum())]
    for _ in range(smooth):
        nodes = _odt_step(nodes, on_shell, radii, clearance * h)
    tet = Delaunay(nodes).simplices
    P = nodes[tet]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    tet = tet[vol > 1e-10 * h**3]
    rc = np.linalg.norm(nodes[tet].mean(1), axis=1)
    tags = _tags_by_radius(rc, radii[:-1])
    return SimplexMesh(nodes, tet, tags)


def gmsh_mesh(build, h, filename, dim=3, algorithm=None, verbose=False):
    """
    Mesh a geometry with the optional external generator gmsh.

    gmsh (https://gmsh.info, GPL) is not a dependency of Wakis; it is only
    imported here, writes `filename` and the mesh is read back with meshio.
    Physical groups defined by `build` become the cell tags.

    Parameters
    ----------
    build : callable
        ``build(gmsh)`` creates the geometry with the gmsh API (e.g.
        ``gmsh.model.occ``), synchronizes it and defines one physical group
        per material region (tag = physical group number).
    h : float
        Maximum element size [m].
    filename : str
        Output ``.msh`` file.
    dim : int, optional
        Mesh dimension (2 or 3).
    algorithm : int, optional
        gmsh ``Mesh.Algorithm3D`` (3D) or ``Mesh.Algorithm`` (2D) option.
    verbose : bool, optional
        Show gmsh terminal output.

    Returns
    -------
    SimplexMesh

    Raises
    ------
    ImportError
        If the gmsh Python module is not installed.
    """
    try:
        import gmsh
    except ImportError as err:
        raise ImportError(
            "gmsh_mesh needs the optional external mesh generator gmsh "
            "(`pip install gmsh`, GPL licensed, not bundled with Wakis). "
            "Alternatively mesh with any tool and load the file with "
            "SimplexMesh.read, or use disk_mesh/ball_mesh."
        ) from err
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.model.add("wakis")
        build(gmsh)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2 * h)
        if algorithm is not None:
            key = "Mesh.Algorithm3D" if dim == 3 else "Mesh.Algorithm"
            gmsh.option.setNumber(key, algorithm)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(dim)
        gmsh.write(filename)
    finally:
        gmsh.finalize()
    return SimplexMesh.read(filename, tag_key="gmsh:physical")


def gmsh_ball_mesh(radii, h, filename, verbose=False):
    """
    Concentric balls meshed with gmsh (see `gmsh_mesh`).

    Parameters
    ----------
    radii : sequence of float
        Increasing radii [m]; the last one is the outer boundary.
    h : float
        Maximum element size [m].
    filename : str
        Output ``.msh`` file.

    Returns
    -------
    SimplexMesh
        Tetrahedra tagged 0 (innermost ball), 1, ... as for `ball_mesh`.
    """
    radii = np.sort(np.atleast_1d(np.asarray(radii, dtype=float)))

    def build(gmsh):
        occ = gmsh.model.occ
        balls = [occ.addSphere(0, 0, 0, r) for r in radii]
        out, _ = occ.fragment([(3, balls[-1])], [(3, b) for b in balls[:-1]])
        occ.synchronize()
        for _, vol in out:
            # each fragment is a shell; its outer radius identifies it
            bb = gmsh.model.getBoundingBox(3, vol)
            k = int(np.argmin(np.abs(radii - 0.5 * (bb[3] - bb[0]))))
            gmsh.model.addPhysicalGroup(3, [vol], k + 1)

    mesh = gmsh_mesh(build, h, filename, dim=3, verbose=verbose)
    mesh.cell_tags -= 1
    return mesh
