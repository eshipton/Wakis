# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
CAD -> tetrahedral mesh import for the unstructured FIT solver.

Three entry points, all returning a `SimplexMesh` whose cell tags index a
``materials`` dict that `UnstructuredFIT3D` accepts directly:

* `mesh_cad` meshes a STEP/BREP/IGES file (or a geometry built with a
  callable) with the optional external mesh generator gmsh.
* `mesh_stl` meshes a closed STL surface (optionally inside a padding box)
  with gmsh.
* `read_mesh` reads an existing ``.msh``/``.vtu``/... file with meshio and
  maps its named physical groups to materials.

gmsh (https://gmsh.info) is GPL licensed and is **not** a Wakis
dependency: it is imported lazily inside the functions that drive it, never
bundled, and only its public API is called. meshio (MIT) is likewise
optional (only `read_mesh` needs it).

Mesh size from cells per wavelength
-----------------------------------
The target edge length of region m is

    h_m = c0 / (f_max n_m cpw),    n_m = sqrt(eps_r mu_r),

so ``cpw`` edges per wavelength are placed in every material at the highest
frequency of interest (project convention: cpw counted in the
highest-index material, 10 for fast runs, 20 for accurate ones).
``size_mode="uniform"`` uses the highest-index h_m everywhere instead.
"""

import numpy as np
from scipy.constants import c as C0

from .mesh import SimplexMesh

__all__ = [
    "size_from_cpw",
    "region_sizes",
    "mesh_cad",
    "mesh_stl",
    "read_mesh",
    "CADMeshInfo",
    "estimate_n_tets",
    "cad_projector",
]


def _require_gmsh():
    try:
        import gmsh
    except ImportError as err:  # pragma: no cover - exercised without gmsh
        raise ImportError(
            "CAD meshing needs the optional external mesh generator gmsh "
            "(`pip install gmsh`, GPL licensed, not bundled with Wakis). "
            "Alternatively mesh with any tool and load the file with "
            "wakis.unstructured.cad.read_mesh."
        ) from err
    return gmsh


def _index(material):
    """Refractive index sqrt(eps_r mu_r) of a material spec (None for PEC)."""
    if isinstance(material, str):
        if material.lower() != "pec":
            raise ValueError(f"unknown material string {material!r}")
        return None
    if hasattr(material, "eps_inf"):  # wakis.dispersive.DispersiveMaterial
        eps = float(np.real(getattr(material, "eps_inf")))
        return np.sqrt(max(eps, 1.0))
    m = np.atleast_1d(np.asarray(material, dtype=float))
    eps = m[0]
    mu = m[1] if m.size > 1 else 1.0
    return float(np.sqrt(eps * mu))


def size_from_cpw(f_max, n=1.0, cpw=10):
    """
    Target edge length h = c0 / (f_max n cpw) [m].

    Parameters
    ----------
    f_max : float
        Highest frequency of interest [Hz].
    n : float
        Refractive index of the material.
    cpw : float
        Cells (edges) per wavelength in that material (10 fast, 20 accurate).
    """
    if f_max <= 0 or cpw <= 0 or n <= 0:
        raise ValueError("f_max, n and cpw must be positive")
    return C0 / (f_max * n * cpw)


def region_sizes(materials, f_max, cpw=10, size_mode="per_material"):
    """
    Target edge length of every named region [m].

    Parameters
    ----------
    materials : dict
        ``{name: material}`` (``[eps_r, mu_r]``, ``[eps_r, mu_r, sigma]``,
        a DispersiveMaterial or ``"pec"``).
    f_max : float
        Highest frequency [Hz].
    cpw : float
        Cells per wavelength in the highest-index material.
    size_mode : {"per_material", "uniform"}
        Own wavelength in every material, or the highest-index size
        everywhere.

    Returns
    -------
    dict
        ``{name: h}``; PEC regions get the smallest dielectric size.
    """
    if size_mode not in ("per_material", "uniform"):
        raise ValueError("size_mode must be 'per_material' or 'uniform'")
    idx = {k: _index(m) for k, m in materials.items()}
    n_max = max([n for n in idx.values() if n is not None] or [1.0])
    h_min = size_from_cpw(f_max, n_max, cpw)
    out = {}
    for k, n in idx.items():
        if n is None or size_mode == "uniform":
            out[k] = h_min
        else:
            out[k] = size_from_cpw(f_max, n, cpw)
    return out


class CADMeshInfo(dict):
    """
    Bookkeeping of a CAD mesh (a dict with attribute access).

    Keys: ``region_tags`` ({name: tag}), ``materials`` ({tag: material},
    ready for `UnstructuredFIT3D`), ``sizes`` ({name: h [m]}),
    ``volumes`` ({name: [gmsh volume tags]}), ``scale``, ``f_max``,
    ``cpw``, ``n_tets``.
    """

    __getattr__ = dict.__getitem__


#: Tetrahedra per h^3 of volume for a size-h Delaunay mesh (regular tetrahedron
#: volume h^3 / (6 sqrt 2) = 0.118 h^3; measured gmsh meshes at cpw 10-20 have
#: 6.5-7.5 tets per h^3 with h the mean edge length)
TETS_PER_H3 = 8.5


def estimate_n_tets(volume, h):
    """Rough number of tetrahedra of a size-h mesh of `volume` (same units)."""
    return TETS_PER_H3 * volume / h**3


def _preflight(n_est, max_tets):
    if max_tets is not None and n_est > max_tets:
        raise MemoryError(
            f"estimated {n_est:.3g} tetrahedra > max_tets = {max_tets:.3g}; "
            "lower cpw or f_max, check `scale` (metres per CAD unit), or raise "
            "max_tets (see wakis.unstructured.estimate_memory)"
        )


def _set_size_fields(gmsh, vol_sizes, curvature_points=0):
    """One Constant field per volume (h inside and on its boundary), Min of all."""
    fld = gmsh.model.mesh.field
    ids = []
    for vols, h in vol_sizes:
        f = fld.add("Constant")
        fld.setNumber(f, "VIn", h)
        fld.setNumber(f, "VOut", 1e22)
        fld.setNumbers(f, "VolumesList", list(vols))
        fld.setNumber(f, "IncludeBoundary", 1)
        ids.append(f)
    fmin = fld.add("Min")
    fld.setNumbers(fmin, "FieldsList", ids)
    fld.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(curvature_points))


def _configure(gmsh, algorithm, optimize):
    if algorithm is not None:
        gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm))
    if optimize not in (None, "gmsh", "netgen", "none"):
        raise ValueError("optimize must be None/'none', 'gmsh' or 'netgen'")
    gmsh.option.setNumber("Mesh.Optimize", 0 if optimize in (None, "none") else 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", int(optimize == "netgen"))
    gmsh.option.setNumber("Mesh.ElementOrder", 1)


def _collect(gmsh, vol_region, names, scale, return_tags=False):
    """Nodes and tetrahedra of the current gmsh mesh, tagged per region."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags = tags[order]
    X = coords.reshape(-1, 3)[order] * scale
    cells, cell_tags = [], []
    for vol, name in vol_region.items():
        etypes, _, enodes = gmsh.model.mesh.getElements(3, vol)
        for et, en in zip(etypes, enodes):
            if et != 4:  # 4-node tetrahedron
                raise ValueError(f"unexpected 3D element type {et} (need tets)")
            con = np.searchsorted(tags, en.astype(np.int64)).reshape(-1, 4)
            cells.append(con)
            cell_tags.append(np.full(len(con), names.index(name)))
    if not cells:
        raise ValueError("no tetrahedra were generated")
    mesh = SimplexMesh(X, np.vstack(cells), np.concatenate(cell_tags))
    if return_tags:
        # SimplexMesh keeps the node order (all gmsh nodes are used by tets
        # once the dropped volumes are removed); check it
        if mesh.n_nodes != len(tags):
            raise RuntimeError("gmsh mesh has nodes outside the tetrahedra")
        return mesh, tags
    return mesh


def _min_volume_ratio(gmsh):
    """Smallest |volume| / largest |volume| of the current gmsh tetrahedra."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    X = coords.reshape(-1, 3)[order]
    _, _, en = gmsh.model.mesh.getElements(3)
    if not en:
        return 1.0
    con = np.searchsorted(tags[order], np.concatenate(en).astype(np.int64)).reshape(-1, 4)
    P = X[con]
    v = np.abs(np.linalg.det(P[:, 1:] - P[:, :1]))
    return float(v.min() / v.max())


def _remove_slivers(gmsh, optimize, tol):
    """
    Delaunay output can contain flat slivers (four nearly coplanar nodes, e.g.
    on a planar interface; zero volume on some platforms). With
    ``optimize=None`` run gmsh's optimizer on those elements only
    (``Mesh.OptimizeThreshold`` 0.01): the rest of the mesh stays Delaunay.
    """
    if optimize not in (None, "none") or _min_volume_ratio(gmsh) >= tol:
        return False
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.01)
    gmsh.model.mesh.optimize("")
    return True


def cad_projector(gmsh, node_tags, scale):
    """
    Sliding rule onto the CAD surfaces for `improve_dt` (gmsh must be live).

    Nodes classified on a CAD surface (dim 2) slide on it (closest-point
    projection with the OpenCASCADE kernel); nodes on CAD curves and points
    stay fixed; volume nodes are free (the projector is not called for them).
    """
    surf_of = {}
    for _, s in gmsh.model.getEntities(2):
        nt, _, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=False)
        for t in nt:
            surf_of[int(t)] = s

    def project(n, Y):
        s = surf_of.get(int(node_tags[n]))
        if s is None:
            return None
        Y = np.atleast_2d(Y)
        out, _ = gmsh.model.getClosestPoint(2, s, (Y / scale).ravel().tolist())
        return np.asarray(out).reshape(-1, 3) * scale

    project.n_sliding = len(surf_of)
    return project


def _match_regions(gmsh, regions, cad_names=None):
    """Map gmsh volumes to region names (point inside, name regex, callable)."""
    import re

    vols = [t for _, t in gmsh.model.getEntities(3)]
    cad_names = cad_names or {}
    out = {}
    for name, spec in regions.items():
        hit = []
        for v in vols:
            if callable(spec):
                ok = bool(spec(gmsh, v))
            elif isinstance(spec, str):
                label = cad_names.get(v) or gmsh.model.getEntityName(3, v)
                ok = re.search(spec, label, flags=re.IGNORECASE) is not None
            else:
                pts = np.atleast_2d(np.asarray(spec, float))
                ok = any(gmsh.model.isInside(3, v, list(p)) for p in pts)
            if ok:
                hit.append(v)
        if not hit:
            raise ValueError(f"region {name!r} matched no CAD volume")
        for v in hit:
            if v in out and out[v] != name:
                raise ValueError(
                    f"volume {v} matched regions {out[v]!r} and {name!r}"
                )
            out[v] = name
    return out


def mesh_cad(
    source,
    regions,
    materials,
    f_max,
    cpw=10,
    scale=1.0,
    size_mode="per_material",
    curvature_points=0,
    algorithm=None,
    optimize="gmsh",
    improve=None,
    filename=None,
    max_tets=2e6,
    sliver_tol=1e-6,
    verbose=False,
):
    """
    Tetrahedral mesh of a CAD model with the optional external tool gmsh.

    All volumes are fragmented (boolean "fragment") so that the interfaces
    between regions are conforming, each volume is assigned to a region,
    volumes assigned to no region are removed (their surfaces become the
    outer boundary: PEC or PMC for the solver), and every region is meshed
    with the cells-per-wavelength size of its material.

    Parameters
    ----------
    source : str or callable
        CAD file (STEP/BREP/IGES, read with the OpenCASCADE kernel) or
        ``build(gmsh)`` that creates OCC volumes (no need to synchronize).
    regions : dict
        ``{name: spec}`` with spec a point (or list of points) inside the
        volume(s) in CAD units, a regular expression searched in the CAD
        volume label (STEP product/material names, case-insensitive; e.g.
        ``r"Vacuum$"``) or ``f(gmsh, volume_tag) -> bool``.
    materials : dict
        ``{name: material}`` for every region (``[eps_r, mu_r]``, ...,
        ``"pec"``).
    f_max : float
        Highest frequency of interest [Hz].
    cpw : float
        Cells per wavelength in the highest-index material.
    scale : float
        Metres per CAD unit (1e-3 for millimetres).
    size_mode : {"per_material", "uniform"}
        See `region_sizes`.
    curvature_points : int
        If > 0, also refine curved surfaces to this many elements per 2 pi
        (gmsh ``Mesh.MeshSizeFromCurvature``).
    algorithm : int, optional
        gmsh ``Mesh.Algorithm3D`` (1 Delaunay, 4 Frontal, 10 HXT).
    optimize : {"gmsh", "netgen", None}
        gmsh's tetrahedral quality optimizer (default), plus Netgen's, or
        none. gmsh's optimizers improve the shape quality but break the
        Delaunay property the diagonal Voronoi Hodge needs (negative dual
        measures); use None with ``improve`` for Voronoi FIT.
    improve : dict or bool, optional
        Run `wakis.unstructured.improve.improve_dt` (time-step-driven node
        relocation) with these keyword arguments before returning; nodes on
        CAD surfaces slide on the exact surfaces, curve and corner nodes stay
        fixed. ``True`` uses the defaults. The history is in
        ``info.improve_history``.
    filename : str, optional
        Also write the gmsh mesh (``.msh``) there.
    max_tets : float or None
        Refuse to mesh (MemoryError) when the estimated number of tetrahedra
        (`estimate_n_tets` per region, before meshing) exceeds this.
    sliver_tol : float
        With ``optimize=None``: if a tetrahedron has |volume| below
        ``sliver_tol`` x the largest, gmsh's optimizer is applied to the
        slivers only (``info.slivers_optimized``); a few faces may then lose
        the Delaunay property, which ``improve`` repairs.
    verbose : bool
        gmsh terminal output.

    Returns
    -------
    mesh : SimplexMesh
        Tags 0, 1, ... in the order of `regions`.
    info : CADMeshInfo
        ``info.materials`` is the ``{tag: material}`` dict for the solver.
    """
    gmsh = _require_gmsh()
    names = list(regions)
    missing = set(names) - set(materials)
    if missing:
        raise ValueError(f"no material for regions {sorted(missing)}")
    sizes = region_sizes({k: materials[k] for k in names}, f_max, cpw, size_mode)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.model.add("wakis_cad")
        occ = gmsh.model.occ
        if callable(source):
            source(gmsh)
        else:
            occ.importShapes(str(source))
        occ.synchronize()
        vols = gmsh.model.getEntities(3)
        names_in = {t: gmsh.model.getEntityName(3, t) for _, t in vols}
        cad_names = dict(names_in)
        if len(vols) > 1:
            _, out_map = occ.fragment(vols[:1], vols[1:])
            occ.synchronize()
            # carry the CAD labels through the boolean fragment
            cad_names = {}
            for (_, t_in), outs in zip(vols, out_map):
                for d, t in outs:
                    if d == 3:
                        cad_names.setdefault(t, names_in[t_in])
        vol_region = _match_regions(gmsh, regions, cad_names)
        drop = [(3, t) for _, t in gmsh.model.getEntities(3) if t not in vol_region]
        if drop:
            occ.remove(drop, recursive=True)
            occ.synchronize()
        h_cad = {k: v / scale for k, v in sizes.items()}
        n_est = sum(
            estimate_n_tets(occ.getMass(3, v), h_cad[n]) for v, n in vol_region.items()
        )
        _preflight(n_est, max_tets)
        _set_size_fields(
            gmsh,
            [([v for v, n in vol_region.items() if n == k], h_cad[k]) for k in names],
            curvature_points,
        )
        _configure(gmsh, algorithm, optimize)
        for k in names:
            gmsh.model.addPhysicalGroup(
                3, [v for v, n in vol_region.items() if n == k], names.index(k) + 1, k
            )
        gmsh.model.mesh.generate(3)
        slivers = _remove_slivers(gmsh, optimize, sliver_tol)
        if filename is not None:
            gmsh.write(str(filename))
        mesh, ntags = _collect(gmsh, vol_region, names, scale, return_tags=True)
        history = None
        if improve:
            from .improve import improve_dt

            kw = dict(improve) if isinstance(improve, dict) else {}
            kw.setdefault("project", cad_projector(gmsh, ntags, scale))
            mats = {i: materials[k] for i, k in enumerate(names)}
            mesh, history = improve_dt(mesh, mats, **kw)
    finally:
        gmsh.finalize()
    info = CADMeshInfo(
        improve_history=history,
        slivers_optimized=slivers,
        region_tags={k: i for i, k in enumerate(names)},
        materials={i: materials[k] for i, k in enumerate(names)},
        sizes=sizes,
        volumes={k: [v for v, n in vol_region.items() if n == k] for k in names},
        scale=scale,
        f_max=f_max,
        cpw=cpw,
        n_tets=mesh.n_cells,
    )
    mesh.region_names = dict(info.region_tags)
    return mesh, info


def _geo_box(geo, lo, hi):
    """Surface loop of an axis-aligned box with the built-in geo kernel."""
    x = [lo[0], hi[0]]
    y = [lo[1], hi[1]]
    z = [lo[2], hi[2]]
    p = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p[i, j, k] = geo.addPoint(x[i], y[j], z[k])

    lines = {}

    def line(a, b):  # shared (oriented) lines so that the box is watertight
        if (b, a) in lines:
            return -lines[b, a]
        if (a, b) not in lines:
            lines[a, b] = geo.addLine(p[a], p[b])
        return lines[a, b]

    def face(c):  # c: 4 corners in cyclic order
        ls = [line(c[i], c[(i + 1) % 4]) for i in range(4)]
        return geo.addPlaneSurface([geo.addCurveLoop(ls)])

    faces = [
        face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
        face([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        face([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        face([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
        face([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
        face([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ]
    return geo.addSurfaceLoop(faces)


def mesh_stl(
    stl_file,
    material,
    f_max,
    cpw=10,
    scale=1.0,
    padding=None,
    background=(1.0, 1.0),
    size_mode="per_material",
    feature_angle=40.0,
    optimize="gmsh",
    max_tets=2e6,
    verbose=False,
):
    """
    Tetrahedral mesh of a closed STL body (optionally inside a padding box).

    The STL triangles are classified into smooth patches (dihedral angle
    below `feature_angle` degrees), turned into discrete geometry and closed
    into a volume with gmsh's built-in kernel. With `padding`, an axis-aligned
    box ``padding`` CAD units larger than the STL bounding box on every side
    is filled with the `background` material (tag 1); the body is tag 0.
    The body surface is remeshed at the target size.

    Parameters
    ----------
    stl_file : str
        Closed (watertight) STL surface.
    material : material spec
        Material of the body (``[eps_r, mu_r]`` or ``"pec"``).
    f_max, cpw, scale, size_mode, optimize, max_tets, verbose
        As in `mesh_cad` (the preflight uses the bounding-box volume).
    padding : float, optional
        Box margin [CAD units]; None meshes the body alone.
    background : material spec
        Material of the padding box.
    feature_angle : float
        Sharp-edge angle [deg] for the patch classification.

    Returns
    -------
    mesh : SimplexMesh
    info : CADMeshInfo
    """
    gmsh = _require_gmsh()
    mats = {"body": material}
    if padding is not None:
        mats["background"] = background
    sizes = region_sizes(mats, f_max, cpw, size_mode)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.model.add("wakis_stl")
        gmsh.merge(str(stl_file))
        gmsh.model.mesh.classifySurfaces(
            np.deg2rad(feature_angle), True, True, np.pi
        )
        gmsh.model.mesh.createGeometry()
        geo = gmsh.model.geo
        surf = [t for _, t in gmsh.model.getEntities(2)]
        body_loop = geo.addSurfaceLoop(surf)
        vol_region = {geo.addVolume([body_loop]): "body"}
        if padding is not None:
            geo.synchronize()
            bb = np.asarray(gmsh.model.getBoundingBox(-1, -1))
            box = _geo_box(geo, bb[:3] - padding, bb[3:] + padding)
            vol_region[geo.addVolume([box, body_loop])] = "background"
        geo.synchronize()
        names = list(mats)
        bb = np.asarray(gmsh.model.getBoundingBox(-1, -1))
        _preflight(
            estimate_n_tets(np.prod(bb[3:] - bb[:3]), min(sizes.values()) / scale),
            max_tets,
        )
        _set_size_fields(
            gmsh,
            [([v for v, n in vol_region.items() if n == k], sizes[k] / scale) for k in names],
        )
        _configure(gmsh, None, optimize)
        gmsh.model.mesh.generate(3)
        mesh = _collect(gmsh, vol_region, names, scale)
    finally:
        gmsh.finalize()
    info = CADMeshInfo(
        region_tags={k: i for i, k in enumerate(names)},
        materials={i: mats[k] for i, k in enumerate(names)},
        sizes=sizes,
        volumes={k: [v for v, n in vol_region.items() if n == k] for k in names},
        scale=scale,
        f_max=f_max,
        cpw=cpw,
        n_tets=mesh.n_cells,
    )
    mesh.region_names = dict(info.region_tags)
    return mesh, info


def read_mesh(filename, materials=None, scale=1.0, tag_key=None):
    """
    Read a tetrahedral mesh file and map named physical groups to materials.

    Works with any format meshio reads (``.msh`` 2.2/4.1, ``.vtu``,
    ``.med``, ...). Gmsh physical names (``field_data``) become region names.

    Parameters
    ----------
    filename : str
    materials : dict, optional
        ``{name or tag: material}``; names are resolved through the physical
        names of the file.
    scale : float
        Metres per file unit.
    tag_key : str, optional
        Cell-data key holding the region tags (see `SimplexMesh.from_meshio`).

    Returns
    -------
    mesh : SimplexMesh
        Cell tags = the file's physical tags; ``mesh.region_names`` maps
        names to tags and ``mesh.surface_groups`` maps 2D physical names (or
        tags) to indices into ``mesh.faces``.
    solver_materials : dict or None
        ``{tag: material}`` for `UnstructuredFIT3D`.
    """
    import meshio

    raw = meshio.read(filename)
    mesh = SimplexMesh.from_meshio(raw, tag_key=tag_key)
    if scale != 1.0:
        mesh = SimplexMesh(mesh.nodes * scale, mesh.cells, mesh.cell_tags)
    names3, names2 = {}, {}
    for name, (tag, dim) in getattr(raw, "field_data", {}).items():
        (names3 if int(dim) == 3 else names2 if int(dim) == 2 else {})[name] = int(tag)
    mesh.region_names = names3
    # Surface groups: physical triangles -> primal face indices
    groups = {}
    key = tag_key or "gmsh:physical"
    if "triangle" in raw.cells_dict and key in raw.cell_data and mesh.dim == 3:
        tri_blocks = [i for i, c in enumerate(raw.cells) if c.type == "triangle"]
        tri = np.concatenate([raw.cells[i].data for i in tri_blocks])
        ttag = np.concatenate([raw.cell_data[key][i] for i in tri_blocks])
        # meshio keeps all points; SimplexMesh may have dropped unused ones
        used = np.zeros(len(raw.points), bool)
        for i, c in enumerate(raw.cells):
            if c.type == "tetra":
                used[c.data.ravel()] = True
        remap = np.cumsum(used) - 1
        ok = used[tri].all(1)
        tri, ttag = np.sort(remap[tri[ok]], 1), ttag[ok]
        fkey = {tuple(f): i for i, f in enumerate(mesh.faces)}
        inv = {v: k for k, v in names2.items()}
        for t in np.unique(ttag):
            ids = [fkey.get(tuple(f), -1) for f in tri[ttag == t]]
            groups[inv.get(int(t), int(t))] = np.array([i for i in ids if i >= 0])
    mesh.surface_groups = groups
    solver_materials = None
    if materials is not None:
        solver_materials = {}
        for k, m in materials.items():
            if isinstance(k, str):
                if k not in names3:
                    raise KeyError(f"no physical volume named {k!r} in {filename}")
                k = names3[k]
            solver_materials[int(k)] = m
    return mesh, solver_materials
