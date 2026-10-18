# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time

import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .field import Field
from .logger import Logger
from .materials import material_colors, material_lib
from .plotting import PlotMixinGrid as PlotMixin

try:
    from mpi4py import MPI

    imported_mpi = True
except ImportError:
    imported_mpi = False


def _nearest(positions, x):
    """Index of the sample nearest to x; exact ties (up to rounding) go to the
    lower index, so serial and MPI runs select the same sample."""
    distance = np.abs(positions - x)
    tol = 1e-9 * (positions.max() - positions.min())
    return np.flatnonzero(distance <= distance.min() + tol)[0]


class GridFIT3D(PlotMixin):
    """
    Class holding the grid information and
    stl importing handling using PyVista
    """

    def __init__(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        Nx=None,
        Ny=None,
        Nz=None,
        x=None,
        y=None,
        z=None,
        use_mpi=False,
        use_mesh_refinement=False,
        refinement_method="insert",
        refinement_tol=1e-8,
        snap_points=None,
        snap_tol=1e-5,
        snap_solids=None,
        stl_solids=None,
        stl_materials=None,
        stl_rotate=[0.0, 0.0, 0.0],
        stl_translate=[0.0, 0.0, 0.0],
        stl_scale=1.0,
        stl_colors=None,
        stl_tol=1e-3,
        stl_method="legacy",
        subpixel_smoothing=False,
        subpixel_smoothing_factor=4,
        subpixel_smoothing_threshold=0.3,
        subpixel_smoothing_bool=True,
        load_from_h5=None,
        boundary_at_domain_edge=True,
        conformal=False,
        conformal_subsamples=4,
        sheets=None,
        sheet_edge_correction=True,
        verbose=2,
    ):
        """
        Class holding the grid information and STL importing/handling using PyVista.

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax : float, optional
            Extent of the simulation domain. If None, must provide x, y, z arrays.
        Nx, Ny, Nz : int, optional
            Number of cells per direction. If None, must provide x, y, z arrays.
        x, y, z : array_like, optional
            Custom grid axis arrays to be used in the meshgrid generation.
            Non-uniform grids are supported.
        use_mpi : bool, optional
            Enable MPI domain decomposition in the z direction. Default is False.
        use_mesh_refinement : bool, optional
            Enable mesh refinement based on snap points extracted from the STL solids.
            Default is False.
        refinement_method : str, optional
            Refinement algorithm for mesh refinement. Default is "insert".
        refinement_tol : float, optional
            Tolerance for mesh refinement. Default is 1e-8.
        snap_points : array_like, optional
            Points to snap the mesh to. Default is None.
        snap_tol : float, optional
            Tolerance for snap point detection. Default is 1e-5.
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is None (all).
        stl_solids : dict or str, optional
            STL files to import in the domain. {'Solid 1': stl_1, ...}
        stl_materials : dict, optional
            Material properties associated with STL solids.
            {'Solid 1': [eps1, mu1], ...}
        stl_rotate : list or dict, optional
            Angle of rotation to apply to the STL models: [rot_x, rot_y, rot_z].
            If dict, must contain the same keys as stl_solids.
        stl_translate : list or dict, optional
            Translation to apply to the STL models: [dx, dy, dz].
            If dict, must contain the same keys as stl_solids.
        stl_scale : float or dict, optional
            Scaling value to apply to the STL model to convert to [m].
            If dict, must contain the same keys as stl_solids.
        stl_colors : str, list, dict, or None, optional
            Color(s) for STL solids. If None, assigned automatically.
        stl_tol : float, optional
            Tolerance factor for STL import, used in grid.select_enclosed_points.
            Default is 1e-3.
        stl_method : str, optional
            Method for marking cells inside STL solids. Options are "legacy" (default),
            "interior_points", "implicit_distance", or "voxelize_rectilinear".
        subpixel_smoothing : bool, optional
            Whether to apply subpixel smoothing to the STL masks after voxelization. Default is False.
        subpixel_smoothing_factor : int, optional
            Factor by which to increase the resolution for subpixel smoothing. Default is 4.
            Memory usage increases with the cube of this factor, so use with caution!
        subpixel_smoothing_threshold : float, optional
            Threshold value (0–1) for classifying smoothed cells as inside/outside the solid.
            Default is 0.3. A lower value includes more boundary cells; typical choice is
            ``1/(subpixel_smoothing_factor**3)``.
        subpixel_smoothing_bool : bool, optional
            If True, convert the smoothed mask to a boolean array after thresholding.
            Default is True.
        load_from_h5 : str, optional
            Load grid from an h5 file previously saved with `save_to_h5`.
        boundary_at_domain_edge : bool, optional
            If True (default), a domain [min, max] meshed with N cells has its
            non-periodic boundaries (PEC, PMC, ABC, CPML) on the planes min and
            max: one ghost cell is appended beyond max along every axis, so
            the field arrays hold N + 1 samples (``Nx = N + 1``) and the last
            index is the boundary node at max. `SolverFIT3D` removes the ghost
            cell again on periodic/Bloch axes. Dual lengths are centred on the
            primary nodes. If False, the legacy convention is kept: N samples
            per axis, the last index is a ghost layer and the high-side
            boundaries sit on the node max - d_last (one cell inside the
            domain), with dual lengths shifted by one node on non-uniform
            meshes. Use False only to reproduce results obtained with earlier
            Wakis versions.
            See "Grid conventions" in the physics guide.
        conformal : bool or {'pec', 'all', 'tensor'}, optional
            Conformal FIT instead of staircased STL solids (applied by
            `SolverFIT3D`). 'pec': PEC solids (and a PEC background around
            the other solids) cut the primary edges and faces, which enter
            the update with their length/area outside the conductor;
            dielectric edges take the material occupying most of their dual
            cell. 'all' (or True): additionally anisotropic subpixel
            averaging of the permittivity of dielectric solids. 'tensor':
            as 'all', plus the off-diagonal couplings of the full averaged
            inverse-permittivity tensor (more accurate when E crosses curved
            or oblique dielectric interfaces; see "Tensor subpixel averaging"
            in the physics guide). Default False (staircase cell masks). See
            "Conformal FIT" in the physics guide.
        conformal_subsamples : int, optional
            Midpoint samples per dual cell along one axis for the dielectric
            volume fractions (the two other axes are integrated exactly).
            Default 4.
        sheets : dict, optional
            Zero-thickness PEC sheets {name: pyvista.PolyData or STL path},
            in metres, planar and normal to x, y or z, lying on a grid
            plane (see `wakis.conformal.rectangle_sheet`). An STL solid with
            material 'sheet' is moved here. Sheets always use the conformal
            edge/face coverage. Default None.
        sheet_edge_correction : bool, optional
            Correct the material matrices near axis-aligned sheet edges with
            the static field of a knife edge, so that the edge is located at
            its physical position for any position relative to the nodes
            (see "Thin PEC sheets" in the physics guide). Default True.
        verbose : int or bool, optional
            Enable verbose output on the terminal. Use `verbose=2` for more detail.

        Attributes
        ----------
        x, y, z : ndarray
            Grid axis arrays.
        Nx, Ny, Nz : int
            Number of cells in each direction.
        dx, dy, dz : ndarray
            Cell sizes in each direction.
        grid : pyvista.StructuredGrid
            PyVista grid object.
        stl_solids, stl_materials, stl_colors : dict
            STL solid file paths, materials, and colors.
        (...)
        """

        t0 = time.time()
        self.logger = Logger()
        self.verbose = verbose
        self.use_mpi = use_mpi
        self.boundary_at_domain_edge = bool(boundary_at_domain_edge)
        # 1 along an axis whose last sample index is a ghost cell beyond max
        self.edge_ghosts = [0, 0, 0]

        # Grid data
        # generate from file
        if load_from_h5 is not None:
            self.load_from_h5(load_from_h5)
            return  # TODO: support MPI decomposition

        # generate from custom x,y,z arrays
        elif x is not None and y is not None and z is not None:
            # allow user to set the grid axis manually
            self.x = x
            self.y = y
            self.z = z
            self.Nx = len(self.x) - 1
            self.Ny = len(self.y) - 1
            self.Nz = len(self.z) - 1
            self.xmin, self.xmax = self.x[0], self.x[-1]
            self.ymin, self.ymax = self.y[0], self.y[-1]
            self.zmin, self.zmax = self.z[0], self.z[-1]
            if self.use_mpi:
                raise ValueError(
                    "[!] Error: use_mpi=True is not compatible with custom x,y,z arrays."
                )

        # generate from domain extents and number of cells [LEGACY]
        elif all(v is not None for v in [xmin, xmax, ymin, ymax, zmin, zmax]):
            # uniform grid from domain extents and number of cells
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            self.zmin, self.zmax = zmin, zmax
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.x = np.linspace(self.xmin, self.xmax, self.Nx + 1)
            self.y = np.linspace(self.ymin, self.ymax, self.Ny + 1)
            self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)

        else:
            raise ValueError(
                "[!] Error initializing GridFIT3D:\n"
                "  - Provide grid axis arrays: x, y, z\n"
                "  - OR domain extents and number of cells: \
                    xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz\n"
                "  - OR load from a HDF5 file using load_from_h5"
            )

        # TODO: allow non uniform dx, dy, dz
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale
        self.stl_colors = stl_colors
        self.update_logger(["stl_solids", "stl_materials"])
        if stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if stl_scale != 1.0:
            self.update_logger(["stl_scale"])

        self.conformal = "all" if conformal is True else conformal
        if self.conformal not in (False, "pec", "all", "tensor"):
            raise ValueError("conformal must be False, True, 'pec', 'all' or 'tensor'.")
        self.conformal_subsamples = int(conformal_subsamples)
        self.sheet_edge_correction = bool(sheet_edge_correction)
        self.sheets = {
            k: pv.read(v) if isinstance(v, str) else v
            for k, v in (sheets or {}).items()
        }
        if stl_solids is not None:
            self._prepare_stl_dicts()
            self._move_sheets_from_stl()

        # refine self.x, self.y, self.z using snap points
        self.use_mesh_refinement = use_mesh_refinement
        self.refinement_method = refinement_method
        self.snap_points = snap_points
        self.snap_tol = snap_tol
        self.snap_solids = snap_solids  # if None, use all stl_solids
        self.update_logger(["use_mesh_refinement"])

        if self.use_mesh_refinement:
            if verbose:
                print("Applying mesh refinement...")
            if self.snap_points is None and stl_solids is not None:
                self._compute_snap_points(snap_solids=snap_solids, snap_tol=snap_tol)
            self._refine_xyz_axis(method=refinement_method, tol=refinement_tol)

        if verbose:
            print(f"Generating grid with {self.Nx * self.Ny * self.Nz} mesh cells...")
        if verbose > 1:
            print(
                f"    * Simulation domain bounds: \n\
                x:[{xmin:.3f}, {xmax:.3f}],\n\
                y:[{ymin:.3f}, {ymax:.3f}],\n\
                z:[{zmin:.3f}, {zmax:.3f}]"
            )
            print(
                "    * Minimum cell sizes: dx={:.3e}, dy={:.3e}, dz={:.3e}".format(
                    np.min(self.dx), np.min(self.dy), np.min(self.dz)
                )
            )

        # MPI subdivide domain
        if self.use_mpi:
            self.ZMIN = None
            self.ZMAX = None
            self.NZ = None
            self.Z = None
            if imported_mpi:
                self._mpi_initialize()
                if self.verbose:
                    print(f"MPI initialized for {self.rank} of {self.size}")
            else:
                raise ImportError(
                    "[!] mpi4py is required when use_mpi=True but was not found"
                )

        # Ghost cells beyond max: the boundary node max gets its own samples
        if self.boundary_at_domain_edge:
            self._add_edge_ghosts(axes="xy" if self.use_mpi else "xyz")

        # grid G and tilde grid ~G, lengths and inverse areas
        self._compute_grid()

        # STL import and mask generation
        if verbose:
            print("Importing STL solids...")
        self.stl_tol = stl_tol
        self.stl_method = stl_method
        self.use_subpixel_smoothing = subpixel_smoothing
        self.subpixel_smoothing_factor = subpixel_smoothing_factor
        self.subpixel_smoothing_threshold = subpixel_smoothing_threshold
        self.subpixel_smoothing_bool = subpixel_smoothing_bool
        if self.subpixel_smoothing_threshold is None:
            self.subpixel_smoothing_threshold = 1 / (self.subpixel_smoothing_factor**3)
        if stl_solids is not None:
            self._mark_cells_in_stl(method=self.stl_method)

        if verbose:
            print(f"Total grid initialization time: {time.time() - t0} s")

        self.gridInitializationTime = time.time() - t0
        self.update_logger(["gridInitializationTime"])

    def _compute_grid(self):
        """
        Compute the PyVista grid and related geometric quantities.

        Sets up the structured grid, cell lengths, inverse areas, and tilde grid.
        """
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
        self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
        self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

        if self.boundary_at_domain_edge:
            self._compute_dual_lengths()
            return

        # Legacy tilde grid ~G: tL[i] = tx[i+1] - tx[i] is centred on node
        # i+1 (not on node i, where E_y, E_z and H_x sit) and is zero at the
        # last index, which freezes the tangential E there
        self.tx = (self.x[1:] + self.x[:-1]) / 2
        self.ty = (self.y[1:] + self.y[:-1]) / 2
        self.tz = (self.z[1:] + self.z[:-1]) / 2

        self.tx = np.append(self.tx, self.tx[-1])
        self.ty = np.append(self.ty, self.ty[-1])
        self.tz = np.append(self.tz, self.tz[-1])

        tX, tY, tZ = np.meshgrid(self.tx, self.ty, self.tz, indexing="ij")

        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        aux = self.tL.field_y * self.tL.field_z
        self.itA.field_x = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.tL.field_x * self.tL.field_z
        self.itA.field_y = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.tL.field_x * self.tL.field_y
        self.itA.field_z = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        del aux

    def _add_edge_ghosts(self, axes="xyz"):
        """
        Append one ghost cell beyond the max end of each axis in `axes`.

        The appended cell copies the size of the last cell. The new last node
        index is then the boundary node at max, and the samples at the edge
        midpoints of the ghost cell are never coupled to the domain.
        """
        for a in axes:
            nodes = getattr(self, a)
            nodes = np.append(nodes, 2 * nodes[-1] - nodes[-2])
            setattr(self, a, nodes)
            setattr(self, "N" + a, len(nodes) - 1)
            setattr(self, "d" + a, np.diff(nodes))
            self.edge_ghosts["xyz".index(a)] = 1

    def remove_edge_ghost(self, axis):
        """
        Remove the domain-edge ghost cell along a periodic axis.

        Called by `SolverFIT3D` for periodic and Bloch axes, whose period is
        exactly the domain length, so that their field arrays keep one sample
        per cell. The axis arrays, metrics and the STL cell masks are trimmed
        in place; the grid can then only be used with that axis periodic.

        Parameters
        ----------
        axis : int
            0, 1 or 2 for x, y or z.
        """
        if not self.edge_ghosts[axis]:
            return
        shape = (self.Nx, self.Ny, self.Nz)
        cells = {}
        for key in self.grid.cell_data.keys():
            # Cell data are ordered with z fastest (C order over x, y, z)
            data = np.reshape(np.asarray(self.grid.cell_data[key]), shape)
            cells[key] = np.take(data, range(shape[axis] - 1), axis=axis).ravel()
        a = "xyz"[axis]
        nodes = getattr(self, a)[:-1]
        setattr(self, a, nodes)
        setattr(self, "N" + a, len(nodes) - 1)
        setattr(self, "d" + a, np.diff(nodes))
        self.edge_ghosts[axis] = 0
        self._compute_grid()
        for key, data in cells.items():
            self.grid.cell_data[key] = data

    def _compute_dual_lengths(self):
        """
        Dual lengths centred on the primary nodes and the inverse dual areas.

        The dual edge through node i spans the midpoints of the primary cells
        i-1 and i: tL[i] = (L[i-1] + L[i]) / 2. The first node and the
        boundary node at max (before the ghost cell) get half a cell, L[0] / 2
        and L[N-1] / 2: the dual cell of a boundary node ends on the boundary.
        Periodic axes overwrite the first node with the length across the
        seam (see `_apply_bc_to_C`).
        """
        tl = []
        for nodes, ghost in zip((self.x, self.y, self.z), self.edge_ghosts):
            L = np.diff(nodes)
            prev = np.concatenate([[0.0], L[:-1]])
            this = L.copy()
            if ghost:  # the ghost cell lies outside the domain
                this[-1] = 0.0
            tl.append(0.5 * (prev + this))
        shape = (self.Nx, self.Ny, self.Nz)
        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = np.broadcast_to(tl[0][:, None, None], shape)
        self.tL.field_y = np.broadcast_to(tl[1][None, :, None], shape)
        self.tL.field_z = np.broadcast_to(tl[2][None, None, :], shape)

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        self.itA.field_x = 1.0 / (self.tL.field_y * self.tL.field_z)
        self.itA.field_y = 1.0 / (self.tL.field_x * self.tL.field_z)
        self.itA.field_z = 1.0 / (self.tL.field_x * self.tL.field_y)

    def _mpi_initialize(self):
        """
        Initialize MPI domain decomposition in the z direction.

        Sets up MPI communicator, rank, size, and subdomain grid quantities.
        """
        comm = MPI.COMM_WORLD  # Get MPI communicator
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Error handling for Nz < size
        if self.Nz < self.size:
            raise ValueError(
                f"Nz ({self.Nz}) must be greater than or equal to the number of \
                MPI processes ({self.size})."
            )

        # global z quantities [ALLCAPS]
        self.ZMIN = self.zmin
        self.ZMAX = self.zmax
        self.NZ = self.Nz - self.Nz % (self.size)  # ensure multiple of MPI size
        self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NZ + 1)[:-1]
        self.Z += (self.ZMAX - self.ZMIN) / (2 * self.NZ)
        # Global number of z samples, as in a serial run: with domain-edge
        # ghosts the boundary node at ZMAX adds one (its edge midpoint lies
        # beyond ZMAX)
        self.NZ_samples = self.NZ + int(self.boundary_at_domain_edge)
        if self.boundary_at_domain_edge:
            self.Z = np.append(self.Z, self.Z[-1] + (self.ZMAX - self.ZMIN) / self.NZ)

        if self.verbose and self.rank == 0:
            print(f" * Global grid ZMIN={self.ZMIN}, ZMAX={self.ZMAX}, NZ={self.NZ}")

        # MPI subdomain quantities [TODO: support non-uniform dz with MPI]
        self.Nz = self.NZ // (self.size)
        self.dz = (self.ZMAX - self.ZMIN) / self.NZ
        self.zmin = self.rank * self.Nz * self.dz + self.ZMIN
        self.zmax = (self.rank + 1) * self.Nz * self.dz + self.ZMIN

        if self.verbose:
            print(
                f"MPI rank {self.rank} of {self.size} initialized with \
                        zmin={self.zmin}, zmax={self.zmax}, Nz={self.Nz}"
            )
        # Add ghost cells
        self.n_ghosts = 1
        if self.rank > 0:
            self.zmin += -self.n_ghosts * self.dz
            self.Nz += self.n_ghosts
        if self.rank < (self.size - 1):
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        # Domain-edge ghost cell beyond ZMAX on the last rank (legacy: only
        # for a single rank, so the wall position depended on the rank count)
        if self.boundary_at_domain_edge and self.rank == self.size - 1:
            self.zmax += self.dz
            self.Nz += 1
            self.edge_ghosts[2] = 1
        elif not self.boundary_at_domain_edge and self.rank == 0 and self.size == 1:
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)
        self.dz = np.diff(self.z)  # only uniform grid possible with MPI

    def mpi_gather_asGrid(self):
        """
        Gather the global grid from all MPI subdomains and return a new GridFIT3D.

        Returns
        -------
        _grid : GridFIT3D or None
            The global grid object (only on rank 0).
        """
        _grid = None
        if self.rank == 0:
            print(f"Generating global grid from {self.ZMIN} to {self.ZMAX}")
            _grid = GridFIT3D(
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                self.ZMIN,
                self.ZMAX,
                self.Nx,
                self.Ny,
                self.NZ,
                use_mpi=False,
                stl_solids=self.stl_solids,
                stl_materials=self.stl_materials,
                stl_scale=self.stl_scale,
                stl_rotate=self.stl_rotate,
                stl_translate=self.stl_translate,
                stl_colors=self.stl_colors,
                verbose=self.verbose,
                stl_tol=self.stl_tol,
                boundary_at_domain_edge=self.boundary_at_domain_edge,
            )
        return _grid

    def _prepare_stl_dicts(self):
        """
        Prepare STL-related dictionaries for rotation, scale, translation, and colors.

        Ensures all STL solids have corresponding entries for rotation, scale,
        translation, and color, converting single values to dicts as needed.
        """
        if type(self.stl_solids) is not dict:
            if type(self.stl_solids) is str:
                self.stl_solids = {"Solid 1": self.stl_solids}
            else:
                raise Exception(
                    "Attribute `stl_solids` must contain a string or a dictionary"
                )

        if type(self.stl_rotate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_rotate = {}
            for key in self.stl_solids.keys():
                stl_rotate[key] = self.stl_rotate
            self.stl_rotate = stl_rotate

        if type(self.stl_scale) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_scale = {}
            for key in self.stl_solids.keys():
                stl_scale[key] = self.stl_scale
            self.stl_scale = stl_scale

        if type(self.stl_translate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_translate = {}
            for key in self.stl_solids.keys():
                stl_translate[key] = self.stl_translate
            self.stl_translate = stl_translate

        if type(self.stl_colors) is not dict:
            if self.stl_colors is None:
                self._assign_colors()
            elif self.stl_colors is str:  # single color for all solids
                stl_colors = {}
                for key in self.stl_solids.keys():
                    stl_colors[key] = self.stl_colors
                self.stl_colors = stl_colors
            elif type(self.stl_colors) is list:
                stl_colors = {}
                try:
                    for i, key in enumerate(self.stl_solids.keys()):
                        stl_colors[key] = self.stl_colors[i]
                    self.stl_colors = stl_colors
                except IndexError:
                    self._assign_colors()
                    print(
                        "[!] If `stl_colors` is a list, it must have \
                        the same length as `stl_solids`."
                    )

        if type(self.stl_materials) is not dict:
            if type(self.stl_materials) is str:
                self.stl_materials = {"Solid 1": self.stl_materials}
            else:
                raise Exception(
                    "Attribute `stl_materials` must contain a string or a dictionary"
                )
        # Material library lookup and conversion to [eps, mu, sigma] format
        for key in self.stl_solids.keys():
            if str(self.stl_materials[key]).lower() == "sheet":
                self.stl_materials[key] = "sheet"
            elif type(self.stl_materials[key]) is str:
                mat_key = self.stl_materials[key].lower()
                mat = material_lib[mat_key]
                self.stl_materials[key] = [
                    mat[0],
                    mat[1],
                    mat[2] if len(mat) == 3 else 0.0,
                ]
            elif len(self.stl_materials[key]) == 2:
                self.stl_materials[key].append(0.0)

    def _move_sheets_from_stl(self):
        """Treat STL solids with material 'sheet' as zero-thickness PEC sheets."""
        for key in [k for k, m in self.stl_materials.items() if isinstance(m, str)]:
            self.sheets[key] = self.read_stl(key)
            del self.stl_solids[key], self.stl_materials[key]

    def _mark_cells_in_stl(self, method):
        """
        Mark grid cells that are inside each STL solid.

        Uses PyVista's select_interior_points as default (equivalent to the deprecated select_enclosed_points),
        and otherwise input method either compute_implicit_distance or voxelize_rectilinear,
        to create boolean masks for each solid.

        See also the PyVista documentation:
        ----------------------------------
        "interior_points": https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.select_interior_points
        "implicit_distance": https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.compute_implicit_distance
        "voxelize_rectilinear" : https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.voxelize_rectilinear

        """
        # Obtain masks with grid cells inside each stl solid
        stl_tolerance = (
            np.min([np.min(self.dx), np.min(self.dy), np.min(self.dz)]) * self.stl_tol
        )
        progress_bar = False
        if self.verbose > 1:
            progress_bar = True

        for key in self.stl_solids.keys():
            surf = self.read_stl(key)

            if self.verbose:
                print(f" * Marking cells inside STL solid '{key}'...")

            # mark cells in stl [True == in stl, False == out stl]
            if method.lower() == "legacy":
                try:
                    select = self.grid.select_interior_points(
                        surf, method="cell_locator", locator_tolerance=stl_tolerance
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"]
                        > stl_tolerance
                    )
                except Exception:
                    select = self.grid.select_interior_points(
                        surf,
                        method="cell_locator",
                        locator_tolerance=stl_tolerance,
                        check_surface=False,
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"]
                        > stl_tolerance
                    )
                    if self.verbose > 1:
                        print(
                            f"[!] Warning: stl solid {key} may have issues with closed surfaces. Consider checking the STL file."
                        )

            elif method.lower() == "interior_points":
                try:
                    select = self.grid.select_interior_points(
                        surf, method="cell_locator", locator_tolerance=stl_tolerance
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"] > 0.5
                    )
                except Exception:
                    select = self.grid.select_interior_points(
                        surf,
                        method="cell_locator",
                        locator_tolerance=stl_tolerance,
                        check_surface=False,
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"] > 0.5
                    )
                    if self.verbose > 1:
                        print(
                            f"[!] Warning: stl solid {key} may have issues with closed surfaces. Consider checking the STL file."
                        )

            elif method.lower() == "implicit_distance":
                # negative distance is inside, positive is outside
                try:
                    select = self.grid.compute_implicit_distance(surf)
                    self.grid[key] = select.point_data_to_cell_data()[
                        "implicit_distance"
                    ] <= 0.5 * np.mean(
                        [np.mean(self.dx), np.mean(self.dy), np.mean(self.dz)]
                    )
                except Exception:
                    print(
                        f"[!] Warning: Implicit distance computation for stl solid {key} failed."
                    )

            elif method.lower() == "voxelize_rectilinear":
                # Extent of the node arrays (includes domain-edge ghost cells)
                dx, dy, dz = (
                    (self.x[-1] - self.x[0]) / (self.Nx),
                    (self.y[-1] - self.y[0]) / (self.Ny),
                    (self.z[-1] - self.z[0]) / (self.Nz),
                )
                reference_vol = pv.ImageData(
                    dimensions=(self.Nx, self.Ny, self.Nz),
                    origin=(self.xmin, self.ymin, self.zmin),
                    spacing=(dx, dy, dz),
                )

                try:
                    vox = surf.voxelize_rectilinear(
                        reference_volume=reference_vol, progress_bar=progress_bar
                    )
                    mask = np.reshape(
                        vox["mask"], (self.Nx, self.Ny, self.Nz), order="F"
                    ).astype(bool)

                    self.grid[key] = np.reshape(mask, (self.Nx * self.Ny * self.Nz))

                    # Apply subpixel smoothing if enabled
                    if self.use_subpixel_smoothing:
                        self._apply_subpixel_smoothing(
                            key,
                            factor=self.subpixel_smoothing_factor,
                            threshold=self.subpixel_smoothing_threshold,
                            make_bool=self.subpixel_smoothing_bool,
                        )

                except Exception:
                    print(
                        f"[!] Warning: voxelization for stl solid {key} failed. Consider checking if the grid is uniform or using a different method."
                    )
            else:
                raise ValueError(
                    f"[!] Error: stl_method {method} not recognized. \
                    Use 'interior_points', 'implicit_distance', or 'voxelize_rectilinear'."
                )

            if self.verbose and np.sum(self.grid[key]) == 0:
                print(
                    f"[!] Warning: no cells were marked inside stl solid {key}. Consider increasing the tolerance factor (currently {self.stl_tol})."
                )

            if self.verbose:
                print(
                    f"    * STL solid {key}: {np.sum(self.grid[key])} cells marked inside the solid."
                )

    def _apply_subpixel_smoothing(
        self,
        key,
        factor=4,
        sigma=0.5,
        make_bool=True,  # default is True
        threshold=None,  # default is 1/(factor**3)
    ):
        """
        Apply subpixel smoothing to the STL mask.

        This method creates a higher resolution reference volume for voxelization,
        applies the voxelization at this higher resolution, and then combines
        the results to create a more accurate mask with the original grid resolution.

        The smoothing is done by averaging the values from the higher resolution grid
        and computing the gradient at mask edges, then applying a Gaussian filter.

        Parameters
        ----------
        key : str
            Key of the STL solid to smooth.
        factor : int, optional
            Factor by which to increase the resolution for subpixel smoothing. Default is 4.
        sigma : float, optional
            Standard deviation for Gaussian filter. Default is 0.5.
        make_bool : bool, optional
            Whether to convert the smoothed mask to boolean. Default is True.
        threshold : float, optional
            Threshold for converting the smoothed mask to boolean. Default is 0.1.
        """

        # Skip for vacuum solids
        if (
            self.stl_materials[key][0] == 1.0
            and self.stl_materials[key][1] == 1.0
            and self.stl_materials[key][2] == 0.0
        ):
            return

        if self.verbose > 1:
            print(f"    * Applying subpixel smoothing with factor {factor}...")

        surface = self.read_stl(key)

        # Create a higher resolution reference volume for voxelization
        Nx, Ny, Nz = self.Nx * factor, self.Ny * factor, self.Nz * factor
        # Extent of the node arrays (includes domain-edge ghost cells)
        dx = (self.x[-1] - self.x[0]) / (Nx)
        dy = (self.y[-1] - self.y[0]) / (Ny)
        dz = (self.z[-1] - self.z[0]) / (Nz)

        reference_vol = pv.ImageData(
            dimensions=(Nx, Ny, Nz),
            origin=(self.xmin, self.ymin, self.zmin),
            spacing=(dx, dy, dz),
        )
        vox = surface.voxelize_rectilinear(
            reference_volume=reference_vol,
        )
        mask_ref = np.reshape(vox["mask"], (Nx, Ny, Nz), order="F")

        # Combine the higher resolution mask back to the original grid resolution
        mask = np.zeros((self.Nx, self.Ny, self.Nz)).astype(float)
        for i in range(factor):
            for j in range(factor):
                for k in range(factor):
                    sub_mask = mask_ref[i::factor, j::factor, k::factor].astype(float)
                    sub_mask += np.sqrt(sum(g**2 for g in np.gradient(sub_mask)))
                    mask += gaussian_filter(np.clip(sub_mask, 0, 1), sigma=sigma)

        # Overwrite the previous binary/boolean mask in the PyVista grid
        mask = np.clip(mask, 0, factor**3) / factor**3  # Normalize to [0, 1]
        mask = np.where(mask < threshold, 0 * mask, mask)  # clip below threshold
        if make_bool:
            mask = mask > threshold

        self.grid[key] = np.reshape(mask, (self.Nx * self.Ny * self.Nz))

    def check_stl_masks_overlap(self):
        """
        Check for overlapping STL masks in the grid
        and print warnings if overlaps are found.

        This method computes the sum of all STL masks and identifies cells
        that are marked as inside more than one solid.
        """
        mask_sum = np.zeros(self.Nx * self.Ny * self.Nz, dtype=int)
        for key in self.stl_solids.keys():
            mask_sum += self.grid[key].astype(int)

        overlap_mask = mask_sum > 1
        num_overlaps = np.sum(overlap_mask)

        if num_overlaps > 0 and self.verbose > 1:
            print(
                f"[!] Warning: {num_overlaps} cells are marked as inside multiple STL solids. \
                Consider checking for overlapping geometries."
            )

    def _get_background_mask(self):
        # Get a mask for the background (cells not inside any STL solid)
        bg_mask = np.ones(self.Nx * self.Ny * self.Nz, dtype=bool)
        for key in self.stl_solids.keys():
            bg_mask &= ~self.grid[key].astype(int).astype(bool)
        return bg_mask

    def _mark_cells_in_surface(self, key):
        # Modify the STL mask to account only for the surface
        # Needed for the SIBC boundary condition when conductivity > 1e3 S/m
        grad = np.array(
            self.grid.compute_derivative(scalars=key, gradient="gradient")["gradient"]
        )

        # Compute normals and transverse cell size dn
        # grad_mag = np.linalg.norm(grad, axis=1)
        # n = grad / (grad_mag[:, None] + 1e-14)
        # dn = np.sqrt((n[mask,0]*self.dx)**2 + (n[mask,1]*self.dy)**2 + (n[mask,2]*self.dz)**2)

        # Get boundary cells via gradient magnitude
        grad = np.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + grad[:, 2] ** 2)
        mask = grad.astype(bool)  # TODO: subpixel smoothing

        return mask

    def read_stl(self, key):
        """
        Read and transform an STL solid by key.

        Parameters
        ----------
        key : str
            Key of the STL solid to read.

        Returns
        -------
        surf : pyvista.PolyData
            The transformed STL surface.
        """
        # import stl (a file path, or a pyvista surface already in memory)
        surf = self.stl_solids[key]
        if not isinstance(surf, pv.PolyData):
            surf = pv.read(surf)

        # rotate
        surf = surf.rotate_x(self.stl_rotate[key][0])
        surf = surf.rotate_y(self.stl_rotate[key][1])
        surf = surf.rotate_z(self.stl_rotate[key][2])

        # translate
        surf = surf.translate(self.stl_translate[key])

        # scale
        surf = surf.scale(self.stl_scale[key])

        return surf

    def _compute_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Compute snap points from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        if self.verbose > 1:
            print("    * Calculating snappy points...")
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        self.snap_points = np.r_[xz_plane_points, yz_plane_points, xy_plane_points]

        # get the unique x, y, z coordinates
        x_snaps = np.unique(np.round(self.snap_points[:, 0], 5))
        y_snaps = np.unique(np.round(self.snap_points[:, 1], 5))
        z_snaps = np.unique(np.round(self.snap_points[:, 2], 5))

        # Include simulation domain bounds
        self.x_snaps = np.unique(np.concatenate(([self.xmin], x_snaps, [self.xmax])))
        self.y_snaps = np.unique(np.concatenate(([self.ymin], y_snaps, [self.ymax])))
        self.z_snaps = np.unique(np.concatenate(([self.zmin], z_snaps, [self.zmax])))

    def plot_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Plot snap points extracted from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        xz_cloud = pv.PolyData(xz_plane_points)
        yz_cloud = pv.PolyData(yz_plane_points)
        xy_cloud = pv.PolyData(xy_plane_points)

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        pl.add_mesh(model, color="white", opacity=0.5, label="base STL")
        pl.add_mesh(
            edges,
            color="black",
            line_width=5,
            opacity=0.8,
        )
        pl.add_mesh(
            xz_cloud,
            color="green",
            point_size=20,
            render_points_as_spheres=True,
            label="XZ plane points",
        )
        pl.add_mesh(
            yz_cloud,
            color="orange",
            point_size=20,
            render_points_as_spheres=True,
            label="YZ plane points",
        )
        pl.add_mesh(
            xy_cloud,
            color="magenta",
            point_size=20,
            render_points_as_spheres=True,
            label="XY plane points",
        )
        pl.add_legend()
        pl.show()

    def region_indices(self, field, component, center, size):
        """
        Indices and coordinates of the field samples inside a box region.

        E components sit at the edge midpoint along their own direction and on
        the nodes transversally; H components sit on the node along their own
        direction and at the cell midpoints transversally. Along directions of
        zero `size` the single nearest sample is selected. With MPI, ghost
        layers are excluded so that each sample belongs to exactly one rank.

        Parameters
        ----------
        field : str
            'E' or 'H'. 'J' samples are colocated with 'E'.
        component : str
            'x', 'y' or 'z'.
        center, size : tuple of float
            Centre and extent of the region (x, y, z) [m].

        Returns
        -------
        indices : list of 3 ndarray of int
            Selected indices along x, y and z (possibly empty along z on MPI
            ranks that do not own the region).
        coords : list of 3 ndarray of float
            Coordinates [m] of the selected samples along x, y and z.
        """
        c = "xyz".index(component)
        on_edge_mid = field.upper() in ("E", "J")
        nodes = (self.x, self.y, self.z)
        indices, coords = [], []
        for d in range(3):
            at_mid = (d == c) == on_edge_mid
            pos = (nodes[d][:-1] + nodes[d][1:]) / 2 if at_mid else nodes[d][:-1]

            owned = np.ones(pos.size, dtype=bool)
            if d == 2 and self.use_mpi:
                if self.rank > 0:
                    owned[: self.n_ghosts] = False
                if self.rank < self.size - 1:
                    owned[-self.n_ghosts :] = False

            if size[d] == 0 and d == 2 and self.use_mpi:
                # Pick the nearest sample on the global (uniform) z grid, so all
                # ranks agree on it even at subdomain interfaces, then map it to
                # the local index of the rank that owns it
                dz = (self.ZMAX - self.ZMIN) / self.NZ
                z_global = self.ZMIN + dz * (np.arange(self.NZ_samples) + 0.5 * at_mid)
                g = _nearest(z_global, center[d])
                local = g - self.rank * (self.NZ // self.size)
                local += self.n_ghosts if self.rank > 0 else 0
                inside = 0 <= local < pos.size and owned[local]
                idx = np.array([local] if inside else [], dtype=int)
            elif size[d] == 0:
                idx = np.array([_nearest(pos, center[d])])
            else:
                tol = 1e-9 * (nodes[d][-1] - nodes[d][0])
                lo, hi = center[d] - size[d] / 2 - tol, center[d] + size[d] / 2 + tol
                idx = np.flatnonzero((pos >= lo) & (pos <= hi))
                if idx.size == 0 and not self.use_mpi:
                    raise ValueError(
                        f"Region contains no {field}{component} samples along "
                        f"{'xyz'[d]}; increase size or refine the mesh."
                    )
            idx = idx[owned[idx]]
            indices.append(idx)
            coords.append(pos[idx])
        return indices, coords

    def refine_axis(self, xmin, xmax, Nx, x_snaps, method="insert", tol=1e-12):
        """
        Refine a grid axis using snap points and a chosen method.

        Parameters
        ----------
        xmin, xmax : float
            Axis bounds.
        Nx : int
            Number of grid points.
        x_snaps : array_like
            Snap points to include in the axis.
        method : str, optional
            Refinement algorithm ('insert', 'neighbor', 'subdivision').
        tol : float, optional
            Convergence tolerance for optimization.

        Returns
        -------
        x : ndarray
            Refined axis array.
        """

        # Loss function to minimize cell size spread
        def loss_function(x, x0, is_snap):
            # avoid moving snap points
            penalty_snap = np.sum((x[is_snap] - x0[is_snap]) ** 2) * 1000
            # avoid gaps < uniform gap
            dx = np.diff(x)
            threshold = 1 / (len(x) - 1)  # or a hardcoded `min_spacing`
            penalty_small_gaps = np.sum((threshold - dx[dx < threshold]) ** 2)
            # avoid large spread in gap length
            dx = np.diff(x)
            penalty_variance = np.std(dx) * 10
            # return penalty_snap + penalty_small_gaps + penalty_variance
            return np.hstack([penalty_snap, penalty_small_gaps, penalty_variance])

        # Uniformly distributed points as initial guess
        x_snaps = (x_snaps - xmin) / (xmax - xmin)  # normalize to [0,1]

        if method == "insert":
            x0 = np.unique(np.append(x_snaps, np.linspace(0, 1, Nx - len(x_snaps))))

        elif method == "neighbor":
            x = np.linspace(0, 1, Nx)
            dx = np.diff(x)[0]
            for s in x_snaps:
                m = np.isclose(x, s, rtol=0.0, atol=dx / 2)
                if np.sum(m) > 0:
                    x[np.argmax(m)] = s
            x0 = x.copy()

        elif method == "subdivision":
            # x = snaps
            while len(x) < Nx:
                # idx of segments sorted min -> max
                idx_max_diffs = np.argsort(np.diff(x))[-1]  # take bigger

                # print(f"Bigger segment starts at {x[idx_max_diffs]}")
                # compute new point in the middle of the segment
                val = x[idx_max_diffs] + (x[idx_max_diffs + 1] - x[idx_max_diffs]) / 2

                # insert the new point
                x = np.insert(x, idx_max_diffs + 1, val)
                x = np.unique(x)
                # print(f"Inserted point {val} at index {idx_max_diffs}")
            x0 = x.copy()
        else:
            raise ValueError(
                f"Method {method} not supported. Use 'insert', 'neighbor' or 'subdivision'."
            )

        # minimize segment length spread for the test points
        is_snap = np.isin(x0, x_snaps)
        result = least_squares(
            loss_function,
            x0=x0.copy(),
            bounds=(0, 1),  # (zmin, zmax),
            jac="3-point",
            method="dogbox",
            loss="arctan",
            gtol=tol,
            ftol=tol,
            xtol=tol,
            verbose=0,
            args=(x0.copy(), is_snap.copy()),
        )
        # transform back to [xmin, xmax]
        return result.x * (xmax - xmin) + xmin

    def _refine_xyz_axis(self, method="insert", tol=1e-6):
        """
        Refine grid axes using snap points extracted from STL solids.

        Uses the stored snap points (``self.x_snaps``, ``self.y_snaps``,
        ``self.z_snaps``) to refine the axis arrays ``self.x``, ``self.y``,
        ``self.z``. The refinement method and convergence tolerance control
        how new grid nodes are inserted.

        Parameters
        ----------
        method : {'insert','neighbor','subdivision'}, optional
            Refinement algorithm to use when inserting snap points. Default is
            'insert'.
        tol : float, optional
            Convergence tolerance passed to the refinement routine.
        """

        if self.verbose > 1:
            print(f"    * Refining x axis with {len(self.x_snaps)} snaps...")
        self.x = self.refine_axis(
            self.xmin,
            self.xmax,
            self.Nx + 1,
            self.x_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining y axis with {len(self.y_snaps)} snaps...")
        self.y = self.refine_axis(
            self.ymin,
            self.ymax,
            self.Ny + 1,
            self.y_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining z axis with {len(self.z_snaps)} snaps...")
        self.z = self.refine_axis(
            self.zmin,
            self.zmax,
            self.Nz + 1,
            self.z_snaps,
            method=method,
            tol=tol,
        )

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)

        if self.verbose > 1:
            print(f"    * Refined grid: Nx = {self.Nx}, Ny ={self.Ny}, Nz = {self.Nz}")

    def _assign_colors(self):
        """
        Assign colors for each STL solid based on material categories.

        Maps entries in ``self.stl_materials`` to color names using the
        ``material_colors`` lookup. Supports string keys referencing the
        material library or explicit material tuples (eps_r, mu_r[, sigma]).
        The resulting mapping is stored in ``self.stl_colors``.
        """
        self.stl_colors = {}

        for key in self.stl_solids:
            mat = self.stl_materials[key]
            if type(mat) is str:
                _color = material_colors.get(mat, material_colors["other"])
                self.stl_colors[key] = _color
            elif len(mat) == 2:
                if mat[0] is np.inf:  # eps_r
                    self.stl_colors[key] = material_colors["pec"]
                elif mat[0] > 1.0:  # eps_r
                    self.stl_colors[key] = material_colors["dielectric"]
                else:
                    self.stl_colors[key] = material_colors["vacuum"]
            elif len(mat) == 3:
                self.stl_colors[key] = material_colors["lossy metal"]
            else:
                self.stl_colors[key] = material_colors["other"]

    def save_to_h5(self, filename="grid.h5"):
        """
        Save the generated grid and STL metadata to an HDF5 file.

        The file contains axis arrays, STL masks suitable for ``grid.cell_data``,
        and all ``stl_`` related variables (materials, colors, transforms).

        Parameters
        ----------
        filename : str, optional
            Output filename for the HDF5 file. Default 'grid.h5'.
        """

        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Saving grid to HDF5 file:", filename)

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("x", data=np.array(self.x))
            hf.create_dataset("y", data=np.array(self.y))
            hf.create_dataset("z", data=np.array(self.z))
            hf.attrs["boundary_at_domain_edge"] = self.boundary_at_domain_edge
            hf.attrs["edge_ghosts"] = np.array(self.edge_ghosts)

            # Save stl_ variables as groups
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                grp = hf.create_group(attr)
                dct = getattr(self, attr)
                for key, val in dct.items():
                    # Use dtype='S' for strings, otherwise np.array
                    if isinstance(val, str):
                        grp.create_dataset(str(key), data=np.bytes_(val))
                    else:
                        grp.create_dataset(str(key), data=np.array(val))

            for key in self.stl_solids.keys():
                hf.create_dataset("grid_" + key, data=np.array(self.grid[key]))

    def load_from_h5(self, filename):
        """
        Load grid axis arrays and STL metadata from an HDF5 file.

        The function restores axis arrays, recomputes grid metrics and fills
        ``self.grid`` cell_data with imported STL masks saved previously by
        ``save_to_h5``.

        Parameters
        ----------
        filename : str
            HDF5 filename to read. The '.h5' suffix is appended if missing.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Loading grid from HDF5 file:", filename)

        with h5py.File(filename, "r") as hf:
            # reconstruct stl dicts
            self.x = hf["x"][()]
            self.y = hf["y"][()]
            self.z = hf["z"][()]
            # Files written before the domain-edge convention hold legacy grids
            self.boundary_at_domain_edge = bool(
                hf.attrs.get("boundary_at_domain_edge", False)
            )
            self.edge_ghosts = [int(g) for g in hf.attrs.get("edge_ghosts", [0, 0, 0])]

            # Load stl_ variables from groups
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                dct = {}
                grp = hf[attr]
                for key in grp.keys():
                    val = grp[key][()]
                    # Decode bytes to string if needed
                    if isinstance(val, bytes):
                        val = val.decode()
                    dct[key] = val
                setattr(self, attr, dct)

        # recompute dx, dy, dz, Nx, Ny, Nz
        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        gx, gy, gz = self.edge_ghosts
        self.xmin, self.xmax = self.x[0], self.x[-1 - gx]
        self.ymin, self.ymax = self.y[0], self.y[-1 - gy]
        self.zmin, self.zmax = self.z[0], self.z[-1 - gz]

        # recommpute grid and L, iA, tL, itA
        self._compute_grid()

        # asign masks to grid.cell_data
        with h5py.File(filename, "r") as hf:
            for key in self.stl_solids.keys():
                self.grid[key] = hf["grid_" + key][()]

        # add verbosity
        if self.verbose > 1:
            print(f"Loaded grid with {self.Nx * self.Ny * self.Nz} mesh cells:")
            print(f"    * Number of cells: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}")
            print(
                f"    * Simulation domain bounds: \n\
                x:[{self.xmin:.3f}, {self.xmax:.3f}],\n\
                y:[{self.ymin:.3f}, {self.ymax:.3f}],\n\
                z:[{self.zmin:.3f}, {self.zmax:.3f}]"
            )
            print(
                f"    * STL solids imported:\n\
                {list(self.stl_solids.keys())}"
            )
            print(
                f"    * STL solids assigned materials [eps_r, mu_r, sigma]:\n\
                {list(self.stl_materials.values())}"
            )

        # update logger
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
        self.update_logger(["stl_solids", "stl_materials"])
        if self.stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if self.stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if self.stl_scale != 1.0:
            self.update_logger(["stl_scale"])

    def update_logger(self, attrs):
        """
        Copy selected Grid attributes into the internal ``Logger``.

        Parameters
        ----------
        attrs : iterable of str
            Names of attributes to copy into ``self.logger.grid``.
        """
        for atr in attrs:
            self.logger.grid[atr] = getattr(self, atr)
