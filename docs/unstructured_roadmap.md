# Unstructured FIT: phase-2 design

Phase 1 (`wakis/unstructured/`, physics guide section "Unstructured FIT")
provides triangular (2D) and tetrahedral (3D) FIT with a circumcentric
(Voronoi) dual, diagonal Hodge matrices assembled per cell, explicit
leapfrog, PEC/PMC/first-order Silver–Müller boundaries, Drude/Lorentz/Debye
ADEs, point/current sources, probes, DFTs and a shift-invert eigen-solver.
This document plans the features needed to use it as a production solver
next to the Cartesian `SolverFIT3D`. Every item keeps the update **linear in
the fields** and expressed with fixed sparse operators, so that the discrete
adjoint stays the transposed update (see "Adjoint" below).

Notation: `e` edge voltages, `b` face fluxes, `C` face–edge incidence,
`M_eps`, `M_nu` diagonal Hodges, `K = C^T M_nu C`.

## 1. CFS-PML on unstructured meshes

The first-order Silver–Müller boundary reflects a few percent at grazing
incidence. Two options, in order of preference:

1. **Layered convolutional PML in a tetrahedral shell.** Generate the
   absorbing region as conforming layers (prisms split into tetrahedra) of a
   convex outer surface (sphere or rounded box) so that every PML edge has a
   well-defined normal coordinate `s` (distance to the inner PML surface)
   and normal direction `n`. Split the field only by direction: the
   stretched Ampère/Faraday laws in the CFS form
   `s(w) = kappa + sigma/(alpha + i w eps_0)` act on the normal component.
   On a Voronoi dual the normal part of an edge voltage is `(t·n) e`; the
   per-edge auxiliary (recursive convolution) variables are then local and
   diagonal, exactly like the dispersive ADE groups in `CapacitiveUpdate`.
   Faces get the analogous update with `(n_f·n)`. Grading `sigma ~ s^3`,
   10 layers, `kappa_max ~ 5` for evanescent waves.
2. **Complex-coordinate stretching of the Hodges** (frequency domain only,
   for the eigen/driven solvers): replace `M_eps`, `M_nu` in the PML shell
   by their stretched values; needs a complex symmetric solver (already
   used by `scipy.sparse.linalg.splu`).

Validation: line source / dipole reflection vs exact Green's function at
normal and grazing incidence (target −40 dB with 10 layers), long-run
stability (CFS alpha > 0 removes the late-time growth of standard PML).

## 2. Ports and mode solver on mesh boundaries

* **2D mode solver on a boundary patch.** A port is a set of boundary faces
  (triangles) with a common plane. The restriction of the 3D complex to the
  patch is a 2D triangular complex: transverse E on the patch edges,
  longitudinal H on patch nodes. The waveguide eigenproblem is the
  triangular FIT of `UnstructuredFIT2D` with the propagation constant as
  eigenvalue (edge-node mixed form, as for the Cartesian `solve_modes`).
* **Mode source / monitor.** Inject the mode as an equivalent surface
  current on the port edges (`i_e = A_e (n x H_mode)·t_e` on dual faces
  adjacent to the port, the unstructured analogue of `ModeSource`) and
  extract amplitudes by overlap of DFT voltages with the mode voltages
  (plain dot products, JAX-friendly).
* **Lumped ports.** An edge chain between two conductors with a series
  resistance enters `CapacitiveUpdate` as a conductance `G = 1/R` on those
  edges plus a voltage source; S-parameters follow `wakis.sparameters`.

## 3. Adjoint (inverse design)

The update is `b' = b - dt C e`, `e' = (a1 e + dt (C^T M_nu b - j) - ADE) / den`
with fixed diagonal `a1`, `den`, so the adjoint run is the same leapfrog
with `C` and `C^T` swapped in role (time-reversed, transposed): exactly what
`wakis/inverse/AdjointStepper` does for the Cartesian solver. Needed:

* `UnstructuredAdjointStepper` applying the transposes of the two steps
  (massless-edge elimination: transposed Schur complement, one extra
  `K_ss` solve with the same factorization; Whitney Hodge: one extra
  `M_1` solve).
* Design variables: per-cell permittivity `eps_T` (a `MaterialGrid` on
  cells). The gradient is `dJ/deps_T = sum_e (dJ/dM_eps[e]) eps_0 A_{e,T}/L_e`,
  a sparse product with `mesh.dual_area_partial`, i.e. a fixed linear map;
  the JAX layer sees it as a `custom_vjp`.
* Objectives from DFT data are plain arrays (`Recorder.dfts`), as now.
* Shape optimization (moving nodes) needs the derivatives of the dual
  measures with respect to node positions (circumcentres are rational
  functions of the nodes); defer to phase 3.

## 4. GPU (CuPy)

All operators are `scipy.sparse` CSR matrices and diagonal vectors: port
`one_step` to `cupyx.scipy.sparse` (two SpMVs, elementwise updates, the
ADE groups with gather/scatter by `idx`). The massless-edge solve becomes a
small sparse direct solve on the host or, better, a precomputed sparse
`K_ss^{-1} K_sf` (massless edges are isolated, `K_ss` is nearly diagonal) so
the step stays two SpMVs. Expected: memory-bound, ~2.1 kB/tet on the host
(see `estimate_memory`), ~1 kB/tet on the device; an L4 (24 GB) holds
~2e7 tetrahedra.

## 5. MPI partitioning

Partition the cells (METIS/`pymetis` or a simple recursive coordinate
bisection, BSD-licensed alternatives only), own edges/faces by the lowest
rank of their cells, and exchange halo voltages and fluxes after each half
step (one neighbour exchange of `e` before the `b` update, one of `b`
before the `e` update). The diagonal Hodges make the exchange the only
communication; Lanczos for `dt` needs a global reduction.

## 6. Hybrid structured/unstructured coupling

Most optical devices are mostly vacuum/layered with a few curved parts.
Couple the Cartesian `SolverFIT3D` (Yee, cheap) to tetrahedral patches
around curved objects:

* Tetrahedralize a box region aligned with the Cartesian grid whose outer
  boundary nodes are the Cartesian nodes (each boundary square split into
  two triangles). The boundary edges are then shared Cartesian edges.
* On the shared surface the unstructured dual faces are cut by the
  boundary; complete them with the Cartesian half-dual faces (the same
  construction as subgridding with grid-line termination, Thoma & Weiland
  1996, which keeps the energy conservation and hence stability).
* The two leapfrogs share `dt` (the smaller one; local time stepping later).

## 7. CAD → mesh workflow

* STL/STEP import with gmsh (external process or optional import; GPL tool,
  not bundled): `gmsh_mesh(build, h, file)` already exists; add a helper
  that takes STL/STEP files with material names, builds the OCC fragments
  (`occ.importShapes`, `occ.fragment`) and maps physical groups to Wakis
  materials.
* Mesh quality for the Voronoi Hodge: request Delaunay-type 3D algorithms
  (`Mesh.Algorithm3D = 1`), run `Mesh.Optimize`/`OptimizeNetgen`, then check
  `SimplexMesh.diagnostics()`. Non-Delaunay meshes fall back to
  `hodge="whitney"` (consistent, non-diagonal) automatically or after a
  user decision.
* Size fields: `h <= lambda/(10 n)` in each material, curvature-adapted
  refinement on interfaces (`Mesh.MeshSizeFromCurvature`).

## 8. Known phase-1 limitations to address

* Time step on 3D Delaunay meshes is ~0.15 h/c (vs 0.58 h/c for a cube
  Yee grid) because Voronoi faces of nearly co-spherical configurations are
  small; massless-edge elimination keeps it bounded. Better meshes (ODT +
  sliver exudation, or lattice-aligned interior with conforming interface
  layers) and the hybrid scheme of item 6 should recover a Yee-like step.
* Eigen-solves use SuperLU (COLAMD); its fill-in dominates memory above
  ~1e5 tetrahedra. Use a symmetric indefinite factorization (e.g. MUMPS via
  an optional dependency) or an iterative solver with an auxiliary-space
  (AMS) preconditioner.
* Curved (higher-order) boundaries: the interfaces are polyhedral; volume
  preservation of interior shells removes the leading error, but outer PEC
  walls stay inscribed.
