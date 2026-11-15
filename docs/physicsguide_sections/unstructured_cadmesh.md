# CAD meshes and the tetrahedral time step

Code: `wakis/unstructured/cad.py` (CAD → tetrahedra), `wakis/unstructured/quality.py` (shape metrics, element
stability bounds), `wakis/unstructured/improve.py` (time-step-driven node relocation),
`wakis/unstructured/analytic.py::layered_pillbox_tm0`. Tests: `tests/test_102_unstructured_cadmesh.py`.
Study: `studies/unstructured-cadmesh/` (`cadmesh_study.py`, `plot_cadmesh.py`).

## 1. From CAD to a FIT mesh

`mesh_cad(source, regions, materials, f_max, cpw)` drives the optional external mesh generator gmsh (GPL; imported
lazily, never bundled, only its public API is called) with the OpenCASCADE kernel:

1. import the STEP/BREP/IGES file (or run a `build(gmsh)` callable);
2. *fragment* all volumes so that every interface between two regions is a shared, conforming surface;
3. assign each CAD volume to a named region by a point inside it, a regular expression on its STEP label
   (labels are carried through the fragment), or a predicate; volumes assigned to no region are removed, and
   their surfaces become the outer boundary of the mesh (PEC or PMC in `UnstructuredFIT3D`): metal parts of a
   CAD assembly (copper walls, beryllium windows) are simply not meshed;
4. set one constant size field per region, $h_m = c_0 / (f_{\max}\, n_m\, \mathrm{cpw})$ with
   $n_m=\sqrt{\varepsilon_r\mu_r}$ (project convention: cpw counted in the highest-index material, 10 fast,
   20 accurate; `size_mode="uniform"` uses the highest-index size everywhere), optionally refined on curved
   surfaces (`curvature_points` elements per $2\pi$);
5. preflight: the number of tetrahedra is estimated before meshing ($\approx 8.5\,V/h^3$, measured
   6.5–7.5 per $h^3$ on gmsh output) and the call refuses meshes above `max_tets` (a forgotten mm → m scale
   is a $10^9$ error);
6. mesh, read nodes and tetrahedra directly from the gmsh API (no file needed), tag cells by region and
   return the `{tag: material}` dict the solver takes.

`mesh_stl` closes a watertight STL into a volume (patch classification at a feature angle, discrete geometry,
remeshing at the target size), optionally inside a padding box of background material. `read_mesh` reads any
meshio format (`.msh` 2.2/4.1, `.vtu`, …) and maps gmsh physical names to materials and surface groups to face
indices.

**Delaunay output matters.** The diagonal (Voronoi) Hodge needs non-negative assembled dual measures. gmsh's 3D
Delaunay output without optimization has none negative on our cases; its tetrahedral *shape* optimizers (the
default `Mesh.Optimize` and Netgen) improve dihedral angles but break the Delaunay property (tens to hundreds of
negative dual areas and dual lengths on a 2e3-tet pillbox) and do not raise — Netgen even halves — the
FIT time step (Section 4). `mesh_cad(optimize=None)` is therefore the right input for Voronoi FIT;
`optimize="gmsh"` remains the default because it is what users of gmsh expect for the Whitney Hodge.

## 2. The stability limit is a sum of element limits

The leapfrog of `UnstructuredFIT3D` advances $M\ddot e = -K e$ on the free edges, $K = C^T M_\nu C$,
$M = M_\varepsilon$, and is stable iff $\Delta t < 2/\sqrt{\lambda_{\max}}$, $\lambda_{\max}$ the largest
eigenvalue of the pencil $(K, M)$. Both matrices are assembled from tetrahedra,
$K=\sum_T P_T^T K_T P_T$, $M=\sum_T P_T^T M_T P_T$ ($P_T$ picks the six edges of $T$), with $K_T\succeq 0$.
If every $M_T$ is positive definite on the free edges of $T$, then for any $x$

$$x^TKx=\sum_T x_T^TK_Tx_T\le\sum_T\lambda_T\,x_T^TM_Tx_T\le\Big(\max_T\lambda_T\Big)\,x^TMx,
\qquad \lambda_T=\lambda_{\max}(K_T,M_T),$$

so $\lambda_{\max}\le\max_T\lambda_T$ and the global step is at least the smallest element step,
$\Delta t_{\max}\ge\min_T\Delta t_T$, $\Delta t_T = 2/\sqrt{\lambda_T}$ (the classical element eigenvalue
bound of finite elements, e.g. Hughes, *The Finite Element Method*, Dover 2000, §9.3; Fried 1972). Restricting
to the free edges (PEC edges removed) is a principal sub-pencil, which by Cauchy interlacing only lowers each
$\lambda_T$. A matching lower bound is the Rayleigh quotient of a unit vector,
$\lambda_{\max}\ge\max_e K_{ee}/M_{ee}$.

**Whitney Hodge.** The consistent element mass matrices of the Whitney 1- and 2-forms are the natural split:
$M_T$ is positive definite on every non-degenerate tetrahedron, and the bound holds on any mesh (tested on a
randomly distorted, non-Delaunay mesh).

**Voronoi Hodge.** $M$ and $M_\nu$ are diagonal; the circumcentric partial measures $A_{e,T}$, $h_{f,T}$ are
*signed* (negative when the circumcentre lies outside $T$, which is common: 11–46 % of the tetrahedra in our
meshes are not well centred), so the geometric split can fail to be positive. But any split
$M_{ee} = \sum_T w_{e,T}M_{ee}$ with $w\ge 0$, $\sum_T w_{e,T}=1$ is exact. Choosing the
**stiffness-proportional** weights $w_{e,T}=(K_T)_{ee}/K_{ee}$ gives $M_T = D_T R_T$ with $D_T=\mathrm{diag}K_T$
and $R_T=\mathrm{diag}(M_{ee}/K_{ee})$, hence

$$\lambda_T\le \nu_T\,\max_{e\in T}\frac{K_{ee}}{M_{ee}},\qquad
\nu_T=\lambda_{\max}\big(D_T^{-1/2}K_TD_T^{-1/2}\big)\le\operatorname{tr}=\#\{\text{edges of }T\text{ with }(K_T)_{ee}>0\}\le 6,$$

and together with the lower bound

$$\max_e\frac{K_{ee}}{M_{ee}}\;\le\;\lambda_{\max}\;\le\;\max_T\lambda_T\;\le\;6\,\max_e\frac{K_{ee}}{M_{ee}} .$$

The time step is bracketed within a factor $\sqrt 6$ by purely local quantities, the **edge rates**
$K_{ee}/M_{ee} = L_e\sum_{f\ni e}(H_f/A_f)/(\mu_0\varepsilon_0 A^\varepsilon_e)$ ($H_f$ the material-weighted dual
length of face $f$, $A^\varepsilon_e$ the material-weighted dual area of edge $e$). Faces with a negative
assembled dual length (non-Delaunay) make $K$ indefinite — a growing mode no step size cures — and are reported
(`n_negative_faces`); dropping them only lowers $x^TKx$, so the bound stays an upper bound for the positive part.
With massless edges eliminated (Schur complement $K_{ff}-K_{fs}K_{ss}^{-1}K_{sf}\preceq K_{ff}$), the bounds
computed on $K_{ff}$ remain upper bounds.

The geometric weights (positive partials) are a poor split: an edge whose dual face barely enters a
tetrahedron gives that tetrahedron a tiny mass share but a full stiffness share, and the bound was 11–12× too
pessimistic on the pillbox and the ball. The stiffness split is the default (`weights="stiffness"`).

## 3. What sets dt on real meshes (measured)

`cadmesh_study.py bounds` (Batch `cadmesh-study-4`, `results/bounds.json`) on ten meshes: the CAD pillbox,
layered pillbox and muon cavity at cpw 10 and 20 (gmsh Delaunay output) and `ball_mesh` at h = 0.25, 0.15.

| | Voronoi (stiffness split) | Whitney |
|---|---|---|
| upper bound $\sqrt{\max_T\lambda_T/\lambda_{\max}}$ | 1.025–1.090 | 1.19–1.55 |
| lower bound $\sqrt{\max_e (K_{ee}/M_{ee})/\lambda_{\max}}$ | 0.986–0.993 | 0.85–1.00 |
| Gershgorin $\sqrt{\lambda_G/\lambda_{\max}}$ | 1.09–1.14 (pillbox, ball) | – |
| energy of the dt-limiting mode on the worst 1 % elements | 0.98–1.00 | 0.00–0.92 |

* **Voronoi FIT: dt is set by single edges.** The largest edge rate is within 1.4 % of $\lambda_{\max}$ and
  the fastest mode puts 98–100 % of its energy on the 1 % of elements with the smallest $\Delta t_T$. The
  global step is a *local* property, and the stiffest edges are interior edges whose dual (Voronoi) face is
  nearly degenerate: four almost co-spherical nodes. Their normalized dual area sits just above the
  `massless_tol` = 1e-2 elimination threshold (pillbox cpw 10: worst edge $A_e/L_e^2$ = 0.0134 against a
  median of 0.215, rate 8.5× the median). Below the threshold the solver eliminates the edge; the next one
  sets dt. On the coarse pillbox the worst edges are chords between boundary nodes whose dual-face ring has at
  most one interior node — fixed boundary nodes are why boundary sliding matters (Section 4).
* **The worst element by dt is not the worst by shape (Voronoi).** On the cpw 20 pillbox the five elements
  with the smallest $\Delta t_T$ have radius ratios 0.71–0.81, and the worst-shaped sliver (radius ratio
  0.007) ranks 11th by $\Delta t_T$; on the cpw 20 muon mesh the worst-shaped element (0.022) ranks 8218th.
  Classical quality measures do not predict the Voronoi-FIT step; the edge rates do. For the Whitney Hodge it
  is the other way round: the element carrying most of the fastest mode's energy is a sliver (radius ratio
  0.01 on the pillbox, 0.14 on the muon cavity), as expected from consistent mass matrices.
* **Whitney FIT: dt is a collective property.** The consistent mass couples neighbouring edges, the fastest
  mode is mostly spread out (0–56 % of its energy on the worst 1 %, 92 % on the muon cpw 20) and the element
  bound is 1.2–1.5× pessimistic; its step is larger than Voronoi's on the same mesh (pillbox cpw 10:
  $c\Delta t/\bar h$ = 0.21 vs 0.15), at the price of a sparse solve per step.

## 4. Raising dt: node relocation driven by the edge rates

`improve_dt(mesh, materials)` minimizes, node by node, the largest edge rate in the node's neighbourhood
(every quantity a move changes lives on the edges of the tetrahedra around the node, so a trial is O(1)):
the candidate positions are the optimal-Delaunay point (volume-weighted mean of the incident circumcentres,
Chen & Xu 2004), halfway to it, and 24 random points within 0.25 local edge lengths. Boundary and interface nodes
slide on their surface — on the exact CAD surface through gmsh's closest-point projection when called from
`mesh_cad(improve=...)`, in their plane otherwise — and nodes on CAD curves and corners stay fixed, so the
geometry and the material regions are unchanged. Moves are compared lexicographically: first the number of
faces with a negative dual length around the node (they make $K$ indefinite), then the largest rate; edges
below `massless_tol` are ignored as the solver eliminates them. Each sweep visits the nodes around the edges
above the 98th percentile of the rates, worst first; 10 sweeps take 6–100 s for 2e3–5e4 tetrahedra (Python).

**Baselines (named):** the same CAD, f_max and cpw with gmsh's Delaunay output ("none"), gmsh's default
optimizer ("gmsh") and Netgen's ("netgen"); solver defaults everywhere (Voronoi Hodge, `massless_tol` 1e-2,
CFL 0.95). Batch `cadmesh-study-4`, `results/improve.json`:

| case, cpw | $c\Delta t/\bar h$ none | gmsh | netgen | **ours** | gain vs none | rms freq. error none → ours | negative faces gmsh / netgen |
|---|---|---|---|---|---|---|---|
| pillbox 10 | 0.158 | 0.159 | 0.077 | **0.235** | 1.49 | 5.0e-3 → 4.5e-3 (5 modes) | 48 / 129 |
| pillbox 15 | 0.135 | 0.024 | 0.098 | **0.199** | 1.47 | 2.8e-3 → 2.1e-3 | 148 / 294 |
| pillbox 20 | 0.114 | 0.114 | 0.089 | **0.196** | 1.73 | 1.4e-3 → 9.5e-4 | 288 / 425 |
| layered 10 | 0.140 | 0.138 | 0.086 | **0.201** | 1.43 | 5.7e-3 → 5.0e-3 (TM01) | 218 / 612 |
| layered 15 | 0.136 | 0.119 | 0.056 | **0.216** | 1.59 | 2.0e-3 → 1.9e-3 | 617 / 1066 |
| layered 20 | 0.123 | 0.118 | 0.081 | **0.203** | 1.66 | 5.1e-4 → 3.9e-4 | 1281 / 1802 |
| muon 10 | 0.136 | 0.137 | 0.067 | **0.206** | 1.51 | see Section 5 | 39 / 242 |
| muon 15 | 0.120 | 0.073 | 0.082 | **0.180** | 1.50 | | 212 / 493 |
| muon 20 | 0.132 | 0.058 | 0.076 | **0.200** | 1.52 | | 269 / 679 |

The time step grows by 1.47–1.73× (pillbox) and 1.50–1.52× (muon cavity) over the unoptimized Delaunay mesh,
at equal or better resonance accuracy on the pillbox. On the layered pillbox, boundary recovery across the planar
material interface and local sliver optimization (`Mesh.OptimizeThreshold`) can leave non-Delaunay faces in the
unoptimized baseline mesh, while gmsh's and Netgen's tetrahedral shape optimizers maximize element aspect ratios
without preserving the local Delaunay circumsphere condition, producing negative dual lengths, an indefinite
curl-curl matrix $K$, and diverging time-domain leapfrog runs; topological flip repair (`flip_to_delaunay`,
invoked before and after node relocation when `improve_dt(..., repair=True)`) restores local Delaunayhood
(measured values: see studies/unstructured-cadmesh/results/repair.json). The number of eliminated (massless) edges changes by −9 % to +29 %; the per-step work
is the same sparse products plus a Schur solve on those edges. Serial wall time per simulated ns
(`cadmesh_study.py timing`, one process, one BLAS thread, median of 5 interleaved repetitions, Batch `cadmesh-timing-1`) at cpw 20: pillbox
0.126 → 0.080 s/ns (1.58× faster), layered 0.697 → 0.403 s/ns (1.73×), muon 0.061 → 0.044 s/ns (1.38×); the
step itself costs +10 %, −4 %, +10 % (more or fewer eliminated edges in the Schur solve), so the wall-time gain
is a little below the dt gain. (The per-arm timings inside the `improve` part ran 8 workers per VM and scatter
by 0.35–1.97×; they are not used.)

## 5. Example: a muon-cooling RF cavity from STEP

`tests/stl/006_muonCavity.stp` (Wakis test data; five solids in mm: two vacuum half cells, two 30 µm
beryllium windows, the copper body) is imported with `regions={"vacuum": r"Vacuum$"}`: the two half cells are
meshed, copper and windows are dropped and become the PEC boundary. The independent reference is the Cartesian
conformal-PEC FIT of Wakis (`SolverFIT3D`, vacuum STL exported from the same CAD, `bg="pec"`,
`conformal="pec"`) ring-down with harminv at h = 4, 3, 2.5 and 2 mm (0.28–2.0 M cells; Batch
`cadmesh-cart-*-1`): TM010-like 0.704164 GHz (spread 2e-5 over the four grids), second mode
1.064767 GHz ($h^2$ extrapolation of 3, 2.5, 2 mm; 1.064705 at 2 mm). Tetrahedral eigenmodes (Batch
`cadmesh-muontet-2`, `results/muon_tet.json`), relative error of the TM010-like mode:

| cpw (f_max 1 GHz) | tets | Voronoi | Voronoi + improve_dt | Whitney | dt gain of improve_dt |
|---|---|---|---|---|---|
| 10 | 1846 | +1.94 % | +2.11 % | +1.07 % | 1.51 |
| 15 | 5572 | +0.41 % | +0.54 % | +0.07 % | 1.50 |
| 20 | 11259 | +0.33 % | +0.49 % | +0.01 % | 1.52 |
| 30 | 36071 | +0.20 % | +0.30 % | −0.04 % | 1.57 |
| 40 | 82836 | +0.18 % | +0.28 % | −0.06 % | 1.76 |

The meshed volume is inscribed (−2.65 % at cpw 10, −0.12 % at cpw 40). Whitney agrees with the Cartesian
reference to 6e-4 from cpw 20 on; the Voronoi Hodge converges slowly towards +0.17 % (open question: the
boundary-adjacent, non-well-centred tetrahedra and the eliminated edges are the candidates). On this cavity
`improve_dt` costs accuracy, +0.10 percentage points (within the factor-2 "equal accuracy" criterion of the
tests, but not equal), for a 1.5–1.8× larger step. Time-domain ring-downs on the tetrahedra
(z-dipole, 60 ns, harminv, leapfrog dispersion removed) agree with the eigen-solve frequencies to the five printed digits
(cpw 15: 0.70705 GHz both) at dt = 9.06 ps (Delaunay) and 13.59 ps (improved).

## 6. Limits & Failure Modes

* **Non-Delaunay meshes are unstable with the Voronoi Hodge.** A face with a negative assembled dual
  length makes $K$ indefinite; the leapfrog then grows exponentially at *any* $\Delta t$ (measured: NaN after
  500 steps on a curvature-refined muon-cavity mesh with 170 such faces). Eigenmode solves do not notice
  (shift-invert finds the positive modes). gmsh's shape optimizers, curvature refinement
  (`curvature_points`) and boundary recovery all produce such faces. `UnstructuredFIT3D` (`hodge="voronoi"`)
  counts faces with negative dual length that couple free edges and raises `ValueError` by default
  (`allow_negative_faces=True` emits a warning instead; `stability_report(solver)["n_negative_faces"]` reports
  the same count): run `improve_dt` (it removes them first, but not all: faces whose nodes are all on CAD curves
  or corners cannot be repaired without topological flips) or use `hodge="whitney"` (stable on any mesh).
* **Connectivity is fixed.** `improve_dt` only moves nodes. Slivers that need edge/face flips (sliver exudation,
  Cheng et al. 2000) or nodes on CAD curves and corners stay as they are; the dt gain saturates after
  5–10 sweeps (Section 4).
* **The gain partly comes from eliminating more edges.** The objective ignores edges below
  `massless_tol * L^2` (the solver eliminates them), so moves that push an edge below the threshold are
  accepted: the number of massless edges changes by −9 % to +29 %. The accuracy cost is measured (Section 4: equal or better
  resonance errors), and the per-step cost of the Schur solve is included in the wall-time comparison, but a
  much larger `massless_tol` *does* cost accuracy (pillbox cpw 10: TM010 error −0.41 %, +0.36 %, +5.4 % at
  1e-2, 3e-2, 1e-1).
* **Sliding changes the polyhedral geometry at O(h²).** Nodes on curved CAD surfaces are projected back onto
  the exact surface; the chord error of the polyhedral boundary is unchanged in order but not identical
  (muon cpw 10: meshed volume −2.64 % → −2.75 % of the CAD volume). Interface clipping of the dual areas is
  re-applied on the final mesh but ignored inside the local search.
* **Element bound tightness is measured, not guaranteed, for Whitney.** The Voronoi bracket within $\sqrt 6$ is
  derived; the Whitney bound was 1.19–1.54× on our meshes with no proven factor.
* **Scope.** PEC/PMC outer boundaries only in the stability analysis (ABC conductances and the PML are not in the
  element matrices; lossy terms do not change the lossless leapfrog limit but the PML lowers dt, see the PML
  section). Material-interface edges use the clipped assembled masses. The STL path needs a watertight surface;
  remeshed STL bodies are inscribed in the STL (volume deficit $\le h^2/(2R^2)$ on a sphere of radius $R$).
* **gmsh is GPL** and optional: without it `mesh_cad`/`mesh_stl` raise ImportError with alternatives
  (`read_mesh` with meshio, or the pure-Python `ball_mesh`/`disk_mesh`); `quality` and `improve` need only
  NumPy/SciPy.

## Topological repair of non-Delaunay meshes

### Derivation and signed dual-length criterion

In the orthogonal Voronoi Finite Integration Technique, the diagonal magnetic-reluctivity Hodge entry on primal
face $f$ is assembled from the signed orthogonal distances between the circumcentres $\mathbf{c}_T$ of its incident
tetrahedra $T \ni f$ and the face circumcentre $\mathbf{c}_f$:

$$H_f = \sum_{T \ni f} \frac{h_{f,T}}{\mu_T}, \qquad h_{f,T} = (\mathbf{c}_T - \mathbf{c}_f)\cdot\mathbf{n}_{f,T},$$

where $\mathbf{n}_{f,T}$ is the unit normal to $f$ pointing into $T$ (`tet_partials` in
`wakis.unstructured._core`). For an interior face $f = T_1 \cap T_2$ within a homogeneous region
($\mu_{T_1} = \mu_{T_2}$), $h_{f,T_1} + h_{f,T_2}$ is the signed length of the Voronoi dual edge joining
$\mathbf{c}_{T_1}$ and $\mathbf{c}_{T_2}$ along $\mathbf{n}_f$; $H_f < 0$ occurs if and only if the pair
$(T_1, T_2)$ violates the local Delaunay empty-circumsphere condition (each apex lies strictly inside the
circumsphere of the adjacent tetrahedron). On a domain boundary face $f \subset \partial\Omega$ carrying at least
one free primal edge (`bc="pec"` or `bc="pmc"`), $H_f = h_{f,T}/\mu_T < 0$ when $\mathbf{c}_T$ lies on the exterior
side of $f$. Whenever $H_f < 0$ on a face that couples free edges, the curl-curl matrix
$K = C^T M_{\mu^{-1}} C$ ceases to be positive semi-definite and explicit leapfrog time stepping is unconditionally
unstable (`UnstructuredFIT3D` raises `ValueError` unless `allow_negative_faces=True`).

Continuous node relocation (`improve_dt` with `repair=False`) cannot always eliminate faces with $H_f < 0$ when
their vertices are constrained to CAD curves, corners, or material interfaces, or when the local tetrahedral
connectivity locks a sliver. Topological flips (`flip_to_delaunay` in `wakis.unstructured.repair`) change the local
connectivity without moving existing vertices:

* **2-3 flip (`_flip_23`):** replaces two interior tetrahedra $(a, c, d, e)$ and $(b, c, d, e)$ sharing the
  non-Delaunay face $(c, d, e)$ by the three tetrahedra $(a, b, c, d)$, $(a, b, d, e)$, and $(a, b, e, c)$ sharing
  the edge $(a, b)$, valid when the segment $ab$ pierces the interior of triangle $(c, d, e)$ so all three
  replacement tetrahedra have strictly positive signed volume.
* **3-2 flip (`_flip_32`):** the topological inverse of the 2-3 flip; when an interior edge $(a, b)$ of degree
  three is surrounded by $(a, b, c, d)$, $(a, b, d, e)$, and $(a, b, e, c)$, replaces the three tetrahedra by the
  two tetrahedra $(a, c, d, e)$ and $(b, c, d, e)$ sharing face $(c, d, e)$.
* **4-4 flip (`_flip_44`):** when an interior edge $(a, b)$ of degree four is surrounded by four tetrahedra over an
  equatorial cycle $(c, d, e, p)$, replaces them by four tetrahedra sharing the alternative equatorial diagonal
  $(c, e)$ or $(d, p)$ (equivalent to a 2-3 flip followed by a 3-2 flip across a coplanar four-node quad).
* **Boundary 2-2 flip (`_flip_22`):** when two boundary-adjacent tetrahedra $(a, c, d, p)$ and $(b, c, d, p)$ share
  an interior face $(c, d, p)$ and their outer faces $(a, b, c)$ and $(a, b, d)$ lie on a common plane of
  $\partial\Omega$ sharing the boundary edge $(a, b)$, flips the boundary diagonal from $(a, b)$ to $(c, d)$ to
  form $(c, d, a, p)$ and $(c, d, b, p)$, accepted only when the two boundary triangles are coplanar and their
  union area and outward unit normals are preserved to relative tolerance $10^{-12}$.

**Locked-node and material-interface rules:**
1. *Material interfaces:* a flip is allowed only if all tetrahedra involved have the same `cell_tags` value; newly
   created tetrahedra inherit that tag. Interface faces separating distinct `cell_tags` are never removed or
   flipped across, preserving both total volume and per-tag volume to relative tolerance $10^{-12}$.
2. *Locked CAD nodes (`locked`):* given an optional boolean mask `locked` over mesh vertices (`True` on fixed CAD
   curves or corners, or `~movable` inside `improve_dt`), any flip that removes an edge whose two endpoints are
   both locked (`refused["locked_edge"]`) or a boundary face whose three vertices are all locked
   (`refused["locked_boundary_face"]`) is refused.

### Algorithm

`flip_to_delaunay(mesh, materials=None, locked=None, bc="pec", steiner=True, max_flips=None, verbose=False, enabled_flips=("2-3", "3-2", "4-4", "2-2"))` uses the shared `neg_faces` criterion from `wakis.unstructured.improve` (the same signed dual-length computation used by `improve._State` and `stability_report` in `wakis.unstructured.quality`):

1. Initialize a queue with all faces satisfying $H_f < -10^{-14}\,\bar{h}/\mu_0$.
2. Pop a face $f$ from the queue; if $f$ is still present in the current mesh and has $H_f < 0$, evaluate candidate
   flips in `enabled_flips` (`_flip_23` on $f$, `_flip_32` and `_flip_44` on the edges of $f$, and `_flip_22` when
   $f$ is incident to a coplanar boundary triangle pair).
3. Accept a candidate flip only when every replacement tetrahedron has strictly positive volume, total and per-tag
   volumes are conserved, and the flip strictly decreases the count of negative faces on the modified cavity (or
   keeps it zero while increasing the minimum signed dual length). Re-enqueue all faces of the newly created
   tetrahedra.
4. If negative interior faces remain after the queue empties and `steiner=True`, `_insert_steiner` inserts an
   interior Steiner node along the segment from the circumcentre of the offending same-tag tetrahedron pair toward
   the host centroid (via a 1-4 or 2-6 split) and cascades local flips, committing the split only if the global
   negative-face count strictly decreases.
5. When `improve_dt(..., repair=True)` (the default), `improve_dt` runs `flip_to_delaunay` $\to$ time-step-driven
   node smoothing $\to$ `flip_to_delaunay` with `locked = ~movable` and appends a `"repair"` report dictionary to
   `history`. Setting `repair=False` runs node smoothing alone.

### Limits & Failure Modes

* **Stuck faces (`report["stuck"]`).** In three dimensions, a non-Delaunay interior face $f = (c, d, e)$ shared by
  $(a, c, d, e)$ and $(b, c, d, e)$ is geometrically non-flippable by a 2-3 move when the segment $ab$ passes
  outside triangle $(c, d, e)$ (the five-vertex polyhedron is non-convex), and a 3-2 flip is unavailable unless the
  reflex edge on the boundary of the pair has degree three in the mesh (Joe 1991; Edelsbrunner and Shah 1996).
* **Locked curves and material interfaces.** Flips that would remove an edge with both endpoints marked `locked` or
  remove a triangle on a material interface (`cell_tags` jump) or non-planar boundary are refused by construction.
  When all vertices of a boundary-adjacent or interface-adjacent sliver are locked by CAD curves or corners, neither
  flips nor node relocation can alter its faces.
* **3D flip-sequence non-termination risk.** Unlike two-dimensional Lawson flips, unconstrained three-dimensional
  local flips can cycle indefinitely if non-improving flips are permitted (Joe 1991; Edelsbrunner and Shah 1996).
  `flip_to_delaunay` avoids cycles by requiring strict local improvement in the negative-face count (or minimum
  dual length at zero negative faces) on every flip and by enforcing `max_flips`.
* **What the Steiner step does and does not guarantee.** `_insert_steiner` operates only on interior same-tag
  tetrahedra and commits an insertion only when the split followed by local flips strictly reduces the global count
  of negative faces. It resolves isolated interior non-convex configurations where 2-3, 3-2, and 4-4 flips are
  geometrically blocked, but it never inserts nodes on $\partial\Omega$ or on material interfaces and does not
  guarantee removal of negative faces constrained by curved boundaries, small dihedral angles, or locked CAD
  features; any remaining offending faces are returned in `report["stuck"]`.

## References

- L. Chen and J. Xu, "Optimal Delaunay triangulations", J. Comput. Math. 22(2), 299–308 (2004).
- I. Fried, "Bounds on the extremal eigenvalues of the finite element stiffness and mass matrices and their
  spectral condition number", J. Sound Vib. 22(4), 407–418 (1972), doi:10.1016/0022-460X(72)90452-X.
- T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover (2000),
  Ch. 9 (element eigenvalue bound of the critical time step).
- M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique", PIER 32, 65–87
  (2001).
- C. Geuzaine and J.-F. Remacle, "Gmsh: a 3-D finite element mesh generator with built-in pre- and
  post-processing facilities", Int. J. Numer. Meth. Eng. 79(11), 1309–1331 (2009), doi:10.1002/nme.2579.
- S.-W. Cheng, T. K. Dey, H. Edelsbrunner, M. A. Facello and S.-H. Teng, "Sliver exudation", J. ACM 47(5),
  883–904 (2000).
- B. Joe, "Construction of three-dimensional Delaunay triangulations using local transformations", Comput. Aided
  Geom. Des. 8(2), 123–142 (1991) (DOI to be checked).
- H. Edelsbrunner and N. R. Shah, "Incremental topological flipping works for regular triangulations",
  Algorithmica 15(3), 223–241 (1996) (DOI to be checked).
