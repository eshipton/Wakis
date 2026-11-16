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
invoked before and after node relocation when `improve_dt(..., repair=True)`) restores local Delaunayhood.
The number of eliminated (massless) edges is reported in the study outputs (measured values: see studies/unstructured-cadmesh/results/repair.json); the per-step work
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
  accepted (measured values: see studies/unstructured-cadmesh/results/repair.json). The accuracy cost is measured (Section 4: equal or better
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
* **Boundary and coplanar-interface 2-2 flip (`_flip_22`):** when two boundary-adjacent tetrahedra $(a, c, d, p)$
  and $(b, c, d, p)$ share an interior face $(c, d, p)$ and their outer faces $(a, b, c)$ and $(a, b, d)$ lie on a
  common plane of $\partial\Omega$ sharing the boundary edge $(a, b)$, flips the boundary diagonal from $(a, b)$ to
  $(c, d)$ to form $(c, d, a, p)$ and $(c, d, b, p)$, accepted only when the two boundary triangles are coplanar and
  their union area and outward unit normals are preserved to relative tolerance $10^{-12}$. Likewise, when two
  coplanar interface triangles $(a, b, c)$ and $(a, b, d)$ separating distinct `cell_tags` are shared by an
  octahedral four-tetrahedron bipyramid $(a, b, c, p)$, $(a, b, d, p)$ on one side and $(a, b, c, q)$, $(a, b, d, q)$
  on the other, flips the interface diagonal $(a, b) \to (c, d)$ into $(c, d, a, p)$, $(c, d, b, p)$ and
  $(c, d, a, q)$, $(c, d, b, q)$ while preserving per-tag volumes and the interface triangle union to relative
  tolerance $10^{-12}$.

**Locked-node and material-interface rules:**
1. *Material interfaces:* 2-3, 3-2, and 4-4 flips are allowed only if all tetrahedra involved have the same
   `cell_tags` value; across distinct `cell_tags`, only the coplanar four-tetrahedron interface 2-2 flip and
   interface-preserving Steiner edge splits are allowed, preserving both total volume and per-tag volume to relative
   tolerance $10^{-12}$.
2. *Locked CAD nodes (`locked`):* given an optional boolean mask `locked` over mesh vertices (`True` on fixed CAD
   curves or corners, or `~movable` inside `improve_dt`), a flip is refused when it removes a boundary edge or
   boundary face whose nodes are all locked (`refused["locked"]`), and likewise for Steiner boundary-edge bisection;
   interior edges and interior faces are never refused by `locked`.

### Algorithm

`flip_to_delaunay(mesh, materials=None, locked=None, bc="pec", steiner=True, max_flips=None, verbose=False, enabled_flips=("2-3", "3-2", "4-4", "2-2"))` uses the shared `neg_faces` criterion from `wakis.unstructured.improve` (the same signed dual-length computation used by `improve._State` and `stability_report` in `wakis.unstructured.quality`):

1. Initialize a queue with all faces satisfying $H_f < -10^{-14}\,\bar{h}/\mu_0$.
2. Pop a face $f$ from the queue; if $f$ is still present in the current mesh and has $H_f < 0$, evaluate candidate
   flips in `enabled_flips` (`_flip_23` on $f$, `_flip_32` and `_flip_44` on the edges of $f$, and `_flip_22` when
   $f$ is part of a coplanar boundary or interface triangle pair).
3. Accept a candidate flip only when every replacement tetrahedron has strictly positive volume, total and per-tag
   volumes are conserved, and the flip strictly decreases the count of negative faces on the modified cavity (or
   keeps it zero while increasing the minimum signed dual length). Re-enqueue all faces of the newly created
   tetrahedra.
4. If negative faces remain after the queue empties and `steiner=True`, `flip_to_delaunay` trials edge splits via
   `split_edge_steiner` and cell-circumcentre splits (`1-4` or `2-6`) via `_insert_steiner` (falling through to
   cell/unlocked candidates when every locked-edge split is refused with `"no_projector"` and the negative-face count
   did not improve), followed by local flips, committing a candidate only if the global negative-face count strictly
   decreases.
5. When `improve_dt(..., repair=True)` (the default), `improve_dt` runs `flip_to_delaunay` $\to$ time-step-driven
   node smoothing $\to$ `flip_to_delaunay` with `locked = ~movable` and appends a `"repair"` report dictionary to
   `history`. Setting `repair=False` runs node smoothing alone.

### Limits & Failure Modes

* **Stuck faces (`report["stuck"]`).** In three dimensions, a non-Delaunay interior face $f = (c, d, e)$ shared by
  $(a, c, d, e)$ and $(b, c, d, e)$ is geometrically non-flippable by a 2-3 move when the segment $ab$ passes
  outside triangle $(c, d, e)$ (the five-vertex polyhedron is non-convex), and a 3-2 flip is unavailable unless the
  reflex edge on the boundary of the pair has degree three in the mesh (B. Joe, Construction of three-dimensional
  Delaunay triangulations using local transformations, Computer Aided Geometric Design 8(2), 123-142 (1991),
  DOI 10.1016/0167-8396(91)90038-d; H. Edelsbrunner and N. R. Shah, Incremental topological flipping works for
  regular triangulations, Algorithmica 15(3), 223-241 (1996), DOI 10.1007/BF01975867).
* **Locked curves and material interfaces.** A flip is refused when it removes a boundary edge or boundary face whose
  nodes are all locked, or when it would alter a non-planar material interface (`cell_tags` jump) or non-planar
  boundary. When all boundary edges of a boundary-adjacent sliver are locked by CAD curves or corners, neither
  boundary flips nor node relocation can alter its boundary faces.
* **3D flip-sequence non-termination risk.** Unlike two-dimensional Lawson flips, unconstrained three-dimensional
  local flips can cycle indefinitely if non-improving flips are permitted (B. Joe, Construction of three-dimensional
  Delaunay triangulations using local transformations, Computer Aided Geometric Design 8(2), 123-142 (1991),
  DOI 10.1016/0167-8396(91)90038-d; H. Edelsbrunner and N. R. Shah, Incremental topological flipping works for
  regular triangulations, Algorithmica 15(3), 223-241 (1996), DOI 10.1007/BF01975867).
  `flip_to_delaunay` avoids cycles by requiring strict local improvement in the negative-face count (or minimum
  dual length at zero negative faces) on every flip and by enforcing `max_flips`.
* **What the Steiner step does and does not guarantee.** `flip_to_delaunay` commits a Steiner split only when the split
  followed by local flips strictly reduces the global count of negative faces (`_RepairState.is_neg_face`, a superset
  of the solver guard in `UnstructuredFIT3D`). When a `split_projector` is supplied (or `split_project="auto"` in
  `improve_dt`), encroached locked CAD-curve edges are bisected and projected onto the underlying curve/surface
  (`split_edge_steiner` in `wakis.unstructured.steiner`); without a projector covering a non-planar boundary edge,
  the split is refused (`report["refused"]["no_projector"]`), and any remaining offending faces are returned in
  `report["stuck"]`.
* **Layered pillbox diagnosis (`test_step_layered_pillbox_repair_complete`).** On the STEP layered pillbox at
  $\mathrm{cpw}=10$ ($8364$ tetrahedra, $1830$ nodes, $17651$ faces, $2$ initial negative faces; recorded in
  `studies/unstructured-cadmesh/results/repair.json`), the wave-3 `imp=None` path ran
  `improve_dt(..., n_sweeps=0, repair=True)` after `gmsh` was finalized with default `project="planar"`, which locked
  every curved cylinder-wall and rim-circle node ($r = a$) and refused all fixing flips and locked-edge splits (`2`
  stuck faces, `refused={"locked": 4}`). With post-gmsh default `split_project="auto"` (`feature_curve_projector`),
  `imp=None` still leaves `2` stuck faces (`refused={"no_projector": 16, "locked": 4}`) because the blocking locked
  edges on those two faces are smooth curved cylinder-wall edges ($r = a$), not sharp rim curves; `imp=None` reaches
  `0` negative faces only when supplied with an analytic pillbox projector (`_pillbox_split_projector` in
  `test_step_layered_pillbox_repair_complete`, committing $2$ projected Steiner nodes on locked cylinder-wall edges:
  `steiner_nodes = 2`, `steiner_on_locked = 2`, `steiner_projected = 2`). By contrast, `imp=True` (`mesh_cad` +
  `CadSplitProjector`) runs while `gmsh` is live so smooth cylinder-wall nodes are unlocked (`dim=2`) and reaches `0`
  negative faces via CAD surface projection (`steiner_nodes = 2`, `steiner_on_locked = 0`, `steiner_projected = 2`),
  with every boundary Steiner node within $\tau$ of $r = a$. In the recorded wave-3
  `studies/unstructured-cadmesh/results/repair.json`, `repair_study.py`'s `flip_only` path (`locked=None`) and
  `improve=True` path (`n_sweeps=10`, `cad_projector`) reached `0` negative faces by committing $2$ unprojected chord
  Steiner nodes on cylinder-wall boundary edges (`1832` nodes, `8372` tetrahedra, `refused={"boundary": 3}`).
* **Repair can lower `dt_max`.** In measured gmsh cases the limiting element after repair has a smaller radius ratio;
  the measured values are in `studies/unstructured-cadmesh/results/repair.json` (key `dt_gain_vs_repair_false`).

### Steiner points on locked CAD curves

#### Encroachment criterion and conforming-Delaunay edge splitting

When an interior face $f$ still has negative Voronoi dual length ($H_f < 0$) after the bistellar flip queue
(`2-3`, `3-2`, `4-4`, `2-2`) empties because every flip that would restore local Delaunayhood removes a constrained
CAD-curve or boundary edge $e = (a, b)$, cell-interior (`1-4`) and face-interior (`2-6`) Steiner splits leave the
constrained segment $e$ intact so the child faces incident to $e$ remain non-Delaunay. In conforming-Delaunay
refinement (Ruppert 1995; Shewchuk 1998; Cheng, Dey & Shewchuk 2012), the remedy is to split the encroached
constrained segment itself. Let $\mathbf{m} = \tfrac{1}{2}(\mathbf{x}_a + \mathbf{x}_b)$ be the midpoint of $e$ and
$L = \|\mathbf{x}_b - \mathbf{x}_a\|$ its chord length. An opposite vertex $\mathbf{x}$ in the tetrahedral edge star
$\mathcal{S}(e) = \{T \ni e\}$ encroaches upon $e$ when it lies strictly inside the diametral sphere of $e$
(`is_edge_encroached` in `wakis.unstructured.steiner`):

$$\|\mathbf{x} - \mathbf{m}\| < \frac{L}{2} \iff (\mathbf{x}_a - \mathbf{x})\cdot(\mathbf{x}_b - \mathbf{x}) = \|\mathbf{x} - \mathbf{m}\|^2 - \frac{L^2}{4} < 0 \iff \angle a x b > 90^\circ.$$

An obtuse apex angle $\angle a x b > 90^\circ$ across $e$ pushes the circumcentre of tetrahedron $(a, b, x, y)$ across
the face opposite $e$, producing $H_f < 0$. Bisecting $e$ at a point $\mathbf{p}$ on the underlying CAD curve replaces
$e$ by two sub-edges $(a, p)$ and $(p, b)$ of length $\approx L/2$, halving the diametral sphere radius so $\mathbf{x}$
is no longer inside the diametral spheres of the sub-edges and the subsequent local `2-3` / `3-2` flips on the split
cavity restore $H_f \ge 0$.

For each face $f$ still satisfying `_RepairState.is_neg_face(f)` after the flip queue empties,
`candidate_edges_for_face` constructs the candidate list $E_f$ ordered as:
1. Locked edges of $f$,
2. Locked edges of the two tetrahedra incident to $f$ not on $f$,
3. Unlocked edges of $f$ and its two incident tetrahedra,

and within each group orders encroached edges first ($\|\mathbf{x} - \mathbf{m}\| < L/2$), then by length $L$
descending.

#### Split projectors and 3-point circle conditioning

A split projector $P$ (`normalize_split_projector` in `wakis.unstructured.steiner`) implements `P.project(a, b, x)`
returning a projected 3D point $\mathbf{p} \in \mathbb{R}^3$ on the CAD curve or surface shared by vertices $a$ and $b$
(or `None` when $(a, b)$ is not on a reconstructible entity), together with `P.register(new_node, a, b)` and
`P.unregister(new_node, a, b)` so sub-edges $(a, \text{new})$ and $(\text{new}, b)$ inherit the parent entity during
recursive splits:

* **OpenCASCADE CAD projector (`CadSplitProjector` in `wakis.unstructured.cad`):** constructed while `gmsh` is live and
  attached as `project.split` on `cad_projector(gmsh, node_tags, scale)`. Each mesh node is mapped to its containing
  1D CAD curves and 2D CAD surfaces via `gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)`. When $(a, b)$
  shares a common 1D curve (preferred) or 2D surface, `CadSplitProjector.project(a, b, x)` returns
  `gmsh.model.getClosestPoint(dim, tag, x)`.
* **Discrete feature-curve fallback (`feature_curve_projector` and `FeatureCurveProjector` in `wakis.unstructured.steiner`):**
  used when `improve_dt(..., split_project="auto")` runs without a live `gmsh` model. A boundary or material-interface
  edge is classified as a sharp feature edge when both endpoints are locked and either (i) the dihedral angle between
  its two incident boundary faces is $\ge 45^\circ$ ($\mathbf{n}_1 \cdot \mathbf{n}_2 \le \cos 45^\circ = 1/\sqrt{2}$,
  which separates $90^\circ$ cylinder rims from smooth curved-wall facet transitions of $\sim h/R \approx 5.7^\circ$ at
  $\mathrm{cpw} = 10$, specifically $5.73^\circ$ for $h = 0.01\text{ m}$ and $R = 0.1\text{ m}$) or (ii) $\ge 3$
  boundary/interface faces meet at the edge (a multi-material triple line such as the dielectric-vacuum rim circle
  $r = a, z = t$). Feature edges are chained across degree-2 vertices whose unit tangents satisfy
  $\hat{\mathbf{t}}_1 \cdot \hat{\mathbf{t}}_2 > \cos 45^\circ$. For an edge $(a, b)$ with a chain neighbour
  $c \in \{p, q\}$, `circle_from_3pts` fits the unique 3D circle through $(\mathbf{x}_a, \mathbf{x}_b, \mathbf{x}_c)$
  (or falls back to the straight chord when collinear within `collinear_tol = 1e-12`, i.e., $10^{-12}$) and projects
  the midpoint $\mathbf{m}$ onto the circular arc:
  $$\mathbf{p} = \mathbf{c}_{\mathrm{circ}} + R\,\frac{\mathbf{m} - \mathbf{c}_{\mathrm{circ}}}{\|\mathbf{m} - \mathbf{c}_{\mathrm{circ}}\|}.$$
  Because the circumcentre $\mathbf{c}_{\mathrm{circ}}$ is obtained by inverting the $2 \times 2$ Gram matrix of two
  chords of length $\ge L_{\min}$ spanning a small sagitta $\sim L_{\min}^2 / (8R)$, the condition number of the
  3-point circle fit scales as $(R / L_{\min})^2$, giving a floating-point radial accuracy bound
  $\|\mathbf{p} - \mathbf{p}_{\mathrm{exact}}\| \le 100\,\varepsilon_{\mathrm{mach}}\,(R / L_{\min})^2 R$ (verified in
  `test_t2_feature_curve_projector_circle_and_line` on a 31-node jittered circle of radius $R = 0.1\text{ m}$ and
  $L_{\min} = 1.5479 \times 10^{-2}\text{ m}$: maximum error $4.163 \times 10^{-17}\text{ m} \le 9.267 \times 10^{-14}\text{ m}$,
  and $1.590 \times 10^{-17}\text{ m}$ on a straight line).

For an unlocked interior edge, no projection is needed ($\mathbf{p} = \mathbf{m}$). For an unlocked boundary or
interface edge, `split_edge_steiner` projects $\mathbf{m}$ when $P$ can; when $P$ returns `None` and the incident
boundary/interface faces are not coplanar (`is_coplanar_constraint_edge`), the split is refused with
`report["refused"]["no_projector"]` because an unprojected chord midpoint on a curved boundary introduces a sagitta
geometry error $R - \sqrt{R^2 - (L/2)^2} \approx L^2 / (8R)$ ($2.747621 \times 10^{-3}\text{ m}$ on the 13-node
cylinder rim in `test_t1_synthetic_rim_locked_edge_steiner`).

#### Exact boundary-flux volume identity for projected splits

In `split_edge_steiner`, every positively oriented tetrahedron $T = (a, b, p, q)$ in the edge star $\mathcal{S}(e)$ is
split into two child tetrahedra $T_a = (a, s, p, q)$ and $T_b = (s, b, p, q)$ at $\mathbf{x}_s = \mathbf{p}$ inheriting
$T$'s material tag, with both child volumes required to exceed $\max(\mathrm{min\_vol}, 10^{-6} V_T)$. When
$\mathbf{p} = \mathbf{m}$ lies on the chord $(a, b)$, $V(T_a) + V(T_b) = V(T)$ holds per tetrahedron. When $\mathbf{p}$
is projected off the chord onto a curved CAD boundary ($\mathbf{p} \ne \mathbf{m}$), individual tetrahedron volumes
change, but the net volume change over the sub-star $\mathcal{S}_t(e) = \{T \in \mathcal{S}(e) : \mathrm{tag}(T) = t\}$
of each material tag $t$ obeys an exact divergence identity (`star_volume_change_identity` in
`wakis.unstructured.steiner`). Because $(\mathbf{p} - \mathbf{x}_a) - (\mathbf{p} - \mathbf{x}_b) = \mathbf{x}_b - \mathbf{x}_a$,
for each $T = (a, b, p, q) \in \mathcal{S}_t(e)$ the signed volume change is `(1/6) det(x_b − x_a, x_q − x_p, p − m)`:

$$\Delta V_T = V(T_a) + V(T_b) - V(T) = \frac{1}{6}\det(\mathbf{x}_b - \mathbf{x}_a,\; \mathbf{x}_q - \mathbf{x}_p,\; \mathbf{p} - \mathbf{m}).$$

Decomposing $T$ relative to the chord midpoint $\mathbf{m}$ shows that shifting the split vertex by $\mathbf{p} - \mathbf{m}$
sweeps out signed tetrahedra over the two triangular faces $F_x = (a, b, x)$ and $F_y = (a, b, y)$ incident to $e$.
When $F = (a, b, x)$ is shared by two tetrahedra of the same tag $t$ in $\mathcal{S}_t(e)$, their outward unit normals
$\mathbf{n}_{F,T_1} = -\mathbf{n}_{F,T_2}$ cancel identically; only triangles $F = (a, b, x) \subset \partial\mathcal{S}_t(e)$
on the outer domain boundary or on a material interface between distinct tags survive:

$$\Delta V_t = \sum_{T \in \mathcal{S}_t(e)} \Delta V_T = \sum_{F = (a,b,x) \subset \partial\mathcal{S}_t(e)} \frac{1}{3}\,A_F\,\mathbf{n}_{F,t}\cdot(\mathbf{p} - \mathbf{m}),$$

where $A_F$ is the area of $F = (a, b, x)$, $\mathbf{n}_{F,t}$ is its unit normal pointing outward from tag $t$, and
$\mathbf{n}_{F,t}\cdot(\mathbf{p} - \mathbf{m})$ is the signed altitude of the added tetrahedron $(a, b, x, p)$ above
the plane of $F$ (since $\mathbf{m} = \tfrac{1}{2}(\mathbf{x}_a + \mathbf{x}_b)$ lies in the plane of $F$). Summing over
all tags cancels the internal material-interface triangles with opposite outward normals, leaving only outer-boundary
triangles $F \subset \partial\Omega \cap \partial\mathcal{S}(e)$ for the total volume change $\Delta V = \sum_t \Delta V_t$.
`split_edge_steiner` enforces this identity both in total and per tag to relative tolerance $10^{-12} V_{\mathcal{S}(e)}$
(measured error $1.146 \times 10^{-15}$ in `test_t1_synthetic_rim_locked_edge_steiner`).

#### Stop criterion and time-step protection caps

* **Stop criterion:** `flip_to_delaunay` uses `_RepairState.is_neg_face` ($H_f < 0$ on a face coupling at least one free
  edge under `_default_free`), which is a superset of the `UnstructuredFIT3D` solver guard (`Mnu_diag < 0` on faces
  coupling at least one free, non-massless edge), and commits a trial split (plus local flip queue) only when the
  global negative-face count strictly decreases.
* **Caps (`max_steiner`, `max_steiner_depth`, `min_edge_ratio`):** `max_steiner` defaults to
  `max(32, 4 * max(neg_before, 1))`, bounding the total number of committed Steiner nodes;
  `max_steiner_depth=2` allows at most one nested sub-edge bisection inside a checkpointed trial before rolling back;
  and `min_edge_ratio=0.2` refuses any split whose sub-edge is shorter than $0.2\,L_0$ (`refused["min_edge"]`), preventing
  arbitrarily short edges that would collapse the explicit CFL limit $\Delta t_{\max} \propto L_{\min}$.

#### Limits & Failure Modes (Steiner curve refinement)

1. **Acute CAD input angles $< 90^\circ$:** in 3D Delaunay refinement, constrained segments or facets meeting at acute
   CAD angles $< 90^\circ$ can trigger mutually encroaching ping-pong splits that do not terminate without protective
   collars (Shewchuk 1998; Cheng, Dey & Shewchuk 2012, Ch. 6 and Ch. 10). Here, refinement is strictly bounded by
   `max_steiner`, `max_steiner_depth=2`, the strict decrease requirement on `len(state.neg_set)`, and the sub-edge floor
   `min_edge_ratio=0.2`; if an acute corner exhausts these caps, the remaining face stays in `report["stuck"]`.
2. **Splitting an edge halves its length and can lower $\Delta t_{\max}$:** bisecting a locked boundary edge of length
   $L$ creates two sub-edges of length $\approx L/2$, which can increase the local Courant eigenvalue and reduce
   $\Delta t_{\max}$ (on the full gmsh benchmark suite in `studies/unstructured-cadmesh/results/repair.json`,
   `dt_gain_vs_repair_false` ranges from $0.9447$ to $1.0536$).
3. **Post-gmsh feature-curve limitation without an explicit projector:** when `improve_dt` is called after `gmsh` has
   been finalized (`split_project="auto"` without `CadSplitProjector`), `feature_curve_projector` reconstructs only sharp
   feature curves with dihedral angle $\ge 45^\circ$ (or multi-material triple junctions). Smooth curved-surface edges
   whose endpoints happen to be locked by `planar_projector` (such as curved cylinder-wall edges on the layered pillbox,
   where `imp=None` with default `split_project="auto"` leaves `2` stuck faces with `refused={"no_projector": 16, "locked": 4}`)
   are not sharp feature curves and are refused with `report["refused"]["no_projector"]`. Consequently, `imp=None` on
   the layered pillbox reaches `0` negative faces only with an analytic pillbox projector, whereas
   `mesh_cad(..., improve=True)` (which runs while `gmsh` is live and passes `CadSplitProjector` covering both 1D curves
   and 2D surfaces) reaches `0` negative faces via surface projection (`steiner_on_locked = 0`, `steiner_projected = 2`).
4. **3-point circle accuracy on general 3D splines:** `FeatureCurveProjector` fits a local 3-point circle (`circle_from_3pts`),
   which is exact to floating-point conditioning $\mathcal{O}(\varepsilon_{\mathrm{mach}}(R/L_{\min})^2 R)$ only for
   circular arcs and straight lines; on general non-circular 3D splines with non-constant curvature $\kappa'(s) \ne 0$,
   a 3-point osculating circle has $\mathcal{O}(h^3)$ geometric deviation without `gmsh`, whereas `CadSplitProjector` is
   exact to the OpenCASCADE kernel tolerance.

### Measured results (cpw 10)

Source: `studies/unstructured-cadmesh/results/repair.json` (commit 6d17729b78, produced by `repair_study.py` on Cloud Batch). Negative-face counts use the solver guard criterion (`Mnu_diag < 0` on faces coupled to free edges) and are shown as negative / total faces.

| case | gmsh optimizer | negative / faces before | after repair=False | after repair=True | dt gain (T/F) | lowest-mode error (F) | lowest-mode error (T) | limiting radius ratio (T) |
|---|---|---|---|---|---|---|---|---|
| pillbox | none | 0 / 4633 | 0 / 4633 | 0 / 4633 | 1.0000 | -2.779e-03 | -2.779e-03 | 0.842 |
| layered | none | 2 / 17651 | 0 / 17651 | 0 / 17669 | 1.0220 | -4.990e-03 | -4.677e-03 | 0.821 |
| muon | none | 0 / 4064 | 0 / 4064 | 0 / 4064 | 1.0000 | +2.113e-02 | +2.113e-02 | 0.738 |
| pillbox | gmsh | 48 / 4531 | 0 / 4531 | 0 / 4629 | 0.9894 | -3.378e-03 | -2.435e-03 | 0.744 |
| layered | gmsh | 218 / 17315 | 4 / 17315 | 0 / 17667 | 0.9447 | -5.257e-03 | -4.918e-03 | 0.758 |
| muon | gmsh | 39 / 3992 | 1 / 3992 | 0 / 4064 | 1.0536 | +2.154e-02 | +2.113e-02 | 0.738 |

In the study, `repair=True` leaves 0 negative faces in all six runs. Step A0 diagnosis (`studies/unstructured-cadmesh/steiner_study.py --quick`) and `studies/unstructured-cadmesh/results/repair.json` explain why the wave-3 `layered` (`none`, $8364$ tetrahedra, $1830$ nodes, $17651$ faces, $2$ initial negative faces) study rows reached 0 while `test_step_layered_pillbox_repair_complete` (`imp=None`) left 2 stuck faces: in `test_step_layered_pillbox_repair_complete`, `imp=None` ran `improve_dt(..., n_sweeps=0, repair=True)` after `gmsh` was finalized using default `project="planar"`, which locked every node on the curved cylinder wall ($r = a$) and rim circles and refused all boundary-edge splits (`"locked": 4`). Post-gmsh default `split_project="auto"` (`feature_curve_projector`) still leaves `2` stuck faces (`refused={"no_projector": 16, "locked": 4}`) because the blocking locked edges on those faces are curved cylinder-wall edges rather than sharp rim curves, so `imp=None` reaches `0` negative faces only with an analytic pillbox projector (`_pillbox_split_projector` in the test, `steiner_on_locked = 2`, `steiner_projected = 2`). By contrast, in wave-3 (`studies/unstructured-cadmesh/results/repair.json`), `repair_study.py`'s `flip_only` path passed `locked=None` and its `improve=True` path used `cad_projector` (unlocking smooth cylinder-wall surface nodes and running `n_sweeps=10`), committing 2 unprojected chord Steiner splits (`1832` nodes, `8372` tetrahedra, `refused={"boundary": 3}`), whereas in wave-4 `imp=True` (`mesh_cad` + `CadSplitProjector`) reaches `0` negative faces via CAD surface projection (`steiner_on_locked = 0`, `steiner_projected = 2`) and refuses unprojected chord splits on curved boundaries (`"no_projector"`). Repair can lower `dt_max`: the gmsh-optimizer rows have dt gain below 1, so the limiting element after repair has a smaller radius ratio. No claim that repair raises dt in general is made.

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
- B. Joe, Construction of three-dimensional Delaunay triangulations using local transformations,
  Computer Aided Geometric Design 8(2), 123-142 (1991), DOI 10.1016/0167-8396(91)90038-d
- H. Edelsbrunner and N. R. Shah, Incremental topological flipping works for regular triangulations,
  Algorithmica 15(3), 223-241 (1996), DOI 10.1007/BF01975867
- J. Ruppert, "A Delaunay refinement algorithm for quality 2-dimensional mesh generation", J. Algorithms 18(3),
  548–585 (1995), doi:10.1006/jagm.1995.1021.
- J. R. Shewchuk, "Tetrahedral mesh generation by Delaunay refinement", Proc. 14th ACM Symp. Comput. Geom. (SoCG),
  86–95 (1998), doi:10.1145/276884.276894.
- S.-W. Cheng, T. K. Dey and J. R. Shewchuk, *Delaunay Mesh Generation*, CRC Press (2012).
- H. Si, "TetGen, a Delaunay-based quality tetrahedral mesh generator", ACM Trans. Math. Softw. 41(2),
  11:1–11:36 (2015), doi:10.1145/2629697.
