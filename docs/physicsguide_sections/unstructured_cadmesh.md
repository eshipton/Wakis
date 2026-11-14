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

STUDY_RESULTS_BOUNDS

## 4. Raising dt: node relocation driven by the edge rates

STUDY_RESULTS_IMPROVE

## 5. Example: a muon-cooling RF cavity from STEP

STUDY_RESULTS_MUON

## 6. Limits & Failure Modes

LIMITS

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
