### 🧱 Prism FIT: extruded triangle meshes (`PrismFIT3D`)

Planar devices are layered in $z$ (substrate, core, cladding, metal films) and have arbitrary shapes in the plane: waveguides, rings, metasurface pillars, patch antennas, printed circuits. The Cartesian grid staircases or sub-pixel-averages the curved in-plane boundaries, and a tetrahedral mesh resolves them but pays for the unstructured third direction. `wakis.unstructured.PrismFIT3D` takes the tensor product of a 2D triangle mesh with a 1D, possibly non-uniform, $z$ grid. It is exact-topology FIT, with a conforming in-plane geometry, a structured $z$ direction and diagonal material matrices. The idea of an unstructured planar grid times layers goes back to the planar generalized Yee algorithm of Gedney and Lansing for MMICs [Gedney & Lansing, 1995; Gedney, Lansing & Rascoe, 1996]. The generalized Yee algorithm uses a non-orthogonal dual with interpolation; here the dual is the orthogonal Voronoi one, as in [Hano & Itoh, 1996] and the Delaunay–Voronoi FIT of [Clemens & Weiland, 2001].

**Cell complex.** Let the triangle mesh have nodes $n$, edges $e$ and triangles $t$, with incidences $\mathbf G_2$ (edge–node) and $\mathbf C_2$ (triangle–edge), and let the levels be $z_0<\dots<z_K$. The primal objects and their fields are:

| primal object | field | dual object (Hodge) |
| --- | --- | --- |
| horizontal edge $(e, l)$ | $e_h$ [V] | 2D dual edge $h_e$ × $\Delta z^*_l$ |
| vertical edge $(n, k+\tfrac12)$ | $e_v$ [V] | 2D Voronoi cell $A_n$ × (layer) |
| triangle face $(t, l)$ | $b_z$ [Wb] | vertical segment $\Delta z^*_l$ |
| vertical quad $(e, k+\tfrac12)$ | $b_q$ [Wb] | 2D dual edge $h_e$ at mid-layer |
| prism $(t, k+\tfrac12)$ | material | — |

Here $\Delta z^*_l$ is the dual spacing at level $l$, the half-sum of the adjacent layers. The curl is a Kronecker assembly:

$$
\mathbf C = \begin{pmatrix} \mathbf C_2\otimes\mathbf I_L & 0\\ \mathbf I_{E}\otimes\mathbf S & \mathbf G_2\otimes\mathbf I_K\end{pmatrix},\qquad
\mathbf G = \begin{pmatrix} \mathbf G_2\otimes\mathbf I_L\\ \mathbf I_N\otimes(-\mathbf S)\end{pmatrix},
$$

where $\mathbf S$ ($K\times L$) is the level difference, $S_{k,k}=1$ and $S_{k,k+1}=-1$, with wrap-around for periodic $z$. Then $\mathbf C\mathbf G = 0$ and $\mathbf D\mathbf C = 0$ hold exactly, because $\mathbf C_2\mathbf G_2 = 0$ and $\mathbf S + (-\mathbf S) = 0$. The time step never forms $\mathbf C$: it is two sparse products with $K$ right-hand sides ($\mathbf C_2 e_h$ and $\mathbf G_2 e_v$ on $(n_\text{2D}\times K)$ arrays) and two contiguous differences along $z$. The $z$ direction is as cheap as in a Cartesian code, and the sparse index overhead is amortized over the layers. The step costs 9–10 ns per unknown on one core, against 16–21 ns for the 2D triangle solver and 27–32 ns for `SolverFIT3D` on the same kind of machine (measured on Cloud Batch n2 machines, see the study).

**Hodge matrices.** The 2D Voronoi dual edge of $e$ is perpendicular to $e$, and the $z$ dual is perpendicular to the levels. Every dual edge is therefore parallel to its primal face normal and every dual face perpendicular to its primal edge, so the Hodge matrices are diagonal. Each prism holds one material and every interface is a union of primal faces. The capacitance of an edge sums $\varepsilon$ over the parts of its dual face in each prism (parallel averaging). This is exact for the tangential $E$ that a conforming edge carries, including edges on the surface of the rod and on the levels between layers. The reluctance of a face sums $h/\mu$ over the parts of its dual edge (series averaging), which is exact for the normal $B$:

$$
C_{e,l} = \frac{\varepsilon_0}{L_e}\sum_{t}\sum_{k\in\{l-1,l\}} \varepsilon_{t,k}\,h_{e,t}\,\frac{\Delta z_k}{2},\qquad
C_{n,k} = \frac{\varepsilon_0}{\Delta z_k}\sum_t \varepsilon_{t,k}A_{n,t},
$$
$$
\nu_{t,l} = \frac{1}{A_t}\sum_{k\in\{l-1,l\}}\frac{\Delta z_k/2}{\mu_0\mu_{t,k}},\qquad
\nu_{e,k} = \frac{1}{L_e\,\Delta z_k}\sum_t\frac{h_{e,t}}{\mu_0\mu_{t,k}} .
$$

**Nearly cocircular triangles.** A horizontal edge has a capacitance proportional to its 2D dual length, $h_e = \tfrac{L_e}{2}(\cot\alpha_1+\cot\alpha_2)$. A right angle opposite an edge makes $h_e\to 0$, and the in-plane curl then gives that edge a frequency $\omega^2\sim L_e/(A_t h_e)\to\infty$. This does not happen in the 2D TE solver, whose $H_z$ lives on the nodes and where a short dual edge only weakens a stiffness. On the ring/lattice meshes of `disk_mesh`, about 1.5 % of the edges have $h_e/L_e<0.05$; they cut the stable time step by 5–14× (prism $c\,\Delta t/h = 0.04$ instead of 0.36 on an ideal hexagonal lattice). Two remedies are combined:

1. ODT smoothing (`wakis.unstructured.smoothing.odt_smooth`, Chen & Xu 2004). Interface and boundary nodes stay fixed, and free nodes move to the area-weighted mean of their triangles' circumcentres, which drives the mesh towards equilateral triangles.
2. Merging (`PrismMesh(merge_tol=...)`, `PolygonComplex2D`). The two triangles of a non-interface edge with $h_e\le$ `merge_tol` $L_e$ become one polygonal cell. For $h_e = 0$ the quadrilateral is cocircular, its Voronoi dual node is unique, and the complex stays exactly orthogonal. For small $h_e$ the merge is an $O(h_e/L_e)$ perturbation of the dual. The cell–edge incidence of the polygon is the sum of the two triangle rows, so the removed edge cancels and $\mathbf C\mathbf G = 0$ still holds.

On the rod-cavity mesh at $h = a/10$, `merge_tol = 0.15` after smoothing removes 10 of 1,000 edges. It raises $c\Delta t/h$ from 0.04 to 0.31 and changes the RMS error of 20 resonances from 0.82 % to 0.84 %. A remaining non-positive Hodge entry (for example an interface edge with an obtuse high-$\varepsilon$ side, which cannot be merged) is clamped to a dual length of `dual_floor` $L_e$, and the clamped entries are counted in `n_floored`.

**Boundaries and sources.**

- *$z$ ends:* PEC (tangential $e_h=0$ on the end levels), PMC (natural, half dual cells) or periodic.
- *Lateral boundary:* PEC, PMC or the first-order Silver–Müller ABC. The ABC is a conductance $l_\text{surf}/(\eta L)$ on the boundary edges: $\Delta z^*$ for horizontal edges, half the adjacent boundary edges for vertical ones.
- *Lateral absorber:* the matched lossy annulus of the 2D solver, $\sigma_e/\varepsilon = \sigma_m/\mu = s(r)$, Crank–Nicolson averaged.
- *z PML:* a Berenger split field. The $z$-derivative parts of $e_h$ and $b_q$ (the only fields with a $\partial_z$) decay at $s(z)$ in the PML layers. Because prism edges are either in-plane or vertical, the uniaxial stretching needs no tensor coupling. The matched condition holds in any medium, so slabs and waveguides may run into the PML.
- *Sources:* point dipoles on the prism Whitney functions (2D Whitney times linear in $z$; $\lambda_n/\Delta z$ for vertical edges) and distributed currents.
- *Lumped ports:* a Norton resistor block on vertical edges, semi-implicit and stable for any $R$.
- *Dispersive media:* the ADEs of `wakis.dispersive`, through the same local capacitive update as the triangle/tetrahedron solvers, with the vacuum capacitance of the dispersive part of each dual face.
- *Monitors and GPU:* running DFTs, energy, $\mathbf D\mathbf b$ and a shift-invert eigensolver with the gradient null space projected out. `use_gpu=True` runs the step with CuPy sparse products; it is lossless or lossy, with absorbers and PML, and has no dispersive media yet.

**Exactness checks.** The discrete energy $W^n = \tfrac12 e^n\mathbf M_\varepsilon e^n + \tfrac12 b^{n-1/2}\mathbf M_\nu b^{n+1/2}$ is constant to $10^{-16}$, and $\mathbf D b = 0$ to round-off. One periodic layer reproduces the eigenfrequencies of the 2D triangle TM solver to $10^{-9}$: $e_v$ is $E_z$ on the nodes and $b_q$ the in-plane flux of the edges. The in-plane horizontal edges with $b_z$ on the triangles form the *edge-based* 2D TE scheme. It is the dual of the solver's node-based TE ($H_z$ on nodes) and exact for the tangential $E$ at interfaces.

**Validation: dielectric rod in a PEC cavity.** A rod (radius $b$, $\varepsilon$) spanning a closed circular PEC cavity (radius $a$, height $h$) is the 3D analogue of the shared cylinder benchmark. The relation $k^2 = k_t^2 + (p\pi/h)^2$ holds only for homogeneous filling or for $p = 0$, where only TM modes exist. For $p\ge1$ the rod couples $E_z$ and $H_z$ into hybrid HE/EH modes. Their exact wavenumbers solve the continuity of $E_\varphi$ and $H_\varphi$ at $r = b$ with $E_z(a) = E_\varphi(a) = 0$:

$$
\Big(\frac{\beta n}{b}\Big)^2\Big(\frac1{k_1^2}-\frac1{k_2^2}\Big)^2
= k^2\Big(\frac{P}{k_1^2}-\frac{Q_H}{k_2^2}\Big)\Big(\frac{\varepsilon P}{k_1^2}-\frac{Q_E}{k_2^2}\Big),
$$

with $\beta = p\pi/h$, $k_1^2 = \varepsilon k^2-\beta^2$ and $k_2^2 = k^2-\beta^2$. $P$ is the logarithmic derivative of $J_n(k_1 r)$ at $b$; $Q_E$ and $Q_H$ are those of the outer radial functions $J_n(k_2r)Y_n(k_2a)-Y_n(k_2r)J_n(k_2a)$ and $\partial_a[\,\cdot\,]$. This is `wakis.unstructured.analytic.rod_cavity_hybrid`, evaluated in a pole-free form that continues to modified Bessel functions for $k_i^2<0$. It reproduces the empty cavity to $10^{-14}$ and `rod_in_cavity` (TM) for $p = 0$. The naive $k_t^2+\beta^2$ list is off by more than 2 % RMS for $\varepsilon = 4$. With $\varepsilon = 4$, $b = a/2$ and $h = a$, the prism solver's 20 lowest modes converge at second order: 0.75 % RMS at $h = a/10$ and 0.38 % at $a/14$, observed order 1.93. Series-averaging the interface edges (the mutation test) is 1.6–1.75× worse.

**Limits & failure modes.**

- *Extrusion only.* Geometry must be $z$-invariant within each layer: vertical sidewalls. Sloped sidewalls, spheres and general 3D shapes need tetrahedra or the Cartesian conformal solver. Layer boundaries must be levels. A thin layer forces a thin $\Delta z$ everywhere in the plane, and $\Delta t$ shrinks with it, since the leapfrog time step depends on the smallest $\Delta z$ and the smallest in-plane dual edges.
- *2D mesh quality governs $\Delta t$.* Without smoothing and merging, near-right angles collapse the time step (above). Merging perturbs the dual by $O(h_e/L_e)$, so `merge_tol` above ~0.2 starts to cost accuracy (1.05 % RMS vs 0.75 % at 0.2 on the rod cavity). Interface edges are never merged.
- *First-order lateral absorber.* The matched isotropic annulus plus ABC is not a PML. It reflects several percent at low $kr$ (as in 2D), so open problems need a generous absorber or the closed-box/resonance formulation. Only the $z$ direction has a true (split-field) PML, and the split-field PML has the known weak instabilities of Berenger's formulation for long runs [Bécache & Joly, 2002].
- *Sources and probes on merged polygons* lose the removed edge's contribution: an error of order the merge tolerance at that point only.
- *No Bloch phase in $z$ yet* (periodic only at $k_z = 0$). No dispersive media on the GPU or inside the PML.
- *Timing claims* compare one-core implementations: the prism step is faster per unknown than `SolverFIT3D` partly because of implementation (array layout), not only the method. The study reports work (unknowns × steps) separately.
