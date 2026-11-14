### 🧩 Hybrid Cartesian / tetrahedral FIT: `wakis.hybrid` (milestone 1)

`wakis.hybrid.HybridFIT3D` keeps a Cartesian FIT grid everywhere and replaces an axis-aligned box of whole Cartesian cells by a tetrahedral FIT mesh, so that curved or small geometry is meshed with tetrahedra only where it is. The two grids are coupled through the tangential trace of the hexahedral Nédélec element on the box surface, using the same hanging-edge construction as the 3D subgrids (`wakis.subgrid3d`, [subgrid3d.md](subgrid3d.md)). The result is one discrete Maxwell system with symmetric positive-definite (SPD) mass matrices. Energy conservation, the stability bound and the exact transpose of a step therefore all follow from the algebra and do not depend on the mesh. Milestone 1 is a CPU (NumPy/SciPy) prototype: one box patch, vacuum or per-tag isotropic $\varepsilon,\mu$ and PEC obstacles in the tetrahedra, per-cell $\varepsilon,\mu$ and PEC cells in the Cartesian grid, PEC/PMC outer walls, soft edge currents and probes, and a frequency-domain solve with complex coordinate stretching.

#### 1. The options for the interface, and the choice

The Cartesian side is the lowest-order hexahedral edge element with trapezoidal (row-sum) mass lumping, which is exactly FIT/Yee. Its tangential trace on a cell face is the bilinear Nédélec field of the four face edges. The tetrahedral side is Whitney 1-forms, with triangular faces. Three ways to make the two meet:

| Option | Idea | For | Against |
|---|---|---|---|
| (a) Pyramid transition layer (Rylander & Bondeson 2000, 2002) | one layer of pyramids whose square bases match the hexahedra and whose triangles match the tetrahedra; one global FEM with mass lumping in the hexahedra | conforming in $H(\mathrm{curl})$; proven stable FEM–FDTD hybrid | needs pyramid edge elements (rational basis functions) and their mass; a structured layer must be generated; Rylander & Bondeson step the FEM part implicitly (Newmark) |
| (b) Free diagonal edges | the diagonals of the split squares become unknowns; the outer hexahedron becomes a 7-faced polyhedron | no constraint | the outer cell needs a non-diagonal, non-orthogonal Hodge for the diagonal edge; the FIT structure of the Cartesian side is lost |
| (c) Hanging tetrahedral surface edges **(chosen)** | every tetrahedral edge on the box surface is the line integral of the Cartesian trace of its square | the Cartesian grid is untouched; one sparse prolongation $\mathbf P$; the subgrid machinery (energy, transpose, Lanczos) carries over unchanged; the tetrahedral surface may refine the squares arbitrarily | the tetrahedral tangential field on the interface is restricted to the Cartesian trace space (the patch surface cannot carry finer tangential detail than the Cartesian squares); the coupling is weak (de Rham-consistent), not $H(\mathrm{curl})$-conforming |

Edelvik & Ledfelt (2000) build an explicit FDTD–FEM hybrid on the same "FDTD = lumped hexahedral FEM" observation, with a hexahedral overlap region in the FEM grid. Option (c) keeps a single mesh with no overlap.

#### 2. The discrete system

**Unknowns.** Cartesian edge voltages and face fluxes of the active cells, i.e. the cells outside the box (`CartesianBlock` of `wakis.subgrid3d`), followed by tetrahedral edge voltages and face fluxes (`SimplexMesh`). The Cartesian edges on the box surface are free, and they are the masters of the interface. The tetrahedral edges on the box surface are hanging. Cartesian edges strictly inside the box belong to no active cell and are dead. Tetrahedral edges on other boundary faces (holes) and on PEC tetrahedra are fixed at zero (PEC).

**The trace.** Let a box face have normal $a$ and tangential axes $(b,c)$, and let the Cartesian square be $[u_0,u_1]\times[v_0,v_1]$ with edge voltages $e_{b0}$ (at $v=v_0$), $e_{b1}$, $e_{c0}$ (at $u=u_0$), $e_{c1}$. The tangential Nédélec field is

```math
E_b(v)=\frac{e_{b0}(v_1-v)+e_{b1}(v-v_0)}{h_u h_v},\qquad
E_c(u)=\frac{e_{c0}(u_1-u)+e_{c1}(u-u_0)}{h_u h_v}.
```

Along any straight segment $p\to q$ inside the square, the integrand is linear, so the midpoint rule $(u_m,v_m)$ is exact:

```math
e_{pq}=\frac{\Delta u}{h_u}\Big[(1-t_v)e_{b0}+t_ve_{b1}\Big]+\frac{\Delta v}{h_v}\Big[(1-t_u)e_{c0}+t_ue_{c1}\Big],\quad t_u=\frac{u_m-u_0}{h_u},\ t_v=\frac{v_m-v_0}{h_v}.
```

`box_trace` assembles these rows. A tetrahedral edge that coincides with a Cartesian edge (or with a piece of one) gets weight $\Delta u/h_u$ on that edge alone. A diagonal of a square gets $\tfrac12$ on each of the four edges, with the signs of $\Delta u,\Delta v$. On an edge shared by two squares the two traces agree (the tangential component along a Cartesian edge is $e/h$ in both), so the choice of square does not matter there. The prolongation is

```math
\mathbf e_\text{all}=\mathbf P\mathbf e,\qquad \mathbf P=\begin{bmatrix}\mathbf I\\ \mathbf T\end{bmatrix}\ \text{(free rows / hanging rows; dead and fixed rows are 0)} .
```

**Operators.** Let $\mathbf C=\mathrm{blockdiag}(\mathbf C_\text{cart},\mathbf C_\text{tet})$ be the face-edge incidence of both complexes and $\mathbf K=\mathbf C\mathbf P$. Then

```math
\mathbf W=\mathrm{diag}(\mathbf W_\text{cart})\oplus\mathbf M_2^\text{tet},\qquad
\mathbf M=\mathbf P^T\big(\mathrm{diag}(\mathbf m_\text{cart})\oplus\mathbf M_1^\text{tet}\big)\mathbf P .
```

* $\mathbf W_\text{cart}$ and $\mathbf m_\text{cart}$ are the FIT half-cell / quarter-cell sums over active cells only. A box-surface square therefore sees only its outer half cell, $W=(h_a/2)/(\mu A)$.
* $\mathbf M_1^\text{tet}$ and $\mathbf M_2^\text{tet}$ are the consistent Whitney 1-form and 2-form mass matrices (`whitney_mass_matrices`). A box-surface triangle sees only its inner tetrahedron.
* The interface Cartesian edges collect the mass of their hanging children through the Galerkin product $\mathbf P^T\mathbf M_1\mathbf P$.

Both $\mathbf W$ and $\mathbf M$ are SPD for any non-degenerate mesh, and $\mathbf M$ is SPD because $\mathbf P$ has full column rank. The leapfrog is

```math
\mathbf b^{n+1/2}=\mathbf b^{n-1/2}-\Delta t\,\mathbf K\mathbf e^n,\qquad
\mathbf M(\mathbf e^{n+1}-\mathbf e^n)=\Delta t\big(\mathbf K^T\mathbf W\mathbf b^{n+1/2}-\mathbf P^T\mathbf i^{n+1/2}\big).
```

#### 3. Properties (proved here, tested in `tests/test_105_hybrid_cart_tet.py`)

1. **Discrete de Rham sequence.** Take a nodal potential $\phi$ on the Cartesian nodes and the interior tetrahedral nodes. On the box-surface tetrahedral nodes, let $\phi$ take the bilinear ($Q_1$) interpolant of its square, which is the trace of the trilinear $Q_1$ nodal element whose gradient lies in the hexahedral Nédélec space. Its edge differences then satisfy $\mathbf P\mathbf e_\text{free}=\mathbf G\phi$ exactly, since the line integral of $\nabla Q_1$ along a segment is the difference of the end values. So $\mathbf K\mathbf e=\mathbf C\mathbf G\phi=0$, and the static null space consists of gradients only: no spurious resonances. *Test:* $|\mathbf C\mathbf P\mathbf G\phi|/|\mathbf G\phi|\approx10^{-15}$ on Kuhn and Delaunay patches. The `nearest` and `inject` trace mutations give $0.04$–$0.29$.
2. **Flux bookkeeping.** The trace field has a constant normal curl on each square. Every triangle inside a square therefore receives the area fraction of the square's circulation, and the triangles of a square sum to its circulation exactly. The square flux and the sum of its triangle fluxes (`interface_flux_mismatch`) differ by a constant in time, zero from rest. Every Cartesian cell and every tetrahedron is divergence-free ($\mathbf D\mathbf C=0$ per complex). *Test:* $\le2\times10^{-15}$ relative after a pulse. The `scale` mutation (all trace weights ×1.01) gives $10^{-2}$.
3. **Energy.** For any SPD $\mathbf W$ and $\mathbf M$ the leapfrog conserves
   ```math
   \mathcal E=\tfrac12\,\mathbf b^{n+1/2\,T}\mathbf W\mathbf b^{n+1/2}+\tfrac12\,\mathbf e^{nT}\mathbf M\mathbf e^{n+1}
   ```
   (source-free). $\mathcal E$ is positive iff $\Delta t<2/\sqrt{\lambda_\max(\mathbf M^{-1}\mathbf K^T\mathbf W\mathbf K)}$, which `dt_max` computes by Lanczos. *Test:* relative drift $<10^{-15}$ over 200 steps (LU and Chebyshev mass solves, Kuhn and Delaunay patches). The `asym` mutation, an e-update with a different trace than the b-update, drifts by $>10^{-4}$.
4. **The finer of the two bounds.** Write $\mathbf e_\text{all}=\mathbf P\mathbf e$ and split it into its Cartesian part $\mathbf x_c$ and its tetrahedral part $\mathbf x_t$. The Rayleigh quotient of the composite operator is
   ```math
   \frac{\mathbf x_c^T\mathbf A_c\mathbf x_c+\mathbf x_t^T\mathbf A_t\mathbf x_t}{\mathbf x_c^T\mathbf M_c\mathbf x_c+\mathbf x_t^T\mathbf M_t\mathbf x_t}\le\max\Big(\frac{\mathbf x_c^T\mathbf A_c\mathbf x_c}{\mathbf x_c^T\mathbf M_c\mathbf x_c},\frac{\mathbf x_t^T\mathbf A_t\mathbf x_t}{\mathbf x_t^T\mathbf M_t\mathbf x_t}\Big)\le\max(\lambda_c,\lambda_t),
   ```
   which is the mediant inequality. Here $\lambda_c$ and $\lambda_t$ are the largest eigenvalues of each region alone with its interface edges free (natural boundary), from `block_lambda_bounds`. The hybrid step is therefore never smaller than the smaller of the two regions' own steps: the coupling adds no stiffness of its own. *Test:* $\lambda_\max\le\max(\lambda_c,\lambda_t)$ to $10^{-8}$. In practice the tetrahedral region sets the step (Section 7).
5. **Exact transpose.** `linear_step_T` applies the transposed sequence $(\mathbf M^{-T},\mathbf W^T,\mathbf K^T,\mathbf P)$. With the Chebyshev solve below, $\mathbf M^{-1}$ is replaced by a fixed symmetric polynomial, which is its own transpose. *Test:* dot-product mismatch $\approx10^{-16}$. A non-symmetric perturbation of the mass block, which the symmetric Chebyshev transpose ignores, fails the test, as it must. With LU (exact `trans='T'`) the perturbed system still passes.

#### 4. Solving with the mass matrix

$\mathbf M$ is diagonal on the Cartesian edges away from the patch and coupled on the tetrahedral edges and the interface Cartesian edges. `MassSolver` splits off the coupled rows. Sparse LU of a 3D consistent mass matrix fills badly: $4.7\times10^7$ factor entries for a 26 000-row block. The default for blocks above 4000 rows is therefore a fixed-degree Chebyshev iteration on the Jacobi-scaled block $\mathbf S=\mathbf D^{-1/2}\mathbf B\mathbf D^{-1/2}$:

```math
\mathbf x=\mathbf D^{-1/2}\,p_k(\mathbf S)\,\mathbf D^{-1/2}\mathbf y,\qquad k=\Big\lceil\frac{\ln(2/\text{tol})}{\ln(1/q)}\Big\rceil,\quad q=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
```

* Mass matrices are spectrally equivalent to their diagonal, with element-wise bounds (Wathen 1987). So $\kappa$ does not grow under refinement. Measured: 5.7 on Kuhn meshes and 49–69 on the Delaunay meshes, which gives $k\approx166$ for tol $=10^{-16}$.
* The spectral bounds come from Lanczos, with factors 0.9 and 1.02.
* $p_k(\mathbf S)$ is a fixed polynomial of a symmetric matrix. The step is exactly linear and its transpose is the same operator. The scheme then conserves the energy of the SPD mass $p_k(\mathbf S)^{-1}$, which differs from $\mathbf M$ by at most tol.
* Measured on the 26 000-row block: residual $9\times10^{-16}$, symmetry error $3\times10^{-16}$, 36 ms per solve on one core.

Lumping the tetrahedral mass instead (the diagonal Voronoi Hodge of `UnstructuredFIT3D`) would make the update fully explicit. On these meshes, however, 1.3–1.7 % of the interior edges have dual areas below $10^{-2}L^2$, and the box-surface edges have negative dual areas, down to $-0.54L^2$, because circumcentres fall outside the box. The Whitney masses are SPD on any mesh.

#### 5. Meshing the patch

* **`kuhn_box_mesh`.** Freudenthal/Kuhn split of every (sub)cube into six tetrahedra around the main diagonal (Freudenthal 1942). It is conforming, deterministic and well shaped, and each boundary square gets one diagonal. Used by the fast tests.
* **`delaunay_box_mesh`.** Delaunay mesh of a box with an optional spherical hole (PEC) or tagged inclusion. The interior nodes form a body-centred cubic lattice, ODT-smoothed (Chen & Xu 2004) with the box nodes fixed and the sphere nodes projected back to the sphere. The sphere surface is recovered by removing interior nodes from the diametral balls of its hull triangles, as in `ball_mesh`. The box surface needs care, because a Delaunay mesh of a regular surface lattice is degenerate. The surface nodes are therefore:
  * the lattice nodes;
  * the midpoints of the surface lattice edges. Each half-edge is then a Gabriel edge: its diametral circle of radius $h/4$ is empty, since the nearest other node is at least $0.35h$ away. It therefore belongs to the face's 2D Delaunay triangulation, which the boundary of the 3D Delaunay mesh reproduces, so no triangle can cross a Cartesian edge;
  * one node near the centre of every square, jittered by up to 15 %, which breaks the cocircular degeneracies inside the squares.

  Lattice nodes on two faces next to a box edge can still form exactly cospherical planar quadrilaterals. Delaunay then emits zero-volume tetrahedra, and dropping them would leave a non-conforming seam. `_resolve_flat` removes each one by a 2-2 flip of the neighbour pair on one side of the quad, then raises if any degenerate cell remains. `box_surface_check` verifies that the triangles cover the box exactly and that each lies inside one square. Tested over 6 seeds × 4 configurations; the worst normalised volume is $10^{-3}$–$3\times10^{-2}$ (slivers).

#### 6. Consistency and the patch test

Both sides reproduce constant fields and fields of constant curl exactly, and so does the trace:

* hexahedral Nédélec reproduces $\mathbf E=\mathbf a+\mathbf b\times\mathbf r$ per cell;
* $\mathbf P$ integrates it exactly along every hanging edge;
* Whitney forms contain $\mathbf a+\mathbf b\times\mathbf r$.

The interface therefore passes the lowest-order patch test. The remaining mismatch is that the tetrahedral tangential field on a square is the Whitney interpolant of a bilinear field, a local $O(h)$ difference with zero circulation on every triangle. In the measurements below it contributes at second order or better.

#### 7. Verification

Study: `studies/hybrid-cart-tet/` (`run_study.py`, `plot_figures.py`, Batch jobs `w2hyb-a` … `w2hyb-d`). Figures in `docs/figures.md`.

RESULTS_PLACEHOLDER

#### 8. Limits & Failure Modes

LIMITS_PLACEHOLDER

#### 9. References

- J. C. Nédélec, "Mixed finite elements in $\mathbb R^3$", *Numer. Math.* **35**, 315–341 (1980), doi:10.1007/BF01396415.
- T. Rylander and A. Bondeson, "Stable FEM-FDTD hybrid method for Maxwell's equations", *Comput. Phys. Commun.* **125**, 75–82 (2000), doi:10.1016/S0010-4655(99)00463-4.
- T. Rylander and A. Bondeson, "Application of stable FEM-FDTD hybrid to scattering problems", *IEEE Trans. Antennas Propag.* **50**(2), 141–144 (2002), doi:10.1109/8.997982.
- F. Edelvik and G. Ledfelt, "Explicit hybrid time domain solver for the Maxwell equations in 3D", *J. Sci. Comput.* **15**, 61–78 (2000), doi:10.1023/A:1007625629485.
- A. J. Wathen, "Realistic eigenvalue bounds for the Galerkin mass matrix", *IMA J. Numer. Anal.* **7**, 449–457 (1987), doi:10.1093/imanum/7.4.449.
- L. Chen and J. Xu, "Optimal Delaunay triangulations", *J. Comput. Math.* **22**, 299–308 (2004), doi:10.4208/jcm.v22.n2.p299.
- H. Freudenthal, "Simplizialzerlegungen von beschränkter Flachheit", *Ann. Math.* **43**, 580–582 (1942), doi:10.2307/1968813.
- F. Labelle and J. R. Shewchuk, "Isosurface stuffing: fast tetrahedral meshes with good dihedral angles", *ACM Trans. Graph.* **26**(3), 57 (2007), doi:10.1145/1276377.1276448.
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley, 1983), Ch. 4 (vector spherical harmonics, perfectly conducting sphere).
- M. Clemens and T. Weiland, "Discrete electromagnetism with the finite integration technique", *Prog. Electromagn. Res.* **32**, 65–87 (2001), doi:10.2528/PIER00080103.
