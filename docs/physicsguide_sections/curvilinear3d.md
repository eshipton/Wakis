### 🧊 3D curvilinear hexahedral FIT: `wakis.curvilinear` (`HexMesh`, `HexBlock3D`, `CurvilinearFIT3D`)

`wakis.curvilinear` extends the 2D nonorthogonal Finite Integration Technique (NFIT) [Holland, 1983; Schuhmann & Weiland, 2000; Clemens & Weiland, 2001] to 3D conforming curvilinear hexahedral meshes (`HexMesh` in `wakis/curvilinear/hexmesh.py` and single-block `HexBlock3D` in `wakis/curvilinear/mesh3d.py`) stepped in the time domain or solved for cavity eigenmodes by `CurvilinearFIT3D` (`wakis/curvilinear/solver3d.py`). The topological curl, gradient, and divergence operators are exact integer incidence matrices on the primal/dual cell complex, while all metric and material couplings are assembled corner by corner into sparse, symmetric positive semi-definite inverse material (Hodge) matrices $\mathbf{W}_\nu$ and $\mathbf{W}_\varepsilon$ so that the explicit leapfrog update requires no linear system solve.

#### 1. Setting and notation

A conforming 3D hexahedral mesh (`HexMesh`) consists of $N_v$ node coordinates $\mathbf{x}_i \in \mathbb{R}^3$ [m] (`nodes` of shape `(n_nodes, 3)`) and $N_c$ hexahedral cells (`hexes` of shape `(n_cells, 8)`) indexed in VTK hexahedron vertex ordering on the reference cube $(\xi_0, \xi_1, \xi_2) = (\xi, \eta, \zeta) \in [0, 1]^3$:

* bottom face ($\zeta = 0$, counter-clockwise in $(\xi, \eta)$): vertices $0:(0,0,0)$, $1:(1,0,0)$, $2:(1,1,0)$, $3:(0,1,0)$;
* top face ($\zeta = 1$, same order in $(\xi, \eta)$): vertices $4:(0,0,1)$, $5:(1,0,1)$, $6:(1,1,1)$, $7:(0,1,1)$.

Each logical corner $(a, b, c) \in \{0, 1\}^3$ of a hexahedron $h$ maps to one of its eight VTK vertices via `_CORNER_VTK[a, b, c]`. At corner $c = (a, b, c)$ of hexahedron $h$ with vertex $\mathbf{x}_{h,c}$, let $\Delta\mathbf{x}_{h,c}^{(0)}, \Delta\mathbf{x}_{h,c}^{(1)}, \Delta\mathbf{x}_{h,c}^{(2)}$ denote the three primal edge vectors incident to that corner, each oriented in the positive logical direction $+\xi_0, +\xi_1, +\xi_2$. Under the trilinear reference-to-physical map $\mathbf{x}_h(\xi, \eta, \zeta)$, the partial derivatives at corner $c$ equal these three edge vectors, giving the corner Jacobian determinant (derived from the trilinear map gradient evaluated at $(\xi, \eta, \zeta) \in \{0, 1\}^3$):

```math
J_{h,c} = \det\!\big[\Delta\mathbf{x}_{h,c}^{(0)},\, \Delta\mathbf{x}_{h,c}^{(1)},\, \Delta\mathbf{x}_{h,c}^{(2)}\big] \quad [\text{m}^3], \qquad
V_h = \frac{1}{8}\sum_{c=0}^{7} J_{h,c}, \qquad
V_{h,c} = \frac{1}{8} V_h \quad [\text{m}^3], \tag{1.1}
```

where every corner must satisfy $J_{h,c} > 0$ (`ValueError` on inverted or degenerate cells) and $V_{h,c} = V_h / 8$ is the uniform corner quadrature weight of hexahedron $h$.

**Global orientation and discrete de Rham sequence.** Across an arbitrary conforming multi-block `HexMesh`, unique primal edges and faces are deduplicated and oriented canonically without global $(i, j, k)$ grid lines:

* **Primal edges ($N_e$):** Each unique edge $e$ is identified by its sorted global node pair $(u, v)$ with $u < v$, sorted lexicographically, and oriented from $u$ to $v$. The topological gradient $\mathbf{G} \in \{-1, 0, +1\}^{N_e \times N_v}$ has $G_{e,v} = +1$ and $G_{e,u} = -1$. A local hex edge along $+\xi_\alpha$ maps to global edge $e$ with sign $s_{h,\text{edge}} = +1$ if its start node index is less than its end node index and $-1$ otherwise.
* **Primal faces ($N_f$):** Each unique quadrilateral face $f$ is identified by its sorted 4-tuple of global node indices $(n_0, n_1, n_2, n_3)$ with $n_0 < n_1 < n_2 < n_3$, sorted lexicographically. Its canonical perimeter cycle $(q_0, q_1, q_2, q_3)$ is anchored at $q_0 = n_0$ and follows the $+\xi_\alpha$ right-handed cycle of the first incident local hex face. Both the topological curl $\mathbf{C} \in \{-1, 0, +1\}^{N_f \times N_e}$ and the bilinear shoelace vector area (derived by integrating $\frac{1}{2}\mathbf{x}\times d\mathbf{x}$ around the bilinear quad perimeter $(q_0, q_1, q_2, q_3)$)

```math
\mathbf{A}_f = \frac{1}{2}(\mathbf{x}_{q_2} - \mathbf{x}_{q_0}) \times (\mathbf{x}_{q_3} - \mathbf{x}_{q_1}) \quad [\text{m}^2] \tag{1.2}
```

use this identical cycle $(q_0, q_1, q_2, q_3)$. Each local hex face $(h, \alpha, \text{side})$ carries sign $s_{h,f} = +1$ if its $q_0$-anchored $+\xi_\alpha$ cycle matches $(q_0, q_1, q_2, q_3)$ and $-1$ if reversed.
* **Dual face areas ($\mathbf{S}_e$):** Within each hexahedron $h$ with centroid $\mathbf{x}_h^{\text{cell}}$, face centroids $\mathbf{x}_{h,f}^{\text{face}}$, and edge midpoints $\mathbf{x}_e^{\text{edge}}$, each local edge along $+\xi_\alpha$ accumulates four dual sub-triangles connecting $\mathbf{x}_e^{\text{edge}}$, the two adjacent face centroids, and $\mathbf{x}_h^{\text{cell}}$, multiplied by $s_{h,\text{edge}}$ to yield the global dual face area vector $\mathbf{S}_e \in \mathbb{R}^3$ [$\text{m}^2$].

Because every edge of face $f$ appears twice with opposite orientation in $\mathbf{C}\mathbf{G}$ and every face of cell $h$ shares each of its edges with a second face of $h$ with opposite outward orientation in the cell–face divergence $\mathbf{D} \in \{-1, 0, +1\}^{N_c \times N_f}$, the discrete de Rham identities (derived from the boundary-of-boundary property $\partial^2 = 0$ on the cell complex [Clemens & Weiland, 2001])

```math
\mathbf{C}\mathbf{G} = \mathbf{0}, \qquad \mathbf{D}\mathbf{C} = \mathbf{0} \tag{1.3}
```

hold identically (`max|D C| = 0.0`, `rel_curl_grad <= 1e-14` in `tests/test_112_curvilinear3d.py`). On a single structured block (`HexBlock3D`), `HexMesh.from_block` stores the bijective edge and face permutations `(perm_e, perm_f)` between C-order family indexing and lexicographical `HexMesh` indexing, producing bit-for-bit identical operators (`max_abs_diff = 0.0`).

**PEC boundary degrees of freedom.** Boundary faces are those incident to a single hexahedron; all nodes and edges of boundary faces form the PEC outer boundary (`boundary_node_mask`, `boundary_edge_mask`). Tangential electric voltages $e_e$ and electric fluxes $d_e$ on boundary edges are fixed to zero, leaving $N_u$ unknown (interior and boundary-normal) edges indexed by `unknown_edges`, with restricted curl $\mathbf{C}_u = \mathbf{C}_{:,\,\text{unknown\_edges}} \in \{-1, 0, +1\}^{N_f \times N_u}$ and interior node gradient $\mathbf{G}_u$ satisfying $\mathbf{C}_u \mathbf{G}_u = \mathbf{0}$.

#### 2. Operators from corner reconstructions

Following [Schuhmann & Weiland, 2000], `CurvilinearFIT3D` assembles the inverse material matrices $\mathbf{W}_\nu = \mathbf{M}_{\mu}^{-1} \in \mathbb{R}^{N_f \times N_f}$ [$\text{H}^{-1}$] and $\mathbf{W}_\varepsilon = \mathbf{M}_{\varepsilon}^{-1} \in \mathbb{R}^{N_u \times N_u}$ [$\text{F}^{-1}$] directly in the flux-to-voltage direction as a sum over all hexahedra $h$ and their eight corners $c \in \{0, \dots, 7\}$.

**Magnetic Hodge $\mathbf{W}_\nu$.** At corner $c = (a, b, d) \in \{0, 1\}^3$ of hexahedron $h$ with permeability $\mu_h = \mu_0 \mu_{r,h}$, three primal faces $f_{h,c} = (f_0, f_1, f_2)$ of logical normals $\xi_0, \xi_1, \xi_2$ meet at the corner with local-to-global orientation signs $\mathbf{s}_{h,c}^f = (s_0, s_1, s_2) \in \{-1, +1\}^3$. For a uniform magnetic flux density $\mathbf{B}_{h,c} \in \mathbb{R}^3$ [T] in the corner octant, the local $+\xi_p$-oriented face fluxes are $\tilde{b}_p = s_p b_{f_p} = (s_p \mathbf{A}_{f_p}) \cdot \mathbf{B}_{h,c}$. Stacking the three signed area rows into $\mathbf{M}_{h,c} = \operatorname{diag}(\mathbf{s}_{h,c}^f)\,\mathbf{A}_{f_{h,c}} \in \mathbb{R}^{3\times 3}$ and inverting gives the exact corner magnetic reconstruction matrix $\mathbf{R}_{h,c}$ and the factored magnetic Hodge (derived by equating the sum of octant magnetic energies $\sum_{h,c} \frac{V_{h,c}}{2\mu_h} \|\mathbf{B}_{h,c}\|^2$ to $\frac{1}{2}\mathbf{b}^T \mathbf{W}_\nu \mathbf{b}$):

```math
\mathbf{B}_{h,c} = \mathbf{R}_{h,c}\,\mathbf{b}_{f_{h,c}}, \qquad
\mathbf{R}_{h,c} = \mathbf{M}_{h,c}^{-1}\operatorname{diag}(\mathbf{s}_{h,c}^f), \qquad
\mathbf{W}_\nu = \sum_{h=0}^{N_c-1}\sum_{c=0}^{7} \frac{V_{h,c}}{\mu_h}\,\mathbf{P}_{f_{h,c}}^T \mathbf{R}_{h,c}^T \mathbf{R}_{h,c} \mathbf{P}_{f_{h,c}} = \tilde{\mathbf{R}}_\nu^T \tilde{\mathbf{R}}_\nu, \tag{2.1}
```

where $\mathbf{P}_{f_{h,c}} \in \{0, 1\}^{3 \times N_f}$ extracts the three face fluxes $\mathbf{b}_{f_{h,c}}$ and $\tilde{\mathbf{R}}_\nu \in \mathbb{R}^{24 N_c \times N_f}$ stacks the scaled rows $\sqrt{V_{h,c}/\mu_h}\,\mathbf{R}_{h,c}\mathbf{P}_{f_{h,c}}$.

**Electric Hodge $\mathbf{W}_\varepsilon$, the constrained PEC wall rule, and the sharp-octant rule.** At corner $c = (a, b, d) \in \{0, 1\}^3$ of hexahedron $h$ with permittivity $\varepsilon_h = \varepsilon_0 \varepsilon_{r,h}$, three primal edges $e_{h,c} = (e_0, e_1, e_2)$ meet with signs $\mathbf{s}_{h,c}^e \in \{-1, +1\}^3$ and dual face area vectors $\mathbf{S}_{e_q}$. Forming $\mathbf{N}_{h,c} = \operatorname{diag}(\mathbf{s}_{h,c}^e)\,\mathbf{S}_{e_{h,c}} \in \mathbb{R}^{3\times 3}$, the unconstrained interior reconstruction of the corner electric displacement $\mathbf{D}_{h,c} = \mathbf{Q}_{h,c}^{\text{int}}\,\mathbf{d}_{e_{h,c}}$ is (derived identically to Eq. (2.1) from $\tilde{d}_q = (s_q \mathbf{S}_{e_q})\cdot\mathbf{D}_{h,c}$):

```math
\mathbf{Q}_{h,c}^{\text{int}} = \mathbf{N}_{h,c}^{-1}\operatorname{diag}(\mathbf{s}_{h,c}^e). \tag{2.2}
```

At a PEC boundary node $v = \text{hexes}[h, c]$, however, two of the three incident edges lie on the PEC wall and have their dual-face fluxes $d_e$ dropped when restricting $\mathbf{W}_\varepsilon$ to `unknown_edges`. On a nonorthogonal curved boundary (such as a sphere or cylinder), the global dual face vectors $\mathbf{S}_{e_q}$ of the two wall edges are not strictly tangent to the wall normal $\hat{\mathbf{n}}_v$; keeping the single remaining column of $\mathbf{Q}_{h,c}^{\text{int}}$ (`_pec_reconstruction="drop_only"`) reconstructs a spurious tangential displacement $\mathbf{D}_{h,c} \times \hat{\mathbf{n}}_v \neq \mathbf{0}$ (`||D_c x n|| / ||D_c|| = 9.294193e-01` on `sphere_nodes(6, 6, 6)` in `test_pec_drop_only_is_not_pec`) and causes the lowest TE cavity eigenvalue to drift away from the analytical resonance as the mesh is refined.

To enforce the physical PEC boundary condition $\mathbf{D}_{h,c} \times \hat{\mathbf{n}} = \mathbf{0}$ at every boundary corner octant (`_pec_reconstruction="constrained"`, the default), `HexMesh` classifies each boundary node $v$ using the smooth wall patch labels `boundary_patch` and generator-supplied `sharp_nodes`:

1. **Smooth PEC wall octant** ($v \in \partial\Omega$ with all incident boundary faces on the same smooth wall patch and `sharp_nodes[v] == False`): the node unit normal $\hat{\mathbf{n}}_v$ is the normalised sum of incident boundary face area vectors. The feasible electric displacement at $v$ lies in $\operatorname{span}(\hat{\mathbf{n}}_v)$, $\mathbf{D}_{h,c} = \alpha\,\hat{\mathbf{n}}_v$. Minimising the least-squares flux residual $\sum_{q \in \mathcal{F}_{h,c}} (\mathbf{S}_{e_q}\cdot(\alpha\hat{\mathbf{n}}_v) - d_{e_q})^2$ over the non-PEC (free) edges $\mathcal{F}_{h,c} = \{q \in \{0,1,2\} : e_q \notin \partial\Omega\}$ of the octant yields the rank-1 constrained corner reconstruction (derived by setting $\frac{d}{d\alpha}\sum_{q \in \mathcal{F}_{h,c}} (\alpha\,\mathbf{S}_{e_q}\cdot\hat{\mathbf{n}}_v - d_{e_q})^2 = 0$):

```math
\mathbf{Q}_{h,c}^{\text{wall}} = \frac{\hat{\mathbf{n}}_v\,(\mathbf{s}_{\perp,h,c})^T}{\|\mathbf{s}_{\perp,h,c}\|^2}, \qquad
(\mathbf{s}_{\perp,h,c})_q = \begin{cases}
\mathbf{S}_{e_q}\cdot\hat{\mathbf{n}}_v, & e_q \notin \partial\Omega,\\[2pt]
0, & e_q \in \partial\Omega.
\end{cases} \tag{2.3}
```

   When the octant has a single interior edge $e_{q^*}$, Eq. (2.3) satisfies $\mathbf{D}_{h,c}\times\hat{\mathbf{n}}_v = \mathbf{0}$ and $\mathbf{S}_{e_{q^*}}\cdot\mathbf{D}_{h,c} = d_{e_{q^*}}$ to machine precision ($\le 10^{-12}$ in `test_pec_constraint_wall_octants`).

2. **Sharp PEC edge/corner octant** ($v \in \partial\Omega$ where incident boundary faces belong to two or more distinct smooth wall patches with non-parallel normals $\hat{\mathbf{n}}_1 \nparallel \hat{\mathbf{n}}_2$, or where `sharp_nodes[v] == True`): the intersection of the feasible PEC subspaces $\{\mathbf{D} : \mathbf{D}\times\hat{\mathbf{n}}_k = \mathbf{0}\}$ across two or more non-parallel wall normals is $\{\mathbf{0}\}$, so the corner reconstruction sets (derived from $\operatorname{span}(\hat{\mathbf{n}}_1) \cap \operatorname{span}(\hat{\mathbf{n}}_2) = \{\mathbf{0}\}$):

```math
\mathbf{Q}_{h,c}^{\text{sharp}} = \mathbf{0}_{3\times 3} \quad \Longrightarrow \quad \mathbf{D}_{h,c} = \mathbf{0}. \tag{2.4}
```

Summing over all cells and corners and restricting to the $N_u$ unknown edges via $\mathbf{P}_u \in \{0, 1\}^{N_u \times N_e}$ gives the symmetric positive semi-definite electric Hodge (derived from the octant electric energy $\sum_{h,c} \frac{V_{h,c}}{2\varepsilon_h}\|\mathbf{D}_{h,c}\|^2 = \frac{1}{2}\mathbf{d}^T \mathbf{W}_\varepsilon \mathbf{d}$):

```math
\mathbf{W}_{\varepsilon,\text{full}} = \sum_{h=0}^{N_c-1}\sum_{c=0}^{7} \frac{V_{h,c}}{\varepsilon_h}\,\mathbf{P}_{e_{h,c}}^T \mathbf{Q}_{h,c}^T \mathbf{Q}_{h,c} \mathbf{P}_{e_{h,c}}, \qquad
\mathbf{W}_\varepsilon = \mathbf{P}_u \mathbf{W}_{\varepsilon,\text{full}} \mathbf{P}_u^T = \tilde{\mathbf{Q}}_\varepsilon^T \tilde{\mathbf{Q}}_\varepsilon. \tag{2.5}
```

#### 3. Consistency and orthogonal-grid reduction to Yee

**Uniform-field consistency.** On any hexahedron $h$ with straight edges and planar faces (an affine image of the unit cube), a constant magnetic flux density $\mathbf{B}_0$ produces face fluxes $b_{f_p} = \mathbf{A}_{f_p}\cdot\mathbf{B}_0$, so Eq. (2.1) recovers $\mathbf{R}_{h,c}\mathbf{b}_{f_{h,c}} = \mathbf{M}_{h,c}^{-1}\mathbf{M}_{h,c}\mathbf{B}_0 = \mathbf{B}_0$ exactly at all eight corners, and $\sum_{c=0}^7 V_{h,c} = V_h$ matches the exact cell volume (`vol_err <= 1e-14`, `af_err <= 1e-14`, `se_err <= 1e-14` in `test_hex_geometry`).

**Exact reduction to the staggered Yee/FIT scheme on orthogonal grids.** On a non-uniform orthogonal Cartesian grid with cell spacings $(h_{x,i}, h_{y,j}, h_{z,k})$ and cell-wise $\varepsilon_{i,j,k}, \mu_{i,j,k}$, the three face area vectors at corner $c$ are mutually orthogonal along the coordinate axes:

```math
\mathbf{M}_{h,c} = \operatorname{diag}(A_{x,h}, A_{y,h}, A_{z,h}) = \operatorname{diag}(h_y h_z,\, h_z h_x,\, h_x h_y), \qquad
\mathbf{N}_{h,c} = \operatorname{diag}(S_{x,e_0}, S_{y,e_1}, S_{z,e_2}), \tag{3.1}
```

where each global dual face area $S_{\alpha,e}$ is the sum of the four quarter-areas $\frac{1}{4}A_{\alpha,h}$ of the cells sharing edge $e$. Because $\mathbf{M}_{h,c}$ and $\mathbf{N}_{h,c}$ are diagonal, $\mathbf{R}_{h,c}^T \mathbf{R}_{h,c}$ and $\mathbf{Q}_{h,c}^T \mathbf{Q}_{h,c}$ are strictly diagonal (zero off-diagonal couplings, `off_nu = 0`, `off_eps = 0`). Summing the four corners sharing face $f$ (each with $V_{h,c} = \frac{1}{8} h_x h_y h_z$) and the four corners along edge $e$ inside each cell yields the exact 1D/3D Yee/FIT diagonal Hodges (derived by substituting Eq. (3.1) into Eqs. (2.1) and (2.5)):

```math
(\mathbf{W}_\nu)_{f,f} = \sum_{h \ni f} \frac{4 V_{h,c}}{\mu_h A_{f}^2} = \sum_{h \ni f} \frac{h_{\alpha,h} / 2}{\mu_h A_f}, \qquad
(\mathbf{W}_\varepsilon)_{e,e} = \sum_{h \ni e} \frac{2 V_{h,c}}{\varepsilon_h S_e^2} = \frac{h_{\alpha,e}}{S_e^2}\sum_{h \ni e} \frac{A_{\alpha,h} / 4}{\varepsilon_h}. \tag{3.2}
```

Furthermore, at any smooth orthogonal PEC wall with normal $\hat{\mathbf{n}}_v = \pm\hat{\mathbf{e}}_\alpha$, the normal edge $e_{q^*}$ has $\mathbf{S}_{e_{q^*}} = S_{e_{q^*}}\hat{\mathbf{e}}_\alpha$ parallel to $\hat{\mathbf{n}}_v$, so Eq. (2.3) gives $\mathbf{Q}_{h,c}^{\text{wall}}\mathbf{d} = (d_{e_{q^*}} / S_{e_{q^*}})\hat{\mathbf{e}}_\alpha$, which is identical to the unconstrained diagonal column of Eq. (2.2) (`_pec_reconstruction="drop_only"`). At sharp box edges and corners, all three octant edges lie on the PEC walls ($\mathcal{F}_{h,c} = \emptyset$), so every column of $\mathbf{Q}_{h,c}$ is dropped upon restriction to `unknown_edges` in both rules. Consequently, on a non-uniform $6 \times 5 \times 4$ Cartesian box with random per-cell $\varepsilon_r \in [1.1, 4.5]$ (`test_pec_orthogonal_box_equals_yee`, Batch job `w3curvhexb-tests`), the constrained and unconstrained electric Hodges agree to machine precision with measured relative difference `1.547466e-16`.

On a uniform $n_x \times n_y \times n_z$ orthogonal Cartesian grid over $[0, a] \times [0, b] \times [0, d]$ with spacings $h_x = a/n_x$, $h_y = b/n_y$, $h_z = d/n_z$ (`perturbed_box_nodes` with $\text{amp} = 0$), this exact operator reduction implies that the discrete $\text{TE}_{101}$ cavity eigenfrequency of `CurvilinearFIT3D` equals the closed-form semi-discrete Yee dispersion eigenvalue (`yee_te101_omega` in `studies/curvilinear3d/perturbed_box_order.py`, derived by substituting the discrete Fourier mode $E_y \propto \sin(\pi x/a)\sin(\pi z/d)$ into the diagonal Yee curl-curl operator):

```math
\omega_h^2 = c_0^2 \left[
    \left(\frac{2}{h_x}\sin\frac{\pi h_x}{2 a}\right)^2
    + \left(\frac{2}{h_z}\sin\frac{\pi h_z}{2 d}\right)^2
\right]. \tag{3.3}
```

On $[0, 1.0]\times[0, 0.7]\times[0, 0.8]\;\text{m}^3$ ($\omega_{\text{ana}} = 1.50765688\times 10^9\;\text{rad/s}$), `CurvilinearFIT3D` with $\text{amp} = 0$ matches Eq. (3.3) to relative mismatch `5.960e-13` at $6\times 6\times 6$ ($\omega_{\text{num}} = 1.490493603473\times 10^9\;\text{rad/s}$ vs $\omega_h = 1.490493603474\times 10^9\;\text{rad/s}$), `5.965e-13` at $8\times 8\times 8$ ($\omega_{\text{num}} = 1.497988047721\times 10^9\;\text{rad/s}$ vs $\omega_h = 1.497988047722\times 10^9\;\text{rad/s}$), and `5.972e-13` at $4\times 3\times 4$ ($\omega_{\text{num}} = 1.469204627425\times 10^9\;\text{rad/s}$ vs $\omega_h = 1.469204627426\times 10^9\;\text{rad/s}$), all $\le 10^{-9}$ (`test_yee_te101_semidiscrete_dispersion_identity` in `tests/test_114_curvilinear3d_diagnostics.py`), whereas comparing against the continuum eigenvalue $\omega_{\text{ana}}$ yields the second-order Yee dispersion errors `1.138407e-02`, `6.413149e-03`, and `2.550464e-02`.

#### 4. Time stepping, energy and stability

**Explicit leapfrog update.** With magnetic fluxes $\mathbf{b}^{n-1/2} \in \mathbb{R}^{N_f}$ [Wb] at half-integer time steps and unknown electric fluxes $\mathbf{d}^n \in \mathbb{R}^{N_u}$ [C] at integer time steps, `CurvilinearFIT3D.step` advances one step $\mathcal{S}: (\mathbf{b}^{n-1/2}, \mathbf{d}^n) \mapsto (\mathbf{b}^{n+1/2}, \mathbf{d}^{n+1})$ without inverting any matrix (derived from Faraday's and Ampère's integral laws on the primal/dual complex):

```math
\begin{aligned}
\mathbf{e}^n &= \mathbf{W}_\varepsilon \mathbf{d}^n, &\qquad
\mathbf{b}^{n+1/2} &= \mathbf{b}^{n-1/2} - \Delta t\,\mathbf{C}_u \mathbf{e}^n, \\
\mathbf{h}^{n+1/2} &= \mathbf{W}_\nu \mathbf{b}^{n+1/2}, &\qquad
\mathbf{d}^{n+1} &= \mathbf{d}^n + \Delta t\,\mathbf{C}_u^T \mathbf{h}^{n+1/2}.
\end{aligned} \tag{4.1}
```

**Symmetric PSD structure and discrete energy conservation.** Because Eqs. (2.1) and (2.5) assemble $\mathbf{W}_\nu = \tilde{\mathbf{R}}_\nu^T \tilde{\mathbf{R}}_\nu \succeq 0$ and $\mathbf{W}_\varepsilon = \tilde{\mathbf{Q}}_\varepsilon^T \tilde{\mathbf{Q}}_\varepsilon \succeq 0$ as sums of symmetric outer products with positive corner weights $V_{h,c}/\mu_h > 0$ and $V_{h,c}/\varepsilon_h > 0$, both Hodges are symmetric positive semi-definite by construction. Define the discrete leapfrog energy at step $n$:

```math
\mathcal{E}^n = \frac{1}{2}(\mathbf{d}^n)^T \mathbf{W}_\varepsilon \mathbf{d}^n + \frac{1}{2}(\mathbf{b}^{n+1/2})^T \mathbf{W}_\nu \mathbf{b}^{n-1/2} \quad [\text{J}]. \tag{4.2}
```

Subtracting $\mathcal{E}^n$ from $\mathcal{E}^{n+1}$ and substituting Eq. (4.1) with $\mathbf{W}_\varepsilon^T = \mathbf{W}_\varepsilon$ and $\mathbf{W}_\nu^T = \mathbf{W}_\nu$ gives (derived by expanding the electric and magnetic increments over step $n \to n+1$):

```math
\begin{aligned}
\mathcal{E}^{n+1} - \mathcal{E}^n
&= \frac{1}{2}(\mathbf{d}^{n+1} + \mathbf{d}^n)^T \mathbf{W}_\varepsilon (\mathbf{d}^{n+1} - \mathbf{d}^n)
 + \frac{1}{2}(\mathbf{b}^{n+3/2} - \mathbf{b}^{n-1/2})^T \mathbf{W}_\nu \mathbf{b}^{n+1/2} \\
&= \frac{\Delta t}{2}(\mathbf{d}^{n+1} + \mathbf{d}^n)^T \mathbf{W}_\varepsilon \mathbf{C}_u^T \mathbf{W}_\nu \mathbf{b}^{n+1/2}
 - \frac{\Delta t}{2}\big(\mathbf{C}_u \mathbf{W}_\varepsilon (\mathbf{d}^{n+1} + \mathbf{d}^n)\big)^T \mathbf{W}_\nu \mathbf{b}^{n+1/2} = 0.
\end{aligned} \tag{4.3}
```

Eliminating $\mathbf{b}^{n-1/2} = \mathbf{b}^{n+1/2} + \Delta t\,\mathbf{C}_u \mathbf{W}_\varepsilon \mathbf{d}^n$ in Eq. (4.2) shows that $\mathcal{E}^n$ is a positive-definite quadratic form on the state if and only if the Courant–Friedrichs–Lewy (CFL) condition holds (derived by completing the square in $\mathbf{b}^{n+1/2}$):

```math
\Delta t < \Delta t_{\max} = \frac{2}{\sqrt{\lambda_{\max}(\mathbf{C}_u^T \mathbf{W}_\nu \mathbf{C}_u \mathbf{W}_\varepsilon)}} = \frac{2}{\sqrt{\lambda_{\max}\!\big((\tilde{\mathbf{R}}_\nu \mathbf{C}_u)\,\mathbf{W}_\varepsilon\,(\tilde{\mathbf{R}}_\nu \mathbf{C}_u)^T\big)}}, \tag{4.4}
```

where `max_stable_dt()` computes $\lambda_{\max}$ via Lanczos iteration (`scipy.sparse.linalg.eigsh`) on the symmetric positive semi-definite factored operator $(\tilde{\mathbf{R}}_\nu \mathbf{C}_u)\mathbf{W}_\varepsilon(\tilde{\mathbf{R}}_\nu \mathbf{C}_u)^T$, and `cavity_frequencies(k)` extracts the lowest positive cavity eigenfrequencies $\omega = \sqrt{\lambda}$ [rad/s] from the same symmetric operator. Over $10^4$ steps at $0.5\,\Delta t_{\max}$, the relative energy drift stays below $10^{-13}$ on both single-block and multi-block meshes (`test_energy_conserved_pec_box`, `test_energy_generic_mesh`), while steps at $0.95\,\Delta t_{\max}$ remain bounded and steps at $1.05\,\Delta t_{\max}$ grow exponentially (`test_stability_bound`, `test_stability_generic_mesh`).

**Exact Euclidean transpose and dot-product test.** Writing the one-step linear map in Eq. (4.1) as a composition of two triangular shears, $\mathbf{b}_1 = \mathbf{b} - \Delta t\,\mathbf{C}_u \mathbf{W}_\varepsilon \mathbf{d}$ followed by $\mathbf{d}_1 = \mathbf{d} + \Delta t\,\mathbf{C}_u^T \mathbf{W}_\nu \mathbf{b}_1$, its exact Euclidean transpose $\mathcal{S}^T: (\mathbf{y}_b, \mathbf{y}_d) \mapsto (\tilde{\mathbf{b}}, \tilde{\mathbf{d}})$ (`CurvilinearFIT3D.transpose_step`) reverses the order of the two shears and transposes each factor (derived from $\mathcal{S} = \mathcal{S}_2 \mathcal{S}_1 \Rightarrow \mathcal{S}^T = \mathcal{S}_1^T \mathcal{S}_2^T$):

```math
\tilde{\mathbf{b}} = \mathbf{y}_b + \Delta t\,\mathbf{W}_\nu^T (\mathbf{C}_u \mathbf{y}_d), \qquad
\tilde{\mathbf{d}} = \mathbf{y}_d - \Delta t\,\mathbf{W}_\varepsilon^T (\mathbf{C}_u^T \tilde{\mathbf{b}}). \tag{4.5}
```

The inner-product identity $\langle (\mathbf{y}_b, \mathbf{y}_d),\, \mathcal{S}(\mathbf{b}, \mathbf{d})\rangle = \langle \mathcal{S}^T(\mathbf{y}_b, \mathbf{y}_d),\, (\mathbf{b}, \mathbf{d})\rangle$ holds to relative error $\le 10^{-14}$ on deformed single-block, butterfly-disk, and cubed-sphere meshes (`test_dot_product_step`, `test_dot_product_generic_mesh`), whereas an un-reversed transpose mutation fails with relative error $> 10^{-3}$.

#### 5. Generators

Single-block mappings of a cube $[-1, 1]^3$ onto a disk or ball (`cylinder_nodes` and `sphere_nodes` in `wakis/curvilinear/mappings3d.py`) collapse two or three cube faces onto a single smooth outer wall at the cube corners $(|u|, |v|) = (1, 1)$ or $(|u|, |v|, |w|) = (1, 1, 1)$. At those corners the continuous Jacobian vanishes and the normalised discrete corner determinant scales as $O(1/n)$ on the elliptic cylinder and $O(1/n^2)$ on the elliptic sphere (`test_mapping_cylinder_corner_jacobian`, `test_sphere_lowest_roots`). To eliminate boundary corner degeneracies and keep the corner reconstruction condition number bounded independently of $n$, `wakis/curvilinear/mappings3d.py` provides two multi-block conforming `HexMesh` generators:

1. **5-block butterfly (O-grid) cylinder** (`butterfly_disk_nodes(n_core, n_arc=None, nz=None, a=1.0, length=1.2)`):
   The circular cross-section of radius $a$ [m] is partitioned into a central $n_{\text{core}} \times n_{\text{core}}$ Cartesian square core $[-b, b]^2$ of half-width $b = \alpha a / \sqrt{2}$ [m] (with $\alpha = 0.65$) and four $n_{\text{core}} \times n_{\text{arc}}$ curved blocks (East $+x$, North $+y$, West $-x$, South $-y$) whose outer arc corners lie at $(\pm a/\sqrt{2}, \pm a/\sqrt{2})$ on $r = a$ along the four rays $\theta = \pm\pi/4, \pm 3\pi/4$. For each curved block $m \in \{0, 1, 2, 3\}$ with azimuthal coordinate $s \in [-1, 1]$, radial blend coordinate $t \in [0, 1]$, inner square side $\mathbf{x}_{\text{in},m}(s)$, and outer circular arc $\mathbf{x}_{\text{out},m}(s) = \big(a\cos\theta_m(s),\, a\sin\theta_m(s)\big)$ where $\theta_m(s) = m\frac{\pi}{2} + \frac{\pi}{4}s$ [rad], linear transfinite blending [Gordon & Hall, 1973] extruded along $z \in [0, L]$ [m] gives (derived by linear interpolation between the core perimeter and the circular arc at fixed $s$ and $z$):

```math
\mathbf{x}_m(s, t, z) = (1 - t)\,\mathbf{x}_{\text{in},m}(s) + t\,\mathbf{x}_{\text{out},m}(s) + (0, 0, z) \quad [\text{m}], \qquad
N_{\text{hex}} = 5 n^3 \quad (n_{\text{core}} = n_{\text{arc}} = n_z = n). \tag{5.1}
```

   Adjacent curved blocks meet along the four straight radial segments connecting $(\pm b, \pm b, z)$ to $(\pm a/\sqrt{2}, \pm a/\sqrt{2}, z)$. The minimum scaled corner Jacobian is $1/\sqrt{2} \approx 0.707107$ for every $n$, and the maximum corner reconstruction condition number is bounded by $4.492163 \le 4.517838$ across $n \in [6, 16]$ (`test_mesh_quality_bounded`). Boundary patches $0\dots 3$ label the four curved lateral walls and $4, 5$ label the flat end caps $z = 0$ and $z = L$; `sharp_nodes` marks only the top and bottom circular rims $(r = a, z \in \{0, L\})$ where the lateral wall meets the end caps at $90^\circ$ ($16 n$ sharp octants).

2. **7-block cubed-sphere ball** (`cubed_sphere_ball_nodes(n, a=1.0)`):
   The ball of radius $a$ [m] is partitioned into an $n \times n \times n$ inner Cartesian cube core $[-b, b]^3$ of half-width $b = \alpha a / \sqrt{3}$ [m] (with $\alpha = 0.56$) and six $n \times n \times n$ radially blended gnomonic shells ($m \in \{0, \dots, 5\}$ for $\pm x, \pm y, \pm z$). Each cube face $m$ is parameterised by $(u, v) \in [-1, 1]^2$ with cube-surface point $\mathbf{p}_m(u, v) \in [-1, 1]^3$ satisfying $\max_i |p_{m,i}| = 1$. Under the equidistant gnomonic projection onto $r = a$ [m], the inner core face point $\mathbf{x}_{\text{in},m}(u,v) = b\,\mathbf{p}_m(u,v)$ and outer sphere point $\mathbf{x}_{\text{out},m}(u,v) = a\,\mathbf{p}_m(u,v)/\|\mathbf{p}_m(u,v)\|$ are collinear along the ray $\mathbf{p}_m(u,v)$, so linear radial blending in $t \in [0, 1]$ gives (derived by linear interpolation along the radial ray $\mathbf{p}_m(u,v)$ from cube radius $b\|\mathbf{p}_m\|$ to sphere radius $a$):

```math
\mathbf{x}_m(u, v, t) = \left[(1 - t)\,b + t\,\frac{a}{\sqrt{1 + u^2 + v^2}}\right]\mathbf{p}_m(u, v) \quad [\text{m}], \qquad
N_{\text{hex}} = 7 n^3. \tag{5.2}
```

   The minimum scaled corner Jacobian equals $1/\sqrt{3} \approx 0.577350$ at the eight inner cube corners $(|u|, |v|, t) = (1, 1, 0)$ for all $n$, and the maximum corner reconstruction condition number stays bounded at $3.336057 \le 3.347598$ across $n \in [6, 16]$ (`test_mesh_quality_bounded`). Boundary patches $0\dots 5$ label the six outer shell faces, and `sharp_nodes` marks only the 8 cube-corner vertices on $r = a$ where three shells meet at $120^\circ$ ($24$ sharp octants across all $n$).

![Multi-block hexahedral meshes](../studies/curvilinear3d/figures/hexmesh_meshes.png)

#### 6. Results

The study script `studies/curvilinear3d/study_hexmesh.py` compares `CurvilinearFIT3D` (`_pec_reconstruction="constrained"`) on `butterfly_disk_nodes` ($a = 1.0\;\text{m}$, $L = 1.2\;\text{m}$, modes $\text{TM}_{010}$ with $\omega_{\text{ana}} = c x_{01}/a = 7.209486\times 10^8\;\text{rad/s}$ and $\text{TE}_{111}$ with $\omega_{\text{ana}} = c\sqrt{(x'_{11}/a)^2 + (\pi/L)^2} = 9.595162\times 10^8\;\text{rad/s}$ at $n = 4, 6, 8, 12$) and on `cubed_sphere_ball_nodes` ($a = 1.0\;\text{m}$, modes $\text{TE}_{l=1}$ from $\frac{d}{dx}[x j_1(x)] = 0$ with $\omega_{\text{ana}} = 8.225427\times 10^8\;\text{rad/s}$ and $\text{TM}_{l=1}$ from $j_1(x) = 0$ with $\omega_{\text{ana}} = 1.347090\times 10^9\;\text{rad/s}$, the second root returned by `sphere_lowest_omegas(a=1.0, l=1)` as selected by `_select_sphere_tm_l1`, at $n = 6, 8, 12, 16$) against `SolverFIT3D` staircase (`conformal=False`) and `SolverFIT3D` conformal (`conformal="pec"`) on a Cartesian bounding-box grid whose per-axis cell count $N_{\text{cart}}$ is chosen so that its active electric-edge unknown count $N_u$ is closest to the curvilinear run's $N_u$. All 48 runs were executed serially on this VM and committed to `studies/curvilinear3d/results/hexmesh_study.json` and `studies/curvilinear3d/results/hexmesh_table.md`:

### Per-run cavity eigenfrequency records

| Geometry | Mode | Method | `n` (`N_cart`) | `n_unknowns` (`n_hex`) | `h` [m] | `omega_disc` [rad/s] | `omega_ana` [rad/s] | `rel_err` | `wall_s` [s] |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `butterfly_disk` | `TM010` | CurvilinearFIT3D | 4 | 748 | 0.250000 | 7.263006e+08 | 7.209486e+08 | +7.4237e-03 | 0.10 |
| `butterfly_disk` | `TM010` | CurvilinearFIT3D | 6 | 2754 | 0.166667 | 7.238285e+08 | 7.209486e+08 | +3.9946e-03 | 0.72 |
| `butterfly_disk` | `TM010` | CurvilinearFIT3D | 8 | 6808 | 0.125000 | 7.227064e+08 | 7.209486e+08 | +2.4382e-03 | 3.52 |
| `butterfly_disk` | `TM010` | CurvilinearFIT3D | 12 | 23940 | 0.083333 | 7.217854e+08 | 7.209486e+08 | +1.1607e-03 | 50.61 |
| `butterfly_disk` | `TM010` | SolverFIT3D staircase | 4 (7) | 725 (748) | 0.285714 | 6.671895e+08 | 7.209486e+08 | -7.4567e-02 | 0.23 |
| `butterfly_disk` | `TM010` | SolverFIT3D staircase | 6 (11) | 3085 (2754) | 0.181818 | 6.730121e+08 | 7.209486e+08 | -6.6491e-02 | 0.40 |
| `butterfly_disk` | `TM010` | SolverFIT3D staircase | 8 (14) | 6448 (6808) | 0.142857 | 6.789720e+08 | 7.209486e+08 | -5.8224e-02 | 1.16 |
| `butterfly_disk` | `TM010` | SolverFIT3D staircase | 12 (22) | 25764 (23940) | 0.090909 | 6.844782e+08 | 7.209486e+08 | -5.0587e-02 | 14.25 |
| `butterfly_disk` | `TM010` | SolverFIT3D conformal | 4 (7) | 680 (748) | 0.285714 | 7.116848e+08 | 7.209486e+08 | -1.2849e-02 | 0.31 |
| `butterfly_disk` | `TM010` | SolverFIT3D conformal | 6 (11) | 2928 (2754) | 0.181818 | 7.173379e+08 | 7.209486e+08 | -5.0081e-03 | 0.54 |
| `butterfly_disk` | `TM010` | SolverFIT3D conformal | 8 (14) | 6138 (6808) | 0.142857 | 7.187511e+08 | 7.209486e+08 | -3.0480e-03 | 1.22 |
| `butterfly_disk` | `TM010` | SolverFIT3D conformal | 12 (22) | 24754 (23940) | 0.090909 | 7.201089e+08 | 7.209486e+08 | -1.1647e-03 | 13.99 |
| `butterfly_disk` | `TE111` | CurvilinearFIT3D | 4 | 748 | 0.250000 | 9.476038e+08 | 9.595162e+08 | -1.2415e-02 | 0.10 |
| `butterfly_disk` | `TE111` | CurvilinearFIT3D | 6 | 2754 | 0.166667 | 9.546274e+08 | 9.595162e+08 | -5.0951e-03 | 0.72 |
| `butterfly_disk` | `TE111` | CurvilinearFIT3D | 8 | 6808 | 0.125000 | 9.568782e+08 | 9.595162e+08 | -2.7493e-03 | 3.52 |
| `butterfly_disk` | `TE111` | CurvilinearFIT3D | 12 | 23940 | 0.083333 | 9.583928e+08 | 9.595162e+08 | -1.1708e-03 | 50.61 |
| `butterfly_disk` | `TE111` | SolverFIT3D staircase | 4 (7) | 725 (748) | 0.285714 | 9.487250e+08 | 9.595162e+08 | -1.1246e-02 | 0.23 |
| `butterfly_disk` | `TE111` | SolverFIT3D staircase | 6 (11) | 3085 (2754) | 0.181818 | 1.000144e+09 | 9.595162e+08 | +4.2342e-02 | 0.40 |
| `butterfly_disk` | `TE111` | SolverFIT3D staircase | 8 (14) | 6448 (6808) | 0.142857 | 9.183673e+08 | 9.595162e+08 | -4.2885e-02 | 1.16 |
| `butterfly_disk` | `TE111` | SolverFIT3D staircase | 12 (22) | 25764 (23940) | 0.090909 | 9.269897e+08 | 9.595162e+08 | -3.3899e-02 | 14.25 |
| `butterfly_disk` | `TE111` | SolverFIT3D conformal | 4 (7) | 680 (748) | 0.285714 | 9.510023e+08 | 9.595162e+08 | -8.8731e-03 | 0.31 |
| `butterfly_disk` | `TE111` | SolverFIT3D conformal | 6 (11) | 2928 (2754) | 0.181818 | 9.565641e+08 | 9.595162e+08 | -3.0766e-03 | 0.54 |
| `butterfly_disk` | `TE111` | SolverFIT3D conformal | 8 (14) | 6138 (6808) | 0.142857 | 9.575803e+08 | 9.595162e+08 | -2.0176e-03 | 1.22 |
| `butterfly_disk` | `TE111` | SolverFIT3D conformal | 12 (22) | 24754 (23940) | 0.090909 | 9.587333e+08 | 9.595162e+08 | -8.1589e-04 | 13.99 |
| `cubed_sphere` | `TE_l1` | CurvilinearFIT3D | 6 | 4350 | 0.166667 | 8.368673e+08 | 8.225427e+08 | +1.7415e-02 | 2.61 |
| `cubed_sphere` | `TE_l1` | CurvilinearFIT3D | 8 | 10408 | 0.125000 | 8.310117e+08 | 8.225427e+08 | +1.0296e-02 | 14.78 |
| `cubed_sphere` | `TE_l1` | CurvilinearFIT3D | 12 | 35484 | 0.083333 | 8.264671e+08 | 8.225427e+08 | +4.7710e-03 | 107.00 |
| `cubed_sphere` | `TE_l1` | CurvilinearFIT3D | 16 | 84560 | 0.062500 | 8.247818e+08 | 8.225427e+08 | +2.7221e-03 | 976.57 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D staircase | 6 (13) | 4221 (4350) | 0.153846 | 9.148980e+08 | 8.225427e+08 | +1.1228e-01 | 0.63 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D staircase | 8 (18) | 10896 (10408) | 0.111111 | 7.319054e+08 | 8.225427e+08 | -1.1019e-01 | 2.81 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D staircase | 12 (27) | 35103 (35484) | 0.074074 | 7.567743e+08 | 8.225427e+08 | -7.9958e-02 | 26.83 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D staircase | 16 (36) | 82290 (84560) | 0.055556 | 7.691562e+08 | 8.225427e+08 | -6.4904e-02 | 368.53 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D conformal | 6 (14) | 4530 (4350) | 0.142857 | 8.206089e+08 | 8.225427e+08 | -2.3511e-03 | 1.29 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D conformal | 8 (18) | 9654 (10408) | 0.111111 | 8.196669e+08 | 8.225427e+08 | -3.4962e-03 | 2.71 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D conformal | 12 (28) | 36204 (35484) | 0.071429 | 8.218722e+08 | 8.225427e+08 | -8.1521e-04 | 36.89 |
| `cubed_sphere` | `TE_l1` | SolverFIT3D conformal | 16 (37) | 82212 (84560) | 0.054054 | 8.217408e+08 | 8.225427e+08 | -9.7500e-04 | 134.94 |
| `cubed_sphere` | `TM_l1` | CurvilinearFIT3D | 6 | 4350 | 0.166667 | 1.393903e+09 | 1.347090e+09 | +3.4751e-02 | 2.61 |
| `cubed_sphere` | `TM_l1` | CurvilinearFIT3D | 8 | 10408 | 0.125000 | 1.374036e+09 | 1.347090e+09 | +2.0003e-02 | 14.78 |
| `cubed_sphere` | `TM_l1` | CurvilinearFIT3D | 12 | 35484 | 0.083333 | 1.359298e+09 | 1.347090e+09 | +9.0621e-03 | 107.00 |
| `cubed_sphere` | `TM_l1` | CurvilinearFIT3D | 16 | 84560 | 0.062500 | 1.354011e+09 | 1.347090e+09 | +5.1375e-03 | 976.57 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D staircase | 6 (13) | 4221 (4350) | 0.153846 | 1.343789e+09 | 1.347090e+09 | -2.4505e-03 | 0.63 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D staircase | 8 (18) | 10896 (10408) | 0.111111 | 1.272831e+09 | 1.347090e+09 | -5.5125e-02 | 2.81 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D staircase | 12 (27) | 35103 (35484) | 0.074074 | 1.321796e+09 | 1.347090e+09 | -1.8777e-02 | 26.83 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D staircase | 16 (36) | 82290 (84560) | 0.055556 | 1.272211e+09 | 1.347090e+09 | -5.5586e-02 | 368.53 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D conformal | 6 (14) | 4530 (4350) | 0.142857 | 1.335321e+09 | 1.347090e+09 | -8.7372e-03 | 1.29 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D conformal | 8 (18) | 9654 (10408) | 0.111111 | 1.353833e+09 | 1.347090e+09 | +5.0051e-03 | 2.71 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D conformal | 12 (28) | 36204 (35484) | 0.071429 | 1.346864e+09 | 1.347090e+09 | -1.6833e-04 | 36.89 |
| `cubed_sphere` | `TM_l1` | SolverFIT3D conformal | 16 (37) | 82212 (84560) | 0.054054 | 1.345542e+09 | 1.347090e+09 | -1.1495e-03 | 134.94 |

### Convergence orders and equal-wall-time comparison

| Geometry | Mode | Method | Resolutions `n` | LS order | Two-level order | Equal-wall-time (`wall_s <= curv finest`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `butterfly_disk` | `TM010` | CurvilinearFIT3D | `[4, 6, 8, 12]` | 1.69 | 1.83 | `n=12`, `rel_err=+1.1607e-03` (`50.61 s` <= `50.61 s`) |
| `butterfly_disk` | `TM010` | SolverFIT3D staircase | `[4, 6, 8, 12]` | 0.35 | 0.31 | `n=12 (N=22)`, `rel_err=-5.0587e-02` (`14.25 s` <= `50.61 s`) |
| `butterfly_disk` | `TM010` | SolverFIT3D conformal | `[4, 6, 8, 12]` | 2.09 | 2.13 | `n=12 (N=22)`, `rel_err=-1.1647e-03` (`13.99 s` <= `50.61 s`) |
| `butterfly_disk` | `TE111` | CurvilinearFIT3D | `[4, 6, 8, 12]` | 2.15 | 2.11 | `n=12`, `rel_err=-1.1708e-03` (`50.61 s` <= `50.61 s`) |
| `butterfly_disk` | `TE111` | SolverFIT3D staircase | `[4, 6, 8, 12]` | -0.92 | 0.52 | `n=12 (N=22)`, `rel_err=-3.3899e-02` (`14.25 s` <= `50.61 s`) |
| `butterfly_disk` | `TE111` | SolverFIT3D conformal | `[4, 6, 8, 12]` | 2.07 | 2.00 | `n=12 (N=22)`, `rel_err=-8.1589e-04` (`13.99 s` <= `50.61 s`) |
| `cubed_sphere` | `TE_l1` | CurvilinearFIT3D | `[6, 8, 12, 16]` | 1.89 | 1.95 | `n=16`, `rel_err=+2.7221e-03` (`976.57 s` <= `976.57 s`) |
| `cubed_sphere` | `TE_l1` | SolverFIT3D staircase | `[6, 8, 12, 16]` | 0.57 | 0.73 | `n=16 (N=36)`, `rel_err=-6.4904e-02` (`368.53 s` <= `976.57 s`) |
| `cubed_sphere` | `TE_l1` | SolverFIT3D conformal | `[6, 8, 12, 16]` | 1.32 | -0.64 | `n=16 (N=37)`, `rel_err=-9.7500e-04` (`134.94 s` <= `976.57 s`) |
| `cubed_sphere` | `TM_l1` | CurvilinearFIT3D | `[6, 8, 12, 16]` | 1.95 | 1.97 | `n=16`, `rel_err=+5.1375e-03` (`976.57 s` <= `976.57 s`) |
| `cubed_sphere` | `TM_l1` | SolverFIT3D staircase | `[6, 8, 12, 16]` | -2.31 | -3.77 | `n=16 (N=36)`, `rel_err=-5.5586e-02` (`368.53 s` <= `976.57 s`) |
| `cubed_sphere` | `TM_l1` | SolverFIT3D conformal | `[6, 8, 12, 16]` | 3.01 | -6.89 | `n=16 (N=37)`, `rel_err=-1.1495e-03` (`134.94 s` <= `976.57 s`) |

![Convergence of cavity eigenfrequencies](../studies/curvilinear3d/figures/hexmesh_convergence.png)

On both curved cavities and all four modes, `CurvilinearFIT3D` converges monotonically with least-squares (LS) log-log slopes of `1.69` (two-level `1.83`) on `butterfly_disk` `TM010`, `2.15` (two-level `2.11`) on `butterfly_disk` `TE111`, `1.89` (two-level `1.95`) on `cubed_sphere` `TE_l1`, and `1.95` (two-level `1.97`) on `cubed_sphere` `TM_l1`. By contrast, on the single-block elliptic mappings (`studies/curvilinear3d/results/cavities_w3curvhex_batch_w3study-cav.json`), the corner Jacobian degeneracy degraded the TE slopes to `-2.56` (`cylinder` `TE111`) and `-1.38` (`sphere` `TE_l1`); replacing the single-block elliptic maps with the 5-block butterfly disk and 7-block cubed-sphere restores monotonic near-second-order convergence across both TM and TE polarisations.

#### 7. Limits & Failure Modes

* **Single-block and conforming-hex-mesh PEC only.** `CurvilinearFIT3D` enforces PEC boundary conditions on the exterior boundary faces of a single structured block (`HexBlock3D`) or a conforming hexahedral mesh (`HexMesh`, `boundary_face_mask`). Interior PEC obstacles, PMC walls, periodic or Bloch boundaries, waveguide/lumped ports, absorbing boundary conditions or PMLs, dispersive ADE media, charged-particle or beam coupling, GPU kernels, and MPI domain decomposition are not implemented.
* **Sharp octants set $\mathbf{D}_{h,c} = \mathbf{0}$ and the `1e-12` cross-product test.** By Eq. (2.4), at any boundary node where incident boundary faces meet at a non-smooth dihedral angle (two or more non-parallel wall normals, or `sharp_nodes[v] == True`), the constrained electric reconstruction sets $\mathbf{Q}_{h,c}^{\text{sharp}} = \mathbf{0}$ ($\mathbf{D}_{h,c} = \mathbf{0}$). On `butterfly_disk_nodes`, only the $8n$ nodes on the top and bottom circular rims $(r = a, z \in \{0, L\})$ are sharp ($16n$ sharp octants), and on `cubed_sphere_ball_nodes`, only the $8$ cube-corner vertices on $r = a$ are sharp ($24$ sharp octants for every $n$). Because each sharp octant has volume $V_{h,c} = V_h / 8 = O(h^3)$, zeroing $\mathbf{D}_{h,c}$ on those octants removes an $O(h^2)$ volume fraction along the cylinder rims and an $O(h^3)$ volume fraction at the $8$ sphere corners, shifting the pre-asymptotic `butterfly_disk` `TM010` LS slope to `1.69` (`1.83` between $n = 8$ and $n = 12$) while preserving asymptotic second-order convergence (`1.89`–`2.15` on the remaining modes). On a single-block sphere (`sphere_nodes(6, 6, 6, a=1.0)` wrapped via `HexMesh.from_block`), multi-wall boundary nodes are classified as sharp by a $\|\hat{\mathbf{n}}_{\text{ref}} \times \hat{\mathbf{n}}_k\| \le 10^{-12}$ cross-product test on the per-wall node normals (`128` sharp octants and `600` smooth wall octants at $N = 6$ in `test_pec_constraint_wall_octants`) that is not checked against the smooth sphere geometry. If `boundary_patch` is omitted on a curved wall so that every boundary facet gets its own patch label (`_ignore_sharp_nodes=True` or `boundary_patch=None`), every multi-face boundary node is misclassified as sharp (`n_sharp_octants` jumps from `64` to `112` on `butterfly_disk(4)` and from `24` to `168` on `cubed_sphere_ball(4)` in `test_hexmesh_conformity_and_orientation`), zeroing $\mathbf{D}_{h,c}$ across the entire curved boundary shell.
* **Magnetic wall check ($\mathbf{B}\cdot\hat{\mathbf{n}}$ on curved facets: Cauchy–Schwarz facet-angle bound vs eigenmode $m_3$ decay).** On a PEC boundary face $f_w \in \partial\Omega$ with outward unit normal $\hat{\mathbf{n}}_{f_w} = s_{\text{out},f_w}\mathbf{A}_{f_w}/\|\mathbf{A}_{f_w}\|$, setting the boundary face flux $b_{f_w} = 0$ makes $\mathbf{A}_{f_w}\cdot\mathbf{B}_{h,c} = s_{f_w} b_{f_w} = 0$ by Eq. (2.1), so the face-level normal ratio $m_0 = |\mathbf{B}_{h,c}\cdot\hat{\mathbf{n}}_{f_w}|/\|\mathbf{B}_{h,c}\|$ vanishes to machine precision (`1.767e-16` on 1-wall octants and `2.110e-16` across all wall octants of `sphere_nodes(6, 6, 6, a=1.0)` in `test_pec_constraint_wall_octants`, and `<= 2.384e-16` across all meshes in `studies/curvilinear3d/bn_wall_diagnosis.py`). Decomposing the area-summed unit node normal $\hat{\mathbf{n}}_v = (\hat{\mathbf{n}}_v\cdot\hat{\mathbf{n}}_{f_w})\hat{\mathbf{n}}_{f_w} + \hat{\mathbf{n}}_{f_w}\times(\hat{\mathbf{n}}_v\times\hat{\mathbf{n}}_{f_w})$ yields the octant bound (derived from the triangle and Cauchy–Schwarz inequalities on the orthogonal complement):

```math
m_1 = \frac{|\mathbf{B}_{h,c}\cdot\hat{\mathbf{n}}_v|}{\|\mathbf{B}_{h,c}\|} \le m_0 + \|\hat{\mathbf{n}}_v \times \hat{\mathbf{n}}_{f_w}\| = m_0 + g \le g + 10^{-12}. \tag{7.1}
```

  This "identity" $m_1 \le m_0 + g$ is Cauchy–Schwarz: it holds identically for any non-zero vector $\mathbf{B}_{h,c} \in \mathbb{R}^3$, so asserting $m_1 \le g + 10^{-12}$ carries no test content beyond $m_0 = 0$ ($m_0 \le 10^{-12}$). Its role is purely geometric—explaining why a random face-flux vector with $b_{f_w} = 0$ saturates $m_1 \approx g$ inside a single corner octant whose facet normal $\hat{\mathbf{n}}_{f_w}$ is tilted by angle $\arcsin g$ relative to $\hat{\mathbf{n}}_v$. Across every boundary octant of `sphere_nodes(N, N, N, a=1.0)`, `cubed_sphere_ball_nodes(n)`, `butterfly_disk_nodes(n)`, and `perturbed_box_nodes(6, 5, 4)`, `studies/curvilinear3d/bn_wall_diagnosis.py` (`studies/curvilinear3d/results/bn_wall_diagnosis.json` and `test_wall_bn_facet_node_angle_identity` in `tests/test_114_curvilinear3d_diagnostics.py`) prints $m_1 - g \le 0 \le 10^{-12}$ (`max_m1_minus_g_globally = 0.0`, `max_m0_globally = 2.384279e-16`; `-3.398e-07` over 1-wall octants on `sphere_nodes(6, 6, 6)`). On `sphere_nodes(N, N, N, a=1.0)` ($N \in \{4, 6, 8, 12, 16\}$), the 1-wall octant maxima are $g = \text{2.651614e-01}$ ($m_1 = \text{2.651581e-01}$ at $N = 4$), $g = \text{1.845888e-01}$ ($m_1 = \text{1.845750e-01}$ at $N = 6$), $g = \text{1.414062e-01}$ ($m_1 = \text{1.414061e-01}$ at $N = 8$), $g = \text{9.634486e-02}$ ($m_1 = \text{9.634485e-02}$ at $N = 12$), and $g = \text{7.310638e-02}$ ($m_1 = \text{7.310631e-02}$ at $N = 16$), decaying at first order with LS slope `0.9307` (`0.9218` over $N \in \{4, 6, 8, 12\}$) and concentrating at the 24 `smooth_wall` nodes adjacent to the 8 singular cube corners $(|u|, |v|, |w|) = (1, 1, 1 - 2/N)$ where the elliptic cube-to-ball facets are most skewed. On `cubed_sphere_ball_nodes(n)` ($n \in \{4, 6, 8\}$), 1-wall $g$ decreases from `3.029054e-01` ($m_1 = \text{3.028829e-01}$ at $n = 4$) to `2.184788e-01` ($m_1 = \text{2.184310e-01}$ at $n = 6$) and `1.691020e-01` ($m_1 = \text{1.687516e-01}$ at $n = 8$, LS slope `0.8387` for $g$ and `0.8414` for $m_1$); on `butterfly_disk_nodes(n)` ($n \in \{4, 6, 8\}$), 1-wall $g$ decreases from `1.950903e-01` ($m_1 = \text{1.894174e-01}$ at $n = 4$) to `1.305262e-01` ($m_1 = \text{1.272257e-01}$ at $n = 6$) and `9.801714e-02` ($m_1 = \text{9.616895e-02}$ at $n = 8$, LS slope `0.9929` for $g$ and `0.9782` for $m_1$); and on the flat-wall control `perturbed_box_nodes(6, 5, 4)`, $m_0 = m_1 = g = \text{0.000e+00}$. The physical evidence that the discrete magnetic field satisfies $\mathbf{B}\cdot\hat{\mathbf{n}} \to 0$ on curved PEC walls is instead the decay with resolution $n$ of the eigenmode node-averaged normal residual $m_3 = |\mathbf{B}_v\cdot\hat{\mathbf{n}}|/\max_w\|\mathbf{B}_w\|$ for the corner-volume-weighted node-averaged magnetic field $\mathbf{B}_v = (\sum_{c \ni v} V_{h,c}\mathbf{B}_{h,c}) / \sum_{c \ni v} V_{h,c}$ of smooth cavity eigenmodes ($\mathbf{b} = \mathbf{C}_u \mathbf{W}_{\varepsilon,u}\mathbf{d}$), evaluated on `smooth_wall` nodes for both $\hat{\mathbf{n}} = \hat{\mathbf{n}}_v$ and $\hat{\mathbf{n}} = \hat{\mathbf{n}}_{\text{exact}}$ (`test_wall_bn_cavity_eigenmode_m3_decay` and `studies/curvilinear3d/bn_wall_diagnosis.py`):
  * `cubed_sphere_ball_nodes(n)` ($n = 4, 6, 8$): lowest TM `m3(n_v) = 5.0124e-03, 1.1154e-03, 6.3948e-04` (LS order `3.0188`, `3.7060` on $4 \to 6$) and `m3(n_exact) = 4.9569e-02, 1.1389e-02, 6.5057e-03` (LS order `2.9754`, `3.6272` on $4 \to 6$); lowest TE `m3(n_v) = 6.9198e-03, 3.4967e-03, 2.0182e-03` (LS order `1.7714`, `1.6834` on $4 \to 6$) and `m3(n_exact) = 8.0907e-02, 3.6502e-02, 2.1063e-02` (LS order `1.9430`, `1.9630` on $4 \to 6$).
  * `butterfly_disk_nodes(n)` ($n = 4, 6, 8$): lowest TM `m3(n_v) = m3(n_exact) = 1.8321e-02, 1.0796e-02, 6.9304e-03` (LS order `1.3961`, `1.3044` on $4 \to 6$); lowest TE `m3(n_v) = m3(n_exact) = 6.5176e-03, 4.5951e-03, 3.1474e-03` (LS order `1.0378`, `0.8620` on $4 \to 6$).
  * `sphere_nodes(N, N, N, a=1.0)` ($N = 4, 6, 8, 12, 16$): lowest TM `m3(n_v) = 1.5014e-02, 1.9583e-02, 1.0944e-02, 1.1429e-02, 5.3435e-03` (LS order `0.7228`) and `m3(n_exact) = 1.3962e-02, 1.9506e-02, 1.1230e-02, 1.1368e-02, 5.2090e-03` (LS order `0.6950`); lowest TE `m3(n_v) = 2.2280e-02, 2.0194e-02, 1.4288e-02, 1.1376e-02, 6.0915e-03` (LS order `0.8994`) and `m3(n_exact) = 1.8898e-02, 1.9728e-02, 1.4886e-02, 1.2129e-02, 5.7770e-03` (LS order `0.8054`).

  ![Wall B.n diagnosis and eigenmode m3 decay](../img/curvilinear3d_bn_wall_diagnosis.png)

* **Perturbed-box convergence order (`1.80` vs `1.999` reconciliation and error decomposition) (updated 2026-11-16).** The earlier `1.80` (`D_drop_only = 1.795576`) came from a results file (`studies/curvilinear3d/results/cavities_w3curvhex_batch_w3study-cav.json`) generated before the PEC reconstruction fix (`_pec_reconstruction="drop_only"`); the released code (`_pec_reconstruction="constrained"`) gives the value printed by `studies/curvilinear3d/perturbed_box_order.py` (`studies/curvilinear3d/results/perturbed_box_order.json`), namely `D = 1.974416` on `F1` $n \in \{6, 8, 12, 16\}$ and `LS(e) = 1.981613` across $n \in \{4, 6, 8, 12, 16, 24\}$. On the rectangular PEC cavity $[0, 1.0]\times[0, 0.7]\times[0, 0.8]\;\text{m}^3$ (`omega_analytic_rad_s = 1.50765688e9` rad/s), `studies/curvilinear3d/perturbed_box_order.py` decomposes the signed relative $\text{TE}_{101}$ error $e(N) = (\omega_h - \omega_{\text{ana}})/\omega_{\text{ana}}$ into the unperturbed Cartesian Yee dispersion error $e_{\text{Yee}}(N) = e(\text{amp}=0) < 0$ and the mesh-distortion shift $e_{\text{dist}}(N) = e(\text{amp}) - e(0) > 0$ across three mesh families (`F1` isotropic $(n, n, n)$ with cell-scaled $\text{amp} = 0.3$, $n \in \{4, 6, 8, 12, 16, 24\}$; `F2` aspect-matched $(4k, 3k, 4k)$ with cell-scaled $\text{amp} = 0.3$, $n = 4k \in \{4, 8, 12, 16, 24\}$; `F3` isotropic $(n, n, n)$ with physical displacement amplitude fixed in metres at the `F1` $n = 6$ value, $n \in \{6, 8, 12, 16, 24\}$). Under the canonical definition **D** (LS slope of $\ln|e|$ vs $\ln n$ on `F1` $n \in \{6, 8, 12, 16\}$, $\text{amp} = 0.3$ cell mode, `compute_d_order`), the released default `_pec_reconstruction="constrained"` achieves `D = 1.974416 >= 1.8` (`test_perturbed_box_order_small`), with signed errors `-1.106566e-02, -6.283780e-03, -2.823016e-03, -1.595393e-03` (`-7.116751e-04` at $n = 24$) and local orders `1.9670, 1.9735, 1.9837` (`1.9909` on $16 \to 24$) rising monotonically toward `2.0`: here $e_{\text{Yee}}$ (`-1.138407e-02, -6.413149e-03, -2.853343e-03, -1.605607e-03`, `-7.137942e-04` at $n = 24$, LS order `1.997050` on $\{6, 8, 12, 16\}$ and `1.996136` on $\{4, \dots, 24\}$) is only slightly reduced in magnitude by the opposite-sign interior distortion term $e_{\text{dist}}$ (`+3.184109e-04, +1.293693e-04, +3.032640e-05, +1.021405e-05`, `+2.119075e-06` at $n = 24$, LS order `3.517161` on $\{6, 8, 12, 16\}`, `3.404419` on $\{6, 8, 12\}` in `test_perturbed_box_error_decomposition`, and `3.247677` on $\{4, \dots, 24\}`). On `F2` ($n \in \{4, 8, 12, 16, 24\}$), `LS(e) = 1.987856` and `LS(e_dist) = 2.901954`, and on $k = 1 \to 2$ ($4\times 3\times 4 \to 8\times 6\times 8$) the legacy two-level estimator gives `1.999214` (`e(amp) = -2.511923e-02 -> -6.283230e-03`, `e_Yee = -2.550464e-02 -> -6.413149e-03`, `e_dist = +3.854070e-04 -> +1.299192e-04`; two-level `F1` at $n = 4 \to 8$ gives `1.986582`), while on `F3` $n \in \{6, 8, 12, 16, 24\}$ `LS(e) = 2.012530` and `LS(e_dist) = 1.621745` (`e(amp) = -1.106566e-02, -6.183393e-03, -2.731579e-03, -1.532492e-03, -6.795790e-04`, `e_dist = +3.184109e-04, +2.297557e-04, +1.217636e-04, +7.311454e-05, +3.421517e-05`). By contrast, the `1.795576` ("1.80") order in `studies/curvilinear3d/results/cavities_w3curvhex_batch_w3study-cav.json` was generated under the pre-PEC-fix `_pec_reconstruction="drop_only"` rule (`D_drop_only = 1.795576`, signed errors `-8.155933e-03, -4.889810e-03, -2.371544e-03, -1.399003e-03`, local orders `1.7783, 1.7846, 1.8346`, `LS(e_dist) = 2.807907`, and `1.957555` on `F2` $k = 1 \to 2$), where unconstrained boundary dual-face skew inflates $e_{\text{dist}}$ at $n = 6$ by `10.14x` (`+3.228137e-03` vs `+3.184109e-04`) and partially cancels $e_{\text{Yee}} < 0$. Thus `0.178840` of the `0.203638` difference between `1.795576` and `1.999214` is accounted for by the `"drop_only"` vs `"constrained"` PEC boundary reconstruction, and the remaining `0.024798` (`D = 1.974416` vs `1.999214`) comes from the opposite-sign $e_{\text{Yee}} + e_{\text{dist}}$ cancellation across mesh family (`F1` vs `F2`) and refinement window (`{6, 8, 12, 16}` vs `k = 1 -> 2`).

  ![Perturbed-box TE101 error decomposition](../img/curvilinear3d_perturbed_box_order.png)
* **Comparison to conformal FIT (`SolverFIT3D(conformal="pec")`) at equal unknowns and equal wall time.**
  * *Equal unknowns:* At the finest butterfly-disk resolution ($n = 12$, $N_u = 23940$ vs $N_u = 24754$ at $N_{\text{cart}} = 22$), `CurvilinearFIT3D` and `SolverFIT3D conformal` achieve nearly identical accuracy on $\text{TM}_{010}$ (`+1.1607e-03` vs `-1.1647e-03`) and comparable accuracy on $\text{TE}_{111}$ (`-1.1708e-03` vs `-8.1589e-04`), while `SolverFIT3D staircase` stalls at `-5.0587e-02` and `-3.3899e-02`. On the cubed-sphere ball at $n = 16$ ($N_u = 84560$ vs $N_u = 82212$ at $N_{\text{cart}} = 37$), `CurvilinearFIT3D` converges smoothly with two-level orders `1.95` ($\text{TE}_{l=1}$, error `+2.7221e-03`) and `1.97` ($\text{TM}_{l=1}$, error `+5.1375e-03`), whereas `SolverFIT3D conformal` reaches smaller absolute errors (`-9.7500e-04` and `-1.1495e-03`) but exhibits non-monotone cut-cell fluctuations as the sphere surface crosses grid planes (two-level slopes `-0.64` and `-6.89` from $N_{\text{cart}} = 28 \to 37$, where $\text{TM}_{l=1}$ moves from `-1.6833e-04` to `-1.1495e-03`).
  * *Equal wall time:* At its finest grid, `CurvilinearFIT3D` is slower than `SolverFIT3D conformal`: `50.61 s` vs `13.99 s` ($3.6\times$ slower) at $n = 12$ on the butterfly disk and `976.57 s` vs `134.94 s` ($7.2\times$ slower) at $n = 16$ on the cubed sphere, with the cause not measured. Consequently, under the equal-wall-time criterion ($\text{wall\_s} \le \text{wall\_s}_{\text{curv,finest}}$), `SolverFIT3D conformal` already finishes its finest grid ($N_{\text{cart}} = 22$ in `13.99 s <= 50.61 s` and $N_{\text{cart}} = 37$ in `134.94 s <= 976.57 s`) with equal or lower error (`-1.1647e-03`, `-8.1589e-04`, `-9.7500e-04`, `-1.1495e-03` vs `+1.1607e-03`, `-1.1708e-03`, `+2.7221e-03`, `+5.1375e-03`).
* **Tetrahedral FIT not compared.** This study compares 3D curvilinear hexahedral FIT only against Cartesian staircase and conformal `SolverFIT3D`; a direct equal-unknowns and equal-wall-time comparison against 3D tetrahedral FIT (`UnstructuredFIT3D` in `wakis.unstructured`) on the same cylinder and sphere cavities is not included here.

References:
- T. Weiland, "A discretization method for the solution of Maxwell's equations for six-component fields", *Electronics and Communication (AEÜ)* **31**, 116–120 (1977).
- R. Holland, "Finite-difference solution of Maxwell's equations in generalized nonorthogonal coordinates", *IEEE Trans. Nucl. Sci.* **30**(6), 4589–4591 (1983).
- B. Krietenstein, R. Schuhmann, P. Thoma and T. Weiland, "The Perfect Boundary Approximation technique facing the big challenge of high precision field computation", *Proc. LINAC'98*, Chicago, pp. 860–862 (1998), [JACoW TH4041](https://epaper.kek.jp/l98/PAPERS/TH4041.PDF).
- R. Schuhmann and T. Weiland, "The nonorthogonal finite integration technique applied to 2D- and 3D-eigenvalue problems", *IEEE Trans. Magn.* **36**(4), 897–901 (2000).
- M. Clemens and T. Weiland, "Discrete electromagnetism with the finite integration technique", *Prog. Electromagn. Res.* **32**, 65–87 (2001), doi:10.2528/PIER00080103.
- S. J. Cooke, R. Shtokhamer, A. A. Mondelli and B. Levush, "A finite integration method for conformal, structured-grid, electromagnetic simulation", *J. Comput. Phys.* **215**, 321–347 (2006).
- W. J. Gordon and C. A. Hall, "Construction of curvilinear co-ordinate systems and applications to mesh generation", *Int. J. Numer. Methods Eng.* **7**, 461–477 (1973), doi:10.1002/nme.1620070405.
