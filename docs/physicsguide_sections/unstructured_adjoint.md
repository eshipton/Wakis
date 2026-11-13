# Exact discrete adjoint of the unstructured FIT solvers

`wakis.unstructured.adjoint` differentiates the time stepping of `UnstructuredFIT2D` (TM and TE) and `UnstructuredFIT3D` (Voronoi and Whitney Hodge) with respect to the relative permittivity of each mesh cell. As in the Cartesian `wakis.inverse` (physics guide, "Inverse design"), the *discrete* step is differentiated (discretize-then-differentiate) [Giles & Pierce, 2000]. The gradient is exact for the simulated fields up to rounding.

* `UnstructuredAdjointStepper(solver, design_cells)` holds a pure-function copy of `one_step` and its exact transpose.
* `CellMaterialGrid` maps a density on the design triangles or tetrahedra to their permittivity, through a cone or Helmholtz filter and a tanh projection.
* `UnstructuredAdjointSimulation` runs the forward and backward sweeps with stored rows or binomial checkpointing. It is duck-type compatible with `wakis.inverse.jax_simulation` (a `jax.custom_vjp`).

## 1. Setting

Let $`\mathbf x^n`$ be the full state after $`n`$ steps. It is a dictionary of 1D arrays and holds:

* the electric unknown $`\mathbf u`$: $`E_z`$ (TM) or $`H_z`$ (TE) on the free nodes in 2D, the voltages on the free edges in 3D;
* the primal-edge flux $`\mathbf v`$ (2D) or the face flux $`\mathbf b`$ (3D);
* the ADE polarizations of every dispersive group;
* with the 2D radial PML, the node flux $`\mathbf q`$ of the layer and the four memory arrays of each of its four CFS filters.

Every update of the solvers is linear in the state with design-dependent but state-independent coefficients:

```math
\mathbf x^{n+1} = \mathbf A(\boldsymbol\varepsilon)\,\mathbf x^n + \mathbf s^n,\qquad \mathbf x^0 = 0 .
```

A monitor records DFT samples of interpolated node or edge fields. With the kernel and time levels of `Recorder.add_dft` (field at the end of step $`n`$, $`t_{n+1} = (n+1)\Delta t`$):

```math
X_{fj} = \sum_{n=0}^{N_t-1} c_{fn}\,(\mathbf W\mathbf u^{n+1})_j,\qquad c_{fn} = \Delta t\, e^{-2\pi i f t_{n+1}} .
```

For a real objective $`F(X)`$, write $`G = \partial F/\partial\,\mathrm{Re}X + i\,\partial F/\partial\,\mathrm{Im}X`$, so that $`dF = \mathrm{Re}\sum \bar G\,dX`$. Reverse-mode differentiation of the recursion gives the adjoint sweep

```math
\boldsymbol\lambda^{N_t} = 0,\qquad
\boldsymbol\lambda^{n} = \mathbf A^T\big(\boldsymbol\lambda^{n+1} + \mathbf e_u\,\mathbf W^T\,\mathrm{Re}\textstyle\sum_f \bar c_{fn} G_{f\cdot}\big),
```

where $`\mathbf e_u`$ places the adjoint source on the $`\mathbf u`$ block. The source is injected at the output level $`n+1`$ of step $`n`$, before that step is transposed: the time-level centring of the monitor. Within a step, every coefficient $`\theta`$ that depends on $`\varepsilon`$ contributes

```math
\frac{dF}{d\theta} \mathrel{+}= \bar{\boldsymbol\lambda}_{\text{internal}}\cdot\frac{\partial(\text{step output})}{\partial\theta}\Big|_{\mathbf x^n} ,
```

i.e. an elementwise product of an adjoint quantity of the transposed step ("sens") and a forward quantity of the same step ("row"). The rows are stored or recomputed (Section 6). The sources $`\mathbf s^n`$ do not depend on $`\varepsilon`$ (Section 8), so they drop out of the transpose.

## 2. Transposed updates

Notation: $`\mathbf G`$ is the edge–node incidence on the free nodes (2D), and $`\mathbf C_f`$, $`\mathbf C_s`$ are the face–edge incidences of the free and massless edges (3D). Bars denote adjoints; inputs of a sub-update receive the transposed contributions of its outputs.

### 2.1 Capacitive update (2D TM, 3D Voronoi) with ADEs

`CapacitiveUpdate` solves, per dof $`k`$ and dispersive group $`g`$ (dofs $`I_g`$, vacuum capacitances $`c_g`$):

```math
\begin{aligned}
\text{acc} &= a_1\,u + \Delta t\,r - \sum_g c_g\Big[\sum_L (p'_L-p_L) + \sum_D(\kappa_D-1)p_D\Big],\\
p'_L &= \alpha p_L + \xi p_{L,\text{old}} + \zeta u_{I_g},\qquad p'_{L,\text{old}} = p_L,\\
u' &= \text{acc}/d,\qquad p'_D = \kappa_D p_D + \beta_D (u'_{I_g} + u_{I_g}),
\end{aligned}
```

with $`a_1 = C_0 - G\Delta t/2 - \sum c\beta`$ and $`d = C_0 + G\Delta t/2 + \sum c\beta`$. Here $`C_0`$ is the capacitance (node mass $`M_\varepsilon`$ in 2D, $`\varepsilon_0 A_e/L_e`$ in 3D), $`G`$ the conductance (conductivity, Silver–Müller boundary, absorber), and $`L`$, $`D`$ index the Lorentz/Drude and Debye poles. The transpose, in reverse order:

```math
\begin{aligned}
\bar p_D &= \kappa_D \bar p'_D,\quad \bar u'_{I_g} \mathrel{+}= \beta_D\bar p'_D,\quad \bar u_{I_g} \mathrel{+}= \beta_D\bar p'_D \quad\text{(Debye feedback)},\\
\overline{\text{acc}} &= \bar u'/d,\quad \bar u \mathrel{+}= a_1\overline{\text{acc}},\quad \bar r = \Delta t\,\overline{\text{acc}},\quad \overline{\text{corr}} = -c_g\overline{\text{acc}}_{I_g},\\
\bar p_D &\mathrel{+}= (\kappa_D-1)\overline{\text{corr}},\quad
\bar p'_{L,\text{tot}} = \bar p'_L + \overline{\text{corr}},\\
\bar p_L &= \bar p'_{L,\text{old}} - \overline{\text{corr}} + \alpha\bar p'_{L,\text{tot}},\quad \bar p_{L,\text{old}} = \xi\bar p'_{L,\text{tot}},\quad \bar u_{I_g} \mathrel{+}= \zeta\bar p'_{L,\text{tot}} .
\end{aligned}
```

The Debye path is the only intra-step feedback: $`p'_D`$ reads the *new* $`u'`$. Its adjoint therefore has to be added to $`\bar u'`$ before $`\bar u'`$ is divided by $`d`$, and the design sensitivity uses that total $`\bar u'`$. Dropping it is one of the mutations of the tests (dot error 0.11).

### 2.2 2D TM step

```math
\mathbf v' = \frac{(1-c_e)\mathbf v - \Delta t\,\mathbf G\,r(\mathbf X)\mathbf u}{1+c_e},\qquad
\mathbf r = \mathbf G^T(\mathbf S\,\mathbf v'),\qquad \mathbf u' = \text{Cap}(\mathbf u, \mathbf r),
```

where $`\mathbf S`$ is the diagonal stiffness $`\sum_T h_{e,T}/(\mu_0\mu_T L_e)`$ and $`c_e = \tfrac12\Delta t\,s(r_e)`$ is the edge loss of the matched absorber ($`c_e=0`$ without one). $`r(\mathbf X)`$ is the LTS polynomial (identity without LTS). The transpose is

```math
\bar{\mathbf v}'_{\text{tot}} = \bar{\mathbf v}' + \mathbf S\,\mathbf G\,\bar{\mathbf r},\qquad
\bar{\mathbf v} = \frac{1-c_e}{1+c_e}\bar{\mathbf v}'_{\text{tot}},\qquad
\bar{\mathbf u} \mathrel{+}= -\,r(\mathbf X)^T\mathbf G^T\frac{\Delta t\,\bar{\mathbf v}'_{\text{tot}}}{1+c_e} .
```

`lts.linear_step_T` already has this form for the source-free, non-dispersive step; here it is extended to the ADE groups and the PML.

### 2.3 2D TE step and the series ADE

The dual-edge voltage of edge $`e`$ is $`V_e = S_e d_e + \sum_{T\,\text{disp}} h_{e,T}E_{e,T}`$, where $`S_e = \sum_{T\,\text{non-disp}} h_{e,T}/(\varepsilon_0\varepsilon_T L_e)`$ and $`E_{e,T}`$ is the segment field of a dispersive triangle (normal-D continuity across the edge). The step is

```math
\mathbf u' = \frac{\mathbf a\,\mathbf u - \Delta t(\mathbf G^T\mathbf V + \mathbf k)}{\mathbf b},\quad
\mathbf v' = \frac{(1-c_e)\mathbf v + \Delta t\,\mathbf G\,r(\mathbf X)\mathbf u'}{1+c_e},\quad
E'_{e,T} = \frac{v'_e/(L_e\varepsilon_0) - \Psi}{\varepsilon_\infty + \sum\beta_D} ,
```

with $`\mathbf a,\mathbf b = M_\mu \mp \tfrac12\Delta t\,G`$, and $`\Psi = \sum_L p'_L + \sum_D(\kappa_D p_D + \beta_D E_{e,T})`$ the ADE polarization of the segment. The segment field $`E^n_{e,T}`$ enters twice, through $`\mathbf V(E^n)`$ and through the ADE, so its adjoint is

```math
\bar E_{e,T} = \underbrace{\sum_L\zeta\bar p'_{L,\text{tot}} + \sum_D\beta_D(\bar p'_D + \bar\Psi)}_{\text{ADE}} + h_{e,T}\bar V_e ,\qquad
\bar V = -\Delta t\,\mathbf G(\bar{\mathbf u}'_{\text{tot}}/\mathbf b) .
```

The segment-to-edge map is a scatter-add (`bincount`). Its transpose is the gather $`\bar v'_e \mathrel{+}= \sum_T \bar D_{e,T}/(L_e\varepsilon_0)`$. Dropping it is a mutation of the tests (dot error 0.05).

### 2.4 Boundaries

* **PEC (TM nodes, 3D edges)**: fixed dofs are excluded from the free set. The step is defined on the free dofs, and the restriction is its own transpose: nothing to do.
* **PMC**: natural, no term.
* **Silver–Müller** first order: a diagonal conductance in $`G`$ (TM, 3D) or in the $`H_z`$ loss (TE). Diagonal, hence symmetric.
* **Matched absorber**: diagonal node loss plus the edge factors $`(1-c_e)/(1+c_e)`$ and $`\Delta t/(1+c_e)`$.

### 2.5 2D radial CFS-PML

Every CFS ratio filter $`y = (s_n/s_d)\,x`$ (`solver2d._CFSRatio`, bilinear transform) is a linear recursion on its memory $`(P_x, P_y, x_0, y_0)`$:

```math
P_x' = c_1P_x + c_2(x+x_0),\quad
y = \frac{\kappa_n x + \sigma_n P_x' - \sigma_d(c_1P_y + c_2y_0)}{\kappa_d+\sigma_d c_2},\quad
P_y' = c_1P_y + c_2(y+y_0),\quad x_0'=x,\ y_0'=y .
```

Its transpose, with $`D = \kappa_d+\sigma_d c_2`$ and an external output adjoint $`\bar y_\text{ext}`$, is

```math
\begin{aligned}
\bar y &= \bar y_\text{ext} + \bar y_0' + c_2\bar P_y',\quad t = \bar y/D,\\
\bar P_y &= c_1\bar P_y' - \sigma_d c_1 t,\quad \bar y_0 = c_2\bar P_y' - \sigma_d c_2 t,\\
\bar P_x &= c_1(\bar P_x' + \sigma_n t),\quad \bar x_0 = c_2(\bar P_x' + \sigma_n t),\quad
\bar x = \kappa_n t + c_2(\bar P_x' + \sigma_n t) + \bar x_0' .
\end{aligned}
```

The PML step overwrites three things:

* the curl of the spokes and chords by $`S\,f_s(\cdot)`$ and $`S\,f_c(\cdot)`$;
* the update of the layer nodes by $`u = f_2(f_1(q/M))`$, where $`q`$ accumulates $`\Delta t\,r`$.

In the transpose, the overwritten entries take no adjoint from the plain update. They are masked, and their adjoint goes through the filter transposes in reverse order ($`f_2^T`$, then $`f_1^T`$, then $`1/M`$, then $`q`$). The layer is a fixed linear map (its coefficients depend on the mesh and $`\Delta t`$, not on $`\varepsilon`$), so it is exactly transposable.

### 2.6 Two-level LTS (2D)

The LTS feeds the flux update with $`r(\mathbf X)\mathbf u`$, where $`\mathbf X = \Delta t^2\mathbf P\mathbf M^{-1}\mathbf K`$ (physics guide, "Local time stepping"). Its transpose is $`r(\mathbf X)^T = r(\mathbf X^T)`$, `LocalTimeStepping.apply_T`. Using `apply` (the untransposed polynomial) in the adjoint fails the dot test (6e-4 TE, 3e-3 TM).

### 2.7 3D: massless edges and Whitney Hodge

**Massless edges** (nearly degenerate Voronoi faces, `massless_tol`) are eliminated from the quasi-static constraint $`(\mathbf C^T\mathbf M_\nu\mathbf b)_s = 0`$ each step. This makes their voltages a fixed linear function of the free ones:

```math
\mathbf e_s = -\mathbf K_{ss}^{-1}\mathbf K_{sf}\mathbf u,\qquad
\mathbf K_{ss} = \mathbf C_s^T\mathbf M_\nu\mathbf C_s,\quad \mathbf K_{sf} = \mathbf C_s^T\mathbf M_\nu\mathbf C_f .
```

The flux update $`\mathbf b' = \mathbf b - \Delta t(\mathbf C_f\mathbf u + \mathbf C_s\mathbf e_s)`$ then transposes to

```math
\bar{\mathbf u} \mathrel{+}= -\Delta t\,\mathbf C_f^T\bar{\mathbf b}' \;+\; \mathbf K_{sf}^T\mathbf K_{ss}^{-T}\big(\Delta t\,\mathbf C_s^T\bar{\mathbf b}'\big) .
```

This is the transposed Schur complement: one extra $`\mathbf K_{ss}`$ solve per adjoint step, with the forward factorization (`splu(...).solve(..., trans="T")`). $`\mathbf K_{ss}`$ is symmetric, so `trans="T"` changes nothing numerically, but it keeps the code correct for a non-symmetric $`\mathbf M_\nu`$. Dropping the term, flipping its sign or replacing $`\mathbf K_{ss}^{-1}`$ by its diagonal fails the dot test (errors 2e-3, 3e-3 and 7e-6).

**Whitney Hodge**: $`\mathbf u' = \mathbf u + \Delta t\,\mathbf M_1^{-1}\mathbf r`$ transposes to $`\mathbf w = \mathbf M_1^{-T}\bar{\mathbf u}'`$, $`\bar{\mathbf r} = \Delta t\,\mathbf w`$: one extra $`\mathbf M_1`$ solve per adjoint step, with the same LU factors.

## 3. Design sensitivities

The design variables are the relative permittivities $`\varepsilon_T`$ of the design cells. Each enters exactly one fixed sparse linear (or inverse-linear) map.

**2D TM and 3D Voronoi: capacitances.** $`C_{0,k} = \varepsilon_0\sum_T\varepsilon_T P_{kT}`$, with the partial dual areas $`P_{kT} = A_{n,T}`$ (2D node masses, `mesh.dual_area_partial`) or $`A_{e,T}/L_e`$ (3D). From $`u' = \text{acc}/d`$ with $`\partial a_1/\partial C_0 = \partial d/\partial C_0 = 1`$, $`\partial u'/\partial C_0 = (u - u')/d`$. Hence

```math
\frac{dF}{d\varepsilon_T} = \varepsilon_0\sum_k P_{kT}\sum_n \bar u'^{\,n}_{k,\text{tot}}\,\frac{u^n_k - u^{n+1}_k}{d_k} .
```

The row of step $`n`$ is $`(u^n-u^{n+1})/d`$ on the design dofs, and the sens is the total $`\bar u'`$ of Section 2.1. This is the roadmap formula $`\sum_e (dF/dM_\varepsilon[e])\,\varepsilon_0A_{e,T}/L_e`$.

**2D TE: series stiffness.** $`S_e = \sum_T h_{e,T}/(\varepsilon_0\varepsilon_T L_e)`$ enters $`V = S\,d`$, so

```math
\frac{dF}{d\varepsilon_T} = -\sum_e \frac{h_{e,T}}{\varepsilon_0\varepsilon_T^2 L_e}\sum_n \bar V^n_e\, d^n_e ,
```

with row $`d^n`$ (the flux *entering* the step) and sens $`\bar V`$.

**3D Whitney.** $`\mathbf M_1 = \sum_T\varepsilon_T\mathbf M_T`$ (element matrices, `whitney_local_mass`), and $`\partial\mathbf u'/\partial\mathbf M_1 = -\mathbf M_1^{-1}(\partial\mathbf M_1)\mathbf M_1^{-1}\mathbf r\,\Delta t`$. Hence

```math
\frac{dF}{d\varepsilon_T} = -\sum_n \mathbf w_n^T\mathbf M_T(\mathbf u^{n+1}-\mathbf u^n),
```

a $`6\times6`$ bilinear form per tetrahedron and step.

**What is frozen.** The time step, the LTS partition and polynomial, the PML coefficients and the massless-edge set are those of the solver that `make_solver` builds. They are not differentiated, because they are not functions of $`\varepsilon`$ in the discrete model actually run. Build that solver with the design at $`\varepsilon_\min`$. Raising $`\varepsilon`$ raises $`\mathbf M`$ (TM, 3D) or lowers $`\mathbf S`$ (TE). By the Rayleigh quotient, either can only lower $`\lambda_\max(\mathbf M^{-1}\mathbf K)`$, so the frozen $`\Delta t`$ stays stable for every $`\rho\in[0,1]`$.

## 4. Parametrization: `CellMaterialGrid`

```math
\varepsilon_T = \varepsilon_\min + (\varepsilon_\max-\varepsilon_\min)\,P_{\beta,\eta}\big((\mathcal F\rho)_T\big) .
```

* **Cone filter** [Bourdin, 2001] on the design cells: $`(\mathcal F\rho)_i = \sum_j w_{ij}A_j\rho_j/\sum_j w_{ij}A_j`$, with $`w_{ij} = \max(0, 1-|\mathbf c_i-\mathbf c_j|/R)`$, centroids $`\mathbf c`$ and areas (volumes) $`A`$. Rows sum to one, so a constant density is preserved.
* **Helmholtz filter** [Lazarov & Sigmund, 2011]: $`-r^2\nabla^2\tilde\rho + \tilde\rho = \rho`$ with $`r = R/(2\sqrt3)`$, discretised by two-point fluxes on the cell adjacency graph. The face weight is $`|f|/|\mathbf c_i-\mathbf c_j|`$, and there is zero flux out of the design region: $`(\mathbf A + r^2\mathbf L)\tilde\rho = \mathbf A\rho`$. Its transpose is $`\mathbf A(\mathbf A+r^2\mathbf L)^{-T}`$, one solve with the same LU.
* **tanh projection** [Wang et al., 2011], pointwise.

Every stage is linear or pointwise, so `vjp` is the exact chain $`\mathcal F^T\,\mathrm{diag}(P')\,(\varepsilon_\max-\varepsilon_\min)`$.

## 5. JAX

`wakis.inverse.jax_simulation(UnstructuredAdjointSimulation(...))` returns a function $`\rho\mapsto\{\text{monitor}:\{\text{"dft"}: X\}\}`$ whose reverse-mode derivative runs `vjp` (`jax.custom_vjp` around `jax.pure_callback`). The test checks that `jax.value_and_grad` equals the NumPy value and gradient to 1e-12.

## 6. Memory

`memory="store"` keeps one row per step: $`N_t\times N_\text{dof,design}`$ float64. `memory="checkpoint"` reuses `wakis.inverse.memory.CheckpointRung`: binomial (revolve) checkpointing over blocks [Griewank & Walther, 2000], with snapshots of the full state dictionary. Replaying step $`n`$ from a restored $`\mathbf x^n`$ performs the same floating-point operations, so the gradient is bitwise equal to the stored-row run. The tests check this with `array_equal` and zero replayed-row mismatches (`verify` mode).

## 7. Verification

`tests/test_100_unstructured_adjoint.py` covers 22 configurations:

* 2D, TM and TE: PEC, PMC, Silver–Müller, Lorentz+Debye, absorber, PML, PML+dispersive, LTS, scattered-field plane wave;
* 3D: PEC and Silver–Müller with 57–66 massless edges, Lorentz+Debye, Whitney.

Each configuration is checked three ways:

* `step` equals `one_step` bitwise over 40 steps with sources. The DFT outputs equal the solver's own `Recorder` to 1e-12.
* The dot test is run **block by block**. For every (input block $`i`$, output block $`j`$), it compares $`\langle\mathbf A_{ji}\mathbf x_i,\mathbf y_j\rangle`$ with $`\langle\mathbf x_i,\mathbf A_{ji}^T\mathbf y_j\rangle`$, relative to $`|\mathbf A_{ji}\mathbf x_i||\mathbf y_j| + |\mathbf x_i||\mathbf A_{ji}^T\mathbf y_j|`$. A single global inner product is dominated by the largest coefficients (here $`\sim10^{21}`$) and can hide an $`O(1)`$ error in a small block. The measured maximum is $`2.8\times10^{-17}`$. The threshold, $`10^{-14}`$, is the round-off bound $`\sqrt n\,\epsilon_\text{mach}`$ for $`n\lesssim 8000`$. Eleven mutations give $`7\times10^{-6}`$ to $`0.2`$.
* Gradients are compared with central finite differences, $`h=10^{-3}`$, on 3 random design cells for each of 9 configurations: TM/TE Silver–Müller, TM/TE dispersive, TM PML, TE LTS, 3D massless, 3D dispersive and 3D Whitney. The relative difference is $`7\times10^{-10}`$ to $`8\times10^{-7}`$, and it equals the truncation estimate $`|FD(2h)-FD(h)|/3`$ to two digits in every case. The assertion is $`|g-FD| \le 1.5\times`$ that estimate, plus a round-off term. Two mutations of the design map fail this test:
  * the barycentric $`A_T/3`$ instead of the Voronoi partial areas (TM);
  * $`-1/\varepsilon`$ instead of $`-1/\varepsilon^2`$ (TE).

  The TM test uses a **jittered** mesh. On the hexagonal lattice of `disk_mesh`, almost every triangle is equilateral, the Voronoi and barycentric partial areas coincide, and the barycentric mutation passed (error $`6.5\times10^{-7}`$). That degenerate test was replaced.

## 8. Limits & Failure Modes

* **Design cells must not:**
  * be PEC or dispersive. The design changes $`\varepsilon_\infty`$ only through $`C_0`$ or $`S`$; a dispersive design material would need the group capacitances $`c_g`$ to depend on $`\rho`$.
  * touch the outer boundary. The Silver–Müller impedance uses the boundary cell's $`\varepsilon`$.
  * lie in the absorber (TM: its node loss is $`M\,s(r)`$).
  * lie in the PML.
  * touch the LTS fine set: TM nodes in $`F`$, or TE edges incident to $`F`$, which enter $`\mathbf X`$.

  The constructor raises in each case.
* **Plane-wave sources** (`add_plane_wave`): their equivalent sources depend on $`\varepsilon`$ (TM node weights $`M_\varepsilon - \varepsilon_0\varepsilon_bA`$, TE segment coefficients $`1-\varepsilon_b/\varepsilon_T`$). The forward and adjoint steps support the non-dispersive plane wave (bitwise and dot-tested in both polarizations). A design with a plane wave raises `NotImplementedError`, and so does a dispersive plane wave at all (`dispersive_planewave` patches the solver's ADE methods). Supporting it needs the derivative of the TF/SF weights, a sparse map like the others.
* **Frozen discrete choices.** $`\Delta t`$, the LTS partition, the massless set and the 3D interface clipping of the dual areas (`_clip_interface_dual_areas`, which attributes a tag's clipped area to one tetrahedron of that tag) are those of the template solver. The gradient is exact for that discrete model. A fresh solver built at another $`\varepsilon`$ could choose differently, for example by flipping an edge across `massless_tol`.
* **Non-deterministic $`\Delta t`$ in 3D.** With massless edges, `UnstructuredFIT3D` computes $`\lambda_\max`$ by an unseeded `eigsh`. Two builds of the same mesh then give $`\Delta t`$ values that differ at the Lanczos tolerance (observed: gradients differing by 1.3e-11 relative between two builds). Build the solver once and pass `dt` explicitly, as the tests do. A seeded start vector, as in `_core.max_eigenvalue`, would fix it.
* **Whitney Hodge**: there is no dispersive or lossy support (the solver ignores `cond` with Whitney). Each adjoint step costs one extra $`\mathbf M_1`$ solve, and each `set_eps` refactorizes $`\mathbf M_1`$.
* **Not covered:**
  * the 3D PML (other branch);
  * the GPU path;
  * `PolygonFIT2D`, `SubTriFIT2D` and `PrismFIT3D`;
  * shape derivatives (node motion);
  * permeability design.
* **Memory**: `store` needs $`8N_tN_\text{dof}`$ bytes; `checkpoint` trades it for at most two recomputations per step (default slots). Time reversal ("reverse" rung) is not implemented for unstructured meshes.
* **Truncated DFT windows.** The objective is the DFT over the simulated window. A high-Q design can exploit ringing that the window cuts off. The demo checks that its objective changes by less than the stated fraction when $`N_t`$ is doubled.

## 9. Worked example

`studies/unstructured-adjoint/shadow_focus.py` (Cloud Batch) focuses a line source into the shadow of a PEC cylinder, using a dielectric annulus meshed with conforming triangles. See the study README and `docs/figures.md`.

## References

- M. B. Giles and N. A. Pierce, "An introduction to the adjoint approach to design", *Flow, Turbulence and Combustion* **65**, 393–415 (2000).
- M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique", *Progress In Electromagnetics Research* **32**, 65–87 (2001).
- A. Bossavit and L. Kettunen, "Yee-like schemes on a tetrahedral mesh, with diagonal lumping", *Int. J. Numer. Model.* **12**(1/2), 129–142 (1999).
- J. Diaz and M. J. Grote, "Energy conserving explicit local time stepping for second-order wave equations", *SIAM J. Sci. Comput.* **31**(3), 1985–2014 (2009), doi:10.1137/070709414.
- F. L. Teixeira and W. C. Chew, "Systematic derivation of anisotropic PML absorbing media in cylindrical and spherical coordinates", *IEEE Microw. Guided Wave Lett.* **7**(11), 371–373 (1997).
- A. Griewank and A. Walther, "Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation", *ACM Trans. Math. Softw.* **26**, 19–45 (2000), doi:10.1145/347837.347846.
- B. Bourdin, "Filters in topology optimization", *Int. J. Numer. Methods Eng.* **50**, 2143–2158 (2001), doi:10.1002/nme.116.
- B. S. Lazarov and O. Sigmund, "Filters in topology optimization based on Helmholtz-type differential equations", *Int. J. Numer. Methods Eng.* **86**(6), 765–781 (2011), doi:10.1002/nme.3072.
- F. Wang, B. S. Lazarov and O. Sigmund, "On projection methods, convergence and robust formulations in topology optimization", *Struct. Multidiscip. Optim.* **43**, 767–784 (2011), doi:10.1007/s00158-010-0602-y.
