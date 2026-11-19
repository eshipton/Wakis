# Local time stepping on 3D tetrahedral meshes (`wakis.unstructured.lts3d`)

Code: `wakis/unstructured/lts3d.py` (`TetLTS`, `partition_levels`, `nest_levels`, `setup_lts`), hooks in
`wakis/unstructured/solver3d.py`, `wakis/unstructured/pml3d.py`, `wakis/unstructured/adjoint.py`,
`wakis/unstructured/gpu.py`, `wakis/unstructured/mpi.py`, `wakis/unstructured/meshers.py::box_mesh`.
Tests: `tests/test_116_unstructured_lts3d.py`.
Study: `studies/unstructured-lts3d/` (`run_study.py`, `plot_figures.py`, `verify_numbers.py`).

## 1. Motivation and scope

In `UnstructuredFIT3D` (`wakis/unstructured/solver3d.py`), the global leapfrog stability limit is determined by the top eigenvalue of the generalized curl-curl pencil $(K, M)$. As documented in `docs/physicsguide_sections/unstructured_cadmesh.md` §3–4 (`studies/unstructured-cadmesh/results/bounds.json`, `improve.json`), for the diagonal Voronoi Hodge the stability limit is governed by single edges whose dual Voronoi faces lie just above the `massless_tol` elimination threshold, concentrating $98\text{--}100\%$ of the top mode's energy on the $1\%$ of tetrahedra with the smallest element stability bound $\Delta t_T$ (where $\sqrt{\max_T \lambda_T / \lambda_{\max}} = 1.025\text{--}1.090$). For the consistent Whitney Hodge, the stability limit is a collective property across coupled edges and the element bounds are $1.19\text{--}1.55\times$ pessimistic, though on meshes with isolated slivers the top mode still localizes on the worst elements. Node relocation (`improve_dt` in `wakis/unstructured/improve.py`) raises the global stable step by $1.47\text{--}1.73\times$ on the pillbox benchmark (`studies/unstructured-cadmesh/results/improve.json`) at zero per-step overhead, and raising `massless_tol` eliminates additional degenerate Voronoi dual faces via the static Schur complement. Local time stepping (LTS) addresses the complementary regime where a mesh has deliberate spatial grading ($h_{\mathrm{coarse}} / h_{\mathrm{fine}} \gg 1$) or irreducible small/sliver elements locked to CAD curves, corners, or material interfaces.

**Milestone 1 scope:**
* **Supported:** CPU (NumPy/SciPy); `UnstructuredFIT3D` with `hodge="voronoi"` (including static massless-edge Schur elimination) and `hodge="whitney"` (consistent sparse $M_1, M_2$); closed PEC and PMC boundaries; the 3D spherical CFS-PML of `wakis.unstructured.pml3d.PML3D` (`hodge="voronoi"`, when all fine levels and their one-ring buffers lie strictly inside the PML inner radius $r_0$); multi-level integer step ratios $p_l \in \{2, 3\}$; automatic level assignment from element bounds (`partition_levels`) paired with a Lanczos stability certificate (`is_stable`, `lts_dt_max`); and the exact discrete transpose in `UnstructuredAdjointStepper` (state transpose always; exact material gradient when no design tetrahedron touches a fine edge or face).
* **Unsupported in milestone 1 (raise `NotImplementedError`):** GPU (`DeviceStepper` with a 3D LTS solver), MPI (`LocalCluster`, `DistributedFIT`), lumped and waveguide ports (`ports3d.attach_ports`), dispersive materials, nonzero conductivity $\sigma > 0$, `bc="abc"`, the hybrid Cartesian/tetrahedral solver, and the material gradient when a design cell touches a fine edge (`allow_inexact_lts_gradient=False`).

---

## 2. Derivation

All quantities are in SI units: coarse time step $\Delta t$ and level substeps $\tau_l$ in $\mathrm{s}$, eigenvalues $\lambda, \mu_l$ in $\mathrm{s}^{-2}$, electric edge mass $M$ in $\mathrm{F}$, and magnetic face Hodge $W$ in $\mathrm{H}^{-1}$. Time-domain fields are real; resonance frequencies are extracted with the Wakis phasor convention $\exp(+\mathrm{j}\omega t)$ (`wakis/resonances.py`). The multi-level recursion, positivity margins, nested-damping law, and global-limit relation follow `docs/physicsguide_sections/subgrid3d.md` §9c and are generalized here to unstructured tetrahedral meshes with massless-edge Schur elimination and non-diagonal consistent Whitney mass matrices.

### 2.1 Semi-discrete system

Let $\mathbf u \in \mathbb R^n$ denote the electric voltages on the $n$ free primal edges (`solver.free`) and $\mathbf b \in \mathbb R^{n_f}$ the magnetic fluxes on the primal faces.

* **Voronoi Hodge (`hodge="voronoi"`):** $M = \mathrm{diag}(\texttt{cap}[f])$ and $W = \mathrm{diag}(\texttt{Mnu\_diag})$. Massless interior edges $s$ (`solver.massless`, with normalized dual area below `massless_tol`) are eliminated by the static Ampère constraint $C_s^T W \mathbf b = \mathbf 0$, yielding $\mathbf e_s = -K_{ss}^{-1} K_{sf} \mathbf u = -S \mathbf u$ with $S = K_{ss}^{-1} K_{sf}$, where $K_{ab} = C_a^T W C_b$. Faraday's law on the eliminated subspace is
  $$\mathbf b' = -C_{\mathrm{eff}}\mathbf u, \qquad C_{\mathrm{eff}} = C_f - C_s S,$$
  and Ampère's law on the free edges is $M \mathbf u' = C_f^T W \mathbf b - \mathbf j$. The static constraint is invariant under the semi-discrete flow:
  $$\frac{\mathrm d}{\mathrm dt}(C_s^T W \mathbf b) = -C_s^T W (C_f - C_s K_{ss}^{-1} K_{sf})\mathbf u = -(K_{sf} - K_{ss} K_{ss}^{-1} K_{sf})\mathbf u = \mathbf 0.$$
* **Whitney Hodge (`hodge="whitney"`):** $M = M_{1,ff}$ (symmetric positive-definite, sparse, non-diagonal) and $W = M_2$ (symmetric positive-definite, sparse); there are no massless edges ($s = \emptyset$) and $C_{\mathrm{eff}} = C_f$.

Eliminating $\mathbf b$ gives the second-order semi-discrete wave equation on the free edges:
$$M \mathbf u'' = -K \mathbf u - \mathbf j', \qquad K = C_f^T W C_{\mathrm{eff}} = C_{\mathrm{eff}}^T W C_{\mathrm{eff}},$$
where for Voronoi $K = K_{ff} - K_{sf}^T K_{ss}^{-1} K_{sf}$ is the symmetric Schur complement and $K \succeq 0$ whenever $W \succeq 0$ (enforced by the negative dual-length guard in `solver3d.py`).

### 2.2 Global leapfrog

With a uniform global step $\Delta t$, the source-free staggered leapfrog
$$\mathbf b^{n+1/2} = \mathbf b^{n-1/2} - \Delta t\, C_{\mathrm{eff}}\mathbf u^n, \qquad M(\mathbf u^{n+1} - \mathbf u^n) = \Delta t\, C_f^T W \mathbf b^{n+1/2}$$
is equivalent to the second-order update
$$M(\mathbf u^{n+1} - 2\mathbf u^n + \mathbf u^{n-1}) = -\Delta t^2 K \mathbf u^n,$$
which is stable if and only if $\Delta t < \Delta t_{\max}^{\mathrm{global}} = 2 / \sqrt{\lambda_{\max}(K, M)}$ (`solver.dt_max`).

### 2.3 Multi-level LTS recursion

Partition the free edges into nested edge sets
$$F_L \subset F_{L-1} \subset \dots \subset F_1 \subset F_0 = \{0, \dots, n-1\},$$
with integer refinement ratios $p_l \in \{2, 3\}$ ($l = 1, \dots, L$) and level step sizes $\tau_0 = \Delta t$, $\tau_l = \tau_{l-1} / p_l$. Let $E_l \in \mathbb R^{n \times |F_l|}$ be the $0/1$ column injection matrix for $F_l$ ($E_l^T E_l = I_{|F_l|}$, $P_l = E_l E_l^T$), and let $M_{ll} = E_l^T M E_l$ be the principal submatrix of $M$ on $F_l$. Since $M \succ 0$, $M_{ll} \succ 0$, and we define the symmetric positive semi-definite local inverse-mass operator
$$Q_l = E_l M_{ll}^{-1} E_l^T \in \mathbb R^{n \times n}.$$
When $M$ is diagonal (Voronoi), $Q_l = P_l M^{-1} = M^{-1} P_l$.

Let $r_l(x) = \sum_{j=0}^{p_l-1} c_{l,j} x^j$ be the damped leapfrog-Chebyshev polynomial of degree $p_l - 1$ (`wakis.unstructured.lts.chebyshev_coefficients`), defined by
$$x\, r_l(x) = 2\left(1 - \frac{T_{p_l}(\delta_l - \beta_l x)}{T_{p_l}(\delta_l)}\right), \qquad \delta_l = 1 + \frac{\nu_l}{p_l^2}, \qquad \beta_l = \frac{T_{p_l}(\delta_l)}{2 T'_{p_l}(\delta_l)}, \qquad r_l(0) = 1.$$
Define the multi-level effective stiffness matrices from the finest level $L$ down to $0$:
$$S_L = K, \qquad S_{l-1} = S_l\, r_l(\tau_{l-1}^2 Q_l S_l) \quad (l = L, \dots, 1),$$
so that
$$S_0 = K R_0, \qquad R_0 = r_L(X_L) \cdots r_1(X_1), \qquad X_l = \tau_{l-1}^2 Q_l K R_l, \qquad R_l = r_L(X_L)\cdots r_{l+1}(X_{l+1}), \quad R_L = I.$$
The source-free second-order LTS update is
$$M(\mathbf u^{n+1} - 2\mathbf u^n + \mathbf u^{n-1}) = -\Delta t^2 S_0 \mathbf u^n = -\Delta t^2 K R_0 \mathbf u^n.$$
For $L = 1$, diagonal $M$, and $\nu = 0$, this is the two-level energy-conserving LTS of Diaz & Grote (2009) ($p_1$ leapfrog substeps of size $\tau_1$ on $F_1$ with the coarse force frozen). For $\nu > 0$, $r_l$ is the stabilized Chebyshev polynomial of Carle, Hochbruck & Sturm (2020) and Grote, Michel & Sauter (2021). Here $\nu_l$ shifts the Chebyshev argument ($\delta_l = 1 + \nu_l / p_l^2$) to give a positivity margin and adds no dissipation — the scheme is $M(\mathbf u^{n+1} - 2\mathbf u^n + \mathbf u^{n-1}) = -\Delta t^2 S_0 \mathbf u^n$ with $S_0$ symmetric, and energy conservation is tested at $\nu = 0.05$ (`test_modified_energy_conserved`). For $L > 1$, the recursion follows the multi-level structure of Diaz & Grote (2015), with the coarse force at level $l$ also evaluated through $S_l$ (`docs/physicsguide_sections/subgrid3d.md` §9c) so that the symmetry induction closes at every level. As derived in `docs/physicsguide_sections/subgrid3d.md` §9c ("Nested levels need increasing damping"), equal damping across nested $p_l = p_{l+1} = 2$ levels gives a level margin identically equal to $1$ at the peak of $x\,r_{l+1}(x)$, so the damping parameter must increase strictly with depth: $\nu_l = l\,\nu$ (default $\nu = 0.05$).

### 2.4 Symmetry for any SPD mass matrix

Expanding the polynomial at level $l$,
$$S_{l-1} = \sum_{j=0}^{p_l - 1} c_{l,j} \tau_{l-1}^{2j}\, S_l (Q_l S_l)^j.$$
If $S_l^T = S_l$ and $Q_l^T = Q_l$, each term $S_l (Q_l S_l)^j = (S_l Q_l)^j S_l$ is a palindrome of symmetric matrices and is therefore symmetric. By downward induction from $S_L = K = K^T$, every $S_l$ ($l = L, \dots, 0$) is symmetric for *any* symmetric positive semi-definite $Q_l$.

For the consistent Whitney edge mass $M = M_1$, the naive operator $P_l M_1^{-1}$ is **not** symmetric because $P_l$ and $M_1$ do not commute ($P_l M_1 \neq M_1 P_l$ whenever an edge in $F_l$ shares a tetrahedron with an edge outside $F_l$). Consequently, replacing $Q_l$ by $P_l M_1^{-1}$ breaks the symmetry of $S_0$ and destroys energy conservation. By contrast, the principal-submatrix inverse $Q_l = E_l M_{ll}^{-1} E_l^T$ is symmetric positive semi-definite on any mesh, requires only a sparse Cholesky factorization of the $|F_l| \times |F_l|$ block $M_{ll}$ once at setup, and restricts all inner substep updates strictly to $F_l$. In the staggered first-order form, $\mathbf z = R_0 \mathbf u^n$ replaces $\mathbf u^n$ in Faraday's law,
$$\mathbf b^{n+1/2} = \mathbf b^{n-1/2} - \Delta t\, C_{\mathrm{eff}} \mathbf z, \qquad M(\mathbf u^{n+1} - \mathbf u^n) = \Delta t\,(C_f^T W \mathbf b^{n+1/2} - \mathbf j^{n+1/2}),$$
while Ampère's law still solves with the full consistent mass $M$ once per coarse step. This principal-submatrix choice $Q_l = E_l M_{ll}^{-1} E_l^T$ for a globally coupled consistent Whitney mass matrix is derived here for Wakis and is not taken from the cited LTS references, which assume a diagonal (lumped) or element-block-diagonal (DG) mass matrix.

### 2.5 Conserved modified energy

Multiplying $M(\mathbf u^{n+1} - 2\mathbf u^n + \mathbf u^{n-1}) = -\Delta t^2 S_0 \mathbf u^n$ on the left by $(\mathbf u^{n+1} - \mathbf u^{n-1})^T$ and using only $M^T = M$ and $S_0^T = S_0$ telescopes identically to the conserved quadratic form (`TetLTS.modified_energy`):
$$E^{n+1/2} = \frac{1}{2}\mathbf w^T\left(M - \frac{\Delta t^2}{4}S_0\right)\mathbf w + \frac{1}{2}\bar{\mathbf y}^T S_0 \bar{\mathbf y}, \qquad \mathbf w = \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}, \qquad \bar{\mathbf y} = \frac{\mathbf u^{n+1} + \mathbf u^n}{2}.$$
Because the telescoping identity uses only the symmetry of $M$ and $S_0$, $E^{n+1/2}$ is algebraically conserved for *any* $\Delta t$ in the absence of sources, losses, and PML. As established in `docs/physicsguide_sections/subgrid3d.md` §9c, $E^{n+1/2}$ is only a **semi-norm** on the first-order state $(\mathbf u, \mathbf b)$: the potential term vanishes on $\ker S_0$ (which contains all discrete gradients $\ker C_{\mathrm{eff}}$), and $\mathbf b^{n+1/2}$ enters only through $\mathbf w = M^{-1} C_f^T W \mathbf b^{n+1/2}$, so $E^{n+1/2}$ also vanishes on $\ker(C_f^T W)$. When the scheme is unstable ($S_0 \not\succeq 0$ or $\frac{\Delta t^2}{4}\lambda_{\max}(S_0, M) > 1$), $E^{n+1/2}$ becomes *indefinite* while remaining conserved to floating-point round-off until overflow (`test_energy_conserved_while_unstable`). Furthermore, there is no conserved first-order energy of the form $\frac{1}{2}\mathbf b^T W \mathbf b + \frac{1}{2}(\mathbf u^n)^T G \mathbf u^{n+1}$ because $M R_0$ is not symmetric ($Q_l K \neq K Q_l$).

### 2.6 Positivity and stability, level by level

Writing $Q_l = L_l L_l^T$ with $L_l = E_l M_{ll}^{-1/2}$, the nonzero eigenvalues of $\tau_{l-1}^2 Q_l S_l$ coincide with the eigenvalues of the symmetric matrix $\tau_{l-1}^2 M_{ll}^{-1/2} (E_l^T S_l E_l) M_{ll}^{-1/2}$, i.e., the generalized pencil $(E_l^T S_l E_l,\, M_{ll})$ on $\mathbb R^{|F_l|}$. If $S_l \succeq 0$ and the spectrum of $\tau_{l-1}^2 (E_l^T S_l E_l,\, M_{ll})$ lies inside $[0, x_+(p_l, \nu_l)]$, where $x_+(p_l, \nu_l) = 2\delta_l/\beta_l$ is the first root of $r_l$ for even $p_l$ ($+\infty$ for odd $p_l$), then $r_l(\tau_{l-1}^2 Q_l S_l)$ is positive semi-definite on the range of $S_l$ and $S_{l-1} \succeq 0$. For odd $p_l$, $x = 2\delta_l / \beta_l$ is the point where $x\, r_l(x)$ reaches $4$.

Stability of the coarse step therefore requires two conditions simultaneously (`TetLTS.is_stable`):
1. **Level margins:** for every level $l = 1, \dots, L$,
   $$m_l := \frac{\tau_{l-1}^2 \mu_l}{x_{\mathrm{adm},l}} < 1 - g, \qquad \mu_l = \lambda_{\max}(E_l^T S_l E_l,\, M_{ll}),$$
   where $x_{\mathrm{adm},l} = 2\delta_l / \beta_l$, further restricted for odd $p_l$ nested under $l - 1 \ge 1$ to the root of $p_{l-1}^2 x\, r_l(x) = x_{\mathrm{adm},l-1}$ (`docs/physicsguide_sections/subgrid3d.md` §9c);
2. **Coarse leapfrog limit:**
   $$\rho := \frac{\Delta t}{2}\sqrt{\lambda_{\max}(S_0, M)} < 1 - g.$$

Because Lanczos Ritz values approximate $\lambda_{\max}$ from below, both checks subtract the safety margin $g = 10\,\mathrm{tol}$ (`is_stable` uses `tol = 1e-5` by default, so $g = 10\,\mathrm{tol} = 10^{-4}$ and $1 - g = 0.9999$, and `setup_lts` re-certifies the final operator at `tol = 1e-8`, where $g = 10^{-7}$). Each check takes the extreme of a block of three Lanczos Ritz values (`K_TOP = 3`, `ncv = 60`, at most 3000 restarts; non-convergence raises an error). With a single Ritz value, a (near-)degenerate top pair on a rotationally symmetric pillbox mesh kept the earlier `tol = 1e-10` certificate running for more than 20 h; the final step sits at 0.995 of the bisected limit, so the margin $1-\rho \approx 5\times10^{-3}$ is far larger than the Ritz error at `tol = 1e-8`. When the global top mode $\lambda_{\max}(K, M) = 4 / (\Delta t_{\max}^{\mathrm{global}})^2$ is supported in the finest level $F_L$ and (for Whitney) off-diagonal mass entries across $\partial F_L$ do not shift it, its level-$L$ argument is $x = \tau_{L-1}^2 \lambda_{\max} = 4\big((\Delta t / \Delta t_{\max}^{\mathrm{global}}) / \prod_{m<L} p_m\big)^2$, giving the global-limit relation (`docs/physicsguide_sections/subgrid3d.md` §9c):
$$\frac{\Delta t_{\max}^{\mathrm{LTS}}}{\Delta t_{\max}^{\mathrm{global}}} = \left(\prod_{m<L} p_m\right)\frac{\sqrt{x_{\mathrm{adm},L}}}{2},$$
which equals $1.9759$, $2.9570$, $3.9070$, and $5.8253$ for ratios $[2]$, $[3]$, $[2, 2]$, and $[2, 3]$ at $\nu_l = 0.05\,l$.

### 2.7 Fast local evaluation of $R_0 \mathbf u$

At level $l$, the operator inside $r_l$ is $X_l = \tau_{l-1}^2 Q_l K R_l = \tau_{l-1}^2 E_l M_{ll}^{-1} E_l^T K R_l$. Because the outermost left factor of $Q_l$ is $E_l$, $X_l \mathbf v$ is strictly zero outside $F_l$ for any vector $\mathbf v$. In the three-term Chebyshev recurrence for $r_l(X_l)\mathbf y$,
$$D_0 = 1, \quad \mathbf w_0 = \mathbf 0, \qquad D_1 = \delta_l, \quad \mathbf w_1 = \mathbf y,$$
$$D_k = 2\delta_l D_{k-1} - D_{k-2}, \qquad \mathbf w_k = \left(2\delta_l \mathbf w_{k-1} - \mathbf w_{k-2}\right) - 2\beta_l X_l \mathbf w_{k-1} + 2\beta_l D_{k-1}\mathbf y \quad (k = 2, \dots, p_l),$$
the difference $\mathbf w_k - D_k(\delta_l)\mathbf y$ is supported exclusively on $F_l$ for every $k \ge 1$. On the complement $C_l = F_0 \setminus F_l$, $\mathbf w_k[C_l] = D_k(\delta_l)\mathbf y[C_l]$ identically, and dividing by $D_{p_l}(\delta_l) = T_{p_l}(\delta_l)$ at the end returns $\mathbf w_{p_l}[C_l] = \mathbf y[C_l]$. Therefore:
* Inside the $k$-loop, only the $|F_l|$-vector $\mathbf w_k[F_l]$ is stored and updated.
* Because $X_l$ has $Q_l$ on the *left* (unlike the 2D `wakis/unstructured/lts.py` formulation, which placed the projector on the right of $M^{-1} K$), $X_l$ has **no nonzero rows outside $F_l$**. The buffer $B_l$ is read only once before the $k$-loop to evaluate $(K\mathbf y)[F_l]$ on the incoming vector $\mathbf y = R_l \mathbf v$ and is never updated inside the Chebyshev recurrence.
* For the inner product $X_l \mathbf w_{k-1}$ inside the $k$-loop, only $\mathbf w_{k-1}[F_l]$ is needed because $R_l$ modifies only $F_{l+1} \subset F_l$ and $(K \mathbf v)[F_l]$ for a vector $\mathbf v$ supported on $F_l$ depends only on $\mathbf v[F_l]$.

To evaluate $(K \mathbf v)[F_l]$ locally without touching the full mesh:
1. Let $\Phi_l$ be the set of primal faces incident to $F_l$ in $C_{\mathrm{eff}}$: for Whitney or Voronoi without massless edges, $\Phi_l = \mathrm{supp}(C_{f,:,F_l})$.
2. For Voronoi with massless edges, let $S_l^{\mathrm{ms}}$ be the union of connected components of $K_{ss}$ that are coupled to $F_l$ through $K_{sf}[:, F_l]$. Then $\Phi_l = \mathrm{supp}(C_{f,:,F_l}) \cup \mathrm{supp}(C_{s,:,S_l^{\mathrm{ms}}})$, and the local Schur elimination solves
   $$K_{ss}[S_l^{\mathrm{ms}}, S_l^{\mathrm{ms}}]\,\mathbf e_s = K_{sf}[S_l^{\mathrm{ms}}, F_l]\,\mathbf v[F_l]$$
   using a per-level sparse LU factorization of the block $K_{ss}[S_l^{\mathrm{ms}}, S_l^{\mathrm{ms}}]$. Because massless Voronoi edges occur as isolated edges or small clusters in non-percolating meshes, $|S_l^{\mathrm{ms}}| \ll n_{\mathrm{massless}}$ (`TetLTS.massless_frac_per_level` records $|S_l^{\mathrm{ms}}| / n_{\mathrm{massless}}$ for every level).
3. The one-ring buffer $B_l \subset F_0 \setminus F_l$ is the set of coarse free edges outside $F_l$ that share a face in $\Phi_l$ or a massless component in $S_l^{\mathrm{ms}}$ with $F_l$. For an arbitrary full-length vector $\mathbf y = R_l \mathbf v$, $(K\mathbf y)[F_l]$ depends only on $\mathbf y[F_l]$ and $\mathbf y[B_l] = \mathbf v[B_l]$ (since $B_l \cap F_{l+1} = \emptyset$ by nestedness).

### 2.8 Partition rules (Voronoi and Whitney)

1. **Nestedness and one-ring buffer (`nest_levels`):** For every level $l \ge 1$, all edges in $F_l$ must belong to $F_{l-1}$, and for $l \ge 2$ the one-ring buffer $B_l$ of $F_l$ in the sparsity graph of $K$ (including massless-component edges for Voronoi) is promoted into $F_{l-1}$ ($\mathrm{level}[B_l] \leftarrow \max(\mathrm{level}[B_l],\, l-1)$). Without this buffer (`mutate="nonest"`), $K R_l \mathbf w$ at level $l-1$ leaks across $F_{l-1}$ into $F_0 \setminus F_{l-1}$.
2. **Massless-component closure (Voronoi):** Every free edge coupled through $K_{sf}$ to a massless component in $S_l^{\mathrm{ms}}$ belongs to $F_l \cup B_l \subset F_{l-1}$. In `partition_levels`, free edges directly coupled to $S_l^{\mathrm{ms}}$ whose local edge rate exceeds the level-$(l-1)$ threshold are promoted to $F_l$.
3. **Whitney mass-ring closure (`whitney_ring=1`):** Because $Q_l = E_l M_{ll}^{-1} E_l^T$ drops the off-diagonal consistent-mass entries $M_{F_l, B_l}$ during substeps, modes localized right on $\partial F_l$ see the boundary of the submatrix inverse. Growing each raw fine set $F_l$ by one edge-adjacency ring in the sparsity graph of $|M_1| + |K_{\mathrm{raw}}|$ before `nest_levels` (`whitney_ring=1`, the default for `hodge="whitney"`) pads the boundary of $M_{ll}$ away from the stiff elements.
4. **Automatic level assignment (`partition_levels`):** From `quality.element_dt_bounds(solver)`, each tetrahedron has an element bound $\Delta t_T$ and each free edge is assigned $\Delta t_e = \min_{T \ni e} \Delta t_T$ (for Voronoi, also capped by $2/\sqrt{K_{ee}/M_{ee}}$). Given ratios $(p_1, \dots, p_L)$ with cumulative products $P_l = \prod_{m \le l} p_m$ (`P_0 = 1`) and a safety factor $\alpha \in (0, 1)$ (default $0.85$), `partition_levels` selects the largest target coarse step $\Delta t_{\mathrm{target}} \le \alpha P_L \min_e \Delta t_e$ such that assigning edges with $\alpha P_{l-1} \Delta t_e < \Delta t_{\mathrm{target}}$ to level $\ge l$ keeps $|F_1| / |F_0| \le \texttt{max\_fine\_fraction}$ after Whitney ring closure and `nest_levels`. The actual time step used by the solver is then certified by `lts_dt_max`.

### 2.9 Consistency, dispersion, and the coarse-step accuracy ceiling

Since $r_l(0) = 1$ and $r'_l(0) = -c_{l,1}$ (`wakis.unstructured.lts3d.r_prime_0`) with
$$c_{l,1} = \frac{\beta_l T''_{p_l}(\delta_l)}{2 T'_{p_l}(\delta_l)} \;\underset{\nu_l \to 0}{\longrightarrow}\; \frac{p_l^2 - 1}{12 p_l^2},$$
the single-level ($L = 1$) effective stiffness expands as
$$S_0 = K - \Delta t^2 c_1 K Q_1 K + \mathcal O(\Delta t^4).$$
Let $(\lambda, \mathbf u)$ be a semi-discrete eigenpair of $(K, M)$, $K \mathbf u = \lambda M \mathbf u$ with $\lambda = \omega^2$ and $\mathbf u^T M \mathbf u = 1$. First-order Rayleigh-Schrödinger perturbation of $(S_0, M)$ around $(K, M)$ shifts the eigenvalue to
$$\tilde\lambda = \mathbf u^T S_0 \mathbf u + \mathcal O(\Delta t^4) = \lambda\Big(1 - c_1 \theta_F (\omega \Delta t)^2 + \mathcal O\big((\omega \Delta t)^4\big)\Big),$$
where the fine-level energy fraction is
$$\theta_F := \frac{(K\mathbf u)_{F_1}^T M_{11}^{-1} (K\mathbf u)_{F_1}}{\lambda^2} = \frac{\mathbf u^T K Q_1 K \mathbf u}{\lambda^2} \in [0, 1]$$
(when $M$ is diagonal and $\mathbf u$ is an eigenvector of $M^{-1}K$, $M^{-1}K\mathbf u = \lambda\mathbf u$ so $\theta_F = \sum_{e \in F_1} M_{ee} u_e^2$). Combining $\tilde\lambda$ with the leapfrog dispersion relation $(2/\Delta t)\sin(\omega_{\mathrm{LTS}}\Delta t / 2) = \sqrt{\tilde\lambda}$ yields the fractional frequency shift relative to the semi-discrete eigenfrequency $\omega$:
$$\frac{\omega_{\mathrm{LTS}} - \omega}{\omega} = \frac{(\omega\Delta t)^2}{24}(1 - 12 c_1 \theta_F) + \mathcal O\big((\omega\Delta t)^4\big) = \underbrace{+\frac{(\omega\Delta t)^2}{24}}_{\delta_t\text{ (coarse leapfrog)}} \underbrace{- \frac{c_1}{2}\theta_F (\omega\Delta t)^2}_{\delta_{\mathrm{LTS}}\text{ (substep correction)}} + \mathcal O\big((\omega\Delta t)^4\big).$$

*Sanity check at $\theta_F = 1$, $p_1 = 2$, $\nu_1 = 0$:* When $F_1 = F_0$ ($\theta_F = 1$) and $\nu_1 = 0$, $c_1 = (2^2 - 1)/(12 \cdot 2^2) = 1/16$ (`r_prime_0(2, 0) = -0.0625` in `wakis/unstructured/lts3d.py`; at the default damping $\nu_1 = 0.05$, `r_prime_0(2, 0.05) = -0.06403368388965096`), so $x\, r_1(x) = x - x^2 / 16 + \mathcal O(x^3)$, $1 - 12 c_1 = 1 / p_1^2 = 1/4$, and the leading phase error is $+\frac{(\omega\Delta t)^2}{96}$. On $F_1 = F_0$, $Q_1 K = M^{-1} K$ has eigenvalue $\omega^2$, and the exact relation $2\sin(\omega_{\mathrm{LTS}}\Delta t/2) = \sqrt{\Delta t^2 \omega^2 (1 - \omega^2\Delta t^2 / 16)}$ gives $\omega_{\mathrm{LTS}}\Delta t / 2 = 2\arcsin(\frac{1}{2}\omega(\Delta t/2))$, i.e., one coarse LTS step with $p_1 = 2, \nu_1 = 0, F_1 = F_0$ is identically two global leapfrog steps of size $\Delta t / 2$, whose dispersion error is $+\frac{1}{24}(\omega\Delta t / 2)^2 = +\frac{(\omega\Delta t)^2}{96}$.

*Coarse-step accuracy ceiling at $\theta_F \ll 1$:* For a low-order cavity or scattering mode whose energy lies mostly in the coarse mesh ($\theta_F \ll 1$, because $F_1$ contains only a small fraction of stiff elements), the LTS perturbation $\delta_{\mathrm{LTS}} = -\frac{1}{2}c_1 \theta_F (\omega\Delta t)^2$ is suppressed by $\theta_F$, leaving the unattenuated coarse leapfrog phase error $\delta_t = +(\omega\Delta t)^2 / 24$. Because $\Delta t = (\Delta t / \Delta t_{\mathrm{global}})\Delta t_{\mathrm{global}}$, $\delta_t$ is $(\Delta t / \Delta t_{\mathrm{global}})^2$ times larger than in a global-step run. When $\Delta t$ is raised several-fold on a mesh with small spatial error, $\delta_t$ can exceed the spatial discretization error $e_{\mathrm{semi}} = (\omega_{\mathrm{semi}} - \omega_{\mathrm{exact}})/\omega_{\mathrm{exact}}$. The study therefore separates the spatial error `err_semi` (after removing leapfrog dispersion) from the total time-domain error `err_td`.

### 2.10 Exact discrete transpose and adjoint gradients

Since $S_0 = K R_0$ is symmetric and $K = K^T$, $K R_0 = R_0^T K$. In the forward step (`solver3d.py::one_step` and `adjoint.py::_step3d`), $\mathbf u^n$ enters Faraday's law through $\mathbf z = R_0 \mathbf u^n$:
$$\mathbf b^{n+1/2} = \mathbf b^{n-1/2} - \Delta t\, C_{\mathrm{eff}} R_0 \mathbf u^n, \qquad \mathbf u^{n+1} = \mathbf u^n + \Delta t\, M^{-1}(C_f^T W \mathbf b^{n+1/2} - \mathbf j^{n+1/2}).$$
Transposing the two substeps in reverse order (`adjoint.py::_step3d_T`) changes only the Faraday transpose: the vector $\mathbf g = -\Delta t\, C_{\mathrm{eff}}^T \bar{\mathbf b}$ is mapped through $R_0^T$ before accumulating into $\bar{\mathbf u}$. Here
$$R_0^T = r_1(X_1^T) \cdots r_L(X_L^T), \qquad X_l^T = \tau_{l-1}^2 R_l^T K Q_l.$$
Because $Q_l = E_l M_{ll}^{-1} E_l^T$ stands on the *right* of $X_l^T$, evaluating $X_l^T \mathbf w$ reads only $\mathbf w[F_l]$, solves with $M_{ll}$, applies $K$ (producing nonzeros only on $F_l \cup B_l$), and applies $R_l^T$ (which modifies only $F_{l+1} \cup B_{l+1} \subset F_l$). Consequently, $X_l^T \mathbf w$ is supported strictly on $F_l \cup B_l$, and `TetLTS.apply_T` evaluates the transpose locally on $F_l \cup B_l$ at every level (avoiding the full-length transpose recursion used in `wakis/subgrid3d/lts.py`).

*Material-gradient restriction:* The LTS operator $R_0$ depends on $\varepsilon$ and $\mu$ through $Q_l$ and $K$ on every tetrahedron incident to $F_1$ or $\Phi_1$. When all design tetrahedra are disjoint from $F_1$ (`len(design_f1_overlap) == 0`), $\partial R_0 / \partial \varepsilon_c = 0$ and the standard adjoint gradient formula differentiating only the outer $M^{-1}$ in Ampère's law is **exact** (`test_gradient_vs_fd`). When a design tetrahedron touches $F_1$, omitting $\partial R_0 / \partial \varepsilon_c$ leaves an $\mathcal O(\Delta t^2)$ inexactness; `UnstructuredAdjointStepper` detects this overlap at initialization and raises `NotImplementedError` unless `allow_inexact_lts_gradient=True` is passed explicitly.

### 2.11 Spherical CFS-PML with LTS

In `wakis.unstructured.pml3d.PML3D`, the absorbing shell occupies the cospherical frustum-prism layers at $r > r_0$. A step of `PML3D.apply(u, b, dt, lts=lts)` consists of Operator 1 (Faraday update plus auxiliary recursive-convolution filter driven by $\mathbf u$), Operator 2 (Ampère update plus auxiliary filter driven by $\mathbf b$), and the static Schur reconstruction on massless edges. When $F_1 \cup B_1$ and $\Phi_1$ are strictly disjoint from the PML shells — specifically, every PML edge damping weight vanishes on $F_1 \cup B_1$, every PML face damping weight vanishes on $\Phi_1$, and the massless-edge PML coupling block `pml._S_bu[:, F_1 ∪ B_1]` has zero nonzeros (verified at setup by `TetLTS.check_pml(pml)`) — $R_0 \mathbf u$ differs from $\mathbf u$ only on $F_1$, where Operator 1 acts as the unmodified interior Faraday update. Passing $\mathbf z = R_0 \mathbf u$ into Operator 1 (and applying $R_0^T$ to the $\bar{\mathbf u}$ increment produced by Operator 1$^T$ in `PML3D.step_T`) therefore steps the interior with LTS while advancing the PML shell at the coarse step $\Delta t$, preserving the exact discrete transpose (`test_dot_product[voronoi_pml]`). Fine edges inside or adjacent to the PML raise `ValueError` at setup.

### 2.12 Work accounting and cost model

One full global leapfrog step performs $W_{\mathrm{full}}$ scalar sparse operations (`TetLTS.full_step_work`):
* for `hodge="voronoi"`, $2\,\mathrm{nnz}(C_f)$ for $C_f \mathbf u$ and $C_f^T(W\mathbf b)$, plus $2\big(\mathrm{nnz}(C_s) + \mathrm{nnz}(K_{sf}) + \mathrm{nnz}(L_{ss}) + \mathrm{nnz}(U_{ss})\big)$ when massless edges are present;
* for `hodge="whitney"`, $2\,\mathrm{nnz}(C_f) + \mathrm{nnz}(M_2) + 2\,\mathrm{nnz}(L_{M_1})$.

Inside $R_0 \mathbf u$, level $l$ executes $n_{\mathrm{calls},l} = (p_l - 1)\prod_{m<l} p_m$ inner applications of $X_l$ on $F_l$, plus $n_{\mathrm{init},l} = \prod_{m<l} p_m$ initial buffer evaluations on $F_l \cup B_l$ when $l < L$. Defining the exact nonzero count $W_{\mathrm{LTS}} = W_{\mathrm{full}} + \sum_{l=1}^L W_l$ (`TetLTS.work_per_coarse_step`), the ratio of arithmetic work to reach a fixed physical time $T$ is
$$G_{\mathrm{work}} = \frac{\Delta t^{\mathrm{LTS}}}{\Delta t^{\mathrm{global}}}\cdot\frac{W_{\mathrm{full}}}{W_{\mathrm{LTS}}} = \frac{\Delta t^{\mathrm{LTS}} / \Delta t^{\mathrm{global}}}{1 + \sum_{l=1}^L (W_l / W_{\mathrm{full}})}.$$
For a single level ($L = 1$, $n_{\mathrm{init},1} = 0$) with per-call fraction $w_1 = (W_1 / (p_1 - 1)) / W_{\mathrm{full}}$,
$$G_{\mathrm{work}} = \frac{\Delta t^{\mathrm{LTS}} / \Delta t^{\mathrm{global}}}{1 + (p_1 - 1) w_1},$$
so **$G_{\mathrm{work}} < 1$ whenever $w_1 > (\Delta t^{\mathrm{LTS}} / \Delta t^{\mathrm{global}} - 1) / (p_1 - 1)$**. On a 3D tetrahedral mesh, each free edge is coupled in $K$ to a large one-ring stencil (on the `_graded_box(seed=2)` test mesh of `tests/test_116_unstructured_lts3d.py`, the mean off-diagonal edge degree $\mathrm{nnz}(K)/n_{\mathrm{free}} - 1$ is $11.14$ for Voronoi at `massless_tol=1e-2`, $23.27$ for Voronoi at `massless_tol=5e-2` after massless Schur elimination, and $12.09$ for Whitney; at $1.5\,\Delta t_{\max}$ with $L = 1$, $|F_1| = 172$ and $|B_1| = 218$, so $|B_1| / |F_1| = 1.27$ and $w_1 = W_1 / W_{\mathrm{full}} = 0.254$ for Voronoi, while $|F_1| = 192$, $|B_1| = 285$, $|B_1| / |F_1| = 1.48$, and $w_1 = 0.257$ for Whitney), which limits the achievable work and wall-clock gain unless the stiff edges occupy a small fraction of the mesh.

---

## 3. Verification (`tests/test_116_unstructured_lts3d.py`)

All 30 parametrized test items in `tests/test_116_unstructured_lts3d.py` pass (`29 passed` in the fast suite, `1 passed` in `-m slow`, `0 xfailed`). Every test includes a concrete mutation that was executed and verified to fail.

| Test function | Hodge / case | Measured value | Threshold | Executed mutation and failing value |
| :--- | :--- | :--- | :--- | :--- |
| `test_known_limits` | `voronoi` | `bitwise_empty=True`, $F_1=F_0$ vs $\Delta t/p$: `3.638e-15` | Bitwise equal ($F_1=\emptyset$); $\le 5\times 10^{-15}$ ($\nu=0, F_1=F_0$) | `mutate="consistency"` ($r_l(0)=1.05$): min rel. diff `4.353e-02` ($> 10^{-2}$) |
| `test_known_limits` | `whitney` | `bitwise_empty=True`, $F_1=F_0$ vs $\Delta t/p$: `1.001e-15` | Bitwise equal ($F_1=\emptyset$); $\le 5\times 10^{-15}$ ($\nu=0, F_1=F_0$) | `mutate="consistency"` ($r_l(0)=1.05$): min rel. diff `2.959e-02` ($> 10^{-2}$) |
| `test_fast_equals_reference_and_dense` | `voronoi`, `[2]` | fast vs ref `1.143e-16`, mat vs ref `1.034e-16`, step vs dense `1.048e-16` | $\le 10^{-14}$ | `mutate="interface"`: `2.374e-04`; `mutate="drop_buffer"`: `2.374e-02` ($> 10^{-4}$) |
| `test_fast_equals_reference_and_dense` | `whitney`, `[2]` | fast vs ref `1.108e-16`, mat vs ref `1.746e-16`, step vs dense `1.246e-16` | $\le 10^{-14}$ | `mutate="interface"`: `3.044e-04`; `mutate="drop_buffer"`: `3.044e-02` ($> 10^{-4}$) |
| `test_fast_equals_reference_and_dense` | `voronoi`, `[2, 2]` | fast vs ref `1.793e-16`, mat vs ref `2.132e-16`, step vs dense `1.614e-16` | $\le 10^{-14}$ | `mutate="interface"`: `1.252e-04`; `mutate="drop_buffer"`: `1.252e-02` ($> 10^{-4}$) |
| `test_fast_equals_reference_and_dense` | `whitney`, `[2, 2]` | fast vs ref `1.933e-16`, mat vs ref `8.064e-16`, step vs dense `4.073e-16` | $\le 10^{-14}$ | `mutate="interface"`: `1.607e-04`; `mutate="drop_buffer"`: `1.607e-02` ($> 10^{-4}$) |
| `test_fast_equals_reference_and_dense` | `voronoi`, `[2, 3]` | fast vs ref `1.622e-16`, mat vs ref `2.669e-16`, step vs dense `2.412e-16` | $\le 10^{-14}$ | `mutate="interface"`: `1.277e-04`; `mutate="drop_buffer"`: `1.277e-02` ($> 10^{-4}$) |
| `test_fast_equals_reference_and_dense` | `whitney`, `[2, 3]` | fast vs ref `1.602e-16`, mat vs ref `8.583e-16`, step vs dense `4.431e-16` | $\le 10^{-14}$ | `mutate="interface"`: `1.636e-04`; `mutate="drop_buffer"`: `1.636e-02` ($> 10^{-4}$) |
| `test_stiffness_symmetric_and_criterion_matches_dense` | `voronoi` | $\Delta t_{\mathrm{LTS}}/\Delta t_{\max}=1.4550$, $\|S_0 - S_0^T\|_F/\|S_0\|_F = 5.861\times 10^{-17}$, $|\rho_{\mathrm{Lanczos}}-\rho_{\mathrm{dense}}| = 1.665\times 10^{-15}$ | Asymmetry $\le 10^{-14}$; $|\Delta\rho| \le 10^{-3}$; `is_stable` matches dense at $0.96, 1.04\times \Delta t_{\mathrm{LTS}}$ | `mutate="nonsym_Q"`: asymmetry `5.039e-04` ($> 10^{-4}$) |
| `test_stiffness_symmetric_and_criterion_matches_dense` | `whitney` | $\Delta t_{\mathrm{LTS}}/\Delta t_{\max}=1.0489$, $\|S_0 - S_0^T\|_F/\|S_0\|_F = 1.738\times 10^{-16}$, $|\rho_{\mathrm{Lanczos}}-\rho_{\mathrm{dense}}| = 1.776\times 10^{-15}$ | Asymmetry $\le 10^{-14}$; $|\Delta\rho| \le 10^{-3}$; `is_stable` matches dense at $0.96, 1.04\times \Delta t_{\mathrm{LTS}}$ | `mutate="nonsym_Q"` ($P_l M_1^{-1}$): asymmetry `1.009e-03` ($> 10^{-4}$) |
| `test_modified_energy_conserved` | `voronoi` | `295` massless edges, 2000-step drift `2.665e-15` | $\le 50\varepsilon_{\mathrm{mach}}\sqrt{n} = 4.965\times 10^{-13}$ | `mutate="interface"` drift `2.246e-05`; `mutate="drop_buffer"` drift `2.238e-03` |
| `test_modified_energy_conserved` | `whitney` | `0` massless edges, 2000-step drift `5.120e-15` | $\le 50\varepsilon_{\mathrm{mach}}\sqrt{n} = 4.965\times 10^{-13}$ | `mutate="interface"` drift `8.171e-06`; `mutate="drop_buffer"` drift `8.699e-04` |
| `test_energy_conserved_while_unstable` | `voronoi` | $\Delta t = 1.12\,\Delta t_{\max}^{\mathrm{LTS}}$, 10-step field growth `1.106e+04`, $|E^{n+1/2}/E^{1/2}-1| = 2.232\times 10^{-15}$ | Growth $> 5$; $|E^{n+1/2}/E^{1/2}-1| \le 10^{-11}$ | Asserting bounded state fails (`growth = 1.106e+04`) |
| `test_stability_enforced` | `voronoi` | $\Delta t_{\mathrm{LTS}}/\Delta t_{\mathrm{global}} = 1.97586$, 5000-step $\|\mathbf u\|_M/\|\mathbf u^0\|_M = 4.135\times 10^{1}$ | Bounded $< 100$ at $0.97\,\Delta t_{\mathrm{LTS}}$; `1.03 * dt_lts` raises `ValueError` | `mutate="nonest"` at same $\Delta t$: `RuntimeError` (`> 1e6` in 200 steps) |
| `test_global_limit_relation` | `voronoi`, `[2]` | Top-mode energy in $F_L$: `1.00000`; meas `1.97575` vs pred `1.97590` (`rel_err = 7.871e-05`) | Relative error $\le 2\times 10^{-3}$ | `mutate="equal_damping"` tested on `[2, 2]` below |
| `test_global_limit_relation` | `voronoi`, `[3]` | Top-mode energy in $F_L$: `1.00000`; meas `2.95679` vs pred `2.95703` (`rel_err = 7.871e-05`) | Relative error $\le 2\times 10^{-3}$ | `mutate="equal_damping"` tested on `[2, 2]` below |
| `test_global_limit_relation` | `voronoi`, `[2, 2]` | Top-mode energy in $F_L$: `1.00000`; meas `3.90667` vs pred `3.90698` (`rel_err = 7.871e-05`) | Relative error $\le 2\times 10^{-3}$ | `mutate="equal_damping"` ($\nu_1=\nu_2=0.05$ at $2.2\,\Delta t_{\max}$): `is_stable=False`, `margins=[0.9999974, 0.9774404]` ($m_1 > 1-g = 0.9999$) |
| `test_box_frequency` | `voronoi` | $\theta_F = 0.7470$, (a) $|f_{\mathrm{TD}}-f_{\mathrm{pred}}|/f_{\mathrm{pred}} = 3.941\times 10^{-14}$, (b) $|\Delta\lambda/\lambda - (-c_1\theta_F\omega^2\Delta t^2)| = 7.702\times 10^{-7}$, (c) $|(f_{\mathrm{TD}}-f_{\mathrm{ana}})/f_{\mathrm{ana}} - (e_{\mathrm{semi}}+\delta_t+\delta_L)| = 5.44\times 10^{-7}$ (`err_c = 4.069e-03 <= 4.644e-03`) | (a) $\le 2\times 10^{-4}$; (b) $\le 2(\omega\Delta t)^4 = 1.789\times 10^{-6}$; (c) $\le 1.1|e_{\mathrm{semi}}| + \delta_t + |\delta_L| = 4.644\times 10^{-3}$ | `mutate="consistency"`: $|f_{\mathrm{TD}}-f_{\mathrm{pred}}|/f_{\mathrm{pred}} = 1.831\times 10^{-2}$ ($> 10^{-2}$) |
| `test_box_frequency` | `whitney` | $\theta_F = 0.9998$, (a) `4.552e-14`, (b) `2.015e-08` ($\le 1.727\times 10^{-5}$), (c) `err_c = 1.114e-02 <= 1.264e-02` | (a) $\le 2\times 10^{-4}$; (b) $\le 2(\omega\Delta t)^4 = 1.727\times 10^{-5}$; (c) $\le 1.264\times 10^{-2}$ | `mutate="consistency"`: $|f_{\mathrm{TD}}-f_{\mathrm{pred}}|/f_{\mathrm{pred}} = 2.467\times 10^{-2}$ ($> 10^{-2}$) |
| `test_dot_product` | `voronoi_pec`, `whitney_pec`, `voronoi_pml` | Relative dot-product error: `5.306e-18`, `1.788e-17`, `4.617e-18`; `bitwise_forward=True` | $\le 10^{-14}$ (`DOT_TOL`) | `mutate="transpose"` ($R_0$ in place of $R_0^T$): `4.079e-03`, `4.689e-04`, `2.763e-05` ($> 10^{-5}$) |
| `test_gradient_vs_fd` | `voronoi` | $|F_1|/|F_0| = 0.287$, max rel. diff vs central FD on 5 design cells: `5.687e-08`; touching $F_1$ raises `NotImplementedError` | $\le 10^{-6}$ | Identity in place of `lts.apply_T` in `_step3d_T`: `3.476e-03` ($> 10^{-3}$) |
| `test_pml_with_lts` | `voronoi` | 10-layer PML vs 24-extra-layer vacuum reference: `4.452e-05`; edge in PML raises `ValueError` | $< 10^{-2}$ | Zeroing `pml.sigma_r`: `9.404e-01` ($21121\times$ worse, $> 10\times$) |
| `test_pml_lts_late_time` (`slow`) | `voronoi` | 20 000 coarse steps: $E(20\text{k})/E(10\text{k}) = 0.950870$, $E(20\text{k})/E_{\mathrm{pulse}} = 5.990\times 10^{-8}$ | $E(20\text{k}) \le 1.05\,E(10\text{k})$ and $\le 10^{-3}\,E_{\mathrm{pulse}}$ | Unstable or non-absorbing PML violates decay bound |
| `test_partition_rules` | `voronoi`, `whitney` | Component closure ratio `0.740973`, `0.693776` ($\le 1$) | Ratio $\le 1$; `nest_levels` leaves 0 level-0 neighbours of $F_2$ | `mutate="nonest"`: leaves `11` (`voronoi`) and `1` (`whitney`) level-0 edges in $B_2$ |
| `test_guards` | 7 unsupported configs | `DeviceStepper`, `LocalCluster`, `DistributedFIT`, `attach_ports`, dispersive, $\sigma > 0$, `bc="abc"` | All 7 raise `NotImplementedError` | Removing any guard allows unsupported execution |
| `test_work_accounting` | `voronoi`, `whitney` | `full=16792, per_level=[7180, 2724], total=26696` (`voronoi`); `full=17428, per_level=[8301, 2640], total=28369` (`whitney`) | Exact match to direct CSR/LU/Cholesky nnz sums; $G_{\mathrm{work}} = (\Delta t_{\mathrm{LTS}}/\Delta t_{\mathrm{global}})\cdot W_{\mathrm{full}}/W_{\mathrm{LTS}}$ | Omitting a level term breaks the identity |

---

## 4. Study: accuracy, cost, and stability on 3D tetrahedral meshes

Driver: `studies/unstructured-lts3d/run_study.py` (`--quick` for smoke verification, `--part {all,preflight,bounds,accuracy,timing,stability}` for the Cloud Batch study), plotter `studies/unstructured-lts3d/plot_figures.py`, verifier `studies/unstructured-lts3d/verify_numbers.py`.

### 4.1 Method and study design

* **Cases (`build_case_mesh`):**
  1. **C1 (`C1_graded_box`):** PEC cavity ($1.0 \times 0.75 \times 0.5\;\mathrm{m}$), raw Delaunay `box_mesh` with $x$-coordinates refined geometrically towards $x = 0$ down to $h/4$, interior jitter $0.2$, seeds $\{1, 2, 3\}$, 4-point $h$ ladder, analytic $\mathrm{TM}_{110}$ mode;
  2. **C2 (`C2_uniform_box`):** Same PEC cavity and $h$ ladder on a uniform tensor grid with interior jitter $0.2$ (sliver-limited only), seeds $\{1, 2, 3\}$, analytic $\mathrm{TM}_{110}$ mode;
  3. **C3 (`C3_pillbox_delaunay`):** CAD PEC pillbox (`make_mesh("pillbox", cpw, optimize="none")` from `studies/unstructured-cadmesh/cadmesh_study.py`), $\mathrm{cpw} \in \{10, 15, 20, 25\}$, analytic $\mathrm{TM}_{010}$ mode;
  4. **C4 (`C4_pillbox_gmsh`):** Same CAD pillbox meshed with `optimize="gmsh"` (non-Delaunay; Whitney Hodge arms only).

* **Arms and named baselines:**
  * *Voronoi Hodge:* `G` (global leapfrog at $\mathrm{cfl}=0.95$, `massless_tol=1e-2`), `G-imp` (`improve_dt` node-relocated mesh with global leapfrog), `G-ml3e-2` and `G-ml1e-1` (global leapfrog with `massless_tol=3e-2` and `1e-1`), `L` (automatic LTS `lts=dict(p=2, dt="auto")`), `L-dt1.25`, `L-dt1.5`, `L-dt2.0`, `L-dt3.0` (LTS at fixed coarse steps $\Delta t \in \{1.25, 1.5, 2.0, 3.0\}\times 0.95\,\Delta t_{\mathrm{global}}$), and `L-imp` (automatic LTS on the `improve_dt` mesh).
  * *Whitney Hodge:* `GW` (global leapfrog), `LW-element` (LTS with `indicator="element"`), and `LW-edge_rate` (LTS with `indicator="edge_rate"`).

* **Parts and outputs (`studies/unstructured-lts3d/results/`):**
  1. `preflight.json`: memory preflight via `wakis.unstructured.estimate.estimate_memory` (`< 2.0 GB`);
  2. `bounds.json` (figure `docs/img/lts3d_levels.png`): element stability bound distributions $\Delta t_T / \Delta t_{\mathrm{global}}$, level assignments, and predicted work curves `work_ratio_vs_G` vs $\Delta t / \Delta t_{\mathrm{global}}$;
  3. `accuracy.json` and `timing.json` (figure `docs/img/lts3d_cost.png`): semi-discrete (`err_semi` from `eigenmodes`) and time-domain (`err_td` from `wakis.resonances.harminv(..., method="pencil")`) frequency errors, setup-time breakdown, break-even simulated time `break_even_ns`, piecewise-linear log-log equal-error wall-time gains (`GAIN`, `LOSS`, `UNRESOLVED`, `NO_OVERLAP`), and 5-repetition interleaved single-core timing (`wall_per_ns_median`, `speedup_vs_G`);
  4. `stability.json` (figure `docs/img/lts3d_stability.png`): $\rho(\Delta t)$ and level positivity margins vs $\Delta t / \Delta t_{\mathrm{global}}$ alongside the derived global-limit prediction, modified-energy trajectories for certified LTS vs `interface` and `drop_buffer` mutations, and the unstable trajectory at $1.05\,\Delta t_{\mathrm{LTS}}$ showing conserved indefinite energy alongside diverging state norm.

### 4.2 Study results

<!-- RESULTS: filled from studies/unstructured-lts3d/results/*.json after the Batch study -->

---

## 5. Limits & failure modes

1. **Modest time-step spread on ungraded meshes (Voronoi).** On quasi-uniform Delaunay or `improve_dt`-smoothed tetrahedral meshes, the bulk of the element bounds $\Delta t_T$ cluster near $\Delta t_{\max}^{\mathrm{global}}$ (`studies/unstructured-cadmesh/results/bounds.json`, `docs/physicsguide_sections/unstructured_cadmesh.md` §3). Even a modest increase in the target coarse step pulls a large share of the mesh into $F_1 \cup B_1$ (for example, on the ungraded `_small_box()` test mesh at $1.5\,\Delta t_{\max}$, $|F_1| = 304$ and $|B_1| = 247$ out of $n_{\mathrm{free}} = 571$ free Voronoi edges) and the buffer overhead erodes the time-step gain. Node relocation (`improve_dt`, which raises $\Delta t_{\max}$ by $1.47\text{--}1.73\times$ on the pillbox benchmark in `studies/unstructured-cadmesh/results/improve.json` with zero extra per-step work) should be applied first on ungraded CAD meshes; LTS is intended for spatially graded meshes or irreducible sliver clusters locked to CAD curves, corners, or material interfaces.

2. **Buffer blow-up in 3D.** In a three-dimensional tetrahedral mesh, each free edge is coupled in $K$ to an average of $11.14$ (Voronoi, `massless_tol=1e-2`), $23.27$ (Voronoi, `massless_tol=5e-2`), or $12.09$ (Whitney) neighbouring free edges on `_graded_box(seed=2)`. Adding the mandatory one-ring stiffness-adjacency buffer $B_l$ expands a sparse fine set substantially: on `_graded_box(seed=2)` (`massless_tol=5e-2`, `test_work_accounting`), the finest level has $|F_2| = 34$ with buffer $|B_2| = 195$ ($|B_2|/|F_2| = 5.74$) for Voronoi and $|F_2| = 58$ with $|B_2| = 218$ ($|B_2|/|F_2| = 3.76$) for Whitney, while promoting $B_2$ into $F_1$ via `nest_levels` grows $|F_{1,\mathrm{raw}}| = 127$ to $|F_1| = 275$ with $|B_1| = 141$ ($|B_1|/|F_{1,\mathrm{raw}}| = 1.11$, $(|F_1| + |B_1|)/|F_{1,\mathrm{raw}}| = 3.28$) for Voronoi and $|F_{1,\mathrm{raw}}| = 216$ to $|F_1| = 366$ with $|B_1| = 271$ ($|B_1|/|F_{1,\mathrm{raw}}| = 1.25$, $(|F_1| + |B_1|)/|F_{1,\mathrm{raw}}| = 2.95$) for Whitney. Whenever the per-substep work fraction $w_1$ exceeds $(\Delta t^{\mathrm{LTS}}/\Delta t^{\mathrm{global}} - 1)/(p_1 - 1)$, the work ratio $G_{\mathrm{work}}$ drops below $1$ and LTS performs more floating-point operations than global leapfrog (§2.12).

3. **Whitney consistent mass.** The local inverse-mass operator $Q_l = E_l M_{ll}^{-1} E_l^T$ drops off-diagonal consistent-mass entries across $\partial F_l$ during substeps, and a sparse Cholesky solve with $M_{ll}$ is executed at every Chebyshev stage inside $R_0$ in addition to the full $M_1$ Cholesky solve once per coarse step in Ampère's law. Moreover, Whitney element bounds $\Delta t_T$ are $1.19\text{--}1.55\times$ pessimistic (`studies/unstructured-cadmesh/results/bounds.json`), so automatic level assignment from $\Delta t_T$ flags more edges than necessary and yields smaller $\Delta t$ gains than Voronoi (`test_stiffness_symmetric_and_criterion_matches_dense`: $\Delta t_{\mathrm{LTS}}/\Delta t_{\max} = 1.0489$ for Whitney vs $1.4550$ for Voronoi on the same graded box) unless an explicit geometric region or level array is supplied.

4. **Conserved energy is a semi-norm and does not detect instability.** The modified energy $E^{n+1/2}$ (§2.5) vanishes on $\ker S_0$ (including all discrete gradients $\ker C_{\mathrm{eff}}$) and on $\ker(C_f^T W)$, and remains algebraically conserved to machine precision even when $S_0$ has negative eigenvalues or $\rho > 1$ (`test_energy_conserved_while_unstable`: relative energy drift $2.232\times 10^{-15}$ while the field norm grows by $1.106\times 10^4$ in 10 steps). Only the Lanczos spectral certificate (`TetLTS.is_stable`) detects instability.

5. **Lanczos Ritz values are lower bounds.** An unconverged Lanczos iteration under-estimates $\mu_l = \lambda_{\max}(E_l^T S_l E_l, M_{ll})$ or $\lambda_{\max}(S_0, M)$, which errs on the unsafe side of the stability threshold. `TetLTS.is_stable` mitigates this with the safety margin $g = 10\,\mathrm{tol}$, a deterministic seeded random starting vector, and `maxiter=2000`, but tightly clustered eigenvalues at the top of the spectrum can slow Lanczos convergence.

6. **Time-dispersion accuracy ceiling.** Eigenmodes supported primarily in the coarse mesh ($\theta_F \ll 1$) experience the coarse time step $\Delta t$, whose leapfrog phase error is $\delta_t = +(\omega\Delta t)^2/24 = (\Delta t / \Delta t_{\mathrm{global}})^2 (\omega\Delta t_{\mathrm{global}})^2/24$ (§2.9). Raising $\Delta t$ several-fold when the spatial mesh is already fine can cause $\delta_t$ to dominate the spatial discretization error $e_{\mathrm{semi}}$; Part B of the study separates `err_semi` from `err_td` to make this trade-off explicit.

7. **Setup cost.** `lts_dt_max` performs up to `n_bisect=22` bisection steps, each running Lanczos on $L+1$ operators where every matvec at level $l < L$ evaluates the finer polynomial operator $R_l$. On large meshes, bisecting $\Delta t$ via Lanczos can take longer than the time-domain run; users should pass an explicit `dt` or construct `TetLTS(..., dt_s=...)` from the element-bound target and call `is_stable` once.

8. **PML geometric restriction.** All fine edges, their one-ring buffers $F_1 \cup B_1$, and their incident faces $\Phi_1$ must be strictly disjoint from the spherical CFS-PML prism shells (`TetLTS.check_pml`). If a mesh has its stiffest tetrahedra adjacent to the PML inner sphere $r = r_0$, those edges cannot be placed in $F_1$ unless $r_0$ is moved outward.

9. **Adjoint material gradient restriction.** Differentiating only the outer $M^{-1}$ in Ampère's law omits $\partial R_0 / \partial \varepsilon$ and $\partial R_0 / \partial \mu$ (§2.10). The material gradient is exact when no design tetrahedron touches $F_1 \cup B_1$ or $\Phi_1$ (`test_gradient_vs_fd`: relative difference $5.687\times 10^{-8}$ vs central finite differences); when a design cell touches $F_1$, `UnstructuredAdjointStepper` raises `NotImplementedError` by default (`allow_inexact_lts_gradient=True` overrides this to return the $\mathcal O(\Delta t^2)$ approximate gradient).

10. **Non-Delaunay meshes (Voronoi).** Faces with negative assembled dual lengths make the spatial curl-curl matrix $K$ indefinite (`docs/physicsguide_sections/unstructured_cadmesh.md` §6). Local time stepping does not cure a semi-discrete spatial instability; the negative-face guard in `UnstructuredFIT3D` still applies, and meshes with negative dual faces must be repaired with `improve_dt(..., repair=True)` / `flip_to_delaunay` or run with `hodge="whitney"`.

11. **Massless-component coupling (Voronoi).** If `massless_tol` is chosen large enough that eliminated edges percolate into large connected components of $K_{ss}$, the per-level massless set $S_l^{\mathrm{ms}}$ and buffer $B_l$ grow and the local Schur solve loses locality. `TetLTS.massless_frac_per_level` records $|S_l^{\mathrm{ms}}| / n_{\mathrm{massless}}$ at each level to monitor this.

12. **Not implemented in milestone 1 (raise `NotImplementedError`).** GPU (`DeviceStepper`), MPI (`LocalCluster`, `DistributedFIT`), lumped and waveguide ports (`ports3d.attach_ports`), dispersive materials, nonzero conductivity $\sigma > 0$, `bc="abc"`, and the hybrid Cartesian/tetrahedral solver (`wakis.hybrid`, which has its own stability bottleneck at the pyramid/tetrahedron transition) raise `NotImplementedError` (`test_guards`).

13. **Porting note for milestone 2 (GPU and MPI).** Because $X_l = \tau_{l-1}^2 Q_l K R_l$ has zero rows outside $F_l$ (§2.7), a Voronoi GPU port only requires gathering $\mathbf y[F_l \cup B_l]$, executing CSR products on $\Phi_l$ and $F_l$, and scattering the result back into $F_l$; the missing theoretical prerequisite for GPU verification is a valid round-off bound replacing $\kappa_E$, since the scaled LTS propagator norm exceeds $\kappa_E$ (`docs/physicsguide_sections/subgrid3d.md` §9e). For MPI domain decomposition, each Chebyshev stage inside $R_0$ requires a halo exchange on $F_l \cup B_l$, adding $(\prod_{m \le L} p_m) - 1$ fine-halo exchanges per coarse step.

---

## 6. References

1. J. Diaz and M. J. Grote, "Energy conserving explicit local time stepping for second-order wave equations", *SIAM J. Sci. Comput.* **31**(3), 1985–2014 (2009), doi:10.1137/070709414.
2. J. Diaz and M. J. Grote, "Multi-level explicit local time-stepping methods for second-order wave equations", *Comput. Methods Appl. Mech. Engrg.* **291**, 240–265 (2015), doi:10.1016/j.cma.2015.03.027.
3. M. J. Grote and T. Mitkova, "Explicit local time-stepping methods for Maxwell's equations", *J. Comput. Appl. Math.* **234**(12), 3283–3302 (2010), doi:10.1016/j.cam.2010.04.028.
4. C. Carle, M. Hochbruck and A. Sturm, "On leapfrog-Chebyshev schemes", *SIAM J. Numer. Anal.* **58**(4), 2404–2433 (2020), doi:10.1137/19M1292524.
5. M. J. Grote, S. Michel and S. A. Sauter, "Stabilized explicit local time-stepping for Maxwell's equations", *Math. Comp.* **90**(332), 2603–2643 (2021), doi:10.1090/mcom/3650.
6. C. Carle and M. Hochbruck, "Error analysis of multirate leapfrog-type methods for second-order semilinear ODEs", *SIAM J. Numer. Anal.* **60**(5), 2897–2924 (2022), doi:10.1137/21M1440893.
7. S. Piperno, "Symplectic local time-stepping in non-dissipative DGTD methods applied to wave propagation problems", *ESAIM: M2AN* **40**(5), 815–841 (2006), doi:10.1051/m2an:2006035.
8. E. Montseny, S. Pernet, X. Ferrières and G. Cohen, "Dissipative terms and local time-stepping improvements in a spatial high order Discontinuous Galerkin scheme for the time-domain Maxwell's equations", *J. Comput. Phys.* **227**(14), 6795–6814 (2008), doi:10.1016/j.jcp.2008.03.032.
9. A. Modave, A. St-Cyr and T. Warburton, "GPU performance analysis of nodal discontinuous Galerkin methods for acoustic and elastic models", and A. Modave et al., "A GPU-accelerated nodal discontinuous Galerkin method with high-order absorbing boundary conditions and corner/edge compatibility", *Int. J. Numer. Meth. Engng.* **112**(11), 1659–1686 (2017), doi:10.1002/nme.5576.
10. M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique", *Prog. Electromagn. Res.* **32**, 65–87 (2001), doi:10.2528/PIER00080103.
11. A. Bossavit, "Whitney forms: a class of finite elements for three-dimensional computations in electromagnetism", *IEE Proc. A* **135**(8), 493–500 (1988), doi:10.1049/ip-a-1.1988.0077.
