# 3D Subgridding: Dispersive and Conformal Media Inside Patches (`wakis.subgrid3d.media`)

This section extends the 3D hierarchical Cartesian FIT subgridding formulation of `docs/physicsguide_sections/subgrid3d.md` (`SubgridFIT3D` and `SubgridLTSFIT3D`) to:
1. **Drude and Lorentz dispersive media** crossing coarse/fine patch interfaces via the auxiliary differential equation (ADE) method (§1), and
2. **Conformal (subpixel) dielectric permittivity** inside coarse and refined blocks via the Kottke–Farjadpour–Johnson dual-volume anisotropic harmonic/arithmetic projection (§2).

All notation follows `docs/physicsguide_sections/subgrid3d.md`: $P = P_e$ is the composite edge interpolation matrix from free composite edges to active block edges, $K = P_f^T C_{\text{all}} P_e$ is the composite curl, $W$ is the diagonal composite face-reluctance matrix, and $D = |P|^T (m_{\text{all}} \circ |P|\mathbf{1})$ is the row-sum-lumped diagonal electric mass on free edges.

---

## §1 Drude and Lorentz ADE Inside Patches

### 1.1 High-Frequency Mass and Lumped Pole Mass

Consider $K_{\text{disp}}$ dispersive material regions $\Omega_k$ ($k = 1, \dots, K_{\text{disp}}$), each associated with a `DispersiveMaterial` of high-frequency relative permittivity $\varepsilon_{\infty,k} \ge 1$ and $P_k$ poles (Drude or Lorentz). On each block of `SubgridMesh3D`, every active leaf cell $c$ is assigned to at most one dispersive material $k$ (by evaluating `region.contains` or the region callable at cell centres, later rules overriding earlier ones):

1. **High-frequency edge mass (`m_all`):** In every active leaf cell $c \in \Omega_k$, the static relative permittivity $\varepsilon_r$ is replaced by $\varepsilon_{\infty,k}$ when assembling the block edge masses $m_{\text{all}}$ and the composite lumped electric mass
   $$D = |P|^T \left(m_{\text{all}} \circ |P|\mathbf{1}\right).$$

2. **Pole mass on active and free edges (`M_k`):** For each dispersive material $k$, the unprojected pole mass on all block edges is accumulated over the active leaf cells $c \in \Omega_k$ using the exact same geometric quarter weights $w_{c,e} = \frac{h_b h_c}{4 h_a}$ of `Block3D.cell_edges`:
   $$m_{p,k,e} = \varepsilon_0 \sum_{c \in \Omega_k} w_{c,e}.$$
   The composite pole mass on the $n_{\text{free}}$ free edges is lumped with the identical row-sum rule used for $D$:
   $$M_k = |P|^T \left(m_{p,k} \circ |P|\mathbf{1}\right) \in \mathbb{R}_{\ge 0}^{n_{\text{free}}}.$$

**Homogeneous identity ($M_k = D / \varepsilon_\infty$):** Suppose a single dispersive material $k$ with high-frequency permittivity $\varepsilon_\infty$ fills every active leaf cell of the mesh. Then on every active block edge $e$,
$$m_{\text{all},e} = \varepsilon_0 \sum_{c \text{ active}} \varepsilon_\infty w_{c,e} = \varepsilon_\infty m_{p,k,e}.$$
By linearity of the entrywise product and matrix-vector product with $|P|^T$,
$$D = |P|^T \left((\varepsilon_\infty m_{p,k}) \circ |P|\mathbf{1}\right) = \varepsilon_\infty |P|^T \left(m_{p,k} \circ |P|\mathbf{1}\right) = \varepsilon_\infty M_k,$$
so $M_k = D / \varepsilon_\infty$ holds identically on every coarse, fine, and interface-parent free edge (`test_d1_homogeneous_lorentz_mass_identity`: measured $\max |M_k \varepsilon_\infty / D - 1| = 4.441 \times 10^{-16}$). Omitting the row-sum factor $|P|\mathbf{1}$ (`mutate="lump_no_rowsum"`, i.e., $M_k^{\text{mut}} = |P|^T m_{p,k}$) underweights hanging-edge rows where $\sum_l |P_{el}| = 1/r < 1$, producing an $\mathcal{O}(1)$ error on interface-parent edges ($\max |M_k^{\text{mut}} \varepsilon_\infty / D - 1| = 3.333 \times 10^{-1}$ on a 2:1 patch).

### 1.2 Second-Order ADE Update and Ampère Coupling

For each pole $p$ of material $k$ with natural frequency $\omega_{0,k,p} \ge 0$ [rad/s], damping rate $\gamma_{k,p} \ge 0$ [rad/s], and strength $s_{k,p} = \Delta\varepsilon_{k,p}\,\omega_{0,k,p}^2$ (Lorentz) or $s_{k,p} = \omega_{p,k,p}^2$ (Drude, $\omega_{0,k,p} = 0$), the normalized polarization $p_{k,p} = P_{k,p} / \varepsilon_0$ lives on the active free-edge support $\mathcal{I}_k = \{i \in \{0,\dots,n_{\text{free}}-1\} : M_{k,i} > 0\}$. Centred second-order discretization of
$$\ddot{p}_{k,p} + \gamma_{k,p} \dot{p}_{k,p} + \omega_{0,k,p}^2 p_{k,p} = s_{k,p} e$$
at time level $t^n$ gives the three-level ADE recursion of `wakis/dispersive.py` (`_ade_coefficients`):
$$p_{k,p}^{n+1} = \alpha_{k,p} p_{k,p}^n + \xi_{k,p} p_{k,p}^{n-1} + \zeta_{k,p} e^n_{\mathcal{I}_k},$$
with
$$\alpha_{k,p} = \frac{2 - \omega_{0,k,p}^2 \Delta t^2}{1 + \tfrac{1}{2}\gamma_{k,p}\Delta t}, \qquad \xi_{k,p} = -\frac{1 - \tfrac{1}{2}\gamma_{k,p}\Delta t}{1 + \tfrac{1}{2}\gamma_{k,p}\Delta t}, \qquad \zeta_{k,p} = \frac{s_{k,p}\Delta t^2}{1 + \tfrac{1}{2}\gamma_{k,p}\Delta t}.$$
Note that $p_{k,p}^{n+1}$ is driven by $e^n$ (the electric field at the start of the step, matching `SolverFIT3D._one_step_dispersive` in `wakis/dispersive.py`). After updating Faraday's law $b^{n+1/2} = b^{n-1/2} - \Delta t K e^n$ and advancing $p_{k,p}^{n+1}$ for all groups and poles, Ampère's law on the free edges updates $e^{n+1}$ via
$$D (e^{n+1} - e^n) = \Delta t \left(K^H W b^{n+1/2} - P^H i^{n+1/2}\right) - \sum_{k=1}^{K_{\text{disp}}} M_k \sum_{p=1}^{P_k} \left(p_{k,p}^{n+1} - p_{k,p}^n\right).$$

### 1.3 Second-Order Leapfrog Form, PSD Pencil, and Conserved Energy ($\gamma = 0$)

In the source-free ($i = 0$), CPML-free, lossless ($\gamma_{k,p} = 0$) regime, the ADE coefficients simplify to $\alpha_{k,p} = 2 - \omega_{0,k,p}^2 \Delta t^2$, $\xi_{k,p} = -1$, and $\zeta_{k,p} = s_{k,p} \Delta t^2$, i.e.,
$$p_{k,p}^{n+1} - 2 p_{k,p}^n + p_{k,p}^{n-1} = \Delta t^2 \left(s_{k,p} e^n - \omega_{0,k,p}^2 p_{k,p}^n\right).$$
Subtracting the Ampère update at step $n-1$ from step $n$ and substituting $b^{n+1/2} - b^{n-1/2} = -\Delta t K e^n$ eliminates $b$ with $S = K^H W K$:
$$D (e^{n+1} - 2 e^n + e^{n-1}) + \sum_{k,p} M_k \left(p_{k,p}^{n+1} - 2 p_{k,p}^n + p_{k,p}^{n-1}\right) = -\Delta t^2 S e^n.$$

- **Drude poles ($\omega_{0,k,p} = 0$, $s_{k,p} = \omega_{p,k,p}^2$):** The second difference of $p_{k,p}$ is $p_{k,p}^{n+1} - 2 p_{k,p}^n + p_{k,p}^{n-1} = \Delta t^2 \omega_{p,k,p}^2 e^n$. Substituting this directly into the $e$ equation eliminates $p_{k,p}$ from the second-order dynamics and adds $\omega_{p,k,p}^2 M_k$ to the $e\text{--}e$ stiffness block.
- **Lorentz poles ($\omega_{0,k,p} > 0$, $s_{k,p} = \Delta\varepsilon_{k,p}\omega_{0,k,p}^2 > 0$):** Substituting $p_{k,p}^{n+1} - 2 p_{k,p}^n + p_{k,p}^{n-1} = \Delta t^2 (s_{k,p} e^n - \omega_{0,k,p}^2 p_{k,p}^n)$ into the $e$ equation and multiplying the $p_{k,p}$ equation by $\frac{\omega_{0,k,p}^2}{s_{k,p}} M_k$ yields the generalized Störmer leapfrog on $x = [e;\, p_{\text{Lorentz}}]$:
  $$x^{n+1} - 2 x^n + x^{n-1} = -\Delta t^2 \mathbb{M}^{-1} \mathbb{K} x^n,$$
  where the mass matrix $\mathbb{M}$ is diagonal and strictly positive definite on the active supports $\mathcal{I}_k$:
  $$\mathbb{M} = \operatorname{diag}\!\left(D,\; \frac{\omega_{0,1}^2}{s_1} M_{k_1,\mathcal{I}_{k_1}},\; \dots\right) \succ 0,$$
  and the stiffness matrix $\mathbb{K}$ has the symmetric block structure (shown for a single Lorentz pole of strength $s$, frequency $\omega_0$, and pole mass $M$, plus Drude poles):
  $$\mathbb{K} = \begin{bmatrix} S + \sum_{\text{Drude}} \omega_p^2 M_{\text{Drude}} + s M & -\omega_0^2 M \\ -\omega_0^2 M & \frac{\omega_0^4}{s} M \end{bmatrix}.$$

**Proof that $\mathbb{K}$ is symmetric positive semi-definite:** Clearly $\mathbb{K}^T = \mathbb{K}$ since $S = K^H W K$ is symmetric and $M$ is diagonal. For each Lorentz pole and any edge $i \in \mathcal{I}_k$, the $2 \times 2$ scalar coupling block is
$$\begin{bmatrix} s & -\omega_0^2 \\ -\omega_0^2 & \omega_0^4 / s \end{bmatrix} = \frac{1}{s} \begin{bmatrix} s \\ -\omega_0^2 \end{bmatrix} \begin{bmatrix} s & -\omega_0^2 \end{bmatrix},$$
which is rank-1 positive semi-definite with eigenvalues $0$ and $s + \omega_0^4 / s > 0$. Therefore, for any state vector $x = [e;\, p_1, \dots]$,
$$x^T \mathbb{K} x = \left\|W^{1/2} K e\right\|_2^2 + \sum_{\text{Drude}} \omega_p^2 e^T M_{\text{Drude}} e + \sum_{\text{Lorentz}} \frac{1}{s} (s e - \omega_0^2 p)^T M (s e - \omega_0^2 p) \ge 0.$$

**Conserved quadratic energy and exact stability threshold:** Because $\mathbb{M} \succ 0$ and $\mathbb{K} \succeq 0$ are real symmetric, the second-order leapfrog analysis of `docs/physicsguide_sections/subgrid3d.md` §9c applies verbatim to $x^n$. Defining $w^{n+1/2} = (x^{n+1} - x^n)/\Delta t$ and $\bar{x}^{n+1/2} = (x^{n+1} + x^n)/2$, multiplying $\mathbb{M}(w^{n+1/2} - w^{n-1/2}) = -\Delta t \mathbb{K} x^n = -\frac{\Delta t}{2}\mathbb{K}(\bar{x}^{n+1/2} + \bar{x}^{n-1/2})$ on the left by $\frac{1}{2}(w^{n+1/2} + w^{n-1/2})^T = \frac{1}{\Delta t}(\bar{x}^{n+1/2} - \bar{x}^{n-1/2})^T$ shows that
$$E^{n+1/2} = \frac{1}{2} (w^{n+1/2})^T \left(\mathbb{M} - \frac{\Delta t^2}{4}\mathbb{K}\right) w^{n+1/2} + \frac{1}{2} (\bar{x}^{n+1/2})^T \mathbb{K} \bar{x}^{n+1/2}$$
is an exact invariant ($E^{n+1/2} = E^{n-1/2}$ in exact arithmetic), and the time step is stable if and only if $\mathbb{M} - \frac{\Delta t^2}{4}\mathbb{K} \succ 0$, or equivalently
$$\Delta t < \Delta t_{\max,\text{disp}} = \frac{2}{\sqrt{\lambda_{\max}(\mathbb{M}^{-1}\mathbb{K})}}.$$
In `wakis/subgrid3d/media.py`, `dispersive_dt_max` computes $\lambda_{\max}(\mathbb{M}^{-1}\mathbb{K})$ via Lanczos (`scipy.sparse.linalg.eigsh` on the symmetrized operator $\mathbb{M}^{-1/2}\mathbb{K}\mathbb{M}^{-1/2}$).

**Homogeneous Drude identity:** When a single lossless Drude medium ($\varepsilon_\infty, \omega_p$) fills the domain, $D = \varepsilon_\infty D_{\text{vac}}$ and $M = D / \varepsilon_\infty = D_{\text{vac}}$, where $D_{\text{vac}}$ is the vacuum lumped mass. There are no Lorentz poles, so $\mathbb{M} = D$ and $\mathbb{K} = S + \omega_p^2 D_{\text{vac}}$, giving
$$\mathbb{M}^{-1}\mathbb{K} = \frac{1}{\varepsilon_\infty}\left(D_{\text{vac}}^{-1} S + \omega_p^2 I\right).$$
Since $\lambda_{\max}(D_{\text{vac}}^{-1} S) = (2 / \Delta t_{\text{vac}})^2$, we obtain $\lambda_{\max}(\mathbb{M}^{-1}\mathbb{K}) = \frac{1}{\varepsilon_\infty}\left(\frac{4}{\Delta t_{\text{vac}}^2} + \omega_p^2\right)$. The stability condition $\frac{\Delta t^2}{4}\lambda_{\max}(\mathbb{M}^{-1}\mathbb{K}) < 1$ is therefore
$$\left(\frac{\Delta t}{\Delta t_{\text{vac}}}\right)^2 + \left(\frac{\omega_p \Delta t}{2}\right)^2 < \varepsilon_\infty,$$
which is the exact Courant–plasma stability relation documented in `DispersiveMaterial.is_stable` (`test_d4_lanczos_dt_max_vs_dense_and_drude_identity`: relative agreement $1.850 \times 10^{-16}$).

### 1.4 Damped Poles ($\gamma > 0$)

For damped poles ($\gamma_{k,p} > 0$), the centred difference $\gamma_{k,p}\frac{p_{k,p}^{n+1} - p_{k,p}^{n-1}}{2\Delta t}$ introduces a first-order dissipative term. In the continuous-time limit, multiplying the pole equation by $\frac{\omega_{0,k,p}^2}{s_{k,p}} M_k \dot{p}_{k,p}$ shows that the total electromagnetic-plus-polarization energy decays at rate
$$\frac{\mathrm{d}E}{\mathrm{d}t} = -\sum_{k,p} \frac{\gamma_{k,p}}{s_{k,p}} \dot{p}_{k,p}^T M_k \dot{p}_{k,p} \le 0.$$
In the discrete-time scheme, for a single homogeneous pole (where $D^{-1} S$ and $M_k = D / \varepsilon_\infty$ commute and share a common eigenbasis with vacuum eigenvalue $\lambda \ge 0$), substituting $(e^n, p^n) = (\hat{e}, \hat{p}) z^n$ into the coupled difference equations and eliminating $b^{n+1/2}$ yields the scalar $4 \times 4$ quartic characteristic polynomial
$$P(z) = (z^2 - 2z + 1)\left[\varepsilon_\infty (z^2 - \alpha z - \xi) + \zeta z\right] + \lambda\Delta t^2 z (z^2 - \alpha z - \xi) = 0.$$
For spatially inhomogeneous multi-pole subgrid media where $D^{-1} S$ and the diagonal pole masks $M_k$ do not commute, simultaneous diagonalization does not apply; stability under $\gamma_{k,p} > 0$ at $\Delta t \le \Delta t_{\max}(\gamma = 0)$ is verified numerically in `test_d3_lossless_and_damped_stability_and_energy`: on a patched mesh containing damped Lorentz and Drude regions stepped at $\Delta t = 0.99\,\Delta t_{\max}(\gamma = 0)$, the dense one-step companion matrix has measured spectral radius `rho_damped = 1.000000000000` ($\le 1 + 10^{-12}$).

### 1.5 Homogeneous Lorentz Cavity Mode Identity

When a single lossless Lorentz material ($\varepsilon_\infty$, pole frequency $\omega_0$, strength $s = \Delta\varepsilon\,\omega_0^2$) fills the entire active mesh, $D = \varepsilon_\infty D_{\text{vac}}$ and $M = D_{\text{vac}}$. Consequently, each vacuum eigenmode $v_{\text{vac}}$ satisfying $D_{\text{vac}}^{-1} S v_{\text{vac}} = \lambda_{\text{vac}} v_{\text{vac}}$ decouples completely from all other spatial modes. Writing $e^n = \hat{e}^n v_{\text{vac}}$ and $p^n = \hat{p}^n v_{\text{vac}}$, the semi-discrete equations $\ddot{\hat{e}} + \ddot{\hat{p}}/\varepsilon_\infty = -(\lambda_{\text{vac}}/\varepsilon_\infty)\hat{e}$ and $\ddot{\hat{p}} = s \hat{e} - \omega_0^2 \hat{p}$ reduce to the $2 \times 2$ eigenvalue problem
$$A_2 \begin{bmatrix} \hat{e} \\ \hat{p} \end{bmatrix} = \Lambda \begin{bmatrix} \hat{e} \\ \hat{p} \end{bmatrix}, \qquad A_2 = \begin{bmatrix} (\lambda_{\text{vac}} + s)/\varepsilon_\infty & -\omega_0^2 / \varepsilon_\infty \\ -s & \omega_0^2 \end{bmatrix}.$$
The two positive eigenvalues $\Lambda_\pm$ of $A_2$ are the lower and upper polariton semi-discrete squared angular frequencies, and the exact time-discrete frequencies of the leapfrog scheme are
$$\omega_\pm = \frac{2}{\Delta t} \arcsin\!\left(\frac{\Delta t}{2}\sqrt{\Lambda_\pm}\right).$$
Equivalently, eliminating $\hat{p} = \frac{s}{\omega_0^2 - \hat{\omega}^2}\hat{e}$ with $\hat{\omega} = \frac{2\sin(\omega\Delta t / 2)}{\Delta t}$ yields the discrete dispersion relation
$$\left(\frac{2\sin(\omega\Delta t / 2)}{\Delta t}\right)^2 \varepsilon_{\text{discrete}}(\omega, \Delta t) = \lambda_{\text{vac}}, \qquad \varepsilon_{\text{discrete}}(\omega, \Delta t) = \varepsilon_\infty + \frac{s}{\omega_0^2 - \hat{\omega}^2},$$
where $\varepsilon_{\text{discrete}}$ is evaluated by `DispersiveMaterial.epsilon_discrete`. In `test_d2_homogeneous_lorentz_cavity_mode_and_perturbation`, the two analytical paths agree to $1.028 \times 10^{-15}$ relative difference and match the time-domain Hann-DFT probe peak to $4.836 \times 10^{-7}$.

---

## §2 Conformal (Subpixel) Permittivity Inside Patches

### 2.1 Dual-Volume Quarter Averages and Anisotropic Projection

On each block `Block3D` of `SubgridMesh3D`, every primal cell $c = (i, j, k)$ contributes 12 `(cell, edge)` quarter entries to `Block3D.cell_edges()`. For an edge $e$ oriented along axis $a \in \{0, 1, 2\}$ with transverse axes $b, c$, its dual volume $V_e$ inside the block is the union of the active quarter bricks $Q \in \mathcal{Q}(e)$ surrounding $e$: each quarter brick $Q$ spans the full primal cell length $h_a$ along axis $a$ and the half-cell widths $h_b / 2$, $h_c / 2$ from the edge toward the cell centre along axes $b, c$, with physical volume $V_Q = \frac{1}{4} h_a h_b h_c$ and FIT geometric weight $w_Q = \frac{h_b h_c}{4 h_a} = \frac{V_Q}{h_a^2}$. Consistent with `SubgridFIT3D._assemble`, only active leaf cells contribute to $\mathcal{Q}(e)$, so hanging edges on coarse/fine interfaces integrate over their fine-block quarters only.

For `averaging="subpixel"`:
1. Within each active quarter brick $Q \in \mathcal{Q}(e)$, the local mean permittivity $\varepsilon_Q = V_Q^{-1}\int_Q \varepsilon(\mathbf{r})\,\mathrm{d}V$ and mean inverse permittivity $\langle \varepsilon^{-1}\rangle_Q = V_Q^{-1}\int_Q \varepsilon(\mathbf{r})^{-1}\,\mathrm{d}V$ are computed either:
   - **Analytically** from exact 3D box-overlap volume fractions when `eps_r` is a `Box` or sequence of `Box` rules (eliminating quadrature sampling noise in convergence studies), or
   - **By tensor-product midpoint quadrature** with $n_{\text{sub}} \times n_{\text{sub}} \times n_{\text{sub}}$ sub-samples per quarter brick when `eps_r` is a callable $\varepsilon(X, Y, Z)$.
2. Weighting each active quarter by its physical volume $V_Q$, the dual-volume arithmetic and harmonic means on edge $e$ are
   $$\langle \varepsilon \rangle_e = \frac{\sum_{Q \in \mathcal{Q}(e)} V_Q \varepsilon_Q}{\sum_{Q \in \mathcal{Q}(e)} V_Q}, \qquad \langle \varepsilon^{-1} \rangle_e = \frac{\sum_{Q \in \mathcal{Q}(e)} V_Q \langle \varepsilon^{-1} \rangle_Q}{\sum_{Q \in \mathcal{Q}(e)} V_Q}.$$
3. An interface unit normal $\hat{\mathbf{n}}_e = (n_0, n_1, n_2)$ is obtained either from an optional user callable `normal(X, Y, Z)` evaluated at the edge midpoint or from the normalized dual-volume permittivity dipole moment $\mathbf{g}_e$ (`_subpixel_block_edge_mass` in `wakis/subgrid3d/media.py`). Following Kottke, Farjadpour & Johnson (2008) and Farjadpour et al. (2006), the inverse effective permittivity along edge direction $a$ is projected as
   $$\varepsilon_{\text{eff},e}^{-1} = n_a^2 \langle \varepsilon^{-1} \rangle_e + (1 - n_a^2) \langle \varepsilon \rangle_e^{-1},$$
   and the block edge mass is
   $$m_e = \varepsilon_0\,\varepsilon_{\text{eff},e} \sum_{Q \in \mathcal{Q}(e)} w_Q,$$
   which enters the unchanged composite row-sum lumping $D = |P|^T (m_{\text{all}} \circ |P|\mathbf{1})$.

### 2.2 Consistency Properties

1. **Exact reduction to staircase on node-plane interfaces:** Suppose all material interfaces lie strictly on grid node planes. Then no quarter brick $Q$ is cut by an interface ($\varepsilon(\mathbf{r}) \equiv \varepsilon_c$ is constant inside every cell $c$ and hence inside every $Q$).
   - If all active quarters $Q \in \mathcal{Q}(e)$ have the same permittivity $\varepsilon_c$, then $\langle \varepsilon \rangle_e = \varepsilon_c$ and $\langle \varepsilon^{-1} \rangle_e = \varepsilon_c^{-1}$, so $\varepsilon_{\text{eff},e} = \varepsilon_c$ for any $\hat{\mathbf{n}}_e$.
   - If $\mathcal{Q}(e)$ straddles a node-plane interface, that node plane necessarily contains the edge $e$, so the interface normal $\hat{\mathbf{n}}_e$ is perpendicular to the edge direction $\hat{\mathbf{e}}_a$, i.e. $n_a = \hat{\mathbf{n}}_e \cdot \hat{\mathbf{e}}_a = 0$. Consequently $\varepsilon_{\text{eff},e} = \langle \varepsilon \rangle_e$. Because every quarter $Q \in \mathcal{Q}(e)$ of an edge $e$ shares the same edge length $h_a$, its volume is $V_Q = h_a^2 w_Q$, so
     $$m_e = \varepsilon_0 \frac{\sum_Q V_Q \varepsilon_Q}{\sum_Q V_Q} \sum_Q w_Q = \varepsilon_0 \frac{\sum_Q h_a^2 w_Q \varepsilon_c}{\sum_Q h_a^2 w_Q} \sum_Q w_Q = \varepsilon_0 \sum_Q \varepsilon_c w_Q = m_e^{\text{staircase}},$$
     even on graded grids (`test_c1_subpixel_node_planes_control`: relative difference $4.441 \times 10^{-16}$ for both `Box` and callable inputs; replacing volume weights $V_Q$ by sample counts `mutate="count_weights"` breaks graded-grid consistency by $3.103 \times 10^{-2}$).
2. **Exact 1D effective medium for axis-normal planar layers:** For a planar interface $z = z_s$ with unit normal $\hat{\mathbf{n}} = \hat{\mathbf{z}}$, tangential edges ($a \in \{x, y\}$) have $n_a = 0$ and receive the exact dual-volume arithmetic mean $\varepsilon_{\text{eff},e} = \langle \varepsilon \rangle_e$, whereas normal edges ($a = z$) have $n_z^2 = 1$ and receive the exact dual-volume harmonic mean $\varepsilon_{\text{eff},e} = \langle \varepsilon^{-1} \rangle_e^{-1}$. This matches the exact 1D homogenization of tangential $E_x, E_y$ and normal $D_z$, restoring $\mathcal{O}(H^2)$ eigenvalue convergence across sub-cell cut positions (`test_c2_subpixel_slab_convergence_slow`: least-squares order $2.032$, whereas swapping $n_a^2 \leftrightarrow 1 - n_a^2$ via `mutate="swap_projection"` degrades the order to $1.466$).
3. **Preservation of energy, stability, and adjoint:** Because subpixel averaging only modifies the positive diagonal entries of $m_{\text{all}}$ (and hence $D = |P|^T (m_{\text{all}} \circ |P|\mathbf{1}) > 0$) while leaving $K$, $W$, and the time stepper untouched, the discrete energy $\frac{1}{2} e^T D e + \frac{1}{2} b^T W (I - \frac{\Delta t^2}{4} K D^{-1} K^H W) b$, the Lanczos stability limit, and the exact transpose `step_vec_T` hold without modification.

---

## Verification Summary

All numbers below are produced directly by `tests/test_120_subgrid3d_media.py` (`NUMBERS` lines) and `studies/subgrid3d/media/run_media.py`.

| check | measured | tolerance (origin) | mutation $\to$ result |
|---|---|---|---|
| **D1** (`test_d1_homogeneous_lorentz_mass_identity`): $\max \|M_k \varepsilon_\infty / D - 1\|$ on 2:1 patched cavity | `4.441e-16` | $\le 10^{-14}$ (identical row-sum lumping, round-off) | `mutate="lump_no_rowsum"` $\to$ `3.333e-01` ($\ge 0.1$) |
| **D2** (`test_d2_homogeneous_lorentz_cavity_mode_and_perturbation`): Hann-DFT peak vs $2\times 2$-$\Lambda$ and `epsilon_discrete` root | `rel_err_lam=4.836e-07`, `rel_err_ed=4.836e-07`, `ref_diff=1.028e-15` | $\le 10^{-5}$ (DFT resolution); refs agree $\le 10^{-12}$ | `mutate="lump_no_rowsum"`: predicted shift `9.318e-03` (`if_energy_share=9.141e-02`), measured shift `8.926e-03` ($> 10^{-4}$) |
| **D3** (`test_d3_lossless_and_damped_stability_and_energy`): 4000-step relative energy drift at $0.99\,\Delta t_{\max}$ on nested 2:1+3:1 graded mesh with Lorentz + Drude; damped spectral radius | `rel_drift=1.025e-14` (bound `7.022e-13`), `rho_damped=1.000000000000` | $\le 50 \varepsilon_{\text{mach}}\sqrt{n_{\text{steps}}}$; $\rho_{\text{damped}} \le 1 + 10^{-12}$ | $\Delta t = 1.02\,\Delta t_{\max} \to$ growth `1.015e+06` ($> 10^3$); `mutate="strength"` $\to$ drift `1.354e-05` (`1.929e+07` $\times$ bound) |
| **D4** (`test_d4_lanczos_dt_max_vs_dense_and_drude_identity`): Lanczos `dispersive_dt_max` vs dense `eigvalsh` ($n_{\text{free}} = 459$, $n_{\text{pencil}} = 644 > \mathrm{ncv} = 40$) and homogeneous Drude identity | `rel_err_dense=1.328e-15`, `rel_err_drude=2.068e-16`, `rel_err_identity=1.850e-16` | $\le 10^{-10}$ (Lanczos tolerance) | High-frequency `dt_max` (ignoring Drude pole) $\to$ `mut_nopole=1.612e-02` ($\ge 10^{-3}$) |
| **D5** (`test_d5_step_vec_transpose_dot_product`): `<y, M x> == <M^T y, x>` with ADE (no PML and with CPML) | `err_nopml=5.951e-18`, `err_pml=6.519e-18` | $\le 10^{-15}$ (machine precision) | `mutate="ade_transpose_no_feedback"` $\to$ `2.184e-03` ($\ge 10^{-8}$) |
| **D6** (`test_d6_slab_resonances_fast` / `_slow`): Lorentz and lossless Drude slab crossing a 2:1 patch ($N_z \in \{9, 18, 27\}$), 3 lowest fundamental modes (`TE_10, TE_01, TM_11`; `vac_1d_err=2.322e-14`) | Lorentz vs 1D orders `2.015, 2.004`, vs fine `2.025, 2.007`; Drude vs 1D orders `1.981, 1.993`, vs fine `1.986, 1.994` | order $\in [1.8, 2.2]$ (second-order interface + bulk accuracy) | `mutate="lump_no_rowsum"` $\to$ Lorentz orders `1.262, 1.177` vs 1D (`1.185, 1.122` vs fine, $< 1.35$), Drude orders `-0.667, 0.007` ($< 1.35$) |
| **C1** (`test_c1_subpixel_node_planes_control`): dielectric boxes on node planes across graded nested 2:1+3:1 mesh | `rel_err_box=4.441e-16`, `rel_err_callable=4.441e-16` | $\le 10^{-15}$ (machine round-off) | `mutate="count_weights"` $\to$ `3.103e-02` ($\ge 10^{-3}$) |
| **C2** (`test_c2_subpixel_slab_convergence_fast` / `_slow`): $\varepsilon=4$ slab at $z_s = 0.361 c$ (non-node-plane) crossing 2:1 patch, $N_z \in \{9, 13, 17, 21, 27\}$ | `err_sub=1.082e-02, 4.553e-03, 2.751e-03, 2.140e-03, 1.055e-03` $<$ `err_st=2.361e-02, 1.943e-02, 6.695e-03, 1.577e-02, 5.494e-03`; pairwise orders `2.354, 1.879, 1.188, 2.813`; LS order `2.166` (fast), `2.032` (all 5) | LS order $\ge 1.7$, `err_sub < err_st` at every $N_z$ | `mutate="swap_projection"` $\to$ LS order `0.358` (fast), `1.466` (all 5) ($< 1.5$) |
| **Study (i–iii)** (`studies/subgrid3d/media/run_media.py`, `studies/subgrid3d/media/media.json`): slab convergence (`slab.d6_dispersive`, `slab.c2_subpixel`), localized dispersive cube work ratio (`dispersive`, $N_x \in \{5, 10, 15\}$), and $\varepsilon_r=4$ sphere convergence (`sphere`, $N_x \in \{6, 8, 10, 12, 16, 20\}$) | quick run: `loc_lor_patch_err=1.384e-02`, `loc_lor_work_ratio=4.45`, `sphere_st_err=1.287e-02`, `sphere_sub4_err=4.087e-03`; full run (`media.json`): D6 LS orders `ls_ord_lor_1d=2.011`, `ls_ord_lor_fine=2.019`, `ls_ord_dru_1d=1.985`, `ls_ord_dru_fine=1.989`; C2 LS orders `ls_sub=2.032` vs `ls_st=1.187` and `ls_mut=1.466`; localized Lorentz cube ($N_x=15$, $h=3.333\text{ cm}$): `patch_r2` error `7.472e-03` (`n_state=18732`, `work_per_ns=2.998e+05`) vs `uniform_fine` error `2.188e-03` (`n_state=79500`, `work_per_ns=1.298e+06`, work ratio `4.33`, wall-per-ns ratio `3.53`); $\varepsilon_r=4$ sphere at $N_x=20$ ($h=2.500\text{ cm}$, `n_free=36958`): `staircase` mean error `5.869e-03` (LS order `1.267`), `subpixel` (`n_sub=2, 4, 6`, moment normal) `3.265e-03, 2.588e-03, 2.535e-03` (LS orders `1.614, 1.821, 1.834`), `subpixel` (`n_sub=4`, exact normal) `2.388e-03` (LS order `1.818`) | — | — |

Figures generated from `studies/subgrid3d/media/media.json` by `studies/subgrid3d/media/plot_media.py` (catalogued in `docs/figures.md`):
- [`docs/img/subgrid3d_media_slab_convergence.png`](../img/subgrid3d_media_slab_convergence.png): (a) $(x, z)$ setup schematic of a dispersive/dielectric slab crossing a 2:1 patch; (b) D6 Lorentz and lossless Drude slab convergence vs fine axial step $h_z$ against the 1D shooting reference and uniform fine `SubgridFIT3D` (`== SolverFIT3D`); (c) C2 Kottke subpixel vs staircase and `swap_projection` convergence for the non-node-plane $\varepsilon_r = 4$ slab ($z_s = 0.361\,c$).
- [`docs/img/subgrid3d_media_dispersive_cube.png`](../img/subgrid3d_media_dispersive_cube.png): localized ($3.3\%$ box volume) vs exterior-dominated ($20\%$ box volume, patch boundary coincident with dielectric jump) Lorentz and Drude 3D cubes inside a 2:1 patch vs uniform fine `SubgridFIT3D` (`== SolverFIT3D`) and uniform coarse `SubgridFIT3D`.
- [`docs/img/subgrid3d_media_subpixel_sphere.png`](../img/subgrid3d_media_subpixel_sphere.png): staircase vs Kottke subpixel ($n_{\text{sub}} \in \{2, 4, 6\}$, moment normal vs exact radial normal) resonance convergence for a dielectric sphere ($\varepsilon_r = 4.0$, $R = 0.22\text{ m}$) inside a 2:1 patch.

---

## Limits & Failure Modes

- **Debye, magnetic, conductive, nonlinear, and gain poles in patches:** `wakis.subgrid3d.media` supports electric Drude and Lorentz poles (`Lorentz` and `Drude` in `wakis.dispersive`). Passing Debye poles (`debye_poles`), dispersive or non-unit magnetic permeability (`has_magnetic`), static conductivity (`sigma != 0`), gain, or Kerr nonlinearity raises `NotImplementedError`.
- **LTS and GPU with dispersive media:** Combining `lts=True` (`SubgridLTSFIT3D` with local time-stepping subcycles) or `backend="cupy"` with a non-empty `dispersive` specification raises `NotImplementedError`.
- **Non-dispersive helper methods on dispersive solvers:** Calling `SubgridFIT3D.linear_step`, `SubgridFIT3D.linear_step_T`, `SubgridFIT3D.energy`, or `SubgridFIT3D.eigenfrequencies` when `dispersive_groups` is non-empty raises `NotImplementedError` (use `SubgridLTSFIT3D.step_vec` / `step_vec_T`, `dispersive_energy`, and `dispersive_eigenfrequencies` instead).
- **Off-diagonal inverse-permittivity tensor entries:** Subpixel averaging in `wakis.subgrid3d.media` retains only the diagonal Kottke projection $\varepsilon_{\text{eff},e}^{-1}$ so that the composite mass matrix $D$ remains strictly diagonal. Full off-diagonal inverse-permittivity coupling (`wakis/tensor_averaging.py`) is implemented only on uniform/non-subgridded `SolverFIT3D` grids.
- **Per-cell permeability $\mu_r$:** Subpixel conformal averaging applies to the electric permittivity $\varepsilon_r$; the magnetic permeability $\mu_r$ in the face reluctance $W$ remains cell-wise staircase averaged.
- **Dispersive pole mass averaging:** For dispersive materials, the pole mass $m_{p,k}$ uses the cell/quarter volume fraction on every edge (arithmetic volume weighting) rather than an orientation-dependent harmonic/arithmetic Kottke projection.
- **Over-patched cavities:** When a global-$\Delta t$ subgrid patch covers a large fraction of the domain or when the resonant mode's wavelength and error are dominated by the coarse exterior rather than the refined inclusion (Panel (c) of `subgrid3d_media_dispersive_cube.png`), the global time step remains pinned to the fine patch stability limit while the spatial error is set by the coarse step $H$, so subgridding does not reduce total work at equal accuracy.

---

## References

- C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for anisotropic dielectric interfaces, and application to subpixel smoothing of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008), doi:[10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
- A. Farjadpour, D. Roundy, A. Rodriguez, M. Ibanescu, P. Bermel, J. D. Joannopoulos, S. G. Johnson, and G. W. Burr, "Improving accuracy by subpixel smoothing in the finite-difference time domain," *Opt. Lett.* **31**(20), 2972–2974 (2006), doi:[10.1364/OL.31.002972](https://doi.org/10.1364/OL.31.002972).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Norwood, MA (2005), Chapter 9 (Dispersive Media / Auxiliary Differential Equations).
