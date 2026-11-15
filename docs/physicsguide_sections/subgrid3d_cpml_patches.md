# 3D Subgridding — CFS-CPML with Patches in the Layer

This section documents the spectral diagnosis of CFS-CPML (`wakis.subgrid3d.pml.SubgridCPML`) when refined subgrid patches reach into or through the absorbing layer, the derivation and testing of Hypothesis C1 (`sigma_sampling="coarse_cell"` and `"coarse_cell_dual"`), and the formulation of Candidate C2 (uniaxial PML in Hodge-mass form) required to eliminate the remaining high-frequency cutoff-mode growth in the deepest backing cells.

---

## 1. Semi-Discrete Generator $G$ and Spectral Diagnosis

### 1.1 Continuous-Time Limit of the Recursive-Convolution Step

In `SubgridCPML`, the directional curl decomposition is $K = \sum_{d=0}^2 K^{(d)}$ with $K^{(d)} = C^{(d)} P_e$, where every nonzero entry of the primal face-edge incidence $C$ with face normal $a$ and edge direction $b$ belongs to derivative axis $d = 3 - a - b$. On active faces $\mathrm{pf}_d$ and active free edges $\mathrm{pe}_d$ in the layer along axis $d$, one time step $\Delta t$ updates the auxiliary variables $\psi_d \in \mathbb{R}^{|\mathrm{pf}_d|}$ and $\phi_d \in \mathbb{R}^{|\mathrm{pe}_d|}$ by

$$
\begin{aligned}
\psi_d^{n+1/2} &= b_{h,d}\,\psi_d^{n-1/2} + a_{h,d}\,K_f^{(d)} z^n, \\
b^{n+1/2} &= b^{n-1/2} - \Delta t \left(K z^n + \sum_{d=0}^2 I_{\mathrm{pf},d}\,\psi_d^{n+1/2}\right), \\
\phi_d^{n+1} &= b_{e,d}\,\phi_d^n + a_{e,d}\,K_e^{(d)} W b^{n+1/2}, \\
e^{n+1} &= e^n + \Delta t\,D^{-1} \left(K^H W b^{n+1/2} + \sum_{d=0}^2 I_{\mathrm{pe},d}\,\phi_d^{n+1} - P^H i^{n+1/2}\right),
\end{aligned}
$$

where $z^n = R_0 e^n$ under LTS ($z^n = e^n$ under a global step), $I_{\mathrm{pf},d}$ and $I_{\mathrm{pe},d}$ inject the active layer rows into the full face and free-edge vectors, and the recursive-convolution coefficients are

$$
b_{x,d} = \exp\!\left(-(\sigma_{x,d} + \alpha)\,\Delta t\right), \qquad
a_{x,d} = \frac{\sigma_{x,d}}{\sigma_{x,d} + \alpha}\,\left(b_{x,d} - 1\right), \qquad x \in \{h, e\}.
$$

Expanding for $\Delta t \to 0$ gives $b_{x,d} = 1 - (\sigma_{x,d} + \alpha)\Delta t + \mathcal{O}(\Delta t^2)$ and $a_{x,d} = -\sigma_{x,d}\Delta t + \mathcal{O}(\Delta t^2)$. Dividing the state differences by $\Delta t$ and taking $\Delta t \to 0$ yields the linear semi-discrete autonomous system $\dot{y} = G y$ on the state vector $y = [b;\, e;\, \psi_0;\, \psi_1;\, \psi_2;\, \phi_0;\, \phi_1;\, \phi_2]$ (`cpml_generator(solver)` in `wakis/subgrid3d/pml.py`):

$$
\begin{aligned}
\dot{b} &= -K R_0 e - \sum_{d=0}^2 I_{\mathrm{pf},d}\,\psi_d, \\
\dot{e} &= D^{-1} \left(K^H W b + \sum_{d=0}^2 I_{\mathrm{pe},d}\,\phi_d\right), \\
\dot{\psi}_d &= -\left(\operatorname{diag}(\sigma_{f,d}) + \alpha I\right)\psi_d - \operatorname{diag}(\sigma_{f,d})\,K_f^{(d)} R_0 e, \\
\dot{\phi}_d &= -\left(\operatorname{diag}(\sigma_{e,d}) + \alpha I\right)\phi_d - \operatorname{diag}(\sigma_{e,d})\,K_e^{(d)} W b.
\end{aligned}
$$

For an eigenmode $y(t) = \hat{y} e^{\lambda t}$ with $\lambda \notin -(\sigma + \alpha)$, eliminating $\hat{\psi}_d = -\frac{\sigma_{f,d}}{\lambda + \sigma_{f,d} + \alpha} K_f^{(d)} R_0 \hat{e}$ and $\hat{\phi}_d = -\frac{\sigma_{e,d}}{\lambda + \sigma_{e,d} + \alpha} K_e^{(d)} W \hat{b}$ recovers the frequency-domain stretched curl equations with diagonal stretch factors $s_{f,d}(\lambda) = 1 + \frac{\sigma_{f,d}}{\lambda + \alpha}$ and $s_{e,d}(\lambda) = 1 + \frac{\sigma_{e,d}}{\lambda + \alpha}$:

$$
\lambda \hat{b} = -\left(\sum_{d=0}^2 S_{f,d}(\lambda)^{-1} K^{(d)}\right) R_0 \hat{e}, \qquad
\lambda D \hat{e} = \left(\sum_{d=0}^2 S_{e,d}(\lambda)^{-1} K^{(d)H}\right) W \hat{b}.
$$

 Comparing the finite-difference quotient $(M(\Delta t) - I)v / \Delta t$ of `step_vec` at $\Delta t = 10^{-15}\text{ s}$ against $G v$ on a patched mesh confirms convergence with relative error `1.031e-06` (`test_cpml_generator_diagnosis_and_nopatch_control`).

### 1.2 Spectral Diagnosis and Unstable Eigenvector Localisation

On an $8 \times 5 \times 5$ coarse mesh ($H = 0.04\text{ m}$) with a 4-cell $+x$ CPML layer ($R_0 = 10^{-6}$, polynomial order $m = 3$, $\sigma_{\max} = 5.177 \times 10^{10}\text{ s}^{-1}$) at $\alpha / (2\pi) = 10\text{ MHz}$ (`test_cpml_generator_diagnosis_and_nopatch_control`):

1. **No-patch control** ($n_{\text{state}} = 1393$):
   $$\frac{\max \operatorname{Re}\lambda(G)}{\sigma_{\max}} = 5.009 \times 10^{-15} \le 10^{-7} \quad (\text{and } 7.095 \times 10^{-15} \text{ at } \alpha = 0).$$
2. **Present fine sampling (`sigma_sampling="fine"`) with a 2:1 patch `(3,1,1)..(7,4,4)`** ($n_{\text{state}} = 3626$):
   $$\max \operatorname{Re}\lambda(G) = 2.818 \times 10^8\text{ s}^{-1}, \qquad \frac{\max \operatorname{Re}\lambda(G)}{\sigma_{\max}} = 5.443 \times 10^{-3} \quad (6.105 \times 10^{-3} \text{ at } \alpha = 0),$$
   with oscillation frequency $|\operatorname{Im}\lambda| / (2\pi) = 8.765 \times 10^7\text{ Hz}$.
3. **Eigenvector localisation (`diagnose_cpml_eigenvector`)**:
   - **Per block**: `32.09%` (`0.3209`) of the electromagnetic energy $\frac{1}{2}(\|b\|_W^2 + \|e\|_D^2)$ resides in the coarse root block 0 and `67.91%` (`0.6791`) resides in the 2:1 patch block 1.
   - **On the coarse-to-fine interface**: **`94.08%`** (`0.9408`) of the magnetic face energy $\|b\|_W^2$ sits on faces touching interface-parent free edges, and **`62.20%`** (`0.6220`) of the electric edge energy $\|e\|_D^2$ sits directly on interface-parent free edges.
   - **By normalized depth $u \in [0, 1]$ in the layer**: `55.7%` (`0.557`) of the energy lies in the first depth quartile $u \in (0, 0.25]$ (first coarse PML cell), `40.4%` (`0.404`) in the second quartile $u \in (0.25, 0.50]$, `0.1%` (`0.001`) in $u \in (0.50, 0.75]$, and `0.0%` (`0.000`) in $u \in (0.75, 1.00]$.

---

## 2. Hypothesis C1: Stretched Nédélec Trace and Coarse-Cell Sampling

### 2.1 Derivation of the Stretched Trace Mismatch

In `SubgridMesh3D`, the edge prolongation matrix $P_e$ maps coarse parent edges on patch boundary faces to hanging fine edges using the unstretched Nédélec face trace with fixed real interpolation weights $t_j = j/r$ ($j = 0, \dots, r$) and collinear segment fraction $1/r$.

Under complex coordinate stretching $\tilde{x}_d(x_d) = \int^{x_d} s_d(\xi)\,d\xi$ along axis $d$ [Chew & Weedon 1994], the constant tangential field of the lowest-order Nédélec space in stretched coordinates $\tilde{x}_d$ interpolates linearly in $\tilde{x}_d$ across a coarse parent cell $[x_k, x_{k+1}]$:

$$
\tilde{t}_j = \frac{\tilde{x}_{k+j/r} - \tilde{x}_k}{\tilde{x}_{k+1} - \tilde{x}_k}
= \frac{\int_{x_k}^{x_{k+j/r}} s_d(\xi)\,d\xi}{\int_{x_k}^{x_{k+1}} s_d(\xi)\,d\xi},
\qquad
\frac{\tilde{L}_{\text{fine},j}}{\tilde{L}_{\text{coarse},k}}
= \frac{\int_{x_{k+j/r}}^{x_{k+(j+1)/r}} s_d(\xi)\,d\xi}{\int_{x_k}^{x_{k+1}} s_d(\xi)\,d\xi}.
$$

These stretched weights equal the fixed real weights $t_j = j/r$ and $1/r$ hard-coded in $P_e$ **if and only if** $s_d(\xi)$ (and hence $\sigma_d(\xi)$) is constant over each coarse parent cell $[x_k, x_{k+1}]$.

Specifically, consider a hanging edge $e_{\text{hang}}(x_{k+j/r})$ on a patch wall parallel to axis $d$, interpolated between coarse parent edges $E_k$ at $x_k$ and $E_{k+1}$ at $x_{k+1}$:

$$
e_{\text{hang}}(x_{k+j/r}) = \left(1 - \frac{j}{r}\right) E_k + \frac{j}{r} E_{k+1}.
$$

Across the $j$-th fine sub-slab $[x_{k+j/r}, x_{k+(j+1)/r}]$ of coarse cell $k$, the primal difference in $K^{(d)} = C^{(d)} P_e$ is

$$
(C^{(d)} P_e e)_{f_j} = e_{\text{hang}}(x_{k+(j+1)/r}) - e_{\text{hang}}(x_{k+j/r}) = \frac{1}{r}\left(E_{k+1} - E_k\right).
$$

When $\sigma_{f,d}$ is sampled at individual fine face centres $x_{k+(j+1/2)/r}$ (`sigma_sampling="fine"`), the $r$ fine sub-slabs of coarse cell $k$ multiply the same coarse difference $E_{k+1} - E_k$ by $r$ different complex stretch factors $s_{f,d}(x_{k+(j+1/2)/r})^{-1}$, whereas the adjacent coarse face in block 0 multiplies $E_{k+1} - E_k$ by $s_{f,d}(x_{k+1/2})^{-1}$. Around the coarse-to-fine interface loop, the stretched circulation fails to close by $\mathcal{O}(H\,\partial_d \sigma_d)$, creating the non-self-adjoint interface mode localised in §1.2.

### 2.2 Candidate C1(i) (`"coarse_cell"`) and C1(ii) (`"coarse_cell_dual"`)

Setting $\sigma_{f,d}$ on fine patch faces equal to $\sigma_d$ at the coarse parent cell midpoint $x_{k+1/2} = \frac{1}{2}(x_k + x_{k+1})$ (`sigma_sampling="coarse_cell"`, Candidate C1(i), or together with coarse-node snapping of $\sigma_{e,d}$ in `"coarse_cell_dual"`, Candidate C1(ii)) makes $s_{f,d}$ constant across all $r$ fine sub-slabs of each coarse cell $k$:

- On the $8 \times 5 \times 5$ mesh with a 2:1 patch reaching 2 cells into the 4-cell $+x$ layer (`(3,1,1)..(6,4,4)`), `sigma_sampling="coarse_cell"` reduces $\max \operatorname{Re}\lambda(G)/\sigma_{\max}$ by six orders of magnitude, from `5.372e-03` (`"fine"`) to **`6.351e-09`** ($\le 10^{-6}$).
- By axis symmetry on `(7, 5, 5)` / `(5, 7, 5)` / `(5, 5, 7)` with a 2:1 patch reaching into the 4-cell layer (`test_t1_cpml_patches_spectrum`), `"coarse_cell"` achieves `5.380e-09` ($x$), `6.361e-09` ($y$), and `5.622e-09` ($z$); on `(5, 3, 3)` with a 3:1 patch reaching through the layer (`x_through_3to1`) it reduces $\max \operatorname{Re}\lambda(G)/\sigma_{\max}$ from `6.405e-03` (`"fine"`) to **`1.582e-09`** ($\le 10^{-6}$); and on the two-layer $+x/+y$ corner `(5, 5, 4)` it reduces $\max \operatorname{Re}\lambda(G)/\sigma_{\max}$ from `6.901e-03` (`"fine"`) to **`4.222e-09`** ($\le 10^{-6}$).
- On a 2:1 patch tube spanning all $x$ through the layer (`(0,1,1)..(6,3,3)` on $6 \times 4 \times 4$), `"coarse_cell"` achieves **`4.308e-09`** ($\le 10^{-6}$).
- The discrete step remains linear and `step_vec_T` is the exact transpose to machine precision: relative dot-product defect `7.804e-18` (`"coarse_cell"`) and `2.373e-17` (`"coarse_cell_dual"`), vs `2.262e-06` when $a_h$ is perturbed by $10\%$ in the transpose (`test_cpml_generator_diagnosis_and_nopatch_control`).

### 2.3 Why 3D Patches Terminating in the Deepest Backing Cells Still Exceed $10^{-6}\sigma_{\max}$

When a 3D patch with side walls inside the cross-section starts in the interior and couples a coarse-to-fine cutoff step into the layer (`x_through_2to1`, `y_through_2to1`, `z_through_2to1`, `x_into_3to1`), `sigma_sampling="coarse_cell"` reduces the growth rate by more than an order of magnitude compared to `"fine"`, but $\max \operatorname{Re}\lambda(G)/\sigma_{\max}$ remains in the range `1.883e-04` to `4.111e-04` ($> 10^{-6}$), so `test_t1_cpml_patches_spectrum` is marked `@pytest.mark.xfail(strict=True)` per Part 3 step 5.

Two coupled mechanisms produce this residual $\sim 10^{-4}\sigma_{\max}$ rate at waveguide cutoff frequencies ($\omega / 2\pi \approx 1.95\text{ GHz}$):

1. **Non-commuting transverse curl $K_\perp$ on side-wall $d$-normal faces**:
   For a layer along axis $d = 0$ ($x$), decomposing edges into longitudinal $e_0$ ($x$-edges) and transverse $e_\perp$ ($y,z$-edges) and faces into $b_x$ ($x$-faces) and $b_{yz}$ ($y,z$-faces) puts the frequency-domain equations into symmetric UPML form if and only if
   $$
   K_{yz,0} S_0 = S_{f,0} K_{yz,0} \quad\text{and}\quad K_x S_{e,0} = S_x K_x.
   $$
   Under C1(i), the first relation holds because every $x$-edge and its incident $y,z$-faces lie within a single coarse $x$-cell $k$. However, on a fine $x$-face $f_x$ at an interior fine plane $x_{k+j/r}$ ($0 < j < r$) touching a patch side wall ($y = y_{\min}, y_{\max}$ or $z = z_{\min}, z_{\max}$), the transverse curl $K_x = (C^{(1)} + C^{(2)})_{f_x} P_e$ couples three free fine edges at $x_{k+j/r}$ with a hanging edge interpolated from coarse parent edges at $x_k$ and $x_{k+1}$. Because $x_k < x_{k+j/r} < x_{k+1}$, no graded diagonal edge stretch $S_{e,0}$ commutes with $K_x$.
2. **Evanescent $\text{TM}_d$ cutoff-mode shift across longitudinal cross-section changes**:
   For a $\text{TM}_x$ waveguide mode with local discrete transverse cutoff frequency $\omega_n(x)$, summing the stretched transmission-line equations by parts yields
   $$
   \lambda^2 \sum_k s_f(x_{k+1/2})\,|I_{k+1/2}|^2 + \sum_k \omega_n(x_{k+1/2})^2\,s_f(x_{k+1/2})\,|I_{k+1/2}|^2 = -\sum_k s_e(x_k)^{-1}\left|\frac{I_{k+1/2}-I_{k-1/2}}{H}\right|^2.
   $$
   Because a refined patch has smaller numerical dispersion than the coarse grid ($\omega_{n,\text{patch}} > \omega_{n,\text{coarse}}$), modes trapped at $\omega \in (\omega_{n,\text{coarse}}, \omega_{n,\text{patch}})$ are propagating in the unpatched region and evanescent inside the patched PML, where $(\omega_{n,\text{pml}}^2 - \omega_{n,\text{int}}^2)\operatorname{Im}(-s_f) > 0$ drives a positive imaginary part in $\lambda^2$ whenever $s_f$ and $s_e$ are staggered along $x$.

---

## 3. Next Step: Candidate C2 (Uniaxial PML in Hodge Form)

To satisfy both commutation conditions identically on an arbitrary hierarchical mesh, the complex coordinate stretch must be placed in the diagonal Hodge material masses rather than on the split topological curl [Gedney 1996]:

1. Keep the topological curl $K = C P_e$ and its adjoint $K^H = P_e^T C^T$ **unstretched and unsplit**, so that $\begin{pmatrix} 0 & -K \\ K^H & 0 \end{pmatrix}$ is skew-adjoint in the unstretched pairing on every patch interface.
2. Associate the uniaxial stretch tensor
   $$
   \bar{\bar{s}}(\omega) = \operatorname{diag}\!\left(\frac{s_y s_z}{s_x},\; \frac{s_z s_x}{s_y},\; \frac{s_x s_y}{s_z}\right)
   $$
   with the cell-level edge and face weights $m_e, m_f$, and lump the stretched edge mass with the exact row-sum rule of `SubgridFIT3D`:
   $$
   D(\omega) = |P_e|^T \left(m_e(\omega) \circ |P_e|\mathbf{1}\right).
   $$
3. In the time domain, each directional factor $s_a(\omega) s_b(\omega) / s_c(\omega)$ is stepped as a two-stage local auxiliary differential equation on the mass entries $D_i$ and $W_j$ rather than on $K^{(d)} e$ and $K^{(d)H} W b$. Because the prolongation $P_e$ then acts before the positive-real Hodge mass lumping on both $e$ and $b$, the coarse-to-fine interface introduces no commutator defect in $K$ or $K^H$.

---

## 4. Verification Summary

All measured numbers below come directly from the `NUMBERS` lines printed by `pytest tests/test_121_subgrid3d_cpml_patches.py -s` and `studies/subgrid3d/cpml_patches/run_cpml_patches.py --quick`.

| Check | Measured | Tolerance (Origin) | Mutation $\to$ Result |
| :--- | :--- | :--- | :--- |
| Generator vs `(step_vec - I)/dt` ($\Delta t = 10^{-15}\text{ s}$) | `fd_err = 1.031e-06` | $< 10^{-5}$ ($\mathcal{O}(\Delta t)$ leapfrog staggering) | — |
| No-patch control ($8\times 5\times 5$, $\alpha/2\pi = 10\text{ MHz}$) | `nop_re_10m = 5.009e-15` (`nop_re_0 = 7.095e-15`) | $\le 10^{-7}\,\sigma_{\max}$ (round-off of stable spectrum) | `sigma_sampling="fine"` with 2:1 patch $\to$ `fine_re_10m = 5.443e-03` ($\operatorname{Re}\lambda = 2.818\times 10^8\text{ s}^{-1}$) |
| Unstable mode localisation (`sigma_sampling="fine"`) | `if_face_b = 0.9408`, `if_edge_e = 0.6220`, `blk1_share = 0.6791`, `depth_q = 0.557,0.404,0.001,0.000` | $> 0.90$ of $E_b$ on interface-touching faces | — |
| C1 (`"coarse_cell"`) on 2:1 patch 2 cells into 4-cell layer | `into2_re_c1 = 6.351e-09` (`tube_re_c1 = 4.308e-09`) | $\le 10^{-6}\,\sigma_{\max}$ (T1 criterion) | `sigma_sampling="fine"` $\to$ `into2_re_fine = 5.372e-03` |
| `step_vec_T` dot-product with C1 (`"coarse_cell"`, `"coarse_cell_dual"`) | `dot_c1 = 7.804e-18`, `dot_dual = 2.373e-17` | $\le 10^{-15}$ (machine precision, §9f) | Perturb $a_h \leftarrow 1.1\,a_h$ in transpose $\to$ `dot_mut = 2.262e-06` ($> 10^{-8}$) |
| T1 full matrix (`test_t1_cpml_patches_spectrum`, strict `xfail`) | `nop_10m = 8.245e-16`; `x_into_2to1 = 5.380e-09`, `y_into_2to1 = 6.361e-09`, `z_into_2to1 = 5.622e-09`, `x_through_3to1 = 1.582e-09`, `corner_xy_2to1 = 4.222e-09`; `x/y/z_through_2to1 = 4.111e-04`, `x_into_3to1 = 1.883e-04` (`max_c1_10m = 4.111e-04`, `max_c1_0 = 4.140e-04`) | $\le 10^{-6}\,\sigma_{\max}$ across all cases (strict `xfail` per Part 3 step 5) | `sigma_sampling="fine"` $\to$ `min_fine_10m = 5.262e-03`, `max_fine_10m = 7.870e-03` ($> 10^{-3}$) |
| Time-gated reflection (`run_cpml_patches.py --quick`, $n_{\text{pml}} = 6$) | `err_nopatch = 2.212e-03`, `err_near = 1.002e-03`, `err_through_c1_a0 = 7.200e-04`, `err_through_a1GHz = 2.495e-03` | Full study: `TBD(cpml_patches.json:reflection)` | $\alpha/2\pi = 1\text{ GHz}$ degrades low-frequency reflection by $3.47\times$ (`2.495e-03` vs `7.200e-04`) |

---

## 5. Limits & Failure Modes

- **Default refusal preserved (`allow_patches=False`)**: Because 3D patches coupling a coarse-to-fine cutoff step into the deepest backing cells of a graded CPML retain a residual growth rate $\max \operatorname{Re}\lambda(G) \le 4.111 \times 10^{-4}\,\sigma_{\max}$ under split-curl CFS-CPML (`sigma_sampling="coarse_cell"`), `SubgridCPML` continues to raise `ValueError` by default when a patch enters the layer (`allow_patches=False`).
- **Low-frequency reflection penalty of large CFS shifts**: Suppressing the split-curl instability via a large CFS shift ($\alpha / 2\pi = 1\text{ GHz}$) weakens the stretch $s_d(\omega) = 1 + \sigma_d / (\alpha + i\omega)$ for $\omega \lesssim \alpha$, increasing the time-gated pulse reflection error from `7.200e-04` ($\alpha = 0$, C1) to `2.495e-03` on the 6-cell quick benchmark (and `TBD(cpml_patches.json:reflection)` on the 4–12 cell full study).
- **LTS fine levels inside CPML**: Local time stepping (`lts=True`) with fine patches inside the CPML layer remains refused (`ValueError`), as the auxiliary convolutions $\psi_d, \phi_d$ are stepped only at the coarse $\Delta t$.

---

## 6. References

1. W. C. Chew and W. H. Weedon, "A 3D perfectly matched medium from modified Maxwell's equations with stretched coordinates," *Microw. Opt. Technol. Lett.*, vol. 7, no. 13, pp. 599–604, 1994, doi:10.1002/mop.4650071304.
2. S. D. Gedney, "An anisotropic perfectly matched layer-absorbing medium for the truncation of FDTD lattices," *IEEE Trans. Antennas Propag.*, vol. 44, no. 12, pp. 1630–1639, 1996, doi:10.1109/8.546249.
3. J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media," *Microw. Opt. Technol. Lett.*, vol. 27, no. 5, pp. 334–339, 2000, doi:10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A.
4. E. Bécache, S. Fauqueux, and P. Joly, "Stability of perfectly matched layers, group velocities and anisotropic waves," *J. Comput. Phys.*, vol. 188, no. 2, pp. 399–433, 2003, doi:10.1016/S0021-9991(03)00184-0.
