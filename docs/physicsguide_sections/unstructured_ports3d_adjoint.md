# Tetrahedral Ports in the Unstructured Adjoint (`wakis.unstructured.ports3d` + `wakis.unstructured.adjoint`)

Scope: exact discrete transpose (`linear_step` / `linear_step_T`) and reverse-mode material-permittivity
gradients (`UnstructuredAdjointStepper`, `UnstructuredAdjointSimulation`, `PortAdjointHelper`,
`PortDFTMonitor`) for 3D tetrahedral FIT solvers (`UnstructuredFIT3D`, Voronoi Hodge) terminated by modal
and lumped ports (`ModalPort`, `LumpedPort`, `attach_ports`). Companion to
`docs/physicsguide_sections/unstructured_ports3d.md` and `docs/physicsguide_sections/unstructured_adjoint.md`.

---

## 1. Port-Augmented Forward Step

With $P$ ports attached to an `UnstructuredFIT3D` solver, let $W = [w_1, \dots, w_P] \in \mathbb{R}^{n_{\text{free}} \times P}$
collect the port weight vectors restricted to the active (free, non-PEC, non-massless) edges `solver.free`, and let
$g = (g_1, \dots, g_P) \ge 0$ be the terminating modal or lumped conductances. Inside step $n \to n+1$, the
semi-implicit trapezoidal port current is

$$I_p^{n+1/2} = I_{s,p}^{n+1/2} - \frac{g_p}{2}\, w_p^T (u^n + u^{n+1}),$$

so the electric update on `solver.free` becomes

$$\underbrace{\left(D_+ + \frac{\Delta t}{2} W \operatorname{diag}(g) W^T\right)}_{A(\varepsilon)} u^{n+1}
= \underbrace{\left(\operatorname{diag}(a_1) - \frac{\Delta t}{2} W \operatorname{diag}(g) W^T\right)}_{B(\varepsilon)} u^n
+ \Delta t\,\bigl(C_f^T M_\nu b^n - j_f^{n+1/2} + W I_s^{n+1/2}\bigr) - \sum_g c_g\,\operatorname{corr}_g(u^n, z^n),$$

where $D_+ = \operatorname{diag}(\text{den})$ (`cu.den = C0 + G dt / 2 + sum c beta`) is the diagonal capacitive
denominator of `_Capacitive` and $a_1$ (`cu.a1 = C0 - G dt / 2 - sum c beta`) is the explicit
numerator inside `_Capacitive`. The massless edge Schur solve and Faraday update follow unchanged:

$$K_{ss} u_s^{n+1} = -K_{sf} u^{n+1}, \qquad b^{n+1} = b^n - \Delta t\,\bigl(C_f u^{n+1} + C_s u_s^{n+1}\bigr).$$

Factoring $A = D_+\bigl(I + \tfrac{\Delta t}{2} D_+^{-1} W \operatorname{diag}(g) W^T\bigr)$ splits the electric update
into the unmodified diagonal `_Capacitive.forward` step

$$u_{\text{diag}}^{n+1} = D_+^{-1}\left[B(\varepsilon) u^n + \Delta t\,\bigl(C_f^T M_\nu b^n - j_f^{n+1/2} + W I_s^{n+1/2}\bigr) - \sum_g c_g\,\operatorname{corr}_g(u^n, z^n)\right]$$

followed by the rank-$P$ Sherman–Morrison–Woodbury correction (Hager 1989)

$$u^{n+1} = A^{-1} D_+ u_{\text{diag}}^{n+1}
= u_{\text{diag}}^{n+1} - D_+^{-1} W \operatorname{diag}\!\left(\frac{\Delta t\, g}{2}\right) S_P^{-1} W^T u_{\text{diag}}^{n+1},
\qquad S_P = I_P + \operatorname{diag}\!\left(\frac{\Delta t\, g}{2}\right) W^T D_+^{-1} W \in \mathbb{R}^{P \times P}.$$

`PortAdjointHelper` (in `wakis/unstructured/ports3d.py`) caches $W$, $g$, $D_+^{-1}$, and $S_P^{-1}$ whenever
`UnstructuredAdjointStepper.set_eps(eps)` updates the cell permittivities:

1. `PortAdjointHelper.pre_step(u, r, n)` adds the port current contribution
   $W\bigl(I_s^{n+1/2} - \tfrac{1}{2} g \odot (W^T u^n)\bigr)$ to `r` before `cu.forward`.
2. `PortAdjointHelper.post_step(u_star)` applies $A^{-1} D_+$ after `cu.forward` and before the massless Schur solve.

Because the arithmetic and operation order match `PortSet.pre_e` / `PortSet.post_e`, `UnstructuredAdjointStepper.step`
reproduces `UnstructuredFIT3D.one_step` bitwise (`np.testing.assert_array_equal`).

---

## 2. Exact Discrete Transpose and Source–Probe Duality

With sources set to zero ($j_f = 0$, $I_s = 0$), one homogeneous step $x^{n+1} = L x^n$ on $x^n = (u^n, b^n, z^n)$ is

$$u_{\text{diag}}^{n+1} = D_+^{-1}\left[\operatorname{diag}(a_1) u^n - \frac{\Delta t}{2} W \operatorname{diag}(g) W^T u^n + \Delta t\, C_f^T M_\nu b^n - \sum_g c_g\,\operatorname{corr}_g(u^n, z^n)\right],$$

$$u^{n+1} = (A^{-1} D_+) u_{\text{diag}}^{n+1}, \qquad
b^{n+1} = b^n - \Delta t\, C_f u^{n+1} + \Delta t\, C_s K_{ss}^{-1} K_{sf} u^{n+1}.$$

Because $D_+$ is diagonal and $\tfrac{\Delta t}{2} W \operatorname{diag}(g) W^T$ is a symmetric rank-$P$ matrix,
$A = D_+ + \tfrac{\Delta t}{2} W \operatorname{diag}(g) W^T$ is symmetric ($A^T = A$). Therefore the transpose of the
post-multiplier $A^{-1} D_+$ is

$$(A^{-1} D_+)^T = D_+ A^{-T} = D_+ A^{-1} = I - W \operatorname{diag}\!\left(\frac{\Delta t\, g}{2}\right) S_P^{-T} W^T D_+^{-1}.$$

Given the incoming cotangent $\bar{x}^{n+1} = (\bar{u}^{n+1}, \bar{b}^{n+1}, \bar{z}^{n+1})$, the exact transpose
$L^T \bar{x}^{n+1}$ executes in reverse order:

1. **Faraday and massless Schur transpose:**
   $$\tilde{u}^{n+1} = \bar{u}^{n+1} - \Delta t\,\bigl(C_f^T \bar{b}^{n+1} - K_{sf}^T K_{ss}^{-T} C_s^T \bar{b}^{n+1}\bigr).$$
2. **Woodbury port transpose (`PortAdjointHelper.pre_step_T`):**
   $$\bar{u}^* = (A^{-1} D_+)^T \tilde{u}^{n+1} = \tilde{u}^{n+1} - W \operatorname{diag}\!\left(\frac{\Delta t\, g}{2}\right) S_P^{-T} \bigl(W^T D_+^{-1} \tilde{u}^{n+1}\bigr).$$
3. **Capacitive and ADE transpose (`_Capacitive.transpose`):**
   Passing $\bar{u}^*$ to `cu.transpose(u_star_b, xb1, xb)` computes
   $$\bar{r} = \Delta t\, D_+^{-1} \bar{u}^* = \Delta t\, A^{-T} \tilde{u}^{n+1}, \qquad
   \bar{u}_{\text{cap}}^n = a_1 \odot (D_+^{-1} \bar{u}^*) + \text{ADE transpose}(\bar{u}^*, \bar{z}^{n+1}).$$
4. **Explicit half-step port conductance transpose (`PortAdjointHelper.post_step_T`) and flux accumulation:**
   $$\bar{u}^n = \bar{u}_{\text{cap}}^n - \frac{1}{2} W \operatorname{diag}(g) W^T \bar{r}, \qquad
   \bar{b}^n = \bar{b}^{n+1} + M_\nu C_f \bar{r}.$$

### Duality of Mode Sources and Mode Probes

In the forward step, the modal source $I_s^{n+1/2}$ enters as the column vector $\Delta t\, W I_s^{n+1/2}$ on the
right-hand side of $A u^{n+1} = \dots$, whereas the port voltage probe evaluates the linear functional
$V^{n+1/2} = W^T u^{n+1/2}$:

$$\frac{\partial u^{n+1}}{\partial I_s^{n+1/2}} = \Delta t\, A^{-1} W, \qquad
\frac{\partial V^{n+1/2}}{\partial u^{n+1}} = \frac{1}{2} W^T, \qquad
\frac{\partial V^{n+1/2}}{\partial u^n} = \frac{1}{2} W^T.$$

Taking transposes swaps their roles: a cotangent $\bar{V}^{n+1/2}$ on the port voltage probe injects the source vector
$\tfrac{1}{2} W \bar{V}^{n+1/2}$ into the adjoint edge equations at levels $n$ and $n+1$, while probing the adjoint
state with $W^T \bar{r} = \Delta t\, W^T A^{-T} \tilde{u}^{n+1}$ yields the sensitivity with respect to the source
current samples $I_s^{n+1/2}$.

---

## 3. Exact Material-Permittivity Gradient Through Ports

For a Voronoi-Hodge 3D solver, the dual-cell capacitance vector on all edges is $C_0(\varepsilon) = \varepsilon_0 P_e (\varepsilon \odot \mathbb{1}_{\text{live}})$,
so only the active-edge entries $C_{0,e}$ ($e \in \text{free}$) depend on the design cell permittivities $\varepsilon_T$.
The port weight matrix $W$ is determined solely by the 2D boundary cross-section eigenmode $\phi$ (or lumped edge
path) and is independent of interior design permittivities, and the reference port conductances $g$ are frozen at the
port boundary.

In the implicit port-augmented step

$$A(\varepsilon) u^{n+1} = B(\varepsilon) u^n + \Delta t\,\bigl(C_f^T M_\nu b^n - j_f^{n+1/2} + W I_s^{n+1/2}\bigr) - \sum_g c_g\,\operatorname{corr}_g(u^n, z^n),$$

both $A(\varepsilon) = D_+(\varepsilon) + \tfrac{\Delta t}{2} W \operatorname{diag}(g) W^T$ and
$B(\varepsilon) = \operatorname{diag}(a_1(\varepsilon)) - \tfrac{\Delta t}{2} W \operatorname{diag}(g) W^T$ depend on
$C_{0,e}$ strictly through their diagonal entries $(D_+)_{ee} = C_{0,e} + \dots$ and
$(a_1)_e = C_{0,e} - \dots$:

$$\frac{\partial A}{\partial C_{0,e}} = e_e e_e^T, \qquad
\frac{\partial B}{\partial C_{0,e}} = e_e e_e^T.$$

Differentiating the step equation with respect to $C_{0,e}$ at fixed $(u^n, b^n, z^n)$ gives

$$A \frac{\partial u^{n+1}}{\partial C_{0,e}} = e_e (u_e^n - u_e^{n+1})
\implies \frac{\partial u^{n+1}}{\partial C_{0,e}} = A^{-1} e_e (u_e^n - u_e^{n+1}).$$

Pairing this Jacobian with the post-Faraday adjoint vector $\tilde{u}^{n+1}$ yields the exact per-step capacitance VJP:

$$\left\langle \tilde{u}^{n+1}, \frac{\partial u^{n+1}}{\partial C_{0,e}} \right\rangle
= (A^{-T} \tilde{u}^{n+1})_e (u_e^n - u_e^{n+1})
= (D_+^{-1} \bar{u}^*)_e (u_e^n - u_e^{n+1})
= \bar{u}^*_e \cdot \frac{u_e^n - u_e^{n+1}}{(D_+)_{ee}}.$$

In `UnstructuredAdjointStepper.step`, the stored sensitivity row is $\text{row}_e = (u_e^n - u_e^{n+1}) / (D_+)_{ee}$
evaluated on the post-Woodbury $u^{n+1}$. In `_step3d_T`, `pre_step_T` maps $\tilde{u}^{n+1} \mapsto \bar{u}^* = D_+ A^{-T} \tilde{u}^{n+1}$
before `cu.transpose`, so `sens = u_star_b[dofs]` multiplied by `row` produces $(A^{-T} \tilde{u}^{n+1})_e (u_e^n - u_e^{n+1})$
identically, and `UnstructuredAdjointStepper.accumulate` requires no modification.

---

## 4. Half-Step Time-Centering and S-Parameter VJP (`PortDFTMonitor`)

### Half-Step Trapezoidal Monitor Kernel

In the leapfrog/trapezoidal port scheme, the source current $I_s^{n+1/2}$ and port voltage
$V_p^{n+1/2} = \tfrac{1}{2} w_p^T (u^n + u^{n+1})$ are centered at $t_{n+1/2} = (n + 1/2)\Delta t$. Because the
simulation starts from rest ($u^0 = 0$), the discrete-time Fourier transform of $V_p^{n+1/2}$ over $N_t$ steps
($n = 0, \dots, N_t - 1$) can be written directly as a linear functional of the post-step edge voltages $u^{n+1}$:

$$\hat{V}_p(f)
= \sum_{n=0}^{N_t - 1} \Delta t\, e^{-2\pi i f (n + 1/2)\Delta t}\, \frac{w_p^T u^n + w_p^T u^{n+1}}{2}
= \sum_{n=0}^{N_t - 1} \phi(f, n, N_t)\, (w_p^T u^{n+1}),$$

with phase weights (`PortDFTMonitor.phase(n, dt, Nt)`)

$$\phi(f, n, N_t) = \begin{cases}
\cos(\pi f \Delta t)\, \Delta t\, e^{-2\pi i f (n+1)\Delta t}, & 0 \le n < N_t - 1, \\[4pt]
\tfrac{1}{2}\,\Delta t\, e^{-2\pi i f (N_t - 1/2)\Delta t}, & n = N_t - 1.
\end{cases}$$

When `UnstructuredAdjointSimulation._phase` delegates to `PortDFTMonitor.phase(n, dt, Nt)`, the forward pass
accumulates $\hat{V}_p(f)$ identically to `_Port.spectra(freqs)` (to $< 10^{-15}$ relative), and the reverse pass
(`UnstructuredAdjointSimulation.vjp`) injects the exact conjugate-transposed source

$$\Delta \bar{u}^{n+1} = \sum_f W \operatorname{Re}\!\bigl\{\phi(f, n, N_t)^*\, \bar{V}(f)\bigr\}.$$

If the monitor instead samples $u^{n+1}$ at integer times $t_{n+1} = (n+1)\Delta t$ (`half_step=False`), the interior
amplitude differs by $1 - \cos(\pi f \Delta t) = O((\omega \Delta t)^2)$ and the relative phase between the source
at $t_{n+1/2}$ and the monitor at $t_{n+1}$ shifts by $\pi f \Delta t = \tfrac{1}{2}\omega \Delta t$ unless compensated,
leaking an $O(\omega \Delta t)$ error into the gradient when differentiated without matching the forward definition.

### Quotient-Rule VJP for $|S_{21}|^2$ and $|S_{11}|^2$

With port $e$ (`excite`) driven by $\hat{I}_{s,e}(f)$ and all other ports passive ($\hat{I}_{s,p} = 0$ for $p \ne e$),
the port current and Kurokawa power waves (Kurokawa 1965) with real reference admittance $Y_p(f) > 0$ are affine
functions of $\hat{V}_p(f)$:

$$\hat{I}_p(f) = \hat{I}_{s,p}(f) - g_p \hat{V}_p(f),$$

$$a_p(f) = \frac{\sqrt{Y_p}\hat{V}_p + \hat{I}_p / \sqrt{Y_p}}{2} = \alpha_p(f)\hat{V}_p(f) + \frac{\hat{I}_{s,p}(f)}{2\sqrt{Y_p(f)}}, \qquad
b_p(f) = \frac{\sqrt{Y_p}\hat{V}_p - \hat{I}_p / \sqrt{Y_p}}{2} = \beta_p(f)\hat{V}_p(f) - \frac{\hat{I}_{s,p}(f)}{2\sqrt{Y_p(f)}},$$

where $\alpha_p = (\sqrt{Y_p} - g_p/\sqrt{Y_p})/2$ and $\beta_p = (\sqrt{Y_p} + g_p/\sqrt{Y_p})/2$ are real.
For the scattering column $S_{p,e}(f) = b_p(f) / a_e(f)$, a variation $\delta \hat{V}$ induces

$$\delta S_{p,e} = \frac{\delta b_p}{a_e} - \frac{b_p}{a_e^2}\,\delta a_e
= \frac{\beta_p}{a_e}\,\delta \hat{V}_p - \frac{b_p \alpha_e}{a_e^2}\,\delta \hat{V}_e.$$

Under the real inner product $\langle G_S, \delta S\rangle = \operatorname{Re}\sum_{f,p} (G_S)_{f,p}^*\, \delta S_{p,e}(f)$,
the exact cotangent pullback `vjp_fn(G_S)` returned by `PortDFTMonitor.s_column` is

$$\bar{b}_p = \frac{(G_S)_{f,p}}{a_e^*}, \qquad
\bar{a}_e = -\sum_{p=1}^P \frac{b_p^*}{(a_e^*)^2}\,(G_S)_{f,p}, \qquad
\bar{V}_p = \beta_p\,\bar{b}_p + \delta_{p,e}\,\alpha_e\,\bar{a}_e.$$

For the power objective $F(\rho) = \sum_{f} \bigl(w_{21}|S_{21}(f)|^2 + w_{11}|S_{11}(f)|^2\bigr)$, the seed
cotangent is $(G_S)_{f,1} = 2 w_{11} S_{11}(f)$ and $(G_S)_{f,2} = 2 w_{21} S_{21}(f)$.

---

## 5. Central Finite-Difference Truncation Bound

For a smooth scalar objective $F(\rho)$, Taylor expansion around $\rho$ along coordinate $e_k$ gives

$$\operatorname{FD}_k(h) = \frac{F(\rho + h e_k) - F(\rho - h e_k)}{2h}
= \frac{\partial F}{\partial \rho_k}(\rho) + \frac{h^2}{6}\frac{\partial^3 F}{\partial \rho_k^3}(\rho) + O(h^4).$$

Evaluating the central difference at step sizes $h$ and $2h$ eliminates the exact derivative:

$$\operatorname{FD}_k(2h) - \operatorname{FD}_k(h) = \frac{4h^2 - h^2}{6}\frac{\partial^3 F}{\partial \rho_k^3}(\rho) + O(h^4)
= 3\left[\operatorname{FD}_k(h) - \frac{\partial F}{\partial \rho_k}(\rho)\right] + O(h^4).$$

Consequently, the leading $O(h^2)$ truncation error of $\operatorname{FD}_k(h)$ is estimated directly from the two
finite-difference evaluations by Richardson extrapolation:

$$\tau_k(h) = \frac{|\operatorname{FD}_k(2h) - \operatorname{FD}_k(h)|}{3}.$$

Adding the IEEE-754 float64 subtractive cancellation floor $r_k(h) = 10^3 \varepsilon_{\text{mach}} |F(\rho)| / h$
(with $\varepsilon_{\text{mach}} = 2^{-52} \approx 2.22\times 10^{-16}$) yields the combined discrepancy bound
$|g_{\text{adj},k} - \operatorname{FD}_k(h)| \le 1.5\,\tau_k(h) + r_k(h)$ enforced in
`tests/test_111_unstructured_ports3d_paths.py`.

---

## 6. Verification and WR-90 Matching Demo

### Test Suite (`tests/test_111_unstructured_ports3d_paths.py`)

All 5 tests in `tests/test_111_unstructured_ports3d_paths.py` run on a 4-layer extruded WR-90 waveguide
($a \times b = 22.86 \times 10.16\text{ mm}$, $L = 12\text{ mm}$, $h = a/4$, 192 tetrahedra, 48 design tetrahedra in
layer 1, dielectric slab $\varepsilon_r = 2.2$ in layer 2, $\text{ TE}_{10}$ modal ports at $z = 0$ and $z = L$, and
an interior 50 $\Omega$ lumped port path):

* **Forward replay and DFT equivalence (`test_adjoint_forward_bitwise_and_dot_product`):**
  `UnstructuredAdjointStepper.step` matches `UnstructuredFIT3D.one_step` bitwise (`assert_array_equal`) over 60 steps
  with two modal ports, one lumped port, and massless edges active. `PortDFTMonitor` matches `_Port.spectra` and
  `power_waves` scattering columns at 9, 10, and 11 GHz to $< 10^{-14}$ relative.
* **Dot-product identity (`test_adjoint_forward_bitwise_and_dot_product`):**
  For random energy-scaled states $x, y$:
  * Blockwise scale-invariant error: $1.378\times 10^{-17} \le 10^{-15}$.
  * Multi-step relative error $|\langle y, L^k x\rangle - \langle (L^T)^k y, x\rangle| / (|\langle y, L^k x\rangle| + |\langle (L^T)^k y, x\rangle|)$:
    $2.072\times 10^{-16}$ ($k = 1$), $3.144\times 10^{-16}$ ($k = 5$), and $3.760\times 10^{-16}$ ($k = 10$), all $\le 10^{-15}$.
* **Scattering-objective gradient vs central FD (`test_adjoint_gradient_s21_s11_vs_finite_differences`):**
  For $F(\rho) = \sum_{f \in \{9, 10, 11\text{ GHz}\}} (|S_{21}(f)|^2 - 0.5|S_{11}(f)|^2)$ with $h = 10^{-3}$ across 4
  sampled design tetrahedra spanning the cross-section:

| Design cell | $g_{\text{adj},k}$ | $\operatorname{FD}_k(h)$ | $|g_{\text{adj},k} - \operatorname{FD}_k(h)|$ | $\tau_k(h) = |\operatorname{FD}(2h)-\operatorname{FD}(h)|/3$ | Relative error |
|---|---|---|---|---|---|
| 11 | $-6.317103\times 10^{-3}$ | $-6.317102\times 10^{-3}$ | $4.869\times 10^{-10}$ | $4.869\times 10^{-10}$ | $7.707\times 10^{-8}$ |
| 4 | $-2.568791\times 10^{-3}$ | $-2.568791\times 10^{-3}$ | $2.356\times 10^{-11}$ | $2.355\times 10^{-11}$ | $9.172\times 10^{-9}$ |
| 1 | $-3.589886\times 10^{-3}$ | $-3.589886\times 10^{-3}$ | $1.603\times 10^{-11}$ | $1.621\times 10^{-11}$ | $4.465\times 10^{-9}$ |
| 23 | $-4.364126\times 10^{-3}$ | $-4.364125\times 10^{-3}$ | $2.282\times 10^{-10}$ | $2.283\times 10^{-10}$ | $5.229\times 10^{-8}$ |

* **Binomial checkpointing (`test_adjoint_gradient_s21_s11_vs_finite_differences`):**
  `sim.vjp(G_cot, memory="checkpoint", snaps=12)` matches `sim.vjp(G_cot, memory="full")` bitwise
  (`assert_array_equal`) with `stepper.replays == 0`.
* **Interior `CellDFTMonitor` with modal + lumped ports (`test_adjoint_cell_dft_monitor_with_lumped_and_modal_ports`):**
  Matches central finite differences within $1.5\,\tau_k(h) + r_k(h)$ on all sampled cells.
* **Mutations (`test_adjoint_dot_product_mutations_fail`, `test_adjoint_gradient_mutations_fail`):**
  * Dropping the port term in `linear_step_T` (`pre_step_T = id`, `post_step_T = 0`) raises the $k=1$ dot-product
    error from $2.072\times 10^{-16}$ to $1.278\times 10^{-3}$ (blockwise $4.120\times 10^{-4}$) and produces a
    $4.194\times 10^{-1}$ ($41.9\%$) relative error in the $|S_{21}|^2 - 0.5|S_{11}|^2$ gradient. When injected directly
    into the unmutated test suite, `test_adjoint_forward_bitwise_and_dot_product` fails with
    `AssertionError: assert np.float64(0.002099648422757077) <= 1e-15` and
    `test_adjoint_gradient_s21_s11_vs_finite_differences` fails with
    `AssertionError: assert 0.0026497043580277473 <= ((1.5 * 4.869114370957087e-10) + 2.4470659161869384e-11)`.
  * Flipping the port sign in `linear_step_T` raises the $k=1$ dot-product error to $3.291\times 10^{-3}$ ($k=3$:
    $1.037\times 10^{-2}$).
  * Omitting the half-step time-centering in the reverse pass (`mon.half_step = False` in `vjp`) raises the relative
    gradient error to $3.631\times 10^{-2}$ ($3.63\%$, exceeding the truncation bound by $> 2.5\times 10^5$).

### WR-90 Permittivity-Matching Study (`studies/unstructured-ports3d/adjoint_wr90_match.py`)

`studies/unstructured-ports3d/adjoint_wr90_match.py` optimises a 3-layer matching section ($2.5 < z < 10.0\text{ mm}$,
144 design tetrahedra with $\varepsilon_T \in [1.0, 2.6]$) in front of a dielectric load ($\varepsilon_r = 2.6$,
$10.0 < z < 17.5\text{ mm}$) inside an 8-layer extruded WR-90 guide ($L = 20\text{ mm}$, $\Delta z = 2.5\text{ mm}$,
960 tetrahedra, 186 active free edges, 144 massless section edges, $\Delta t = 2.610\text{ ps}$, $N_t = 320$) to
maximise $F(\rho) = |S_{21}(f_0)|^2 - |S_{11}(f_0)|^2$ at $f_0 = 10\text{ GHz}$. Results from
`out/adjoint_wr90_match.json`:

* **Dot-product errors on the 960-tet mesh:** $7.281\times 10^{-16}$ ($k = 1$), $3.003\times 10^{-16}$ ($k = 5$), and
  $7.080\times 10^{-16}$ ($k = 10$); dropping the port term in $L^T$ gives $1.664\times 10^{-2}$ and flipping its sign
  gives $4.055\times 10^{-2}$.
* **Adjoint vs central FD ($h = 10^{-3}$) across 6 design tetrahedra (2 per matching layer):** maximum relative error
  $2.551\times 10^{-7}$ (down to $4.783\times 10^{-9}$ on cell 60, where $|g_{\text{adj}} - \operatorname{FD}(h)| = 1.685\times 10^{-10}$
  matches $\tau(h) = 1.676\times 10^{-10}$). Dropping the port transpose gives up to $53.14$ relative error; using an
  integer-step $t_{n+1}$ monitor in `vjp` gives $6.575\times 10^{-2}$ relative error.
* **L-BFGS-B optimisation (6 iterations, wall time 4.87 s):**
  * Unmatched initial state ($\rho = 0$, $\varepsilon_r = 1$ in layers 1–3): $|S_{11}(f_0)| = -8.342\text{ dB}$
    ($0.3827$), $|S_{21}(f_0)|^2 = 0.8223$, objective $F = 0.6758$.
  * Optimised 6-iteration state: $|S_{11}(f_0)| = -37.235\text{ dB}$ ($0.01375$, a $28.89\text{ dB}$ return-loss
    reduction), $|S_{21}(f_0)|^2 = 0.9576$, objective $F = 0.9574$, with monotonically graded layer-mean permittivities
    $\bar{\varepsilon}_r = (1.0533, 1.2669, 1.4200)$ between air ($1.0$) and the $\varepsilon_r = 2.6$ load.

---

## 7. Limits & Failure Modes

* **Design cells must not touch port edges.** `PortAdjointHelper` assumes the port weight vectors $w_p$ and reference
  conductances $g_p$ are independent of the design variables $\rho$. If a design cell shares an edge with a `PortPlane`
  or `LumpedPort`, changing $\varepsilon_T$ alters the 2D cross-section eigenproblem (`te_modes` / `tem_mode`) and the
  reference admittance $Y_h(f_0)$; those boundary-mode sensitivities $\partial w_p / \partial \varepsilon_T$ and
  $\partial g_p / \partial \varepsilon_T$ are not differentiated. Keep design cells strictly inside the interior layers.
* **Design cells must not touch massless boundary edges.** As in `UnstructuredAdjointStepper` without ports, the
  massless edge subset `solver.massless` (where $C_{0,e}$ is below `massless_tol * median(C_0)` and zeroed into the
  magnetostatic Schur complement $K_{ss}$) is frozen when the solver is built. Perturbing a cell whose edges belong
  to `solver.massless` does not change $C_0$ on those edges.
* **Voronoi Hodge only (`hodge="voronoi"`).** Both `attach_ports` and `PortAdjointHelper` require diagonal dual-cell
  capacitances $D_+$. The sparse Whitney 1-form mass matrix (`hodge="whitney"`) has no port conductance path.
* **Narrowband conductance termination.** Because each `ModalPort` terminates the guide with a frequency-independent
  conductance $g_p = \operatorname{Re}\{Y_h(f_0)/\cos(\omega_0\Delta t/2)\}$ matched at $f_0$, waves at $f \ne f_0$
  partially reflect at the ports during the transient. While `PortDFTMonitor.s_column` accounts for $g_p \ne Y_p(f)$
  in the Kurokawa power waves $a_p, b_p$, the finite integration window $N_t \Delta t$ must be long enough for internal
  reflections to decay; otherwise both the forward DFT and its adjoint gradient include the exact discrete
  finite-window truncation of the $N_t$-step run.
* **3D spherical PML + ports combination is not supported.** `UnstructuredAdjointStepper._step3d` / `_step3d_T`
  dispatch to `_step3d_pml` / `_step3d_pml_T` when `solver._pml` is present, which uses its own extended state layout
  and does not apply `PortAdjointHelper`.

---

## References

* M. B. Giles and N. A. Pierce, "An introduction to the adjoint approach to design," Flow Turbul. Combust. 65,
  393–415 (2000). doi:10.1023/A:1011430410075
* M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique," Prog.
  Electromagn. Res. 32, 65–87 (2001). doi:10.2528/PIER00080103
* K. Kurokawa, "Power waves and the scattering matrix," IEEE Trans. Microw. Theory Techn. 13(2), 194–202 (1965).
  doi:10.1109/TMTT.1965.1125964
* W. W. Hager, "Updating the inverse of a matrix," SIAM Rev. 31(2), 221–239 (1989). doi:10.1137/1031049
* A. Griewank and A. Walther, "Algorithm 799: Revolve: an implementation of checkpointing for the reverse or
  adjoint mode of computational differentiation," ACM Trans. Math. Softw. 26(1), 19–45 (2000).
  doi:10.1145/347837.347846
