# GPU port path: tetrahedral lumped and waveguide ports on the device

`wakis.unstructured.gpu.DeviceStepper` supports 3D tetrahedral solvers (`UnstructuredFIT3D`, Voronoi Hodge)
with attached lumped and waveguide ports (`LumpedPort`, `ModalPort`, `PortSet`, `attach_ports` in
`wakis.unstructured.ports3d`). Companion to `docs/physicsguide_sections/unstructured_ports3d.md` and
`docs/physicsguide_sections/unstructured_gpu.md`.

---

## GPU port path

### 1. Derivation of the device port update

On a 3D tetrahedral mesh with Voronoi dual Hodges $M_\varepsilon = \operatorname{diag}(\mathbf{C}_0)$ and
$M_\nu = \operatorname{diag}(\mathbf{M}_{\nu,\mathrm{diag}})$ (Clemens & Weiland 2001), `DeviceStepper` folds the
quasi-static massless-edge elimination $\mathbf{u}_s^n = -K_{ss}^{-1} K_{sf} \mathbf{u}^n$ into the effective
curl operator $C_{\mathrm{eff}} = C_f - C_s K_{ss}^{-1} K_{sf}$ at setup (`docs/physicsguide_sections/unstructured_gpu.md`,
§1). At step $n \to n+1$ ($t_{n+1/2} = (n + 1/2)\Delta t$), Faraday's law and the interior curl-minus-source
vector on the $n_{\mathrm{free}}$ active edges `solver.free` are

$$\mathbf{b}^{n+1/2} = \mathbf{b}^{n-1/2} - \Delta t\, C_{\mathrm{eff}} \mathbf{u}^n, \qquad
\mathbf{r} = C_f^T \bigl(M_\nu \mathbf{b}^{n+1/2}\bigr) - \mathbf{j}_{\mathrm{src}}^{n+1/2}.$$

With $P$ ports attached via `PortSet`, let $W = [\mathbf{w}_1, \dots, \mathbf{w}_P] \in \mathbb{R}^{n_{\mathrm{free}} \times P}$
collect the port weight vectors on the free edges, $\mathbf{g} = (g_1, \dots, g_P) \ge 0$ be the port
conductances [S], and $\mathbf{I}_s^{n+1/2} \in \mathbb{R}^P$ be the impressed Norton port currents [A].
Trapezoidal integration of the port currents $\mathbf{I}^{n+1/2} = \mathbf{I}_s^{n+1/2} - \frac{1}{2}\mathbf{g}\odot W^T(\mathbf{u}^n + \mathbf{u}^{n+1})$
gives the semi-implicit electric update (`docs/physicsguide_sections/unstructured_ports3d.md`, §2)

$$\Bigl(D_+ + \frac{\Delta t}{2} W \operatorname{diag}(\mathbf{g}) W^T\Bigr) \mathbf{u}^{n+1}
= \Bigl(\operatorname{diag}(\mathbf{a}_1) - \frac{\Delta t}{2} W \operatorname{diag}(\mathbf{g}) W^T\Bigr) \mathbf{u}^n
+ \Delta t\,\bigl(\mathbf{r} + W \mathbf{I}_s^{n+1/2}\bigr) - \sum_m c_m\,\mathrm{corr}_m,$$

where $D_+ = \operatorname{diag}(\mathrm{den})$ (`den = C0 + G dt / 2 + sum c beta`) and
$\mathbf{a}_1 = \mathbf{C}_0 - \mathbf{G}\Delta t/2 - \sum \mathbf{c}\beta$. Applying the Sherman–Morrison–Woodbury
identity (Hager 1989) with $\mathbf{d}_{\mathrm{inv}} = \Delta t / \mathrm{den} = \Delta t\, D_+^{-1}\mathbf{1}$,
$Q = W^T \operatorname{diag}(\mathbf{d}_{\mathrm{inv}}) W \in \mathbb{R}^{P \times P}$, and
$S_P = I_P + \frac{1}{2}\operatorname{diag}(\mathbf{g}) Q \in \mathbb{R}^{P \times P}$ splits the port update
into a pre-step before `_cap_step` and a post-step after `_cap_step`:

1. **Port pre-step (`DeviceStepper._port_pre`)**: one sparse dot product per port ($W^T \mathbf{u}^n$) and one
   sparse scatter-axpy into $\mathbf{r}$ on the $n_p$ non-empty port-edge rows of $W$:
   $$\mathbf{V}^n = W^T \mathbf{u}^n, \qquad
   \mathbf{c}_{\mathrm{pre}} = \mathbf{I}_s^{n+1/2} - \frac{1}{2}\mathbf{g} \odot \mathbf{V}^n, \qquad
   \mathbf{r}_{\mathrm{port}} = \mathbf{r} + W \mathbf{c}_{\mathrm{pre}}.$$
2. **Diagonal / ADE capacitive update (`DeviceStepper._cap_step`)**:
   $$\mathbf{u}^* = D_+^{-1}\Bigl(\mathbf{a}_1 \odot \mathbf{u}^n + \Delta t\,\mathbf{r}_{\mathrm{port}} - \sum_m c_m\,\mathrm{corr}_m\Bigr).$$
3. **Port post-step (`DeviceStepper._port_post`)**: one sparse dot product per port ($W^T \mathbf{u}^*$), the
   $P \times P$ Woodbury solve, and one sparse scatter-axpy on the $n_p$ port edges:
   $$\mathbf{V}^* = W^T \mathbf{u}^*, \qquad
   \boldsymbol{\alpha} = S_P^{-1}\Bigl(\frac{1}{2}\mathbf{g} \odot \mathbf{V}^*\Bigr), \qquad
   \mathbf{u}^{n+1} = \mathbf{u}^* - \mathbf{d}_{\mathrm{inv}} \odot (W \boldsymbol{\alpha}).$$
4. **Device port monitor recording**: the half-step port voltages
   $$\mathbf{V}^{n+1/2} = \frac{1}{2}\bigl(\mathbf{V}^n + W^T \mathbf{u}^{n+1}\bigr)$$
   and impressed currents $\mathbf{I}_s^{n+1/2}$ are written into a preallocated device buffer `ports["buf"]` of
   shape `(n_cap, 2 * P)`. Calling `to_host()` copies `V` and `Is` back to `solver.ports.ports` so `_Port.spectra`
   and `power_waves` (Kurokawa 1965) work identically to a host run.

*(Algebraic note)*: Because port edges are non-dispersive and $\frac{1}{2}\operatorname{diag}(\mathbf{g}) Q = S_P - I_P$,
substituting $\mathbf{u}^* = \tilde{\mathbf{u}} + \mathbf{d}_{\mathrm{inv}} \odot (W \mathbf{c}_{\mathrm{pre}})$
(where $\tilde{\mathbf{u}}$ is the port-free capacitive update) yields
$\mathbf{c}_{\mathrm{pre}} - \boldsymbol{\alpha} = S_P^{-1}\bigl(\mathbf{I}_s^{n+1/2} - \frac{1}{2}\mathbf{g}\odot W^T(\mathbf{u}^n + \tilde{\mathbf{u}})\bigr)$,
so the entire port update is a single rank-$P$ update $\mathbf{u}^{n+1} = \tilde{\mathbf{u}} + \mathbf{d}_{\mathrm{inv}}\odot(W S_P^{-1}[\mathbf{I}_s^{n+1/2} - \frac{1}{2}\mathbf{g}\odot(\mathbf{V}^n + W^T\tilde{\mathbf{u}})])$.
`DeviceStepper` executes the two-stage `_port_pre` / `_port_post` form so that its floating-point parenthesisation
matches `PortSet.step` on the host term-for-term.

**No host synchronisation.** Impressed port currents $\mathbf{I}_s((n + 1/2)\Delta t)$ are pre-evaluated into a
device table `ports["Is"]` of shape `(n_max, P)` in `_port_table` / `prepare`. Inside `step()`, `Is[self.n]` is a
zero-copy device slice, $W^T \mathbf{u}$ uses the 4-lane sub-warp CSR kernel `uns_spmv`, the port-edge updates use
`uns_spmv_scatter` restricted to the $n_p$ active port edges `ports["rows"]`, and the $P$-component vector updates
run in `uns_port_pre`, `uns_port_post`, and `uns_port_avg`. No `.get()`, `asnumpy`, `float()`, or host-to-device
transfer occurs inside `step()`.

---

### 2. Derivation of the CPU–GPU round-off bound

The CPU (`UnstructuredFIT3D.one_step`) and GPU (`DeviceStepper.step`) paths evaluate the same linear recurrence
$\mathbf{x}_{n+1} = A\mathbf{x}_n + B\mathbf{f}_n$ on $\mathbf{x}_n = (\mathbf{u}^n, \mathbf{b}^{n-1/2}, \dots)$.
By standard floating-point error analysis (Higham 2002, §3.1, eq. 3.4 and Lemma 3.3), each step incurs a local
rounding error $|\boldsymbol{\delta}_n| \le \gamma_m |A|(|\hat{\mathbf{x}}_n|) + \mathcal{O}(u^2)$ componentwise
for any summation order and FMA contraction, where $u = 2^{-53}$ and $\gamma_m = mu/(1 - mu)$. As derived in
`docs/physicsguide_sections/unstructured_gpu.md` (§2) and enforced by `wakis.unstructured.gpu.agreement_bound`
in `tests/test_099_unstructured_gpu.py`, the weighted energy-norm difference
$\mathbf{d}_n = \hat{\mathbf{x}}_n^{\mathrm{GPU}} - \hat{\mathbf{x}}_n^{\mathrm{CPU}}$ is bounded by

$$\|\mathbf{d}_N\|_W \le \varepsilon_N := 2 K \gamma_m \sum_{n < N} \bigl\| |A|(|\mathbf{x}_n|) \bigr\|_W, \qquad
K = \sup_{k \le N} \|A^k\|_W.$$

With `PortSet` attached, the three ingredients of `agreement_bound` extend as follows:

1. **Absolute-value twin $|A|(|\mathbf{x}_n|)$**: In `DeviceStepper(solver, absolute=True)` (`self.m = +1`),
   `_build_ports` replaces $W$, $W^T$, $S_P^{-1}$, and $\mathbf{I}_s^{n+1/2}$ by their entrywise absolute values
   $|W|$, $|W|^T$, $|S_P^{-1}|$, and $|\mathbf{I}_s^{n+1/2}|$, and `_port_pre` / `_port_post` replace subtractions
   with additions:
   $$|\mathbf{r}_{\mathrm{port}}| = |\mathbf{r}| + |W|\Bigl(|\mathbf{I}_s^{n+1/2}| + \frac{1}{2}\mathbf{g}\odot\bigl(|W|^T|\mathbf{u}^n|\bigr)\Bigr),$$
   $$|\mathbf{u}^{n+1}| = |\mathbf{u}^*| + \mathbf{d}_{\mathrm{inv}} \odot \Bigl(|W|\Bigl(|S_P^{-1}|\Bigl(\frac{1}{2}\mathbf{g}\odot\bigl(|W|^T|\mathbf{u}^*|\bigr)\Bigr)\Bigr)\Bigr).$$
2. **Operation depth $m$ (`DeviceStepper.depth`)**: By Higham (2002, Lemma 3.3 / §3.5), chained matrix-vector
   products add their maximum row lengths to the operation count feeding any output component:
   * `_port_pre`: $k_{W^T} = \max_p \mathrm{nnz}(W^T_{p,:})$ for $\mathbf{V}^n = W^T \mathbf{u}^n$, $2$ flops for
     $I_{s,p} - \frac{1}{2}g_p V_p^n$, and $k_W + 1 = \max_e \mathrm{nnz}(W_{e,:}) + 1$ flops for
     $\mathbf{r} + W \mathbf{c}_{\mathrm{pre}}$.
   * `_port_post`: $k_{W^T}$ flops for $\mathbf{V}^* = W^T \mathbf{u}^*$, $P + 1$ flops for
     $S_P^{-1}(\frac{1}{2}\mathbf{g}\odot \mathbf{V}^*)$, and $k_W + 2$ flops for
     $\mathbf{u}^* - \mathbf{d}_{\mathrm{inv}} \odot (W \boldsymbol{\alpha})$.
   Summing these contributions gives the exact depth increment added in `_build_ports`:
   $$\Delta m_{\mathrm{port}} = 2\,\max\mathrm{row\text{-}nnz}(W^T) + 2\,\max\mathrm{row\text{-}nnz}(W) + P + 6.$$
3. **Amplification factor $K$**: On the physical subspace $\mathcal{V}_{\mathrm{phys}} = \{(\mathbf{u}, \mathbf{b}) : C_s^T M_\nu \mathbf{b} = 0\}$
   (invariant under $\mathbf{b}^{n+1/2} = \mathbf{b}^{n-1/2} - \Delta t\, C_{\mathrm{eff}}\mathbf{u}^n$ since
   $C_s^T M_\nu C_{\mathrm{eff}} = 0$), the homogeneous leapfrog ($I_s = 0$) with passive ports ($g_p \ge 0$)
   dissipates the modified energy $E_{\mathrm{mod}}^n = \frac{1}{2}(\mathbf{u}^n)^T M_\varepsilon \mathbf{u}^n + \frac{1}{2}(\mathbf{b}^{n+1/2})^T M_\nu \mathbf{b}^{n-1/2}$:
   $$E_{\mathrm{mod}}^{n+1} - E_{\mathrm{mod}}^n = -\Delta t\,\bar{\mathbf{u}}^T\bigl(G + W\operatorname{diag}(\mathbf{g})W^T\bigr)\bar{\mathbf{u}} \le 0, \qquad
   \bar{\mathbf{u}} = \frac{\mathbf{u}^n + \mathbf{u}^{n+1}}{2}.$$
   Since $(1 - \rho)E \le E_{\mathrm{mod}} \le (1 + \rho)E$ with Courant number
   $\rho = \frac{\Delta t}{2}\sqrt{\lambda_{\max}}$ (Diaz & Grote 2009), the lossless amplification bound
   $$K \le \sqrt{\frac{1 + \rho}{1 - \rho}}$$
   (`theory_amplification`) holds for any passive `PortSet`.
4. **Componentwise bound on recorded port voltages $\mathbf{V}^{n+1/2}$**: Using $|d_{n,e}^u| \le W_{u,e}^{-1/2}\varepsilon_n$
   (with $\varepsilon_{-1} = 0$ at $n = 0$) and the local error $\gamma_{k_{W^T} + 1}$ of each side's half-step
   average $\frac{1}{2}(W^T\mathbf{u}^n + W^T\mathbf{u}^{n+1})$, `agreement_bound` bounds the recorded port voltage
   difference by
   $$\bigl|\Delta \mathbf{V}^{n+1/2}\bigr| \le \frac{\varepsilon_{n-1} + \varepsilon_n}{2}\,\bigl(|W|^T W_u^{-1/2}\bigr)
   + \gamma_{k_{W^T} + 1}\,|W|^T\bigl(|\mathbf{u}^n| + |\mathbf{u}^{n+1}|\bigr).$$

---

### 3. Verification (`tests/test_111_unstructured_ports3d_paths.py`, `out/gpu_ports.json`)

All numbers below are produced on an NVIDIA L4 GPU by running `python tests/test_111_unstructured_ports3d_paths.py`
(saved in `out/gpu_ports.json`, Batch task `w3p-b-gpu`). The test problem is the 4-layer extruded WR-90 guide
($a = 22.86\text{ mm}$, $b = 10.16\text{ mm}$, $L = 12.0\text{ mm}$, $h = a/4 = 5.715\text{ mm}$, 480 tetrahedra,
1048 faces, 102 active free edges, 72 massless quad-diagonal edges with $\kappa_{\max} = 1.0$ and block size $1$,
dielectric slab $\varepsilon_r = 2.2$ in layer 2, $\Delta t = 2.6634\text{ ps}$, $\rho = 0.95$,
$K = \sqrt{(1+\rho)/(1-\rho)} = 6.2450$, $N = 120$ steps, probe in layer 2, and 3-frequency DFT at
$f \in \{9.0, 10.0, 11.0\}\text{ GHz}$) in two port configurations:

* `modal`: two $\text{TE}_{10}$ `ModalPort` terminations at $z = 0$ (excited) and $z = L$ (passive), $P = 2$,
  $n_p = 36$ port edges, $\max\mathrm{row\text{-}nnz}(W^T) = 18$, $\max\mathrm{row\text{-}nnz}(W) = 1$,
  $\Delta m_{\mathrm{port}} = 46$, total depth $m = 75$, $\gamma_{75} = 8.3267\times 10^{-15}$, and subspace
  empirical amplification $K_{\mathrm{emp,sub}} = 1.2209 \le K = 6.2450$.
* `mixed`: the two $\text{TE}_{10}$ modal ports plus an interior $50\,\Omega$ `LumpedPort` path, $P = 3$,
  $n_p = 38$ port edges, $\Delta m_{\mathrm{port}} = 47$, total depth $m = 76$,
  $\gamma_{76} = 8.4377\times 10^{-15}$, and $K_{\mathrm{emp,sub}} = 1.2109 \le K = 6.2450$.

#### CPU–GPU agreement (`test_gpu_ports_match_cpu_within_roundoff_bound`)

| Case | Backend | $\|\mathbf{d}_N\|_W / \max\|\mathbf{x}\|_W$ | $\varepsilon_N / \max\|\mathbf{x}\|_W$ | $\|\mathbf{d}_N\|_W / \varepsilon_N$ | Max probe ratio | Max DFT ratio | Max $|\Delta V^{n+1/2}|$ / bound | S-param rel. diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `modal` | `cupy` (L4) | `1.0870e-15` | `6.2015e-12` | `1.7529e-04` | `3.2651e-05` | `6.4739e-05` | `2.6350e-04` | `2.4041e-16` |
| `modal` | `numpy` | `1.0702e-15` | `6.2015e-12` | `1.7257e-04` | `2.2948e-05` | `6.1532e-05` | `1.6958e-04` | `2.4041e-16` |
| `mixed` | `cupy` (L4) | `8.7159e-16` | `6.3697e-12` | `1.3683e-04` | `1.2462e-05` | `4.7972e-05` | `2.6006e-04` | `3.4867e-16` |
| `mixed` | `numpy` | `7.6342e-16` | `6.3697e-12` | `1.1985e-04` | `1.8693e-05` | `4.4210e-05` | `1.6735e-04` | `3.5348e-16` |

#### Mutation checks (`test_gpu_ports_mutations_exceed_bound`, `mixed`, L4 GPU)

Each mutation perturbs only the device port step while comparing against the unmodified CPU reference and bound
$\varepsilon_N = 2.6689\times 10^{-16}$:

| Mutation | Description | $\|\mathbf{d}_N\|_W$ | $\|\mathbf{d}_N\|_W / \varepsilon_N$ |
|---|---|---:|---:|
| `drop_port_step` | Drop `_port_pre` and `_port_post` on device (`dev.ports = None`) | `4.1899e-05` | `1.5699e+11` |
| `drop_port_post` | Drop Woodbury implicit solve `_port_post` on device | `1.4910e-05` | `5.5868e+10` |
| `zero_port_pre_g` | Zero explicit half-step conductance (`gh = 0`) | `4.9354e-05` | `1.8492e+11` |
| `scale_port_sinv_1e-3` | Perturb Woodbury inverse $S_P^{-1} \times (1 + 10^{-3})$ | `1.3603e-08` | `5.0968e+07` |
| `flip_port_pre_conductance` | Flip sign of explicit half-step conductance (`gh -> -gh`) | `9.7365e+49` | `3.6481e+65` |

#### Stream non-synchronisation (`test_gpu_ports_step_does_not_synchronize`)

After 5 warm-up steps and `prepare(50)`, three `DeviceStepper.step()` calls on `mixed` are enqueued behind a
$100\text{ ms}$ device `_spin_port` kernel on the default CUDA stream and return while `ev.done` is `False`;
injecting a `float(u[0])` device-to-host read inside `_port_post` forces stream synchronisation (`ev.done == True`)
and is caught by the test.

---

### 4. Limits & Failure Modes

* **Adjoint reverse sweep is host-only (`wakis.unstructured.adjoint`).** `DeviceStepper` implements the forward
  time-stepping path on the GPU (fields, probes, DFTs, and port signals). The exact transpose and reverse-mode
  permittivity-gradient machinery (`UnstructuredAdjointStepper`, `UnstructuredAdjointSimulation`, and
  `PortAdjointHelper` in `wakis/unstructured/adjoint.py` and `wakis/unstructured/ports3d.py`) run on host
  NumPy/SciPy arrays and are not wired to `DeviceStepper` on the GPU.
* **Voronoi Hodge without spherical PML (`hodge="voronoi"`, `pml=None`).** As on the host (`PortSet`), device
  ports require diagonal dual-cell capacitances and are not supported together with the 3D spherical CFS-PML
  extended-state stage matrices (`_step_pml3d`).
* **Non-dispersive port edges.** Port edges must not belong to any dispersive ADE pole group (`PortSet` raises
  `NotImplementedError` at construction if a port edge is dispersive); interior non-port edges may carry Debye,
  Drude, or Lorentz poles.
* **Unfused capacitive launch when ports are active.** Without ports or ADE groups, `_step_3d` fuses the curl
  SpMV and diagonal capacitive update into a single `uns_spmv_cap` launch. When ports are attached, `_step_3d`
  evaluates `CfT.spmv_into`, `_port_pre`, `_cap_step`, and `_port_post` as separate kernel launches so that
  $\mathbf{r}$ and $\mathbf{u}^*$ receive the pre- and post-step port corrections; on small meshes where stepping
  is launch-bound, this adds kernel launch overhead per step.
* **Magnetostatic constraint on massless edges.** On meshes with massless edges (`solver.massless`), the energy
  amplification bound $K = \sqrt{(1+\rho)/(1-\rho)}$ holds on the invariant physical subspace
  $\mathcal{V}_{\mathrm{phys}} = \{(\mathbf{u}, \mathbf{b}) : C_s^T M_\nu \mathbf{b} = 0\}$ entered from
  $\mathbf{b}^0 = 0$ ($K_{\mathrm{emp,sub}} = 1.2209$ on `modal` and $1.2109$ on `mixed`, vs $K = 6.2450$).
  Unprojected random initial face fluxes $\mathbf{b}^0 \notin \ker(C_s^T M_\nu)$ in `empirical_amplification`
  excite the static difference between $C_f^T M_\nu \mathbf{b}^0$ and $C_{\mathrm{eff}}^T M_\nu \mathbf{b}^0$.

---

## References

* M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique," *Prog.
  Electromagn. Res.* 32, 65–87 (2001). doi:10.2528/PIER00080103
* W. W. Hager, "Updating the inverse of a matrix," *SIAM Rev.* 31(2), 221–239 (1989). doi:10.1137/1031049
* K. Kurokawa, "Power waves and the scattering matrix," *IEEE Trans. Microw. Theory Techn.* 13(2), 194–202 (1965).
  doi:10.1109/TMTT.1965.1125964
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002).
  doi:10.1137/1.9780898718027
* J. Diaz and M. J. Grote, "Energy conserving explicit local time stepping for second-order wave equations,"
  *SIAM J. Sci. Comput.* 31, 1985–2014 (2009). doi:10.1137/070709414
