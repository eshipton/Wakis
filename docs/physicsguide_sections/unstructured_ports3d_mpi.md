# Tetrahedral Ports under MPI (`wakis.unstructured.ports3d` + `wakis.unstructured.mpi`)

Scope: domain-decomposed execution (`DistributedFIT`, `LocalCluster`) of 3D tetrahedral FIT solvers
(`UnstructuredFIT3D`, Voronoi Hodge) terminated by modal and lumped ports (`ModalPort`, `LumpedPort`, `PortSet`,
`attach_ports`), with bitwise IEEE-754 equivalence (`np.testing.assert_array_equal`) to the serial solver across
arbitrary rank counts $R \ge 1$. Companion to `docs/physicsguide_sections/unstructured_mpi.md` and
`docs/physicsguide_sections/unstructured_ports3d.md`.

---

## 1. Distributed Port Step and Global Inner Products

With $P$ ports attached to an `UnstructuredFIT3D` solver, let $W = [w_1, \dots, w_P] \in \mathbb{R}^{n_{\text{free}} \times P}$
be the sparse port weight matrix on the active (free, non-PEC, non-massless) edges `solver.free`, and let
$g = (g_1, \dots, g_P) \ge 0$ be the terminating modal or lumped conductances. As derived in
`docs/physicsguide_sections/unstructured_ports3d.md`, one leapfrog step $n \to n+1$ of `UnstructuredFIT3D.one_step()`
with `PortSet` executes five operations in sequence:

1. **Port pre-step (`PortSet.pre_e`):** evaluate the $P$-vector of explicit port voltages and inject the net explicit
   port current into the curl residual $r = C_f^T M_\nu b^n - j_f^{n+1/2}$:
   $$V^n = W^T u^n \in \mathbb{R}^P, \qquad q_{\text{pre}}^n = I_s^{n+1/2} - \frac{1}{2} g \odot V^n, \qquad
   r \leftarrow r + W q_{\text{pre}}^n.$$
2. **Diagonal capacitive update (`_Capacitive.forward`):** advance every free edge independently using the diagonal
   capacitance denominator $D_+ = \operatorname{diag}(\text{den})$ (`cu.den`):
   $$u_*^{n+1} = \mathcal{U}(u^n, r).$$
3. **Port post-step (`PortSet.post_e`):** apply the Sherman–Morrison–Woodbury rank-$P$ inverse (Hager 1989) with cached
   $P \times P$ matrix $S_P^{-1} = \bigl(I_P + \operatorname{diag}(\tfrac{1}{2}\Delta t\, g)\, W^T D_+^{-1} W\bigr)^{-1}$
   and accumulate the trapezoidal port voltage record $V^{n+1/2}$:
   $$V_*^{n+1} = W^T u_*^{n+1} \in \mathbb{R}^P, \qquad
   \delta V^{n+1} = S_P^{-1} V_*^{n+1}, \qquad
   u^{n+1} = u_*^{n+1} - D_+^{-1} W \bigl(\tfrac{1}{2}\Delta t\, g \odot \delta V^{n+1}\bigr),$$
   $$V^{n+1/2} = \frac{1}{2}\bigl(V^n + V_*^{n+1} - (W^T D_+^{-1} W)\bigl(\tfrac{1}{2}\Delta t\, g \odot \delta V^{n+1}\bigr)\bigr).$$
4. **Massless edge Schur solve (when `solver.massless.size > 0`):**
   $$K_{ss} e_s^{n+1} = -K_{sf} u^{n+1}.$$
5. **Faraday update:**
   $$b^{n+1} = b^n - \Delta t\, C_f u^{n+1} - \Delta t\, C_s e_s^{n+1}.$$

Under the domain decomposition of `wakis.unstructured.mpi`, the free edges $\{0, \dots, n_{\text{free}}-1\}$ are
partitioned into disjoint owned index sets $O_u(0) \sqcup \dots \sqcup O_u(R-1)$ (sorted in increasing serial order on
each rank). Once the $P$-vectors $V^n = W^T u^n$ and $V_*^{n+1} = W^T u_*^{n+1}$ are known on all ranks, the rank-$P$
outer-product updates in steps 1 and 3 are strictly row-local on each rank's owned free edges $O_u(r)$:

$$r[O_u(r)] \leftarrow r[O_u(r)] + W[O_u(r), :]\, q_{\text{pre}}^n, \qquad
u^{n+1}[O_u(r)] = u_*^{n+1}[O_u(r)] - D_+^{-1}[O_u(r)] \odot \bigl(W[O_u(r), :]\, (\tfrac{1}{2}\Delta t\, g \odot \delta V^{n+1})\bigr).$$

No halo exchange is needed for $W$; the only communication introduced by the ports is the global reduction that forms
$V^n = W^T u^n$ in step 1 and $V_*^{n+1} = W^T u_*^{n+1}$ in step 3 (plus the analogous inner product in `apply_T`).

---

## 2. Deterministic Reduction Order and Bitwise Equality

### Floating-Point Non-Associativity of Per-Rank Partial Sums

For each port $p \in \{0, \dots, P-1\}$, let $I_p = (i_{p,0} < i_{p,1} < \dots < i_{p,m_p-1})$ be the support of column
$w_p$ in `solver.free`, sorted in strictly increasing serial free-edge index order. In the serial solver, `PortSet`
stores $W$ as a `scipy.sparse.csc_matrix` of shape $(n_{\text{free}}, P)$, so `W.T` is a `csr_matrix` of shape
$(P, n_{\text{free}})$ whose row $p$ has column indices $I_p$ in strictly increasing order. The serial inner product
`(W.T @ u)[p]` therefore evaluates the left-to-right IEEE-754 recurrence

$$V_p^{\text{ser}} = \bigoplus_{k=0}^{m_p - 1} \operatorname{fl}\!\bigl(W_{i_{p,k}, p}\, u_{i_{p,k}}\bigr),$$

accumulated in increasing serial free-edge index order $i_{p,0} < i_{p,1} < \dots < i_{p,m_p-1}$.

When the partition cuts across a port face or a lumped port path, the edges $I_p$ are split across multiple ranks
$r \in \{0, \dots, R-1\}$. If each rank first forms its own scalar partial sum

$$s_{r,p} = \bigoplus_{i \in I_p \cap O_u(r)} \operatorname{fl}\!\bigl(W_{i,p}\, u_i\bigr)$$

and the ranks combine $s_{0,p} \oplus s_{1,p} \oplus \dots \oplus s_{R-1,p}$ (via `MPI.Allreduce(..., MPI.SUM)` or fixed
rank-order summation), the resulting summation tree groups terms by owner rank $O_u(r)$ rather than by serial index
$i_{p,k}$ whenever $I_p$ is interleaved across ranks (or when the reduction order is permuted). By standard floating-point
error analysis (Higham 2002, §3.1, eq. 3.5), two summation orders of $m_p$ IEEE-754 binary64 products ($u_{\text{mach}} = 2^{-53}$)
can differ by up to

$$\bigl|\hat{V}_p^{(1)} - \hat{V}_p^{(2)}\bigr| \le 2\gamma_{m_p} \sum_{k=0}^{m_p - 1} |W_{i_{p,k}, p}|\,|u_{i_{p,k}}|,
\qquad \gamma_m = \frac{m u_{\text{mach}}}{1 - m u_{\text{mach}}},$$

which injects a $\sim 10^{-16}\text{–}10^{-14}$ perturbation into $\delta V^{n+1}$ at every step and destroys bitwise
equality (`assert_array_equal`) with the serial solver.

### Serial Free-Edge Order Reduction in `wakis/unstructured/mpi.py`

To guarantee bitwise equality with the serial run for any partition and any number of ranks $R$, `wakis/unstructured/mpi.py`
combines the port edge contributions in **serial free-edge order**:

1. **Setup (`_port_pieces`):** Let
   $$I_{\text{port}} = \bigcup_{p=0}^{P-1} I_p = (i_0 < i_1 < \dots < i_{m_{\text{port}}-1}) \subset \{0, \dots, n_{\text{free}}-1\}$$
   be the sorted union of serial free-edge indices with at least one non-zero entry in $W$, and define the compressed
   port matrix $W_{\text{port}} = W[I_{\text{port}}, :] \in \mathbb{R}^{m_{\text{port}} \times P}$ (`csc_matrix`).
   Because $I_{\text{port}}$ preserves the relative order of $I_p$ for every port $p$, row $p$ of the CSR matrix
   $W_{\text{port}}^T$ has the exact same non-zero values in the exact same order as row $p$ of $W^T$. Consequently,
   for any vector $u \in \mathbb{R}^{n_{\text{free}}}$ and its restriction $u_{\text{port}} = u[I_{\text{port}}]$,
   $$W_{\text{port}}^T u_{\text{port}} \equiv W^T u \quad \text{(bitwise in IEEE-754)}.$$
2. **Per-rank index maps:** For each rank $r$, let $I_{\text{port}}(r) = I_{\text{port}} \cap O_u(r)$. Rank $r$ stores:
   * `loc_idx`: the local positions of $I_{\text{port}}(r)$ inside rank $r$'s owned vector $u_{\text{own},r}$;
   * `port_pos`: the positions of $I_{\text{port}}(r)$ inside $I_{\text{port}}$, i.e. $I_{\text{port}}[\text{port\_pos}_r] = I_{\text{port}}(r)$;
   * `W_own`: the row slice $W_{\text{port}}[\text{port\_pos}_r, :]$ (`csc_matrix`) and `dinv_own`: $D_+^{-1}[I_{\text{port}}(r)]$.
3. **Deterministic gather–assemble–SpMV–broadcast (`_combine_port_reduction`):** At each port reduction stage
   (`"u_n"`, `"u_star"`, `"u_T"`):
   * Each rank $r$ extracts its owned port-edge values $u_r = u_{\text{own},r}[\text{loc\_idx}_r]$ (1D `float64` array of
     length $|I_{\text{port}}(r)|$, typically $O(10\text{–}10^2)$ doubles).
   * `comm.gather(u_r, root=root)` collects $[u_0, u_1, \dots, u_{R-1}]$ in fixed rank order on `root`.
   * On `root`, `_assemble_serial_port_vector` places each rank's slice into $u_{\text{port}}[\text{port\_pos}_r] = u_r$.
     Because $\{\text{port\_pos}_r\}_{r=0}^{R-1}$ partitions $\{0, \dots, m_{\text{port}}-1\}$, the assembled vector
     $u_{\text{port}}$ is ordered by strictly increasing serial free-edge index $I_{\text{port}}$ regardless of $R$ or
     the mesh partitioner.
   * `root` evaluates $V = W_{\text{port}}^T u_{\text{port}} \in \mathbb{R}^P$ via the same SciPy CSR SpMV kernel as the
     serial solver and broadcasts $V$ to all ranks with `comm.bcast(V, root=root)`.

### Two-Step Massless Faraday Update When Ports Are Attached

On extruded waveguide meshes (`extrude_section`), prism-diagonal cross-section edges have degenerate Voronoi dual faces
and belong to `solver.massless` ($72$ massless edges on the 4-layer WR-90 mesh). Without ports, `wakis.unstructured.mpi`
fuses the massless Schur complement into $C_{\text{eff}} = C_f - C_s K_{ss}^{-1} K_{sf}$ (`DeviceStepper` convention,
see `docs/physicsguide_sections/unstructured_mpi.md` §4). However, serial `UnstructuredFIT3D.one_step()` updates $b$ in
two separate steps:

$$b \leftarrow b - \Delta t\, (C_f u^{n+1}), \qquad
e_s^{n+1} = -K_{ss}^{-1}(K_{sf} u^{n+1}), \qquad
b \leftarrow b - \Delta t\, (C_s e_s^{n+1}).$$

Because the connected components of $K_{ss}$ are strictly local (§4 of `unstructured_mpi.md`), when `s.ports is not None`
and `s.massless.size > 0`, `_field_pieces` extracts for each rank $r$ the local sub-blocks $C_{s,\text{loc}}$,
$K_{sf,\text{loc}}$, and $K_{ss,\text{loc}}$ on the massless cluster edges incident to rank $r$'s owned faces $O_b(r)$
(extending the edge halo $H_u(r)$ by all free edges coupled to those clusters through $K_{sf}$). Each rank factors its
small local $K_{ss,\text{loc}}$ with `scipy.sparse.linalg.splu` at setup and executes the same two-step Faraday update
as `UnstructuredFIT3D.one_step()`. Every floating-point operation on $u$, $b$, probes, running field DFTs, and port
voltage/current records is therefore bit-for-bit identical to the serial run.

---

## 3. Exact Distributed Transpose (`apply_T`)

For the homogeneous linear step $x^{n+1} = L x^n$ on $x^n = (u^n, b^n)$, the exact transpose $\bar{x}^n = L^T \bar{x}^{n+1}$
derived in `docs/physicsguide_sections/unstructured_ports3d_adjoint.md` §2 runs in reverse stage order in
`_FieldRank.program_T`:

1. **`("port", "u_T")` and `_t1` (Woodbury, capacitive, and explicit port conductance transpose):** gathers
   $\bar{u}^{n+1}[I_{\text{port}}(r)]$ into serial free-edge order $I_{\text{port}}$, evaluates on `root`
   $$z = W_{\text{port}}^T (D_+^{-1} \bar{u}^{n+1}) \in \mathbb{R}^P, \qquad
   \bar{u}^* = \bar{u}^{n+1} - W_{\text{port}} \operatorname{diag}\!\left(\frac{\Delta t\, g}{2}\right) S_P^{-T} z, \qquad
   \bar{r} = \Delta t\, D_+^{-1} \bar{u}^*, \qquad
   w_y = W_{\text{port}}^T \bar{r},$$
   broadcasts $(z, w_y)$, and updates each rank's owned free edges with
   $$\bar{u}_{\text{cap}}^n = a_1 \odot (D_+^{-1} \bar{u}^*) - \frac{1}{2} W \operatorname{diag}(g)\, w_y.$$
2. **`("u", "ru")` and `_t2` (Ampere transpose):** exchanges the edge halo on $\bar{r}$ and computes on owned faces
   $$\bar{b}^n = \bar{b}^{n+1} + M_\nu C_f \bar{r}.$$
3. **`("f", "rf")` and `_t3` (Faraday and massless Schur transpose):** exchanges the face halo on $\bar{b}^n$ and
   computes on each rank's owned free edges
   $$\bar{u}^n = \bar{u}_{\text{cap}}^n - \Delta t\,(C_f^T \bar{b}^n) - K_{sf}^T K_{ss}^{-T}\bigl(-\Delta t\,(C_s^T \bar{b}^n)\bigr).$$

Because every sparse row product and port reduction matches `UnstructuredAdjointStepper.linear_step_T` in the same
order, `d.apply_T` reproduces `UnstructuredAdjointStepper.linear_step_T` bitwise (`np.testing.assert_array_equal`).

---

## 4. Verification and Mutation Analysis (`tests/test_111_unstructured_ports3d_paths.py`)

The MPI port test suite (`test_mpi_*` in `tests/test_111_unstructured_ports3d_paths.py`) and its summary generator
`compute_mpi_ports_summary()` evaluate three port configurations on the 4-layer extruded WR-90 waveguide mesh
($a \times b = 22.86 \times 10.16\text{ mm}$, $L = 12\text{ mm}$, $192$ tetrahedra, $n_{\text{free}} = 102$ free edges,
$n_{\text{massless}} = 72$ massless section edges, $\varepsilon_r = 2.2$ slab in layer 2, $N_t = 60$ steps, frequencies
$f \in \{9, 10, 11\}\text{ GHz}$) across $R = 2$ and $R = 4$ RCB partitions in both `scatter` and `replicated` modes:

* **`modal`:** two $\text{TE}_{10}$ `ModalPort`s (`p1` at $z = 0$ excited by a 10 GHz Gaussian-modulated sine pulse,
  `p2` at $z = 12\text{ mm}$ matched). The 18 port edges of each `ModalPort` are split $[10, 8]$ across ranks $[0, 1]$
  at $R = 2$ and $[6, 4, 5, 3]$ across ranks $[0, 1, 2, 3]$ at $R = 4$.
* **`lumped`:** two 50 $\Omega$ `LumpedPort`s (`lp1` along $x \in [0, a]$ at $y = b/2, z = z_1$ driven by the pulse,
  and `lp2` along $y \in [0, b]$ at $x = a/2, z = z_3$). The 4 free edges of `lp1` are split $[2, 2]$ across ranks
  $[0, 1]$ at $R = 2$ and $[2, 0, 2, 0]$ across ranks $[0, 2]$ at $R = 4$; the 2 free edges of `lp2` have per-rank
  counts $[2, 0]$ at $R = 2$ and $[1, 1, 0, 0]$ at $R = 4$.
* **`mixed`:** both $\text{TE}_{10}$ modal ports (`p1`, `p2`) plus an interior 50 $\Omega$ lumped port (`lp` at $z = z_2$,
  counts $[2, 0]$ at $R = 2$ and $[1, 1, 0, 0]$ at $R = 4$).

Two mutations (`_mutation` in `DistributedFIT` / `LocalCluster`) verify that the tests fail when either completeness or
reduction order is violated:

* **`drop_port_rank`:** omits the highest rank owning port edges from the port inner-product reduction, producing an
  $O(1)$ error in the fields and spectra.
* **`swap_port_order`:** computes per-rank scalar partial sums $s_r = W_{\text{own},r}^T u_{\text{own},r}$ on each rank
  and sums them in reverse rank order $r = R-1, \dots, 0$ instead of assembling $u_{\text{port}}$ in serial free-edge
  order $I_{\text{port}}$. Because the port edges are cut across ranks in all three cases, changing the IEEE-754
  summation order produces a non-zero $10^{-28}\text{–}10^{-13}$ discrepancy that fails `np.testing.assert_array_equal`.

Exact values from `out/mpi_ports.json` (`MPI_NT = 60`):

| Ranks $R$ | Case | Port edges per rank | Unmutated $\max|\Delta u|$, $\max|\Delta b|$, $\max|\Delta \hat{E}_{\text{probe}}|$, $\max|\Delta \hat{V}_p|$ | `swap_port_order` $\max|\Delta u|$ (rel.) / $\max|\Delta \hat{E}_{\text{probe}}|$ / $\max|\Delta \hat{V}_p|$ | `drop_port_rank` $\max|\Delta u|$ (rel.) / $\max|\Delta \hat{E}_{\text{probe}}|$ / $\max|\Delta \hat{V}_p|$ |
| :---: | :---: | :--- | :---: | :---: | :---: |
| $2$ | `modal` | `p1`: $[10, 8]$<br>`p2`: $[10, 8]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $8.438\times 10^{-15}$ ($2.078\times 10^{-15}$)<br>$1.965\times 10^{-24}$ / $5.170\times 10^{-26}$ | $5.442\times 10^{-1}$ ($1.340\times 10^{-1}$)<br>$1.149\times 10^{-9}$ / $4.312\times 10^{-11}$ |
| $2$ | `lumped` | `lp1`: $[2, 2]$<br>`lp2`: $[2, 0]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $2.515\times 10^{-17}$ ($2.787\times 10^{-15}$)<br>$5.626\times 10^{-27}$ / $2.258\times 10^{-28}$ | $1.630\times 10^{-2}$ ($1.806\times 10^{0}$)<br>$1.676\times 10^{-12}$ / $4.310\times 10^{-13}$ |
| $2$ | `mixed` | `p1`: $[10, 8]$, `p2`: $[10, 8]$<br>`lp`: $[2, 0]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $4.607\times 10^{-15}$ ($1.194\times 10^{-15}$)<br>$2.490\times 10^{-24}$ / $5.170\times 10^{-26}$ | $5.765\times 10^{-1}$ ($1.494\times 10^{-1}$)<br>$1.242\times 10^{-9}$ / $4.542\times 10^{-11}$ |
| $4$ | `modal` | `p1`: $[6, 4, 5, 3]$<br>`p2`: $[6, 4, 5, 3]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $6.217\times 10^{-15}$ ($1.531\times 10^{-15}$)<br>$2.482\times 10^{-24}$ / $5.170\times 10^{-26}$ | $5.390\times 10^{-1}$ ($1.327\times 10^{-1}$)<br>$1.137\times 10^{-9}$ / $4.266\times 10^{-11}$ |
| $4$ | `lumped` | `lp1`: $[2, 0, 2, 0]$<br>`lp2`: $[1, 1, 0, 0]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $2.515\times 10^{-17}$ ($2.787\times 10^{-15}$)<br>$5.626\times 10^{-27}$ / $2.258\times 10^{-28}$ | $2.737\times 10^{-4}$ ($3.032\times 10^{-2}$)<br>$1.450\times 10^{-13}$ / $1.364\times 10^{-15}$ |
| $4$ | `mixed` | `p1`: $[6, 4, 5, 3]$, `p2`: $[6, 4, 5, 3]$<br>`lp`: $[1, 1, 0, 0]$ | $0.0,\ 0.0,\ 0.0,\ 0.0$ | $6.883\times 10^{-15}$ ($1.784\times 10^{-15}$)<br>$2.068\times 10^{-24}$ / $7.755\times 10^{-26}$ | $5.501\times 10^{-1}$ ($1.425\times 10^{-1}$)<br>$1.181\times 10^{-9}$ / $4.699\times 10^{-11}$ |

---

## 5. Limits & Failure Modes

* **Two collective port reductions per time step.** Each step with ports performs two gather-plus-broadcast reductions
  (`"u_n"` before `_Capacitive.forward` and `"u_star"` after `_Capacitive.forward`, transferring $m_{\text{port}}$
  doubles to `root` and $P$ doubles back). While $m_{\text{port}}$ and $P$ are small compared to the volume unknowns,
  the two collectives add $2\alpha_{\text{MPI}}$ latency per step and synchronize all ranks twice per step.
* **Serial port setup and root-side reduction.** As in `wakis.unstructured.mpi` generally (`unstructured_mpi.md`
  §Limits & Failure Modes), `PortPlane` 2D eigenmodes (`te_modes`, `tm_modes`, `tem_mode`), `PortSet`, and the $P \times P$
  Woodbury inverse $S_P^{-1}$ are constructed on the serial solver before partitioning, and `W_port.T @ u_port` is
  evaluated on `root`. Port time histories (`V_hist`, `Is_hist`, `I_hist`) are recorded on `root` and broadcast on demand
  when `d.port_signals(k)` or `d.port_spectra(freqs, k)` is called.
* **Picklable source waveforms in `scatter` mode.** In `DistributedFIT(..., mode="scatter")`, `root` scatters the port
  metadata to all ranks (`_port_pieces` keeps the waveform callables `i_src` on `root` and sends `[None] * P` to
  non-root ranks, while `LumpedPort` wraps its Norton source in the top-level `_NortonSource` callable). If a custom
  subclass stores unpicklable state inside a piece sent to non-root ranks, `comm.scatter` raises `PicklingError`;
  `mode="replicated"` avoids pickling solver pieces.
* **Homogeneous binary and CPU requirement for bitwise equality.** Bitwise identity (`assert_array_equal`) between
  serial and MPI runs requires that `root` and all ranks execute the same compiled SciPy CSR/CSC kernels on the same CPU
  instruction set. Heterogeneous ranks with different FMA contraction behavior can differ in the last bit.
* **Unsupported combinations (`NotImplementedError`).** Ports require `3D` Voronoi Hodge (`hodge="voronoi"`) without
  spherical PML (`s._pml is None`). Combining ports with `hodge="whitney"` or `PML3D` raises `NotImplementedError`.

---

## References

* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002).
  doi:10.1137/1.9780898718027
* W. W. Hager, "Updating the inverse of a matrix," *SIAM Rev.* 31(2), 221–239 (1989).
  doi:10.1137/1031049
* M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique," *Prog. Electromagn. Res.*
  32, 65–87 (2001). doi:10.2528/PIER00080103
* K. Kurokawa, "Power waves and the scattering matrix," *IEEE Trans. Microw. Theory Techn.* 13(2), 194–202 (1965).
  doi:10.1109/TMTT.1965.1125964
