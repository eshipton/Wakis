### Fused GPU Kernels for 3D Cartesian Subgridding (`wakis/subgrid3d/gpu_fused.py`)

 Section 9e of `docs/physicsguide_sections/subgrid3d.md` showed that on an NVIDIA L4 GPU the array-slicing `BlockOps` backend (`wakis/subgrid3d/gpu.py`) and the CuPy CSR backend (`kernels="csr", backend="cupy"`) are launch-latency bound: on the uniform $H = 0.1\text{ m}$ and $H = 0.05\text{ m}$ loaded-cavity grids (`sg3d-lts-cost-d`), `BlockOps` takes $0.825\text{--}0.840\text{ ms/step}$ and `csr` takes $0.257\text{--}0.564\text{ ms/step}$, compared with $0.0291\text{--}0.416\text{ ms/step}$ for `SolverFIT3D(..., use_gpu=True, fused=True)` with per-step Python `one_step()` (and $0.0066\text{--}0.0070\text{ ms/step}$ at $H = 0.1\text{--}0.05\text{ m}$ when `SolverFIT3D` is replayed as 64-step CUDA graphs via `wakis.gpu_fused.FusedLoop` / `GraphRunner`; see §6). With a 2:1 or 3:1 refinement patch and local time stepping (LTS), `BlockOps` issues a separate CuPy slice kernel per block and per axis in `curl_all` and `curl_T_all`, separate CuPy CSR matrix-vector products for $P_h$, $P_h^T$, the LTS level operators $K_{F,l}, K_{C,l}, K_{F,l}^H$ and the CPML curl rows $K_f^{(d)}, K_e^{(d)}$, plus per-step temporary allocations (`self.xp.empty(self._nb)` in `SubgridLTSFIT3D._Kz`), bringing the per-step cost to $1.8215\text{--}2.8940\text{ ms/step}$ on an L4.

`wakis/subgrid3d/gpu_fused.py` (`FusedTables` and `FusedSubgridStepper`) replaces those slice and sparse launches with a fixed sequence of zero-allocation CUDA kernels compiled with `--fmad=false` and replayed in chunks of `graph_chunk` steps via `wakis.gpu_fused.GraphRunner`.

---

#### 1. Data Layout of the Concatenated Dense Block Arrays

A `SubgridMesh3D` with $n_{\text{blocks}}$ Cartesian blocks concatenates the 3D edge arrays $(E_x, E_y, E_z)$ of block $b \in \{0, \dots, n_{\text{blocks}}-1\}$ at global offset `M.edge_off[b]` ($n_E = \text{M.n\_edges}$ total all-edges) and the 3D face arrays $(F_x, F_y, F_z)$ at global offset `M.face_off[b]` ($n_F = \text{M.n\_faces}$ total all-faces). Within block $b$ of cell counts $\mathbf{n}_c^{(b)} = (n_{x}, n_{y}, n_{z})$:

* Edges along axis $a \in \{0, 1, 2\}$ have 3D shape $\mathbf{s}_e^{(b,a)}$ with $s_{e,a}^{(b,a)} = n_a$ and $s_{e,d}^{(b,a)} = n_d + 1$ for $d \ne a$, occupying the contiguous slice $[\text{eo}_a^{(b)}, \text{eo}_{a+1}^{(b)})$ in row-major ($C$-contiguous) order $(i, j, k)$.
* Faces normal to axis $a \in \{0, 1, 2\}$ have 3D shape $\mathbf{s}_f^{(b,a)}$ with $s_{f,a}^{(b,a)} = n_a + 1$ and $s_{f,d}^{(b,a)} = n_d$ for $d \ne a$, occupying $[\text{fo}_a^{(b)}, \text{fo}_{a+1}^{(b)})$ in row-major order $(i, j, k)$.

`FusedTables` packs these offsets and dimensions into a single `int32` metadata array `blk_meta` of stride 32 per block:

| `blk_meta[32 * b + ...]` | Contents | Description |
| :--- | :--- | :--- |
| `0 .. 3` | `eo[0 .. 3]` | Global all-edge component boundaries $[\text{eo}_a^{(b)}, \text{eo}_{a+1}^{(b)})$ |
| `4 .. 7` | `fo[0 .. 3]` | Global all-face component boundaries $[\text{fo}_a^{(b)}, \text{fo}_{a+1}^{(b)})$ |
| `8 .. 16` | `es[0 .. 2][0 .. 2]` | 3D edge shapes $\mathbf{s}_e^{(b,a)}$ for $a = 0, 1, 2$ |
| `17 .. 25` | `fs[0 .. 2][0 .. 2]` | 3D face shapes $\mathbf{s}_f^{(b,a)}$ for $a = 0, 1, 2$ |
| `26 .. 28` | `nc[0 .. 2]` | Cell counts $(n_x, n_y, n_z)$ of block $b$ |

Because $n_{\text{blocks}}$ is small ($1\text{--}3$ in typical subgrid hierarchies), the entire `blk_meta` table ($32 n_{\text{blocks}} \times 4\text{ B} \le 384\text{ B}$) stays in constant/L1 cache. A thread assigned to global face $f \in [0, n_F)$ or global all-edge $e_{\text{all}} \in [0, n_E)$ finds its block $b$ by scanning `fo[0]` or `eo[0]`, determines its component axis $a = (f \ge \text{fo}_1^{(b)}) + (f \ge \text{fo}_2^{(b)})$, and decodes $(i, j, k)$ from the local offset $\ell = f - \text{fo}_a^{(b)}$ via two integer divisions/moduli with $s_{f,1}^{(b,a)}$ and $s_{f,2}^{(b,a)}$.

---

#### 2. Fused Faraday Kernel (`sg_faraday`): One Thread per Face of All Blocks

In the Nédélec trace formulation (`docs/physicsguide_sections/subgrid3d.md` §3 and §8), every all-edge $e_{\text{all}} \in \{0, \dots, n_E - 1\}$ is either:

1. a **free edge** ($e_{\text{all}} = \text{free\_edges}[q]$ for a unique $q \in \{0, \dots, n_{\text{free}} - 1\}$), with weight $1$;
2. a **hanging interface edge**, whose voltage is a linear combination of at most two coarse parent free edges $(q_0, w_0)$ and $(q_1, w_1)$ from its row in $P_h = P_e[\text{hanging}]$ ($w_0 = 1/r$ when collinear with a parent edge; $w_0 = (1-t)/r, w_1 = t/r$ inside a parent face); or
3. a **dead edge** (PEC boundary, embedded PEC, or parent edge inside a refined box), whose row in $P_e$ is identically zero.

`FusedTables` folds $P_e$ into four arrays of length $n_E$: `src_i0`, `src_i1` (`int32`, initialized to $-1$) and `src_w0`, `src_w1` (`float64`), storing the column indices and weights of each edge's $\le 2$ parent free edges in the exact CSR column order of $P_h$. Evaluating an all-edge voltage from the free-edge vector $z$ on the device is the branch-minimal inline helper `eval_edge`:

$$(P_e z)_{e_{\text{all}}} = \begin{cases}
0, & \text{src\_i0}[e_{\text{all}}] < 0, \\
\text{src\_w0}[e_{\text{all}}]\,z[\text{src\_i0}[e_{\text{all}}]], & \text{src\_i1}[e_{\text{all}}] < 0, \\
\text{src\_w0}[e_{\text{all}}]\,z[\text{src\_i0}[e_{\text{all}}]] + \text{src\_w1}[e_{\text{all}}]\,z[\text{src\_i1}[e_{\text{all}}]], & \text{src\_i1}[e_{\text{all}}] \ge 0.
\end{cases}$$

When LTS is active ($L \ge 1$), the recursive step $\mathcal{R}_0 e$ only modifies free edges in $\mathcal{F}_1$ (`lts_levels[1]["uF"]`). Passing `z_f1 = uF` and the lookup array `f1_pos` ($n_{\text{free}}$ `int32`s mapping $q \in \mathcal{F}_1$ to its local index in $\mathcal{F}_1$, and $-1$ outside $\mathcal{F}_1$) allows `eval_free` to select $z_q = u_{\mathcal{F}_1}[\text{f1\_pos}[q]]$ when $\text{f1\_pos}[q] \ge 0$ and $e_q$ otherwise, avoiding a scatter kernel before Faraday.

In `sg_faraday`, thread $f \in \{0, \dots, n_F - 1\}$ decodes $(b_i, a, i, j, k)$, sets $b = (a + 1) \bmod 3$ and $c = (a + 2) \bmod 3$, computes the four bounding all-edge indices
$$e_{b,0} = \text{idx}_b(i, j, k), \quad e_{b,1} = e_{b,0} + \text{stride}_{b,c}, \quad e_{c,0} = \text{idx}_c(i, j, k), \quad e_{c,1} = e_{c,0} + \text{stride}_{c,b},$$
evaluates $v_{b,0}, v_{c,1}, v_{b,1}, v_{c,0}$ via `eval_edge`, and forms the discrete curl:
$$(K z)_f = (v_{b,0} + v_{c,1}) - (v_{b,1} + v_{c,0}).$$
Where a CPML auxiliary $\psi$ exists on face $f$ (`psi_slot0[f] >= 0` or `psi_slot1[f] >= 0`), the curl splits into the derivative along axis $b$, $(K^{(b)} z)_f = v_{c,1} - v_{c,0}$, and the derivative along axis $c$, $(K^{(c)} z)_f = v_{b,0} - v_{b,1}$. Ordering the two transverse axes as $d_{\text{lo}} = \min(b, c)$ and $d_{\text{hi}} = \max(b, c)$ updates $\psi$ and accumulates into $(Kz)_f$ in increasing axis order $d = 0, 1, 2$, followed by $b_f \leftarrow b_f - \Delta t\,(Kz)_f$.

---

#### 3. Fused Ampère Kernel (`sg_ampere`): One Thread per Free Edge

For an all-edge $e_{\text{all}}$ of direction $\text{dir}_e \in \{0, 1, 2\}$ at block-local grid index $(i, j, k)$ in block $b_i$, the transposed topological curl $C^T y$ (with $y = W_{\text{all}} \odot b$) receives contributions from the two face-normal directions $a \ne \text{dir}_e$. For each such $a \in \{0, 1, 2\}$, let $d = 3 - a - \text{dir}_e$ be the transverse derivative axis and let
$$f_{\text{curr}} = \text{idx}_a(i, j, k) \quad (\text{if } q_d < n_{c,d}), \qquad f_{\text{prev}} = \text{idx}_a(i, j, k) - \text{stride}_{a,d} \quad (\text{if } q_d \ge 1)$$
be the two faces of normal $a$ incident to $e_{\text{all}}$ inside block $b_i$. By the cyclic orientation of $(a, b, c)$ in `BlockOps.curl_T_all`:

* if $\text{dir}_e = (a + 1) \bmod 3$, then $f_+ = f_{\text{curr}}$ carries $+1$ and $f_- = f_{\text{prev}}$ carries $-1$;
* if $\text{dir}_e = (a + 2) \bmod 3$, then $f_+ = f_{\text{prev}}$ carries $+1$ and $f_- = f_{\text{curr}}$ carries $-1$.

The device helper `edge_curl_T` loops over $a = 0, 1, 2$ ($a \ne \text{dir}_e$), adds $+y_{f_+}$ then subtracts $-y_{f_-}$ into the edge accumulator $g$, and records the derivative-axis split $q_d$ (restricting to active faces where $W_{\text{all}}[f] \ne 0$, matching `SubgridCPML` which builds $K_e^{(d)}$ on `solver.faces`) into $q_{\text{lo}}$ ($d < a$) and $q_{\text{hi}}$ ($d > a$).

In `sg_ampere`, thread $q \in \{0, \dots, n_{\text{free}} - 1\}$:

1. Evaluates `edge_curl_T` on its own free edge $\text{free\_edges}[q]$ to obtain $\text{rhs}_q$, $q_{\text{lo}}$, $q_{\text{hi}}$.
2. Loops over its row $j \in [\text{pht\_ptr}[q], \text{pht\_ptr}[q+1])$ in the CSR transpose table $P_h^T$, calling `edge_curl_T` on each hanging child $\text{hang\_edges}[\text{pht\_col}[j]]$ and accumulating $w_j g_{\text{child}}$ into $s_{\text{hang}}$ (then $\text{rhs}_q \leftarrow \text{rhs}_q + s_{\text{hang}}$) and $w_j q_{\text{child},\text{lo}/\text{hi}}$ into $q_{\text{lo}}, q_{\text{hi}}$.
3. Updates any CPML auxiliary slots `phi_slot0[q]` ($d_{\text{lo}}$) and `phi_slot1[q]` ($d_{\text{hi}}$): $\phi_s \leftarrow b_{e,s}\phi_s + a_{e,s} q_{\text{lo}/\text{hi}}$, $\text{rhs}_q \leftarrow \text{rhs}_q + \phi_s$.
4. Subtracts the precomputed current-source sample `src_table[step * n_src_edges + src_slot[q]]` when `src_slot[q] >= 0`.
5. Updates $e_q \leftarrow e_q + (\Delta t \, D_q^{-1}) \, \text{rhs}_q$.

---

#### 4. Local Time Stepping Kernels (`sg_lts_face` and `sg_lts_edge`)

In the multi-level damped Chebyshev LTS recursion (`wakis/subgrid3d/lts.py`, `SubgridLTS._corr` and `_R`), level $m \in \{1, \dots, L\}$ advances its fine free edges $\mathcal{F}_m$ ($n_{F,m} = |\mathcal{F}_m|$) and active faces $\Phi_m$ ($n_{\Phi,m} = |\Phi_m|$) through $p_m - 1$ Chebyshev substeps using the restricted operators $K_{F,m}$ ($n_{\Phi,m} \times n_{F,m}$), $K_{C,m}$ ($n_{\Phi,m} \times |\mathcal{C}_m|$) and $K_{F,m}^H$ ($n_{F,m} \times n_{\Phi,m}$).

At initialization, `FusedSubgridStepper._build_lts` merges each level's $K_{F,m}$ and $K_{C,m}$ row-wise via `wakis.gpu_fused.merge_rows(K_F, K_C)` into a single `RowOp(..., max_width=0)` with row pointers `ptr`, split pointers `split`, column indices `col` and values `val`. Each substep $k \in \{1, \dots, p_m - 1\}$ at level $m$ launches exactly two kernels:

1. **`sg_lts_face` ($n_{\Phi,m}$ threads):** thread $i \in [0, n_{\Phi,m})$ sums $(K_{F,m} z)_i$ over $j \in [\text{ptr}[i], \text{split}[i])$ and $(K_{C,m} y_C)_i$ over $j \in [\text{split}[i], \text{ptr}[i+1])$, reading $z$ and $y_C$ directly through the prebuilt index maps (`F_root`/`iF`, `C_root`/`iC`, and `sub_pos` when $m < L$ so finer-level recursion output $u_{\mathcal{F}_{m+1}}$ is read in place without a scatter kernel), and writes
   $$w_{\Phi,i} = W_{\Phi_m,i}\,\bigl((K_{F,m} z)_i + \kappa_{m,k}\,(K_{C,m} y_C)_i\bigr).$$
2. **`sg_lts_edge` ($n_{F,m}$ threads):** thread $i \in [0, n_{F,m})$ sums $(K_{F,m}^H w_\Phi)_i$ over its CSR row in $K_{F,m}^H$, forms $X_{d,i} = (\tau_m^2 D_{\mathcal{F}_m,i}^{-1})\,(K_{F,m}^H w_\Phi)_i$, and advances the three-term recurrence
   $$d_{\text{new},i} = \bigl((2\delta_m d_{k,i} - 2\beta_m X_{d,i}) + 2 T_{m,k} y_{F,i}\bigr) - d_{\text{old},i},$$
   with $d_1 = y_F, d_{\text{old},1} = 0$ at $k = 1$ and $d_{\text{old},2} = y_F$ at $k = 2$. On the final substep $k = p_m - 1$, it writes the completed level-$m$ update $u_{F,i} = y_{F,i} + (\text{out}_m \, d_{\text{new},i} - y_{F,i})$ directly to the target buffer (scattering through `iF` when nested inside a coarser level).

**Kernel count per coarse step:** Let $R(m)$ be the number of kernel launches in `_launch_R_into_uF(m, ...)`. At the finest level $m = L$, `_launch_corr(L, ...)` runs $p_L - 1$ substeps of 2 kernels each, so $R(L) = 2(p_L - 1)$. For $m < L$, each of the $p_m - 1$ substeps of `_launch_corr(m, ...)` invokes `_launch_R_into_uF(m + 1, ...)` ($R(m+1)$ kernels) plus 2 kernels, followed by one final `_launch_R_into_uF(m + 1, ...)`:
$$R(m) = (p_m - 1)\bigl(R(m + 1) + 2\bigr) + R(m + 1) = p_m\bigl(R(m + 1) + 2\bigr) - 2.$$
By induction from $R(L) + 2 = 2 p_L$, the full LTS recursion $\mathcal{R}_0$ launches $R(1) = 2\prod_{l=1}^L p_l - 2 = 2(P - 1)$ kernels. Adding `sg_faraday` and `sg_ampere` gives **exact $2P = 2\prod_{l=1}^L p_l$ field-update kernels per coarse step** (2 for global $\Delta t$, 4 for a 2:1 patch, 6 for a 3:1 patch, and 12 for nested 2:1 + 3:1 patches), plus 1 counter kernel (`sg_step_add`) and 1 probe/DFT kernel (`sg_probe_dft`) when probes are present.

---

#### 5. Operation Order and 0-ULP Bitwise Equality

The CUDA module in `wakis/subgrid3d/gpu_fused.py` is compiled with `--std=c++17` and `--fmad=false`, disabling fused multiply-add contraction so every IEEE-754 `binary64` addition and multiplication rounds to nearest-even independently, as in NumPy on x86-64. Every sum and elementwise expression in `sg_faraday`, `sg_ampere`, `sg_lts_face`, `sg_lts_edge` and `sg_probe_dft` is ordered to match the NumPy `kernels="block"` path (`BlockOps` + `SubgridLTS._corr` + `SubgridLTSFIT3D.one_step`):

1. **Prolongation and curl in `sg_faraday`:** For a hanging edge with two parent free edges, `eval_edge` evaluates `w0 * z[i0] + w1 * z[i1]` in CSR column order of $P_h$, matching `self.Ph @ e` in `BlockOps.prolong`. The four edge voltages of face $f$ are combined as `(vb0 + vc1) - (vb1 + vc0)`, matching `F[...] = _sl(E[b], c, 0, -1) + _sl(E[c], b, 1, None); F -= _sl(E[b], c, 1, None) + _sl(E[c], b, 0, -1)` in `BlockOps.curl_all`.
2. **Transposed curl and restriction in `sg_ampere`:** `edge_curl_T` iterates over face-normal axes $a = 0, 1, 2$ ($a \ne \text{dir}_e$) in increasing order and within each $a$ applies `g += yp` (`f_plus`) followed by `g -= ym` (`f_minus`), replicating the exact pass sequence of `BlockOps.curl_T_all`. For free edges that are parents of hanging edges, `s_hang` accumulates `pht_val[j] * gc` in the CSR row order of `PhT @ g[self.hang]` and then adds `rhs += s_hang`, matching `r = g[self.free].copy(); r += self.PhT @ g[self.hang]` in `BlockOps.restrict`. The final scaling `(dt * Dinv[f_idx]) * rhs` matches Python's left-to-right `dt * self._Dinv * rhs`.
3. **CPML auxiliary updates:** Both `sg_faraday` ($\psi$) and `sg_ampere` ($\phi$) apply the two transverse derivative axes in increasing order $d_{\text{lo}} < d_{\text{hi}}$, matching `for d, p in enumerate(self._pml):` in `SubgridLTSFIT3D.one_step`.
4. **LTS Chebyshev substeps:** `RowOp(..., max_width=0)` keeps $K_{F,m} \mid K_{C,m}$ and $K_{F,m}^H$ in CSR format so `sg_lts_face` and `sg_lts_edge` sum each row in CSR column order (and avoid out-of-bounds ELL padding when $n_{\Phi,m} > |\mathcal{C}_m|$). In `sg_lts_edge`, `xd = (tau2 * Dinv[i]) * kh`, `d_new = (((2.0 * delta) * dk - (2.0 * beta) * xd) + (2.0 * Tk) * yf) - dold`, and `u = yf + (out_scale * d_new - yf)` match the parenthesization of `SubgridLTS._corr`.
5. **Sources, probes and DFT phasors:** Host-evaluated source waveforms are not CUDA-graph safe, and device `sincos` (`libdevice`) can differ by $1\text{--}2\text{ ULP}$ from host glibc `sincos` in `np.exp`. `FusedSubgridStepper` precomputes the source table at `t_half = (st * dt) + 0.5 * dt` and the DFT phasors `np.exp((-1j * 2 * np.pi * freqs) * (st * dt))` on the host, and `sg_probe_dft` evaluates `(p_w0 * e[i0] + p_w1 * e[i1]) / p_L` and `acc[f] + (cv * ph[f]) * cdt`.

With these choices, both single operator applications (G1, G2) and 500-step trajectories with sources, probes, DFTs, CPML and nested 2:1 + 3:1 LTS (G3) are **0-ULP bitwise equal** (`np.array_equal == True`) to the NumPy `kernels="block"` reference.

---

#### 6. Roofline Model and GPU Timings

Let $n_F = \text{M.n\_faces}$, $n_E = \text{M.n\_edges}$, $n_{\text{free}}$ be the number of free edges, $n_{\text{hang}}$ the number of hanging edges, and $U = n_{\text{free}} + n_{\text{faces}}$ the number of active electromagnetic unknowns. In the compulsory-traffic (cache-resident working-set) model implemented in `studies/subgrid3d/gpu/bench_fused.py` (`bytes_per_step_model`):

* **Faraday (`sg_faraday`):** reads and writes $b$ ($16 n_F\text{ B}$), reads the edge-source table `(src_i0, src_i1, src_w0, src_w1)` ($24 n_E\text{ B}$) and the free-edge voltages $e$ ($8 n_{\text{free}}\text{ B}$), plus $4 n_{\text{free}} + 8|\mathcal{F}_1|\text{ B}$ when LTS is active and $8 n_F + 32 n_\psi\text{ B}$ when CPML is active:
  $$B_{\text{Faraday}} = 16 n_F + 24 n_E + 8 n_{\text{free}} + \mathbb{I}_{\text{LTS}}(4 n_{\text{free}} + 8|\mathcal{F}_1|) + \mathbb{I}_{\text{CPML}}(8 n_F + 32 n_\psi).$$
* **Ampère (`sg_ampere`):** reads `free_edges` ($4 n_{\text{free}}\text{ B}$), $b$ and $W_{\text{all}}$ ($16 n_F\text{ B}$), $D^{-1}$ ($8 n_{\text{free}}\text{ B}$), reads and writes $e$ ($16 n_{\text{free}}\text{ B}$), plus the $P_h^T$ CSR table ($4(n_{\text{free}}+1) + 4 n_{\text{hang}} + 12\,\text{nnz}(P_h^T)\text{ B}$) when hanging edges are present and $8 n_{\text{free}} + 32 n_\phi\text{ B}$ when CPML is active:
  $$B_{\text{Ampère}} = 28 n_{\text{free}} + 16 n_F + \mathbb{I}_{\text{hang}}\bigl(4(n_{\text{free}} + 1) + 4 n_{\text{hang}} + 12\,\text{nnz}(P_h^T)\bigr) + \mathbb{I}_{\text{CPML}}(8 n_{\text{free}} + 32 n_\phi).$$
  On a uniform PEC grid without hanging edges or CPML ($n_F \approx n_E \approx n_{\text{free}} \approx U/2$), $B_{\text{Faraday}} + B_{\text{Ampère}} \approx 92\,(U/2) = 46\text{ B/unknown}$ ($47.8\text{--}49.9\text{ B/unknown}$ including boundary edges/faces on the $H = 0.05\text{--}0.1\text{ m}$ cavity).
* **LTS Chebyshev substeps (`sg_lts_face` + `sg_lts_edge`):** each substep at level $l$ moves
  $$B_{\text{LTS,face}}^{(l)} = 24|\Phi_l| + 12\,\text{nnz}(K_{F,l}\mid K_{C,l}) + 8(|\mathcal{F}_l| + |\mathcal{C}_l|), \qquad B_{\text{LTS,edge}}^{(l)} = 44|\mathcal{F}_l| + 8|\Phi_l| + 12\,\text{nnz}(K_{F,l}^H),$$
  and level $l$ runs $\bigl(\prod_{m < l} p_m\bigr)(p_l - 1)$ substeps per coarse step, giving total compulsory traffic
  $$B_{\text{step}} = B_{\text{Faraday}} + B_{\text{Ampère}} + \sum_{l=1}^L \left(\prod_{m=1}^{l-1} p_m\right)(p_l - 1)\bigl(B_{\text{LTS,face}}^{(l)} + B_{\text{LTS,edge}}^{(l)}\bigr), \qquad t_{\text{roofline}} = \frac{B_{\text{step}}}{\text{BW}_{\text{copy}}}.$$

On the NVIDIA L4 in this worktree (`bench_fused.py --quick`, `steps=200`, `graph_chunk=64`), a 20M-element `float64` device-to-device copy (`320,000,000 B` moved per iteration) measures $\text{BW}_{\text{copy}} = 230.04\text{ GB/s}$ (`1.3911 ms` per copy).

| Case (`bench_fused.py`) | Unknowns $U$ | Kernels / step | $B_{\text{step}}$ [B] ($B/U$ [B]) | Roofline [ms/step] | `subgrid_block` [ms/step] | `subgrid_csr` [ms/step] | `subgrid_fused` (direct) [ms/step] | `subgrid_fused` (CUDA graph) [ms/step] | Speedup vs `block` / `csr` | Full-study Batch (`bench.json`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `uniform` $H=0.1\text{ m}$ | 2,716 | 2 | 135,424 (49.9) | 0.0006 | 0.8249 | 0.2603 | 0.0415 | 0.0075 | $110.1\times$ / $34.7\times$ | `TBD(bench.json:uniform_H0.1000_r1)` |
| `uniform` $H=0.05\text{ m}$ | 22,336 | 2 | 1,068,736 (47.8) | 0.0046 | 0.8201 | 0.2627 | 0.0480 | 0.0077 | $107.1\times$ / $34.3\times$ | `TBD(bench.json:uniform_H0.0500_r1)` |
| `uniform` $H=0.025\text{ m}$ | `TBD(bench.json:uniform_H0.0250_r1:unknowns)` | 2 | `TBD(bench.json:uniform_H0.0250_r1:bytes_per_step)` | `TBD(bench.json:uniform_H0.0250_r1:roofline_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_block_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_csr_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_fused_direct_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0250_r1:speedup)` | `TBD(bench.json:uniform_H0.0250_r1)` |
| `uniform` $H=0.0125\text{ m}$ | `TBD(bench.json:uniform_H0.0125_r1:unknowns)` | 2 | `TBD(bench.json:uniform_H0.0125_r1:bytes_per_step)` | `TBD(bench.json:uniform_H0.0125_r1:roofline_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_block_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_csr_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_fused_direct_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0125_r1:speedup)` | `TBD(bench.json:uniform_H0.0125_r1)` |
| `global` $H=0.1\text{ m}, r=2$ | 7,881 | 2 | 441,300 (56.0) | 0.0019 | 1.8215 | 0.2634 | 0.0393 | 0.0171 | $106.4\times$ / $15.4\times$ | `TBD(bench.json:global_H0.1000_r2)` |
| `lts` $H=0.1\text{ m}, r=2$ | 7,881 | 4 | 1,071,192 (135.9) | 0.0047 | 2.5349 | 1.0616 | 0.0695 | 0.0270 | $94.0\times$ / $39.4\times$ | `TBD(bench.json:lts_H0.1000_r2)` |
| `lts` $H=0.1\text{ m}, r=3$ | 21,771 | 6 | 5,039,532 (231.5) | 0.0219 | 2.8940 | 1.3839 | 0.0949 | 0.0583 | $49.6\times$ / $23.7\times$ | `TBD(bench.json:lts_H0.1000_r3)` |
| `lts` $H=0.05\text{ m}, r=2$ | 63,966 | 4 | 8,253,012 (129.0) | 0.0359 | 2.5129 | 1.0195 | 0.0695 | 0.0334 | $75.2\times$ / $30.5\times$ | `TBD(bench.json:lts_H0.0500_r2)` |
| `lts` $H=0.05\text{ m}, r=3$ | `TBD(bench.json:lts_H0.0500_r3:unknowns)` | 6 | `TBD(bench.json:lts_H0.0500_r3:bytes_per_step)` | `TBD(bench.json:lts_H0.0500_r3:roofline_ms)` | `TBD(bench.json:lts_H0.0500_r3:subgrid_block_ms)` | `TBD(bench.json:lts_H0.0500_r3:subgrid_csr_ms)` | `TBD(bench.json:lts_H0.0500_r3:subgrid_fused_direct_ms)` | `TBD(bench.json:lts_H0.0500_r3:subgrid_fused_graph_ms)` | `TBD(bench.json:lts_H0.0500_r3:speedup)` | `TBD(bench.json:lts_H0.0500_r3)` |
| `lts` $H=0.025\text{ m}, r=2$ | `TBD(bench.json:lts_H0.0250_r2:unknowns)` | 4 | `TBD(bench.json:lts_H0.0250_r2:bytes_per_step)` | `TBD(bench.json:lts_H0.0250_r2:roofline_ms)` | `TBD(bench.json:lts_H0.0250_r2:subgrid_block_ms)` | `TBD(bench.json:lts_H0.0250_r2:subgrid_csr_ms)` | `TBD(bench.json:lts_H0.0250_r2:subgrid_fused_direct_ms)` | `TBD(bench.json:lts_H0.0250_r2:subgrid_fused_graph_ms)` | `TBD(bench.json:lts_H0.0250_r2:speedup)` | `TBD(bench.json:lts_H0.0250_r2)` |

| Uniform case (`bench_fused.py`) | Unknowns $U$ | `fit3d_csr` (per-step Python) [ms/step] | `fit3d_fused` (per-step Python) [ms/step] | `fit3d_fused_graph` (CUDA graph, `chunk=64`) [ms/step] | `subgrid_fused_direct` (`graph_chunk=0`) [ms/step] | `subgrid_fused_graph` (CUDA graph, `graph_chunk=64`) [ms/step] | `subgrid_fused_direct / fit3d_fused` | `subgrid_fused_graph / fit3d_fused_graph` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` $H=0.1\text{ m}$ | 2,716 | 0.1779 | 0.0297 | 0.0066 | 0.0415 | 0.0075 | $1.39\times$ | $1.14\times$ |
| `uniform` $H=0.05\text{ m}$ | 22,336 | 0.1792 | 0.0298 | 0.0070 | 0.0480 | 0.0077 | $1.61\times$ | $1.10\times$ |
| `uniform` $H=0.025\text{ m}$ | `TBD(bench.json:uniform_H0.0250_r1:unknowns)` | `TBD(bench.json:uniform_H0.0250_r1:fit3d_csr_ms)` | `TBD(bench.json:uniform_H0.0250_r1:fit3d_fused_ms)` | `TBD(bench.json:uniform_H0.0250_r1:fit3d_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_fused_direct_ms)` | `TBD(bench.json:uniform_H0.0250_r1:subgrid_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0250_r1:ratio_direct_vs_fit3d_py)` | `TBD(bench.json:uniform_H0.0250_r1:ratio_graph_vs_fit3d_graph)` |
| `uniform` $H=0.0125\text{ m}$ | `TBD(bench.json:uniform_H0.0125_r1:unknowns)` | `TBD(bench.json:uniform_H0.0125_r1:fit3d_csr_ms)` | `TBD(bench.json:uniform_H0.0125_r1:fit3d_fused_ms)` | `TBD(bench.json:uniform_H0.0125_r1:fit3d_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_fused_direct_ms)` | `TBD(bench.json:uniform_H0.0125_r1:subgrid_fused_graph_ms)` | `TBD(bench.json:uniform_H0.0125_r1:ratio_direct_vs_fit3d_py)` | `TBD(bench.json:uniform_H0.0125_r1:ratio_graph_vs_fit3d_graph)` |

On the uniform $H = 0.1\text{ m}$ and $H = 0.05\text{ m}$ grids (same grid and $\Delta t$, Event-timed over `steps=200` after warm-up), `SolverFIT3D(use_gpu=True)` measures `0.1779 ms/step` and `0.1792 ms/step` (`fit3d_csr`, per-step Python), `0.0297 ms/step` and `0.0298 ms/step` (`fit3d_fused`, per-step Python `one_step()`), and `0.0066 ms/step` and `0.0070 ms/step` (`fit3d_fused_graph`, CUDA-graph replay via `wakis.gpu_fused.FusedLoop` / `GraphRunner` with `chunk=64`). On the same uniform grids, `FusedSubgridStepper` measures `0.0415 ms/step` and `0.0480 ms/step` with direct launches (`graph_chunk=0`, $1.39\times$ and $1.61\times$ `fit3d_fused` per-step Python) and `0.0075 ms/step` and `0.0077 ms/step` with CUDA-graph replay (`graph_chunk=64`, $1.14\times$ and $1.10\times$ `fit3d_fused_graph`, and $107.1\text{--}110.1\times$ faster than `subgrid_block` and $34.3\text{--}34.7\times$ faster than `subgrid_csr`). On the 2:1 and 3:1 LTS meshes (`H = 0.1 m` and `H = 0.05 m`), `FusedSubgridStepper` with CUDA-graph replay takes `0.0270--0.0583 ms/step`, which is $49.6\text{--}94.0\times$ faster than `subgrid_block` (`2.5129--2.8940 ms/step`) and $23.7\text{--}39.4\times$ faster than `subgrid_csr` (`1.0195--1.3839 ms/step`). On `lts_H0.0500_r2` ($U = 63,966$, compulsory traffic $8,253,012\text{ B}$, which fits within the L4's $48\text{ MB}$ L2 cache), the measured `0.0334 ms/step` slightly beats the DRAM compulsory-traffic roofline `0.0359 ms/step`.

---

#### 7. Verification (`tests/test_123_subgrid3d_gpu.py`)

Every row in the table below comes from a `NUMBERS` line printed by `tests/test_123_subgrid3d_gpu.py`:

| check | measured | tolerance (origin) | mutation -> result |
| :--- | :--- | :--- | :--- |
| **G1 (CPU):** `FusedTables.apply_K` and `apply_KH` vs `BlockOps.K` and `BlockOps.KH` (`closed_mesh`, `_open_box` with CPML, `13x11x9` with 3:1 patch) + transpose dot-product | `K_ulp=0`, `KH_ulp=0` (`np.array_equal`), `dot_defect=7.67e-18` | `0 ULP` (`np.array_equal`, identical operation order); dot-product `< 1e-15` | `mutate="no_hanging"` -> rel. 2-norm `1.366e-01` ($K$), `1.315e-01` ($K^H$) on `closed_mesh`, `5.056e-02` on `_open_box`, `6.942e-02` on `13x11x9` ($> 10^{-3}$, §9f); `mutate="face_offset"` -> `1.526e+00` ($K$), `1.513e+00` ($K^H$) |
| **G2 (GPU):** `FusedSubgridStepper.apply_K` and `apply_KH` on device vs NumPy `BlockOps.K` and `BlockOps.KH` + transpose dot-product | `K_ulp=0`, `KH_ulp=0` (`np.array_equal`), `dot_defect=7.67e-18` | `0 ULP` (`np.array_equal`, `--fmad=false` + identical operation order); `<= 4 eps` fallback | `mutate="no_hanging"` on device -> `1.366e-01` ($K$), `1.315e-01` ($K^H$); `mutate="face_offset"` on device -> `1.526e+00` ($K$), `1.513e+00` ($K^H$) |
| **G3 (GPU):** 500 steps of `closed_mesh` (global $\Delta t$ & 2:1+3:1 LTS), `_open_box` with CPML (global $\Delta t$ & LTS), and `13x11x9` 3:1 LTS, with source + 2 probes + DFT; `step_vec_T` dot-product | `state_ulp=0`, `probe_ulp=0`, `dft_ulp=0`; `dot_closed_lts=1.24e-17`, `dot_open_cpml_lts=4.64e-18`; `kernels_global=2`, `kernels_2to1=4`, `kernels_3to1=6`, `kernels_2x3=12` | `0 ULP` (`np.array_equal`; empirical round-off bounds `6.994e-12`, `3.024e-11`, `3.830e-12`, `1.645e-11`, §9e); dot-product `< 1e-15` | `mutate="lts_kappa"` ($\kappa_k g$ dropped in `sg_lts_face`) -> rel. 2-norm `1.922e-01` ($\gg \text{tol} = 3.024 \times 10^{-11}$) |
| **G4 (GPU):** CUDA-graph replay (`graph_chunk=32`) vs direct launches (`direct_chunk=0`) over `steps=132` (`warmup_steps=32` + `alloc_steps=100`); CuPy memory pool `used_bytes()` over `alloc_steps=100` steps | `graph_vs_direct_ulp=0`, `probe_ulp=0`, `dft_ulp=0` (`np.array_equal` on state, probes, DFT); `used_bytes_before=6705152`, `used_bytes_after=6705152`, `used_bytes_delta=0` | `0 ULP` (`np.array_equal`); `used_bytes_delta == 0` (exact) | — |
| **G5 (GPU):** device `modified_energy` over 2000 LTS steps on `closed_mesh` at $0.95\,\Delta t_{\max}^{\text{LTS}}$ | `drift=2.887e-15` (`E0=8.908234e+10`) | `<= 50 eps sqrt(n) = 4.965e-13` (round-off random walk, §9f) | — |
| **G6 (CPU+GPU):** refusals for `dispersive_groups`, mismatched `n_state`, Bloch-periodic blocks, complex fields, and invalid `mutate` | `cpu_refusals=5`, `gpu_checked=1` | `NotImplementedError` (`ValueError` for invalid `mutate`) | — |

---

#### 8. Limits & Failure Modes

* **Real fields and non-Bloch boundaries only:** `FusedTables` and `FusedSubgridStepper` require `dtype == float` and non-periodic blocks (`any(b.periodic)` or complex fields raise `NotImplementedError`, falling back to the CSR path).
* **No dispersive media:** Solvers with a non-empty `dispersive_groups` attribute or a state length `n_state` different from `len(b) + len(e) + cpml.n_aux` raise `NotImplementedError` at construction.
* **Single GPU only:** All blocks and LTS levels reside on one CUDA device; multi-GPU / MPI domain decomposition across subgrid blocks is not implemented in `gpu_fused.py`.
* **`run(..., energy=True)` bypasses CUDA-graph replay:** Calling `FusedSubgridStepper.run(n_steps, energy=True)` bypasses `GraphRunner` even when `graph_chunk > 1`, executing the per-step Python loop so that $e^n$ can be copied into `_e_old` and `modified_energy(e^n, e^{n+1})` (which performs device inner products `cp.vdot` and host scalar readback `.item()`) can be evaluated after every coarse step.
* **FP64 throughput on Ada Lovelace (NVIDIA L4):** The L4's double-precision (`float64`) shader rate is $1/64$ of its FP32 rate. Because the FIT curl + trace + Hodge updates have low arithmetic intensity ($\lesssim 0.25\text{ FLOP/B}$), the kernels remain memory-bandwidth or launch-latency bound rather than FP64 compute bound.
* **Launch-latency floor on small meshes:** Even inside a captured CUDA graph, each kernel node incurs $\sim 3\text{--}4\text{ }\mu\text{s}$ of hardware dispatch overhead on an L4. On small meshes ($U \le 22,336$, where compulsory byte traffic is $135,424\text{--}1,071,192\text{ B/step}$ and the bandwidth roofline is $0.0006\text{--}0.0047\text{ ms/step}$), the measured step time (`0.0075--0.0270 ms/step` for `subgrid_fused_graph` and `0.0066--0.0070 ms/step` for `fit3d_fused_graph`) is set by the $2P$ kernel nodes per graph step rather than DRAM bandwidth; only for $U = 63,966$ (`lts_H0.0500_r2`, `0.0334 ms/step` vs `0.0359 ms/step` roofline) does memory traffic dominate.
