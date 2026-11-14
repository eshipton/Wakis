#### Unstructured FIT on many ranks: domain decomposition, halo exchange and bitwise agreement

`wakis.unstructured.mpi` distributes the time loop of a configured `UnstructuredFIT3D` (Voronoi Hodge, with or without the spherical PML) or `UnstructuredFIT2D` (TM, TE) over MPI ranks. The solver classes are not changed: the serial solver is built once (on the root rank, or on every rank), its operators are cut into per-rank row blocks, and each rank advances only the unknowns it owns.

```python
s = UnstructuredFIT3D(mesh, materials, bc="abc")           # serial set-up (root)
s.add_point_dipole(r0, d, pulse); s.add_probe(pts, 2); s.add_dft(freqs)
d = DistributedFIT(s if comm.rank == 0 else None, comm)   # one per rank (mode="scatter")
d.run(n_steps)
u = d.gather("u")                     # full edge voltages on rank 0
E = d.energy()                        # Allreduce of the owned partial energies
lam = d.apply_T(adjoint_state)        # exact transpose of the source-free step

c = LocalCluster(s, n_parts=4)        # the same per-rank arithmetic in one process (tests)
```

##### 1. Partition and ownership

The cells are split into $`P`$ parts by `partition_cells(mesh, P, method)`:

* `"metis"`: multilevel k-way partitioning of the dual graph (cells connected through a shared face in 3D or a shared edge in 2D) with METIS (Karypis & Kumar 1998). It uses the optional `pymetis`, which is MIT-licensed bindings to the Apache-2 METIS.
* `"rcb"`: recursive coordinate bisection of the cell centroids along the longest extent (Berger & Bokhari 1987). It is pure NumPy, and part sizes are balanced to one cell per split level.
* `"rib"`: recursive inertial bisection along the principal axis of the centroid cloud (Simon 1991).

`"auto"` uses METIS when it is installed and RCB otherwise. Every unknown is owned by the **lowest part among the cells that contain it**: edges and faces in 3D, nodes and edges in 2D. This rule is deterministic, and it puts every owned unknown into a cell of its own rank.

##### 2. Owned rows and the halo

One leapfrog step of the 3D solver is

```math
\mathbf b \leftarrow \mathbf b - \Delta t\,A\,\mathbf u,\qquad
\mathbf r = C_f^{T}(M_\nu\mathbf b) - \mathbf j,\qquad
\mathbf u \leftarrow \mathcal U(\mathbf u, \mathbf r),
```

with $`A = C_f`$, or $`A = C_{\mathrm{eff}}`$ when there are massless edges (§4). $`\mathcal U`$ is the capacitive update, which is local per edge: capacitance, conductivity, Silver–Müller conductance and ADE poles. The 2D TM step has the same form with $`A = G`$, $`C_f^T \to G^T`$ and $`M_\nu \to`$ `stiff`. In 2D TE the roles of the two vectors are swapped.

Rank $`p`$ owns the index sets $`O_u(p)`$ and $`O_b(p)`$. It stores the rows $`A[O_b(p),:]`$ and $`C_f^T[O_u(p),:]`$, with their columns renumbered into the local vectors $`[\,\text{owned}\mid\text{halo}\,]`$. The halo is

```math
H_u(p) = \operatorname{cols}\!\big(A[O_b(p),:]\big)\cup\operatorname{cols}\!\big(C_f[O_b(p),:]\big)\cup\operatorname{cols}(P_{\mathrm{probe}}[\cdot,:])\setminus O_u(p),
\qquad
H_b(p) = \operatorname{cols}\!\big(C_f^T[O_u(p),:]\big)\cup\operatorname{cols}\!\big(A^T[O_u(p),:]\big)\setminus O_b(p).
```

The extra column sets ($`C_f`$ and $`A^T`$) are those read by the transposed step (§6). Each halo is computed from the actual sparsity of the operators, not from a geometric layer, so it is exact for any stencil, including the wider $`C_{\mathrm{eff}}`$ stencil. Halo values are refreshed by one neighbour exchange (`Isend`/`Irecv`/`Waitall`) per vector and step. The order inside the step is what makes each read current:

```
b[own] -= dt A_loc u_ext          reads u halo   (refreshed at the end of the previous step)
exchange b halo
r = Cf^T_loc (M_nu b)_ext - j     reads b halo   (refreshed just above)
u[own] = U(u[own], r)
exchange u halo
record probes (P_loc u_ext) and DFTs (owned u)
```

Every value a rank reads is either owned or a halo copy refreshed after the last update of its owner. The distributed step is therefore the serial step restricted to each rank's owned rows: no approximation, and no condition on the mesh or the partition. Sources are restricted to the owned entries, and dispersive groups to the owned edges of the group. Probe rows go to the lowest owner of their columns, and DFTs run on the owned entries.

##### 3. Why the result is bitwise equal to the serial run

The serial step evaluates

* CSR SpMVs: each row is summed in stored order starting from 0;
* $`C_f^T\mathbf x`$, a CSC SpMV of the transposed CSR: SciPy accumulates every output entry over the columns in increasing order, which is the sorted CSR row of $`C_f^T`$;
* elementwise operations.

Each local operator is the same CSR row, cut out with the entry order kept (`_ref_csr` turns a CSC transpose into the sorted CSR row, which has the same summation order). The elementwise updates see the same operands. **Every owned entry is therefore computed by exactly the same sequence of IEEE operations as in the serial run, whatever $`P`$ and the partition.** The step contains no reduction across ranks, so the result is bitwise identical, not merely equal to round-off: fields, probe samples and DFTs. The derivation assumes the same SciPy binary and CPU on all ranks: the FMA contraction of the compiled kernel is the same code path for serial and distributed rows. Different binaries or architectures may differ in the last bit.

The only quantities that are reduced across ranks are diagnostics. The global energy $`W = \tfrac12\sum_k m_k u_k^2 + \tfrac12\sum_f w_f b_f^2`$ is an `Allreduce` of the per-rank partial sums. It differs from the serial dot product only by the summation order, and because all terms are non-negative (Higham 2002, eq. 3.5)

```math
|W_{\mathrm{dist}} - W_{\mathrm{ser}}| \le 2\gamma_{n}\,W,\qquad \gamma_n = \frac{n u}{1-nu},\ u = 2^{-53},
```

with $`n`$ the number of terms. `gather("u")` and the probe signals are exact copies.

##### 4. The massless-edge elimination is local

Massless edges $`s`$ (degenerate Voronoi faces) are eliminated by the quasi-static constraint $`(C^TM_\nu\mathbf b)_s = 0`$: $`\mathbf e_s = -K_{ss}^{-1}K_{sf}\mathbf u`$, with $`K_{ss} = C_s^TM_\nu C_s`$. The question is whether $`K_{ss}^{-1}`$ couples unknowns across the partition, which would make a global solve necessary.

* $`(K_{ss})_{st} = \sum_f C_{fs}\,M_{\nu,f}\,C_{ft} \neq 0`$ only if the massless edges $`s`$ and $`t`$ share a face $`f`$ with $`M_{\nu,f} > 0`$.
* $`K_{ss}`$ is therefore block diagonal over the connected components (clusters) of the "share-a-face" graph of the massless edges, and $`K_{ss}^{-1}`$ has the same blocks.
* $`\mathbf e_s`$ on cluster $`k`$ depends only on the free edges that share a face with the cluster ($`K_{sf}`$), and $`b_f`$ reads $`\mathbf e_s`$ only through faces that contain a massless edge.

The fused operator $`C_{\mathrm{eff}} = C_f - C_s(K_{ss}^{-1}K_{sf})`$ (built by `gpu.massless_schur_operator`, block inverses computed densely) is therefore sparse. Its row for face $`f`$ reaches at most one cluster plus one face beyond the edges of $`f`$. The elimination is **local, and needs no communication beyond a wider halo**: §2 derives the halo from the sparsity of $`C_{\mathrm{eff}}`$. On `ball_mesh` meshes the clusters have at most 6 edges. A cluster that percolates across the mesh would make $`C_{\mathrm{eff}}`$ dense; clusters larger than 64 edges raise an error.

The distributed step uses $`C_{\mathrm{eff}}`$, so it is bitwise equal to `DeviceStepper(xp=numpy)`, which uses the same fused, sorted operator. It differs from the serial `one_step`, which solves with a sparse LU in every step, by round-off. The two are implementations of the same linear recurrence, and `gpu.agreement_bound` bounds their energy-norm difference:

```math
\|d_N\|_W \le 2K\gamma_m\sum_{n<N}\big\|\,|A|(|x_n|)\big\|_W .
```

Here $`|A|`$ is the absolute-value twin of the step, including the $`(1+\kappa)`$ factor of the LU forward error, $`m`$ is the operation depth, and $`K = \sqrt{(1+\rho)/(1-\rho)}`$ (see `unstructured_gpu.md`). The difference does not depend on $`P`$ (it is the same for 1 to $`P`$ ranks).

##### 5. The PML: stage by stage

The spherical CFS-PML (`pml3d.PML3D`) replaces the step by four fixed sparse stages on the extended state $`\mathbf x = [\mathbf u, \mathbf b, \mathbf d, \text{filter states}]`$:

```math
\mathbf x \leftarrow S_4\big(S_3S_2S_1\mathbf x + B\,\mathbf j\big).
```

Every entry of $`\mathbf x`$ belongs to an edge or a face: $`\mathbf d`$ and the edge-filter states to the PML edge, and the face-filter states to the PML face. It inherits that unknown's owner. Each stage gets its own halo (the columns of its owned rows), and the step makes four exchanges plus one for the probes. $`B`$ couples only entries of the same edge, so it needs no halo (this is checked at set-up). The argument of §3 applies stage by stage, so the distributed PML step is bitwise equal to `PML3D.apply`, i.e. to `one_step`.

##### 6. Exact transpose (adjoint) of the distributed step

The source-free step $`F:(\mathbf u,\mathbf b)\mapsto(\mathbf u',\mathbf b')`$ of §2, without dispersion, is

```math
\mathbf b' = \mathbf b - \Delta t A\mathbf u,\qquad \mathbf u' = (a_1\mathbf u + \Delta t\,C_f^TM_\nu\mathbf b')/\mathrm{den}.
```

Its transpose, from the adjoints $`(\bar{\mathbf u}',\bar{\mathbf b}')`$, is

```math
\bar{\mathbf a} = \bar{\mathbf u}'/\mathrm{den},\quad
\bar{\mathbf r} = \Delta t\,\bar{\mathbf a},\quad
\bar{\mathbf b} = \bar{\mathbf b}' + M_\nu C_f\bar{\mathbf r},\quad
\bar{\mathbf u} = a_1\bar{\mathbf a} - \Delta t\,A^T\bar{\mathbf b}.
```

That is two distributed SpMVs ($`C_f`$ on face rows, reading the $`\bar{\mathbf r}`$ halo on the u layout, and $`A^T`$ on edge rows, reading the $`\bar{\mathbf b}`$ halo), with the same halo spaces as the forward step (§2). For the PML, $`F^T = S_1^TS_2^TS_3^TS_4^T`$ and $`B^TS_4^T`$ gives the source sensitivity, one exchange per stage. These are the operations of the serial `UnstructuredAdjointStepper.step_T` and `PML3D.step_T` in the same order, so they too are bitwise equal to the serial adjoint (no massless edges). The dot-product identity is checked with the rigorous bound

```math
\big|\langle F\mathbf x,\mathbf y\rangle - \langle \mathbf x, F^T\mathbf y\rangle\big| \le (2\gamma_m + 2\gamma_N)\,|\mathbf y|^T|F|\,|\mathbf x|.
```

$`m`$ is the operation depth of the step, $`N`$ the state size, and $`|F|`$ the absolute-value twin. The random vectors are energy-balanced, $`\mathbf x = W^{-1/2}\mathbf z_1`$ and $`\mathbf y = W^{1/2}\mathbf z_2`$, with $`W`$ the capacitances, $`M_\nu`$ and the corresponding PML block weights. Raw $`\mathcal N(0,1)`$ vectors make $`|\mathbf y|^T|F||\mathbf x|`$ dominated by the $`M_\nu\sim1/\mu_0`$ entries, and a wrong coefficient of an $`\mathcal O(\Delta t)`$ term then stays below the bound. The first version of the test had exactly this problem, and the mutation was not detected.

##### 7. Communication volume and scaling model

Per step, rank $`p`$ receives $`|H_u(p)| + |H_b(p)|`$ doubles (3D, no PML) from at most a few neighbours. Each cut face adds the edges and faces of the one-cell layer around it to some halo, so the total halo volume is proportional to the partition surface $`S_P`$ (cut faces). For a 3D mesh of $`N`$ cells split into $`P`$ compact parts, $`S_P \approx c\,N^{2/3}P^{1/3}`$, so the communication-to-computation ratio per rank grows as $`(N/P)^{-1/3}`$. The time per step is

```math
t(P) \approx \frac{t_1}{P}\,\beta(P) + n_{\mathrm{ex}}\,(\alpha + \tau_{\mathrm{py}}) + \frac{8\,|H|_{\max}}{B_{\mathrm{link}}}.
```

* $`t_1`$: the serial SpMV-bound step.
* $`\beta(P)\ge1`$: the slowdown when $`P`$ ranks share the memory bandwidth of one machine. The SpMVs are bandwidth bound (CLAUDE.md rule 14).
* $`n_{\mathrm{ex}} = 2`$: exchanges per step (5 with the PML).
* $`\alpha`$: the MPI latency.
* $`\tau_{\mathrm{py}}`$: the Python overhead per program item, a few µs.

The measured numbers are in §8.

##### 8. Validation and measurements

**Tests** (`tests/test_103_unstructured_mpi.py`; serial pytest runs the in-process `LocalCluster`, and `mpiexec -n 2`/`-n 4` runs the real `DistributedFIT` in both `scatter` and `replicated` modes). The results are given in the report table below (Batch jobs `w2-mpi-tests-*`).

* Bitwise equality with `one_step` (fields, every probe, DFT) for 3D PEC without massless edges, 3D PML, 2D TM (PEC; ABC with conductivity and a Drude–Lorentz medium; Debye medium with an absorber) and 2D TE (ABC with an absorber), for 2, 3 and 4 parts with RCB and RIB.
* Massless edges (Debye–Lorentz with ABC; a PEC ball with a lossy shell): bitwise equal to the fused host `DeviceStepper`, and different from the LU `one_step` by $`0 < \|d\|_W \le \varepsilon_N`$.
* Energy `Allreduce` within $`2\gamma_n W`$.
* Mutation: one halo entry is never refreshed (3D, massless, PML, TM, TE). The relative difference must exceed $`10^{-6}`$; round-off would be about $`10^{-15}`$.
* Transpose: the dot-product identity holds within the bound above. One coefficient of one rank's transpose is perturbed by $`10^{-3}`$, which must exceed $`10\times`$ the bound. The transpose is bitwise equal to the serial adjoint.

**Study** (`studies/unstructured-mpi/`): `correctness.py` gives the per-step difference against the bound (figure `unstructured_mpi_correctness.png`), and `scaling.py` the strong and weak scaling and communication volume on one n2-standard-32 VM (figures `unstructured_mpi_*_scaling.png`, `unstructured_mpi_comm_volume.png`). The baseline is the same mesh stepped by the serial SciPy `one_step` in one process with one thread.

##### Limits & Failure Modes

* **Set-up is not distributed.** The serial solver (mesh, operators, Lanczos for $`\Delta t`$) is built on the root (`scatter`) or on every rank (`replicated`). Peak set-up memory is that of the serial run on the root (plus the pieces before the scatter), or that times $`P`$ in replicated mode on one machine. Only the time loop is distributed. A mesh that does not fit on one node cannot be run.
* **Single node measured.** The scaling study runs on one VM: the Batch tooling starts one VM per job, so multi-node MPI is not tested. On one node the SpMVs share the memory bandwidth, so strong scaling saturates before the core count. The study reports this as measured, not as a property of the method.
* **No overlap** of communication and computation. Interior rows could be updated while the halo is in flight; this is not implemented.
* **Python overhead** of a few µs per program item and per exchange limits strong scaling once a rank holds about $`10^4`$ cells.
* **Ownership rule.** "Lowest part" puts all interface unknowns on the lower rank: the owned-dof imbalance is up to a few % and the halo sizes are asymmetric. It is deterministic but not optimal.
* **Bitwise equality** needs the same SciPy build and CPU on all ranks (§3). With massless edges the reference is the fused elimination. The serial LU path differs by round-off within the derived bound.
* **Not distributed** (`NotImplementedError`): `hodge="whitney"` (a sparse solve with the consistent mass matrix in every step: it would need a distributed preconditioned CG), the 2D PML, LTS and plane waves, 2D TE with dispersive media, the transpose of dispersive steps, the 2D TE and absorber transposes, eigenmodes, and the GPU stepper (multi-GPU with NCCL is not done).
* **Source amplitudes** must be picklable in `scatter` mode (module-level functions or callable objects; no lambdas). In `replicated` mode every rank must build an identical solver (deterministic mesh and seeded Lanczos). The partition is broadcast from the root, so METIS randomness cannot make ranks disagree.
* **Massless clusters** larger than 64 edges raise an error (the dense block inverse would fill in, and the stencil would stop being local).

**References.** G. Karypis and V. Kumar, A fast and high quality multilevel scheme for partitioning irregular graphs, *SIAM J. Sci. Comput.* 20, 359–392 (1998), doi:10.1137/S1064827595287997. M. J. Berger and S. H. Bokhari, A partitioning strategy for nonuniform problems on multiprocessors, *IEEE Trans. Comput.* C-36, 570–580 (1987), doi:10.1109/TC.1987.1676942. H. D. Simon, Partitioning of unstructured problems for parallel processing, *Comput. Syst. Eng.* 2, 135–148 (1991), doi:10.1016/0956-0521(91)90014-V. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002), doi:10.1137/1.9780898718027. M. Clemens and T. Weiland, Discrete electromagnetism with the Finite Integration Technique, *PIER* 32, 65–87 (2001), doi:10.2528/PIER00080103.
