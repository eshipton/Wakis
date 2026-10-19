# High-order time-domain schemes vs second-order FIT: Mie cylinder, metals and an electrically large cylinder

**Question.** In the cylinder studies, conformal Cartesian FIT and triangle FIT both converge at second order, and for x > 6 their error comes from bulk numerical dispersion: resonances shift. How many unknowns, how much work and how much time does a high-order scheme need to reach 1 %, 0.1 % (and 0.01 %) error, against the best second-order baselines?

**Answer (short).**
- **Nodal DGTD on curved triangles (p = 4–5) is a different accuracy class.** On the shared Mie benchmark it reaches 0.1 % with about 17k unknowns, against an extrapolated ~2M for graded triangle FIT. That is about 120× fewer unknowns and ~550× fewer unknown-steps, and it runs 29× (TM) and 55× (TE) faster on one core.
  - Against the best competitor *measured* at 0.1 %, the Cartesian (2,4) scheme with RK4, DG is 11× faster.
  - At the loose 1–2 % targets the picture is different. DG still needs 15× fewer unknowns, but its wall time is about equal to graded triangle FIT. The **(2,4) leapfrog** scheme is the fastest at 1 %, about 2× faster than the triangle baseline.
- **Curved geometry is essential.** Straight-sided DG at p = 4 is no better than FIT (it reaches 3–5 % at λ_d/1.5–2). The same holds on metals: on the Au nanocylinder (B2) curved DG reaches 0.021 % Q_ext NRMS with 8.9k unknowns (p = 3) and 0.002 % with 13.3k unknowns (p = 4), while straight-sided DG at p = 4 gives 1.6 % (TE) and 3.9 % (TM) at h = 10 nm (15.8k unknowns).
- **Error split.** For x > 6, 95–99 % of the Yee/FIT error is bulk dispersion. The interface error measured with the (2,4) scheme is only 0.09 % (TM) and 0.19 % (TE) at 28 cells per λ_d. So the high-order *stencil* alone gains about 10× before the second-order sub-pixel interface caps it, and TE (diagonal tensor) caps first.

## Methods (details and derivations: physics guide, "High-order time-domain schemes")

| scheme | module | order | interfaces | time | absorber |
|---|---|---|---|---|---|
| nodal DGTD, p = 1–5 | `wakis.highorder.DGTD2D` | p + 1 (space) | curved isoparametric elements (Szabó–Babuška blend) | LSRK(5,4) | CFS stretched-coordinate PML (ADE) or Silver–Müller ABC |
| (2,4) staggered FD | `wakis.highorder.FDTD24(order=4)` | 4 (space) | sub-pixel averaging (2nd order) | leapfrog or LSRK(5,4) | CPML |
| Yee (2,2) | `FDTD24(order=2)` | 2 | same averaging | leapfrog | CPML |

- **Validation of the FD module against FIT.** Yee in `FDTD24` reproduces the conformal Cartesian FIT baseline: 3.58 % vs 3.50 % RMS at res 14, and 2.45 % vs 2.40 % at res 20 (TM). The bulk/interface split therefore uses the same grid, the same averaging, the same source and the same extraction for (2,2) and (2,4).
- **Literature (verified metadata, see the guide for the list).**
  - Hesthaven & Warburton, J. Comput. Phys. 181 (2002), and the 2008 book (DGTD, upwind flux);
  - Warburton, J. Eng. Math. 56 (2006) (nodes);
  - Carpenter & Kennedy, NASA TM-109112 (1994) (LSRK);
  - Yefet & Petropoulos, J. Comput. Phys. 168 (2001), and Hwang & Cangellaris, Electromagnetics 23 (2003) ((2,4) schemes);
  - Zhao & Wei, J. Comput. Phys. 200 (2004) (high-order FD with interface matching);
  - Busch, König & Niegemann, Laser Photonics Rev. 5 (2011) (DGTD in nanophotonics);
  - Rapetti & Bossavit, SIAM J. Numer. Anal. 47 (2009) (high-degree Whitney forms).
- **Why not option (iii), high-order mimetic FIT.** High-degree Whitney forms exist (Rapetti & Bossavit 2009), but we found no verified *explicit, diagonal-Hodge* high-order edge-element scheme on triangles. Mass lumping is established for nodal (scalar) triangles (Cohen–Joly–Roberts–Tordjman 2001) and lowest-order edge elements (Elmkies–Joly 1997). Without lumping, every step needs a sparse solve. Decision D9 in `decisions.md`.

## B1: shared Mie benchmark (n = 2, a = 1 µm, x = 2–10, 1000 fs window)

![](figures/a_error_vs_cost.png)
**(a)** RMS error of Q_sca vs the exact series, plotted against unknowns, unknowns × RHS evaluations, and uncontended 1-core time. Markers are Wakis runs; dashed lines are the baselines from `../mie-cylinder-graded` (graded triangle FIT with PML or compact, graded conformal Cartesian). The DG curves flatten at 0.14 % (TM) and 0.87 % (TE), and that floor comes from the run window, not from the mesh (next paragraph).

**The run-window floor (new finding).**
- **Cause.** With the shared 1000 fs DFT window, the error against the exact series cannot go below a geometry-dependent floor. One high-Q TE resonance near x = 6.4 has not rung down by 1000 fs.
- **Evidence.** Converged DG (three resolutions agree to < 1e-5) plateaus at 0.140 % (TM) and 0.866 % (TE) in the compact geometry. In the other geometries the plateaus are 0.12–0.37 % (0.119 % TM / 0.368 % TE in the PML box, 0.212 % / 0.170 % in `cc`, 0.191 % / 0.238 % in `std`). With a 3000 fs window the same DG runs reach 0.012 % (TM) and 0.036 % (TE) against the exact series (`data_long/`).
- **Consequence.** The baselines' "1 % TE" numbers are partly this floor. The benchmark suite has since moved B1 to 3000 fs.
- **Our protocol.** For targets ≤ 0.1 % we measure every arm against a *converged windowed reference*: the finest DG p = 5 run with the same extraction circles, the same boundary and the same 1000 fs window (`dg5-compact`, `dg5-std`, `dg5-cc`, `dg5`). This removes the floor without rerunning the baselines. The geometry match for `tri-graded-pml` (radial PML) against `dg5-std` (ABC disk) is approximate, so its sub-0.5 % values carry about ±0.1 % uncertainty.

![](figures/g_windowed_error_vs_cost.png)
**(g)** The same comparison against the converged windowed reference. DG keeps converging to about 1e-6.

### Cost to reach a target

Values interpolate log-log between runs. `*` means extrapolated with the arm's last measured slope. `≤` means the coarsest run already beats the target. Time is the uncontended n2-standard-2 core time-stepping time.

| target (reference) | pol | arm | unknowns | unknowns × steps | time [s] |
|---|---|---|---|---|---|
| 1 % (exact series) | TM | tri-graded-pml (best 2nd-order) | 240k | 1.07e10 | 104 |
| | TM | cart-graded-compact | 362k | 1.67e10 | 531 |
| | TM | fd24 leapfrog | 120k | 2.4e9 | **58** |
| | TM | **DG p = 5 curved** | **≤ 15k** | **≤ 4.5e8** | ≤ 79 |
| 1 % (exact series) | TE | tri-graded-pml | 245k | 1.11e10 | 185 |
| | TE | fd24 leapfrog | 159k | 3.6e9 | **76** |
| | TE | **DG p = 4 curved** | **18k** | **4.9e8** | 117 |
| 0.1 % (windowed) | TM | tri-graded-pml | 2.1M* | 2.9e11* | 2600* |
| | TM | cart-graded-compact | 2.9M* | 4.1e11* | 12600* |
| | TM | fd24 (2,4) LSRK4 (measured) | 331k | 6.5e9 | 1020 |
| | TM | **DG p = 5 curved** | **17k** | **5.3e8** | **91** |
| 0.1 % (windowed) | TE | tri-graded-pml | 2.1M* | 2.9e11* | 6030* |
| | TE | fd24 (2,4) LSRK4 | 3.5M* (interface-capped) | 2.2e11* | 28300* |
| | TE | **DG p = 5 curved** | **17k** | **5.2e8** | **109** |
| 0.01 % (windowed) | TM/TE | DG p = 5 curved | 25–26k | 1.0e9 | 163–189 |

- DG takes 5 RHS evaluations per step, and the work column counts steps. In unknowns × RHS evaluations, the 0.1 % advantage over tri-graded-pml is ~110× instead of ~550×.
- Per unknown-step, DG costs 150–300 ns on an n2 core, against 17–35 ns for the numpy (2,2)/(2,4) leapfrog and ~19 ns for triangle FIT. That is about 10× more per unknown, offset by 15–120× fewer unknowns.
- All tables, per-run data and the full target table are in `data/summary.md` (exact series) and `data/summary_windowed.md` (windowed).

![](figures/b_qsca_dg4.png)
**(b)** Q_sca vs x: Wakis DGTD p = 4 (curved) at 2 elements per λ_d (28,665 unknowns), against the exact series. The remaining error is the 1000 fs window floor at the x = 6.4 TE resonance.

### Bulk vs interface error (Cartesian, same grid)

![](figures/f_error_split.png)
**(f)** Signed per-frequency errors against the windowed reference. e(fd24 LSRK4) estimates the interface error plus a small 4th-order bulk residual. e(Yee) − e(fd24 LSRK4) is Yee's bulk dispersion. For x > 6:

| pol | res | Yee bulk | interface (2,4) |
|---|---|---|---|
| TM | 14 | 4.55 % | 0.17 % |
| TM | 28 | 2.00 % | 0.09 % |
| TE | 14 | 6.10 % | 0.41 % |
| TE | 28 | 1.43 % | 0.19 % |

- **Bulk dominates second order.** The bulk error is 95–99 % of the Yee error at x > 6. A high-order *stencil* therefore gains 10–25× in error at equal h, until the interface takes over. In TE that happens at about 0.2–0.5 %, where the diagonal sub-pixel tensor converges at about first order.
- **Time error matters for (2,4).** With leapfrog at CFL 0.95 it dominates: res 14 TM gives 0.74 % with leapfrog, 0.26 % at CFL 0.3 and 0.23 % with LSRK4. Yoshida's symplectic 4th-order composition is unstable with the CPML, because its negative sub-step amplifies inside the absorber.
- **DG interface.** On DG the analogous split is curved vs straight-sided at equal p and h. Straight-sided p = 4 gives 2.9–5.4 % at λ_d/1.5–2, against 0.14–1.03 % (vs series) or 0.022–0.52 % (windowed) for curved.

## B1 with the suite's 3000 fs window (benchmark-suite arms, `../highorder-bench/data/B1`)

The arms run through `wakis.benchmarks`, all with the 3000 fs window: DG on a de-embedded ABC disk at the suite circles (1.9, 2.2 and 2.5 µm), and the suite's `tri-graded` (radial PML). Metric: `q_sca_rms_rel` against the exact series. The values are from 32 of the 40 runs; the 8 finest were still running when this was written.

| pol | arm | unknowns | time [s] | RMS error |
|---|---|---|---|---|
| TM | tri-graded (finest, 40 cells/λ_d) | 368k | 679 | 1.24 % |
| TM | DG p = 3, λ_d/2 | 22.6k | 259 | 0.83 % |
| TM | DG p = 4, λ_d/2 | 37.2k | 544 | 0.066 % |
| TM | DG p = 5, λ_d/1.5 | 40.1k | 536 | 0.016 % |
| TE | tri-graded (finest) | 369k | 1062 | 1.22 % |
| TE | DG p = 4, λ_d/2 | 37.2k | 443 | 0.044 % |
| TE | DG p = 5, λ_d/1.5 | 40.1k | 471 | 0.014 % |

- **DG floor.** DG flattens at about 0.012 %, which is the residual window floor at 3000 fs (compare `data_long/`).
- **tri-graded does not reach 1 % at 3000 fs** in the suite configuration: it sits at 1.2 % with 368k unknowns. At 1000 fs the graded study's tri-graded-pml reached 1 % at 240k (different h_out cap and PML). This plateau is flagged to the suite owner.
- **Headline at 3000 fs.** DG p = 4–5 reaches ≤ 0.07 % with about 40k unknowns in about 500 s, against 1.2 % for 368k unknowns in 680–1060 s for the suite's graded triangles. That is more than 17× lower error with 9× fewer unknowns, at about equal or lower time.

## B2 / B3: metals (benchmark suite, `../highorder-bench`)

Arms `dgtd-p1…p4` (curved) and `dgtd-p4-straight`, box with CFS-PML, Rakić Drude–Lorentz Au/Ag through ADEs driven by the total field. The records are published in the shared store (`gs://…/bench-results/<case>/dgtd-*`).

![](../highorder-bench/figures/B2_error_vs_unknowns.png)

![](../highorder-bench/figures/B3_error_vs_unknowns.png)

- **B2 (`B2-Au-TE`, `B2-Au-TM`, `B2-Ag-TE`, `B2-Ag-TM`, LSPR $a = 20\,\mathrm{nm}$), $Q_{\mathrm{ext}}$ NRMS (`../highorder-bench/data/B2/cost_to_target.md`).**
  - On `B2-Au-TE` at $h = a/1.5 = 13.33\,\mathrm{nm}$: curved DG $p = 2$ reaches $0.238\,\%$ with $5,328$ unknowns ($47\,\mathrm{s}$; $0.199\,\%$ on `B2-Ag-TE` in $52\,\mathrm{s}$); $p = 3$ reaches $0.021\,\%$ with $8,880$ unknowns ($110\,\mathrm{s}$ on Au, $98\,\mathrm{s}$ on Ag); $p = 4$ reaches $0.002\,\%$ ($0.003\,\%$ on Ag) with $13,320$ unknowns ($143$–$161\,\mathrm{s}$) and $\le 0.001\,\%$ at $h = 10\,\mathrm{nm}$ ($15,750$–$16,020$ unknowns).
  - Best second-order baselines on `B2-Au-TE`: `tri-uniform` reaches $0.122\,\%$ at $73,745$ unknowns ($55\,\mathrm{s}$) and $0.014\,\%$ at $516,254$ unknowns ($990\,\mathrm{s}$); `tri-graded` / `tri-graded-pml` reach $0.114\,\%$ at $69,018$ unknowns ($327\,\mathrm{s}$ / $242\,\mathrm{s}$); `hybrid-quadtri` reaches $0.166\,\%$ at $56,537$ unknowns ($287\,\mathrm{s}$); `cart-conformal-all-graded` reaches $0.576\,\%$ at $97,200$ unknowns ($207\,\mathrm{s}$).
  - Cost to the $0.1\,\%$ $Q_{\mathrm{ext}}$ NRMS target: on `B2-Au-TE` and `B2-Ag-TE`, only `tri-uniform` among the FIT baselines reaches $0.1\,\%$, requiring $87,971$ unknowns ($2.58 \times 10^9$ unknown-steps, $64\,\mathrm{s}$) on `B2-Au-TE` and $82,680$ unknowns ($2.33 \times 10^9$ unknown-steps, $95\,\mathrm{s}$) on `B2-Ag-TE`. Curved `dg2-pml` reaches $0.1\,\%$ with $5,959$ unknowns ($9.26 \times 10^7$ unknown-steps, $75\,\mathrm{s}$) on `B2-Au-TE` (**$14.8\times$ fewer unknowns**, **$27.9\times$ fewer unknown-steps**) and $5,901$ unknowns ($1.05 \times 10^8$ unknown-steps, $72\,\mathrm{s}$) on `B2-Ag-TE` (**$14.0\times$ fewer unknowns**, **$22.2\times$ fewer unknown-steps**), while `dg3-pml` is already at $0.021\,\%$ with $\le 8,880$ unknowns ($98$–$110\,\mathrm{s}$). On TM (`B2-Au-TM`, `B2-Ag-TM`), `dg2-pml` already meets $0.1\,\%$ at $h = 10\,\mathrm{nm}$ ($\le 6,300$ unknowns at $0.012\,\%$ on Au, $\le 6,408$ unknowns at $0.018\,\%$ on Ag), compared with $15,594$ (`hybrid-quadtri` on `B2-Au-TM`) and $10,872$ (`tri-graded` on `B2-Ag-TM`).
- **Order on metals.** No order loss is visible down to the error floor of about 1e-5. The interface is smooth (a circle), represented exactly to O(h^{p+1}), and the ADE is integrated by RK4 with no time-discrete permittivity mismatch.
  - The straight-sided control (`dg4-pml-straight`) gives $1.63\,\%$ (`B2-Au-TE`), $3.90\,\%$ (`B2-Au-TM`), $1.42\,\%$ (`B2-Ag-TE`), and $2.53\,\%$ (`B2-Ag-TM`) at $h = 10\,\mathrm{nm}$ ($15.8\text{k}$–$16.0\text{k}$ unknowns), reaches $1\,\%$ only at $21.6\text{k}$ (`B2-Ag-TE`), $23.5\text{k}$ (`B2-Au-TE`), and $49.2\text{k}$ (`B2-Au-TM`) unknowns, and never reaches $0.1\,\%$. For metals the interface geometry is the *whole* error, because a 20 nm particle has no bulk dispersion at λ/a ≈ 25.
  - Corners (sharp metal edges) would bring field singularities and reduce DG to the singularity's order. They are not tested here.
- **B3 (Au dimer, gaps $g \in \{2, 5, 10\}\,\mathrm{nm}$, `../highorder-bench/data/B3/cost_to_target.md`).**
  - Curved `dg2-pml` ($\le 8,028$–$8,496$ unknowns), `dg3-pml` ($\le 13,380$–$14,160$), and `dg4-pml` ($\le 20,070$–$21,240$) all meet both the $5\,\%$ and $2\,\%$ `enh_peak` targets and the $1\,\%$ `q_ext_nrms` target already at the coarsest mesh $h = a/2 = 10\,\mathrm{nm}$ ($0.64$–$1.99\,\%$ `enh_peak` for $p = 2$, $0.19$–$0.98\,\%$ for $p = 3$, $0.09$–$0.24\,\%$ for $p = 4$). Straight-sided `dg4-pml-straight` gives $2.27$–$12.03\,\%$ `enh_peak` across $h = 5$–$10\,\mathrm{nm}$ and never reaches the $2\,\%$ `enh_peak` target (reaching $5\,\%$ at $44,571$ unknowns on `B3-g10`, $66,809$ on `B3-g5`, and $\le 20,070$ at $4.13\,\%$ on `B3-g2`). At the $2\,\%$ `enh_peak` target, the best FIT baselines require $31,679$ (`tri-uniform`) or $39,938$ (`tri-graded`) unknowns on `B3-g10`, $8,981$ (`tri-graded`) or $17,126$ (`hybrid-quadtri`) on `B3-g5`, and $\le 34,756$ (`tri-graded`) on `B3-g2`.
  - The time step is set by the gap elements. At $h = 10\,\mathrm{nm}$, reducing the gap from $g = 10\,\mathrm{nm}$ to $g = 2\,\mathrm{nm}$ increases the step count by $2.6$–$2.7\times$ ($46,479 \to 125,011$ steps for $p = 4$; global RK, no local time stepping). See `../highorder-bench/README.md` and `../highorder-bench/data/B3/summary.md` for full tables.

## Showcase: electrically large lossy cylinder (x = 20–30)

**Why this case.** Dispersion error grows with the electrical size: the phase error accumulates over the path length, so second-order schemes need ever more cells per wavelength. A lossless n = 2 cylinder at x = 20–30 has whispering-gallery modes with enormous Q, and no finite window converges for them (see the floor above). We therefore give the cylinder σ = 5000 S/m. That is Im n ≈ 0.02 and Q < ~50 for every resonance, so a 200 fs window converges and the error measures the scheme. The exact series uses the same complex ε(ω). The domain is a disk (DG) or square (FD) of half-width 1.45 µm, which is 14 λ_0 across at x = 30.

![](figures/c_large_error_vs_cost.png)
**(c)** RMS error of Q_sca vs cost, x = 20–30.

![](figures/d_large_qsca.png)
**(d)** Q_sca of Wakis DGTD p = 4 at res 4 (632,070 unknowns) against the exact series; the RMS error is 0.0003 % in both TM and TE (0.018 % TM and 0.026 % TE already at res 2 with 170,055 unknowns).

| target | pol | arm | unknowns | time [s] |
|---|---|---|---|---|
| 1 % | TM | Yee (2,2) | 3.23M | 2990 |
| | TM | (2,4) leapfrog | ≤ 215k (0.62 % already) | **≤ 46** |
| | TM | DG p = 4 | ≤ 99k (0.09 % already) | ≤ 525 |
| 0.1 % | TM | Yee (2,2) | 31.9M* | 1.48e5* |
| | TM | (2,4) LSRK4 | 1.29M | 2820 |
| | TM | (2,4) leapfrog | 2.81M (TE: 3.75M*; res 32 at 3.45M gives 0.108 % in 3697 s) | 3400 (TE: 4170*) |
| | TM | **DG p = 4** | **≤ 99k** | **≤ 525** |
| 0.1 % | TE | (2,4) LSRK4 | 784k | 1680 |
| | TE | **DG p = 4** | **116k** | **424** |
| 0.01 % | TM/TE | DG p = 5 | 149–159k (brackets: 139–238k) | 952–964 (brackets: 673–2829) |

- The DG times come from final-code timing reruns (`ho-ltime`).
- At 0.1 % on the large cylinder, DG is 4–5.4× faster than the best (2,4) arm, and 280× (TM) to 400× (TE) faster than Yee (extrapolated).
- At 1 %, the (2,4) leapfrog is the clear winner: 46 s (TM) / 41 s (TE), against at least ~369–525 s for our coarsest DG run, which already overshoots to 0.09–0.17 %.

## Exact-adjoint and GPU assessment

| scheme | linear, transposable step? | adjoint in practice | GPU |
|---|---|---|---|
| DGTD + LSRK4 | yes: X' = A X + f(t). `operator_matrix()` assembles A exactly by coloured probing; a step's adjoint is the same LSRK with Aᵀ; dot test < 1e-12 (`test_operator_matrix_and_adjoint_dot_product`, mutation A↔Aᵀ fails) | 5 stage vectors per step to checkpoint or recompute (vs 1 for leapfrog); PML and ADE auxiliaries included in X | dense element GEMMs ([Dr;Ds] (Np × 3K), lift) plus gathers; arithmetic intensity grows with p, better suited to GPU than FIT's sparse memory-bound updates |
| (2,4) leapfrog | yes (difference operators and diagonal scalings, like Yee; CN loss is a linear symmetric-in-time map) | same structure as the existing FIT adjoint | same as Yee: 2× wider stencil, same bytes per cell |
| (2,4) LSRK4 | yes (as DG) | 5 stages | as Yee |

## Limits & caveats

- **Window floor.** The B1 results against the exact series are limited by the 1000 fs window; see the protocol above. The windowed reference for `tri-graded-pml` (radial PML) comes from a DG run with a de-embedded ABC disk at the same radii, an approximate geometry match (±0.1 %).
- **Extrapolation.** The baselines' 0.1 % values are extrapolated: the second-order arms do not reach 0.1 % in the measured ladders. Treat the ratios as ±30 %.
- **PML late-time growth.** Fine-mesh runs with the rectangular CFS-PML on the 1000 fs B1 window grow without bound late in the window and are excluded from the tables and plots: specifically `dg3` box at `res 6` (TM RMS $3.3 \times 10^{34}$, TE RMS $1.1 \times 10^{34}$) and `res 8` (TM RMS $8.0 \times 10^{51}$, TE RMS $5.6 \times 10^{51}$), and `dg4` box at `res 4` (TM RMS $3.0 \times 10^{22}$, TE RMS $2.4 \times 10^{21}$) and `res 6` (TM RMS $3.1 \times 10^{92}$, TE RMS $9.3 \times 10^{91}$). Diagnostic runs in `data_stab/` confirm that this is not a time-step / CFL stability limit: reducing the CFL factor from $0.8$ to $0.65$ (`dg4-c65_TM_4`, $92,350$ steps, RMS $3.0 \times 10^{22}$) and $0.5$ (`dg4-c50_TM_4`, $120,055$ steps, RMS $3.0 \times 10^{22}$; `dg3-c50_TM_6`, $129,977$ steps, RMS $3.3 \times 10^{34}$) still produces the same blowup, whereas the identical meshes without the PML (`dg3-nopml_TM_6` at $241,980$ unknowns / $81,234$ steps and `dg4-nopml_TM_4` at $164,070$ unknowns / $75,032$ steps, terminated by the Silver–Müller ABC on the box boundary) are completely stable and converge to $0.143\,\%$ RMS ($0.1429\,\%$ and $0.1427\,\%$). All compact-disk (ABC) runs across all orders and resolutions are likewise stable.
- **Mesh limits.** Elements with two curved faces are rejected, so h ≥ a/1.5 on a circle: the B2 runs at h = a failed by design. Circles only (no CAD). A gap narrower than h is forced to one element across.
- **Timing.** One core of n2-standard-2 spot machines; the same configuration varies by ±30 % between machines. The numpy implementations of DG and of the (2,4)/Yee schemes are not optimized, so a compiled or GPU kernel would change the time ratios (not the unknown or work ratios).
- **Single geometry.** Smooth convex scatterers only. Sharp corners (singular fields) would reduce the DG convergence rate, and the Cartesian interface treatment would lose more.

## Reproduce

```bash
python ho_study.py list                                   # B1 tasks (112)
python ho_study.py run --shard I --nshards 112 --outdir out      # Cloud Batch, one task per machine
python ho_study.py run --tasklist wref --shard I --nshards 10    # windowed references
python ho_study.py run --tasklist long --t-run 3000 ...          # window check
python ho_study.py run --tasklist large --t-run 200 ...          # showcase (52)
python ho_study.py plot                                   # figures + data/*.md
python plot_mesh.py                                       # figure e
```

Tests: `tests/test_055_highorder.py` (12 tests, each with the failure it catches):
- the reference element is exact;
- LSRK4 is 4th order (mutated coefficient → order < 2.5);
- DG converges at > p (square cavity);
- the central flux conserves energy while upwind dissipates;
- curved elements restore order > 3.5 on a circular cavity, with straight-sided > 100× worse;
- the probed operator equals the matrix-free one and passes the adjoint dot test (mutation fails);
- the PML is > 6× better than the ABC (σ = 0 mutation fails);
- the Drude cavity frequency is √(ω_p² + c²k²) (mutation: no current);
- the (2,4) scheme is 4th order and Yee 2nd order.

![](figures/e_curved_meshes.png)
**(e)** The curved DG meshes: B1 at h = λ_d/2 (637 triangles, 28,665 unknowns at p = 4), and the B3 dimer with a 5 nm gap.
