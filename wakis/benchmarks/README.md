# `wakis.benchmarks`: shared benchmark suite for meshes and schemes

A small set of cases that every mesh/solver "arm" (Cartesian staircase or conformal, graded grids, subgridding, triangles, prisms, polygons, high-order, curvilinear NFIT, ...) can run and be scored on. Each case has the same parts:

- an exact or independently validated reference;
- a band and a source;
- metrics and accuracy targets;
- a cost estimate.

The single smooth n = 2 Mie cylinder used so far (B1) is too easy. It has no sharp features, no surface waves, no resonance and no near-field hot spot. B2–B4 add these one at a time.

```python
from wakis.benchmarks import get_case, list_cases, run, Arm
case = get_case("B2-Au-TE")
ref = case.reference          # exact spectra: q_ext, q_sca, q_abs (and enh for B3)
rec = run(case, my_arm, h)    # one simulation -> metrics + cost (JSON-serialisable)
```

Baseline results, figures and cost-to-target tables are in `studies/benchmarks/`. The shared result store is described in `wakis/benchmarks/results.py`.

## Cases

| case | what | reference | band | primary metrics |
|---|---|---|---|---|
| `B1-TM`, `B1-TE` | dielectric cylinder, n = 2, a = 1 µm (the earlier shared case; DFT window now 3000 fs, as 1000 fs leaves a 0.87 % TE / 0.14 % TM floor from a high-Q mode near x = 6.4) | Mie series (`wakis.analytic.mie_cylinder`) | x = k a = 2–10 (41 pts) | RMS relative error of Q_sca |
| `B2-Au-TE/TM` | gold nanocylinder, a = 20 nm, in a host of n = 2.0 | complex-m cylinder series with the **same** Rakić (1998) Lorentz–Drude ε(ω) as the solver | 450–900 nm (91 pts) | Q_ext/Q_sca/Q_abs NRMS; LSPR wavelength, peak height, FWHM |
| `B2-Ag-TE/TM` | silver nanocylinder, a = 20 nm, vacuum (sharp LSPR) | same | 300–500 nm (81 pts) | same |
| `B3-g10`, `B3-g5`, `B3-g2` | gold dimer, a = 20 nm, gap 10/5/2 nm, host n = 2.0, TE with E along the dimer axis, incidence along +y | exact 2D multi-cylinder T-matrix (`wakis.analytic.multi_cylinder`) | 500–1000 nm (101 pts) | gap \|E/E0\|² (mean over a 9-point stencil: centre + ring of radius g/4) peak and its wavelength; Q_ext NRMS |
| `B4` | centre-fed PEC dipole, L = 150 mm, a = 0.5 mm, gap 1 mm, free space | Hallén thin-wire MoM with the same finite gap (`wakis.analytic.thin_wire`) | 0.7–1.2 GHz (51 pts) | resonance frequency (X_in = 0), R_in at resonance, Z_in NRMS, broadside directivity |
| `B5-TM`, `B5-TE` | PEC cylinder, a = 1 µm | series in the PEC limit | x = 0.5–10 | Q_ext NRMS |

### Why these parameters

**B2: why gold needs a host of index 2.**
- A 2D cylinder in TE has its dipole (LSPR) resonance where ε_metal(ω) = −ε_b. A sphere resonates at −2ε_b.
- For gold in vacuum or water, ε = −1 … −1.8 falls at 450–500 nm, inside the interband absorption. The extinction peak is then absent (vacuum) or broad and weak (water: peak/valley prominence 1.3, Q ≈ 2.3).
- In a host of n = 2.0 (Si₃N₄-like), the condition ε = −4 is met at about 550 nm, in the Drude-dominated region. This gives a clear LSPR at 553 nm with Q ≈ 3.5 and prominence 2.1 (`test_b2_reference_has_clear_resonance`).
- Silver in vacuum is the sharp variant: 351 nm, Q ≈ 9.
- a = 20 nm keeps the particle quasi-static (k_b a ≤ 0.56). The surface fields then dominate, while the dynamic red shift and radiative damping are still visible in the series.

**B2 physics that makes it hard:**
- The skin depth of gold is about 25 nm, so the field penetrates only the outer part of the particle.
- The LSPR is a surface charge oscillation, with |E| largest just outside the surface. Its frequency is set by the *shape*.
- A staircased circle has corners and a jagged perimeter:
  - its polarisability is wrong (first-order geometry error);
  - the corners support extra, red-shifted local resonances.

  The staircase arms show both: a red-shifted shoulder and a 15 % → 2.7 % first-order Q_ext error.
- Subpixel (Kottke) averaging of a metal and its host gives mixed cells whose averaged ε(ω) can cross zero (parallel average) or have a pole (series average) somewhere in the band. The result is weak spurious resonances and non-monotone convergence in TE.

**B3: why the dimer gap.**
- Two particles coupled across a gap g form a bonding dipole mode. It red-shifts as the gap closes (Q_ext peak 626 → 666 → 738 nm for g = 10 → 5 → 2 nm at a = 20 nm).
- The field concentrates in the gap: |E/E0|² up to about 60, 240 and 1900.
- The gap field varies on the scale √(a g), not a. Meshing it is the whole difficulty, which is why this case is the one where meshing matters most.

**B4: why a thick-ish wire.**
- L/a = 300 (Ω = 2 ln(L/a) = 11.4) gives a resonance at L = 0.471 λ (thick-wire shortening).
- The reduced-kernel Hallén reference is accurate there: the spread over N = 150–300 segments is 0.09 % in f_res and 0.06 % in R_res.
- Classical anchors: the induced-EMF λ/2 limit 73.1 + j 42.5 Ω (Balanis, *Antenna Theory*, sec. 4.6 and eqs. 8-60). The MoM approaches it as a → 0, and the power radiated by the MoM current equals Re(V I*)/2 to 2e-3 (`tests/test_bench_thin_wire.py`).
- The same finite gap g is used in the MoM (uniform generator field over |z| < g/2) and in the lumped port, because the input reactance of a delta gap does not converge.

## Metrics

All defined in `wakis.benchmarks.metrics` and `case.evaluate`:
- **NRMS** = sqrt(mean (y − y_ref)²) / max |y_ref| over the band. Unlike the relative RMS of B1, it stays meaningful where a metal's Q is small.
- **Peak wavelength** (parabola through the three largest samples), **peak height**, **FWHM** (interpolated half-maximum crossings), as relative errors.
- **B3**: `enh_peak` (relative error of the maximum of the gap enhancement, the mean of |E/E0|² over `case.gap_points`: the gap centre plus 8 points at radius g/4) and `enh_lam` (its wavelength). `enh_centre_peak` is the pointwise centre value, a diagnostic only: it depends on where the point falls inside a cell.
- **Diagnostics**: `boundary_reflection` (|incoming| / |outgoing| harmonic power on the extraction circles) and the fit residual.
- **Cost to target**: `cost_to_target(records, metric, target)` interpolates (log-log) the unknowns, work (= unknowns × steps) and run time from which the error stays below the target on finer meshes. Default targets live in `case.targets` (e.g. B2: Q_ext NRMS 2/1/0.5 %, LSPR wavelength 0.5/0.2 %, height 2/1 %; B3: gap enhancement 5/2 %).

## Writing an arm (adapter interface)

Subclass `wakis.benchmarks.Arm`:

```python
class MyArm(Arm):
    name = "prism-graded"
    def supports(self, case): return hasattr(case, "cylinders")
    def resolutions(self, case): return [4e-9, 2e-9, 1e-9]      # any labels, coarse -> fine
    def run(self, case, h):
        ...  # build mesh from case.cylinders / case.materials / case.n_background
        return dict(samples=S, u0=U0, radii=list(case.circles),
                    e_gap=E, e0_gap=E0,                         # B3 only
                    cost=dict(unknowns=N, steps=n, dt=dt, run_s=t, setup_s=t0))
```

What a 2D scattering arm must deliver:

- **Geometry and materials.**
  - `case.cylinders`: centres and radii.
  - `case.materials[key]`: a `DispersiveMaterial` (use its poles in your ADE), `[eps_r, mu_r]` or `"pec"`.
  - `case.n_background`: the host fills the whole domain, including the absorber.
- **Excitation.**
  - A plane wave travelling at angle `case.direction`.
  - TE has E_inc = (−sin φ, cos φ), so H_z,inc = +|E|/η.
  - Use `case.source()` (Gaussian, band ±2σ) and run `case.t_run`.
- **Output.**
  - Scattered E_z (TM) or H_z (TE) as **Wakis DFT phasors** (kernel exp(−iωt)) on `circle_points(r, case.n_points)` for each r in `case.circles`, at `case.freqs`.
  - The incident phasor `u0` at the origin, at the same time levels. TE: shift H by −dt/2 if it lives at half steps.
  - TF/SF arms: the circles must lie outside the total-field box.
  - Instead of samples, an arm may return `c_ext`, `c_sca` widths [m] (e.g. from a flux box).
- **B3 gap field.**
  - `e_gap`: the total in-plane E phasors at `case.gap_points`, shape (n_freq, 9, 2).
  - `e0_gap`: the incident amplitude phasor (n_freq,).
- **Cost.**
  - `unknowns`: field degrees of freedom.
  - `steps`, and `run_s`: time-stepping wall time on one uncontended core.

What the B4 dipole arm must deliver:
- `z_in` over `case.freqs` (exp(+jωt): R + jX, the lumped-port V/I).
- Optionally `directivity` (broadside) with `d_freqs`.

`run(case, arm, h)` returns a JSON record. `bench_study.py run --publish` (or `wakis.benchmarks.results.publish`) uploads it to `gs://cloud-code-511302-autoresearch/bench-results/<case>/<arm>/<tag>.json`. `results.fetch(case)` returns all arms as a DataFrame.

### Baseline arms (in `arms/`)

| arm | file | notes |
|---|---|---|
| `cart-staircase-{uniform,graded}`, `cart-conformal-all-{uniform,graded}` | `cartesian2d.py` | `SolverFIT3D` on a one-cell periodic slab; TF/SF box; CPML; dispersive subpixel network for `'all'`; graded = h on the scatterers' bounding box + 2h, growth 1.2 to min(4h, λ_host/20) |
| `tri-{uniform,graded}` | `triangles2d.py` | `UnstructuredFIT2D`, Voronoi Hodge, `meshing.cluster_mesh` (interface rings + greedy Poisson-disk fill + polar PML rings); scattered-field plane wave with the dispersive source of `wakis.unstructured.dispersive_planewave`; radial CFS-PML |
| `cart-staircase-edges-graded` (reference baseline), `cart-staircase-stl-graded`, `cart-conformal-pec-graded` | `cartesian_dipole.py` | 3D dipole: graded transverse grid, h_z = min(g, 4 h_t), `LumpedPort`, CPML, `Near2FarMonitor`. `edges`: an E edge is PEC when its midpoint is in the wire. `stl` and `conformal-pec` currently inherit the legacy STL cell masks (any node inside), which thicken the wire by up to a cell; the conformal-PEC records are marked as affected by this bug until `fix/conformal-pec-legacy-mask` lands. `BENCH_GPU=1` runs the arm on CuPy |

## Cost estimates (one core, uncontended n2-standard-2)

| case | typical converged run | memory |
|---|---|---|
| B1 | 0.2–1 M unknowns, 1–20 min | < 1 GB |
| B2 | 5k–100k unknowns (graded), 4–400 s | < 0.5 GB |
| B3 | 10k–300k unknowns, 0.5–30 min (g = 2 nm needs h ≤ 1 nm) | < 1 GB |
| B4 | 0.7–3 M Cartesian cells, 20 ns window, 0.3–3 h | 1–4 GB (use n2-highmem-2) |

## Limits & failure modes

- **Reference validity.**
  - The B2/B3 references solve the continuous ε(ω). The ADE's discrete ε differs at second order in ω dt (≈1e-5 here).
  - The B4 MoM uses the reduced kernel. It needs segment lengths ≳ a, and its reactance drifts by about 1 Ω between N = 150 and 300 (reported as `reference_uncertainty`).
- **DFT window.** The window floor (1× vs 3× window) is ≤ 1e-5 on every B2/B3 metric (`studies/benchmarks/data/window_floor.md`).
- **De-embedding.** The boundary-reflection de-embedding is exact only for a rotationally symmetric scatterer centred on the origin (B1, B2, B5). B3 uses the raw outgoing coefficients, so arms need a good absorber (`boundary_reflection` < ~1e-3).
- **Weak lossless scatterers.** Their Q_ext − Q_sca is a small difference of large terms, set by phase errors. Use Q_sca or the volume absorption to check energy balance instead.
- **Outer-mesh errors.** In TM the large metal-scattered field makes the coarse outer region the dominant error source. In B2 TM the graded arms are worse than the uniform ones at equal h; refine h_out with h.
