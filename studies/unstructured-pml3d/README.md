# Spherical CFS-PML for 3D tetrahedral FIT

`UnstructuredFIT3D(mesh, bc="pec", pml=dict(r0, r1, order=3, R=1e-8, kappa_max=1, alpha=0))` absorbs outgoing
waves in a shell r0 < r < r1, meshed by `wakis.unstructured.pml3d.ball_with_pml_mesh` /
`spherical_shell_mesh`. Derivation and limits: `docs/physicsguide_sections/unstructured_pml3d.md`.

![](../../docs/img/unstructured_pml3d_reflection.png)
![](../../docs/img/unstructured_pml3d_angle_layers.png)

## Setup

* **Mesh.** `ball_mesh` of radius 8 h (edge h), extruded radially in layers dr = h. The prisms are split into
  tetrahedra; the PML has N = 4, 6, 10 or 16 layers backed by PEC.
* **Source.** Derivative-of-Gaussian current moment, centre wavelength 16 h, so the spectrum covers λ = 10–32 h.
  The point dipole sits off the lattice nodes.
* **Reference.** The *same* ball and first layers, continued by 40 vacuum layers and a 16-layer PML. Its own
  echo arrives after the window. The reflection is the probe-field difference test − reference (all three E
  components, Whitney interpolation from *all* edge voltages). The interior discretisation and source-model error
  cancel exactly. All runs share one dt (0.0926 h/c, 0.9 × the most restrictive limit).
* **Normal case.** z-dipole near the centre; probes at r = 7 h, θ = 90°, 60°, 30°.
* **Oblique case.** Radial dipole 1 h below the PML surface; probes on the same sphere at separations
  ψ = 20°–150°. The incidence angle at the specular point is 48°–61°; a sphere of radius 8 h allows no more for
  this source depth.
* **Baselines on the identical ball.** First-order Silver–Müller (`bc="abc"`) and a bare PEC wall, both at r0.

## Results (`reflection.json`; Cloud Batch `pml3d-refl-v2`, n2-highmem-8, 18 runs in 158 s)

`max_t |E_test − E_ref| / max_t |E_ref|` (worst probe), and the worst spectral ratio over λ = 10–32 h:

| boundary | normal (time) | normal (spectral, worst λ) | oblique (time, ψ = 20°…150°) |
|---|---|---|---|
| PEC wall at r0 | 0.81–1.9 | 5.6 | 0.69–10.7 |
| 1st-order Silver–Müller at r0 | 0.14–0.22 | 0.30 | 0.46–0.66 |
| PML, 4 layers | 1.9e-2 | 2.3e-2 | 2.2e-2–7.2e-2 |
| PML, 6 layers | 1.4e-3 | 1.4e-3 | 4.1e-3–1.5e-2 |
| **PML, 10 layers** | **4.5e-4 (−67 dB)** | **5.2e-4** | **7.5e-5–2.0e-3 (≤ −54 dB)** |
| PML, 16 layers | 4.6e-4 | 5.1e-4 | 4.5e-5–1.5e-3 |
| PML, 10 layers, κ_max = 5, α = ω0/10 | 4.6e-4 | 7.0e-4 | 7.0e-5–2.0e-3 |
| PML, 10 layers, α = ω0/10 | 4.6e-4 | 5.2e-4 | 6.8e-5–1.8e-3 |

* **Against the target.** 10 layers beat the −40 dB target by 27 dB at normal incidence and by ≥ 14 dB obliquely.
  They are 300–500× better than first-order Silver–Müller on the same ball.
* **Layer count.** The reflection falls steeply from N = 4 to 10, by 40× in a factor 2.5 of N. This regime is
  dominated by the per-cell discretisation of the grading: σ_max dr / c = 9.2 at N = 4 and 3.7 at N = 10. I did
  not derive a closed-form exponent, and the N⁻² guess I started from is not supported by the data.
* **Floor.** From N = 10 to 16 the normal-incidence reflection does not change (4.5e-4). The maxima of the
  difference occur when the wave that crossed the whole layer and reflected off its PEC backing returns. The
  floor is therefore a property of the layer, not of the reference; its exact origin is open. The candidate is
  the quasi-static near-field content of the pulse (p(t) has a DC part, which a PML does not attenuate) reaching
  the PEC backing. At large ψ in the oblique case the path crosses the whole ball, and the reference's coarse
  outer layers may add to the floor.
* **CFS parameters.** κ_max = 5 and α > 0 change propagating-wave absorption by less than about 30 %, as in 2D.
* **Reference vs the exact dipole Green's function.** 14–59 % (field `ref_vs_exact`). This is the error of the
  point-dipole source model and of the interior discretisation at 7 h from a point source at 16 cells per
  wavelength. A Whitney point dipole puts its charge on one tetrahedron, which gives O(h/r) multipoles in the near
  field; on a lattice node the error was 2.4×. This error is why the identical-mesh reference is needed; it
  cancels in the reflection.

## Late-time behaviour (`stability.json`, Cloud Batch `pml3d-stab-v1`)

Three runs of 1e5 steps each (ball radius 5 h + 10 layers, 40 k tetrahedra; ~25 min each on n2-standard-4;
Cloud Batch `pml3d-stab-v2`, exit 0; the first attempt `pml3d-stab-v1` failed in code shipping, exit 4, and never ran):

| layer | max\|e\| / peak at 2 %, 25 %, 100 % of the run | growth (max last 10 % / max 40–50 %) | interior energy at end / peak |
|---|---|---|---|
| α = 0, κ = 1 | 1.2e-4, 1.1e-5, 3.7e-6 (slowly decaying quasi-static field held in the layer) | 0.51 | 1.5e-10 |
| α = ω0/10 | 1.3e-7 flat | 0.98 | 1.9e-14 |
| κ_max = 5, α = ω0/10 | 1.3e-7 flat | 0.99 | 1.9e-14 |

None of the runs grows. With α = 0 the layer keeps a quasi-static field (the 1/(iω) pole) that decays only slowly.
Some of that field leaks back: the interior energy climbs from 1e-12 to 2e-9 of the peak around ct ≈ 1400 h
before it decays. CFS α > 0 lets it relax within a few hundred steps; what remains is a flat 1.3e-7 residual on edges near the
source, which is not curl-free. It is not a PML effect: the field is bounded and lives in the interior. Its
nature (a weakly radiating local mode of the tetrahedral mesh?) is open.

![](../../docs/img/unstructured_pml3d_late_time.png)

## Tests (`tests/test_098_unstructured_pml3d.py`, ball radius 4 h, Cloud Batch `pml3d-test-v3`)

See the module docstring; the measured values are listed in the physics guide section.

## Reproduce

```bash
python pml3d_reflection.py run 8        # 18 runs, ~3 min on 8 cores (Cloud Batch)
python pml3d_reflection.py stability    # 3 x 1e5 steps
python pml3d_reflection.py plot
```
