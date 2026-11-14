# Spherical CFS-PML for tetrahedral FIT

`UnstructuredFIT3D(mesh, bc="pec", pml=dict(r0=..., r1=..., ...))` absorbs outgoing waves in a spherical shell
r0 < r < r1, meshed with `wakis.unstructured.pml3d.spherical_shell_mesh` (or `ball_with_pml_mesh`). Code:
`wakis/unstructured/pml3d.py`; tests: `tests/test_098_unstructured_pml3d.py`; study: `studies/unstructured-pml3d/`.

## 1. Continuum: complex radial stretching

Following Chew & Weedon (1994) and Teixeira & Chew (1997), the PML is the analytic continuation of Maxwell's
equations to a complex radial coordinate

$$\tilde r(r) = r_0 + \int_{r_0}^{r} s_r(r')\,dr', \qquad s_r = \kappa_r + \frac{\sigma_r}{\alpha + i\omega},$$

which is the complex-frequency-shifted (CFS) stretch of Kuzuoglu & Mittra (1996) (time convention $e^{+i\omega t}$,
as Wakis; $\sigma$ in s$^{-1}$). The two angular metric factors become $s_t = \tilde r / r$. With a *uniform*
$\alpha$, $s_t$ keeps the CFS form

$$s_t = \kappa_t + \frac{\sigma_t}{\alpha+i\omega},\quad
\kappa_t = 1 + \frac{1}{r}\int_{r_0}^{r}(\kappa_r-1)\,dr',\quad \sigma_t = \frac{1}{r}\int_{r_0}^{r}\sigma_r\,dr'.$$

Stretched-coordinate Maxwell equations in $(r,\theta,\phi)$ are identical to the unstretched ones in an anisotropic
medium $\varepsilon\Lambda$, $\mu\Lambda$ with (Teixeira & Chew 1997)

$$\Lambda = s_r s_\theta s_\phi\,\mathrm{diag}(s_r^{-2}, s_\theta^{-2}, s_\phi^{-2})
= \mathrm{diag}\!\left(\frac{s_t^2}{s_r},\; s_r,\; s_r\right).$$

The medium is reflectionless for every angle and frequency in the continuum, and an outgoing wave
$e^{-ik\tilde r}$ decays as $\exp(-\int\sigma_r dr/c)$. With polynomial grading $\sigma_r = \sigma_{max}\xi^m$,
$\xi = (r-r_0)/d$, $d = r_1-r_0$, and a PEC wall at $r_1$, the normal-incidence round-trip design reflection is
$R_{th} = \exp(-2\sigma_{max} d/((m+1)c))$, so $\sigma_{max} = (m+1)c\ln(1/R_{th})/(2d)$ (default $R_{th}=10^{-8}$,
$m = 3$). $\kappa_r = 1+(\kappa_{max}-1)\xi^m$ damps evanescent fields; $\alpha > 0$ removes the $\omega\to0$
singularity of $1/(i\omega)$ (the CFS motivation of Roden & Gedney 2000).

## 2. Diagonal stretched Hodge on a layered tetrahedral shell

FIT unknowns: edge voltages $e = \int E\cdot dl$, face fluxes $b = \int B\cdot dA$, and their duals (flux $d$
through the Voronoi face of an edge, voltage $h$ along the Voronoi edge of a face). In an anisotropic medium the
diagonal Hodge $d_e = M_{\varepsilon,e}\,\lambda\,e_e$ is consistent only if the edge tangent $t$ is an
eigenvector of $\Lambda$ ($t\cdot\Lambda E = \lambda\, t\cdot E$); likewise each face normal. So every *coupling*
dof in the shell must be radial or tangential. The shell mesher guarantees this:

* **Frustum prisms.** Each boundary triangle $abc$ of the interior ball (all on $r = r_0$) is extruded radially to
  the layer radii; the prism between $r_k$ and $r_{k+1}$ is a frustum of the cone with apex at the centre. Its
  bottom and top circumcircles are coaxial, so **all six vertices lie on one sphere**, centred on the cone axis at
  $(r_k+r_{k+1})/(2\cos\theta_0)$ ($\theta_0$ the angular circumradius of $abc$).
* **Split into three tetrahedra** by the min-index diagonal rule of Dompierre, Labbé, Vallet & Camarero (1999),
  which is conforming between neighbours and never produces the untetrahedralizable cyclic configuration. The
  three tetrahedra share the prism's circumcentre.

Consequences (all checked by `test_shell_mesh_geometry`):

| object | geometry | dual measure | stretch factor |
|---|---|---|---|
| spoke (radial edge) | along $\hat r$ | dual face ⟂ $\hat r$, positive | $\lambda = s_t^2/s_r$: $e = (d/M_\varepsilon)\,s_r/s_t^2$ |
| chord on $r=r_k$ | $t\cdot(a+b)=0$: tangential at its midpoint | positive (Delaunay surface) | $\lambda = s_r$: $e = (d/M_\varepsilon)/s_r$ |
| quad diagonal | oblique | **zero** (the two halves of a cyclic planar quad share $c_Q$; the pieces cancel) | massless |
| sphere face on $r=r_k$ | normal through the circumcentre, radial | dual edge radial, length $dr/\cos\theta_0>0$ | $h = M_\nu b\, s_r/s_t^2$ |
| side face | plane through the centre: normal ⟂ every $\hat r$ in it | positive | $h = M_\nu b/s_r$ |
| inner prism face | oblique | **zero** (both sides end at the common circumcentre) | inert |

Because $\Lambda$ is isotropic in the tangent plane ($s_\theta = s_\phi$), the chords and side faces need not be
mutually orthogonal. The quad diagonals and inner faces have exactly zero measure up to round-off (~1e-17);
`_snap_degenerate_duals` sets them to exact zeros. Positivity is guaranteed in the shell for any radius by a
Delaunay surface triangulation (the convex hull of the sphere nodes), since the spoke and chord dual faces are
the (scaled, gnomonically projected) spherical Voronoi cells and edges.

Stretch factors are sampled at the centre of the dual object (spoke midpoint, chord radius $r_k$, sphere-face
radius $r_k$, side-face mid-layer radius): the radial analogue of Yee staggering, second order in $dr$.

## 3. Time stepping

Each ratio of CFS factors is applied as a cascade of two first-order filters
$\kappa_d y + \sigma_d P[y] = \kappa_n x + \sigma_n P[x]$, $P' + \alpha P = f$, with the trapezoidal rule
$P^n = c_1P^{n-1} + c_2(f^n+f^{n-1})$, $c_1 = (1-\alpha\Delta t/2)/(1+\alpha\Delta t/2)$,
$c_2 = (\Delta t/2)/(1+\alpha\Delta t/2)$ — the bilinear transform $i\omega \to i\Omega = (2/\Delta t)\,i\tan(\omega\Delta t/2)$
(`bilinear_response`, verified to 1e-6 by `test_filter_matches_bilinear_response`). This is a recursive
convolution of the Roden–Gedney (2000) type, but applied to the constitutive (Hodge) relation rather than to a
split curl. One step:

1. $b^{n+1/2} = b^{n-1/2} - \Delta t\,(C_f u^n + C_s e_s^n)$ (Faraday, unchanged).
2. PML faces: $h = F_f(M_\nu b^{n+1/2})$; other faces $h = M_\nu b$.
3. $r = C^T h - j$. Outside the layer $u^{n+1}$ follows the solver's own capacitive update (conductivity
   allowed); PML edges accumulate the dual flux $d^{n+1} = d^n + \Delta t\, r$.
4. PML edges: $u^{n+1} = F_e(d^{n+1}/M_\varepsilon)$.

**Massless edges in the layer.** Quad diagonals (zero area) and chords whose spherical Voronoi edge is tiny
(below `massless_tol`, from nearly cocircular surface triangles) carry no capacitance; their voltage is fixed by the
constraint $C_s^T h^{n+1/2} = 0$. With stretched $h$ the plain solver's elimination (unstretched $M_\nu$) is wrong
whenever the faces around the edge carry different filters (a massless chord touches sphere faces at $r_k$ and side
faces of two layers). Each face filter output is affine in its input, $h = G\,M_\nu b + \eta(\text{state})$ with
instantaneous gain $G > 0$, so the exact elimination is the local SPD solve

$$\left(C_s^T G M_\nu C_s\right) e_s\,\Delta t = C_s^T\!\left[G M_\nu (b^{n-1/2} - \Delta t\,C_f u^n) + \eta\right].$$

`test_massless_constraint_stretched` measures $\max|C_s^T h| / \max|C_f^T h| = 8\times10^{-18}$; the unstretched
elimination (mutation) gives $3.5\times10^{-4}$.

**Linearity and transpose.** All four stages are fixed sparse matrices on the extended state
$x = [u, b, d_{pml}, \text{filter states}]$: $x^{n+1} = S_4S_3S_2S_1x^n + B\,j^{n+1/2}$. `PML3D.step_T` applies
$S_1^TS_2^TS_3^TS_4^T$ and returns $B^TS_4^T\lambda$ (source sensitivity), the building block of a discrete adjoint
(dot-product test ~1e-15, finite differences 1e-9, `test_step_transpose_dot_product`). With $\sigma = 0$,
$\kappa = 1$ the assembled step equals the plain solver step to round-off (`test_identity_filters_match_plain_solver`).

**Stability (heuristic, not proved).** Each filter is passive for $\kappa\ge1$, $\sigma\ge0$, $\alpha\ge0$
(bilinear map of a positive-real function), the leapfrog CFL limit is that of the unstretched mesh, and the long-run
test shows no late-time growth over $10^5$ steps with $\alpha>0$. A proof for the stretched discrete system (Bécache &
Joly 2002 show the split Bérenger PML can be weakly ill-posed; CFS removes the low-frequency growth) is open.

## 4. Results

**Study** (`studies/unstructured-pml3d/`, ball radius 8 h, dr = h, centre wavelength 16 h, λ = 10–32 h; reflection
= field difference to an identical-interior reference with 40 extra vacuum layers; all runs one dt). The PML rows are
from `pml3d_floor.py` after the per-column massless fix of §4.1 (Cloud Batch `pmlfloor-v3`); the PEC and
Silver–Müller rows are unchanged from `pml3d_reflection.py`:

| boundary | normal incidence (worst probe, time domain) | oblique 48–61° (time domain, worst probe) |
|---|---|---|
| PEC wall at r0 | 0.81–1.9 | 0.69–10.7 |
| first-order Silver–Müller at r0 (same ball) | 0.14–0.22 | 0.46–0.66 |
| PML 4 / 6 layers | 1.9e-2 / 1.4e-3 | 7.2e-2 / 1.5e-2 |
| **PML 10 layers** | **3.0e-4 (−70 dB)** | **2.0e-3 (−54 dB)** |
| PML 16 / 24 layers | 7.0e-5 / 1.7e-5 | 3.9e-4 / 7.4e-5 |
| reference error (ref40 vs a 32-layer PML) | 6.1e-6 | 2.2e-5 |

Before the fix the normal-incidence values were 4.5e-4 (10), 4.6e-4 (16) and 4.6e-4 (24 layers): a floor. It is gone;
the reflection now falls as roughly $N^{-3.1}$ (10 → 16) to $N^{-3.4}$ (16 → 24) at fixed dr. The 10-layer value is set
by the discretised grading: the same 10 h thick layer with dr = h/2 (20 layers) reflects 7.3e-5, 4.1× less
($\approx dr^2$), measured against a 24 h thick dr = h/2 layer. A design reflection $R = 10^{-12}$ (larger
$\sigma_{max}$) is *worse* (4.0e-4 at 10, 9.8e-5 at 16 layers): the layer is grading-limited, not design-limited. κ_max = 5,
α = ω0/10 change propagating-wave absorption by < 30 %. The reference itself differs from the exact dipole field by
14–59 % at 7 h from the point source (Whitney point-dipole multipoles and interior dispersion), which is why the
identical-mesh reference is used.

### 4.1 The 4.5e-4 "floor" was the reference

The floor of the first study (10 and 16 layers both 4.5e-4) was tested against four hypotheses
(`studies/unstructured-pml3d/pml3d_floor.py`, 24 runs per variant, same dt and window):

| hypothesis | prediction | measured (normal incidence, before the fix) | verdict |
|---|---|---|---|
| backing echo (quasi-static content reaching the PEC wall) | $(E_{PEC}-E_{PMC})/2$ ≈ floor | 1.1e-7 (10 layers), 3.2e-8 (16) | rejected (4000× below) |
| design reflection not reached | $R=10^{-12}$ lowers it | unchanged vs ref40 (4.6e-4) | rejected |
| grading discretisation | dr = h/2 lowers it | 4.5e-4 → see the fixed column above | not the floor |
| **reference shell reflects** | all PMLs agree far below the floor; ref40 differs from a thick PML by ≈ floor | ref40 vs PML32 = **4.6e-4**; PML16 vs PML32 = 7.9e-5 | **confirmed** |

The difference ref40 − PML32 is zero until the wave returns from r ≈ 16 h and then grows (figure
`unstructured_pml3d_floor_mechanism.png`): the vacuum shell of the reference is not a homogeneous discrete medium.
The cause is the massless-edge test of the solver, `area < massless_tol · L²`. For the chord of one surface edge at
radius r of the shell, the dual area grows as $r\,dr$ and $L^2$ as $r^2$, so the test value falls as $dr/r$: on the
4 h test ball, 4 chords per ring are massless at r = 5 h and 29 at r = 33 h. Each change of the massless set changes
the discrete medium; the shell reflects at the rings where it happens. The fix (`_column_massless_factor`,
`mesh.massless_area_factor`) makes the decision once per radial column, with the value of the column's chord on the
first shell sphere. Along a column the chord's capacitance and the stiffness of its faces both scale as $1/dr$
(angular sizes fixed), so the local stiffness-to-capacitance ratio does not depend on r and keeping a column does not
lower dt (checked: dt_max unchanged to 1e-15). With the fix the reference error falls 75× (4.6e-4 → 6.1e-6) and the
layer-count curve has no floor down to 1.7e-5.

**What the identical-mesh reference does not measure.** The transition from the unstructured ball to the prism shell
cancels in test − reference. Shells with other dr change the field in the ball by 0.6 % (dr = 0.75 h), 1.0 %
(dr = h/2) and 2.0 % (dr = 1.5 h) relative to dr = h (thick PMLs, `pml3d_floor.py`), so the transition reflects at the
1e-2 level, far above the PML. Its absolute size is measured in §4.2.

**Tests** (`tests/test_098_unstructured_pml3d.py`, ball radius 4 h, centre wavelength 12 h): 10 layers 7.5e-5–1.3e-4
normal, 5.9e-4–1.8e-3 oblique; Silver–Müller 0.064–0.92; PEC 0.17–2.0. Layers 3 / 6 / 12: 1.6e-2 / 2.4e-4 / 2.5e-5
(ratio 12/6 = 0.10; with the per-chord massless decision it was 0.57, the test requires < 0.25). The floor test
(`test_floor_was_the_reference_shell`): per chord, the vacuum reference differs from a 20-layer PML by 1.1e-4 = the
floor; per column by 3.9e-6. The backing echo is 3.2e-8 (R = 1e-8) and 2.7e-4 with R = 1e-3 (mutation, must exceed
1e-5). `test_massless_decision_is_per_column`: 4 massless chords on every ring (per chord: 4 → 29). Mutations: negative
σ blows up (1e87); radial/tangential filters swapped 1.3–1.8; auxiliary term dropped 0.69–0.93; s_t omitted
3.0e-3–4.5e-3 (still −47 dB: a planar stretch on the sphere only adds a curvature error; the test requires > 10× the
correct layer); α < 0 grows by 1e18 in 1e4 steps. After the source is off, the ball energy falls to 2.8e-10 of its
peak with at most 5.2e-8 of the peak of transient rise (bound R² = 1e-4); in a PEC ball the same energy form is
conserved to 1e-17. Over 1e5 steps (CFS α > 0) the field stays at 1.6e-7 of its peak without growth (last-10 % /
40–50 % maximum ratio 0.98).

**Late time** (study, 3 × 1e5 steps, ball 5 h + 10 layers): no growth in any variant. With α = 0 a slowly
decaying quasi-static field stays in the layer (3.7e-6 of peak at the end). With α = ω0/10 (with or without
κ_max = 5) the field is flat at 1.3e-7 of peak and the interior energy is 1.9e-14 of peak. The flat residual sits
on interior edges near the source and is not a PML effect (open question).

## 5. Limits & Failure Modes

* **Geometry.** Only a spherical shell around a ball whose boundary nodes lie on one sphere. A rounded box needs
  the same co-spherical property per prism (it holds for any radial extrusion of a sphere-inscribed surface, not for a
  general convex surface: there the prism normals are not radial and the Hodge stops being diagonal). Non-layered
  tetrahedra in r0 < r < r1 are rejected (`test_unlayered_mesh_rejected`).
* **Tangential coarsening.** Shell cells grow as r/r0 tangentially. Spherical waves from a centred source are fine,
  but obliquely travelling waves are under-resolved in a thick shell; keep the PML thin (10 layers) and the source
  inside the ball.
* **Timestep.** Nearly cocircular surface triangles give chords with small (above `massless_tol`) dual area in
  every layer; on the meshes tested the shell lowered the stable dt by 27 % (0.154 → 0.113 h/c). A smoothed or
  centroidal surface triangulation would remove this.
* **Materials.** Homogeneous, lossless, non-dispersive medium in the shell (the profile uses the slowest speed of the
  mesh). Dispersive ADE groups with `pml` raise `NotImplementedError` (next milestone: run the ADE groups on the
  non-PML edges inside the stage S3).
* **Uniform α.** Graded α (as often used in CPML) breaks the CFS form of $s_t$; it would need one extra pole per
  edge.
* **Grazing incidence.** A sphere of radius $r_0$ only offers incidence angles at the specular point up to about
  $\arccos\sqrt{2a/r_0}$ for a source at depth $a$ below the layer; the study reaches 48–61° (r0 = 8 h, a = 1 h).
  Truly grazing (> 75°) incidence was not measured; it needs a larger ball or a source closer to the layer.
* **Massless chords.** Handled exactly, but with large blocks of coupled massless edges (> 200 in one connected
  component) the sparse inverse is refused.
* **Stability** is shown empirically, not proved.
* **Mesh transition.** The reflection numbers are relative to the same shell mesh; the tetrahedra → prism transition
  at r0 reflects separately and depends on dr (§4.2). It is the dominant reflection of the whole truncation.
* **Massless decision per column.** `spherical_shell_mesh` sets `mesh.massless_area_factor`; a shell built by other
  means (or a mesh edited afterwards) falls back to the per-edge decision and its vacuum layers reflect at ~1e-4.
* **Adjoint and GPU.** `UnstructuredAdjointStepper` replays and transposes the PML step (physics guide "Unstructured
  adjoint" §2.8); `DeviceStepper` runs it on the GPU ("Unstructured FIT on the GPU" §5).

## References

* J.-P. Bérenger, "A perfectly matched layer for the absorption of electromagnetic waves," J. Comput. Phys. 114,
  185–200 (1994). doi:10.1006/jcph.1994.1159
* W. C. Chew and W. H. Weedon, "A 3D perfectly matched medium from modified Maxwell's equations with stretched
  coordinates," Microw. Opt. Technol. Lett. 7, 599–604 (1994). doi:10.1002/mop.4650071304
* M. Kuzuoglu and R. Mittra, "Frequency dependence of the constitutive parameters of causal perfectly matched
  anisotropic absorbers," IEEE Microw. Guided Wave Lett. 6, 447–449 (1996). doi:10.1109/75.544545
* F. L. Teixeira and W. C. Chew, "Systematic derivation of anisotropic PML absorbing media in cylindrical and
  spherical coordinates," IEEE Microw. Guided Wave Lett. 7, 371–373 (1997). doi:10.1109/75.641424
* J. A. Roden and S. D. Gedney, "Convolution PML (CPML): an efficient FDTD implementation of the CFS-PML for
  arbitrary media," Microw. Opt. Technol. Lett. 27, 334–339 (2000).
  doi:10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A
* E. Bécache and P. Joly, "On the analysis of Bérenger's perfectly matched layers for Maxwell's equations,"
  ESAIM: M2AN 36, 87–119 (2002). doi:10.1051/m2an:2002004
* J. Dompierre, P. Labbé, M.-G. Vallet and R. Camarero, "How to subdivide pyramids, prisms and hexahedra into
  tetrahedra," Proc. 8th International Meshing Roundtable, 195–204 (1999) (no DOI).

All DOIs were checked against Crossref on 2026-11-13.
