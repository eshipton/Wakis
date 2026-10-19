### 🎓 High-order time-domain schemes: nodal DGTD on curved triangles and the Cartesian (2,4) scheme

**Why.** Yee/FIT is second order in space and time. Its phase velocity error is $\delta v/v \approx -(kh)^2/24$ per axis (the leapfrog adds $+(\omega\Delta t)^2/24$). For a resonator, the phase error accumulates over $Q$ round trips and shows up as a shifted resonance. In the Mie studies, for $x = ka > 6$ the error of both conformal Cartesian and triangular FIT is dominated by these bulk shifts, not by the interface. Halving the error costs $2^{d/2}\cdot 2^{1/2}$ more work in $d = 2$ dimensions (unknowns × steps $\propto h^{-3}$). A scheme of order $q$ needs $h \propto \varepsilon^{1/q}$, so its cost to reach a small target $\varepsilon$ falls like $\varepsilon^{-3/q}$ instead of $\varepsilon^{-3/2}$. `wakis.highorder` holds two 2D prototypes that test this. Both are separate solver cores and do not change FIT.

**Nodal DGTD (`DGTD2D`).** Both polarizations are one system for $(u, v_x, v_y)$ with coefficients $(a, b)$:

$$
a\,\partial_t u = \partial_x v_y - \partial_y v_x,\qquad
b\,\partial_t v_x = -\partial_y u,\qquad
b\,\partial_t v_y = \partial_x u ,
$$

with TM: $u = E_z$, $\mathbf v = \mathbf H$, $a = \varepsilon$, $b = \mu$, and TE: $u = H_z$, $\mathbf v = -\mathbf E$, $a = \mu$, $b = \varepsilon$ (duality).

- **Discretization.** On every triangle, the fields are polynomials of degree $p$ in a nodal (Lagrange) basis on Warburton's warp-and-blend nodes [Warburton, 2006]. The derivatives are exact on the polynomial space. Elements couple only through a numerical flux on their faces. This is the strong form of [Hesthaven & Warburton, 2002; 2008]:

$$
\mathbf M_J \frac{d q}{dt} = \mathbf M_J\,\nabla\!\cdot\!F(q) + \oint \ell\,\mathbf n\cdot(F^* - F^-)\,ds .
$$

- **Upwind flux.** Projected on the face normal, the system is the 1D pair $a\,\partial_t u = \partial_\xi H_\tau$, $b\,\partial_t H_\tau = \partial_\xi u$, with $H_\tau = n_x v_y - n_y v_x$ the tangential component. Its characteristics are $u \mp Z H_\tau$ with $Z = \sqrt{b/a}$. Solving the Riemann problem between the media $(-)$ and $(+)$ (continuity of $u^*$ and $H_\tau^*$) gives

$$
H_\tau^* - H_\tau^- = \frac{Z^+\,[H_\tau] + \alpha\,[u]}{Z^+ + Z^-},\qquad
u^* - u^- = \frac{Y^+\,[u] + \alpha\,[H_\tau]}{Y^+ + Y^-},
$$

  with $[f] = f^+ - f^-$ and $Y = 1/Z$. $\alpha = 1$ is the exact upwind flux for piecewise-constant media; it dissipates only unresolved modes. $\alpha = 0$ is the impedance-weighted central flux, which conserves energy. The flux couples only *tangential* fields, which are continuous across every interface. The normal-field jump of TE at a dielectric or metal interface is therefore free inside the discretization, with no averaging.
- **Curved elements.** Faces that are chords of a mesh circle are curved isoparametrically. The element nodes move by the Szabó–Babuška blend $\mathbf d = 4\lambda_i\lambda_j\,\boldsymbol\delta(\xi)/(1-\xi^2)$, with $\xi = \lambda_j - \lambda_i$ and $\boldsymbol\delta$ the arc-minus-chord displacement of the edge. The blend is smooth, so the polynomial interpolant of the map converges at order $p+1$. The metric terms come from the interpolated map at the nodes. The $J$-weighted mass matrix and the face mass matrices with the varying line Jacobian come from Gauss cubature, so every curved element has its own lift matrix.
  - The first version blended radially from the opposite vertex: $\mathbf d = (\lambda_i+\lambda_j)\,\Delta(t)$, $t = \lambda_j/(\lambda_i+\lambda_j)$. That map is only Lipschitz at the opposite vertex, and it cut the circular-cavity convergence of $p = 4$ from about 5 to about 2. `test_curved_elements_restore_high_order` guards this.
- **Time stepping.** The 5-stage, 4th-order, 2N-storage Runge–Kutta of [Carpenter & Kennedy, 1994]. $\Delta t$ is set from the spectral radius of the semi-discrete operator, computed by ARPACK on the matrix-free operator, times the scheme's imaginary-axis limit $\beta \approx 3.34$ and a CFL factor. On the same mesh $\Delta t \propto h/p^2$, so high $p$ pays in steps what it saves in unknowns.
- **Absorbing boundaries.** The first-order Silver–Müller condition (zero exterior state in the flux) is used on a disk, with the reflections de-embedded by the harmonic extraction. The other option is a complex-frequency-shifted stretched-coordinate PML in a rectangular frame:
  - $1/s_q = 1 - \sigma_q/(s+\alpha+\sigma_q)$ acts on each weak derivative $\partial_q f$ through an auxiliary field $\Psi$ with $\dot\Psi = \sigma_q\,\partial_q f - (\sigma_q+\alpha)\Psi$;
  - the weak derivative includes the lifted flux term, split by components: $v^* - v^- = \hat{\mathbf t}\,(H_\tau^*-H_\tau^-)$;
  - four auxiliary fields live on the PML elements only.

  A first version without the CFS shift and with direct $\sigma$ damping of $\mathbf v$ gave an $O(1)$ low-frequency error in a 1000 fs Mie run.
- **Plane-wave source.** The scattered-field formulation $a\,\partial_t u_s = \partial_x v_{s,y} - \partial_y v_{s,x} - (a - a_b)\,\partial_t u_{inc}$, plus the analogous source for $\mathbf v$ in TE. It is a volume source on the nodes of the non-background elements. It is exact for curved elements because the flux only sees continuous tangential components.
- **Dispersive media.** `DispersiveMaterial` with Lorentz/Drude ($\dot J = -\gamma J - \omega_0^2 P + \varepsilon_0 S\,E$, $\dot P = J$), Debye and conductivity. The poles are driven by the *total* field $E_s + E_{inc}$ and are integrated by the same RK scheme. The ADE is solved without a time-discrete permittivity (no $\sin(\omega\Delta t/2)$ factors): the time error is RK4's $O(\Delta t^4)$.
- **Adjoint compatibility.** The semi-discrete system $\dot X = A X + f(t)$ is linear. `operator_matrix()` assembles $A$ exactly as a sparse matrix by probing the matrix-free operator with a distance-2 colouring of the elements: about $n_{colours}\cdot N_p\cdot n_{fields}$ applications, independent of the mesh size. The discrete adjoint of an LSRK step is the same scheme applied to $A^T$ in reverse. `test_operator_matrix_and_adjoint_dot_product` checks $\langle y, S x\rangle = \langle S^T y, x\rangle$ to $10^{-12}$; a mutation (using $A$ instead of $A^T$) fails. This is a different structure from FIT's leapfrog adjoint (it needs the stage vectors, i.e. 5 states per step to checkpoint), but it is equally exact.
- **GPU suitability.** All work is dense element-local GEMMs ($[D_r; D_s]\,(N_p\times 3K)$, lift $N_p\times 3N_{fp}$) plus gathers, which is the textbook GPU pattern. Arithmetic intensity grows with $p$, so DG is *more* GPU-friendly than FIT's memory-bound sparse updates. The numpy prototype runs on one CPU core.

**Cartesian (2,4) (`FDTD24`).** The Yee staggering with the fourth-order staggered difference

$$
\partial_x f\big|_{i+1/2} \approx \frac{1}{h}\Big[\tfrac98\,(f_{i+1}-f_i) - \tfrac1{24}\,(f_{i+2}-f_{i-1})\Big],
$$

[Yefet & Petropoulos, 2001; Hwang & Cangellaris, 2003] gives a spatial phase error of $-\tfrac{3}{640}(kh)^4$. The leapfrog stability limit drops by $6/7$ ($\max|\text{symbol}| = 9/8 + 1/24$).

- **Time error.** With leapfrog, the time error $+(\omega\Delta t)^2/24$ dominates the (2,4) spatial error at practical CFL numbers. At 14 cells per dielectric wavelength it is about 7 times larger. Option `time_scheme='lsrk4'` integrates the same semi-discrete operator, including the CPML auxiliaries in ADE form, with LSRK4.
  - Yoshida's 4th-order symplectic composition [Yoshida, 1990] was tried and abandoned: its negative sub-step turns the CPML into an amplifier, and the run went unstable.
- **Interfaces.** Sub-pixel averaging is the Hwang–Cangellaris/Farjadpour type. TM uses the area-averaged $\varepsilon$ ($E_z$ is tangential). TE uses the diagonal of the averaged inverse tensor, $(\varepsilon^{-1})_{ii} = n_i^2\langle\varepsilon^{-1}\rangle + (1-n_i^2)/\langle\varepsilon\rangle$. Both are at best second order, so the interface caps the (2,4) scheme at second order once the bulk error is gone. The study measures where that happens.
- **Source and loss.** The scattered-field source is written in constitutive form: $E_s = \kappa D_s/\varepsilon_0 + (\kappa - 1)E_{inc}$ with $\kappa = \varepsilon_{\rm eff}^{-1}$. A conductivity is handled by Crank–Nicolson on $D_s$, driven by the total field. PEC walls use image ghosts (integer-point fields odd, half-point fields even), so the wide stencil keeps 4th order at a PEC wall.
- **Adjoint and GPU.** Every update is a sum of difference operators and diagonal scalings, so it is linear and transposable like Yee, and it is as GPU-friendly as Yee: the stencil is twice as wide, the memory traffic per cell is the same, and there are about 1.5 times as many flops.

**Results** (study `studies/highorder-mie/`, benchmark arms in `studies/highorder-bench/`). The numbers are in the study README: unknowns, work and uncontended time to reach 2 %, 1 % and 0.1 % RMS error in $Q_{sca}$, against the best second-order baselines (graded triangle FIT with PML, graded conformal Cartesian FIT).

#### Limits & Failure Modes

- **Straight-sided DG elements are second order at curved interfaces.** With polygonal interfaces, DG at $p = 4$ stalls at about 3 % RMS on the Mie cylinder (B1), the same as a coarse FIT mesh. High order needs the geometry to high order too. Curved elements are supported for circles only (any number, including the outer boundary). General curves need a CAD projection.
- **Cost per unknown.** A DG unknown costs 5 RHS evaluations per step and dense element operations, with $\Delta t \propto h/p^2$. Measured on one n2 core, it is about 10× the cost of a Yee/FIT unknown-step. DG wins only where its error per unknown falls much faster, at tight tolerances or electrically large problems; the study quantifies the crossover. At loose tolerances (a few %), graded FIT is cheaper.
- **Time step set by the smallest element.** Global RK has no local time stepping. A small element at a gap or a thin feature sets $\Delta t$ for the whole mesh (B3 gaps).
- **Upwind dissipation.** $\alpha = 1$ damps unresolved modes, which is welcome, but it also damps resolved high-Q resonances slightly. $\alpha = 0$ conserves energy but leaves spurious modes undamped.
- **Mesh quality.** The force-equilibrium mesher (`circles_mesh`) keeps circle chords Delaunay through a node clearance. It does not handle circles that touch or overlap. A gap narrower than the local $h$ is forced to one element across.
- **PML.** It is rectangular and needs vacuum or background in the frame. Its reflection is set by the element resolution of the $\sigma$ profile (about $10^{-3}$ field error for 2 elements of $p = 3$; under $10^{-4}$ for 4). No curved elements may lie inside it.
- **(2,4) scheme.** The sub-pixel averaging is second order at best. The diagonal TE tensor drops the off-diagonal terms, so TE is first order near oblique interfaces. With leapfrog the scheme is second order in time; it needs LSRK4 or a small CFL to be 4th order. Its benefit disappears once the interface error dominates. Metals were not implemented in `FDTD24`; the mixed-cell averaging problem of conformal Cartesian metals is unchanged.

References:
- J. S. Hesthaven and T. Warburton, "Nodal high-order methods on unstructured grids I. Time-domain solution of Maxwell's equations", *J. Comput. Phys.* **181**, 186–221 (2002), doi:10.1006/jcph.2002.7118.
- J. S. Hesthaven and T. Warburton, *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*, Springer (2008), doi:10.1007/978-0-387-72067-8.
- T. Warburton, "An explicit construction of interpolation nodes on the simplex", *J. Eng. Math.* **56**, 247–262 (2006), doi:10.1007/s10665-006-9086-6.
- M. H. Carpenter and C. A. Kennedy, "Fourth-order 2N-storage Runge–Kutta schemes", NASA TM-109112 (1994).
- M. Ainsworth, "Dispersive and dissipative behaviour of high order discontinuous Galerkin finite element methods", *J. Comput. Phys.* **198**, 106–130 (2004), doi:10.1016/j.jcp.2004.01.004.
- K. Busch, M. König and J. Niegemann, "Discontinuous Galerkin methods in nanophotonics", *Laser Photonics Rev.* **5**, 773–809 (2011), doi:10.1002/lpor.201000045.
- W. J. Gordon and C. A. Hall, "Construction of curvilinear co-ordinate systems and applications to mesh generation", *Int. J. Numer. Methods Eng.* **7**, 461–477 (1973), doi:10.1002/nme.1620070405.
- A. Yefet and P. G. Petropoulos, "A staggered fourth-order accurate explicit finite difference scheme for the time-domain Maxwell's equations", *J. Comput. Phys.* **168**, 286–315 (2001), doi:10.1006/jcph.2001.6691.
- K.-P. Hwang and A. C. Cangellaris, "Computational efficiency of Fang's fourth-order FDTD schemes", *Electromagnetics* **23**, 89–102 (2003), doi:10.1080/02726340390159450.
- K.-P. Hwang and A. C. Cangellaris, "Effective permittivities for second-order accurate FDTD equations at dielectric interfaces", *IEEE Microw. Wireless Compon. Lett.* **11**, 158–160 (2001), doi:10.1109/7260.916329.
- Z. Zhao and G. W. Wei, "High-order FDTD methods via derivative matching for Maxwell's equations with material interfaces", *J. Comput. Phys.* **200**, 60–103 (2004), doi:10.1016/j.jcp.2004.03.008.
- H. Yoshida, "Construction of higher order symplectic integrators", *Phys. Lett. A* **150**, 262–268 (1990), doi:10.1016/0375-9601(90)90092-3.
- J. A. Roden and S. D. Gedney, "Convolution PML (CPML)", *Microw. Opt. Technol. Lett.* **27**, 334–339 (2000).
- F. Rapetti and A. Bossavit, "Whitney forms of higher degree", *SIAM J. Numer. Anal.* **47**, 2369–2386 (2009), doi:10.1137/070705489 (the high-order FIT route, not implemented; see the study README).

