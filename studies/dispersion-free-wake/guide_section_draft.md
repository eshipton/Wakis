### 🚄 Dispersion-free wake computation: the explicit TE/TM scheme (`scheme='tete'`)

`SolverFIT3D(..., scheme='tete')` replaces the Yee leapfrog by the explicit
TE/TM scheme of Dohlus and Zagorodnov [J. Comput. Phys. 228, 2822 (2009),
arXiv:0901.1053], the explicit successor of the implicit TE/TM scheme of
Zagorodnov and Weiland [J. Comput. Phys. 207, 69 (2005); Phys. Rev. ST
Accel. Beams 8, 042001 (2005)]. At the time step $c\Delta t=\Delta z$ waves
along $z$ propagate at exactly $c$, so an ultrarelativistic bunch does not
emit numerical Cherenkov radiation. The default (`scheme='yee'`) is
unchanged.

#### Why Yee radiates

In vacuum the Yee scheme obeys

$$
\frac{\sin^2(\omega\Delta t/2)}{c^2\Delta t^2}
=\frac{\sin^2(k_x\Delta x/2)}{\Delta x^2}
+\frac{\sin^2(k_y\Delta y/2)}{\Delta y^2}
+\frac{\sin^2(k_z\Delta z/2)}{\Delta z^2}.
$$

Along $z$ the phase velocity equals $c$ only for $c\Delta t=\Delta z$, which
violates the 3D Courant limit
$c\Delta t\le(\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2})^{-1/2}$. For any
stable step, short waves along $z$ are slower than the bunch: the bunch field
(a plane-wave-like TEM "pancake") is continually re-radiated behind it. The
phase error over a length $L$ grows as
$L\,\delta k\sim L\,\Delta z^2/\sigma_z^3$ [Dohlus & Zagorodnov, eq. (12)],
so Yee needs $\Delta z\lesssim\sigma_z^{3/2}L^{-1/2}$ for a fixed accuracy,
while a scheme without longitudinal dispersion only needs
$\Delta z\propto\sigma_z$.

#### The scheme

Split the fields into the subsystems (relative to $z$)
$u=(h_x,h_y,e_z)$ ("TM") and $v=(e_x,e_y,h_z)$ ("TE"), and split the FIT
operators $A=$ `tDsiDmuiDaC` ($\partial_t h=-Ae$) and $B=$
`itDaiDepsDstC` ($\partial_t e=Bh-\varepsilon^{-1}j$) into their
longitudinal parts $A_L$, $B_L$ (the $P_z$ couplings $e_{x,y}\leftrightarrow
h_{y,x}$) and their transverse parts ($A_T: e_z\to h_{xy}$,
$A_Z: e_{xy}\to h_z$, $B_T: h_z\to e_{xy}$, $B_Z: h_{xy}\to e_z$). One step
is a leapfrog between TE and TM in which each subsystem's transverse
dynamics is integrated with a symmetric (Störmer–Verlet) split:

$$
\begin{aligned}
l_E &= B_L h_{xy}^{n}, &
e_{xy} &\mathrel{+}= \tfrac{\Delta t}{2}\,(B_T h_z + l_E - \varepsilon^{-1}j_{xy}^{n}),\\
h_z^{n+1/2} &= h_z^{n-1/2}-\Delta t\,A_Z e_{xy}, &
e_{xy}^{n+1/2} &= e_{xy} + \tfrac{\Delta t}{2}\,(B_T h_z^{n+1/2} + l_E - \varepsilon^{-1}j_{xy}^{n}),\\
l_H &= A_L e_{xy}^{n+1/2}, &
h_{xy} &\mathrel{-}= \tfrac{\Delta t}{2}\,(A_T e_z^{n} + l_H),\\
e_z^{n+1} &= e_z^{n}+\Delta t\,(B_Z h_{xy}-\varepsilon^{-1}j_z^{n+1/2}), &
h_{xy}^{n+1} &= h_{xy} - \tfrac{\Delta t}{2}\,(A_T e_z^{n+1} + l_H).
\end{aligned}
$$

This is the detailed form on p. 7 of Dohlus & Zagorodnov with the step cut
before the TE block instead of the TM block (the sequence of sub-updates is
identical). Wakis stores $e_z,h_x,h_y$ at integer and $e_x,e_y,h_z$ at
half-integer times, so $E_z$, the field the wake post-processing integrates,
is sampled at the same instants $n\Delta t$ as with Yee. The longitudinal
terms $l_E$, $l_H$ are evaluated once per step, so a step costs 4/3 of the
Yee sparse matrix-vector work (not the 5/3 of 10 independent sub-updates
quoted in the paper).

Dispersion relation in vacuum on a uniform grid [eq. (27)], verified to
$10^{-12}$ by the Fourier symbol of the six sub-updates and by measurement
(`tests/test_045_tete.py`):

$$
\frac{\sin^2(\omega\Delta t/2)}{c^2\Delta t^2}
=\frac{\sin^2(k_z\Delta z/2)}{\Delta z^2}
+\Big(\frac{\sin^2(k_x\Delta x/2)}{\Delta x^2}
+\frac{\sin^2(k_y\Delta y/2)}{\Delta y^2}\Big)
\Big(1-\frac{c^2\Delta t^2}{\Delta z^2}\sin^2\frac{k_z\Delta z}{2}\Big).
$$

For $k_x=k_y=0$ and $c\Delta t=\Delta z$ it gives $\omega=ck_z$ for every
$k_z$ (both polarisations).

#### Stability

With the energy method (the step is $B_e(y^{n+1}-y^n)/\Delta t+Ay^n=f$ with
skew $A$), the scheme conserves the discrete energy
$\tfrac12\langle Q_e y,y\rangle$, $Q_e=B_e-\tfrac{\Delta t}{2}A$, and is
stable if $Q_e>0$ [Dohlus & Zagorodnov, sec. 4.4], i.e.

$$
\Delta t\le\frac{2}{\sqrt{\lambda_{\max}(B_LA_L)}},\qquad
\Delta t\le\frac{2}{\sqrt{\lambda_{\max}(B_ZA_T)}},\qquad
\Delta t\le\frac{2}{\sqrt{\lambda_{\max}(B_TA_Z)}},
$$

which on a uniform staircase grid in vacuum reads [eq. (25)]

$$
c\Delta t\le\Delta z,\qquad c\Delta t\le(\Delta x^{-2}+\Delta y^{-2})^{-1/2}.
$$

So the magic step needs $\Delta z\le(\Delta x^{-2}+\Delta y^{-2})^{-1/2}$,
e.g. $\Delta x=\Delta y\ge\sqrt2\,\Delta z$: the transverse mesh may be
coarser than the longitudinal one, which suits beam-pipe geometries. The
solver certifies the step with Gershgorin bounds,
$\lambda_{\max}(XY)\le\max_i(|X||Y|\mathbf 1)_i$, of the three operators
(`wakis.tete.gershgorin_limits`). They are tight on uniform staircase grids
(they reproduce the two conditions above) and cost one sparse product, so no
eigenvalue solve is needed; a time step above the bound raises
`ValueError`. The test suite checks that the limit is sharp: at $0.99$ of the
transverse limit the one-step map has spectral radius 1, at $1.03$ it grows.

**Conformal PEC.** The conformal grid of the Yee path sub-samples the face
areas from the STL surface and stabilises small faces by linear
extrapolation from two neighbours (weights $2,-1$). Both raise
$\lambda_{\max}(B_LA_L)$ above $4c^2/\Delta z^2$, and merging a small face
normal to $x$ or $y$ with a neighbour couples two $(x,y)$ columns, so the
bunch self-field is no longer an exact travelling solution. Measured stable
steps relative to $\Delta z/c$ ($\Delta x=2\Delta z$):

| geometry | Yee conformal operators | x/y faces merged (weight 1) | TE/TM recipe below |
|---|---|---|---|
| round pipe, $r=5\Delta x$, longitudinal | 0.887 | 1.000 | 1.000 |
| pillbox + pipes ($\Delta z$ = 0.8 mm), longitudinal | – | 0.861 (sub-sampled areas) | 1.000 |

With `scheme='tete'` the conformal setup therefore

1. gives the faces normal to $x$ and $y$ the trapezoidal open area
   $A=\Delta z\,(l_{k}+l_{k+1})/2$ from the open lengths of their two
   transverse edges and does not redistribute them. Every row of $B_LA_L$
   then sums to $\frac{\tilde l}{\varepsilon\mu\tilde A}
   \big[\frac{l_{k+1}+l_k}{A_r}+\frac{l_k+l_{k-1}}{A_l}\big]=4c^2/\Delta z^2$,
   and Gershgorin's theorem gives $\lambda_{\max}(B_LA_L)\le4c^2/\Delta z^2$
   for any wall shape;
2. restores the transverse (TM) stability lost by the small unmerged faces
   by scaling the rows of $B_Z$, i.e. $\varepsilon^{-1}$ of the $e_z$ edges next
   to them, by $s=\min\big(1,(2/\Delta t)^2/r\big)$, $r=|B_Z||A_T|\mathbf 1$
   (Gershgorin bound of $B_ZA_T$); if the TE limit is also too small, the
   rows of $A_Z$ ($\mu^{-1}$ of $h_z$) are scaled the same way (the z faces
   keep the extrapolating redistribution of the Yee path, whose absolute
   weights inflate the Gershgorin sums). Only $e_z$
   and $h_z$, which vanish in the TEM self-field, get heavier, so the
   self-field stays exact and the longitudinal operator is untouched.

The trapezoidal area is exact for walls that are invariant or linear along
$z$ (pipes, tapers); for faces cut by a wall normal to $z$ (cavity end
plates) it is a local first-order area error, and the mass scaling is a
local first-order error for fields with $E_z$ ($H_z$) next to small cut
faces. On the study pillbox ($\Delta x=2\Delta z$) 184 (0.8 mm) and 504
(0.4 mm) $e_z$ edges (smallest factors 0.43 and 0.07) and 1032 and 4772 $h_z$
faces (about 2 % of them; smallest factors 0.60 and 0.53) were scaled. On a
small conformal pipe the Arnoldi eigenvalues confirm the certified limits
(`tests/test_045_tete.py`). The conformal `Beam` TFSF template is built from the
solver's own operators (`Beam.tfsf_template='operator'`, the default for
TE/TM), which equals the staircase Poisson template to $10^{-12}$ on
staircase grids.

#### Source consistency

The self-field of a bunch with $v=c$ in a uniform pipe is a TEM wave
$E_\perp=-\nabla_\perp\phi\,\lambda(z-ct)$, $H_\perp=\hat z\times E_\perp/Z_0$,
$E_z=0$. On the grid it is an exact discrete solution of the TE/TM scheme
at $c\Delta t=\Delta z$ if (i) $\phi$ solves the discrete transverse Poisson
problem (as in the TFSF injection of `Beam`), (ii) the 1D longitudinal
leapfrog at Courant number 1 shifts any sampled profile by exactly one cell
per step, and (iii) the current that drives $e_z$ matches the transverse
curl of $h$ used in the $e_z$ update. Because the $e_z$ kick sees
$h_{xy}$ after one half kick, i.e. the average of $h_{xy}$ at $t^n$ and
$t^{n+1}$, condition (iii) requires the trapezoidal current
$j_z^{n+1/2}=\tfrac12[j_z(t^n)+j_z(t^{n+1})]$ instead of the midpoint value
(the difference is $O(\Delta z^2\lambda'')$). `Beam` uses this average in
TE/TM mode and samples the injected TFSF fields at the TE/TM time levels
($E_\perp$ at $t+\Delta t/2$, $H_\perp$ at $t$). With these, the smooth
pipe gives $|W|\lesssim10^{-8}$ V/pC (round-off) where Yee gives 1–170 V/pC
(table below).

The TFSF injection of `Beam` stops when the bunch centre is
`Beam.tfsf_truncation` $\sigma_z$ (default 5) past the downstream plane; the
truncated tail leaves a residual field of $e^{-n^2/2}$ of the peak
($3.7\times10^{-6}$ at 5). The study and tests use 8.

**Charge conservation.** The scheme conserves the modified discrete charge
$q_e^{n+1/2}=P_x^e\frac{e_x^{n+1}+e_x^{n}}{2}+P_y^e\frac{e_y^{n+1}+e_y^n}{2}
+(I+\tfrac{\Delta t^2}{4}r_0)P_z^e e_z^{n+1/2}$ to second order
[Dohlus & Zagorodnov, eq. (35)]; a rigid beam current therefore does not
accumulate spurious static charge.

#### Exact adjoint

Every sub-update is a shear $x_i\mathrel{+}=\alpha\,Mx_j$ (or a diagonal loss
factor), so the step $U$ is linear and its Hermitian transpose is the
composition of the transposed shears in reverse order, including the z-CPML
auxiliary variables (`wakis.tete.TeteTranspose`, used by
`wakis.inverse.AdjointStepper`). The dot-product test
$\langle Ux,y\rangle=\langle x,U^Hy\rangle$ holds to $\sim10^{-15}$ (PEC, lossy,
periodic, z-CPML, complex Bloch) and fails at $5\times10^{-7}$ when one
transposed block is perturbed by $10^{-3}$. `AdjointSimulation` (design
gradients) is not supported: $\varepsilon^{-1}$ enters three sub-updates per
step, and the single-increment gradient formula of the Yee path does not
apply.

#### Validation (`studies/dispersion-free-wake/`)

STUDY_TABLE

#### Limits and failure modes

- **Only the $z$ direction is dispersion-free.** Off-axis waves have
  dispersion comparable to Yee (larger between ~30° and 60° on
  $\Delta x=2\Delta z$; see `figures/dispersion_vs_angle.png`). Cavity modes
  and transverse resonances keep a second-order error in the transverse
  cell size.
- **Magic step requires $\Delta z\le(\Delta x^{-2}+\Delta y^{-2})^{-1/2}$**
  and a uniform $\Delta z$ (with a graded $z$ mesh the step is set by the
  smallest $\Delta z$ and only those cells are dispersion-free). If the
  transverse limit is smaller, the solver refuses the magic step; with an
  explicit smaller `dt` the scheme runs with longitudinal dispersion
  ($v_{ph}<c$, like Yee but smaller).
- **Periodic $z$ with an even number of cells**: the Nyquist mode
  $k_z\Delta z=\pi$ is a double root of the amplification polynomial at
  $c\Delta t=\Delta z$, so its amplitude can grow linearly (not
  exponentially) if excited (random initial fields); smooth bunches excite it
  only at round-off level.
- **Conformal grids** change the x/y face areas and the merging rule for the
  TE/TM run (see above), so Yee and TE/TM runs on the same conformal grid do
  not share exactly the same geometry model; transverse small-cell limits
  can still forbid the magic step on coarse transverse meshes.
- **Not supported** (raise `NotImplementedError`): CPML on x/y faces, PML,
  Mur ABC, SIBC, dispersive (ADE) materials, tensor averaging
  (`conformal='tensor'`), lumped elements, magnetic current sources,
  plane-wave TFSF without CPML, design gradients (`AdjointSimulation`), and
  the TE/TM adjoint with MPI. MPI forward runs are supported (z slabs, two
  ghost exchanges per step, equal to the serial run to $10^{-10}$).
- **Conductivity** ($\sigma>0$) is treated explicitly as in the Yee path
  ($j=\sigma e$ at the start of each sub-update); first order in time, and
  the conductivity time-constant check of the solver may lower $\Delta t$
  below the magic step (a warning is printed).
- **Time levels**: $e_x,e_y,h_z$ are stored half a step later than with Yee;
  field monitors that mix components see this half-step offset.
- **Wakis post-processing** samples $E_z(z,t)$ with `int(ts/dt)`
  (truncation); at $c\Delta t=\Delta z$ the sampling instants fall on the
  grid exactly, so round-off can shift a sample by one step for both schemes.
