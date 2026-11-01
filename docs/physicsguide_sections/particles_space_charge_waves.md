#### Space-charge waves on a drifting sheet beam: cross-validation of particles against `LinearizedBeam`

When a cold relativistic electron beam of DC velocity $\mathbf{u}_0 = u_0\hat{\mathbf{z}}$ ($u_0 = c\sqrt{1 - \gamma_0^{-2}}$, Lorentz factor $\gamma_0 = 1 + e V_0 / (m_e c^2)$) and unperturbed charge density $\rho_0 = -e n_0 < 0$ drifts through a conducting tunnel under strong axial magnetic confinement ($\mathbf{B}_{\text{ext}} = B_{z,\text{ext}}\hat{\mathbf{z}}$ with $\omega_c = e B_{z,\text{ext}} / (\gamma_0 m_e) \gg \omega_p$), longitudinal density perturbations excite electrostatic restoring forces that propagate along the beam as a pair of **fast and slow space-charge waves** [Ramo, 1939; Hahn, 1939]. Cross-validating the Lagrangian macroparticle solver (`wakis.particles.ParticleSolver` + `BeamInjector`) against the Eulerian linearized cold-fluid solver (`wakis.beam_fluid.LinearizedBeam`) on the identical `GridFIT3D` and `SolverFIT3D` setup tests the entire coupled particle–field loop—continuous time-modulated injection, relativistic Boris pushing, 3D Esirkepov current deposition, and Finite Integration Technique (FIT) field updates—against an independent continuum discretization and analytical dispersion theory (`tests/test_087_particles_vs_linearized_beam.py`, `examples/024_particles_space_charge_waves.py`).

##### 1. Governing equations and reduced plasma wavenumber $\beta_q$

In the small-signal regime ($|v_1| \ll u_0$, $|\rho_1| \ll |\rho_0|$) with transverse motion suppressed by $B_{z,\text{ext}}$ ($\mathbf{v}_{1,\perp} = \mathbf{0}$), linearizing the relativistic longitudinal momentum equation $\mathrm{d}(\gamma m_e v_z)/\mathrm{d}t = -e E_z$ around $v_z = u_0 + v_1$ gives

```math
\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial z}\right) v_1(x, z, t) = -\frac{e}{\gamma_0^3 m_e}\,E_z(x, z, t),
```

```math
\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial z}\right) \rho_1(x, z, t) = -\rho_0\,\frac{\partial v_1(x, z, t)}{\partial z},
```

```math
J_{1,z}(x, z, t) = \rho_0\,v_1(x, z, t) + u_0\,\rho_1(x, z, t).
```

For a monochromatic perturbation $\propto \exp\bigl(\mathrm{i}(\omega t - k_z z)\bigr)$ with electronic wavenumber $\beta_e = \omega / u_0$ and non-relativistic plasma frequency $\omega_p = \sqrt{n_0 e^2 / (\varepsilon_0 m_e)} = \sqrt{-\rho_0 e / (\varepsilon_0 m_e)}$, eliminating $v_1$ and $\rho_1$ gives the spatially dispersive axial current response

```math
J_{1,z}(\omega, k_z) = -\mathrm{i}\omega\varepsilon_0\,\frac{\omega_{p,\parallel}^2}{(\omega - k_z u_0)^2}\,E_z(\omega, k_z), \qquad \omega_{p,\parallel} = \frac{\omega_p}{\gamma_0^{3/2}},
```

corresponding to an effective axial relative permittivity inside the beam cross-section:

```math
\frac{\varepsilon_{zz}(\omega, k_z)}{\varepsilon_0} = 1 - \frac{\omega_{p,\parallel}^2}{(\omega - k_z u_0)^2}.
```

In an infinitely wide 1D beam ($\nabla_\perp^2 E_z = 0$), Gauss's law requires $\varepsilon_{zz}(\omega, k_z) = 0$, giving $\omega - k_z u_0 = \pm \omega_{p,\parallel}$. In a beam of finite transverse extent inside a conducting drift tunnel, part of the longitudinal displacement flux terminates as image charge on the surrounding Perfectly Electric Conducting (PEC) walls rather than on adjacent space-charge bunches. This fringe-field screening reduces the restoring axial electric field $E_z$ by the geometric **plasma-frequency reduction factor** $R \in (0, 1)$ [Ramo, 1939; Branch & Mihran, 1955]:

```math
\omega_q = R\,\omega_p\,\gamma_0^{-3/2}, \qquad \beta_q = \frac{\omega_q}{u_0} = \frac{R\,\omega_p}{\gamma_0^{3/2} u_0},
```

splitting a modulation at angular frequency $\omega$ into two co-propagating space-charge waves with axial wavenumbers

```math
k_{\text{fast}} = \beta_e - \beta_q \quad \left(v_{p,\text{fast}} = \frac{\omega}{\beta_e - \beta_q} > u_0\right), \qquad k_{\text{slow}} = \beta_e + \beta_q \quad \left(v_{p,\text{slow}} = \frac{\omega}{\beta_e + \beta_q} < u_0\right).
```

##### 2. Planar sheet-beam reduction factor (`reduction_factor_planar`)

While Branch & Mihran (1955) tabulated reduction factors for cylindrical beams in cylindrical tunnels, the 2D planar analogue implemented in `wakis.beam_fluid.reduction_factor_planar` follows from the parallel-plate slow-wave transverse-magnetic ($\mathrm{TM}$) boundary-value problem (see also the main physics guide section "Linearized 1D Relativistic Cold-Fluid Beam (`wakis.beam_fluid`)"). Consider a planar sheet electron beam of half-thickness $b$ ($|x| \le b$) centered inside a parallel-plate PEC drift tunnel of half-width $a$ ($|x| \le a$, $b \le a$), uniform (periodic) along $y$. For a slow wave with axial wavenumber $k_z \approx \beta_e$ and transverse vacuum decay constant

```math
\tau = \sqrt{\beta_e^2 - \left(\frac{\omega}{c}\right)^2} = \frac{\beta_e}{\gamma_0},
```

the transverse wave equation for the $n$-th even $\mathrm{TM}$ space-charge eigenmode is

```math
\left(\frac{\mathrm{d}^2}{\mathrm{d}x^2} + T_n^2\right) E_z(x) = 0 \quad (|x| \le b), \qquad \left(\frac{\mathrm{d}^2}{\mathrm{d}x^2} - \tau^2\right) E_z(x) = 0 \quad (b < |x| \le a),
```

where $T_n^2 = -\tau^2\bigl(\varepsilon_{zz}/\varepsilon_0\bigr) = \tau^2\bigl(R_n^{-2} - 1\bigr) > 0$. Enforcing $E_z(\pm a) = 0$ on the PEC tunnel walls and continuity of $E_z$ and $\mathrm{d}E_z/\mathrm{d}x$ across the beam boundary $x = \pm b$ with $E_z(x) = A\cos(T_n x)$ ($|x| \le b$) and $E_z(x) = B\sinh\bigl(\tau(a - |x|)\bigr)$ ($b \le |x| \le a$) yields the transcendental eigenvalue equation:

```math
(T_n b)\tan(T_n b) = (\tau b)\coth\!\bigl(\tau(a - b)\bigr), \qquad (n - 1)\pi < T_n b < \left(n - \frac{1}{2}\right)\pi,
```

from which the fundamental ($n = 1$) planar reduction factor evaluated by `wakis.beam_fluid.reduction_factor_planar(omega, u0, b, a=a)` is

```math
R(\tau b, a/b) = \frac{1}{\sqrt{1 + \left(\dfrac{T_1}{\tau}\right)^2}}.
```

##### 3. DC space-charge potential and velocity depression in an unneutralized sheet beam

In `ParticleSolver`, the beam is injected into an empty PEC tunnel without a neutralizing ion background, so the steady DC beam charge density $\rho_0 < 0$ generates a transverse electrostatic potential trough $\phi_{\text{DC}}(x) < 0$ across the tunnel. Away from the entrance and exit planes ($z \gg a$, $L_z - z \gg a$), solving the 1D Poisson equation

```math
\frac{\mathrm{d}^2 \phi_{\text{DC}}}{\mathrm{d}x^2} = -\frac{\rho_0}{\varepsilon_0} \quad (|x| \le b), \qquad \frac{\mathrm{d}^2 \phi_{\text{DC}}}{\mathrm{d}x^2} = 0 \quad (b < |x| \le a),
```

subject to $\phi_{\text{DC}}(\pm a) = 0$, $\mathrm{d}\phi_{\text{DC}}/\mathrm{d}x\big|_{x=0} = 0$, and continuity of $\phi_{\text{DC}}$ and $\mathrm{d}\phi_{\text{DC}}/\mathrm{d}x$ at $x = \pm b$ gives

```math
\phi_{\text{DC}}(x) = \frac{\rho_0}{2\varepsilon_0}\bigl(2 a b - b^2 - x^2\bigr) \quad (|x| \le b), \qquad \langle \phi_{\text{DC}} \rangle_{|x|\le b} = \frac{1}{2b}\int_{-b}^{b} \phi_{\text{DC}}(x)\,\mathrm{d}x = \frac{\rho_0 b}{\varepsilon_0}\left(a - \frac{2}{3}b\right).
```

As electrons enter through the grounded plane $z = 0$ ($\phi_{\text{DC}}(x, 0) = 0$) into the interior potential trough, relativistic energy conservation $\bigl(\gamma(x) - \gamma_0\bigr)m_e c^2 = -e\phi_{\text{DC}}(x)$ with $\mathrm{d}\gamma = \gamma_0^3 (u_0/c^2)\,\mathrm{d}u$ reduces the cross-section-averaged axial drift velocity from $u_0$ to $u_{\text{dep}} = u_0(1 - \delta_{u,\text{DC}})$ and increases the steady interior charge density by DC continuity ($J_{0,z} = \rho_{\text{dep}} u_{\text{dep}} = \rho_0 u_0$):

```math
\delta_{u,\text{DC}} = \frac{u_0 - u_{\text{dep}}}{u_0} = \frac{-e\langle\phi_{\text{DC}}\rangle_{|x|\le b}}{\gamma_0^3 m_e u_0^2} = \frac{\omega_p^2\,b\left(a - \dfrac{2}{3}b\right)}{\gamma_0^3 u_0^2}, \qquad \rho_{\text{dep}} = \rho_0\,\frac{u_0}{u_{\text{dep}}}, \qquad \omega_{p,\text{dep}} = \omega_p\sqrt{\frac{u_0}{u_{\text{dep}}}}.
```

Because $\beta_q = R(\omega, u, b, a)\,\omega_p / (\gamma^{3/2} u)$ depends on both $u$ and $\rho$, this DC depression shifts $\beta_q$ upward by $\mathcal{O}(2\delta_{u,\text{DC}})$ independently of grid refinement. To compare both codes on the identical physical equilibrium without confounding boundary artifacts, we adopt **Option 3(ii)**: both the analytical reference wavenumber $\beta_{q,\text{dep}}$ and the Eulerian `LinearizedBeam` equilibrium are evaluated at the depressed state $(u_{\text{dep}}, \rho_{\text{dep}}, \gamma_{\text{dep}})$ that the particles experience in the interior of the tunnel, while the particle run directly measures $\langle u_z\rangle(z)$ and $\rho_{\text{DC}}(z)$ over $z \in [0.2 L_z, 0.8 L_z]$ and verifies agreement with $\delta_{u,\text{DC}}$. Neutralization via an external field `E_ext(x)` or `solver._particles_q_bg` (Option 3(i)) is not used because:
- A 1D transverse `E_ext(x)` only acts during particle field gather (not on `solver.E`) and cannot cancel the 2D axial cathode image-charge sheath field $E_{z,\text{DC}}(x, z) = -\partial_z\phi_{\text{DC}}(x, z)$ near $z = 0$ that decelerates entering electrons along $z$.
- The solver's fixed background array `solver._particles_q_bg` is used only in the diagnostic `gauss_residual()` check and does not enter Ampère's law $\partial_t \mathbf{E} = c^2\nabla\times\mathbf{B} - \mathbf{J}/\varepsilon_0$; pre-filling discrete macroparticle slices at $t = 0$ with $\mathbf{E}(0) = \mathbf{0}$ imprints a static grid-scale Moiré pattern into $\mathbf{q}_{\text{bg}}$ and impulsively excites undamped PEC cavity modes at $t = 0$.

##### 4. Standing-wave current envelope, effective launch offset $z_0$, and spatial spectrum

Suppose the electron beam is driven at the entrance with a pure sinusoidal density modulation at frequency $\omega$:

```math
I(0, t) = I_0\bigl(1 + m\sin(\omega t)\bigr), \qquad \rho_1(0, t) = m\rho_{\text{dep}}\sin(\omega t), \qquad v_1(0, t) = 0, \qquad m \ll 1.
```

In `LinearizedBeam`, this entrance drive is supplied through the public `inflow_rho1` callable parameter, which injects the upstream ghost-edge density perturbation at $z_{-1} = -\Delta z / 2$ into the Crank–Nicolson continuity step using the library's own `_c_rho` advection coefficient and `_w_perp_eff` transverse profile (`tests/test_090_linearized_beam_inflow_drive.py`). Near the grounded PEC entrance plane $z = 0$, image-charge screening of the transverse electric field $E_x(x, 0, t) = 0$ over the evanescent cathode zone $\ell_{\text{cath}} \sim 1/T_1$ together with weak higher-order slab modes ($n \ge 2$) excited by the uniform top-hat profile introduces a small effective launch offset $z_0$ into the fundamental space-charge standing wave:

```math
I_1(z, \omega) \simeq I_1(0)\cos\!\bigl(\beta_q(z - z_0)\bigr)\,\mathrm{e}^{-\mathrm{i}\beta_e z}, \qquad |I_1(z)|^2 \simeq A\cos\!\bigl(2\beta_q(z - z_0)\bigr) + c.
```

Consequently:
1. **Standing-wave envelope and all-nulls least-squares fit**: The squared current envelope $|I_1(z)|^2$ exhibits periodic interference nulls at
   ```math
   z_m = z_0 + \left(m + \frac{1}{2}\right)\frac{\pi}{\beta_q}, \qquad m = 0, 1, \dots,
   ```
   Fitting $z_m = z_0 + (m + \tfrac{1}{2})L_{\text{half}}$ ($L_{\text{half}} = \pi/\beta_q$) by ordinary least squares over all interior nulls in $[0, z_{\max}]$ determines $\beta_q = \pi/L_{\text{half}}$, the effective launch offset $z_0$, and the per-null residuals $r_m = z_m - \bigl(z_0 + (m + \tfrac{1}{2})L_{\text{half}}\bigr)$ (`fit_beta_q_from_envelope`). Because both solvers experience the grounded PEC entrance boundary condition $E_x(x, 0, t) = 0$ over the evanescent cathode zone $\ell_{\text{cath}} \sim 1/T_1$ and share the same top-hat multi-mode launch profile, their launch offsets agree within the half-cell entrance stagger plus cathode-sheath depression scale, $|z_{0,\text{part}} - z_{0,\text{LB}}| \le \frac{1}{2}\Delta z + \delta_{u,\text{DC}}/\beta_q$.
2. **Spatial wavenumber spectrum**: The spatial Fourier transform over $z \in [0, \lambda_q]$,
   ```math
   \hat{I}_1(k_z) = \int_0^{\lambda_q} I_1(z, \omega)\,\mathrm{e}^{+\mathrm{i}k_z z}\,\mathrm{d}z,
   ```
   splits into a symmetric doublet of peaks at $k_{\text{fast}} = \beta_e - \beta_q$ and $k_{\text{slow}} = \beta_e + \beta_q$, separated by a deep null at the carrier wavenumber $k_z = \beta_e$.

##### 5. Small-signal linearity condition, ballistic bunching, and $m$-scan verification

In the Lagrangian particle description (`wakis.particles`), the convective nonlinearities $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$ are retained automatically. For a pure entrance density modulation $I(0, t) = I_0(1 + m\sin\omega t)$ with $v_1(0, t) = 0$, the first-order relative density and velocity perturbations along the drift tunnel are

```math
\frac{\rho_1(z, t)}{\rho_{\text{dep}}} = m\cos(\beta_q z)\sin(\omega t - \beta_e z), \qquad \frac{v_1(z, t)}{u_{\text{dep}}} = -m\,\frac{\beta_q}{\beta_e}\sin(\beta_q z)\cos(\omega t - \beta_e z).
```

Along unperturbed trajectories $t(z, t_0) \simeq t_0 + z/u_{\text{dep}} - u_{\text{dep}}^{-2}\int_0^z v_1(z', t_0 + z'/u_{\text{dep}})\,\mathrm{d}z'$, the velocity modulation produces an arrival-phase bunching parameter $X(z) \sim \frac{1}{2}m\beta_q z$ in the charge-conservation mapping $I(z, t)\,\mathrm{d}t = I(0, t_0)\,\mathrm{d}t_0$ [Webster, 1939; Paschke, 1957]. Expanding the harmonic series ($2 J_2(X)/J_1(X) \simeq X$ for $X \ll 1$) gives a second-harmonic current ratio scaling linearly with $m$:

```math
\frac{|I_2(z)|}{|I_1(0)|} \sim X(z) \sim \frac{1}{2}\,m\,\beta_q z, \qquad \max_{z \in [0, \lambda_q]} \frac{|I_2(z)|}{|I_1(0)|} \le \frac{1}{2}\,m\,\beta_q\lambda_q = \pi m.
```

To verify small-signal linearity directly (`test_dc_depression_and_dft_leakage_and_linearity`), the suite runs an $m$-scan at $m$ and $m/2$ and checks that:
1. The fitted reduced plasma wavenumber $\beta_q$ is invariant between $m$ and $m/2$ within the noise budget $\varepsilon_{\text{noise}}$.
2. In `ParticleSolver`, $\max_{z \in [0, \lambda_q]} |I_2(z)|/|I_1(0)| \le \pi m$ and the coherent quadrature ballistic-bunching component of $|I_2(z)|/|I_1(0)|$ scales linearly with $m$ (doubling between $m/2$ and $m$).
3. In `LinearizedBeam` (which is strictly linear in $(\rho_1, v_1)$), the steady-state $2\omega$ response is identically zero; the observed ratio $|I_2|/|I_1(0)|$ is an $m$-independent start-up transient leaking from the $C^1$ raised-cosine turn-on ramp ($N_{\text{rise}} = 3$ RF periods) across the finite $N_{\text{per}} = 6$-period rectangular DFT window, bounded by $\varepsilon_{\text{tr},2\omega} = \bigl[(2 N_{\text{rise}})^2 \pi N_{\text{per}}\bigr]^{-1}$.

##### 6. Numerical setup, boundaries, injection, and term-by-term error budget

To compare `ParticleSolver` and `LinearizedBeam` cleanly (`examples/_space_charge_waves_common.py`, `tests/test_087_particles_vs_linearized_beam.py`, `examples/024_particles_space_charge_waves.py`):

1. **Shared grid, integer-period time step, and DFT mean subtraction**: Both solvers run on the identical `GridFIT3D` ($x \in [-a, a]$ with `bc="pec"`, $y \in [-L_y/2, L_y/2]$ with `ny = 1` and `bc="periodic"`, $z \in [0, L_z]$ with `bc="pec"`). To eliminate spectral leakage of the large DC beam current $I_0 \gg I_1(0)$ into the fundamental DFT at $\omega$, the time step is chosen as $\Delta t = 1/(f N_{\text{spp}})$ with $N_{\text{spp}} = \lceil 1/(f\Delta t_{\text{CFL}})\rceil$ ($\Delta t \le \Delta t_{\text{CFL}}$), so the 6-period DFT window $N_{\text{DFT}} = 6 N_{\text{spp}}$ holds an exact integer number of RF cycles ($\bigl|\sum_n \mathrm{e}^{-\mathrm{i}\omega t^n}\bigr| \sim \epsilon_{\text{mach}}$). In addition, each axial edge's time mean over the $N_{\text{DFT}}$ window is subtracted before evaluating the $1\omega$ and $2\omega$ DFTs, bounding residual DC leakage by $\varepsilon_{\text{DFT}} = 100\,\epsilon_{\text{mach}}/m$.
2. **Outflow boundary treatment and fit-window margin**: Both runs use identical PEC/periodic field boundary conditions on `SolverFIT3D`. In the particle run, `ParticleBoundaries` absorbs macroparticles reaching $z = L_z$ (`z_high`) via exact ray-cast segment clipping. In the fluid run, `LinearizedBeam` applies a quarter-wave sine taper to the background density and velocity coupling over the final `ramp_cells` cells (`wakis/beam_fluid.py:884`):
   ```math
   w_{\text{ax}}(m) = \sin\!\left(\frac{\pi}{2}\,\frac{N_z - m}{N_{\text{ramp}} + 1}\right), \qquad m = N_z - N_{\text{ramp}}, \dots, N_z - 1,
   ```
   paired with a one-sided upwind difference at $m = N_z - 1$. The envelope and spectrum fit window $z \in [0, \lambda_q]$ terminates upstream of the outflow ramp $z_{\text{ramp}} = L_z - L_{\text{ramp}}$ with a positive margin $(L_z - L_{\text{ramp}}) - \lambda_q \ge 2.5\text{ mm} > 5\Delta z_{\text{coarse}}$.
3. **Time-modulated weight injection and 8-seed empirical noise**: Every time step, `BeamInjector.inject` (`wakis/particles/boundaries.py:759`) places `n_macro_per_step` macroparticles on the entrance plane $z = z_0 = 0$ with longitudinal momentum $u_z = \gamma_0 u_0$, $\mathbf{u}_\perp = \mathbf{0}$, and time-modulated charge weight
   ```math
   w(t^n) = \frac{|I(t^n)|\,\Delta t}{|q|\,N_{\text{macro}}}.
   ```
   Transverse coordinates $x_p \in [-b, b]$, $y_p = 0$ are placed using `BeamInjector` planar sheet profiles (`wakis/particles/boundaries.py`): `profile="sheet_lattice"` (deterministic stratum midpoints $x_k = b(2(k+1/2)/N_{\text{macro}} - 1)$, zero transverse shot noise, identical for all seeds) for the deterministic reference run, signed physics checks, particle mutations, two-resolution check, and $m$-scan; and `profile="sheet"` (one seeded uniform draw per stratum per step from `_step_rng`) for the 8-seed noise ensemble (`measure_empirical_noise`).
4. **Guide magnetic field and Boris cyclotron step**: The axial guide field $B_{z,\text{ext}} = 50\text{ T}$ freezes out transverse space-charge expansion ($\omega_c / \omega_p \gg 1$). On the coarse and fine grids, the dimensionless cyclotron step is $\omega_c \Delta t \approx 3.5$ (coarse) and $\omega_c \Delta t \approx 1.8$ (fine). Although the explicit Boris rotation aliases the Larmor gyration phase when $\omega_c \Delta t \gtrsim 2$ ($\omega_{\text{eff}}\Delta t = 2\arctan(\omega_c\Delta t/2)$), the Boris push is unconditionally energy-conserving in a uniform magnetic field, reproduces the exact guiding-center $\mathbf{E}\times\mathbf{B}$ drift, and suppresses transverse velocity excursions $\mathbf{v}_\perp$ for a beam injected with $\mathbf{v}_\perp(0) = \mathbf{0}$ whose space-charge wave dynamics are purely longitudinal along $\hat{\mathbf{z}}$ [Birdsall & Langdon, 1991].
5. **Derived term-by-term error budget and convergence-order bands**: Unlike the Pierce travelling-wave-tube helper (which divides $u_0$ by $\sin(\beta_e\Delta z)/(\beta_e\Delta z)$ to maintain phase synchronism with an external slow-wave circuit), both solvers here use the physical equilibrium velocity $u_{\text{dep}}$, and all operator discretization terms are included explicitly in the error budget (`derived_tolerances`):
   - *Shared `SolverFIT3D` field operators (`eps_x_fit`, `eps_z_fit` — in BOTH predictions)*: Both codes couple to the same `SolverFIT3D` fields. From $R^2 = \tau^2/(\tau^2 + T_1^2)$, a relative error $-(T_1\Delta x)^2/12$ in the 3-point transverse FIT Laplacian $T_1^2$ shifts $\beta_q$ upward in **both** codes by
     ```math
     \varepsilon_{x,\text{FIT}} = +\frac{1}{2}\bigl(1 - R^2\bigr)\frac{(T_1\Delta x)^2}{12}.
     ```
     Simultaneously, the staggered Yee axial curl-curl stencil in `SolverFIT3D` evaluates axial derivatives with symbol $\hat{k}_{z,\text{FIT}}(k) = \frac{2}{\Delta z}\sin(k\Delta z/2) = k\bigl(1 - (k\Delta z)^2/24 + \mathcal{O}(\Delta z^4)\bigr)$, reducing the effective axial decay constant $\tau$ in the field Helmholtz equation and shifting $R(\tau)$ downward in **both** codes by
     ```math
     \varepsilon_{z,\text{FIT}} = \frac{\mathrm{d}\ln R}{\mathrm{d}\ln\tau}\,\frac{(\beta_e\Delta z)^2}{24}, \qquad \frac{\mathrm{d}\ln R}{\mathrm{d}\ln\tau} = \bigl(1 - R^2\bigr)\!\left(1 - \frac{1 - 2\tau(a-b)/\sinh(2\tau(a-b))}{1 + 2 T_1 b/\sin(2 T_1 b)}\right).
     ```
   - *Eulerian advection and numerical carrier wavenumber in `LinearizedBeam` (`eps_z_lb`, `eps_tau_lb` — in `LinearizedBeam` only)*: Centered axial differences $(f_{m+1} - f_{m-1})/(2\Delta z)$ have discrete group velocity $u_g = u_{\text{dep}}\cos(\beta_e\Delta z)$, shifting $\beta_{q,\text{LB}} = \omega_q/u_g$ upward by $\varepsilon_{z,\text{LB}} = \frac{1}{\cos(\beta_e\Delta z)} - 1 = \frac{1}{2}(\beta_e\Delta z)^2 + \mathcal{O}(\Delta z^4)$. In addition, `SolverFIT3D` sees `LinearizedBeam`'s numerical carrier wavenumber $k_{\text{car}} = \arcsin(\beta_e\Delta z)/\Delta z = \beta_e\bigl(1 + (\beta_e\Delta z)^2/6 + \mathcal{O}(\Delta z^4)\bigr)$ rather than $\beta_e$, which increases $\tau$ and shifts $R(\tau)$ upward by $\varepsilon_{\tau,\text{LB}} = \frac{\mathrm{d}\ln R}{\mathrm{d}\ln\tau}\frac{(\beta_e\Delta z)^2}{6}$. Combining the `LinearizedBeam`-specific and shared field-solver terms gives the signed leading prediction:
     ```math
     \delta_{\text{LB,lead}} = +\varepsilon_{z,\text{LB}} + \varepsilon_{\tau,\text{LB}} - \varepsilon_{z,\text{FIT}} + \varepsilon_{x,\text{FIT}} > 0.
     ```
   - *CIC smoothing in `ParticleSolver` (`eps_cic` — in `ParticleSolver` only)*: Lagrangian particles advect in continuous space with carrier wavenumber $\beta_e$, and staggered axial CIC field gather plus Esirkepov current deposition satisfy $\kappa_z^2(\beta_e\Delta z) \equiv 1$ (`wakis/particles/solver.py:420`), while nodal transverse CIC deposit/gather ($p = p_g = 1$) attenuates $T_1$ by $\kappa_x^2(T_1\Delta x) = 1 - \frac{2}{3}\sin^2(T_1\Delta x/2)$, shifting $\beta_q$ downward by $\varepsilon_{\text{CIC}} = 1 - \sqrt{1 - \frac{2}{3}\sin^2(T_1\Delta x/2)}$. Combining `-eps_cic` with the shared `SolverFIT3D` terms gives the signed leading prediction:
     ```math
     \delta_{\text{Part,lead}} = -\varepsilon_{\text{CIC}} - \varepsilon_{z,\text{FIT}} + \varepsilon_{x,\text{FIT}} < 0.
     ```
   - *DC space-charge depression tolerance (`dep_tol`, `eps_dc_res`)*: Self-consistent quadratic density enhancement $\delta_{u,\text{DC}}^2$ plus transverse grid/CIC quadrature $(\Delta x/(2b))^2\delta_{u,\text{DC}}$ across the beam thickness $2b$ bounds the DC depression difference $|\delta_{u,\text{meas}} - \delta_{u,\text{DC}}|$ and $|\delta_{\rho,\text{meas}} - \delta_{u,\text{DC}}|$ (with $\rho_{\text{DC}}$ measured directly from the deposited particle charge per cell) by $\text{dep\_tol} = \delta_{u,\text{DC}}^2 + (\Delta x/(2b))^2\delta_{u,\text{DC}}$, which shifts $\beta_q$ by at most $\varepsilon_{\text{DC,res}} = (\mathrm{d}\ln\beta_q/\mathrm{d}\delta_u)\,\text{dep\_tol}$.
   - *Multi-mode envelope-fit shift (`eps_fit`)*, *residual DFT leakage (`eps_dft`)*, and *empirical 8-seed noise (`eps_noise`)*.
   - *Total asserted bounds*:
     ```math
     \text{tol}_{\text{LB}} = |\delta_{\text{LB,lead}}| + \varepsilon_{\text{fit}} + \varepsilon_{\text{DFT}}, \qquad \text{tol}_{\text{Part}} = |\delta_{\text{Part,lead}}| + \varepsilon_{\text{DC,res}} + \varepsilon_{\text{fit}} + \varepsilon_{\text{DFT}} + \varepsilon_{\text{noise}}, \qquad \text{tol}_{\text{cross}} = \text{tol}_{\text{LB}} + \text{tol}_{\text{Part}}.
     ```
   - *Entrance current amplitude bounds (`tol_amp_lb`, `tol_amp_part`)*: At the first primary edge midpoint $z_0 = \Delta z / 2$, the standing-wave envelope varies from $z = 0$ by $1 - \cos(\beta_q\Delta z) \simeq \frac{1}{2}(\beta_q\Delta z)^2$, while transverse CIC deposition across the beam boundary cells $x = \pm b$ adds the quadrature bound $\frac{1}{2}(\Delta x / (2b))^2$ in the particle run.
   - *Derived convergence-order bands*: Across the $2\times$ grid refinement from `COARSE_GRID` to `FINE_GRID`, the allowed convergence-order band for each solver and for the cross-code difference is derived directly from the signed leading predictions $e_c, e_f$ and their slacks $s_c, s_f$:
     ```math
     p \in \left[\log_2\!\left(\frac{|e_c| - s_c}{|e_f| + s_f}\right),\; \log_2\!\left(\frac{|e_c| + s_c}{|e_f| - s_f}\right)\right].
     ```
     To inspect the printed budget and run the suite:
     ```bash
     pytest tests/test_087_particles_vs_linearized_beam.py -v -s
     python examples/024_particles_space_charge_waves.py
     ```

##### 7. Limits & Failure Modes

- **Small-signal linear regime only ($m \ll 1$, $X(z) \ll 1$)**: `LinearizedBeam` linearizes the cold-fluid equations around a time-independent equilibrium and omits the quadratic convective terms $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$. When the modulation index $m$ or drift length $z$ is increased until the bunching parameter $X \sim \mathcal{O}(1)$, fast electrons overtake slow electrons (wave breaking / multi-stream trajectory crossing), generating strong higher harmonics ($2\omega, 3\omega$) and electrostatic trapping in `ParticleSolver` that `LinearizedBeam` cannot represent.
- **Planar sheet-beam geometry and strong axial magnetic focusing**: The analytical reduction factor `reduction_factor_planar` assumes a 1D planar sheet beam uniform along $y$ (`ny = 1`, `bc="periodic"`) and confined by an axial guide field $B_{z,\text{ext}}$ strong enough ($\omega_c \gg \omega_p$) to freeze out transverse velocity perturbations $\mathbf{v}_\perp$. At weak or zero guide field, transverse space-charge forces cause beam spreading or Brillouin scalloping and couple longitudinal plasma oscillations to transverse cyclotron/diocotron modes; 3D pencil or rectangular beams require `reduction_factor_cylindrical` or `reduction_factor_rectangular` [Branch & Mihran, 1955].
- **Mildly relativistic operating point and longitudinal mass**: At $V_0 = 10\text{ kV}$ ($\gamma_0 \approx 1.0196$), the longitudinal relativistic mass factor $\gamma_0^{3/2}$ in $\beta_q = R\omega_p/(\gamma_0^{3/2}u_0)$ differs from the transverse factor $\gamma_0^{1/2}$ by $\approx \gamma_0 - 1 \approx 2\%$, which is smaller than the coarse discretization tolerance bounds. Consequently, this $10\text{ kV}$ benchmark validates the coupled space-charge-wave dispersion $\beta_e \pm \beta_q$ and geometric reduction factor $R$, but does **not** independently isolate the $\gamma_0^3 m_e$ relativistic longitudinal mass exponent (which requires a higher-voltage operating point $V_0 \ge 100\text{ kV}$).
- **What this cross-check does not test — synchronous beam–circuit coupling**: A smooth parallel-plate PEC drift tunnel supports only fast waveguide modes ($v_p > c > u_0$, or cutoff evanescent modes when $f < f_{c,10} = c/(4a)$), which have zero synchronous Pierce interaction impedance ($K_n = 0$). Thus this benchmark tests passive space-charge-wave propagation rather than convective traveling-wave-tube (TWT) gain or backward-wave-oscillator (BWO) interaction.

References:
- S. Ramo, "Space Charge and Field Waves in an Electron Beam", *Phys. Rev.* **56**(3), 276–283 (1939), [doi:10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276).
- W. C. Hahn, "Small Signal Theory of Velocity-Modulated Electron Beams", *Gen. Elec. Rev.* **42**, 258–270 (1939).
- D. L. Webster, "Cathode-Ray Bunching", *J. Appl. Phys.* **10**(7), 501–508 (1939), [doi:10.1063/1.1707335](https://doi.org/10.1063/1.1707335).
- G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in Electron Beams", *IRE Trans. Electron Devices* **2**(2), 3–11 (1955), [doi:10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065).
- F. Paschke, "On the Nonlinear Behavior of Electron-Beam Devices", *RCA Rev.* **18**(2), 221–242 (1957).
- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, Taylor & Francis / IOP Publishing, Bristol (1991), ISBN 978-0-7503-0117-6.
- T. Zh. Esirkepov, "Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor", *Comput. Phys. Commun.* **135**(2), 144–153 (2001), [doi:10.1016/S0010-4655(00)00228-9](https://doi.org/10.1016/S0010-4655(00)00228-9).
