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

##### 3. DC space-charge potential and exact energy-conserving velocity depression in an unneutralized sheet beam

In `ParticleSolver`, the beam is injected into an empty PEC tunnel without a neutralizing ion background, so the steady DC beam charge density $\rho_0 < 0$ generates a transverse electrostatic potential trough $\phi_{\text{DC}}(x) < 0$ across the tunnel. Away from the entrance and exit planes ($z \gg a$, $L_z - z \gg a$), solving the 1D Poisson equation

```math
\frac{\mathrm{d}^2 \phi_{\text{DC}}}{\mathrm{d}x^2} = -\frac{\rho_0}{\varepsilon_0} \quad (|x| \le b), \qquad \frac{\mathrm{d}^2 \phi_{\text{DC}}}{\mathrm{d}x^2} = 0 \quad (b < |x| \le a),
```

subject to $\phi_{\text{DC}}(\pm a) = 0$, $\mathrm{d}\phi_{\text{DC}}/\mathrm{d}x\big|_{x=0} = 0$, and continuity of $\phi_{\text{DC}}$ and $\mathrm{d}\phi_{\text{DC}}/\mathrm{d}x$ at $x = \pm b$ gives the undepressed slab potential profile

```math
\phi_0(x) = \frac{\rho_0}{2\varepsilon_0}\bigl(2 a b - b^2 - x^2\bigr) \quad (|x| \le b), \qquad \langle \phi_0 \rangle_{|x|\le b} = \frac{1}{2b}\int_{-b}^{b} \phi_0(x)\,\mathrm{d}x = \frac{\rho_0 b}{\varepsilon_0}\left(a - \frac{2}{3}b\right).
```

As electrons enter through the grounded plane $z = 0$ ($\phi_{\text{DC}}(x, 0) = 0$) into the interior potential trough, exact relativistic energy conservation along each beam line $\ell$ (at transverse coordinate $x_\ell$) determines the depressed Lorentz factor $\gamma_\ell$ and axial drift velocity $u_\ell$, while DC charge continuity ($J_{0,z,\ell} = \rho_{\text{dep},\ell} u_\ell = \rho_{0,\ell} u_0$) scales each line's charge density by $u_0 / u_\ell$. Both the continuum depression $\delta_{\text{th}}$ (`compute_analytic_parameters`) and the profile-consistent discrete 1D 3-point Poisson depression $\delta_{\text{disc}}$ (`compute_discrete_dc_depression`, where at each fixed-point iteration the 3-point discrete Laplacian on $N_x$ cells with PEC boundary conditions $\phi(\pm a) = 0$ is solved with the order-$p$ nodal B-spline depressed charge deposit $\propto w_\ell (u_0 / u_\ell)$ and gathered back onto the actual line positions $x_\ell$ with the same order-$p$ nodal B-spline weight matrix `transverse_weight_matrix(x_lines, nx, p)` built from `bspline_weights_1d(xi, p, staggered=False)` at $\xi = (x + a)/\Delta x$) iterate the per-line energy-conserving fixed point to machine precision ($10^{-14}$ tolerance):

```math
\mathbf{L}_x\,\boldsymbol{\phi}_{\text{nodes}} = -\frac{\boldsymbol{\rho}_{\text{nodes}}(\mathbf{u})}{\varepsilon_0}, \qquad \gamma_\ell = \gamma_0 - \frac{e\,|\phi_\ell|}{m_{\text{eff}} c^2}, \qquad u_\ell = c\sqrt{1 - \frac{1}{\gamma_\ell^2}}, \qquad \bar{\delta} = 1 - \frac{\sum_\ell w_{\text{avg},\ell} u_\ell}{u_0 \sum_\ell w_{\text{avg},\ell}}.
```

Three line-weighting conventions (`weighting` in `compute_discrete_dc_depression`, verified in `tests/test_094_dc_depression_weighting.py`) are evaluated and reported in `DEPCHECK` for normalized injected line current fractions $\rho_\ell$ ($\sum_\ell \rho_\ell = 1$):
- **Steady-state line-charge weighting (`weighting="charge"`, $w_{\text{avg},\ell} = \rho_\ell / u_\ell$, `delta_disc_charge`) — harmonic mean**: In steady state, a stream $\ell$ injected with current fraction $\rho_\ell$ and drifting through the interior measurement slice $z \in [0.2 L_z, 0.8 L_z]$ at velocity $u_\ell$ spends transit time $\propto 1/u_\ell$ in the slice, so its resident line charge density (and accumulated macroparticle weight $\sum_{p \in \ell} w_p$ at any instant) is proportional to $\rho_\ell / u_\ell$. Consequently, the bin-accumulated particle estimator `u_z_mean` (`sum(w_p v_{z,p}) / sum(w_p)` over $z \in [0.2 L_z, 0.8 L_z]$) weights each stream by $w_{\text{avg},\ell} = \rho_\ell / u_\ell$, yielding the **harmonic mean** of the per-line drift velocities $u_\ell$:
  ```math
  \bar{u}_{\text{charge}} = \frac{\sum_\ell (\rho_\ell / u_\ell)\,u_\ell}{\sum_\ell (\rho_\ell / u_\ell)} = \frac{\sum_\ell \rho_\ell}{\sum_\ell (\rho_\ell / u_\ell)}, \qquad 1 - \delta_{\text{disc,charge}} = \frac{\sum_\ell \rho_\ell}{u_0 \sum_\ell (\rho_\ell / u_\ell)}.
  ```
  `delta_disc_charge` is the reference asserted against the particle measurement in `DEPCHECK` across the coarse and fine `"sheet_lattice"` runs, the 8-seed `"sheet"` ensemble, and the center-peaked Fibonacci disk profile in `DEPCHECK id=M3`.
- **Injected-fraction density weighting (`weighting="density"`, $w_{\text{avg},\ell} = \rho_\ell$, `delta_disc_density`)**: Arithmetic mean weighted by the injected line fractions $\rho_\ell$.
- **Current/flux weighting (`weighting="flux"`, $w_{\text{avg},\ell} = \rho_\ell u_\ell$, `delta_disc_flux`, default `delta_disc`)**: Weights by $\rho_\ell u_\ell$. Whenever $u_\ell$ varies across lines, Jensen's inequality enforces the strict ordering $\delta_{\text{disc,charge}} > \delta_{\text{disc,density}} > \delta_{\text{disc,flux}}$.

At the converged continuum equilibrium $u_{\text{dep}} = u_0(1 - \delta_{\text{th}})$, the depressed parameters passed to `LinearizedBeam` (Option 3(ii)) are

```math
\gamma_{\text{dep}} = \frac{1}{\sqrt{1 - (u_{\text{dep}}/c)^2}}, \qquad \rho_{\text{dep}} = \rho_0\,\frac{u_0}{u_{\text{dep}}}, \qquad \omega_{p,\text{dep}} = \omega_p\sqrt{\frac{u_0}{u_{\text{dep}}}}.
```

For the deterministic `"sheet_lattice"` profile (and **M3**), the measured interior DC velocity depression $\delta_{\text{meas,lattice}}$ is checked against the charge-weighted discrete prediction $\delta_{\text{disc,charge}}$ up to the second-order AC velocity-modulation scale $\varepsilon_{\text{AC}}$ (which covers the charge-weighted correlation $\langle\rho_1 v_1\rangle/(\rho_0 u) = -\langle v_1^2\rangle/u^2 \le \frac{1}{2}(m\beta_q/\beta_e)^2$; for the 8-seed `"sheet"` ensemble, the bound adds $t_{0.995,7}\,\text{SEM}$ for seed scatter):

```math
\bigl|\delta_{\text{meas,lattice}} - \delta_{\text{disc,charge}}\bigr| \le \varepsilon_{\text{AC}} + 10^{-12}, \qquad \varepsilon_{\text{AC}} = \frac{1}{2}\left(m\,\frac{\beta_q}{\beta_e}\right)^2.
```

Pre-filling `solver._particles_q_bg` (Option 3(i)) is not used because `solver._particles_q_bg` only enters `gauss_residual()` and not Ampère's law $\partial_t \mathbf{E} = c^2\nabla\times\mathbf{B} - \mathbf{J}/\varepsilon_0$.

##### 4. Two-wave variable-projection fit and spatial wavenumber spectrum

Suppose the electron beam is driven at the entrance with a pure sinusoidal density modulation at angular frequency $\omega$:

```math
I(0, t) = I_0\bigl(1 + m\sin(\omega t)\bigr), \qquad \rho_1(0, t) = m\rho_{\text{dep}}\sin(\omega t), \qquad v_1(0, t) = 0, \qquad m \ll 1.
```

In `LinearizedBeam`, this entrance drive is supplied through the public `inflow_rho1` callable parameter, which injects the upstream ghost-edge density perturbation at $z_{-1} = -\Delta z / 2$ into the Crank–Nicolson continuity step (`tests/test_090_linearized_beam_inflow_drive.py`). Inside the drift tunnel, the fundamental space-charge wave is a superposition of the fast ($k_f$) and slow ($k_s$) waves:

```math
I_1(z) = a_f\,\mathrm{e}^{-\mathrm{i} k_f z} + a_s\,\mathrm{e}^{-\mathrm{i} k_s z} \equiv \mathrm{e}^{-\mathrm{i} k_c z}\bigl[A\cos(\beta_q z) + B\sin(\beta_q z)\bigr],
```

with complex wave amplitudes $a_f, a_s \in \mathbb{C}$, real wavenumbers $k_f < k_s$, and the exact algebraic identities

```math
k_c = \frac{k_f + k_s}{2}, \qquad \beta_q = \frac{k_s - k_f}{2}, \qquad A = a_f + a_s, \qquad B = \mathrm{i}(a_s - a_f).
```

1. **Complex two-wave variable-projection fit (`fit_beta_q_from_envelope`)**:
   - **Shared fit window (`fit_window(params, outflow_ramp_m)`) derived from discrete evanescent near-field decay**: On each grid $(N_x, 1, N_z)$ with $\Delta x = 2a/N_x$, $\Delta z = L_z/N_z$, and integer-period time step $\Delta t$, a localized discontinuity at the PEC entrance ($z = 0$), the `LinearizedBeam` outflow ramp start ($z_{\text{ramp}} = L_z - L_{\text{ramp}}$), or the particle collector ($z = L_z$) perturbs the field over the discrete axial decay length of the slowest even $\mathrm{TM}$ evanescent waveguide mode of the parallel-plate tunnel at frequency $\omega$:
     ```math
     \lambda_x = \left(\frac{2}{\Delta x}\right)^2\sin^2\!\left(\frac{\pi\Delta x}{4a}\right), \qquad \gamma_{1,\text{disc}} = \frac{2}{\Delta z}\,\operatorname{arsinh}\!\left(\frac{\Delta z}{2}\sqrt{\lambda_x - \left(\frac{\Omega_f}{c}\right)^2}\right), \qquad \kappa_{\text{DC}} = T_1\sqrt{1 - \left(\frac{\omega_{p,\text{dep}}}{u_{\text{dep}} T_1}\right)^2} > \gamma_{1,\text{disc}},
     ```
     where $\Omega_f = (2/\Delta t)\sin(\omega\Delta t/2)$ uses that grid's $\Delta t$ and beam loading only adds faster-decaying transients ($\kappa_{\text{DC}} > \gamma_{1,\text{disc}}$, verified as an identity check and printed in `TERMS`). Requiring the relative evanescent near-field amplitude at the window edges to decay below the design choice $\varepsilon_w = 10^{-5}$ (`EPS_WINDOW = 1e-5`) gives the per-grid margin $d_{w,\text{grid}} = \ln(1/\varepsilon_w)/\gamma_{1,\text{disc}}$, and taking the largest $d_{w,\text{grid}}$ (smallest $\gamma_{1,\text{disc}}$ across the coarse and fine grids) defines the single common fit window used across all fits:
     ```math
     d_{w,\text{common}} = \max_{\text{coarse, fine}} d_{w,\text{grid}}, \qquad z_{\min} = d_{w,\text{common}}, \qquad z_{\max} = (L_z - L_{\text{ramp}}) - d_{w,\text{common}}, \qquad z_{\max} - z_{\min} \ge 1.25\,\lambda_q,
     ```
     with the implied window-edge $\beta_q$ bias bound $\text{eps\_w\_bias} \approx \varepsilon_w\,(1/\gamma_{1,\text{disc}})/(z_{\max} - z_{\min})$ printed in `TERMS` as a report-only diagnostic.
   - **Fitted carrier $k_c$ and variable projection**: The carrier wavenumber $k_c$ is fitted simultaneously with $\beta_q$ rather than fixed to $\beta_e = \omega/u_{\text{dep}}$ because `LinearizedBeam`'s centered Eulerian advection shifts its numerical carrier by $\simeq \frac{1}{6}(\beta_e\Delta z)^2$, and fixing $k_c = \beta_e$ would bias $\beta_q$. For any trial pair $(k_c, \beta_q)$, the linear parameters $(a_f, a_s)$ follow from a 2-column complex linear least-squares solve (`numpy.linalg.lstsq`) on the design matrix $[\mathrm{e}^{-\mathrm{i} k_f z_k},\,\mathrm{e}^{-\mathrm{i} k_s z_k}]$. Minimizing the complex residual sum of squares $\text{RSS}(k_c, \beta_q) = \sum_k |I_1(z_k) - I_{1,\text{fit}}(z_k)|^2$ on a $201\times 401$ grid over $k_c \in k_{c,0}(1 \pm 0.03)$ (where $k_{c,0}$ is the unwrapped phase slope of $I_1(z)$) and $\beta_q \in [0.5, 1.8]\,\beta_{q,\text{guess}}$ followed by Nelder–Mead polishing (`xatol = 1e-10 * beta_q_guess`) yields $(k_c, \beta_q, A, B, a_f, a_s)$. The fit uncertainty $\sigma_{\beta_q}$ is obtained from the Gauss–Newton covariance $s^2(\mathbf{J}^{\mathsf{H}}\mathbf{J})^{-1}$ with $s^2 = \text{RSS}/(2N - 6)$ ($N$ complex points, $6$ real parameters).
   - **Outflow ramp-sensitivity and parasitic-branch diagnostics (`DIAG id=lb_ramp`, `DIAG id=lb_parasitic`)**:
     - `DIAG id=lb_ramp` compares `LinearizedBeam` on the coarse and fine grids with $L_{\text{ramp}} = 8\,\text{mm}$ and $4\,\text{mm}$, both fitted on the $8\,\text{mm}$ window $[d_w, L_z - 8\,\text{mm} - d_w]$ so both ramps lie at least $d_w$ downstream of $z_{\max}$, alongside the four-wave forward/backward reflection fit (`fit_four_wave_reflection`).
     - `DIAG id=lb_parasitic` (`fit_lb_parasitic_waves`) augments the variable-projection basis for `LinearizedBeam` with the two centered-difference odd–even parasitic branches $k_{p,\mp} \in (\pi/(2\Delta z),\,\pi/\Delta z)$ (propagating with negative group velocity) that solve the discrete `LinearizedBeam` dispersion relation $\Omega_{\text{LB}} \mp \omega_{q,1}(k_p) = u_{\text{dep}}\sin(k_p\Delta z)/\Delta z$ via `scipy.optimize.brentq` with the fundamental transverse eigenvalue $\lambda_1(k_p)$ and reduction factor evaluated at $k_p$, adding two linear complex amplitudes and no new wavenumber parameters. Because the resulting relative shift $|\Delta\beta_q/\beta_q|$ on the coarse, fine, and **M2** runs is orders of magnitude smaller than the `LinearizedBeam` residual, the 2-wave fit is retained as the `LinearizedBeam` estimator across all checks.
   - **Amplitude sanity check vs report-only launch phase**: For a pure density modulation, $|a_f| = |a_s|$ and $B/A = \tan(\beta_q z_0)$ where $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$ is the effective launch offset induced by cathode image-charge screening and weak higher-order modes. Because no closed-form boundary-layer derivation predicts $z_0$ or $B/A$, $|a_s|/|a_f|$, $B/A$, and $z_0$ are reported as diagnostics without assertion, while $|A| > \frac{1}{2}m I_0$ is asserted to verify that the fit locked onto the driven wave rather than noise.
2. **Spatial wavenumber spectrum (`compute_spatial_spectrum`, report-only `DIAG id=spectrum`)**: The spatial Fourier transform over $z \in [0, \lambda_q]$,
   ```math
   \hat{I}_1(k_z) = \int_0^{\lambda_q} I_1(z, \omega)\,\mathrm{e}^{+\mathrm{i}k_z z}\,\mathrm{d}z,
   ```
   splits into a doublet of peaks near $k_f = \beta_e - \beta_q$ and $k_s = \beta_e + \beta_q$ separated by a deep null at $k_z = \beta_e$. Because the rectangular window $[0, \lambda_q]$ begins at $z = 0$ and includes the unpredicted cathode launch offset $z_0$, the two-wave Dirichlet-kernel sidelobe interaction shift of the local spectral maxima cannot be derived in closed form without $z_0$; the peak wavenumbers and valley depth are therefore printed in `DIAG id=spectrum` as report-only diagnostics, while the primary wavenumber estimator is `fit_beta_q_from_envelope` on $[z_{\min}, z_{\max}]$.

##### 5. Small-signal linearity condition, ballistic bunching, and Richardson extrapolation to $m \to 0$

In the Lagrangian particle description (`wakis.particles`), the convective nonlinearities $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$ are retained automatically. For a pure entrance density modulation $I(0, t) = I_0(1 + m\sin\omega t)$ with $v_1(0, t) = 0$, the first-order relative density and velocity perturbations along the drift tunnel are

```math
\frac{\rho_1(z, t)}{\rho_{\text{dep}}} = m\cos(\beta_q z)\sin(\omega t - \beta_e z), \qquad \frac{v_1(z, t)}{u_{\text{dep}}} = -m\,\frac{\beta_q}{\beta_e}\sin(\beta_q z)\cos(\omega t - \beta_e z).
```

Along unperturbed trajectories, the velocity modulation produces an arrival-phase bunching parameter $X(z) \sim \frac{1}{2}m\beta_q z$ in the charge-conservation mapping $I(z, t)\,\mathrm{d}t = I(0, t_0)\,\mathrm{d}t_0$ [Webster, 1939; Paschke, 1957], generating a second-harmonic current ratio $|I_2(z)|/|I_1(0)| \sim X(z)$ and an even-in-$m$ nonlinear amplitude shift $\beta_q(m) = \beta_0 + c\,m^2 + d\,m^4 + \mathcal{O}(m^6)$ starting at third order in the wave amplitude. Because every linear modal prediction $\beta_{\text{pred}}$ applies to the $m \to 0$ limit, every particle check against $\beta_{\text{pred}}$ extrapolates $\beta_q(m)$ to $m \to 0$ first (`compute_m_extrapolation`, `H1CHECK`, `test_m_scan_linearity`):
1. **Three-point Richardson extrapolation on the coarse grid and two-point extrapolation on the fine grid and mutations (`compute_m_extrapolation`)**: The deterministic `"sheet_lattice"` profile is evaluated at $m, m/2, m/4$ on the coarse grid and at $m, m/2$ on the fine grid (and in **M1** and **M3**). On the coarse grid, fitting $\beta_q(m) = \beta_0 + c\,m^2 + d\,m^4$ exactly through $(m, m/2, m/4)$ gives the three-point Richardson extrapolate, two-point remainder $e_R$, and propagated Gauss–Newton standard error $\sigma_{\text{prop}}$:
   ```math
   \beta_{0,\text{3pt}} = \frac{64\,\beta_q(m/4) - 20\,\beta_q(m/2) + \beta_q(m)}{45}, \qquad \beta_{0,\text{2pt}}(m/2, m/4) = \frac{4\,\beta_q(m/4) - \beta_q(m/2)}{3},
   ```
   ```math
   e_R = \bigl|\beta_{0,\text{3pt}} - \beta_{0,\text{2pt}}(m/2, m/4)\bigr|, \qquad \sigma_{\text{prop}} = \sqrt{\left(\frac{\sigma_m}{45}\right)^2 + \left(\frac{20\,\sigma_{m/2}}{45}\right)^2 + \left(\frac{64\,\sigma_{m/4}}{45}\right)^2},
   ```
   where $\sigma_m, \sigma_{m/2}, \sigma_{m/4}$ are the Gauss–Newton standard errors $\sigma_{\beta_q}$ of the three single-amplitude variable-projection fits. The coarse-grid assertion checks
   ```math
   \left|\frac{\beta_{0,\text{3pt}}}{\beta_{\text{pred}}} - 1\right| \le s_{\text{part}} + \frac{e_R + \sigma_{\text{prop}}}{\beta_{\text{pred}}},
   ```
   while the successive difference ratio $q_m = \bigl(\beta_q(m) - \beta_q(m/2)\bigr)/\bigl(\beta_q(m/2) - \beta_q(m/4)\bigr)$ and observed order $p = \log_2(q_m)$ (together with the finite-$m$ shift $|\beta_q(m)/\beta_q(m/2) - 1|$) are reported as report-only diagnostics without an empirical validity band. On the fine grid, the two-point extrapolate $\beta_0 = \bigl(4\beta_q(m/2) - \beta_q(m)\bigr)/3$ (`m2_assumed=1`) is used for both the fine lattice check and the fine cross-code check. In mutations **M1** and **M3**, $\beta_{0,\text{mut}} = \bigl(4\beta_{\text{mut}}(m/2) - \beta_{\text{mut}}(m)\bigr)/3$ is checked against $s_{\text{part,mut}} + e_{R,\text{mut}}/\beta_{\text{pred,mut}}$ with $e_{R,\text{mut}} = \bigl(e_R / |\beta_q(m) - \beta_q(m/2)|\bigr)_{\text{baseline}}\,|\beta_{\text{mut}}(m) - \beta_{\text{mut}}(m/2)|$.
2. **Exact $m$-invariance in `LinearizedBeam`**: In `LinearizedBeam`, $m$-invariance of $\beta_q$ and $|I_2|/|I_1|$ is verified to round-off as an algebraic **identity** of the linearized fluid equations.

##### 6. Numerical setup, D18 total-current observable, D19 entrance kernel, and D20 per-line multistream prediction

To compare `ParticleSolver` and `LinearizedBeam` without adjustable parameters (`examples/_space_charge_waves_common.py`, `tests/test_087_particles_vs_linearized_beam.py`, `tests/test_092_particle_current_lines.py`, `tests/test_093_entrance_amplitude.py`, `tests/test_094_dc_depression_weighting.py`, `tests/test_095_multistream_prediction.py`, `examples/024_particles_space_charge_waves.py`):

1. **Shared grid, integer-period time step, DFT mean subtraction, and D18 total-current observable**:
   - **Shared grid and integer-period DFT**: Both solvers run on the identical `GridFIT3D` ($x \in [-a, a]$ with `bc="pec"`, $y \in [-L_y/2, L_y/2]$ with `ny = 1` and `bc="periodic"`, $z \in [0, L_z]$ with `bc="pec"`). The time step is chosen as $\Delta t = 1/(f N_{\text{spp}})$ with $N_{\text{spp}} = \lceil 1/(f\Delta t_{\text{CFL}})\rceil$ ($\Delta t \le \Delta t_{\text{CFL}}$), so the 6-period DFT window $N_{\text{DFT}} = 6 N_{\text{spp}}$ spans an exact integer number of RF cycles. Per-edge mean subtraction over the $N_{\text{DFT}}$ window eliminates DC current leakage; the integer-$N_{\text{spp}}$ Dirichlet kernel leakage is bounded by the round-off **identity** $\varepsilon_{\text{DFT}} = 100\,\epsilon_{\text{mach}}/m$.
   - **Decision D18 — Total-current observable over all non-PEC transverse lines (`_particle_current_lines`, `run_particle_beam_drift(..., meas_lines="all")`, `tests/test_092_particle_current_lines.py`)**: `LinearizedBeam.axial_current()` sums `J1` over its `keep_lines` mask where the top-hat transverse weight `_w_perp_eff` is strictly positive. For particles with B-spline shape order $p \ge 2$, the transverse nodal deposit $W_p$ places part of the beam current onto non-PEC transverse $E_z$ lines outside `keep_lines` ($5/512$ of the total current on the $10\times 474$ grid with $16$ `"sheet_lattice"` lines, derived from the quadratic-spline weights), whereas the analytical/discrete model counts the total beam current. Summing `ps.J_particles` only over `keep_lines` therefore compared different observables and biased the fitted particle $\beta_q$ low. Under Decision D18 (`meas_lines="all"`, default), `_particle_current_lines(beam_ref, solver, mode="all")` selects **all non-PEC transverse primal $E_z$ lines** (`ieps != 0`) across `beam_ref.k_indices` using the same flat-index and dual-facet area ($\mathrm{d}A_\perp$) conventions as `LinearizedBeam._initialize`:
     ```math
     I_{\text{part}}(z_k, t) = \sum_{\ell \in \text{all non-PEC}} J_{\text{particles},z}(x_\ell, z_k, t)\,\mathrm{d}A_{\perp,\ell},
     ```
     measuring the physical total beam current $I(z)$ independently of the `LinearizedBeam` top-hat mask for any shape order $p$ and beam width (`meas_lines="lb_keep"` retains the legacy top-hat-restricted observable).
2. **Transverse sheet injection profiles (`"sheet_lattice"` vs `"sheet"`)**: Every time step, `BeamInjector.inject` (`wakis/particles/boundaries.py`) places $N_{\text{macro}}$ macroparticles on $z = 0$ with $y_p = 0$ exactly, $|x_p| \le b$, longitudinal momentum $u_z = \gamma_0 u_0$, $\mathbf{u}_\perp = \mathbf{0}$, and time-modulated weight $w(t^n) = |I(t^n)|\Delta t / (|q| N_{\text{macro}})$:
   - **Deterministic reference profile (`profile="sheet_lattice"`)**: Places one macroparticle at the midpoint of each equal-width transverse stratum, $x_\ell = b\bigl(2(\ell + \tfrac{1}{2})/N_{\text{macro}} - 1\bigr)$ ($\ell = 0, \dots, N_{\text{macro}} - 1$). This profile has zero transverse shot noise, is identical for all random seeds, and is used for every signed physics check, all three mutations (**M1**, **M2**, **M3**), the two-resolution check, and the $m$-scan.
   - **Seeded stratified profile (`profile="sheet"`)**: Draws one uniform sample per stratum per step, $x_\ell \in [-b + 2b\ell/N_{\text{macro}}, -b + 2b(\ell+1)/N_{\text{macro}})$, from `BeamInjector._step_rng(step_idx)`. Because `seed` only affects `profile="sheet"`, the 8-seed ensemble (`measure_empirical_noise`) calls `run_particle_beam_drift(..., profile="sheet", seed=s)` at $N_{\text{macro}} = 256$ (coarse) and $512$ (fine), the same per-cell density across both grids, and compares against the `"sheet_lattice"` reference at the same $N_{\text{macro}}$.
   - `test_injected_sheet_geometry` directly verifies these geometric invariants as an **identity (geometry) check** after 3 steps on CPU.
3. **Guide magnetic field**: A strong uniform axial guide field $B_{z,\text{ext}}$ ($\omega_c \gg \omega_p$) freezes out transverse space-charge expansion while leaving longitudinal motion along $\hat{\mathbf{z}}$ unperturbed [Birdsall & Langdon, 1991].
4. **Exact discrete transverse operator, D20 per-line multistream prediction, and D19 entrance-edge kernel (`discrete_modal_prediction`, `multistream_waves`, `entrance_edge_transfer`)**: Rather than approximating transverse finite-difference and B-spline shape effects at leading order in $(T_1\Delta x)^2$, `discrete_modal_prediction(code, nx, nz, params, ..., shape=PARTICLE_SHAPE, method="multistream")` solves the exact discrete transverse operator, profile-consistent DC depression, and for particles (`method="multistream"`, Decision D20) the exact $k$-dependent per-line multistream boundary-value problem for general B-spline shape order $p \in \{1, 2, 3\}$ (`PARTICLE_SHAPE = 2` in V7) in seven steps:
   1. **Axial and temporal symbols, and the drifting-beam axial shape factor $S_z(k)^2$**:
      - Staggered Yee field operator along $z$: $\kappa(k) = \frac{2}{\Delta z}\sin(k\Delta z/2)$, and leapfrog field time symbol $\Omega_f = \frac{2}{\Delta t}\sin(\omega\Delta t/2)$.
      - **Drifting-beam axial shape factor $S_z(k)^2 = \operatorname{sinc}^{2p}(k\Delta z/2)$ (particles, shape order $p$)**: Along $z$, the order-$p$ charge-conserving Esirkepov segment deposit onto primary $z$-edges deposits the $(p - 1)$-th order spline $W_{p-1}$ integrated over each sub-step segment $[z^n, z^{n+1}]$, giving the one-way axial deposit symbol $S_{\text{dep}}(k) = \operatorname{sinc}^p(k\Delta z/2)$ (`wakis/particles/solver.py::cold_plasma_frequency_grid`). Conversely, the Galerkin primary-edge electric-field gather uses staggered order $p_g = p - 1$ along the edge axis $z$ (`wakis/particles/shapes.py`), interpolating $E_z$ with $W_{p-1}(z - z_{k+1/2})$ and giving the one-way axial gather symbol $S_{\text{gath}}(k) = \operatorname{sinc}^{p_g+1}(k\Delta z/2) = \operatorname{sinc}^p(k\Delta z/2)$. For a stationary cold plasma where every Brillouin alias $k_m = k + 2\pi m/\Delta z$ ($m \in \mathbb{Z}$) shares the same temporal denominator $\Omega_p^2$, summing over all aliases enters $\kappa_z^2$ (`wakis/particles/solver.py::cold_plasma_frequency_grid`). For a **drifting** beam, however, the non-principal electrostatic aliases $m \ne 0$ are Doppler-detuned in the beam frame by $|\omega - k_m u_{\text{dep}}| \simeq 2\pi|m|u_{\text{dep}}/\Delta z \gg \omega_q$, so their off-resonant contribution to the electrostatic space-charge response $(\omega - k_m u_{\text{dep}})^{-2}$ is suppressed by the factor
        ```math
        \text{alias\_supp} = \left(\frac{\omega_q\Delta z}{2\pi u_{\text{dep}}}\right)^2 \ll 1.
        ```
        Note that `alias_supp` bounds only the off-resonant electrostatic space-charge aliases; it does not cover the resonant crossing of the $m = -1$ beam alias with the backward electromagnetic light line (see Section 7, *Aliased numerical Cherenkov instability of drifting beams*). In the principal ($m = 0$) Brillouin zone, the round-trip deposit-and-gather coupling for Galerkin gather ($p_g = p - 1$) therefore acquires the axial shape factor
        ```math
        S_z(k)^2 = S_{\text{dep}}(k)\,S_{\text{gath}}(k) = \operatorname{sinc}^p\!\left(\frac{k\Delta z}{2}\right)\operatorname{sinc}^{p_g+1}\!\left(\frac{k\Delta z}{2}\right) = \operatorname{sinc}^{2p}\!\left(\frac{k\Delta z}{2}\right), \qquad \operatorname{sinc}(\xi) \equiv \frac{\sin\xi}{\xi}.
        ```
      - `LinearizedBeam`: centered finite-difference derivative $k_s(k) = \sin(k\Delta z)/\Delta z$ in the continuity and advection operators. Because `J1` and `E_s` are co-located on the same primary $z$-edges, `LinearizedBeam` has no sub-cell axial shape factor ($S_z \equiv 1$).
      - Beam time stepping: Crank–Nicolson symbol $\Omega = \frac{2}{\Delta t}\tan(\omega\Delta t/2)$ for `LinearizedBeam`, and leapfrog symbol $\Omega_p = \frac{2}{\Delta t}\sin(\omega\Delta t/2)$ for particles.
   2. **Exact discrete TM transverse field operator (no quasi-static approximation)**: For $E_z$-only motion under infinite axial guide field ($J_x = J_y = 0$, $\partial_y = 0$), combining the discrete curl-curl equation for $E_z$ with exact discrete charge continuity $-\mathrm{i}\Omega_f\rho + \mathrm{i}\kappa(k)J_z = 0$ eliminates $E_x$ and $B_y$ identically to give
      ```math
      \bigl(\mathbf{L}_x - \tau_{\text{Y}}(k)^2\,\mathbf{I}\bigr) E_z = -\frac{\mathrm{i}\,\tau_{\text{Y}}(k)^2}{\Omega_f\varepsilon_0}\,J_z, \qquad \mathbf{A}(k) = -\mathbf{L}_x + \tau_{\text{Y}}(k)^2\,\mathbf{I},
      ```
      where on the $N_x - 1$ interior $x$-nodes with PEC boundary conditions at $x = \pm a$,
      ```math
      \mathbf{L}_x = \frac{\operatorname{tridiag}(1, -2, 1)}{\Delta x^2}, \qquad \tau_{\text{Y}}(k)^2 = \kappa(k)^2 - \left(\frac{\Omega_f}{c}\right)^2.
      ```
      Because this relation is the exact discrete $\mathrm{TM}$ reduction of the staggered Yee system, there is no quasi-static approximation to bound.
   3. **Beam lines, profile-consistent DC depression, and transverse coupling**:
      - `LinearizedBeam`: the beam lines are the active interior $x$-nodes with transverse top-hat weights $\mathbf{w}_{\perp,\text{eff}}$ read directly from a constructed `LinearizedBeam` instance (pointwise $E_z$ gather onto $v_1$ and line-weighted $J_z$ deposit). Because `LinearizedBeam` has no self-consistent DC Poisson solve, it always uses the uniform depressed velocity `params["u_dep"]` (including in **D3**).
      - Particles: for $E_z$ and $J_z$ located on $z$-edges at transverse primal $x$-nodes, both the order-$p$ Esirkepov $J_z$ (and $\rho$) deposit and the Galerkin $E_z$ gather ($p_g = p - 1$) evaluate the order-$p$ nodal B-spline $W_p$ on primal $x$-nodes via `bspline_weights_1d(xi, p, staggered=False)` with $\xi = (x + a)/\Delta x$, assembled by `transverse_weight_matrix(x_lines, nx, p)`. Weights landing on PEC wall nodes $x = \pm a$ or outside are dropped ($E_z(\pm a) = 0$, matching `_wrap_or_mask_1d` for non-periodic axes; for $|x| \le b$, no $p \le 3$ stencil reaches a wall node). The beam lines are the injected particle positions $x_\ell$ (`"sheet_lattice"` midpoints at the active $N_{\text{macro}}$, or any non-uniform `line_x` array such as the Fibonacci disk positions in **M3**), with coupling matrix $S_{i,\ell} = W_p\bigl((x_\ell - x_i)/\Delta x\bigr)\,(2b / N_{\text{lines}})/\Delta x$ on the interior nodes $i = 1, \dots, N_x - 1$. Before assembling the AC operator, `discrete_modal_prediction` and `multistream_waves` call `compute_discrete_dc_depression(nx, params, line_x=line_x, line_w=line_w, shape=shape)` on the **actual line set** $(\mathbf{x}_\ell, \mathbf{w}_\ell)$ to obtain the profile-consistent per-line DC velocity $u_\ell$, Lorentz factor $\gamma_\ell$, per-line depressed plasma frequency $\omega_{p,\ell}^2 = \omega_{p,0}^2(u_0 / u_\ell)$, and profile-consistent mean depressed velocity $\bar{u}$ and depression $\bar{\delta}$, never reusing the uniform-beam `params["u_dep"]` for a non-uniform profile.
   4. **Transverse modal operator (`compute_transverse_modal_operator`) and `LinearizedBeam` modal branch (`method="modal"`)**: With line charge fractions $\mathbf{D} = \operatorname{diag}(d_\ell)$ ($\sum_\ell d_\ell = 1$), the discrete transverse modal operator
      ```math
      \mathbf{G}(k) = C\,\tau_{\text{Y}}(k)^2\,\mathbf{S}^{\mathsf{T}}\,\mathbf{A}(k)^{-1}\,\mathbf{S}\,\mathbf{D}, \qquad C = \frac{N^2\Delta x}{2b}, \qquad \widetilde{\mathbf{G}}(k)\,\mathbf{y}_j = \lambda_j(k)\,\mathbf{y}_j,
      ```
      is normalized so that for a uniform slab as $\Delta x \to 0$, $\lambda_1(k) \to R\bigl(\tau_{\text{Y}}(k)\bigr)^2$ (`test_discrete_modal_operator_unit`, `DIAG id=modal_unit`). For `LinearizedBeam`, the modal plasma frequencies are $\omega_{q,j}(k)^2 = \lambda_j(k)\,\omega_{p,\text{dep}}^2 / \gamma_{\text{dep}}^3$, the fast ($k_{-,j}$) and slow ($k_{+,j}$) wavenumbers solve $\Omega \mp \omega_{q,j}(k_{\pm,j}) = u_{\text{dep}}\,k_s(k_{\pm,j})$ with $\beta_{q,j} = (k_{+,j} - k_{-,j})/2$ and $k_{c,j} = (k_{+,j} + k_{-,j})/2$, and expanding the entrance pattern $\mathbf{e}_{\text{in}} \propto \mathbf{w}_{\perp,\text{eff}}$ in the biorthonormal eigenvectors $(\mathbf{l}_j, \mathbf{y}_j)$ at $\beta_e$ gives $c_j = (\mathbf{d}^{\mathsf{T}}\mathbf{y}_j)(\mathbf{l}_j^{\mathsf{T}}\mathbf{e}_{\text{in}})$ and the `LinearizedBeam` modal synthesis
      ```math
      I_{1,\text{LB,model}}(z_k) = m I_0 \sum_j c_j \cos(\beta_{q,j} z_k)\,\mathrm{e}^{-\mathrm{i} k_{c,j} z_k}
      ```
      (for particles, this frozen-eigenvector cos-form synthesis is retained only as `method="modal"` for regression and negative-control checks).
   5. **Decision D20 — Exact per-line cold multistream prediction for particles (`multistream_waves`, `discrete_modal_prediction("particles", ..., method="multistream")`, `tests/test_095_multistream_prediction.py`)**:
      - **Coupled line equations and nonlinear eigenproblem $\mathbf{T}(k)\mathbf{X} = \mathbf{0}$**: Each macroparticle stream $\ell \in \{0, \dots, N - 1\}$ at transverse coordinate $x_\ell$ has injected current fraction $d_\ell$, DC drift velocity $u_\ell$, Lorentz factor $\gamma_\ell$, discrete electronic wavenumber $\beta_\ell = \Omega_p / u_\ell$ ($\Omega_p = (2/\Delta t)\sin(\omega\Delta t/2)$), and plasma susceptibility
        ```math
        \sigma_\ell = \frac{\omega_{p,0}^2\,(u_0 / u_\ell)}{\gamma_\ell^3 u_\ell^2}.
        ```
        From linearized cold-stream continuity and $i_{1,\ell} = \rho_{0,\ell} v_{1,\ell} + u_\ell\rho_{1,\ell}$, the convective operator $D_\ell = \mathrm{i}\Omega_p + u_\ell\partial_z$ satisfies the exact identity $D_\ell i_{1,\ell} = \mathrm{i}\Omega_p\rho_{0,\ell} v_{1,\ell}$ (which holds even when $u_\ell(z)$ and $\rho_{0,\ell}(z)$ vary with $z$). Combining with the relativistic longitudinal momentum equation in the uniform bulk yields
        ```math
        \left(\frac{\mathrm{d}}{\mathrm{d}z} + \mathrm{i}\beta_\ell\right)^2 i_\ell(z) = -\sigma_\ell\,\bigl(S_z^2 \circledast \mathbf{G}[i]\bigr)_\ell(z),
        ```
        where $i_\ell(z)$ is the line current phasor in units of $d_\ell I_{1,0}$. Spatial harmonics $\mathbf{i}(z) = \mathbf{X}\,\mathrm{e}^{-\mathrm{i} k z}$ therefore satisfy the $N\times N$ **nonlinear eigenvalue problem**
        ```math
        \mathbf{T}(k)\,\mathbf{X}_n = \mathbf{0}, \qquad \mathbf{T}(k) = \operatorname{diag}\!\bigl((k - \beta_\ell)^2\bigr) - \mathbf{M}(k), \qquad \mathbf{M}(k) = S_z(k)^2\,\operatorname{diag}(\sigma_\ell)\,\mathbf{G}(k),
        ```
        which has $2N$ multistream roots $(k_n, \mathbf{X}_n)$ ($n = 1, \dots, 2N$; solved in the mirror-even subspace $\mathbf{P}$ of dimension $N/2$ with $P_{\ell,j} = (\delta_{\ell,j} + \delta_{\ell,N-1-j})/\sqrt{2}$ when $x_\ell$, $d_\ell$, $\beta_\ell$, and the entrance profile are mirror-symmetric, or in the full $N$-line space for asymmetric profiles such as **M3**). Seeds from the companion quadratic eigenproblem at $k_{\text{ref}} = \omega / \bar{u}$ are polished root-by-root via SVD Newton updates $\Delta k = -\mathbf{y}^{\mathsf{H}}\mathbf{T}(k)\mathbf{x} / (\mathbf{y}^{\mathsf{H}}\mathbf{T}'(k)\mathbf{x})$ (with central-difference $\mathbf{T}'(k)$ at $h = 10^{-3}\;\text{rad/m}$ and successive-substitution fallback) to $\sigma_{\min}/\sigma_{\max} < 10^{-12}$ and pairwise separation $|\Delta k| > 10^{-8}\;\text{rad/m}$.
        The legacy cos-form synthesis (`method="modal"`) is exact only when $u_\ell$ is uniform and $\mathbf{G}$ is frozen at $k_{\text{ref}}$ (`freeze_g=True`, reduction check R1: $|\Delta I_1|/|I_1| \le 1.3\times 10^{-12}$). With $k$-dependent $\mathbf{G}(k)$ even without shear (reduction check R2), $\tau_{\text{Y}}(k)^2$ differs by $\pm\beta_q$ between the fast and slow waves, so the fundamental mode's transverse eigenvector $\mathbf{X}$ differs between $k_f$ and $k_s$; enforcing $v_{1,\ell}(0) = 0$ per line then excites mode 3 asymmetrically ($|O_{3s}/O_{3f}| = 3.6$, $3.1$, and $8.8$ on coarse $q/m=1$, fine $q/m=1$, and coarse $q/m=2$), shifting the fitted $\beta_q$ by $-4.9\times 10^{-5}$, $-4.7\times 10^{-5}$, and $+4.0\times 10^{-5}$. With transverse DC velocity shear, the null-space ("ballistic") band near $\bar{\beta}$ splits into complex conjugate pairs with $|\operatorname{Im} k| \le 0.184\;\text{rad/m}$ ($q/m = 1$) and $\le 0.426\;\text{rad/m}$ ($q/m = 2$)—a weak convective multistream instability with amplitude growth $\le 1.08$ over the tunnel and observable weight $|o_n| \le 10^{-3}$.
      - **DC entrance transit phase $\theta_\ell$ (`_multistream_dc_entrance_phase`) and exact per-line matching**: Near the grounded PEC cathode wall $z = 0$ (Dirichlet $\phi_{\text{DC}}(x, 0) = 0$), each transverse eigenmode $(\lambda_m, \mathbf{e}_m)$ of $-\mathbf{L}_x$ relaxes toward its bulk depressed nodal potential $\boldsymbol{\phi}_\infty$ as $1 - r_m^{z/\Delta z} = 1 - \mathrm{e}^{-q_m z}$ over $\sim 1/\sqrt{\lambda_1} \approx 1.3\;\text{mm}$, with $r_m + r_m^{-1} = 2 + \lambda_m\Delta z^2$ and $q_m = -\ln(r_m)/\Delta z$. Substituting $i_\ell(z) = \psi_\ell(z)\exp\bigl(-\mathrm{i}\int_0^z (\Omega_p / u_\ell(z'))\,\mathrm{d}z'\bigr)$ into $D_\ell i_\ell = \mathrm{i}\Omega_p\rho_{0,\ell} v_{1,\ell}$ makes the exact entrance condition $v_{1,\ell}(0) = 0 \iff \psi_\ell'(0) = 0$. To leading order in the ratio of the DC entrance length to $\lambda_q$, matching onto the asymptotic bulk streams shifts each line's entrance phase by $\mathrm{e}^{-\mathrm{i}\theta_\ell}$ with
        ```math
        \theta_\ell = \int_0^\infty \left(\frac{\Omega_p}{u_\ell(z)} - \frac{\Omega_p}{u_\ell}\right)\mathrm{d}z, \qquad \gamma_\ell(z) = \gamma_0 - \frac{e\,|\phi_\ell(z)|}{m_{\text{eff}} c^2}, \qquad u_\ell(z) = c\sqrt{1 - \gamma_\ell(z)^{-2}},
        ```
        where $\theta_\ell \approx -8\times 10^{-3}$ to $-10^{-2}\;\text{rad}$ (cross-line spread $1.9\times 10^{-3}\;\text{rad}$ for $q/m = 1$ and $4.1\times 10^{-3}\;\text{rad}$ for $q/m = 2$, shifting $\beta_q$ by $+2.0\times 10^{-6}$, $+2.7\times 10^{-6}$, and $+2\times 10^{-7}$). Because `BeamInjector` places the sinusoidal current modulation in the macro-weights and injects every line at $\gamma_0$ ($v_{1,\ell}(0) = 0$), the $2N$ initial wave amplitudes $\mathbf{a}^{(0)}$ satisfy the exact $2N \times 2N$ per-line entrance matching system ($e_\ell = 1$ for uniform modulation):
        ```math
        \sum_{n=1}^{2N} a_n^{(0)}\,X_{\ell,n} = e_\ell\,\mathrm{e}^{-\mathrm{i}\theta_\ell}, \qquad \sum_{n=1}^{2N} a_n^{(0)}\,(k_n - \beta_\ell)\,X_{\ell,n} = 0 \qquad (\ell = 0, \dots, N - 1).
        ```
      - **Neumann PEC-wall near-field correction (`_multistream_wall_correction`) and image-sign derivation ($E_x(x, 0) = 0 \Rightarrow J_z$ even)**: The bulk symbol $\mathbf{G}(k)$ assumes each wave $\mathrm{e}^{-\mathrm{i} k z}$ continues without boundary into $z < 0$. At the $z = 0$ PEC wall, the tangential electric field vanishes: $E_x(x, 0) = 0$. By Gauss's law $\partial_x E_x + \partial_z E_z = \rho/\varepsilon_0$, $E_x(x, 0) = 0$ requires $\rho(x, z)$ to be odd across $z = 0$, and by charge continuity $\partial_t\rho + \partial_z J_z = 0$ the axial current density $J_z(x, z)$ and axial electric field $E_z(x, z)$ (or axial Hertz potential $\Pi_z$, $\partial_z\Pi_z|_{z=0} = 0$) have an **even (Neumann) image continuation** across $z = 0$ ($g_m'(0) = 0$, `wall_image="neumann"`). Splitting $\mathbf{G}(k)$ in the eigenbasis $(\lambda_m, \mathbf{e}_m)$ of $-\mathbf{L}_x$ into its local contact term and its evanescent sum,
        ```math
        \mathbf{G}(k) = C\,\mathbf{S}^{\mathsf{T}}\mathbf{S}\mathbf{D} - C\sum_{m=1}^{N_x - 1} \frac{\lambda_m}{\lambda_m + \tau_{\text{Y}}(k)^2}\,\mathbf{S}^{\mathsf{T}}\mathbf{e}_m\mathbf{e}_m^{\mathsf{T}}\mathbf{S}\mathbf{D},
        ```
        shows that the local term $C\,\mathbf{S}^{\mathsf{T}}\mathbf{S}\mathbf{D}$ produces no wall correction, while each evanescent transverse mode $m$ with $s_m^2 = \lambda_m - (\Omega_f / c)^2 > 0$ has the discrete 1D Green's function $A_m\rho_m^{|j|}$ with $\rho_m + \rho_m^{-1} = 2 + s_m^2\Delta z^2$, $q_m = -\ln(\rho_m)/\Delta z$, and $H_m = \lambda_m\rho_m\Delta z^2 / (1 - \rho_m^2)$. Subtracting the unbounded $z < 0$ harmonic continuation from the even (Neumann) image reflection $j \mapsto -j - 1$ across $z = 0$ yields the discrete evanescent wall kernel at edge midpoint $z_j = (j + 1/2)\Delta z$:
        ```math
        \Delta F_{m,n}(z_j) = H_m\,\rho_m^{j+1}\left[\frac{\mathrm{e}^{-\mathrm{i} k_n\Delta z/2}}{1 - \rho_m\mathrm{e}^{-\mathrm{i} k_n\Delta z}} - \frac{\mathrm{e}^{+\mathrm{i} k_n\Delta z/2}}{1 - \rho_m\mathrm{e}^{+\mathrm{i} k_n\Delta z}}\right] \equiv \operatorname{amp}_{n,m}\,\mathrm{e}^{-q_m z_j}
        ```
        (continuum limit $\Delta_m(z) = -\mathrm{i}\lambda_m k_n\,\mathrm{e}^{-s_m z}/\bigl(s_m(s_m^2 + k_n^2)\bigr)$). Including the axial deposit factor $\operatorname{sinc}^p(k_n\Delta z/2)$ and the evanescent gather factor $g_{q,m} = \bigl(\sinh(q_m\Delta z/2)/(q_m\Delta z/2)\bigr)^p$, the first-order (Born) wave-amplitude shift propagated with the causal beam-wave resolvent $\mathbf{R} = \mathbf{V}^{-1}[\mathbf{0};\,\mathbf{I}]$ ($\mathbf{V} = [\mathbf{X};\;-\mathrm{i}\mathbf{X}\operatorname{diag}(k_n)]$) is
        ```math
        \mathbf{f}_m = -C\,g_{q,m}\,(\mathbf{S}^{\mathsf{T}}\mathbf{e}_m)\sum_{n=1}^{2N} a_n^{(0)}\,(\mathbf{e}_m^{\mathsf{T}}\mathbf{S}\mathbf{D}\mathbf{X}_n)\,\operatorname{amp}_{n,m}\operatorname{sinc}^p\!\left(\frac{k_n\Delta z}{2}\right), \qquad \Delta a_n = \sum_{m=1}^{N_x - 1} \frac{\bigl[\mathbf{R}\bigl(-\boldsymbol{\sigma}\odot\mathbf{f}_m\bigr)\bigr]_n}{q_m - \mathrm{i}k_n},
        ```
        giving $a_n = a_n^{(0)} + \Delta a_n$. This PEC-wall near-field term shifts $\beta_q$ by $+6.4\times 10^{-5}$ (coarse $q/m = 1$), $+8.4\times 10^{-5}$ (fine $q/m = 1$), and $-3.1\times 10^{-5}$ (coarse $q/m = 2$), and agrees with the exact half-line solution of the augmented ODE $[\mathbf{i}, \mathbf{i}', \mathbf{g}, \mathbf{g}']$ with $g_m'(0) = 0$ to $0.13\%$–$2.3\%$ of the wall effect per wave ($\le 5\times 10^{-7}$ on the fitted $\beta_q$; `tests/test_095_multistream_prediction.py`).
      - **D18 per-wave deposited-current observable and envelope synthesis**: Each multistream wave $(k_n, \mathbf{X}_n)$ contributes to the D18 total-current observable at primary edge midpoints $z_k = (k + 1/2)\Delta z$ with its own one-way order-$p$ Esirkepov axial deposit symbol $\operatorname{sinc}^p(k_n\Delta z/2)$ and total line-weight projection $\mathbf{d}^{\mathsf{T}}\mathbf{X}_n$ (while the common time factor $\operatorname{sinc}(\omega\Delta t/2)$ drops out of the fit):
        ```math
        I_{1,\text{model}}(z_k) = m I_0 \sum_{n=1}^{2N} o_n\,\mathrm{e}^{-\mathrm{i} k_n z_k}, \qquad o_n = a_n\,\operatorname{sinc}^p\!\left(\frac{k_n\Delta z}{2}\right)\bigl(\mathbf{d}^{\mathsf{T}}\mathbf{X}_n\bigr).
        ```
        Fitting $I_{1,\text{model}}(z_k)$ with `fit_beta_q_from_envelope` over $[z_{\min}, z_{\max}]$ yields the parameter-free prediction $\beta_{\text{P,pred}}$ (for $s_{\text{modes}}$, each wave $n$ is assigned to the transverse mode $j$ at $k_{\text{ref}}$ with maximal $|\mathbf{l}_j^{\mathsf{T}}\mathbf{X}_n|$ and the waves of the highest excited mode are zeroed).
   6. **Decision D19 — PEC-wall entrance-edge transfer kernel (`entrance_edge_transfer`, `tests/test_093_entrance_amplitude.py`) and its analytic bound**: On the first primary edge $z_0 = \Delta z / 2$ ($k = 0$), `LinearizedBeam` evaluates $\cos(\beta_q\Delta z/2)$ with bound $\frac{1}{2}(\beta_q\Delta z)^2$. For particles with shape order $p \ge 2$, the order-$p$ Esirkepov edge-$k$ kernel at the PEC wall $z = 0$ is cut (no particles at $z < 0$) and the 1D normal-axis birth wall-closure current delivers the interior tail charge fraction $C_k(0) = \sum_{m \ge k+1} W_p(-m)$ ($C_0(0) = 0, 1/8, 1/6$ for $p = 1, 2, 3$) as an impulse at birth ($j = 0$). Consequently, the first-edge current is less smeared than the bulk symbol $\operatorname{sinc}^p(\beta_e\Delta z/2)\cos(\beta_q\Delta z/2)$ (which missed the $p = 2$ measurement by $+1.48\times 10^{-3}$ on $10\times 474$ and $+3.76\times 10^{-4}$ on $20\times 948$, $6$–$10\times$ outside a Taylor-only bound). `entrance_edge_transfer(p, k, dz, dt, omega, u, wall=True)` computes the exact discrete-time unperturbed transfer modulus $H_k$ of a particle born at $z_0 = 0$ and drifting at $u = u_{\text{dep}}$ along $z_j = u\,\Delta t\,j$:
      ```math
      C_k(\zeta) = \sum_{m \ge k + 1} W_p(\zeta - m), \qquad \Delta C_j = C_k\!\left(\frac{z_{j+1}}{\Delta z}\right) - C_k\!\left(\frac{z_j}{\Delta z}\right) + \delta_{j,0}\,C_k(0), \qquad H_k = \left|\sum_{j=0}^{n-1} \Delta C_j\,\mathrm{e}^{-\mathrm{i}\omega\Delta t\,j}\right|,
      ```
      giving the first-edge ($k = 0$) entrance amplitude prediction and analytic bound
      ```math
      \frac{|I_{1,\text{part}}(z_0)|}{m I_0} = H_0(p, \Delta z, \Delta t, \omega, u_{\text{dep}})\cos\!\left(\frac{\beta_q\Delta z}{2}\right), \qquad \text{tol}_{\text{amp,part}} = 2\left(\frac{v_{\text{inj}}}{u_{\text{dep}}} - 1\right)\frac{p\,(\beta_e\Delta z)^2}{24} + \left(1 - \cos\!\left(\frac{\beta_q\Delta z}{2}\right)\right) + 10^{-12},
      ```
      where the first term bounds the unmodelled entrance velocity difference between $v_{\text{inj}} = u_0$ (beam not yet slowed at the wall) and $u_{\text{dep}}$ (the factor $2$ follows from $\mathrm{d}\operatorname{sinc}^p/\mathrm{d}\ln\beta_e$), and the second term bounds the near-wall plasma-phase factor (in the small-$\Delta z$ limit at CFL $\sigma = u\,\Delta t/\Delta z$, $(H_{\text{wall}} - H_{\text{bulk}})/(\beta_e\Delta z)^2 \to \frac{61}{4608} + \frac{25}{768}\sigma + \mathcal{O}(\sigma^2)$ for $p = 2$). Because the D19 wall kernel only alters edges at $z < (p + 1)\Delta z$ while the variable-projection fit window starts at $z_{\min} = 15.0\;\text{mm}$ ($\ge 34$ cells), D19 does not enter the fitted $\beta_q$.
   7. **D20 attribution, residuals within budget, and factorial residual separation (`DIAG id=shear`, `DIAG id=particle_factorial`)**:
      - Across the cumulative D20 steps (with no fitted parameters), the relative residuals against the $m \to 0$ extrapolated targets ($\beta_0 = 53.332757\;\text{rad/m}$ on coarse $10\times 474$, $54.362241\;\text{rad/m}$ on fine $20\times 948$, and $77.11659\;\text{rad/m}$ on **M1** $q/m = 2$) evolve as:

        | step | $q/m = 1$ coarse ($10\times 474$) | $q/m = 1$ fine ($20\times 948$) | $q/m = 2$ coarse (**M1**) |
        |---|---|---|---|
        | legacy cos-form (`method="modal"`) | `+1.12e-4` | `+8.1e-6` | `-5.63e-4` |
        | exact roots, first-order amplitudes | `+3.28e-4` | `+2.32e-4` | `-1.18e-4` |
        | + exact per-line matching ($k$-dependent $\mathbf{G}$) | `-6.88e-5` | `-8.84e-5` | `+1.78e-5` |
        | + per-wave observable $\operatorname{sinc}^p$ (D18) | `-6.91e-5` | `-8.85e-5` | `+1.92e-5` |
        | + Neumann PEC-wall near field | `-5.3e-6` | `-4.3e-6` | `-1.19e-5` |
        | + DC entrance phase $\theta_\ell$ = **D20** | **`-3.3e-6`** | **`-1.7e-6`** | **`-1.16e-5`** |

        At $q/m = 1$, the legacy cos-form model passed only by an accidental cancellation of three errors—first-order roots ($\approx -2.2\times 10^{-4}$), cos-form amplitudes ($\approx +3\times 10^{-4}$), and the missing PEC-wall near field ($-6\times 10^{-5}$ to $-8\times 10^{-5}$)—which failed at $q/m = 2$ (**M1**, $-5.63\times 10^{-4}$ vs bound $1.82\times 10^{-4}$ / $1.90\times 10^{-4}$) and on the blind out-of-sample Fibonacci disk profile (**M3**, where D20 predicts $55.191183\;\text{rad/m}$ vs measured $\beta_0 = 55.191143\;\text{rad/m}$, relative residual $\approx -1\times 10^{-6}$, whereas the cos-form model predicts $55.196609\;\text{rad/m}$ and misses by $+9.9\times 10^{-5}$ against a $5.9\times 10^{-5}$ bound). In `test_axial_vs_transverse_refinement` (`DIAG id=particle_factorial`), D20 also reproduces the **D6** transverse line-count shift ($N_{\text{macro}} = 16 \to 64$ on $10\times 474$, where $\beta_0 = 53.319720\;\text{rad/m}$ at $n=64$ and $54.358707\;\text{rad/m}$ at $n=128$ on $20\times 948$, both at $-3.3\times 10^{-6}$ and $-1.7\times 10^{-6}$): measured $-1.3038\times 10^{-2}\;\text{rad/m}$, D20 $-1.3037\times 10^{-2}\;\text{rad/m}$, cos-form $-1.2803\times 10^{-2}\;\text{rad/m}$.
      - **Residuals are within budget, not $10^{-6}$ accuracy; coarse-test variant separation**: Because the $m \to 0$ residuals of $-3.3\times 10^{-6}$ (coarse) and $-1.7\times 10^{-6}$ (fine) lie below D20's own unmodelled terms—discrete-particle alias/Doppler terms ($\sim 10^{-5}$ to $3\times 10^{-5}$), wall-term discretization ($\le 9.3\times 10^{-6}$), complex ballistic roots evaluated at $\operatorname{Re} k$ ($\le 4.8\times 10^{-6}$), and the AC part of the D17 birth kick ($-2.0\times 10^{-6}$)—they are reported as **within budget**, not as $10^{-6}$-level accuracy. Furthermore, on the coarse baseline grid ($10\times 474$, $q/m = 1$), $s_{\text{part}} = 4.76\times 10^{-5}$ ($s_{\text{win}} = 4.61\times 10^{-5}$; $5.14\times 10^{-5}$ for the cos-form model), so the coarse `bound0` of $2.44\times 10^{-4}$ ($2.48\times 10^{-4}$ in the design table) is $\sim 80\%$ $e_R + \sigma_{\text{prop}}$. Consequently, **the coarse baseline test alone cannot separate D20 from the cos-form ($+1.12\times 10^{-4}$), Dirichlet-image (`wall_image="dirichlet"`, $-1.33\times 10^{-4}$), or no-wall (`wall_image="none"`, $-6.7\times 10^{-5}$) variants** (and **M1** with bound $1.90\times 10^{-4}$ also passes Dirichlet at $+5.6\times 10^{-5}$ and no-wall at $+2.0\times 10^{-5}$, while on the fine grid no-wall at $-8.6\times 10^{-5}$ exceeds $s_{\text{part}} = 7.8\times 10^{-5}$ by $10\%$ and Dirichlet misses by $-1.70\times 10^{-4}$). Separating the variants requires combining **M1**, **M3**, the fine grid, and the direct fast checks in `tests/test_095_multistream_prediction.py`.
5. **Parameter-free prediction slack (`prediction_slack`) and signed checks**:
   - The remaining model sensitivities are evaluated explicitly and summed without empirical multipliers or squares:
     - $s_{\text{win}} = \max_{\pm}\bigl|\beta_{\text{pred}}(\text{window shifted by }\pm 1/T_1) - \beta_{\text{pred}}\bigr|/\beta_{\text{pred}}$ (sensitivity to the cathode-zone and outflow-ramp window margins);
     - $s_{\text{modes}} = \bigl|\beta_{\text{pred}}(\text{all modes}) - \beta_{\text{pred}}(\text{all modes except highest})\bigr|/\beta_{\text{pred}}$ (modal truncation);
     - $s_{\text{sym}} = \bigl|\beta_{\text{pred}}(\Omega_f\text{ in }\tau_{\text{Y}}^2) - \beta_{\text{pred}}(\omega\text{ in }\tau_{\text{Y}}^2)\bigr|/\beta_{\text{pred}}$ (time-symbol ambiguity at the field/beam interface);
     - $s_{\text{round}} = 10^{-12}$.
   - For the linear code (`LinearizedBeam` on coarse and fine grids and mutation **M2** with $\rho_0\times 2$), the signed check asserts
     ```math
     \left|\frac{\beta_{q,\text{LB}}}{\beta_{\text{LB,pred}}} - 1\right| \le s_{\text{LB}} = s_{\text{win,LB}} + s_{\text{modes,LB}} + s_{\text{sym,LB}} + s_{\text{round}}.
     ```
   - Every check comparing particle measurements with the linear prediction uses the $m \to 0$ extrapolated wavenumber $\beta_0$:
     - **Coarse lattice**: $\beta_0 = \beta_{0,\text{3pt}}$ from the three-point fit through $(m, m/2, m/4)$, asserting $|\beta_{0,\text{3pt}}/\beta_{\text{P,pred}} - 1| \le s_{\text{part}} + (e_R + \sigma_{\text{prop}})/\beta_{\text{P,pred}}$.
     - **Fine lattice**: $\beta_0 = \bigl(4\beta(m/2) - \beta(m)\bigr)/3$ (`m2_assumed=1`), asserting $|\beta_0/\beta_{\text{P,pred}} - 1| \le s_{\text{part}}$.
     - **Mutations M1 ($|q/m|\times 2$) and M3 (Fibonacci disk `line_x`)**: each mutated configuration is run at $m$ and $m/2$ with $\beta_{0,\text{mut}} = \bigl(4\beta_{\text{mut}}(m/2) - \beta_{\text{mut}}(m)\bigr)/3$, and the bound adds the two-point remainder scaled from the coarse baseline:
       ```math
       \left|\frac{\beta_{0,\text{mut}}}{\beta_{\text{pred,mut}}} - 1\right| \le s_{\text{mut}} + \frac{e_{R,\text{mut}}}{\beta_{\text{pred,mut}}}, \qquad e_{R,\text{mut}} = \left(\frac{e_R}{|\beta(m) - \beta(m/2)|}\right)_{\text{baseline}}\,\bigl|\beta_{\text{mut}}(m) - \beta_{\text{mut}}(m/2)\bigr|.
       ```
     - **Cross-code check (coarse and fine)**: uses the respective extrapolated $\beta_0$:
       ```math
       \left|\frac{\beta_0}{\beta_{q,\text{LB}}} - \frac{\beta_{\text{P,pred}}}{\beta_{\text{LB,pred}}}\right| \le s_{\text{LB}} + s_{\text{part}}.
       ```
     - **8-seed ensemble (`profile="sheet"`)**: `resid_mean` vs $\beta_{\text{P,pred}}$ is report-only; the asserted falsifier is $\bigl|\overline{\beta}_{q,\text{P}} - \beta_{q,\text{P,lattice}}\bigr| \le t_{0.995,7}\,\text{SEM}$ at the same finite $m$ on both sides.
   - To inspect the printed `*CHECK`, `TERMS`, and `DIAG` lines at run time:
     ```bash
     pytest tests/test_087_particles_vs_linearized_beam.py -v -s
     python examples/024_particles_space_charge_waves.py --smoke
     ```

##### 7. Limits & Failure Modes

- **Linear in modulation index $m$ (compare at $m \to 0$)**: Both `multistream_waves` (D20) and `LinearizedBeam` solve the $m \to 0$ linearized equations and omit the quadratic convective terms $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$. At finite modulation ($m = 0.015$), the $\mathcal{O}(m^2)$ ballistic-bunching shift raises the measured particle wavenumber by $\beta_q(m)/\beta_0 - 1 = +4.6\times 10^{-5}$ (coarse) and $+5.1\times 10^{-5}$ (fine), so against finite-$m$ measurements D20 sits $-4.0\times 10^{-5}$ to $-5.5\times 10^{-5}$ below across all cases ($-4.95\times 10^{-5}$ on $10\times 474$, $-5.27\times 10^{-5}$ on $20\times 948$, $-4.03\times 10^{-5}$ on **M1**, $-5.5\times 10^{-5}$ on $10\times 948$, and $-4.5\times 10^{-5}$ on $20\times 474$). Every particle check against the linear prediction must therefore compare at $m \to 0$ via Richardson extrapolation. When $m$ or drift length $z$ is increased until $X(z) \sim \mathcal{O}(1)$, fast electrons overtake slow electrons (wave breaking / trajectory crossing), generating strong higher harmonics ($2\omega, 3\omega$) and electrostatic trapping in `ParticleSolver` that linear models cannot represent.
- **Discrete-particle alias/Doppler terms are not modelled ($\sim 10^{-5}$ to $3\times 10^{-5}$)**: `multistream_waves` retains only the principal ($m = 0$) Brillouin-zone axial shape factor $S_z(k)^2 = \operatorname{sinc}^{2p}(k\Delta z/2)$ and omits Doppler-shifted discrete-particle stream aliases $k_m = k + 2\pi m/\Delta z$ ($m \ne 0$), which are estimated at $\sim 10^{-5}$ ($q/m = 1$) to $\sim 3\times 10^{-5}$ ($q/m = 2$, the leading candidate for the $-1.16\times 10^{-5}$ remainder in **M1**).
- **PEC-wall near-field term is first order ($\le 9.3\times 10^{-6}$)**: The $z = 0$ Neumann wall correction in `_multistream_wall_correction` uses the first-order Born approximation in the evanescent image field (Born truncation $\le 5\times 10^{-7}$ relative to the exact augmented half-line ODE, $0.13\%$–$2.3\%$ of the wall effect per wave) and models the first axial cell through Fourier/modal symbols ($H_m\rho_m^{j+1}$, $\operatorname{sinc}^p$, and $g_{q,m}$) rather than the exact cut near-wall Yee stencil with the D19 kernel, leaving a wall-discretization uncertainty of $\le 9.3\times 10^{-6}$ on the coarse grid ($\le 1.7\times 10^{-6}$ on the fine grid and at $q/m = 2$). In addition, the AC part of the D17 PEC-wall birth kick ($\ell = C_0(0)\Delta z = \Delta z/8$ for $p = 2$, shifting $\beta_q$ by $-2.0\times 10^{-6}$ coarse, $-1.7\times 10^{-6}$ fine, and $-1.1\times 10^{-6}$ at $q/m = 2$) is omitted, and the DC entrance transit phase $\theta_\ell$ uses a thin-region approximation that ignores near-wall density depletion ($\approx 1\%$ of a $2\times 10^{-6}$ effect).
- **Complex roots evaluated at $\operatorname{Re} k$ ($\le 4.8\times 10^{-6}$)**: When DC velocity shear splits the null-space ("ballistic") band near $\bar{\beta}$ into complex conjugate pairs ($|\operatorname{Im} k| \le 0.184\;\text{rad/m}$ for $q/m = 1$ and $\le 0.426\;\text{rad/m}$ for $q/m = 2$), $\mathbf{M}(k)$ and `root_resid` in `multistream_waves` are evaluated at $\operatorname{Re} k$. Dropping the complex ballistic band entirely shifts the fitted $\beta_q$ by $+4\times 10^{-7}$ (coarse $q/m = 1$), $-1.5\times 10^{-7}$ (fine $q/m = 1$), and $-4.8\times 10^{-6}$ (coarse $q/m = 2$), bounding this approximation by $\le 4.8\times 10^{-6}$.
- **Residuals are within budget (not $10^{-6}$ accuracy) and the coarse test alone cannot separate D20 from its variants**: Because the $m \to 0$ residuals ($3.3\times 10^{-6}$ coarse, $1.7\times 10^{-6}$ fine) are smaller than D20's unmodelled alias ($\sim 10^{-5}$ to $3\times 10^{-5}$), wall-discretization ($\le 9.3\times 10^{-6}$), complex-root ($\le 4.8\times 10^{-6}$), and AC birth-kick ($-2\times 10^{-6}$) terms, they must be interpreted as **within budget** rather than claimed as $10^{-6}$-level accuracy. Furthermore, because the coarse baseline test bound of $2.44\times 10^{-4}$ ($s_{\text{part}} = 4.76\times 10^{-5}$) is dominated ($\sim 80\%$) by $e_R + \sigma_{\text{prop}}$, the coarse test alone cannot separate D20 from the cos-form ($+1.12\times 10^{-4}$), Dirichlet-image ($-1.33\times 10^{-4}$), or no-wall ($-6.7\times 10^{-5}$) variants. `LinearizedBeam` also retains the legacy cos-form modal prediction (`code="lb"`).
- **Seeded sheet vs lattice at low macroparticle density**: With `profile='sheet'` at low `n_macro_per_step`, step-to-step random transverse positions raise the fitted $\beta_q$ and the DC depression, and cause a small energy deficit. The effect is not a fit artefact, not static disorder and not a warm-beam shift (see DIAG A1–A5), and it vanishes at higher macroparticle density. Use `profile='sheet_lattice'` for deterministic checks.
- **Planar sheet-beam geometry and strong axial magnetic focusing only**: The analytical reduction factor `reduction_factor_planar` and the `"sheet"` / `"sheet_lattice"` injector profiles assume a 1D planar sheet beam uniform along $y$ (`ny = 1`, `bc="periodic"`) and confined by an axial guide field $B_{z,\text{ext}}$ strong enough ($\omega_c \gg \omega_p$) to freeze out transverse velocity perturbations $\mathbf{v}_\perp$. At weak or zero guide field, transverse space-charge forces cause beam spreading or Brillouin scalloping; 3D pencil or rectangular beams require `reduction_factor_cylindrical` or `reduction_factor_rectangular` [Branch & Mihran, 1955].
- **Two-wave fit launch offset $z_0$ and sine-quadrature amplitude $B$ are report-only diagnostics**: While D20 includes the first-order Neumann PEC-wall near field and DC entrance phase $\theta_\ell$ in the particle prediction $I_{1,\text{model}}(z)$, the two-wave variable-projection fit parameters $|a_s|/|a_f|$, $B/A$, and $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$ remain report-only diagnostics and are not asserted directly.
- **Two-wave fit window requires $z_{\max} - z_{\min} \ge 1.25\,\lambda_q$**: Separating two closely spaced complex exponentials $k_f = k_c - \beta_q$ and $k_s = k_c + \beta_q$ alongside a free carrier $k_c$ requires the interior fit window $[d_w, L_z - L_{\text{ramp}} - d_w]$ ($d_w = \ln(1/\varepsilon_w)/\gamma_1$) to span at least $1.25\,\lambda_q$ (`test_preflight_and_resolution_criteria`). On shorter tunnels where $z_{\max} - z_{\min} \lesssim \lambda_q$, the two wave columns become nearly collinear and the variable-projection estimate of $(k_c, \beta_q)$ becomes ill-conditioned.
- **Transverse injection profile and profile-consistent DC depression must match (mutation M3 and diagnostic D3)**: `LinearizedBeam` with `half_thickness=b` represents a uniform top-hat planar slab across $x \in [-b, b]$ and excites its inflow perturbation proportional to `_w_perp_eff`. If the particle injector instead samples a Fibonacci disk profile (**M3**, `test_mutation_disk_profile_instead_of_sheet_fails`), its center-peaked $x$-marginal both increases the fundamental transverse eigenvalue $\lambda_1$ and deepens the DC potential trough (larger $\delta_{\text{disc,charge,M3}}$ and stronger transverse velocity shear, `DEPCHECK id=M3`), shifting $\beta_{q,\text{P}}$ above the uniform-sheet prediction (failing the baseline check while matching `discrete_modal_prediction` evaluated with the full-space D20 multistream solve and profile-consistent DC depression at the Fibonacci `line_x`). Comparing `LinearizedBeam` on the same non-uniform profile (**D3**) requires depositing the **M3** particle positions onto the $x$-nodes via `transverse_weight_matrix(line_x, nx, shape)` (`_CicNodeRho0`) while keeping uniform `u_dep` because `LinearizedBeam` has no DC space-charge physics.

###### Aliased numerical Cherenkov instability of drifting beams

On a staggered Yee grid of axial cell size $\Delta z$, a beam drifting at velocity $u$ carries spatial Brillouin aliases with dispersion $\omega = (k - 2\pi m/\Delta z)u$ ($m \in \mathbb{Z}$). For $m = -1$, the backward-propagating alias $\omega = (2\pi/\Delta z - k)u$ crosses the backward electromagnetic light line $\omega = c k$ in the principal Brillouin zone at (`studies/p5b-mscan/README.md` Section 3)

```math
k = \frac{2\pi}{\Delta z}\,\frac{u}{c + u}, \qquad f = \frac{c k}{2\pi} = \frac{c}{\Delta z}\,\frac{u}{c + u}.
```

For $u = 5.78\times 10^7\;\text{m/s}$ and $\Delta z = 0.186/428\;\text{m}$, this crossing occurs at $k = 2336\;\text{rad/m}$ and $f = 111.5\;\text{GHz}$, observed directly in the 2D FFT of the unmodulated ($m = 0$) axial current history $I_z(t, z)$ at settle factor $1.5$ alongside its harmonic at $222\;\text{GHz}$, with the rms current fluctuation peaking at $14.8\;\text{mA}$ near the entrance ($z \sim 0.05\;\text{m}$) and falling to $\sim 4\;\text{mA}$ at the exit (`studies/p5b-mscan/README.md` Section 3).

1. **Why the instability is absolute (backward-wave feedback)**: At the $m = -1$ alias / backward light-line crossing, the electromagnetic wave propagates upstream with negative group velocity ($v_g < 0$), transporting energy back toward the cathode where it reflects off the PEC entrance wall at $z = 0$ and re-modulates the incoming beam—an absolute, backward-wave-oscillator (BWO)-like numerical instability (`studies/p5b-mscan/README.md` Section 3). Consequently, the off-resonant electrostatic bound $\text{alias\_supp} = (\omega_q\Delta z / (2\pi u_{\text{dep}}))^2$ does not bound this electromagnetic crossing.
2. **Why axial grid refinement does not cure it**: The dimensionless crossing phase
   ```math
   k\Delta z = \frac{2\pi u}{c + u}
   ```
   depends only on $u/c$ and is independent of $\Delta z$ (`studies/p5b-mscan/README.md` Section 3; $k\Delta z = 1.015$ in `studies/p5b-mscan/README.md` Section 4). Refining $\Delta z$ only shifts $k$ and $f$ upward in proportion to $1/\Delta z$; the crossing does not leave the grid's resolved band.
3. **Measured settle-factor scan with `shape = 1` (`studies/p5b-mscan/README.md` Section 3)**: On the coarse grid $(10, 1, 474)$ (`sheet_lattice`, $16$ macroparticles/step, $q/m \times 1$), increasing the settle factor grows the $m = 0$ current fluctuation across the 6-period DFT window and biases the fitted $\beta_q$ at $m = 0.015$ (`studies/p5b-mscan/README.md` Section 3):

   | settle factor | $\beta_q$ rel. residual | $m = 0$ rms first $\to$ last sixth [A] |
   |---|---|---|
   | `1.05` | `-2.05e-4` | `3.50e-3 -> 1.49e-3` (transient decaying) |
   | `1.10` | `-4.63e-4` | `7.10e-4 -> 1.91e-3` |
   | `1.15` | `-1.16e-3` | `4.65e-4 -> 2.73e-3` |
   | `1.20` | `-2.83e-3` | `7.05e-4 -> 3.40e-3` |
   | `1.30` | `-1.04e-2` | `1.68e-3 -> 5.18e-3` |
   | `1.50` | `-3.62e-2` | `4.81e-3 -> 9.23e-3` |

4. **Measured mitigation scan and cure (`studies/p5b-mscan/README.md` Section 4)**: At settle factor $1.5$ on the coarse grid $(10, 1, 474)$, a single-pass binomial current filter alone does not suppress the instability because its symbol $\cos^2(k\Delta z/2)$ is $0.76$ at $k\Delta z = 1.015$, whereas increasing the B-spline shape order to `shape = 2` (or `shape = 3`) eliminates the growth (`studies/p5b-mscan/README.md` Section 4):

   | configuration | $m = 0$ rms first $\to$ last sixth [A] | growth [1/s] | run time [s] |
   |---|---|---|---|
   | `shape 1` | `4.81e-3 -> 9.23e-3` | `8.5e8` | `1928` |
   | `shape 2` | `1.38e-5 -> 1.41e-5` | `~0 (2.9e7)` | `3806` |
   | `shape 3` | `1.29e-5 -> 1.26e-5` | `~0 (-2.8e7)` | `7218` |
   | `shape 1 + binomial` | `1.59e-3 -> 6.36e-3` | `1.8e9` | `1982` |
   | `shape 2 + binomial` | `8.74e-6 -> 9.50e-6` | `~0` | `3906` |
   | `shape 3 + binomial` | `8.70e-6 -> 8.32e-6` | `~0` | `7410` |

5. **Cure adopted in V7 (`PARTICLE_SHAPE = 2`) and general order-$p$ symbols**: The cross-validation runs use quadratic B-spline shape `shape = 2` (`PARTICLE_SHAPE = 2` in `examples/_space_charge_waves_common.py`, at a coarse-grid runtime of $3806\;\text{s}$ vs $1928\;\text{s}$ for `shape = 1` and $7218\;\text{s}$ for `shape = 3`; `studies/p5b-mscan/README.md` Section 4), with the discrete DC-depression, D19 entrance-edge kernel, and D20 multistream predictions generalised to arbitrary B-spline order $p \in \{1, 2, 3\}$:
   - **Transverse nodal B-spline weights**: both the $J_z$ (and $\rho$) deposit and the Galerkin $E_z$ gather ($p_g = p - 1$) on $z$-edges at primal $x$-nodes use `bspline_weights_1d(xi, p, staggered=False)` with $\xi = (x + a)/\Delta x$ (`transverse_weight_matrix(x_lines, nx, p)`), dropping weights on PEC wall nodes $x = \pm a$ or outside ($E_z(\pm a) = 0$).
   - **Principal-mode axial symbol**: $S_z(k)^2 = \operatorname{sinc}^p(k\Delta z/2)\operatorname{sinc}^{p_g+1}(k\Delta z/2) = \operatorname{sinc}^{2p}(k\Delta z/2)$ for Galerkin gather ($p_g = p - 1$), and per-wave D18 deposited-current observable factor $\operatorname{sinc}^p(k_n\Delta z/2)$.
   - **First-cell PEC-wall entrance transfer kernel (D19, `entrance_edge_transfer`)**: $|I_{1,\text{part}}(z_0)|/(m I_0) = H_0(p, \Delta z, \Delta t, \omega, u_{\text{dep}})\cos(\beta_q\Delta z/2)$ with analytic bound $\text{tol}_{\text{amp,part}} = 2(v_{\text{inj}}/u_{\text{dep}} - 1)p(\beta_e\Delta z)^2/24 + \bigl(1 - \cos(\beta_q\Delta z/2)\bigr) + 10^{-12}$, accounting for the truncation of the order-$p$ edge-0 kernel at $z < 0$ and the birth wall-closure impulse $C_0(0) \in \{0, 1/8, 1/6\}$ for $p \in \{1, 2, 3\}$.

###### PEC wall-closure current for shape $p \ge 2$ particle creation and absorption

In FIT particle-in-cell coupling, the electromagnetic field sees charge changes exclusively through the deposited dual-face charge flux $W_e$ ($\mathbf{J} = \mathbf{W}\odot\text{itA}/\Delta t$, satisfying $\tilde{\mathbf{S}}\tilde{\mathbf{d}}^{n+1} - \tilde{\mathbf{S}}\tilde{\mathbf{d}}^n = -\tilde{\mathbf{S}}\mathbf{W}$). For a particle created at a PEC cathode wall $\zeta = 0$ ($k_w = 0$) or absorbed at a PEC domain wall $k_w \in \{0, N_n\}$ with B-spline shape order $p \ge 2$, the order-$p$ nodal deposit $q_{\text{vis}}$ (`deposit_charge`) places non-zero weight on interior vacuum nodes adjacent to the wall: at $\zeta = 0$, the axial B-spline weights on $(k = 0, k = 1)$ are $(1, 0)$ for $p = 1$, $(3/4, 1/8)$ for $p = 2$ (with $1/8$ on the masked exterior ghost node $k = -1$), and $(2/3, 1/6)$ for $p = 3$ (with $1/6$ on $k = -1$).

1. **Unclosed birth/absorption artefact for $p \ge 2$ (`studies/p5b-mscan/out6/*.json`, `studies/p5b-mscan/out7/*.json`)**:
   - Without a birth current on injection, the field never sees the interior stencil tail $q_{\text{vis}}[k \ge 1]$ appear, leaving $-q_{\text{vis}}[k \ge 1]$ behind on interior vacuum nodes ($k = 1, \dots$) as an effective immobile charge of opposite sign. Symmetrically, when Esirkepov current deposition terminates at $\boldsymbol{\xi}_1^{\text{dep}}$ upon PEC domain-wall absorption, the remaining interior stencil charge $q_{\text{vis}}$ stays on nodes $N_n - 1, N_n - 2, \dots$ (or $1, 2, \dots$) inside the vacuum.
   - On the grid $(10, 1, 474)$ over $3$ transits ($22471$ steps, $\Delta t = 4.704775\times 10^{-13}\;\text{s}$) in the bulk window $z \in [0.3, 0.7]\,L_z$, the bulk DC depression at the end of the run recorded in `studies/p5b-mscan/out6/*.json` (before the fix) and `studies/p5b-mscan/out7/*.json` (after the fix) is:

     | | before fix (`out6`) | after fix (`out7`) | $\delta_{\text{disc}}$ |
     |---|---|---|---|
     | $p = 2$ | `2.751 %` | `0.993 %` | `1.0312 %` |
     | $p = 3$ | `4.04 %` | `0.971 %` | `1.0207 %` |

   - After the fix, $\delta$ is constant over the last two transits (`studies/p5b-mscan/out7/*.json`).

2. **1D normal-axis telescoping wall-closure current (`_pec_wall_closure_1d`, `tests/test_090_particles_wall_closure.py`)**:
   - Applied exclusively to events on PEC domain faces (`emitter_kind == "pec_wall"` injectors and absorptions on domain walls with `bc in ("pec", "electric")`; internal user `Collector`s and `emitter_kind == "interior_sheet"` emitters retain their stationary space-charge sheets).
   - Let $n \in \{0, 1, 2\}$ be the wall normal axis, $k_w$ the wall node index ($0$ on the low face, $N_n$ on the high face), and $q_{\text{vis}}$ the order-$p$ nodal charge deposit of the event's particles (at birth position for injection, or at truncated stop coordinate $\boldsymbol{\xi}_1^{\text{dep}}$ for absorption). Column by column in the two transverse indices, the normal-axis edge charge flux $W_n$ (on primal edges between nodes $k$ and $k + 1$) receives the 1D telescoping sum that moves all visible event charge between the wall node $k_w$ and its interior stencil nodes:
     ```math
     \Delta W_n[k] = \begin{cases}
     +\sum_{k' > k} q_{\text{vis}}[k'] & \text{(low face } k_w = 0\text{, injection; } -\text{ for absorption)}, \\[4pt]
     +\sum_{k' \le k} q_{\text{vis}}[k'] & \text{(high face } k_w = N_n\text{, absorption; } -\text{ for injection)},
     \end{cases}
     ```
     so that $-(\tilde{\mathbf{S}}\Delta\mathbf{W})_k = +q_{\text{vis}}[k]$ on interior nodes $k \ne k_w$ for injection (and $-q_{\text{vis}}[k]$ for absorption), placing the entire compensating image charge on the conductor wall node $k_w$. For $p = 1$ at $\zeta = 0$ (or $\zeta = N_n$), $q_{\text{vis}}[k \ne k_w] = 0$ and $\Delta W_n$ is identically zero.
   - **Surface-charge bookkeeping and wall-plane Gauss check**: By B-spline symmetry at the wall node $k_w$, the masked exterior ghost weight $w_{-1}$ equals the interior stencil tail $\sum_{k' \ne k_w} q_{\text{vis}}[k']$, so `solver._particles_q_bg` records the full particle charge $2\sum_{k'} q_{\text{vis}}[k'] - q_{\text{vis}}[k_w]$ on the wall node plane $k_w$ (`-=` for injection, `+=` for absorption) so that both $\sum(\tilde{\mathbf{S}}\tilde{\mathbf{d}}) = \sum(q_{\text{nodes}} + q_{\text{bg}})$ and $(\tilde{\mathbf{S}}\tilde{\mathbf{d}})_{k_w} = (q_{\text{nodes}} + q_{\text{bg}})_{k_w}$ hold through injection, transit, and absorption. For PEC domain faces, `_interior_vacuum_node_mask` already evaluates all interior vacuum node planes $1, \dots, N_n - 1$ by default (excluding only the wall nodes $k_w$ and CPML nodes); the `include_wall_planes=True` keyword only shrinks the B-spline stencil margin next to CPML inner interfaces to $1$ (`tests/test_090_particles_wall_closure.py`).
   - **PEC-wall birth kick for $p \ge 2$ (`_apply_pec_wall_birth_kick`, `wall_kick_energy()`)**: For shape order $p \ge 2$, a macroparticle born by a `"pec_wall"` injector at the wall node $k_w$ deposits visible stencil weight on adjacent vacuum nodes ($w_1 = 1/8$ on node $1$ for $p = 2$ at $\zeta = 0$, $w_1 = 1/6$ for $p = 3$). Because the wall-closure current $\Delta\mathbf{W}_{\text{cl}}$ carries that charge from $k_w$ to the interior stencil nodes, the electric field does work on it while an un-kicked particle starts with effective potential energy $q\sum_{\mathbf{k}} W_{\mathbf{k}}^{(p)}\phi_{\mathbf{k}} \ne q\phi_{\text{wall}}$ and reaches the bulk with excess kinetic energy, leaving the bulk $\delta$ low by about $w_1\phi_1/\phi_{\text{bulk}}$ (entrance diagnostic: $-3.7\,\%$ for $p = 2$, $-4.9\,\%$ for $p = 3$; in slow `test_087` V7 `M3 DEPCHECK` with the wall-closure current alone, the measured bulk DC depression was $\delta_{\text{meas}} = 1.053027\times 10^{-2}$ vs the discrete model $\delta_{\text{disc}} = 1.094299\times 10^{-2}$, an error of $4.1\times 10^{-4}$). To make injection energy-consistent (decision D17 option (b), adopted for injection; with the kick, the entrance-diagnostic $\delta$ error is $+1.0\times 10^{-4}$ for $p = 2$ and $+3.7\times 10^{-5}$ for $p = 3$, and the slow M3 $\delta$ error falls from $4.1\times 10^{-4}$ to $1.7\times 10^{-6}$), at each `"pec_wall"` injection event with $p \ge 2$ (`_ENABLE_PEC_WALL_KICK = True`; skipped for $p = 1$ so $p = 1$ remains bit-identical), the solver builds the 1D line integral $\Psi$ of the normal electric field $E_n^n$ from the wall node $k_w$ ($\Psi[k_w] = 0$) along the wall normal using primal edge lengths $\Delta l_e$:
     ```math
     \Psi[k] = \begin{cases}
     +\sum_{e = k_w}^{k - 1} E_n^n[e]\,\Delta l_e & \text{(low face } k_w = 0,\; k > k_w\text{)}, \\[4pt]
     -\sum_{e = k}^{k_w - 1} E_n^n[e]\,\Delta l_e & \text{(high face } k_w = N_n,\; k < k_w\text{)},
     \end{cases}
     ```
     and gathers the birth kinetic-energy change for each newly injected particle $i$ with the order-$p$ nodal B-spline weights and ghost-node masking of `deposit_charge`:
     ```math
     \Delta K_i = q_i w_i \sum_{\mathbf{k}} W^{(p)}(\mathbf{x}_i - \mathbf{x}_{\mathbf{k}})\,\Psi_{\mathbf{k}}.
     ```
     Summing over the event's particles and rearranging the prefix sum yields the exact discrete identity with the closure work on the event's edges at field $E_n^n$:
     ```math
     \sum_i \Delta K_i = \sum_{\mathbf{k}} q_{\text{vis}}[\mathbf{k}]\,\Psi[\mathbf{k}] = \sum_e \Delta W_{\text{cl},e}\,E_n^n[e]\,\Delta l_e.
     ```
     Each particle's birth proper velocity $\mathbf{u}_i(t_n)$ is updated at constant direction before `_backpush_injected_slice`:
     ```math
     \gamma_i' = \gamma_i + \frac{\Delta K_i}{w_i m_s c^2}, \qquad \mathbf{u}_i' = \mathbf{u}_i \frac{\sqrt{{\gamma_i'}^2 - 1}}{\sqrt{\gamma_i^2 - 1}},
     ```
     raising `ValueError` if $\gamma_i' < 1$ (particle cannot leave the wall), and `self._K_wall_kick` accumulates $\sum_i \Delta K_i$ locally (allreduced in `ParticleSolver.wall_kick_energy()`, persisted under key `"wall_kick"` in checkpoints).
   - **Wall-closure work and birth kick in `ParticleSolver.energy_balance()`**: Because $\Delta\mathbf{W}_{\text{cl}}$ is added to `ps.J_particles` (after MPI guard reduction, optional binomial filtering, and $\text{itA}/\Delta t$ scaling into $\mathbf{J}_{\text{cl}}$), the electromagnetic field performs work on the closure current (`field_work()` includes it). When `track_energy=True`, `ParticleSolver.wall_closure_work()` accumulates $\sum_m \Delta t\,\mathbf{J}_{\text{cl}}^{m+1/2}\cdot\mathbf{V}_E\bar{\mathbf{e}}^{m+1/2}$ as `"W_wall_closure"`, `ParticleSolver.wall_kick_energy()` reports the cumulative birth-kick kinetic energy change $\sum \Delta K_i$ as `"K_wall_kick"` (`"K_injected"` stays the pre-kick injected kinetic energy), and `energy_balance()` computes `"residual"` as `(K_injected + E_init) - K_absorbed - K_particles - W_field - W_wall_closure + K_wall_kick`.
   - **Limits & failure modes**: The closure work per event scales as $q\,w_1 E_{\text{wall}}\,h$, so $W_{\text{wall\_closure}} / K_{\text{injected}} = \mathcal{O}(h)$ is a grid artefact that vanishes as $h \to 0$ and does not depend on $\Delta t$; for injection the work is given to the particle (birth kick), whereas for absorption it is reported as `W_wall_closure` (the particle is already gone), and `interior_sheet` emitters and in-vacuum `Collector`s are not closed (intentional).

References:
- S. Ramo, "Space Charge and Field Waves in an Electron Beam", *Phys. Rev.* **56**(3), 276–283 (1939), [doi:10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276).
- W. C. Hahn, "Small Signal Theory of Velocity-Modulated Electron Beams", *Gen. Elec. Rev.* **42**, 258–270 (1939).
- D. L. Webster, "Cathode-Ray Bunching", *J. Appl. Phys.* **10**(7), 501–508 (1939), [doi:10.1063/1.1707335](https://doi.org/10.1063/1.1707335).
- G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in Electron Beams", *IRE Trans. Electron Devices* **2**(2), 3–11 (1955), [doi:10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065).
- F. Paschke, "On the Nonlinear Behavior of Electron-Beam Devices", *RCA Rev.* **18**(2), 221–242 (1957).
- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, Taylor & Francis / IOP Publishing, Bristol (1991), ISBN 978-0-7503-0117-6.
- T. Zh. Esirkepov, "Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor", *Comput. Phys. Commun.* **135**(2), 144–153 (2001), [doi:10.1016/S0010-4655(00)00228-9](https://doi.org/10.1016/S0010-4655(00)00228-9).
