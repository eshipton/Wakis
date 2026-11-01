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

As electrons enter through the grounded plane $z = 0$ ($\phi_{\text{DC}}(x, 0) = 0$) into the interior potential trough, exact relativistic energy conservation along each beam line $\ell$ (at transverse coordinate $x_\ell$) determines the depressed Lorentz factor $\gamma_\ell$ and axial drift velocity $u_\ell$, while DC charge continuity ($J_{0,z,\ell} = \rho_{\text{dep},\ell} u_\ell = \rho_{0,\ell} u_0$) scales each line's charge density by $u_0 / u_\ell$. Both the continuum depression $\delta_{\text{th}}$ (`compute_analytic_parameters`) and the profile-consistent discrete 1D 3-point Poisson depression $\delta_{\text{disc}}$ (`compute_discrete_dc_depression`, where at each fixed-point iteration the 3-point discrete Laplacian on $N_x$ cells with PEC boundary conditions $\phi(\pm a) = 0$ is solved with the per-line depressed charge deposit $\propto w_\ell (u_0 / u_\ell)$ and CIC-gathered back onto the actual line positions $x_\ell$) iterate the per-line energy-conserving fixed point to machine precision ($10^{-14}$ tolerance):

```math
\mathbf{L}_x\,\boldsymbol{\phi}_{\text{nodes}} = -\frac{\boldsymbol{\rho}_{\text{nodes}}(\mathbf{u})}{\varepsilon_0}, \qquad \gamma_\ell = \gamma_0 - \frac{e\,|\phi_\ell|}{m_{\text{eff}} c^2}, \qquad u_\ell = c\sqrt{1 - \frac{1}{\gamma_\ell^2}}, \qquad \bar{\delta} = 1 - \frac{\sum_\ell w_{\text{avg},\ell} u_\ell}{u_0 \sum_\ell w_{\text{avg},\ell}}.
```

Two line-weighting conventions are evaluated and reported in `DEPCHECK`:
- **Density weighting ($w_{\text{avg},\ell} \propto w_\ell / u_\ell$)**: Since injected beam lines with current weights $w_\ell$ carry steady line charge density $\rho_\ell \propto w_\ell / u_\ell$, density weighting matches the bin-accumulated particle measurement `u_z_mean` (`sum(w_p v_{z,p}) / sum(w_p)` over $z \in [0.2 L_z, 0.8 L_z]$) and is used as `delta_disc_exact` (both for the uniform `"sheet_lattice"` profile and for the center-peaked Fibonacci disk profile in `DEPCHECK id=M3`).
- **Current/flux weighting ($w_{\text{avg},\ell} \propto w_\ell$)**: Injected-current weighting (`delta_disc_flux`).

At the converged continuum equilibrium $u_{\text{dep}} = u_0(1 - \delta_{\text{th}})$, the depressed parameters passed to `LinearizedBeam` (Option 3(ii)) are

```math
\gamma_{\text{dep}} = \frac{1}{\sqrt{1 - (u_{\text{dep}}/c)^2}}, \qquad \rho_{\text{dep}} = \rho_0\,\frac{u_0}{u_{\text{dep}}}, \qquad \omega_{p,\text{dep}} = \omega_p\sqrt{\frac{u_0}{u_{\text{dep}}}}.
```

For the deterministic `"sheet_lattice"` profile, the measured interior DC velocity depression $\delta_{\text{meas,lattice}}$ is checked against $\delta_{\text{disc}}$ up to the second-order AC velocity-modulation shift $\varepsilon_{\text{AC}}$:

```math
\bigl|\delta_{\text{meas,lattice}} - \delta_{\text{disc}}\bigr| \le \varepsilon_{\text{AC}} + 10^{-12}, \qquad \varepsilon_{\text{AC}} = \frac{1}{2}\left(m\,\frac{\beta_q}{\beta_e}\right)^2.
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
     where $\Omega_f = (2/\Delta t)\sin(\omega\Delta t/2)$ uses that grid's $\Delta t$ and beam loading only adds faster-decaying transients ($\kappa_{\text{DC}} > \gamma_{1,\text{disc}}$, verified as an identity check and printed in `TERMS`). Requiring the relative evanescent near-field amplitude at the window edges to decay below $\varepsilon_w = 10^{-5}$ gives the per-grid margin $d_{w,\text{grid}} = \ln(1/\varepsilon_w)/\gamma_{1,\text{disc}}$, and taking the largest $d_{w,\text{grid}}$ (smallest $\gamma_{1,\text{disc}}$ across the coarse and fine grids) defines the single common fit window used across all fits:
     ```math
     d_{w,\text{common}} = \max_{\text{coarse, fine}} d_{w,\text{grid}}, \qquad z_{\min} = d_{w,\text{common}}, \qquad z_{\max} = (L_z - L_{\text{ramp}}) - d_{w,\text{common}}, \qquad z_{\max} - z_{\min} \ge 1.25\,\lambda_q.
     ```
   - **Fitted carrier $k_c$ and variable projection**: The carrier wavenumber $k_c$ is fitted simultaneously with $\beta_q$ rather than fixed to $\beta_e = \omega/u_{\text{dep}}$ because `LinearizedBeam`'s centered Eulerian advection shifts its numerical carrier by $\simeq \frac{1}{6}(\beta_e\Delta z)^2$, and fixing $k_c = \beta_e$ would bias $\beta_q$. For any trial pair $(k_c, \beta_q)$, the linear parameters $(a_f, a_s)$ follow from a 2-column complex linear least-squares solve (`numpy.linalg.lstsq`) on the design matrix $[\mathrm{e}^{-\mathrm{i} k_f z_k},\,\mathrm{e}^{-\mathrm{i} k_s z_k}]$. Minimizing the complex residual sum of squares $\text{RSS}(k_c, \beta_q) = \sum_k |I_1(z_k) - I_{1,\text{fit}}(z_k)|^2$ on a $201\times 401$ grid over $k_c \in k_{c,0}(1 \pm 0.03)$ (where $k_{c,0}$ is the unwrapped phase slope of $I_1(z)$) and $\beta_q \in [0.5, 1.8]\,\beta_{q,\text{guess}}$ followed by Nelder–Mead polishing (`xatol = 1e-10 * beta_q_guess`) yields $(k_c, \beta_q, A, B, a_f, a_s)$. The fit uncertainty $\sigma_{\beta_q}$ is obtained from the Gauss–Newton covariance $s^2(\mathbf{J}^{\mathsf{H}}\mathbf{J})^{-1}$ with $s^2 = \text{RSS}/(2N - 6)$ ($N$ complex points, $6$ real parameters).
   - **Outflow ramp-sensitivity and parasitic-branch diagnostics (`DIAG id=lb_ramp`, `DIAG id=lb_parasitic`)**:
     - `DIAG id=lb_ramp` compares `LinearizedBeam` on the coarse and fine grids with $L_{\text{ramp}} = 8\,\text{mm}$ and $4\,\text{mm}$, both fitted on the $8\,\text{mm}$ window $[d_w, L_z - 8\,\text{mm} - d_w]$ so both ramps lie at least $d_w$ downstream of $z_{\max}$, alongside the four-wave forward/backward reflection fit (`fit_four_wave_reflection`).
     - `DIAG id=lb_parasitic` (`fit_lb_parasitic_waves`) augments the variable-projection basis for `LinearizedBeam` with the two centered-difference odd–even parasitic branches $k_{p,\mp} \in (\pi/(2\Delta z),\,\pi/\Delta z)$ (propagating with negative group velocity) that solve the discrete `LinearizedBeam` dispersion relation $\Omega_{\text{LB}} \mp \omega_{q,1}(k_p) = u_{\text{dep}}\sin(k_p\Delta z)/\Delta z$ via `scipy.optimize.brentq` with the fundamental transverse eigenvalue $\lambda_1(k_p)$ and reduction factor evaluated at $k_p$, adding two linear complex amplitudes and no new wavenumber parameters. Because the resulting relative shift $|\Delta\beta_q/\beta_q|$ on the coarse, fine, and **M2** runs is orders of magnitude smaller than the `LinearizedBeam` residual, the 2-wave fit is retained as the `LinearizedBeam` estimator across all checks.
   - **Amplitude sanity check vs report-only launch phase**: For a pure density modulation, $|a_f| = |a_s|$ and $B/A = \tan(\beta_q z_0)$ where $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$ is the effective launch offset induced by cathode image-charge screening and weak higher-order modes. Because no closed-form boundary-layer derivation predicts $z_0$ or $B/A$, $|a_s|/|a_f|$, $B/A$, and $z_0$ are reported as diagnostics without assertion, while $|A| > \frac{1}{2}m I_0$ is asserted to verify that the fit locked onto the driven wave rather than noise.
2. **Spatial wavenumber spectrum (`compute_spatial_spectrum`)**: The spatial Fourier transform over $z \in [0, \lambda_q]$,
   ```math
   \hat{I}_1(k_z) = \int_0^{\lambda_q} I_1(z, \omega)\,\mathrm{e}^{+\mathrm{i}k_z z}\,\mathrm{d}z,
   ```
   splits into a doublet of peaks at $k_f = \beta_e - \beta_q$ and $k_s = \beta_e + \beta_q$ separated by a deep null at $k_z = \beta_e$.

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

##### 6. Numerical setup, injection profiles, and exact discrete transverse modal model

To compare `ParticleSolver` and `LinearizedBeam` without adjustable parameters (`examples/_space_charge_waves_common.py`, `tests/test_087_particles_vs_linearized_beam.py`, `examples/024_particles_space_charge_waves.py`):

1. **Shared grid, integer-period time step, and DFT mean subtraction**: Both solvers run on the identical `GridFIT3D` ($x \in [-a, a]$ with `bc="pec"`, $y \in [-L_y/2, L_y/2]$ with `ny = 1` and `bc="periodic"`, $z \in [0, L_z]$ with `bc="pec"`). The time step is chosen as $\Delta t = 1/(f N_{\text{spp}})$ with $N_{\text{spp}} = \lceil 1/(f\Delta t_{\text{CFL}})\rceil$ ($\Delta t \le \Delta t_{\text{CFL}}$), so the 6-period DFT window $N_{\text{DFT}} = 6 N_{\text{spp}}$ spans an exact integer number of RF cycles. Per-edge mean subtraction over the $N_{\text{DFT}}$ window eliminates DC current leakage; the integer-$N_{\text{spp}}$ Dirichlet kernel leakage is bounded by the round-off **identity** $\varepsilon_{\text{DFT}} = 100\,\epsilon_{\text{mach}}/m$.
2. **Transverse sheet injection profiles (`"sheet_lattice"` vs `"sheet"`)**: Every time step, `BeamInjector.inject` (`wakis/particles/boundaries.py`) places $N_{\text{macro}}$ macroparticles on $z = 0$ with $y_p = 0$ exactly, $|x_p| \le b$, longitudinal momentum $u_z = \gamma_0 u_0$, $\mathbf{u}_\perp = \mathbf{0}$, and time-modulated weight $w(t^n) = |I(t^n)|\Delta t / (|q| N_{\text{macro}})$:
   - **Deterministic reference profile (`profile="sheet_lattice"`)**: Places one macroparticle at the midpoint of each equal-width transverse stratum, $x_\ell = b\bigl(2(\ell + \tfrac{1}{2})/N_{\text{macro}} - 1\bigr)$ ($\ell = 0, \dots, N_{\text{macro}} - 1$). This profile has zero transverse shot noise, is identical for all random seeds, and is used for every signed physics check, all three mutations (**M1**, **M2**, **M3**), the two-resolution check, and the $m$-scan.
   - **Seeded stratified profile (`profile="sheet"`)**: Draws one uniform sample per stratum per step, $x_\ell \in [-b + 2b\ell/N_{\text{macro}}, -b + 2b(\ell+1)/N_{\text{macro}})$, from `BeamInjector._step_rng(step_idx)`. Because `seed` only affects `profile="sheet"`, the 8-seed ensemble (`measure_empirical_noise`) calls `run_particle_beam_drift(..., profile="sheet", seed=s)` at $N_{\text{macro}} = 256$ (coarse) and $512$ (fine), the same per-cell density across both grids, and compares against the `"sheet_lattice"` reference at the same $N_{\text{macro}}$.
   - `test_injected_sheet_geometry` directly verifies these geometric invariants as an **identity (geometry) check** after 3 steps on CPU.
3. **Guide magnetic field**: A strong uniform axial guide field $B_{z,\text{ext}}$ ($\omega_c \gg \omega_p$) freezes out transverse space-charge expansion while leaving longitudinal motion along $\hat{\mathbf{z}}$ unperturbed [Birdsall & Langdon, 1991].
4. **Exact discrete transverse modal model (`discrete_modal_prediction`)**: Rather than approximating transverse finite-difference and CIC effects at leading order in $(T_1\Delta x)^2$, `discrete_modal_prediction(code, nx, nz, params, ...)` solves the exact discrete transverse operator, profile-consistent DC depression, and multi-mode axial superposition in seven steps:
   1. **Axial and temporal symbols, and the drifting-beam axial shape factor $S_z(k)^2$**:
      - Staggered Yee field operator along $z$: $\kappa(k) = \frac{2}{\Delta z}\sin(k\Delta z/2)$, and leapfrog field time symbol $\Omega_f = \frac{2}{\Delta t}\sin(\omega\Delta t/2)$.
      - **Drifting-beam axial shape factor $S_z(k)^2 = \operatorname{sinc}^2(k\Delta z/2)$ (particles)**: Along $z$, the $p = 1$ charge-conserving Esirkepov segment deposit onto primary $z$-edges deposits the $0$-th order piecewise-constant indicator $W_0$ over each sub-step segment $[z^n, z^{n+1}]$, giving the one-way axial deposit symbol $S_{\text{dep}}(k) = \operatorname{sinc}(k\Delta z/2)$ (`sinc^{p}` along the current direction, `wakis/particles/solver.py`). Conversely, the Galerkin primary-edge electric-field gather (`gather_order` $p_g = 0$ along the edge axis $z$) interpolates $E_z$ with the $0$-th order nearest-cell indicator $W_0(z - z_{k+1/2})$, giving the one-way axial gather symbol $S_{\text{gath}}(k) = \operatorname{sinc}^{p_g+1}(k\Delta z/2) = \operatorname{sinc}(k\Delta z/2)$. For a stationary cold plasma where every Brillouin alias $k_m = k + 2\pi m/\Delta z$ ($m \in \mathbb{Z}$) shares the same temporal denominator $\Omega_p^2$, summing over all aliases restores $\sum_m \operatorname{sinc}^2(k_m\Delta z/2) = 1$ (`\kappa_z^2 = 1` in `wakis/particles/solver.py`). For a **drifting** beam, however, the non-principal aliases $m \ne 0$ are Doppler-detuned in the beam frame by $|\omega - k_m u_{\text{dep}}| \simeq 2\pi|m|u_{\text{dep}}/\Delta z \gg \omega_q$, so their contribution to the space-charge response $(\omega - k_m u_{\text{dep}})^{-2}$ is suppressed by the factor
        ```math
        \text{alias\_supp} = \left(\frac{\omega_q\Delta z}{2\pi u_{\text{dep}}}\right)^2 \ll 1.
        ```
        Consequently, only the principal ($m = 0$) Brillouin zone survives, and the round-trip deposit-and-gather coupling acquires the axial shape factor
        ```math
        S_z(k)^2 = S_{\text{dep}}(k)\,S_{\text{gath}}(k) = \operatorname{sinc}^2\!\left(\frac{k\Delta z}{2}\right), \qquad \operatorname{sinc}(\xi) \equiv \frac{\sin\xi}{\xi}.
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
      - Particles: the beam lines are the injected particle positions $x_\ell$ (`"sheet_lattice"` midpoints at the active $N_{\text{macro}}$, or any non-uniform `line_x` array such as the Fibonacci disk positions in **M3**), with coupling matrix $S_{i,\ell} = W_{\text{CIC}}(x_i - x_\ell)\,(2b / N_{\text{lines}})/\Delta x$ and Galerkin ($p = 1$) CIC gather along $x$. Before assembling the AC modal operator, `discrete_modal_prediction` calls `compute_discrete_dc_depression(nx, params, line_x=line_x, line_w=line_w)` on the **actual line set** $(\mathbf{x}_\ell, \mathbf{w}_\ell)$ to obtain the profile-consistent per-line DC velocity $u_\ell$, Lorentz factor $\gamma_\ell$, per-line depressed plasma frequency $\omega_{p,\ell}^2 = \omega_{p,0}^2(u_0 / u_\ell)$, and profile-consistent mean depressed velocity $\bar{u}$ and depression $\bar{\delta}$, never reusing the uniform-beam `params["u_dep"]` for a non-uniform profile.
   4. **Eigenproblem (`compute_transverse_modal_operator`)**: With line charge fractions $\mathbf{D} = \operatorname{diag}(d_\ell)$ ($\sum_\ell d_\ell = 1$), the discrete transverse modal operator
      ```math
      \mathbf{G}(k) = \frac{\Delta x}{2b}\,\tau_{\text{Y}}(k)^2\,\mathbf{S}^{\mathsf{T}}\,\mathbf{A}(k)^{-1}\,\mathbf{S}\,\mathbf{D}, \qquad \widetilde{\mathbf{G}}(k)\,\mathbf{y}_j = \lambda_j(k)\,\mathbf{y}_j,
      ```
      is normalized so that for a uniform slab as $\Delta x \to 0$, $\lambda_1(k) \to R\bigl(\tau_{\text{Y}}(k)\bigr)^2$ (`test_discrete_modal_operator_unit`, `DIAG id=modal_unit`). For particles, each line's susceptibility includes its profile-consistent DC velocity and Lorentz factor via $\sigma_\ell = (\omega_{p,\ell}^2 / \omega_{p,\text{ref}}^2)(\bar{u}/u_\ell)^2(\bar{\gamma}/\gamma_\ell)^3 = (\bar{u}/u_\ell)^3(\bar{\gamma}/\gamma_\ell)^3$ in $\widetilde{\mathbf{G}}(k) = \operatorname{diag}(\sigma_\ell)\mathbf{G}(k)$. Including the axial shape factor, the modal plasma frequencies are
      ```math
      \omega_{q,j}(k)^2 = \lambda_j(k)\,\frac{\omega_{p,\text{dep}}^2}{\gamma_{\text{dep}}^3} \quad (\text{LinearizedBeam}), \qquad \omega_{q,j}(k)^2 = \lambda_j(k)\,\frac{\omega_{p,\text{ref}}^2}{\bar{\gamma}^3}\,S_z(k)^2 \quad (\text{particles}).
      ```
   5. **Axial dispersion per mode**: For each excited mode $j$, the fast ($k_{-,j}$) and slow ($k_{+,j}$) axial wavenumbers solve
      ```math
      \Omega \mp \omega_{q,j}(k_{\pm,j}) = u_{\text{dep}}\,k_s(k_{\pm,j}) \quad (\text{LinearizedBeam}), \qquad \Omega_p \mp \omega_{q,j}(k_{\pm,j}) = \bar{u}\,k_{\pm,j} \quad (\text{particles}),
      ```
      giving $\beta_{q,j} = (k_{+,j} - k_{-,j})/2$ and $k_{c,j} = (k_{+,j} + k_{-,j})/2$.
   6. **Entrance excitation, first-cell entrance amplitude, and model envelope fit**: Expanding the entrance density pattern $\mathbf{e}_{\text{in}}$ (`LinearizedBeam`: `inflow_rho1` $\propto \mathbf{w}_{\perp,\text{eff}}$; particles: equal charge per line) in the biorthonormal left and right eigenvectors $(\mathbf{l}_j, \mathbf{y}_j)$ gives the modal current amplitudes $c_j = (\mathbf{1}^{\mathsf{T}}\mathbf{D}\mathbf{y}_j)(\mathbf{l}_j^{\mathsf{T}}\mathbf{e}_{\text{in}})$. Synthesizing
      ```math
      I_{1,\text{model}}(z_k) = m I_0 \sum_j c_j \cos(\beta_{q,j} z_k)\,\mathrm{e}^{-\mathrm{i} k_{c,j} z_k}
      ```
      on the primary edge midpoints $z_k$ and fitting $I_{1,\text{model}}(z_k)$ with `fit_beta_q_from_envelope` over $[z_{\min}, z_{\max}]$ yields the parameter-free modal prediction $\beta_{\text{pred}}$. On the first primary edge $z_0 = \Delta z / 2$, `LinearizedBeam` evaluates the point value $\cos(\beta_q\Delta z/2)$ with truncation bound $(\beta_q\Delta z/2)^4/24 + 10^{-12}$, whereas the particle edge-0 current is the Esirkepov flux averaged over a one-cell transit $z \in [0, \Delta z]$:
      ```math
      \frac{|I_{1,\text{part}}(z_0)|}{m I_0} = \operatorname{sinc}\!\left(\frac{\beta_e\Delta z}{2}\right)\cos\!\left(\frac{\beta_q\Delta z}{2}\right), \qquad \text{tol}_{\text{amp,part}} = \frac{(\beta_e\Delta z / 2)^4}{120} + \frac{(\beta_q\Delta z / 2)^4}{24} + 10^{-12}.
      ```
   7. **Particle DC velocity shear (`DIAG id=shear`) and factorial residual separation (`DIAG id=particle_factorial`)**: `DIAG id=shear` compares the uniform-$u_{\text{dep}}$ modal prediction against the profile-consistent shear-weighted prediction $\widetilde{\mathbf{G}}(k) = \operatorname{diag}(\sigma_\ell)\mathbf{G}(k)$. In `test_axial_vs_transverse_refinement`, `DIAG id=particle_factorial` separates the remaining particle lattice residual across grid resolution ($10\times 474$, $20\times 948$, $10\times 948$, $20\times 474$), transverse line count ($N_{\text{macro}} = 16$ vs $64$ on coarse; $32$ vs $128$ on fine, each evaluated against its own `line_x`-matched modal prediction), and modulation amplitude ($m$ vs $m/2$, with Richardson extrapolation $\beta_0(m, m/2)$), verifying that the **D6** shift between $N_{\text{macro}} = 16$ and $64$ is matched by the transverse midpoint quadrature shift of `discrete_modal_prediction`.
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
   - Every check comparing particle measurements with the linear modal prediction uses the $m \to 0$ extrapolated wavenumber $\beta_0$:
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

- **Seeded sheet vs lattice at low macroparticle density**: With `profile='sheet'` at low `n_macro_per_step`, step-to-step random transverse positions raise the fitted β_q and the DC depression, and cause a small energy deficit. The effect is not a fit artefact, not static disorder and not a warm-beam shift (see DIAG A1–A5), and it vanishes at higher macroparticle density. Use `profile='sheet_lattice'` for deterministic checks.
- **Small-signal linear regime only ($m \ll 1$, $X(z) \ll 1$)**: `LinearizedBeam` linearizes the cold-fluid equations around a time-independent equilibrium and omits the quadratic convective terms $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$. When the modulation index $m$ or drift length $z$ is increased until the bunching parameter $X \sim \mathcal{O}(1)$, fast electrons overtake slow electrons (wave breaking / multi-stream trajectory crossing), generating strong higher harmonics ($2\omega, 3\omega$) and electrostatic trapping in `ParticleSolver` that `LinearizedBeam` cannot represent.
- **Planar sheet-beam geometry and strong axial magnetic focusing only**: The analytical reduction factor `reduction_factor_planar` and the `"sheet"` / `"sheet_lattice"` injector profiles assume a 1D planar sheet beam uniform along $y$ (`ny = 1`, `bc="periodic"`) and confined by an axial guide field $B_{z,\text{ext}}$ strong enough ($\omega_c \gg \omega_p$) to freeze out transverse velocity perturbations $\mathbf{v}_\perp$. At weak or zero guide field, transverse space-charge forces cause beam spreading or Brillouin scalloping; 3D pencil or rectangular beams require `reduction_factor_cylindrical` or `reduction_factor_rectangular` [Branch & Mihran, 1955].
- **Effective launch offset $z_0$ and sine-quadrature amplitude $B$ are not predicted analytically**: Near the grounded PEC cathode plane $z = 0$, image-charge screening of $E_x$ over the evanescent zone $\ell_{\text{cath}} \sim 1/\gamma_1$ and multi-mode excitation produce a small launch offset $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$. Because no closed-form cathode boundary-layer derivation exists for $z_0$ or $B/A$, these quantities are reported as diagnostics only and are not asserted.
- **Two-wave fit window requires $z_{\max} - z_{\min} \ge 1.25\,\lambda_q$**: Separating two closely spaced complex exponentials $k_f = k_c - \beta_q$ and $k_s = k_c + \beta_q$ alongside a free carrier $k_c$ requires the interior fit window $[d_w, L_z - L_{\text{ramp}} - d_w]$ ($d_w = \ln(1/\varepsilon_w)/\gamma_1$) to span at least $1.25\,\lambda_q$ (`test_preflight_and_resolution_criteria`). On shorter tunnels where $z_{\max} - z_{\min} \lesssim \lambda_q$, the two wave columns become nearly collinear and the variable-projection estimate of $(k_c, \beta_q)$ becomes ill-conditioned.
- **Transverse injection profile and profile-consistent DC depression must match (mutation M3 and diagnostic D3)**: `LinearizedBeam` with `half_thickness=b` represents a uniform top-hat planar slab across $x \in [-b, b]$ and excites its inflow perturbation proportional to `_w_perp_eff`. If the particle injector instead samples a Fibonacci disk profile (**M3**, `test_mutation_disk_profile_instead_of_sheet_fails`), its center-peaked $x$-marginal both increases the fundamental transverse eigenvalue $\lambda_1$ and deepens the DC potential trough (larger $\delta_{\text{disc,M3}}$ and stronger transverse velocity shear, `DEPCHECK id=M3`), shifting $\beta_{q,\text{P}}$ above the uniform-sheet prediction (failing the baseline check while matching `discrete_modal_prediction` evaluated with the profile-consistent DC depression at the Fibonacci `line_x`). Comparing `LinearizedBeam` on the same non-uniform profile (**D3**) requires depositing the **M3** particle positions onto the $x$-nodes via CIC while keeping uniform `u_dep` because `LinearizedBeam` has no DC space-charge physics.

References:
- S. Ramo, "Space Charge and Field Waves in an Electron Beam", *Phys. Rev.* **56**(3), 276–283 (1939), [doi:10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276).
- W. C. Hahn, "Small Signal Theory of Velocity-Modulated Electron Beams", *Gen. Elec. Rev.* **42**, 258–270 (1939).
- D. L. Webster, "Cathode-Ray Bunching", *J. Appl. Phys.* **10**(7), 501–508 (1939), [doi:10.1063/1.1707335](https://doi.org/10.1063/1.1707335).
- G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in Electron Beams", *IRE Trans. Electron Devices* **2**(2), 3–11 (1955), [doi:10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065).
- F. Paschke, "On the Nonlinear Behavior of Electron-Beam Devices", *RCA Rev.* **18**(2), 221–242 (1957).
- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, Taylor & Francis / IOP Publishing, Bristol (1991), ISBN 978-0-7503-0117-6.
- T. Zh. Esirkepov, "Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor", *Comput. Phys. Commun.* **135**(2), 144–153 (2001), [doi:10.1016/S0010-4655(00)00228-9](https://doi.org/10.1016/S0010-4655(00)00228-9).
