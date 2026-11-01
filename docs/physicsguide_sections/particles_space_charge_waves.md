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

As electrons enter through the grounded plane $z = 0$ ($\phi_{\text{DC}}(x, 0) = 0$) into the interior potential trough, relativistic energy conservation $\bigl(\gamma(x) - \gamma_0\bigr)m_e c^2 = -e\phi_{\text{DC}}(x)$ with $\mathrm{d}\gamma = \gamma_0^3 (u_0/c^2)\,\mathrm{d}u$ reduces the cross-section-averaged axial drift velocity from $u_0$ to $u_{\text{dep}} = u_0(1 - \delta_{u,\text{DC}})$ and increases the steady interior charge density by DC continuity ($J_{0,z} = \rho_{\text{dep}} u_{\text{dep}} = \rho_0 u_0$). Starting from the leading-order fractional velocity depression

```math
\delta_0 = \frac{-e\langle\phi_{\text{DC}}\rangle_{|x|\le b}}{\gamma_0^3 m_e u_0^2} = \frac{\omega_p^2\,b\left(a - \dfrac{2}{3}b\right)}{\gamma_0^3 u_0^2},
```

the self-consistent continuum depression $\delta_{\text{th}} = \delta_{u,\text{DC}}$ iterates density enhancement ($\rho_{\text{dep}} \propto u_{\text{dep}}^{-1}$), the $u_{\text{dep}}^{-2}$ kinematic factor, and the depressed Lorentz factor $\gamma(u_{\text{dep}})^3$ to convergence:

```math
\delta \leftarrow \delta_0\,(1 - \delta)^{-3}\left(\frac{\gamma_0}{\gamma\bigl(u_0(1 - \delta)\bigr)}\right)^3, \qquad \rho_{\text{dep}} = \rho_0\,\frac{u_0}{u_{\text{dep}}}, \qquad \omega_{p,\text{dep}} = \omega_p\sqrt{\frac{u_0}{u_{\text{dep}}}}.
```

On a finite transverse grid of $N_x$ cells ($x_i = -a + i\Delta x$, $\Delta x = 2a/N_x$) with PEC walls at $x = \pm a$, solving the 1D discrete 3-point Poisson equation for the cloud-in-cell (CIC) deposited top-hat sheet charge and taking the particle-weighted mean potential followed by the same fixed-point iteration yields the discrete DC depression prediction $\delta_{\text{th,disc}}$ (`compute_discrete_dc_depression`). To compare both codes on the same physical equilibrium, we adopt **Option 3(ii)**: `LinearizedBeam` is initialized at the depressed continuum state $(u_{\text{dep}}, \rho_{\text{dep}}, \gamma_{\text{dep}})$, while the particle prediction $\beta_{q,\text{P,pred}}$ evaluates the analytical dispersion $\beta_{q,\text{th}}(\delta_{\text{meas}})$ directly at the interior velocity depression $\delta_{\text{meas}}$ measured in that same particle run over $z \in [0.2 L_z, 0.8 L_z]$. Pre-filling `solver._particles_q_bg` (Option 3(i)) is not used because `solver._particles_q_bg` only enters `gauss_residual()` and not Ampère's law $\partial_t \mathbf{E} = c^2\nabla\times\mathbf{B} - \mathbf{J}/\varepsilon_0$.

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
   - **Shared fit window (`fit_window`)**: Both solvers are fitted on the primary edge midpoints $z_k = (k + \tfrac{1}{2})\Delta z$ inside the identical interior window
     ```math
     z_{\min} = \frac{3}{T_1}, \qquad z_{\max} = (L_z - L_{\text{ramp}}) - \frac{3}{T_1}, \qquad z_{\max} - z_{\min} \ge 1.25\,\lambda_q,
     ```
     which excludes three transverse decay lengths $1/T_1$ of the evanescent cathode zone at $z = 0$ and three decay lengths upstream of the outflow taper at $L_z - L_{\text{ramp}}$.
   - **Fitted carrier $k_c$ and variable projection**: The carrier wavenumber $k_c$ is fitted simultaneously with $\beta_q$ rather than fixed to $\beta_e = \omega/u_{\text{dep}}$ because `LinearizedBeam`'s centered Eulerian advection shifts its numerical carrier by $\simeq \frac{1}{6}(\beta_e\Delta z)^2$, and fixing $k_c = \beta_e$ would bias $\beta_q$. For any trial pair $(k_c, \beta_q)$, the linear parameters $(a_f, a_s)$ follow from a 2-column complex linear least-squares solve (`numpy.linalg.lstsq`) on the design matrix $[\mathrm{e}^{-\mathrm{i} k_f z_k},\,\mathrm{e}^{-\mathrm{i} k_s z_k}]$. Minimizing the complex residual sum of squares $\text{RSS}(k_c, \beta_q) = \sum_k |I_1(z_k) - I_{1,\text{fit}}(z_k)|^2$ on a $201\times 401$ grid over $k_c \in k_{c,0}(1 \pm 0.03)$ (where $k_{c,0}$ is the unwrapped phase slope of $I_1(z)$) and $\beta_q \in [0.5, 1.8]\,\beta_{q,\text{guess}}$ followed by Nelder–Mead polishing (`xatol = 1e-10 * beta_q_guess`) yields $(k_c, \beta_q, A, B, a_f, a_s)$. The fit uncertainty $\sigma_{\beta_q}$ is obtained from the Gauss–Newton covariance $s^2(\mathbf{J}^{\mathsf{H}}\mathbf{J})^{-1}$ with $s^2 = \text{RSS}/(2N - 6)$ ($N$ complex points, $6$ real parameters).
   - **Amplitude sanity check vs report-only launch phase**: For a pure density modulation, $|a_f| = |a_s|$ and $B/A = \tan(\beta_q z_0)$ where $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$ is the effective launch offset induced by cathode image-charge screening and weak higher-order modes. Because no closed-form boundary-layer derivation predicts $z_0$ or $B/A$, $|a_s|/|a_f|$, $B/A$, and $z_0$ are reported as diagnostics without assertion, while $|A| > \frac{1}{2}m I_0$ is asserted to verify that the fit locked onto the driven wave rather than noise.
2. **Spatial wavenumber spectrum (`compute_spatial_spectrum`)**: The spatial Fourier transform over $z \in [0, \lambda_q]$,
   ```math
   \hat{I}_1(k_z) = \int_0^{\lambda_q} I_1(z, \omega)\,\mathrm{e}^{+\mathrm{i}k_z z}\,\mathrm{d}z,
   ```
   splits into a doublet of peaks at $k_f = \beta_e - \beta_q$ and $k_s = \beta_e + \beta_q$ separated by a deep null at $k_z = \beta_e$.

##### 5. Small-signal linearity condition, ballistic bunching, and $m$-scan verification

In the Lagrangian particle description (`wakis.particles`), the convective nonlinearities $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$ are retained automatically. For a pure entrance density modulation $I(0, t) = I_0(1 + m\sin\omega t)$ with $v_1(0, t) = 0$, the first-order relative density and velocity perturbations along the drift tunnel are

```math
\frac{\rho_1(z, t)}{\rho_{\text{dep}}} = m\cos(\beta_q z)\sin(\omega t - \beta_e z), \qquad \frac{v_1(z, t)}{u_{\text{dep}}} = -m\,\frac{\beta_q}{\beta_e}\sin(\beta_q z)\cos(\omega t - \beta_e z).
```

Along unperturbed trajectories, the velocity modulation produces an arrival-phase bunching parameter $X(z) \sim \frac{1}{2}m\beta_q z$ in the charge-conservation mapping $I(z, t)\,\mathrm{d}t = I(0, t_0)\,\mathrm{d}t_0$ [Webster, 1939; Paschke, 1957], generating a second-harmonic current ratio $|I_2(z)|/|I_1(0)| \sim X(z)$. In `test_m_scan_linearity`, running the deterministic `"sheet_lattice"` profile at $m$ and $m/2$ verifies that:
1. In `ParticleSolver`, the relative nonlinear amplitude shift in $\beta_q$ between $m$ and $m/2$ remains bounded by the discretization slack, $|\beta_q(m)/\beta_q(m/2) - 1| \le s_{\text{part}}$, while the raw and coherent-bunching second-harmonic ratios $|I_2|/|I_1|$ at $m$ and $m/2$ and their ratio are reported as diagnostics.
2. In `LinearizedBeam`, $m$-invariance of $\beta_q$ and $|I_2|/|I_1|$ is verified to round-off as an algebraic **identity** of the linearized fluid equations.

##### 6. Numerical setup, injection profiles, and derived tolerance formulas

To compare `ParticleSolver` and `LinearizedBeam` cleanly (`examples/_space_charge_waves_common.py`, `tests/test_087_particles_vs_linearized_beam.py`, `examples/024_particles_space_charge_waves.py`):

1. **Shared grid, integer-period time step, and DFT mean subtraction**: Both solvers run on the identical `GridFIT3D` ($x \in [-a, a]$ with `bc="pec"`, $y \in [-L_y/2, L_y/2]$ with `ny = 1` and `bc="periodic"`, $z \in [0, L_z]$ with `bc="pec"`). The time step is chosen as $\Delta t = 1/(f N_{\text{spp}})$ with $N_{\text{spp}} = \lceil 1/(f\Delta t_{\text{CFL}})\rceil$ ($\Delta t \le \Delta t_{\text{CFL}}$), so the 6-period DFT window $N_{\text{DFT}} = 6 N_{\text{spp}}$ spans an exact integer number of RF cycles. Per-edge mean subtraction over the $N_{\text{DFT}}$ window eliminates DC current leakage; the integer-$N_{\text{spp}}$ Dirichlet kernel leakage is bounded by the round-off **identity** $\varepsilon_{\text{DFT}} = 100\,\epsilon_{\text{mach}}/m$.
2. **Transverse sheet injection profiles (`"sheet_lattice"` vs `"sheet"`)**: Every time step, `BeamInjector.inject` (`wakis/particles/boundaries.py`) places $N_{\text{macro}}$ macroparticles on $z = 0$ with $y_p = 0$ exactly, $|x_p| \le b$, longitudinal momentum $u_z = \gamma_0 u_0$, $\mathbf{u}_\perp = \mathbf{0}$, and time-modulated weight $w(t^n) = |I(t^n)|\Delta t / (|q| N_{\text{macro}})$:
   - **Deterministic reference profile (`profile="sheet_lattice"`)**: Places one macroparticle at the midpoint of each equal-width transverse stratum, $x_k = b\bigl(2(k + \tfrac{1}{2})/N_{\text{macro}} - 1\bigr)$ ($k = 0, \dots, N_{\text{macro}} - 1$). This profile has zero transverse shot noise, is identical for all random seeds, and is used for every signed physics check, both particle mutations, the two-resolution check, and the $m$-scan.
   - **Seeded stratified profile (`profile="sheet"`)**: Draws one uniform sample per stratum per step, $x_k \in [-b + 2bk/N_{\text{macro}}, -b + 2b(k+1)/N_{\text{macro}})$, from `BeamInjector._step_rng(step_idx)`. Because `seed` only affects `profile="sheet"`, the 8-seed ensemble (`measure_empirical_noise`) calls `run_particle_beam_drift(..., profile="sheet", seed=s)`.
   - `test_injected_sheet_geometry` directly verifies these geometric invariants as an **identity (geometry) check** after 3 steps on CPU.
3. **Guide magnetic field**: A strong uniform axial guide field $B_{z,\text{ext}}$ ($\omega_c \gg \omega_p$) freezes out transverse space-charge expansion while leaving longitudinal motion along $\hat{\mathbf{z}}$ unperturbed [Birdsall & Langdon, 1991].
4. **Derived predictions and tolerances (`derived_tolerances`)**:
   - **Shared `SolverFIT3D` transverse operator (`eps_x_fit`) and multi-mode fit shift (`eps_fit`)**: From $R^2 = \tau^2/(\tau^2 + T_1^2)$, the 3-point transverse FIT Laplacian under-estimates $T_1^2$ by $-(T_1\Delta x)^2/12$, shifting $\beta_q$ upward in both codes by
     ```math
     \varepsilon_{x,\text{FIT}} = +\frac{1}{2}\bigl(1 - R^2\bigr)\frac{(T_1\Delta x)^2}{12}.
     ```
     Because the top-hat entrance profile excites weak higher-order even slab eigenmodes $\cos(T_n x)$ ($n \ge 2$), running `fit_beta_q_from_envelope` over $[z_{\min}, z_{\max}]$ on the multi-mode superposition $\sum_n w_n\cos(\beta_{q,n} z)\mathrm{e}^{-\mathrm{i}\beta_e z}$ gives a signed relative shift $\varepsilon_{\text{fit}} = \varepsilon_{\text{fit,cont}}$ when $w_n$ are evaluated with continuum overlaps and $\varepsilon_{\text{fit,grid}}$ when $w_n$ are evaluated with the discrete transverse CIC mode weights on the $N_x$-cell grid.
   - **`LinearizedBeam` deterministic prediction and slack**: On the discrete grid $(\Delta z, \Delta t)$, `LinearizedBeam`'s Crank–Nicolson axial advection coupled to the staggered Yee curl-curl symbol $\kappa(k) = \frac{2}{\Delta z}\sin(k\Delta z/2)$ obeys the exact 1D discrete dispersion relation for the fast ($k_-$) and slow ($k_+$) branches:
     ```math
     \frac{2}{\Delta t}\tan\!\left(\frac{\omega\Delta t}{2}\right) \mp \omega_q(k_\pm) = u_{\text{dep}}\,\frac{\sin(k_\pm\Delta z)}{\Delta z}, \qquad \omega_q(k) = R\bigl(\tau_{\text{Y}}(k)\bigr)\,\frac{\omega_{p,\text{dep}}}{\gamma_{\text{dep}}^{3/2}}, \qquad \tau_{\text{Y}}(k) = \sqrt{\kappa(k)^2 - \left(\frac{\omega}{c}\right)^2}.
     ```
     Solving both branches by `brentq` gives the predicted `LinearizedBeam` wavenumber and higher-order slack:
     ```math
     \beta_{q,\text{LB,pred}} = \frac{k_+ - k_-}{2}\,\bigl(1 + \varepsilon_{x,\text{FIT}}\bigr)\bigl(1 + \varepsilon_{\text{fit}}\bigr), \qquad \varepsilon_{\text{ax}} = \frac{k_+ - k_-}{2\beta_{q,\text{th}}} - 1,
     ```
     ```math
     s_{\text{LB}} = \bigl(|\varepsilon_{\text{ax}}| + |\varepsilon_{x,\text{FIT}}| + |\varepsilon_{\text{fit}}|\bigr)^2 + \bigl|\varepsilon_{\text{fit,grid}} - \varepsilon_{\text{fit,cont}}\bigr|, \qquad \left|\frac{\beta_{q,\text{LB}}}{\beta_{q,\text{LB,pred}}} - 1\right| \le s_{\text{LB}}.
     ```
   - **Particle deterministic prediction (`"sheet_lattice"`) and slack**: Lagrangian particles advect in continuous space without Eulerian carrier dispersion, so the staggered Yee axial curl-curl symbol acts on the physical carrier $\beta_e$, reducing $\tau = \beta_e/\gamma_{\text{dep}}$ via $\tau^2 = \beta_e^2 - (\omega/c)^2$ ($\mathrm{d}\ln\tau / \mathrm{d}\ln\beta_e = \gamma_{\text{dep}}^2$) and shifting $R(\tau)$ downward by
     ```math
     \varepsilon'_{z,\text{FIT}} = \frac{\mathrm{d}\ln R}{\mathrm{d}\ln\tau}\,\gamma_{\text{dep}}^2\,\frac{(\beta_e\Delta z)^2}{24},
     ```
     while nodal transverse CIC charge/current assignment and field gather attenuate $T_1$ by
     ```math
     \varepsilon_{\text{CIC}} = 1 - \sqrt{1 - \frac{2}{3}\sin^2\!\left(\frac{T_1\Delta x}{2}\right)}.
     ```
     Evaluating $\beta_{q,\text{th}}(\delta_{\text{meas}})$ at the measured interior DC depression $\delta_{\text{meas}}$ of that same run gives the deterministic particle prediction and slack:
     ```math
     \beta_{q,\text{P,pred}} = \beta_{q,\text{th}}(\delta_{\text{meas}})\,\bigl(1 - \varepsilon_{\text{CIC}}\bigr)\bigl(1 - \varepsilon'_{z,\text{FIT}}\bigr)\bigl(1 + \varepsilon_{x,\text{FIT}}\bigr)\bigl(1 + \varepsilon_{\text{fit}}\bigr),
     ```
     ```math
     s_{\text{part}} = \bigl(|\varepsilon_{\text{CIC}}| + |\varepsilon'_{z,\text{FIT}}| + |\varepsilon_{x,\text{FIT}}| + |\varepsilon_{\text{fit}}|\bigr)^2 + \bigl|\varepsilon_{\text{fit,grid}} - \varepsilon_{\text{fit,cont}}\bigr|, \qquad \left|\frac{\beta_{q,\text{P,lattice}}}{\beta_{q,\text{P,pred}}} - 1\right| \le s_{\text{part}}.
     ```
   - **Particle 8-seed `"sheet"` ensemble**: Over the $N_{\text{seeds}} = 8$ seeded `"sheet"` runs with sample mean $\bar{\beta}_q$, standard error $\text{SEM} = s/\sqrt{8}$ (`ddof=1`), and two-sided $99.73\%$ Student-$t$ critical value $t = t_{0.99865,\,7}$:
     ```math
     \left|\frac{\bar{\beta}_q}{\beta_{q,\text{P,pred}}} - 1\right| \le s_{\text{part}} + \frac{t\,\text{SEM}}{\beta_{q,\text{P,pred}}}, \qquad \bigl|\bar{\beta}_q - \beta_{q,\text{P,lattice}}\bigr| \le t\,\text{SEM}.
     ```
   - **Cross-code agreement (`"sheet_lattice"`)**:
     ```math
     \left|\frac{\beta_{q,\text{P,lattice}}}{\beta_{q,\text{LB}}} - \frac{\beta_{q,\text{P,pred}}}{\beta_{q,\text{LB,pred}}}\right| \le s_{\text{LB}} + s_{\text{part}}.
     ```
   - **DC velocity depression (`"sheet"` ensemble vs discrete Poisson)**:
     ```math
     \bigl|\bar{\delta}_{\text{meas}} - \delta_{\text{th,disc}}\bigr| \le t\,\text{SEM}(\delta_{\text{meas}}) + 10^{-12}.
     ```
   - **Entrance current amplitude at $z_0 = \Delta z / 2$**: Because CIC deposition is exactly charge-conserving across the beam cross-section, the entrance current amplitude at the first primary edge midpoint $z_0 = \Delta z / 2$ is bounded by the standing-wave envelope variation over half a cell:
     ```math
     \text{tol}_{\text{amp,LB}} = \frac{1}{2}(\beta_q\Delta z)^2, \qquad \text{tol}_{\text{amp,part}} = \text{tol}_{\text{amp,LB}} + 10^{-12},
     ```
     plus $t\,\text{SEM}(|I_1(z_0)|)/(m I_0)$ for the 8-seed `"sheet"` ensemble.
   - To inspect the printed `*CHECK` metrics and error budgets at run time:
     ```bash
     pytest tests/test_087_particles_vs_linearized_beam.py -v -s
     python examples/024_particles_space_charge_waves.py --smoke
     ```

##### 7. Limits & Failure Modes

- **Small-signal linear regime only ($m \ll 1$, $X(z) \ll 1$)**: `LinearizedBeam` linearizes the cold-fluid equations around a time-independent equilibrium and omits the quadratic convective terms $v_1\partial_z v_1$ and $\partial_z(\rho_1 v_1)$. When the modulation index $m$ or drift length $z$ is increased until the bunching parameter $X \sim \mathcal{O}(1)$, fast electrons overtake slow electrons (wave breaking / multi-stream trajectory crossing), generating strong higher harmonics ($2\omega, 3\omega$) and electrostatic trapping in `ParticleSolver` that `LinearizedBeam` cannot represent.
- **Planar sheet-beam geometry and strong axial magnetic focusing only**: The analytical reduction factor `reduction_factor_planar` and the `"sheet"` / `"sheet_lattice"` injector profiles assume a 1D planar sheet beam uniform along $y$ (`ny = 1`, `bc="periodic"`) and confined by an axial guide field $B_{z,\text{ext}}$ strong enough ($\omega_c \gg \omega_p$) to freeze out transverse velocity perturbations $\mathbf{v}_\perp$. At weak or zero guide field, transverse space-charge forces cause beam spreading or Brillouin scalloping; 3D pencil or rectangular beams require `reduction_factor_cylindrical` or `reduction_factor_rectangular` [Branch & Mihran, 1955].
- **Effective launch offset $z_0$ and sine-quadrature amplitude $B$ are not predicted analytically**: Near the grounded PEC cathode plane $z = 0$, image-charge screening of $E_x$ over the evanescent zone $\ell_{\text{cath}} \sim 1/T_1$ and multi-mode excitation produce a small launch offset $z_0 = \arctan(\operatorname{Re}(B/A))/\beta_q$. Because no closed-form cathode boundary-layer derivation exists for $z_0$ or $B/A$, these quantities are reported as diagnostics only and are not asserted.
- **Two-wave fit window requires $z_{\max} - z_{\min} \ge 1.25\,\lambda_q$**: Separating two closely spaced complex exponentials $k_f = k_c - \beta_q$ and $k_s = k_c + \beta_q$ alongside a free carrier $k_c$ requires the interior fit window $[3/T_1, L_z - L_{\text{ramp}} - 3/T_1]$ to span at least $1.25\,\lambda_q$ (`test_preflight_and_resolution_criteria`). On shorter tunnels where $z_{\max} - z_{\min} \lesssim \lambda_q$, the two wave columns become nearly collinear and the variable-projection estimate of $(k_c, \beta_q)$ becomes ill-conditioned.
- **Transverse injection profile must match between codes (mutation M3)**: `LinearizedBeam` with `half_thickness=b` represents a uniform top-hat planar slab across $x \in [-b, b]$. If the particle injector instead samples a circular disk profile (such as the Fibonacci disk `"lattice"` profile tested in mutation **M3**, `test_mutation_disk_profile_instead_of_sheet_fails`), its semicircular $x$-marginal $\propto \sqrt{1 - (x/b)^2}$ concentrates charge near $x = 0$, increasing the effective plasma reduction factor $R$ and shifting $\beta_{q,\text{P}}$ significantly above the top-hat sheet prediction $\beta_{q,\text{P,pred}}$ until Step 4.2 fails.

References:
- S. Ramo, "Space Charge and Field Waves in an Electron Beam", *Phys. Rev.* **56**(3), 276–283 (1939), [doi:10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276).
- W. C. Hahn, "Small Signal Theory of Velocity-Modulated Electron Beams", *Gen. Elec. Rev.* **42**, 258–270 (1939).
- D. L. Webster, "Cathode-Ray Bunching", *J. Appl. Phys.* **10**(7), 501–508 (1939), [doi:10.1063/1.1707335](https://doi.org/10.1063/1.1707335).
- G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in Electron Beams", *IRE Trans. Electron Devices* **2**(2), 3–11 (1955), [doi:10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065).
- F. Paschke, "On the Nonlinear Behavior of Electron-Beam Devices", *RCA Rev.* **18**(2), 221–242 (1957).
- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, Taylor & Francis / IOP Publishing, Bristol (1991), ISBN 978-0-7503-0117-6.
- T. Zh. Esirkepov, "Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor", *Comput. Phys. Commun.* **135**(2), 144–153 (2001), [doi:10.1016/S0010-4655(00)00228-9](https://doi.org/10.1016/S0010-4655(00)00228-9).
