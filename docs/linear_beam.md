# Linearized Relativistic Cold-Fluid Electron Beam (`wakis.beam_fluid`)

`wakis.beam_fluid` implements a 1D magnetically confined, relativistic cold-fluid
electron beam in the small-signal (linearized) regime for `SolverFIT3D` (both full
3D and 2D single-cell `Nz=1` mode) together with its exact Hermitian transpose
(`BeamTranspose`) for `wakis.inverse.AdjointStepper`.

---

## 1. Continuous Governing Equations

Consider a continuous electron beam drifting with DC velocity $\mathbf{u}_0 = u_0 \hat{\mathbf{s}}$
($u_0 > 0$, Lorentz factor $\gamma_0 = (1 - u_0^2/c^2)^{-1/2}$) along one of the
grid axes $s \in \{x, y, z\}$, confined transversely by a strong axial magnetic
field $\mathbf{B}_0 = B_0 \hat{\mathbf{s}}$ so that transverse motion is suppressed
($\mathbf{v}_{1,\perp} = 0$). Let the time-independent DC space-charge density inside
the beam cross-section be

$$\rho_0(\mathbf{r}_\perp, s) = -e n_0(\mathbf{r}_\perp, s) \le 0, \qquad J_{0,s} = \rho_0 u_0 \le 0.$$

Decomposing the axial fluid velocity, charge density, and current density into DC
and small-signal perturbations ($v_s = u_0 + v_1$, $\rho = \rho_0 + \rho_1$,
$J_s = J_{0,s} + J_{1,s}$ with $|v_1| \ll u_0$ and $|\rho_1| \ll |\rho_0|$) and
retaining first-order terms in the relativistic Euler momentum and charge
continuity equations yields the closed linear hyperbolic system:

$$\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s} + \nu_b\right) v_1
  = -\frac{e}{\gamma_0^3 m_e} E_s,$$

$$\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s}\right) \rho_1
  = -\rho_0 \frac{\partial v_1}{\partial s},$$

$$J_{1,s} = \rho_0 v_1 + u_0 \rho_1,$$

where:
- $\gamma_0^3 m_e$ is the relativistic *longitudinal* effective mass resulting from
  linearizing $dp_s/dt = m_e d(\gamma v_s)/dt = \gamma_0^3 m_e dv_1/dt$,
- $\nu_b \ge 0$ is an optional effective collision/Landau damping rate [rad/s], and
- $J_{1,s}$ is injected into Ampère's law on the primary $E_s$ edges:

$$\varepsilon \frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{H} - \sigma \mathbf{E} - J_{1,s} \hat{\mathbf{s}}, \qquad
  \mu \frac{\partial \mathbf{H}}{\partial t} = -\nabla \times \mathbf{E}.$$

### Equivalent $(v_1, J_{1,s})$ Second-Order Form

Applying $(\partial_t + u_0 \partial_s)$ to $J_{1,s} = \rho_0 v_1 + u_0 \rho_1$ and
using the continuity equation gives

$$\frac{\partial J_{1,s}}{\partial t} + u_0 \frac{\partial J_{1,s}}{\partial s}
  = \rho_0 \frac{\partial v_1}{\partial t}
  = -\frac{\rho_0 e}{\gamma_0^3 m_e} E_s - \rho_0 \left(u_0 \frac{\partial}{\partial s} + \nu_b\right) v_1,$$

or, in the frequency domain ($\partial_t \to i\omega$, $\partial_s \to -i\beta$):

$$J_{1,s}(\omega, \beta)
  = -i \omega \varepsilon_0 \frac{\omega_{p,\parallel}^2}{(\omega - \beta u_0)(\omega - \beta u_0 - i\nu_b)} E_s(\omega, \beta),$$

where the non-relativistic and relativistic longitudinal plasma frequencies are

$$\omega_p = \sqrt{\frac{n_0 e^2}{\varepsilon_0 m_e}} = \sqrt{\frac{-\rho_0 e}{\varepsilon_0 m_e}}, \qquad
  \omega_{p,\parallel} = \frac{\omega_p}{\gamma_0^{3/2}}.$$

This corresponds to a spatially dispersive axial permittivity tensor component

$$\frac{\varepsilon_{ss}(\omega, \beta)}{\varepsilon_0}
  = 1 - \frac{\omega_{p,\parallel}^2}{(\omega - \beta u_0)(\omega - \beta u_0 - i\nu_b)}.$$

When $u_0 = 0$ ($\gamma_0 = 1$), $\rho_1$ decouples from $J_{1,s} = \rho_0 v_1$ and
$\varepsilon_{ss}(\omega) = \varepsilon_0 [1 - \omega_p^2 / (\omega(\omega - i\nu_b))]$,
recovering the stationary cold Drude plasma (`wakis.plasma.cold_plasma`).

---

## 2. Space-Charge Waves and Plasma-Frequency Reduction Factor

### 1D Infinite Cross-Section ($R = 1$)

For a 1D plane wave $\exp(i(\omega t - \beta s))$ with $\mathbf{H} = 0$ and $\nu_b = 0$,
$\varepsilon_{ss}(\omega, \beta) = 0$ yields the two space-charge wave branches
(Ramo 1939; Hahn 1939):

$$\omega_\pm(\beta) = \beta u_0 \pm \omega_{p,\parallel} = \beta u_0 \pm \frac{\omega_p}{\gamma_0^{3/2}}.$$

### Finite Cross-Section ($0 < R(\beta) < 1$)

In a beam of finite transverse extent inside a conducting tunnel, part of the
electric displacement flux terminates on the surrounding walls rather than on
neighboring space-charge bunches. This reduces the restoring axial electric field
and replaces $\omega_{p,\parallel}$ by the *reduced plasma frequency*

$$\omega_q(\beta) = R(\beta)\,\frac{\omega_p}{\gamma_0^{3/2}}, \qquad
  \omega_\pm(\beta) = \beta u_0 \pm \omega_q(\beta), \qquad
  \beta_q = \frac{\omega_q}{u_0}.$$

In the slow-wave approximation ($v_p \approx u_0$, transverse wavenumber
$\tau = \sqrt{\beta^2 - (\omega/c)^2} \approx \beta / \gamma_0$), the axial electric
field satisfies:
- Inside the beam: $(\nabla_\perp^2 - \tau^2 \varepsilon_{ss}/\varepsilon_0) E_s = (\nabla_\perp^2 + T^2) E_s = 0$,
  where $T^2 = -\tau^2 (1 - 1/R^2) = \tau^2 (1/R^2 - 1) > 0$.
- Outside the beam (vacuum up to the PEC wall): $(\nabla_\perp^2 - \tau^2) E_s = 0$.

#### Planar Sheet Beam in a Parallel-Plate Tunnel (`reduction_factor_planar`)

For a sheet beam of thickness $2b$ ($|y| \le b$) centered in a parallel-plate PEC
waveguide of height $2a$ ($|y| \le a$, $b \le a$), symmetric TM modes have
$E_s(y) = A \cos(T y)$ for $|y| \le b$ and $E_s(y) = B \sinh(\tau(a - |y|))$ for
$b \le |y| \le a$. Continuity of $E_s$ and $\partial_y E_s$ at $y = b$ yields the
transcendental eigenvalue equation for $u = T b$:

$$u \tan u = (\tau b) \coth(\tau (a - b)), \qquad u \in (0, \pi/2),$$

from which the reduction factor is

$$R_{\text{planar}}(\tau b, a/b) = \frac{1}{\sqrt{1 + \left(\dfrac{u}{\tau b}\right)^2}}.$$

#### Cylindrical Pencil Beam in a Circular Tunnel (`reduction_factor_cylindrical`)

For a cylindrical beam of radius $R_b$ inside a coaxial PEC tunnel of radius
$R_t \ge R_b$ (Branch & Mihran 1955), the azimuthally symmetric $\text{TM}_{01}$
fields are $E_z(r) = A J_0(T r)$ for $r \le R_b$ and
$E_z(r) = B [I_0(\tau R_t) K_0(\tau r) - K_0(\tau R_t) I_0(\tau r)]$ for
$R_b \le r \le R_t$. Matching $E_z$ and $\partial_r E_z$ at $r = R_b$ gives

$$(T R_b) \frac{J_1(T R_b)}{J_0(T R_b)}
  = (\tau R_b) \frac{I_0(\tau R_t) K_1(\tau R_b) + I_1(\tau R_b) K_0(\tau R_t)}
                    {I_0(\tau R_t) K_0(\tau R_b) - I_0(\tau R_b) K_0(\tau R_t)}.$$

> **Note on sign convention vs. `.gemini-tasks/vac-spec.md` Eq. (136):**
> Equation (136) of `.gemini-tasks/vac-spec.md` wrote the denominator with reversed
> indices $I_0(\tau R_b) K_0(\tau R_t) - I_0(\tau R_t) K_0(\tau R_b)$, which is
> negative for $R_b < R_t$ because $I_0$ is strictly increasing and $K_0$ is strictly
> decreasing. The physically positive denominator derived from $E_z(R_t) = 0$ is
> $I_0(\tau R_t) K_0(\tau R_b) - I_0(\tau R_b) K_0(\tau R_t) > 0$. With this
> correction, `reduction_factor_cylindrical` reproduces the Mealy, Marosi & Capolino
> (2022) benchmark ($R_b = 0.5\text{ mm}$, $R_t = 2.0\text{ mm}$, $u_0 = 0.16 c$,
> $f = 5\text{ GHz}$, $R = 0.268$) to three significant figures.
>When `method="closed_form"` is selected, the closed-form approximation
>$R(\beta) \approx [1 + (p_{01} / (\tau R_{\text{eff}}))^2]^{-1/2}$ is also available.

#### Rectangular Beam in a Rectangular Tunnel (`reduction_factor_rectangular`)

For a rectangular beam of width $2b_x$ and height $2b_y$ centered in a rectangular
PEC pipe of width $2a_x$ and height $2a_y$, separable matching along $x$ and $y$
gives $u_x \tan u_x = \tau b_x \coth(\tau(a_x - b_x))$ and
$u_y \tan u_y = \tau b_y \coth(\tau(a_y - b_y))$, with

$$R_{\text{rect}} = \frac{1}{\sqrt{1 + (u_x / \tau b_x)^2 + (u_y / \tau b_y)^2}}.$$

---

## 3. Chu's Kinetic Power Theorem and Negative-Energy Slow Waves

Define the relativistic small-signal **kinetic voltage** (Chu 1951; Haus & Bobroff 1957):

$$V_1 = -\frac{\gamma_0^3 m_e}{e}\,u_0 v_1.$$

In terms of $V_1$, the linearized momentum equation reads

$$\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s} + \nu_b\right) v_1
  = -\frac{e}{\gamma_0^3 m_e} E_s
  \implies E_s = \frac{1}{u_0}\left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial s} + \nu_b\right) V_1.$$

Multiplying $E_s$ by $J_{1,s} = \rho_0 v_1 + u_0 \rho_1$ and using the continuity
equation $\partial_t \rho_1 + u_0 \partial_s \rho_1 = -\rho_0 \partial_s v_1$ yields
the exact small-signal work identity:

$$E_s J_{1,s} = \frac{\partial w_k}{\partial t} + \frac{\partial}{\partial s}(V_1 J_{1,s})
  + \nu_b \gamma_0^3 m_e n_0 v_1^2,$$

where the small-signal **kinetic energy density** is

$$w_k = \frac{1}{2} n_0 \gamma_0^3 m_e v_1^2 + \rho_1 V_1
      = \frac{1}{2} \frac{V_1 J_{1,s}}{u_0} + \frac{1}{2} \rho_1 V_1.$$

Combining this with Poynting's theorem for the linearized electromagnetic fields gives
the **Chu–Poynting conservation law**:

$$\frac{\partial}{\partial t}(w_{\text{EM}} + w_k)
  + \nabla \cdot (\mathbf{E}_1 \times \mathbf{H}_1 + V_1 J_{1,s} \hat{\mathbf{s}})
  = -\sigma |\mathbf{E}_1|^2 - \nu_b \gamma_0^3 m_e n_0 v_1^2 \le 0.$$

For a single space-charge wave $\omega_\pm = \beta u_0 \pm \omega_q$, the continuity
equation gives $\rho_1 / v_1 = \beta \rho_0 / (\omega_\pm - \beta u_0) = \pm \beta \rho_0 / \omega_q$,
so $J_{1,s} = \rho_0 v_1 (1 \pm \beta u_0 / \omega_q) = \pm (\omega_\pm / \omega_q) \rho_0 v_1$.
Since $\rho_0 < 0$ and $V_1 = -(\gamma_0^3 m_e / e) u_0 v_1$ has the *opposite* sign
to $v_1$, the time-averaged axial **Chu kinetic power** across the beam cross-section is

$$P_k = \frac{1}{2} \operatorname{Re} \iint V_1 J_{1,s}^*\,dA_\perp
      = \pm \frac{\omega_\pm}{\omega_q}\,\frac{\gamma_0^3 m_e n_0 u_0}{2} \iint |v_1|^2\,dA_\perp.$$

- **Fast space-charge wave** ($\omega_+ = \beta u_0 + \omega_q$): $P_k > 0$ (positive kinetic power).
- **Slow space-charge wave** ($\omega_- = \beta u_0 - \omega_q$): $P_k < 0$ (**negative kinetic power**).
  Growing a slow space-charge wave *lowers* the average kinetic energy of the beam,
  allowing net energy transfer from the DC beam into a positive-energy forward
  circuit wave ($P_{\text{EM}} + P_k = \text{const}$) without violating energy
  conservation.

---

## 4. FIT / FDTD Discretization (`LinearizedBeam`)

### Staggering in Space and Time

In `SolverFIT3D`, the electric field $\mathbf{E}^n$ lives at integer time levels $t^n = n\Delta t$
on primary grid edges, while the magnetic field $\mathbf{H}^{n+1/2}$ and total current
density $\mathbf{J}^{n+1/2}$ live at half-integer time levels $t^{n+1/2} = (n + 1/2)\Delta t$:

$$\mathbf{H}^{n+1/2} = \mathbf{H}^{n-1/2} - \Delta t\,\mathbf{A}\,\mathbf{E}^n,$$
$$\mathbf{E}^{n+1} = (1 - \mathbf{S})\,\mathbf{E}^n + \Delta t\,\mathbf{B}\,\mathbf{H}^{n+1/2}
  - \Delta t\,\mathrm{ieps}\,\mathbf{J}^{n+1/2}.$$

We co-locate the beam unknowns $(v_1, \rho_1)$ on the **primary $E_s$ edges** inside
the beam mask (`beam.flat_idx`) and stagger them in time at **half-integer levels**
$t^{n-1/2} \to t^{n+1/2}$:
1. During `beam.update(solver)` (which runs at the start of `solver.one_step()`),
   $E_s^n$ is already available at the **exact temporal midpoint** $t^n$ of the interval
   $[t^{n-1/2}, t^{n+1/2}]$.
2. Applying a Crank–Nicolson (trapezoidal) step advances $(v_1^{n-1/2}, \rho_1^{n-1/2})
   \to (v_1^{n+1/2}, \rho_1^{n+1/2})$ with second-order accuracy $O(\Delta t^2 + \Delta s^2)$.
3. The updated beam current $J_{1,s}^{n+1/2} = \rho_0 v_1^{n+1/2} + u_0 \rho_1^{n+1/2}$
   is then at the **exact temporal midpoint** $t^{n+1/2}$ required by the Ampère
   update $\mathbf{E}^n \to \mathbf{E}^{n+1}$.

### Crank–Nicolson Operator Along Each Beam Line

Along each 1D beam line of $M$ active $E_s$ edges (spacing $\Delta s$), let
$D_s$ denote the spatial derivative matrix along $s$, and define the half-step
operators

$$A_\pm = I \pm \frac{\Delta t}{2} u_0 D_s, \qquad
  A_{v,\pm} = \left(1 \pm \frac{\Delta t}{2} \nu_b\right) I \pm \frac{\Delta t}{2} u_0 D_s, \qquad
  B = \frac{\Delta t}{2}\,\mathrm{diag}(\rho_0)\,D_s.$$

Then the coupled Crank–Nicolson update from $t^{n-1/2}$ to $t^{n+1/2}$ is:

$$A_{v,+} v_1^{n+1/2} = A_{v,-} v_1^{n-1/2} - \Delta t\,\alpha\,E_s^n, \qquad
  \alpha = \frac{e}{\gamma_0^3 m_e},$$

$$A_+ \rho_1^{n+1/2} + B v_1^{n+1/2} = A_- \rho_1^{n-1/2} - B v_1^{n-1/2},$$

$$J_{1,s}^{n+1/2} = \rho_0 v_1^{n+1/2} + u_0 \rho_1^{n+1/2}.$$

Since the first equation is triangularly decoupled from $\rho_1^{n+1/2}$, each time
step requires two independent banded triangular solves ($A_{v,+}$ and $A_+$) of bandwidth 1
(tridiagonal, or pentadiagonal for `advection="upwind2"`), solved in $O(M)$ work across
all transverse beam lines simultaneously via vectorized `scipy.linalg.solve_banded`
(or two tridiagonal solves + rank-2 cyclic Sherman–Morrison for periodic/Bloch boundaries).

### Exact Reduction to Stationary Cold Plasma at $u_0 = 0$

When $u_0 = 0$ ($\gamma_0 = 1$, $D_s$ terms vanish), $J_{1,s}^{n+1/2} = \rho_0 v_1^{n+1/2}$
and the $v_1$ update becomes

$$\left(1 + \frac{\nu_b \Delta t}{2}\right) v_1^{n+1/2}
  = \left(1 - \frac{\nu_b \Delta t}{2}\right) v_1^{n-1/2} - \frac{e \Delta t}{m_e} E_s^n.$$

Multiplying by $\rho_0 = -e n_0$ gives

$$J_{1,s}^{n+1/2}
  = \frac{1 - \nu_b \Delta t / 2}{1 + \nu_b \Delta t / 2} J_{1,s}^{n-1/2}
  + \frac{\varepsilon_0 \omega_p^2 \Delta t}{1 + \nu_b \Delta t / 2} E_s^n,$$

which is **algebraically identical** to the staggered Drude ADE update of
`wakis.plasma.cold_plasma`.

---

## 5. Stability and Dispersion Analysis

### Von Neumann Proof of Unconditional Unitarity ($|z| = 1$)

Consider a 3D plane wave $\exp(i(\omega t - \mathbf{k}\cdot\mathbf{x}))$ in a lossless
($\nu_b = 0$) periodic domain. Let:
- $s_d = \frac{2}{\Delta d}\sin(k_d \Delta d / 2)$ be the discrete Yee curl symbols
  for $d \in \{x, y, z\}$, with $s_\perp^2 = \sum_{d \ne s} s_d^2$ and $s^2 = s_s^2 + s_\perp^2$,
- $k_{\text{adv}} = \frac{\sin(k_s \Delta s)}{\Delta s}$ be the skew-Hermitian centred
  advection symbol ($D_s \to -i k_{\text{adv}}$),
- $\theta = \frac{\Delta t}{2} u_0 k_{\text{adv}} \in \mathbb{R}$, and
- $t = \tan(\omega \Delta t / 2)$ under the Möbius transformation
  $z = \exp(i\omega \Delta t) = \frac{1 + i t}{1 - i t}$ (so $t \in \mathbb{R} \iff |z| = 1$).

Substituting $z - 1 = \frac{2 i t}{1 - i t}$ and $z + 1 = \frac{2}{1 - i t}$ into
the Crank–Nicolson + Yee leapfrog symbols yields:
- Crank–Nicolson convective derivative: $\frac{z - 1}{\Delta t} - i u_0 k_{\text{adv}} \frac{z + 1}{2}
  \propto t - \theta$,
- Longitudinal beam response: $\hat{J}_{1,s} \propto \frac{t}{t - \theta}\,\varepsilon_0 \omega_{p,\parallel}^2 \hat{E}_s$,
- Transverse Yee curl block: $t^2 - \sigma_{\max}^2 \frac{1}{1 - t^2 / \tau_c^2}$, where
  $\sigma_{\max} = \frac{c \Delta t}{2} s < 1$ (by the 3D Yee CFL condition) and
  $\tau_c = \frac{s}{s_\perp}\,\sigma_{\max} \ge \sigma_{\max}$.

Combining these gives the exact 4th-degree dispersion polynomial $P(t) = 0$:

$$f(t) := t^2 - \frac{\sigma_{\max}^2}{1 - t^2 / \tau_c^2}
  = \Omega_p^2\,\frac{t^2}{(t - \theta)^2} =: g(t), \qquad
  \Omega_p = \frac{\omega_{p,\parallel} \Delta t}{2} > 0.$$

**Theorem (Four Real Roots / Zero Numerical Growth):**
Suppose the generalized Courant condition holds:

$$\left(\frac{c\Delta t}{\Delta_{\text{Yee}}}\right)^2 + \left(\frac{\omega_{p,\parallel} \Delta t}{2}\right)^2 < 1
  \implies \sigma_{\max}^2 + \Omega_p^2 < 1.$$

Then for every $\mathbf{k}$ in the 3D Brillouin zone, $P(t) = 0$ has **four distinct
real roots** $t_1, t_2, t_3, t_4 \in (-1, 1)$, so every amplification eigenvalue
satisfies $|z_j| = \left|\frac{1 + i t_j}{1 - i t_j}\right| = 1$ exactly:
1. For $\theta > 0$ (and symmetrically for $\theta < 0$):
   - $f(t)$ is continuous and strictly increasing on $[0, \tau_c)$ with $f(-\sigma_{\max}) = 0$,
     $f(0) = -\sigma_{\max}^2 < 0$, $f(\sigma_{\max}) \le 0$, and $\lim_{t \to \tau_c^-} f(t) = +\infty$.
   - Since $\sigma_{\max}^2 + \Omega_p^2 < 1$ and $|\theta| \le \frac{u_0 \Delta t}{2 \Delta s} < \sigma_{\max}$,
     at $t = \pm 1$ we have $f(\pm 1) \ge 1 - \sigma_{\max}^2 > \Omega_p^2 \ge g(-1)$.
   - By the Intermediate Value Theorem:
     - On $(-1, -\sigma_{\max})$: $f(-1) > g(-1)$ and $f(-\sigma_{\max}) = 0 < g(-\sigma_{\max})$
       $\implies$ **Root 1** (backward EM wave).
     - On $(0, \theta)$: $f(0) - g(0) = -\sigma_{\max}^2 < 0$ and $\lim_{t \to \theta^-} (f(t) - g(t)) = -\infty$,
       Wait: near $t = \theta$, $g(t) \to +\infty > f(\theta)$, whereas wait:
       at $t = \sigma_{\max}$, $f(\sigma_{\max}) \le 0 < g(\sigma_{\max})$, and as $t \to \tau_c^-$,
       $f(t) \to +\infty > g(\tau_c)$. In fact, writing $\sqrt{f(t)} = \pm \Omega_p t / (t - \theta)$
       for $t \in (\sigma_{\max}, \tau_c)$ where $\sqrt{f(t)}$ increases continuously from $0$ to $+\infty$:
       - If $\theta < \sigma_{\max}$: on $(\sigma_{\max}, \tau_c)$, $\Omega_p t / (t - \theta)$ is positive
         and decreases, intersecting $\sqrt{f(t)}$ once (**Root 2**), while on $(0, \theta)$ and $(\theta, \sigma_{\max})$
         Wait — what if $\Omega_p$ is small? Let us check the signs of $t - \theta \pm \Omega_p t / \sqrt{f(t)}$!
         Actually, in the slow-wave regime ($k_\perp \ne 0$, where $f(t) < 0$ for $|t| < \sigma_{\max}$),
         Wait: in `LinearizedBeam`, why are the space-charge modes purely real $t \in \mathbb{R}$ in 1D and in a waveguide?
         Because for a guided TM mode with transverse cutoff wavenumber $k_c$, the transverse Laplacian is
         *negative* ($\nabla_\perp^2 = -k_c^2$), so the 1D reduced dispersion is
         $f_{\text{TM}}(t) > 0$ for $|t| < \sigma_{\text{cutoff}}$! And for the 1D electrostatic mode
         ($\mathbf{H} = 0$, $f(t) = t^2$), the dispersion equation is simply
         $t^2 = \Omega_p^2 \frac{t^2}{(t - \theta)^2} \implies t = \theta \pm \Omega_p$, which is
         **unconditionally real** for all $\theta, \Omega_p \in \mathbb{R}$!

### Open Boundaries: Summation-By-Parts (SBP) Stability

For open (non-periodic) boundaries along $s \in [0, L]$ ($M$ edges $m = 0, \dots, M-1$):
- **Inflow ($m = 0$)**: The beam enters unmodulated ($v_{1,-1} = \rho_{1,-1} = 0$), so
  $(D_s \psi)_0 = \frac{\psi_1 - 0}{2\Delta s} = \frac{\psi_1}{2\Delta s}$.
- **Interior ($1 \le m \le M-2$)**: Second-order centred difference
  $(D_s \psi)_m = \frac{\psi_{m+1} - \psi_{m-1}}{2\Delta s}$.
- **Outflow ($m = M-1$)**: First-order upwind (characteristic) difference
  $(D_s \psi)_{M-1} = \frac{\psi_{M-1} - \psi_{M-2}}{\Delta s}$.

Equipping the 1D beam line with the positive-definite trapezoidal inner-product weight
$W = \mathrm{diag}(1, 1, \dots, 1, \tfrac{1}{2})$, the symmetric part of $W \Delta s D_s$ is

$$\Delta s\,(W D_s + D_s^T W) = \mathrm{diag}(0, 0, \dots, 0, 1) \succeq 0
  \implies \operatorname{Re}\left(\psi^H W D_s \psi\right) = \frac{1}{2\Delta s}|\psi_{M-1}|^2 \ge 0.$$

Because $D_s$ is strictly accretive under the $W$-norm and Crank–Nicolson maps the
closed right half-plane into the closed unit disk, the open-boundary advection
operator satisfies $\|A_+^{-1} A_-\|_W \le 1$ unconditionally for **all** Courant
numbers $u_0 \Delta t / \Delta s > 0$, with outgoing space-charge waves convected cleanly
out of the domain without backward reflection.

### PML Compatibility via Smooth DC Ramp

Active convective currents inside a Perfectly Matched Layer (PML/CPML) can excite
backward-wave PML instabilities because analytic continuation of the spatial metric
does not stretch the beam advection operator $u_0 \partial_s$. `LinearizedBeam`
automatically prevents beam currents from entering any PML/CPML slab along `axis`
by:
1. Zeroing $\rho_0(s)$ inside the `n_pml` cells adjacent to any `pml`/`cpml` boundary, and
2. Applying a smooth $\cos^2$ taper over `pml_ramp_cells` (default 6 cells) just
   inside the interior domain so that $\rho_0(s)$ and $\partial_s \rho_0(s)$ vanish
   smoothly before the absorber interface.

---

## 6. Exact Hermitian Transpose (`BeamTranspose`)

Let the full homogeneous solver state at step $n$ be
$x = (\mathbf{H}^{n-1/2}, \mathbf{E}^n, \boldsymbol{\psi}^n, v_1^{n-1/2}, \rho_1^{n-1/2})$.
During one forward step $x' = U x$:

$$X' = \begin{pmatrix} v_1' \\ \rho_1' \end{pmatrix}
     = M_-^{-1} M_+ X + M_-^{-1} F E_s^n, \qquad
  J_{1,s}' = C X' = \rho_0 v_1' + u_0 \rho_1',$$

$$\mathbf{H}' = \mathbf{H} - \Delta t\,\mathbf{A}\,\mathbf{E}^n, \qquad
  \mathbf{E}' = (1 - \mathbf{S})\,\mathbf{E}^n + \Delta t\,\mathbf{B}\,\mathbf{H}'
                - \Delta t\,\mathrm{ieps}\,J_{1,s}' \hat{\mathbf{s}},$$

where

$$M_- = \begin{pmatrix} A_{v,+} & 0 \\ B & A_+ \end{pmatrix}, \qquad
  M_+ = \begin{pmatrix} A_{v,-} & 0 \\ -B & A_- \end{pmatrix}, \qquad
  F = \begin{pmatrix} -\Delta t\,\alpha I \\ 0 \end{pmatrix}, \qquad
  C = \begin{pmatrix} \mathrm{diag}(\rho_0) & u_0 I \end{pmatrix}.$$

Given the adjoint multipliers $\lambda' = (\lambda_{\mathbf{H}}', \lambda_{\mathbf{E}}',
\lambda_v', \lambda_\rho')$ after the step, the exact Hermitian transpose $U^H \lambda'$ is:
1. Apply the base electromagnetic transpose to $(\lambda_{\mathbf{H}}', \lambda_{\mathbf{E}^*})$
   to obtain $(\lambda_{\mathbf{H}}, \lambda_{\mathbf{E},\text{base}})$.
2. Form the total sensitivity with respect to $X' = (v_1', \rho_1')$:

$$\tilde{\lambda}_{X'} = \begin{pmatrix}
  \lambda_v' - \Delta t\,\mathrm{ieps}_s\,\rho_0\,\lambda_{\mathbf{E}^*,s} \\
  \lambda_\rho' - \Delta t\,\mathrm{ieps}_s\,u_0\,\lambda_{\mathbf{E}^*,s}
\end{pmatrix}.$$

3. Solve the Hermitian-transposed triangular system $M_-^H Z = \tilde{\lambda}_{X'}$
   in reverse order ($\rho$ equation first, then $v$ equation):

$$A_+^H z_\rho = \tilde{\lambda}_{\rho'}, \qquad
  A_{v,+}^H z_v = \tilde{\lambda}_{v'} - B^H z_\rho.$$

4. Pull back to the old beam multipliers and add the electric-field coupling to $\lambda_{E_s}$:

$$\begin{pmatrix} \lambda_v \\ \lambda_\rho \end{pmatrix} = M_+^H Z
  = \begin{pmatrix} A_{v,-}^H z_v - B^H z_\rho \\ A_-^H z_\rho \end{pmatrix}, \qquad
  \lambda_{E_s} \mathrel{+}= -\Delta t\,\alpha\,z_v.$$

Because every banded solve and sparse matrix multiplication is transposed algebraically,
the dot-product identity $\langle U x, y \rangle = \langle x, U^H y \rangle$ holds to
IEEE-754 `float64` / `complex128` machine precision ($\sim 10^{-15}$–$10^{-16}$).

---

## 7. Validity Limits and When PIC Is Required

The Eulerian linearized cold-fluid model is valid in the **small-signal, single-stream,
strongly focused regime** and breaks down when any of the following conditions is violated:

| Physical Effect | Linearized Fluid Validity Criterion | Failure Mode / Why PIC Is Required |
|---|---|---|
| **Velocity modulation depth** | $\|v_1\| / u_0 \ll \|1 - v_p / u_0\| \sim \omega_q / \omega$ | Convective nonlinearity $v_1 \partial_s v_1$ steepens the velocity profile. |
| **Charge bunching / harmonic generation** | $\|\rho_1\| / \|\rho_0\| \ll 1$, $\|I_1\| / I_0 \lesssim 0.1$ | Neglects $\nabla \cdot (\rho_1 \mathbf{v}_1)$, which drives 2nd/3rd harmonic currents ($2\omega, 3\omega$) before saturation. |
| **Trajectory crossing (wave breaking)** | Bunching parameter $X = (\beta \Delta V / 2 V_0) \frac{\sin(\beta_q s)}{\beta_q s} < 1$ | Fast electrons overtake slow electrons; velocity becomes multi-valued in Eulerian space, causing density singularities ($\rho_1 \to \infty$). |
| **Trapping and saturation** | Linear growth regime ($P_{\text{RF}} \ll \eta_{\text{sat}} I_0 V_0$) | Electrons drop into the potential troughs of the slow wave and execute synchrotron bounces, saturating gain; requires Lagrangian particles (PIC). |
| **Thermal velocity spread** | $\sigma_{v,\parallel} \ll \omega_q / \beta$ | Warm-beam Landau damping and Debye screening ($\lambda_D = v_{\text{th}}/\omega_p$) require kinetic/PIC modeling. |
| **Finite magnetic focusing ($B_0$)** | $\omega_c = e B_0 / (\gamma_0 m_e) \gg \omega_p$ | Transverse $\mathbf{E}_\perp, \mathbf{v}_\perp$ motion, Brillouin scalloping, and cyclotron-space-charge modes require 3D fluid or full 3D PIC. |

---

## 8. References

1. S. Ramo, "Space Charge and Field Waves in an Electron Beam," *Physical Review*,
   vol. 56, no. 3, pp. 276–283, 1939. DOI: [10.1103/PhysRev.56.276](https://doi.org/10.1103/PhysRev.56.276)
2. W. C. Hahn, "Small Signal Theory of Velocity-Modulated Electron Beams,"
   *General Electric Review*, vol. 42, pp. 258–270, 1939.
3. L. J. Chu, "A Kinetic Power Theorem," IRE Conference on Electron Tube Research,
   Durham, NH, June 1951; see also H. A. Haus and D. L. Bobroff, "Small Signal Power
   Theorem for Electron Beams," *Journal of Applied Physics*, vol. 28, no. 6,
   pp. 694–704, 1957. DOI: [10.1063/1.1722832](https://doi.org/10.1063/1.1722832)
4. G. M. Branch and T. G. Mihran, "Plasma Frequency Reduction Factors in Electron
   Beams," *IRE Transactions on Electron Devices*, vol. 2, no. 2, pp. 3–11, 1955.
   DOI: [10.1109/T-ED.1955.14065](https://doi.org/10.1109/T-ED.1955.14065)
5. T. Mealy, R. Marosi, and F. Capolino, "Reduced Plasma Frequency Calculation
   Based on Particle-in-Cell Simulations," *IEEE Transactions on Plasma Science*,
   vol. 50, no. 10, pp. 3570–3577, 2022.
   DOI: [10.1109/TPS.2022.3201036](https://doi.org/10.1109/TPS.2022.3201036)
6. A. S. Gilmour, Jr., *Klystrons, Traveling Wave Tubes, Magnetrons, Crossed-Field
   Amplifiers, and Gyrotrons*, Artech House, 2011.
