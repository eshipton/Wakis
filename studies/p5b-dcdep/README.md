# Per-Line Decomposition of the $m = 0$ DC-Depression Gap (`studies/p5b-dcdep`)

## Problem Statement

`compute_discrete_dc_depression(nx, params, line_x=..., weighting="charge")` in
`examples/_space_charge_waves_common.py` predicts the charge-weighted discrete
DC velocity depression $\delta_{\mathrm{charge}}$. In the `sheet_lattice` runs
(rerun `p5b-ufin-1b`), the $m \to 0$ Richardson value
$\delta_0 = (4\delta(m/2) - \delta(m))/3$ of $\delta = (u_0 - \bar{u}_z)/u_0$
lies below the model prediction $\delta_{\mathrm{charge}}$ by:

- $2.68 \times 10^{-7}$ on the coarse grid (`10 x 474`, `16` macro lines;
  seeded ensemble: $3.10 \times 10^{-7}$),
- $8.80 \times 10^{-7}$ on the fine grid (`20 x 948`, `32` macro lines;
  seeded ensemble: $8.88 \times 10^{-7}$).

Because the per-particle energy check $\bar{\gamma} - \bar{\gamma}_{\mathrm{pred}}$
against the simulated transverse potential is $\sim +3.5 \times 10^{-9}$
(holding to $\sim 10^{-7}$ in $\delta$), `studies/p5b-dcdep/dc_decomp.py`
decomposes the gap $\delta_{\mathrm{meas}} - \delta_{\mathrm{charge,model}}$ at
pure DC ($m = 0$) per injected line $l \in \{0, \dots, n_{\mathrm{macro}} - 1\}$.

## Method and Per-Line Decomposition

`BeamInjector` and `ParticleSolver` accept `m_mod = 0.0` directly, emitting a
pure DC beam $I(t) = I_0$ on the deterministic `sheet_lattice` positions
$x_{l,\mathrm{inj}} = b \,(2(l + 0.5)/n_{\mathrm{macro}} - 1)$.

`_LineTrackingBeamInjector` subclasses `BeamInjector` to bin each newly injected
macroparticle's transverse coordinate $x$ to the nearest $x_{l,\mathrm{inj}}$
at injection and store the line index in a per-particle array indexed by
`Species.id` (with an injection-order fallback if `track_ids` is disabled).

Over the settled measurement window (first RF period $k \in \{0, \dots, n_{\mathrm{spp}} - 1\}$
after `n_settle`) and interior axial cells $z \in [0.2 L_z, 0.8 L_z]$, the
driver accumulates per line $l$:

1. $N_l$ [C] (time-averaged physical charge magnitude in the interior window),
   charge-weighted mean transverse position $\bar{x}_l = \langle x \rangle_l$ [m],
   and transverse RMS displacement $\mathrm{rms}(x - x_{l,\mathrm{inj}})$ [m];
2. Charge-weighted mean Lorentz factor $\bar{\gamma}_l = \langle \gamma \rangle_l$ [-],
   mean axial velocity $\bar{v}_{z,l} = \langle v_z \rangle_l$ [m/s], and mean
   dimensionless transverse proper velocity squared
   $\langle (u_x^2 + u_y^2)/c^2 \rangle_l$ [-];
3. $\phi_{\mathrm{sim},l} = \langle -\phi_{\mathrm{sim}}(x_p, z_p) \rangle_l$ [V],
   where $\phi_{\mathrm{sim}}(x_p, z_p) \le 0$ is gathered at each particle's $x_p$
   using `transverse_weight_matrix(x_p, nx, shape)` from the node potential
   integrated across $E_x$ with PEC boundary conditions $\phi(\pm a, z) = 0$
   (matching the library's `gamma_pred` calculation and the positive
   potential-depression sign convention of `model_det["phi_l"]`);
4. Model per-line arrays from
   `compute_discrete_dc_depression(nx, params, weighting="charge", return_details=True)`,
   whose returned dictionary keys are:
   `['delta_disc', 'delta_flux', 'delta_density', 'delta_charge', 'weighting', 'u_l', 'gamma_l', 'phi_l', 'rho_l', 'x_sheet', 'w_mat']`.

Let $\mathcal{D}(v_l; w_l) = 1 - \frac{\sum_l w_l v_l}{u_0 \sum_l w_l}$ and
$\mathcal{D}_{\mathrm{hm}}(v_l) = \mathcal{D}(v_l; \rho_l / v_l)$. Define the
intermediate stage depressions:

- $\delta_0 = \delta_{\mathrm{charge,model}} = \mathcal{D}_{\mathrm{hm}}(u_{\mathrm{model},l})$
  at $x_{l,\mathrm{inj}}$;
- $\delta_C$: model 1D discrete Poisson depression re-evaluated at the measured
  mean line positions `line_x = mean_x_l` (plus fixed-node-potential gather
  variant $\delta_{C,\mathrm{gather}}$);
- $\delta_A = \mathcal{D}_{\mathrm{hm}}(u_{\mathrm{sim},l})$ with
  $\gamma_{\mathrm{sim},l} = \gamma_0 - e \phi_{\mathrm{sim},l}/(m_{\mathrm{eff}} c^2)$
  and $u_{\mathrm{sim},l} = c\sqrt{1 - 1/\gamma_{\mathrm{sim},l}^2}$;
- $\delta_{\gamma,\mathrm{noperp}} = \mathcal{D}_{\mathrm{hm}}(v_{z,\gamma 0,l})$
  with $v_{z,\gamma 0,l} = c\sqrt{1 - 1/\bar{\gamma}_l^2}$;
- $\delta_{\gamma,\mathrm{withperp}} = \mathcal{D}_{\mathrm{hm}}(v_{z,\gamma\perp,l})$
  with $v_{z,\gamma\perp,l} = c\sqrt{1 - (1 + \langle (u_x^2 + u_y^2)/c^2 \rangle_l)/\bar{\gamma}_l^2}$;
- $\delta_{vz,\mathrm{hm}} = \mathcal{D}_{\mathrm{hm}}(\bar{v}_{z,l})$;
- $\delta_{\mathrm{meas}} = \mathcal{D}(\bar{v}_{z,l}; N_l)$.

The four contributions and closure residuals are:

- `(C)` Line shift in $x$: $C = \delta_C - \delta_0$ (and $C_{\mathrm{gather}} = \delta_{C,\mathrm{gather}} - \delta_0$);
- `(A)` Simulated DC potential vs model 1D discrete Poisson potential:
  $A = \delta_A - \delta_C$ (net of line shift) and $A_{\mathrm{direct}} = \delta_A - \delta_0 = A + C$;
- `(B)` Transverse kinetic energy ($u_x, u_y \ne 0$):
  $B = \delta_{\gamma,\mathrm{withperp}} - \delta_{\gamma,\mathrm{noperp}}$
  (and $B_{\mathrm{total}} = B + \mathrm{jensen\_vz} = \delta_{vz,\mathrm{hm}} - \delta_{\gamma,\mathrm{noperp}}$);
- `(D)` Measured charge weighting $N_l$ vs model harmonic-mean weighting $\rho_l / v_{z,l}$:
  $D = \delta_{\mathrm{meas}} - \delta_{vz,\mathrm{hm}}$ (also reported against
  fixed model weights $\rho_l / u_{\mathrm{model},l}$ as $D_{\mathrm{vs\_w\_model}}$);
- Non-ABCD kinematic/energy terms:
  $\mathrm{energy\_residual} = \delta_{\gamma,\mathrm{noperp}} - \delta_A$ and
  $\mathrm{jensen\_vz} = \delta_{vz,\mathrm{hm}} - \delta_{\gamma,\mathrm{withperp}}$;
- Closure checks:
  - `sum_abcd = A + B + C + D`, `closure_abcd = sum_abcd - total = -(energy_residual + jensen_vz)`,
  - `sum_parts = A + B + C + D + energy_residual + jensen_vz`, `closure = sum_parts - total` (exact telescoping identity).

## CLI Usage

Run coarse, fine, or both grids on GPU (when `USE_GPU` is available in the library):

```bash
python studies/p5b-dcdep/dc_decomp.py --grid coarse
python studies/p5b-dcdep/dc_decomp.py --grid fine
python studies/p5b-dcdep/dc_decomp.py --grid both
```

Run only the synthetic self-check and single-channel mutation checks:

```bash
python studies/p5b-dcdep/dc_decomp.py --self-check-only
```

Run the tiny CPU smoke check (`nx=10`, reduced `nz=16`, scaled tunnel length, a few steps on CPU):

```bash
python studies/p5b-dcdep/dc_decomp.py --grid smoke
```

## Verification Performed (Self-Check and Tiny CPU Smoke Only)

> **Note:** Only the synthetic self-check and the tiny CPU smoke check (`--grid smoke`)
> were executed here. The full `coarse` (`10 x 474`) and `fine` (`20 x 948`) grids
> were **not** run in this CPU environment and are intended to be run on GPU.

### 1. Synthetic Model-Identity Self-Check and Mutation Checks

Command:

```bash
/root/computational-electromagnetics/bin/memguard -m 2000 -- /root/computational-electromagnetics/wakis/.venv/bin/python studies/p5b-dcdep/dc_decomp.py --self-check-only
```

Feeding synthetic per-line arrays equal to the model's own values
($\phi_{\mathrm{sim},l} = \phi_{\mathrm{model},l}$, $\bar{x}_l = x_{l,\mathrm{inj}}$,
$\langle u_\perp^2/c^2 \rangle_l = 0$, $N_l = \rho_l / u_{\mathrm{model},l}$)
on both `coarse` (`nx=10`, `n_macro=16`) and `fine` (`nx=20`, `n_macro=32`) gave:

- `A = 0.0`, `B = 0.0`, `C = 0.0`, `D = 0.0`, `energy_residual = 0.0`, `jensen_vz = 0.0`,
  `total = 0.0`, `closure_abcd = 0.0`, `closure = 0.0`, `max_abs_part = 0.0 <= 1e-15`.
- Perturbing each individual channel (`mut_A`, `mut_B`, `mut_C`, `mut_D`) caused
  `assert_self_check_zero` to fail as expected (`failed_as_expected = True`) and
  isolated the change to the perturbed channel (`A_minus_total = 0.0`,
  `B_minus_total = 0.0`, `C_minus_total = 0.0`, `D_minus_total = 0.0`,
  `abs_closure_abcd = 0.0`).

### 2. Tiny CPU Smoke Check (`--grid smoke`)

Command:

```bash
/root/computational-electromagnetics/bin/memguard -m 2000 -- /root/computational-electromagnetics/wakis/.venv/bin/python studies/p5b-dcdep/dc_decomp.py --grid smoke
```

Smoke configuration and output summary (tiny short-tunnel unsettle smoke run on CPU,
`nx = 10`, `nz = 16`, `n_macro = 16`, `n_settle = 263`, `n_window = 10`,
`lz_m = 0.006953271028037383` m, `m_mod = 0.0`):

- `MODEL_DETAILS`: `keys=['delta_disc', 'delta_flux', 'delta_density', 'delta_charge', 'weighting', 'u_l', 'gamma_l', 'phi_l', 'rho_l', 'x_sheet', 'w_mat']`
- `max_inj_x_err_m = 0.0`
- `cells_vs_lines_delta_diff = 2.419939248987646e-16` (`delta_meas_cells = 0.007360697748183174` vs `delta_meas = 0.007360697748182932`)
- `CLOSURE`:
  `total = -2.952680298729e-03`,
  `sum_abcd = -2.221703670499e-03`,
  `|closure_abcd| = 7.309766282307e-04` (equal to `|energy_residual + jensen_vz|` on the short unsettled smoke box),
  `sum_parts = -2.952680298729e-03`,
  `|closure| = 0.000000000000e+00`.

---

## Localising the Steady-State Energy Excess of Wall-Born Particles (`single_particle.py` and `path_identity.py`)

### Background from `dc_decomp.py` Production Runs

In the settled interior of the $m = 0$ sheet beam, `dc_decomp.py` shows that
particles carry $\gamma$ above $\gamma_0 + (q / (m c^2))\,\phi_{\mathrm{sim}}(x)$ by:

- $+1.01 \times 10^{-8}$ on the coarse grid (`10 x 474`) and $+3.45 \times 10^{-8}$
  on the fine grid (`20 x 948`);
- doubling the settle time changes this by $< 11\%$ ($1.12 \times 10^{-8}$ coarse,
  $3.44 \times 10^{-8}$ fine);
- halving $\Delta t$ at fixed $\Delta z$ gives $7.7 \times 10^{-9}$ ($-24\%$);
- disabling the birth kick gives $+1.51 \times 10^{-5}$, and disabling the PEC wall
  closure current gives $-1.56 \times 10^{-5}$.

In a static field, the discrete particle energy identity with order-$p$ nodal
gathered potential $\Phi(\mathbf{x})$ ($\Phi = 0$ on the wall node) is

$$\gamma_{\mathrm{int}} - \gamma_0 + \frac{q}{m c^2}\,\Phi(\mathbf{x}_{\mathrm{int}})
= \frac{q}{m c^2} \sum_{m > n_b} \bigl[\Phi^m(\mathbf{x}^m) - \Phi^{m-1}(\mathbf{x}^m)\bigr]
+ O\bigl((v\,\Delta t)^2 \phi''\bigr).$$

Two scripts isolate the origin of the interior excess:

1. `studies/p5b-dcdep/single_particle.py`: single macroparticle of unit weight
   injected from the $z = 0$ PEC wall into a frozen, exact discrete gradient field
   $\mathbf{E} = -\nabla_h \phi$ on the coarse `10 x 1 x 474` grid;
2. `studies/p5b-dcdep/path_identity.py`: production driver comparing the transverse
   potential $\phi_{\mathrm{trans}}$ and axial potential $\phi_{\mathrm{axial}}$
   (Part a) and evaluating the step-by-step transit sum
   $\sum_{m > n_b} [\Phi^m(\mathbf{x}^m) - \Phi^{m-1}(\mathbf{x}^m)]$ for 3
   tracked particles per lattice line (Part b).

### 1. Frozen-Potential Single-Particle Study (`single_particle.py`)

`single_particle.py` sets the nodal potential

$$\phi(x_i, z_k) = \phi_{\min}\!\left(1 - \frac{x_i^2}{a^2}\right)\!\left(1 - e^{-z_k / L}\right)$$

with $\phi = 0$ on all PEC walls ($x = \pm a$, $z = 0$, $z = L_z$), sets
$E_x$ and $E_z$ on their primal Yee edges via $\mathbf{E} = -\nabla_h \phi$
($E_y = 0$), and freezes the field advance (`solver._update_H = lambda: None`,
`solver._update_E = lambda: None`) while keeping `solver.one_step = solver._one_step`
so `ParticleSolver._mid_step_hook` executes normally. A single electron is
injected at $z = 0$ on the central sheet lattice line $x_{\mathrm{inj}} = 5 \times 10^{-5}\text{ m}$
and pushed to $z > 5 L$ ($L = a = 2.0 \times 10^{-3}\text{ m}$) for $\Delta t$,
$\Delta t / 2$, and $\Delta t / 4$.

Command:

```bash
/root/computational-electromagnetics/bin/memguard -m 2000 -- /root/computational-electromagnetics/wakis/.venv/bin/python studies/p5b-dcdep/single_particle.py
```

Measured runtime output (`SP1` records):

| Quantity | $\Delta t$ (`dt_div=1`) | $\Delta t / 2$ (`dt_div=2`) | $\Delta t / 4$ (`dt_div=4`) |
| --- | --- | --- | --- |
| `dt` [s] | `4.704775e-13` | `2.352388e-13` | `1.176194e-13` |
| `z_final` [m] ($z / L$) | `0.010819` (`5.410`) | `0.010819` (`5.410`) | `0.010819` (`5.410`) |
| `phi_gathered` [V] | `-226.646891` | `-226.646890` | `-226.646890` |
| `excess_kick_on_raw` | `+2.511019e-08` | `+1.332267e-08` | `+6.166209e-09` |
| `excess_kick_on_sync` | `+1.145634e-08` | `+6.495747e-09` | `+2.752744e-09` |
| `excess_before_kink_sync` ($\zeta = 0.4427 < 0.5$) | `+6.008961e-11` | `+1.502220e-11` | `+3.755628e-12` |
| `kink_jump_sync` (crossing $\zeta = 0.5$) | `+1.773071e-08` | `+7.939953e-09` | `+3.049048e-09` |
| `kink_jump_pred` (closed-form $\zeta = 0.5$ formula) | `+1.772046e-08` | `+7.938816e-09` | `+3.048946e-09` |
| `kink_theta` ($\theta = (\zeta_{m^*} - 0.5)/\Delta\zeta$) | `0.094115` | `0.188220` | `0.376438` |
| `excess_smooth_interior_sync` | `-6.274377e-09` | `-1.444205e-09` | `-2.963042e-10` |
| `excess_kick_off_raw` ($z > 5 L$) | `+1.090247e-05` | `+1.089041e-05` | `+1.088314e-05` |
| `excess_kick_off_sync` ($z > 5 L$) | `+1.088883e-05` | `+1.088359e-05` | `+1.087973e-05` |
| `pred_birth_kick_dgamma` ($(q/mc^2)\sum_k W_k \Psi_k$) | `-1.087697e-05` | `-1.087697e-05` | `-1.087697e-05` |
| `pred_birth_kick_J` ($q \sum_k W_k \Psi_k$ [J]) | `-8.905090e-19` | `-8.905090e-19` | `-8.905090e-19` |
| `birth_kick_rel_err_dgamma` / `birth_kick_rel_err_J` | `0.0` | `0.0` | `0.0` |

**Diagnosis from `single_particle.py`:**

1. Even in a strictly frozen, curl-free gradient field $\mathbf{E} = -\nabla_h \phi$,
   a wall-born particle acquires $+1.145634 \times 10^{-8}$ (`sync`) /
   $+2.511019 \times 10^{-8}$ (`raw`) at nominal coarse $\Delta t$, reproducing the
   $+1.01 \times 10^{-8}$ excess measured in `dc_decomp.py`.
2. **Why the excess arises at $\zeta = 0.5$ ($z = 0.5\,\Delta z$) and why a single
   $\Delta t$ halving reduces it by less than $4\times$:**
   For `shape = 2` (`gather = "galerkin"`), `_gather_component_3d(..., periodic=(False, True, False))`
   masks exterior ghost indices $k < 0$ to zero at the non-periodic $z = 0$ PEC wall.
   Consequently, for $0 \le \zeta < 0.5$, the staggered axial gather of $E_z$
   retains only edge $k = 0$ with weight $1/2 + \zeta$, so
   $E_{z,\mathrm{gathered}}(x, \zeta) = (1/2 + \zeta)\,E_{z,0}(x) = -\partial_z \Phi_{\mathrm{gathered}}(x, z)$
   holds identically. However, at the first cell midpoint $\zeta = 0.5$, the axial
   slope $\partial_\zeta E_{z,\mathrm{gathered}}$ jumps from $+E_{z,0}$ (for $\zeta < 0.5$)
   to $E_{z,1} - E_{z,0}$ (for $0.5 \le \zeta < 1.5$), creating a jump discontinuity
   $\Delta(\partial_z^2 \Phi_{\mathrm{gathered}}) = (2 E_{z,0} - E_{z,1})/\Delta z$
   in the second derivative of $\Phi_{\mathrm{gathered}}$.
3. When the leapfrog integrator steps across $\zeta = 0.5$ on step $m^* = \lceil 0.5 / \Delta\zeta \rceil$
   with normalized step size $\Delta\zeta = v_z\,\Delta t / \Delta z$ and crossing
   fraction $\theta = (\zeta_{m^*} - 0.5)/\Delta\zeta \in (0, 1)$, the single-step
   trapezoidal quadrature error across the kink is

   $$\Delta\gamma_{\mathrm{kink}} = \frac{e\,\Delta z}{m_e c^2}\,\frac{1}{2}\,\theta(1 - \theta)\,(\Delta\zeta)^2\,(2 E_{z,0} - E_{z,1}),$$

   which matches the measured step-8 jump `kink_jump_sync` to four significant
   digits (`1.7720e-08` vs `1.7731e-08` at $\Delta t$; `7.9388e-09` vs `7.9400e-09`
   at $\Delta t/2$; `3.0489e-09` vs `3.0490e-09` at $\Delta t/4$). Because
   $0.5 / \Delta\zeta_1 \approx 7.906$ at `cfln = 0.25`, $\theta_1 = 0.0941$ is near
   the edge of $(0, 1)$ and doubles to $\theta_2 = 0.1882$ and $\theta_4 = 0.3764$,
   so $\theta(1 - \theta)(\Delta\zeta)^2$ decreases by $2.23\times$ ($\Delta t \to \Delta t/2$)
   and $2.60\times$ ($\Delta t/2 \to \Delta t/4$). Meanwhile, both the pre-kink
   excess (`excess_before_kink_sync`, ratios `4.0001` and `3.9999`) and the smooth
   interior excess after subtracting the single-step $\zeta = 0.5$ kink jump
   (`excess_smooth_interior_sync`, ratios `4.3445` and `4.8741`) exhibit the expected
   $O(\Delta t^2)$ ($\ge 4\times$ per halving) convergence.

### 2. 3D Path-Potential and Transit-Identity Study (`path_identity.py`)

`path_identity.py` implements:

- **Part (a):** On the settled $m = 0$ beam, saves $\mathbf{E}$ and $\mathbf{H}$ on
  the full grid and computes `phi_trans` ($-\int_{-a}^x E_x\,\mathrm{d}x'$),
  `phi_axial` ($-\int_0^z E_z\,\mathrm{d}z'$), and both gathered at each sheet
  lattice line $x_l$ with order-$p$ B-spline weights, together with `Ny` and $y$-BC
  consistency, per-line `max` and `mean` $|\phi_{\mathrm{axial}} - \phi_{\mathrm{trans}}| / |\phi|$,
  interior $\max|\nabla_h \times \mathbf{E}| / (\max|\mathbf{E}|/\Delta z)$, and
  interior $\max|\mathbf{H}|$.
- **Part (b):** Tracks 3 particles per lattice line for one transit and accumulates
  $(q / m c^2)\sum_{m > n_b} [\Phi^m(\mathbf{x}^m) - \Phi^{m-1}(\mathbf{x}^m)]$
  step by step with $\Phi$ from the axial integral, reporting it beside each
  particle's interior $\gamma$ excess in a single `PATH {json}` output line per grid.

Production CLI usage (intended for GPU):

```bash
python studies/p5b-dcdep/path_identity.py --grid coarse
python studies/p5b-dcdep/path_identity.py --grid fine
```

Synthetic self-check command run in this job (tiny `6 x 1 x 16` grid; production
`coarse` and `fine` grids were **not** run):

```bash
/root/computational-electromagnetics/bin/memguard -m 2000 -- /root/computational-electromagnetics/wakis/.venv/bin/python studies/p5b-dcdep/path_identity.py --self-check-only
```

Self-check results:

- `Ny = 1`, `bc_y = ['periodic', 'periodic']`
- Curl-free analytic discrete potential on `(6, 1, 16)`:
  - `clean_node_max_rel_diff = 7.105427357601002e-17` ($< 10^{-12}$)
  - `clean_line_max_rel_diff = 1.610802171748338e-16` ($< 10^{-12}$)
  - `clean_3d_max_rel_diff = 0.0` ($< 10^{-12}$)
  - `clean_rel_curl_E = 1.1175724189210196e-16` ($< 10^{-12}$)
- Mutation check (perturbing one interior $E_z$ edge by $10 / \Delta z$):
  - `mutated_line_max_rel_diff = 0.03429561200923799` ($> 10^{-12}$, detected as expected)
  - `mutated_rel_curl_E = 0.08999881856582609`
- Synthetic Part (b) transit-tracking check (linearly ramping potential on `(6, 1, 16)`,
  3 particles $\times$ 4 lines = 12 tracked particles):
  - `tiny_tracked_completed = 12`
  - `tiny_mean_sum_dphi_over_mc2 = 1.765432e-04` vs `tiny_mean_gamma_excess_sync = 1.761968e-04`.


---

## Removing the First-Cell Entrance Error: Image Wall Stencil (`wall_stencil="image"`, 2026-11-10)

### Options evaluated from first principles

The leapfrog energy gain in a static field is the trapezoid rule of the gathered field $E_g$ on the
step positions $z^n$; it is exact on linear pieces of $E_g$, and a slope jump $J$ crossed at phase
$\theta$ adds $(|q|\Delta z/mc^2)\,\tfrac12\theta(1-\theta)\Delta\zeta^2|J|$.

- **Staggered birth phases (rejected).** $\theta(1-\theta) \ge 0$ for every phase, so spreading the
  births uniformly over the step does not cancel the error: it replaces the coherent coefficient
  $\tfrac12\theta(1-\theta)$ by its mean $1/12$ (coarse: $0.042 \to 0.083$, i.e. 2x worse; fine:
  $0.112 \to 0.083$). Only $\theta \to 0$ (a step landing exactly on $\zeta = 1/2$) removes it, which
  cannot hold for modulated or self-consistent beams.
- **Image (mirrored-ghost) stencil (implemented, opt-in).** Fold the out-of-domain stencil back with
  the PEC image symmetry (charge odd; normal $E$ even, tangential $E$ odd; $B_n$ odd, $B_t$ even),
  i.e. deposit the real and the image path with the standard masked Esirkepov deposit (Gauss exact
  per path) and gather with its transpose. A particle on the wall then has zero visible charge and
  zero effective potential, so the D17 closure current and birth kick vanish identically (no
  re-derivation needed: both are zero). The first-half-cell gather becomes $E_g = E_0$ (constant,
  so the birth half step is exact) and the knot at $\zeta = 1/2$ jumps by $E_1 - E_0$ instead of
  $E_1 - 2E_0$: the coherent error becomes second order in $\Delta z$.

### Prediction vs measurement

| quantity | coarse masked | coarse image | fine masked | fine image |
| --- | --- | --- | --- | --- |
| frozen-field knot jump (`single_particle.py`, dt) | 1.773e-8 (pred 1.772e-8) | 2.914e-9 (pred 2.915e-9) | — | — |
| static 1D model cell-0 excess (`wall_gather_model.py`) | 2.65e-8 | 6.2e-9 | 3.82e-8 | 5.7e-9 |
| settled beam RPROF cell 0 (`path_identity.py`) | 3.105e-8 | 1.253e-8 | 3.894e-8 | 6.45e-9 |
| settled beam total excess (RPROF cum) | +8.96e-9 | -9.78e-9 | +3.450e-8 | +1.08e-9 |
| static model total excess at zeta = 6 | +1.30e-8 | -7.3e-9 | +3.59e-8 | +3.4e-9 |
| m -> 0 delta0 - delta_disc_charge (`dc_image.py`) | -2.676e-7 | +2.438e-7 | -8.800e-7 | -5.18e-8 |
| delta0 - delta_disc_flux | +1.130e-6 | +1.641e-6 | +6.01e-7 | +1.429e-6 |
| derived entrance bound in delta | 3.62e-6 | 9.60e-7 | 2.05e-6 | 3.31e-7 |

The masked runs reproduce the earlier `p5b-path5` RPROF and DEPEXTRAP gaps to all printed digits
(default path unchanged). On the fine grid the image stencil removes the entrance term (gap 17x
smaller, total excess 32x smaller). On the coarse grid the first-cell remainder falls 2.5x
(predicted 4.3x) and the remaining near-wall leapfrog error (cells 1-5, the same in both stencils,
O(dt^2); interior knots of the near-wall field) now dominates with the opposite sign, so the gap
magnitude is unchanged (+2.4e-7 vs -2.7e-7) but lies inside the 3.8x tighter derived bound. The
coarse cell-0 remainder is 2x above the static-model prediction (open: non-static settled field or
the coarse knot-phase sequence; the fine grid agrees to 13 %).

Jobs: `fce-dc-{coarse,fine}-m{10,05}-{image,masked}*`, `fce-path-{coarse,fine}-{image,masked}*`.
