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
