# Chapter 8 · Vacuum Electronics

Slow-wave structures, space-charge waves on drifting electron beams, traveling-wave tubes, backward-wave oscillators, klystrons, and microwave amplifier circuits modeled with the Finite Integration Technique (FIT) in Wakis.

For cold-test dispersion and Pierce interaction impedance of corrugated and glide-symmetric slow-wave structures, see [Tutorial 6.2 · Slow-wave structure cold test: corrugated and glide-symmetric waveguides](../06_periodic_structures/06-02_sws_cold_test/sws_cold_test.ipynb) in Chapter 6.

## Tutorials

- **[Tutorial 8.1 · Space-charge waves on a drifting electron beam](08-01_space_charge_waves/space_charge_waves.ipynb)** ([`space_charge_waves.py`](08-01_space_charge_waves/space_charge_waves.py)) — Fast and slow space-charge waves on a drifting relativistic electron beam (`LinearizedBeam`) inside a conducting drift tube: Doppler-shifted dispersion $\omega = k_z v_0 \pm R\,\omega_p/\gamma_0^{3/2}$, plasma-frequency reduction factor $R(\gamma_0 b, \gamma_0 a)$ in 2D slab and 3D cylindrical geometry, klystron ballistic current-bunching beat $\lambda_q = \pi v_0 / \omega_q$, and Chu's small-signal kinetic power theorem.
- **[Tutorial 8.2 · Cherenkov radiation from a moving charge](08-02_cherenkov_radiation/cherenkov_radiation.ipynb)** ([`cherenkov_radiation.py`](08-02_cherenkov_radiation/cherenkov_radiation.py)) — 3D Cherenkov radiation from a moving Gaussian bunch (`Beam`): Mach cone angle $\cos\theta_c = 1/(\beta n)$ and shock half-angle $90^\circ - \theta_c$, Frank–Tamm spectrum $d^2W/(d\omega\,dz)$ with the coherent bunch form factor $F(\omega)$, sub-threshold contracted Coulomb field ($\beta n < 1$), dispersive Lorentz passband/stopband edge, and elimination of spurious numerical Cherenkov radiation in vacuum ($\beta = 0.999$) using the longitudinal dispersion-free explicit TE/TM scheme (`scheme="tete"`).
- **[Tutorial 8.4 · Smith–Purcell radiation and the dielectric laser accelerator grating](08-04_smith_purcell_dla/smith_purcell_dla.ipynb)** ([`smith_purcell_dla.py`](08-04_smith_purcell_dla/smith_purcell_dla.py)) — Relativistic charge passing over metallic (PEC) and fused-silica gratings (`WakeSolver` / `SolverFIT3D`): Smith–Purcell space-harmonic dispersion $\lambda = (d/n)(\beta^{-1} - \cos\theta)$, evanescent impact-parameter decay $\exp(-4\pi h/(\beta\gamma\lambda))$, and the inverse process (dielectric laser acceleration, Peralta *et al.* 2013 & Sapra *et al.* 2020) with synchronous period $\lambda_p = \beta\lambda_0$, wake-integral accelerating gradient vs tooth depth and duty cycle, and flat-surface Lawson–Woodward null.

### Planned

- Folded-waveguide and helix slow-wave structures (3D dispersion and interaction impedance).
- Metamaterial and frequency-selective-surface slow-wave structures.
- Traveling-wave tube small-signal convective gain vs Pierce three-wave theory.
- Backward-wave oscillator start-oscillation condition and voltage tuning.
- Klystron cavities ($R/Q$, $Q_0$) and two-cavity ballistic bunching gain.
- Microwave amplifier circuit with microstrip matching networks and transistor S-parameters.
