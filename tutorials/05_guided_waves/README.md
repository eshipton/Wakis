# Chapter 5 · Guided Waves

Worked examples of passive silicon-on-insulator (SOI) and silicon-nitride ($\text{Si}_3\text{N}_4$) photonic integrated circuit (PIC) waveguides, splitters, bends, mode converters, couplers, and ring resonators modeled with the Finite Integration Technique (FIT) in Wakis (`wakis.templates.photonics`).

Every tutorial runs as a **2D effective-index (EIM)** simulation (`Ex`, `Ey`, `Hz`; one periodic cell along $z$) with **Kottke (2008) subpixel permittivity smoothing** on extruded polygonal slabs (`PolySlab`), broadband eigenmode port sources and monitors (`ModeSource`, `ModeMonitor`), and automated $N$-port S-parameter extraction (`PICSimulation.run_s_parameters`). All geometry primitives (`PolySlab`, `straight`, `taper`, `spline_taper`, `arc`, `euler_bend`, `sbend`, `ring`) retain their full vertical $z$ bounds so that selected tutorials can be extended to 3D on GPU.

## Tutorials

- **[Tutorial 5.1 · Y-junction power splitter](05-01_y_junction/y_junction.ipynb)** ([`y_junction.py`](05-01_y_junction/y_junction.py)) — Particle-swarm-optimized $2\;\mu\text{m}$ multi-mode spline Y-junction of Zhang *et al.* (*Opt. Express* **21**, 1310, 2013) on $220\;\text{nm}$ SOI: labelled setup schematic, 1D vertical slab effective-index reduction, 13-knot cubic-spline taper + cosine S-bend construction, staircase vs Kottke subpixel permittivity maps on the bifurcation tip, graded non-uniform Cartesian mesh ($-34.2\%$ cells, $-44.8\%$ wall time at equal accuracy), broadband $3\times 3$ S-parameters ($1500\text{–}1600\;\text{nm}$), and 2D Poynting flux-box energy budget.
- **[Tutorial 5.2 · Euler waveguide bend](05-02_euler_bend/euler_bend.ipynb)** ([`euler_bend.py`](05-02_euler_bend/euler_bend.py)) — Hybrid clothoid–circular $90^\circ$ silicon waveguide bend vs constant-radius circular arcs: labelled setup schematic, continuous curvature ramp $\kappa(s)$, mode-mismatch suppression at straight-to-bend junctions, radius sweep ($2.5\text{–}4.0\;\mu\text{m}$), diagonal-corridor graded non-uniform mesh ($-19.3\%$ cells, $-38.6\%$ wall time), and Kottke vs staircase convergence.
- **[Tutorial 5.3 · Mode size converters: tapers, Luneburg lens and semi-lens](05-03_size_converter/size_converter.ipynb)** ([`size_converter.py`](05-03_size_converter/size_converter.py)) — Adiabatic linear/parabolic tapers, graded-index Luneburg lens (with body-fitted `wakis.unstructured` hybrid square+triangle circular rim mesh), and ultra-compact piecewise-constant semi-lens beam expander from $0.5\;\mu\text{m}$ strip to $12\;\mu\text{m}$ output waveguide, with labelled setup schematics, graded non-uniform semi-lens mesh ($-26.6\%$ cells, $-33.2\%$ wall time), multi-mode $\text{TE}_0/\text{TE}_2/\text{TE}_4$ overlap decomposition, and $50\;\text{nm}$ minimum-feature resolution audit.
- **[Tutorial 5.4 · Strip-to-slot waveguide mode converters](05-04_strip_to_slot/strip_to_slot.ipynb)** ([`strip_to_slot.py`](05-04_strip_to_slot/strip_to_slot.py)) — Symmetric multimode-interference (MMI) strip-to-slot converter (Deng *et al.* 2016) and asymmetric/symmetric rail-taper converters (Palmer *et al.* 2013; Wang *et al.* 2009) coupling into a $100\;\text{nm}$ oxide slot: labelled setup schematics, graded non-uniform Cartesian mesh ($-38.1\%$ cells, $-58.6\%$ wall time), and Kottke subpixel smoothing vs staircase snapping across slot-gap resolution levels.
- **[Tutorial 5.5 · Waveguide crossing with cosine tapers](05-05_crossing/crossing.ipynb)** ([`crossing.py`](05-05_crossing/crossing.py)) — Orthogonal $90^\circ$ SOI waveguide crossing with convex cosine multimode self-imaging tapers at O-band ($1310\;\text{nm}$, Chandran *et al.* 2020): labelled setup schematics, $\text{TE}_0 + \text{TE}_2$ two-mode beating, insertion loss ($0.090\;\text{dB}$ in 2D EIM), crosstalk ($<-43\;\text{dB}$), $D_4$-symmetric graded non-uniform mesh ($-55.2\%$ cells), and cosine vs linear vs untapered comparison.
- **[Tutorial 5.6 · 1×4 multimode-interference power splitter](05-06_mmi_1x4/mmi_1x4.ipynb)** ([`mmi_1x4.py`](05-06_mmi_1x4/mmi_1x4.py)) — Symmetric $1\times 4$ $\text{Si–GaN–Si}$ slot-waveguide MMI splitter (Malka *et al.* 2016): labelled setup schematic, Soldano & Pennings self-imaging theory ($L_{\text{MMI}} \approx 3 L_\pi / 16$), 5-port S-parameter extraction, channel uniformity and excess loss across $W_{\text{MMI}} \times L_{\text{MMI}}$, graded non-uniform non-square Cartesian mesh ($-26.4\%$ cells, $-30.3\%$ wall time, $2.5\times$ smaller error vs $40\;\text{cpw}$ reference), and Kottke vs staircase convergence.
- **[Tutorial 5.7 · 2×2 multimode-interference coupler](05-07_mmi_2x2/mmi_2x2.ipynb)** ([`mmi_2x2.py`](05-07_mmi_2x2/mmi_2x2.py)) — General-interference $2\times 2$ SOI $3\;\text{dB}$ MMI coupler ($L_{\text{mmi}} \approx 3 L_\pi / 2$): labelled setup schematic, 4-mode Talbot self-imaging carpet, bar/cross power splitting ratio, $-90^\circ$ quadrature phase relation $\Delta\phi = \arg(S_{31} S_{41}^*)$, graded non-uniform non-square Cartesian mesh ($-34.4\%$ cells, $-41.8\%$ wall time, $1.87\times$ lower 2-port RMS error vs baseline **L3** against the $40\;\text{cpw}$ reference), and MMI box length sweep.
- **[Tutorial 5.8 · Directional coupler: supermode beating and sidewall-angle sensitivity](05-08_directional_coupler/directional_coupler.ipynb)** ([`directional_coupler.py`](05-08_directional_coupler/directional_coupler.py)) — Evanescently coupled parallel SOI waveguides: labelled setup schematics (single-port and dual-port $\pi/2$ quadrature routing), symmetric/antisymmetric supermode beating ($L_\pi = \lambda / [2(n_{\text{even}} - n_{\text{odd}})]$), coupling-length ($0\text{–}8\;\mu\text{m}$) and trapezoidal sidewall-angle ($\theta_{\text{sw}} = 0^\circ\text{–}30^\circ$) sweeps, mirror-symmetric graded non-uniform Cartesian mesh ($-38.3\%$ cells, $-53.0\%$ wall time at equal accuracy), and Kottke vs staircase convergence.
- **[Tutorial 5.9 · Waveguide-to-ring point coupler](05-09_ring_coupling/ring_coupling.ipynb)** ([`ring_coupling.py`](05-09_ring_coupling/ring_coupling.py)) — Bus-to-microring directional point coupler with CPML-truncated ring arms: labelled setup schematics, extraction of complex transmission $t(\lambda)$ and cross-coupling $\kappa(\lambda)$, exponential decay of $|\kappa|^2$ vs coupling gap ($50\text{–}250\;\text{nm}$), mirror-symmetric graded non-uniform Cartesian mesh ($165\text{k}$ cells, $-37.2\%$ cells, $-54.8\%$ wall time at equal accuracy), body-fitted `wakis.unstructured` hybrid square+triangle circular rim mesh, and gap-cell resolution study.
- **[Tutorial 5.10 · Microring add-drop filter](05-10_ring_add_drop/ring_add_drop.ipynb)** ([`ring_add_drop.py`](05-10_ring_add_drop/ring_add_drop.py)) — Four-port SOI microring resonator add-drop filter ($R = 5\;\mu\text{m}$): labelled setup schematics, free spectral range ($\text{FSR} \approx 23.7\;\text{nm}$), group index $n_g$, loaded quality factor $Q_L$, extinction ratio, mirror-symmetric graded non-uniform Cartesian mesh ($183\text{k}$ cells, $-35.3\%$ cells, $-59.2\%$ wall time at equal accuracy), body-fitted `wakis.unstructured` hybrid square+triangle circular rim mesh, and coupled-mode-theory Lorentzian lineshape validation across coupling gaps.
- **[Tutorial 5.11 · Kerr ($\chi^{(3)}$) nonlinear media in FIT and FIT-informed soliton microcombs](05-11_kerr_microcomb/kerr_microcomb.ipynb)** ([`kerr_microcomb.py`](05-11_kerr_microcomb/kerr_microcomb.py)) — Sub-cycle $\chi^{(3)}$ Kerr + delayed Raman time stepping in `SolverFIT3D` (`wakis.KerrMaterial`) and hybrid FIT $\to$ Lugiato–Lefever Equation (`wakis.solve_lle`) microcomb modeling: labelled setup schematics, cylindrical Bessel whispering-gallery dispersion benchmark ($m = 14\text{–}20$), anomalous ($D_2 > 0$) vs normal ($D_2 < 0$) integrated dispersion $D_{\text{int}}(\mu)$, bus-coupled quality factor extraction, direct 2D FIT Kerr ring resonance shift vs $V_{\text{eff,NL}}$ perturbation theory, locally refined graded mesh with a 3-level convergence study (observed order and extrapolation band), and split-step Fourier pump-detuning scan from modulation instability and chaotic breathers onto the single dissipative Kerr soliton step ($\text{sech}^2$ comb bandwidth $\propto D_2^{-1/2}$).

## Running a guided-wave tutorial

```bash
# Quick local run (< 60 s on CPU, writes data/*.npz and all figures in figures/)
python tutorials/05_guided_waves/05-01_y_junction/y_junction.py --quick

# Converged 2D study across all resolutions
python tutorials/05_guided_waves/05-01_y_junction/y_junction.py --full

# Re-render figures from saved data/*.npz without re-running FIT
python tutorials/05_guided_waves/05-01_y_junction/y_junction.py --plot

# Regenerate the Jupyter notebook from the Jupytext percent source
python tutorials/build.py 05-01
```

## Cloud Batch parameter sweeps (`tutorials/05_guided_waves/cloud/`)

Each guided-wave tutorial exposes `--list-tasks` and `--task I --out DIR` so its study points can run in parallel as a single Google Cloud Batch array job:

```bash
# Inspect task manifest and Batch job JSON locally (no cloud calls)
python tutorials/05_guided_waves/cloud/submit.py --dry-run tutorials/05_guided_waves/05-01_y_junction/y_junction.py

# Submit CPU array job (n2-standard-4, OMP_NUM_THREADS=4) or GPU job (--gpu)
python tutorials/05_guided_waves/cloud/submit.py tutorials/05_guided_waves/05-01_y_junction/y_junction.py

# Download completed task_*.npz files into data/ and regenerate figures
python tutorials/05_guided_waves/cloud/collect.py RUN_ID tutorials/05_guided_waves/05-01_y_junction --plot
```
