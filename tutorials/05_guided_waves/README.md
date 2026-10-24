# Chapter 5 · Guided Waves

Worked examples of passive silicon-on-insulator (SOI) and silicon-nitride ($\text{Si}_3\text{N}_4$) photonic integrated circuit (PIC) waveguides, splitters, bends, mode converters, couplers, and ring resonators modeled with the Finite Integration Technique (FIT) in Wakis (`wakis.templates.photonics`).

Every tutorial runs as a **2D effective-index (EIM)** simulation (`Ex`, `Ey`, `Hz`; one periodic cell along $z$) with **Kottke (2008) subpixel permittivity smoothing** on extruded polygonal slabs (`PolySlab`), broadband eigenmode port sources and monitors (`ModeSource`, `ModeMonitor`), and automated $N$-port S-parameter extraction (`PICSimulation.run_s_parameters`). All geometry primitives (`PolySlab`, `straight`, `taper`, `spline_taper`, `arc`, `euler_bend`, `sbend`, `ring`) retain their full vertical $z$ bounds so that selected tutorials can be extended to 3D on GPU.

## Tutorials

- **[Tutorial 5.1 · Y-junction power splitter](05-01_y_junction/y_junction.ipynb)** ([`y_junction.py`](05-01_y_junction/y_junction.py)) — Particle-swarm-optimized $2\;\mu\text{m}$ multi-mode spline Y-junction of Zhang *et al.* (*Opt. Express* **21**, 1310, 2013) on $220\;\text{nm}$ SOI: 1D vertical slab effective-index reduction, 13-knot cubic-spline taper + cosine S-bend construction, staircase vs Kottke subpixel permittivity maps on the bifurcation tip, broadband $3\times 3$ S-parameters ($1500\text{–}1600\;\text{nm}$), 2D Poynting flux-box energy budget, and 3-resolution convergence study.
- **[Tutorial 5.2 · Euler waveguide bend](05-02_euler_bend/euler_bend.ipynb)** ([`euler_bend.py`](05-02_euler_bend/euler_bend.py)) — Hybrid clothoid–circular $90^\circ$ silicon waveguide bend vs constant-radius circular arcs: continuous curvature ramp $\kappa(s)$, mode-mismatch suppression at straight-to-bend junctions, radius sweep ($2.5\text{–}4.0\;\mu\text{m}$), and Kottke vs staircase convergence.
- **[Tutorial 5.3 · Mode size converters: tapers, Luneburg lens and semi-lens](05-03_size_converter/size_converter.ipynb)** ([`size_converter.py`](05-03_size_converter/size_converter.py)) — Adiabatic linear/parabolic tapers, graded-index Luneburg lens, and ultra-compact piecewise-constant semi-lens beam expander from $0.5\;\mu\text{m}$ strip to $12\;\mu\text{m}$ output waveguide, with multi-mode $\text{TE}_0/\text{TE}_2/\text{TE}_4$ overlap decomposition and 50 nm minimum-feature resolution audit.
- **[Tutorial 5.4 · Strip-to-slot waveguide mode converters](05-04_strip_to_slot/strip_to_slot.ipynb)** ([`strip_to_slot.py`](05-04_strip_to_slot/strip_to_slot.py)) — Symmetric double-tip adiabatic strip-to-slot converter (Wang *et al.* 2016) and multimode-interference (MMI) strip-to-slot converter coupling into a $100\;\text{nm}$ air slot, comparing Kottke subpixel smoothing against staircase snapping across slot-gap resolution levels.
- **[Tutorial 5.5 · Waveguide crossing with cosine tapers](05-05_crossing/crossing.ipynb)** ([`crossing.py`](05-05_crossing/crossing.py)) — Orthogonal $90^\circ$ SOI waveguide crossing with $\cos^2$ multimode self-imaging tapers at O-band ($1310\;\text{nm}$): $\text{TE}_0 + \text{TE}_2$ two-mode beating, insertion loss ($0.19\;\text{dB}$), crosstalk ($<-44\;\text{dB}$), and geometry optimization sweep.
- **[Tutorial 5.6 · 1×4 multimode-interference power splitter](05-06_mmi_1x4/mmi_1x4.ipynb)** ([`mmi_1x4.py`](05-06_mmi_1x4/mmi_1x4.py)) — Symmetric $1\times 4$ $\text{Si–GaN–Si}$ slot-waveguide MMI splitter: Soldano & Pennings self-imaging theory ($L_{\text{MMI}} \approx 3 L_\pi / 16$), 5-port S-parameter extraction, channel uniformity and excess loss across $W_{\text{MMI}} \times L_{\text{MMI}}$, and Kottke vs staircase convergence.
- **[Tutorial 5.7 · 2×2 multimode-interference coupler](05-07_mmi_2x2/mmi_2x2.ipynb)** ([`mmi_2x2.py`](05-07_mmi_2x2/mmi_2x2.py)) — Paired-interference $2\times 2$ SOI $3\;\text{dB}$ MMI coupler ($L_{\text{MMI}} \approx L_\pi / 2$): bar/cross power splitting ratio, $90^\circ$ quadrature phase relation $\Delta\phi = \arg(S_{41}/S_{31}) = \pm\pi/2$, and length/width sensitivity sweeps.
- **[Tutorial 5.8 · Directional coupler: supermode beating and sidewall-angle sensitivity](05-08_directional_coupler/directional_coupler.ipynb)** ([`directional_coupler.py`](05-08_directional_coupler/directional_coupler.py)) — Evanescently coupled parallel SOI waveguides: symmetric/antisymmetric supermode beating ($L_x = \lambda_0 / [2(n_{\text{sym}} - n_{\text{asym}})]$), coupling-length and gap sweeps, and trapezoidal sidewall-angle sensitivity ($\theta_{\text{sw}} = 90^\circ \to 80^\circ$).
- **[Tutorial 5.9 · Waveguide-to-ring point coupler](05-09_ring_coupling/ring_coupling.ipynb)** ([`ring_coupling.py`](05-09_ring_coupling/ring_coupling.py)) — Bus-to-microring directional point coupler with PML-truncated ring arms: extraction of complex transmission $t(\lambda)$ and cross-coupling $\kappa(\lambda)$, exponential decay of $|\kappa|^2$ vs coupling gap ($50\text{–}250\;\text{nm}$), and gap-cell resolution study.
- **[Tutorial 5.10 · Microring add-drop filter](05-10_ring_add_drop/ring_add_drop.ipynb)** ([`ring_add_drop.py`](05-10_ring_add_drop/ring_add_drop.py)) — Four-port SOI microring resonator add-drop filter ($R = 5\;\mu\text{m}$): free spectral range ($\text{FSR} \approx 23.7\;\text{nm}$), group index $n_g$, loaded quality factor $Q_L$, extinction ratio, and coupled-mode-theory Lorentzian lineshape validation across coupling gaps.

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
