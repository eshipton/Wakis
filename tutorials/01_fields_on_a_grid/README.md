# Chapter 1 · Fields on a Grid

Foundational time-domain electromagnetics on staggered Yee / Finite Integration Technique (FIT) grids: plane-wave propagation, numerical dispersion, Courant stability, total-field/scattered-field (`TFSFSource`) injection, CPML absorbing boundaries, and flux-box energy balance.

## Tutorials

- **[Tutorial 1.1 · Plane wave in a box](01-01_plane_wave/plane_wave.ipynb)** ([`plane_wave.py`](01-01_plane_wave/plane_wave.py)) — Yee/FIT grid staggering, Courant stability limit and supercritical Nyquist blow-up, 1D and 2D angular numerical dispersion ($\mathcal{O}(\text{cpw}^{-2})$ and the 20 cpw rule), soft current sources (`CurrentSource` + `GaussianSource`), `DFTMonitor` source-spectrum normalization, and total-field/scattered-field (`TFSFSource`) plane-wave injection at round-off leakage.
- **[Tutorial 1.2 · Reflection and transmission at a slab](01-02_slab_rt/slab_rt.ipynb)** ([`slab_rt.py`](01-02_slab_rt/slab_rt.py)) — Normal and oblique plane-wave reflection and transmission at a dielectric slab, Fresnel coefficients and Brewster's angle ($\theta_B = \arctan(n_2/n_1)$), Fabry–Pérot Airy fringes, two-run `FluxMonitor` reference normalization ($R = 1 - \Phi_r/\Phi_{r,\text{ref}}$, $T = \Phi_t/\Phi_{t,\text{ref}}$), convolutional PML (`cpml`) reflection benchmark vs layer count and grading order (compared with `pec` and Mur `abc`), and off-grid Kottke subpixel smoothing ($\mathcal{O}(\Delta z^2)$) vs staircasing ($\mathcal{O}(\Delta z^1)$).
