# 📚 Physics guide

This section provides the theoretical foundations behind Wakis, offering a clear and concise explanation of the physics principles and numerical methods that power the simulation engine.

We begin by revisiting Maxwell’s equations in their integral form and explaining how they are discretized using the Finite Integration Technique (FIT) on a structured Cartesian grid. The formulation naturally leads to Maxwell Grid Equations (MGEs), which are solved in time using a leapfrog scheme. The fields and material properties are represented on a Yee-type staggered lattice, and anisotropic or spatially varying materials are handled via sparse metric tensors.

Boundary conditions (PEC, PMC, periodic, Mur ABC, PML, and CPML) are discussed along with the treatment of sources and initial conditions. We also highlight how the solver supports geometry import and sub-pixel smoothing for embedded CAD models.

Finally, we cover the implementation aspects, including GPU acceleration with CuPy and parallelization with mpi4py, enabling high-resolution 3D simulations across multiple devices.

```{contents}
:depth: 3
```

## 1. Introduction

### 🎯 Motivation

In modern accelerators, precise knowledge of beam-coupling impedance and wakefields is essential to ensure beam quality, mitigate heating, and optimize component design. Analytical methods, while powerful, often fall short for realistic 3D geometries — this is where full electromagnetic solvers like Wakis become indispensable.

### 📖 Background

The evaluation of **beam-coupling impedance** and **wakefields** is fundamental to the design and operation of particle accelerators. As charged particle bunches traverse beamline components, they interact with material boundaries and geometric discontinuities, generating electromagnetic fields collectively known as **wakefields**.

These wakefields can:
- Degrade beam quality and cause **coherent instabilities**
- Induce **energy spread** and **emittance growth**
- Lead to **beam-induced heating** and **power loss** in accelerator components

In the **frequency domain**, the response of a structure to the passing beam is quantified by its **beam-coupling impedance** — a complex-valued function that encapsulates how each device stores and dissipates electromagnetic energy. The inverse Fourier transform of this impedance yields the **wake function**, describing the time-domain interaction between successive charged particles.

Accurate computation of the impedance and wake function is essential for:
- Predicting **collective beam dynamics**, such as instabilities and bunch deformation
- Estimating **power deposition** and guiding the **mechanical design** of beamline components

```{seealso}
* **🎯 Beam dynamic simulations**

To simulate collective beam dynamics and stability using impedance models, you can use the CERN-developed Python package [**Xsuite**](https://github.com/xsuite/xsuite), an open-source macroparticle tracking simulation code suite.
```

```{seealso}
* **💡 Beam Induced Heating**

The **dissipated power** due to beam-induced heating can be estimated using [BIHC](https://github.com/ImpedanCEI/BIHC), a tool within the Wakis ecosystem. It takes as input the impedance profile of a device and the beam spectrum, returning power loss predictions that are critical for **vacuum design**, **cooling**, and **material compliance**.
```



#### Analytical vs Numerical Approaches


Analytical methods offer elegant and accurate solutions for the beam-coupling impedance of **simplified geometries**. These are invaluable for physical insight and quick estimates in idealized cases such as:

- **Resistive wall pipes**
  Analytical expressions exist for round and elliptical beam pipes, characterizing impedance due to finite wall conductivity [Yokoya, 1993](https://cds.cern.ch/record/248630), [Migliorati et al., 2019](https://cds.cern.ch/record/2705426).

- **RF cavities**
  Resonant mode impedance and wake contributions can be calculated using modal expansion or equivalent circuits [Hofmann & Zotter, 1990](https://cds.cern.ch/record/196446), [Jensen, 2014](https://cds.cern.ch/record/1982429).

- **Tapers and step transitions**
  Important for matching sections and bellows, these can be described with slowly varying cross-section approximations [Yokoya, 1990](https://cds.cern.ch/record/210347), [Stupakov, 2007](https://link.aps.org/doi/10.1103/PhysRevSTAB.10.094401), [Palumbo et al., 1994](https://arxiv.org/abs/physics/0309023).


However, real-world accelerator components like:
- Beam Position Monitors (BPMs)
- Injection kickers
- RF-shielded bellows
- Complex collimators or cavities

...often have **no analytical solution**. These must be addressed through **full 3D numerical simulations** of Maxwell’s equations [Maxwell, 1865](https://royalsocietypublishing.org/doi/10.1098/rstl.1865.0008), using finite differences, finite elements, or finite integration.

```{tip}
* **🧑‍🏫 Why Use Time-Domain Solvers?**

Wakis employs a **time-domain approach** using the Finite Integration Technique (FIT), which offers key benefits:
- **Broadband response** in a single simulation
- Natural support for transient excitation (e.g., Gaussian bunch)
- Efficient use of explicit solvers with GPU and MPI support

This makes Wakis well-suited for impedance characterization across a **wide frequency range**, complementing frequency-domain solvers like CST or HFSS.

For a broader overview of impedance modeling, see [Metral et al., 2020](https://cds.cern.ch/record/2743945).
```


## 3. Electromagnetic Formulation

### ⚡🧲 Maxwell's Equations (Integral Form)

Wakis numerically solves Maxwell's equations in their **integral form**, which is fundamental to the Finite Integration Technique (FIT). This approach preserves the physical laws in their conservative form and naturally fits the discretization on structured grids.

The time-domain integral form of Maxwell's equations is:

$$
\begin{align}
\oint_{\partial A} \mathbf{E}\cdot \mathrm{d}\mathbf{s} &= -\iint_{A}\frac{\partial \mathbf{B}}{\partial t}\cdot \mathrm{d}\mathbf{A} \tag{1a}\\[6pt]
\oint_{\partial A} \mathbf{H}\cdot \mathrm{d}\mathbf{s} &= \iint_{A}\left(\frac{\partial \mathbf{D}}{\partial t} + \mathbf{J}\right)\cdot \mathrm{d}\mathbf{A} \tag{1b}\\[6pt]
\iint_{\partial V} \mathbf{B}\cdot \mathrm{d}\mathbf{A} &= 0 \tag{1c}\\[6pt]
\iint_{\partial V} \mathbf{D}\cdot \mathrm{d}\mathbf{A} &= \iiint_{V}\rho\, \mathrm{d}V \tag{1d}\\[6pt]
\mathbf{D} = \varepsilon \mathbf{E},\quad
\mathbf{B} &= \mu \mathbf{H},\quad
\mathbf{J} = \sigma \mathbf{E} + \rho\mathbf{v} \tag{1e}
\end{align}
$$

These laws describe:

- The evolution of electric $\mathbf{E}$ and magnetic $\mathbf{H}$ fields over time via their circulation around surfaces (Eqs. 1a–1b) and fluxes (magnetic flux density $\mathbf{B}$, electric displacement field $\mathbf{D}$)
- The coupling to sources through current density $\mathbf{J}$ and charge density $\rho$ (Eqs. 1b, 1d),
- The absence of magnetic monopoles (Eq. 1c),
- And the constitutive relations of the materials (Eq. 1e), which relate the physical fields to the medium’s electromagnetic properties: permittivity $\varepsilon$, permeability $\mu$, and conductivity $\sigma$, with $\mathbf{v}$ denoting the velocity of moving charges.

In these equations, $\varepsilon$, $\mu$, $\sigma$ can be considered tensors and frequency independent. To account for frequency or time dependency, the multiplication should be exchanged for a convolution $\ast$.

### 🧱 Discretization with the Finite Integration Technique (FIT)

Wakis discretizes the integral form of Maxwell's equations using the **Finite Integration Technique (FIT)** on a structured three-dimensional Cartesian grid.

$$
N_\text{cells} = N_x \times N_y \times N_z
$$

This approach maps:
- Line integrals → to grid **edges**
- Surface integrals → to grid **faces**
- Volume integrals → to grid **cells**


The resulting discretization yields the **Maxwell Grid Equations (MGE)**, which evolve the fields on a **staggered Yee-like lattice**. Specifically:
- $\vec{E}$ and $\vec{H}$ components are stored on **edges**
- $\vec{D}$ and $\vec{B}$ components are defined on **faces**
- Scalar quantities such as charge density reside at **cell centers**

This structure ensures that discrete curl, divergence, and gradient operators obey their continuous counterparts' conservation properties, which is critical for numerical stability and accuracy. By adopting the **Yee staggered grid** formulation and initially charge-free conditions, the divergence equations (1c–1d) are **satisfied implicitly** by construction.


#### Maxwell Grid Equations (MGE)

Following the FIT numerical method, the continuous Maxwell equations are converted into discrete update rules for the electric and magnetic fields:

$$
\begin{align}
\mathbf{C}\mathbf{D}_s \, \mathbf{e} &= -\mathbf{D}_A \, \frac{\partial (\mathbf{M}_{\mu} \mathbf{h})}{\partial t}  \tag{2a} \\[6pt]
\widetilde{\mathbf{C}}\widetilde{\mathbf{D}}_s \, \mathbf{h} &= \widetilde{\mathbf{D}}_A \left( \frac{\partial (\mathbf{M}_{\varepsilon} \mathbf{e})}{\partial t} + \mathbf{M}_{\sigma} \mathbf{e} + \mathbf{j}_{\text{src}} \right)  \tag{2b} \\[6pt]
\end{align}
$$

Where:
- $\mathbf{C}$ is the discrete **curl matrix**
- $\mathbf{C}^T$ is its transpose (used for magnetic curl)
- $\mathbf{D}_s$, $\widetilde{\mathbf{D}}_s$, $\mathbf{D}_A$, and $\widetilde{\mathbf{D}}_A$ are diagonal matrices representing cell edge lengths and face areas in the primal and dual~($\sim$) grids.
- The electromagnetic fields $\mathbf{e}, \mathbf{h}, \mathbf{j}$ are stored in memory as **1D vectors** of length $\{3N_\text{cells}\}$ stored in **lexicographic order**, encapsulated in the `Field` class.


The numerical method is implemented in `SolverFIT3D` class. The curl matrices are $\{3N_{cells}\times3N_{cells}\}$  sparse matrices with bands of +1 and -1. They are implemented efficiently in Wakis using `scipy.sparse` CSR format. The diagonal matrices also benefit from the `scipy.sparse.diags` object for optimized storage.

The electromagnetic fields are stored in a `numpy`-based `Field` object in Wakis:
- Supports `.toarray()` and `.fromarray()` for optimized modification during the time-stepping
- `.from_matrix()`, `.to_matrix()` to go from 1d to 3d matrix,automatically reshaped to the simulation grid dimensions.
- Interoperates with CuPy and MPI through magic methods and flags.
- `.inspect()` and other handy plotting methods
- Custom magic methods for multiplication, addition, division: `__div__`, `__add__`, `__mul__`
- `__getitem__`, `__setitem__` Getters and setters to access the 3D coordinates on-the-fly: apply intial conditions, sources, save states...

```{tip}
Thank's to the `Field` class, fields are stored in memory-continuous arrays for optimized performance, but can be accessed as a 3D matrix.
```

Some examples on how to access and operate `Field`s:
```python
# modify field slice in z-direction
solver.E[100, 20:30, :, 'z'] = 0.

# access cell value of the 123456th cell in lexico-graphic order
solver.H[123456]

# sum or multiply two field objects, keeping the Field
E_tot = solver_1.E + solver_2.E

# Calculate the energy (T_00)
T_00 = 0.5*(solver.E.get_abs()**2 + solver.H.get_abs()**2)

# Inspect the 3 components of the field in one line
solver.J.inspect()
```

This formulation enables stable, explicit time stepping and modular extensions to lossy media, materials, and sources.

#### Material tensors and grid information

Wakis distinguishes between **primal** and **dual** grid geometries as part of its Finite Integration Technique (FIT) formulation. The grid operations are implemented in Wakis' `GridFIT3D` class. Each quantity is mapped to a geometric entity and stored as a sparse diagonal matrix to enable fast, memory-efficient computations:

| Quantity                  | Description                                    |
|--------------------------|------------------------------------------------|
| $\mathbf{M}_\varepsilon^{-1}$ | Diagonal matrix of inverse permittivities              |
| $\mathbf{M}_\mu^{-1}$    | Diagonal matrix of inverse permeabilities      |
| $\mathbf{M}_\sigma$      | Diagonal matrix of electrical conductivities   |
| $\mathbf{D}_s$, $\widetilde{\mathbf{D}}_s$, $\mathbf{D}_A$, and $\widetilde{\mathbf{D}}_A$ | Edge lengths and face areas (primal/dual) |

To support **anisotropic materials** and **imported geometries**, Wakis stores the raw material data in structured `Field` objects — similar to 3D tensors — where values can be specified independently along the **x, y, and z directions** for each cell.

Before time-stepping, these directional fields are assembled into the corresponding **sparse diagonal matrices** using `scipy.sparse.diags`. This preserves the **locality of FIT updates** while enabling efficient CPU and GPU execution.

Wakis also supports spatially varying media, embedded materials from CAD imports, and subpixel smoothing, ensuring accurate representation of complex geometries and composite structures.

### 🕒 Time-Stepping Routine

Wakis uses the **Leapfrog scheme**, a second-order accurate and explicit time integrator. This method updates the magnetic and electric fields in a staggered fashion:

$$
\begin{align}
\mathbf{h}^{n+1} &= \mathbf{h}^n - \Delta t \, \widetilde{\mathbf{D}}_s \, \mathbf{M}_\mu^{-1} \, \mathbf{D}_A^{-1} \, \mathbf{C} \, \mathbf{e}^{n+0.5} \tag{3a} \\[6pt]
\mathbf{e}^{n+1.5} &= \mathbf{e}^{n+0.5} + \Delta t \, \mathbf{D}_s \, \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \widetilde{\mathbf{D}}_A^{-1} \, \widetilde{\mathbf{C}} \, \mathbf{h}^n
- \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \mathbf{j}_{\text{src}}^n
- \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \widetilde{\mathbf{M}}_\sigma \, \mathbf{e}^{n+0.5} \tag{3b}
\end{align}
$$

Where:
- $\mathbf{C}$ and $\widetilde{\mathbf{C}}$ are discrete curl matrices on the primal and dual grids
- $\mathbf{D}_s$, $\mathbf{D}_A$ are diagonal matrices containing metric coefficients (edge lengths and face areas)
- Material properties enter through $\mathbf{M}_\mu$, $\mathbf{M}_\varepsilon$, and $\mathbf{M}_\sigma$

The timestep $\Delta t$ is constrained by:
- The **Courant–Friedrichs–Lewy (CFL)** condition
- Material relaxation times (for dispersive or lossy media)

Most matrix operations are precomputed and cached, enabling large-scale simulations with modest memory usage (~20M cells in <8 GB RAM/GPU).

### 🔌 Sources and Initial Conditions

Wakis allows arbitrary initial conditions on $\vec{E}$, $\vec{H}$, and $\vec{J}$. Sources can be defined in multiple ways:
- **User-defined time-dependent callbacks** placed after each step
- **Predefined source types**: Gaussian beams, dipoles, plane waves, laser pulses, available in `sources.py`

```{tip}
#### ⚙️ Source callbacks

A source callback can be easily created as:
- A function like `def update(solver, time)` placed in a `for` loop after each step `solver.one_step()`.
- a class with the method `Source.update(solver, time)`, passed to the `solver.emsolve()` routine.
```

#### Example: Gaussian Beam Current $J_z$

A rigid Gaussian bunch current is modeled as a line distribution:

$$
\mathbf{J}_z(x_{\text{src}}, y_{\text{src}}, \vec{z}) =
\frac{q \beta c}{\sqrt{2\pi} \sigma_z} \,
\exp\left( -\frac{(\vec{s} - s_0)^2}{2\sigma_z^2} \right)
$$

with:
- $\vec{s} = \vec{z} - \beta c t$: beam-frame coordinate
- $s_0 = z_{\min} - \beta c t_{\text{inj}}$: center of bunch
- $q$ the charge in $\text{nC}$
- $\sigma_z$ the bunch length in $\text{m}$

This supports both **ultra-relativistic** ($\beta \approx 1$) and **low-beta** scenarios.

To avoid parasitic fields interacting with the Perfect Matched Layer (PML) boundary, particle beams are only injected into the physical domain, excluding the PML region. To mitigate the divergence error fields that arise from this truncation, a Total-Field/Scattered-Field (TF/SF) formulation is applied [M.C. Balk et al., 2006].

The computational domain is divided into total-field and scattered-field regions, with the TF/SF boundary offset by one cell from the CPML interface. The beam is injected strictly within the total-field region, where the total field is defined as:

$$E^t = E^s + E^i$$

At the boundary, the curl operator is modified to add or subtract the incident fields, rendering the boundary transparent to field propagation. For ultrarelativistic beams, incident transverse electric fields are pre-calculated using a 2D Poisson solver and dynamically scaled by the instantaneous longitudinal current density during time integration.

#### Soft current sources: `CurrentSource`

Sources that overwrite a field value every timestep (*hard* sources, e.g. `Dipole`, `Pulse` or `PlaneWave` with `field='E'`) pin the field at the source cells. Any wave scattered back onto them is reflected as if by a small obstacle, which spoils antenna, scattering and resonance calculations. A *soft* source instead prescribes an impressed electric current density $\mathbf{J}_s$, which only adds a forcing term to Ampère's law [Taflove & Hagness, 2005; Oskooi & Johnson, 2013]:

$$
\widehat{\mathbf{e}}^{\,n+1} = \widehat{\mathbf{e}}^{\,n} + \Delta t\, \mathbf{M}_{\varepsilon}^{-1}\left(\widetilde{\mathbf{C}}\,\widehat{\mathbf{h}}^{\,n+1/2} - \mathbf{j}_s^{\,n+1/2}\right)
$$

The field equations stay linear in $\mathbf{J}_s$, so the source is transparent to incoming fields and the fields of several sources superpose. `CurrentSource` evaluates its time profile at $t + \Delta t/2$, the time level at which $\mathbf{j}$ enters the update. It adds only the change of its current since the previous step, so it coexists with the ohmic current $\sigma\mathbf{E}$ and with other sources sharing the same edges.

**Normalization.** The amplitude is integrated over the directions in which the source has zero size. For a point source along $c$ on edge $i$, the current density is

$$
J_{s,c} = \frac{I\ell\; s(t)}{L_{c,i}\,\widetilde{A}_{c,i}},
$$

where $I\ell$ [A m] is the current moment of a Hertzian dipole, $L_{c,i}$ is the primary edge length and $\widetilde{A}_{c,i}$ the dual facet area pierced by the edge, both taken from the (possibly non-uniform) mesh. The radiated fields are therefore independent of the mesh. In free space such a dipole radiates [Jackson, 1999, Ch. 9]

$$
P = \frac{\eta_0 k^2 |I\ell|^2}{12\pi},
$$

which Wakis reproduces to second order in the mesh size; see `tests/test_017_current_sources.py`. Line, sheet and volume sources have amplitudes in A, A/m and A/m², respectively. An optional `amp_func(x, y, z)` shapes the spatial profile, e.g. a mode profile or a phase gradient.

**Time profiles.**
- `GaussianSource(frequency, fwidth)`: $s(t) = \frac{1}{\omega_0}\frac{d}{dt}\left[g(t)\sin\omega_0\tau\right]$ with $g(t)=e^{-\tau^2/2\sigma_t^2}$, $\sigma_t = 1/(2\pi f_w)$, $\tau = t - t_0$. Its spectrum is Gaussian with standard deviation $f_w$, and `fourier_transform(f)` returns it analytically for normalizing frequency-domain results. Taking the time derivative makes $\int s\,dt = 0$. A Gaussian-modulated cosine would deposit a small net charge, leaving a static dipole field that never decays and corrupts Fourier transforms of the fields.
- `ContinuousSource(frequency, ramp_time)`: a sinusoid switched on with a $\sin^2$ ramp to avoid broadband transients.
- `CustomSource(func)`: an arbitrary user-defined $s(t)$.

`SolverFIT3D.emsolve` accepts a single source or a list of sources.

References:
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).
- A. Oskooi and S. G. Johnson, "Electromagnetic wave source conditions", in *Advances in FDTD Computational Electrodynamics*, Artech House (2013), [arXiv:1301.5366](https://arxiv.org/abs/1301.5366).
- J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999).

### 📡 Frequency-domain monitors: `DFTMonitor` and `FluxMonitor`

Frequency-domain quantities are obtained from a single broadband run by accumulating running discrete Fourier transforms of the time-domain fields [Taflove & Hagness, 2005; Oskooi et al., 2010]:

$$
\mathbf{F}(f) = \sum_n \mathbf{F}(t_n)\, e^{-2\pi i f t_n}\, \Delta t ,
$$

which approximates the continuous transform with the same convention as `GaussianSource.fourier_transform`, so spectra can be divided by the source spectrum directly. Each sample is transformed at its own position and time level: after the step starting at $t = n\Delta t$, the electric field (and $\mathbf{J}$) is known at $t_n = (n+1)\Delta t$ and the magnetic field at $t_n = (n+\tfrac12)\Delta t$. Because radiated power is the small real part of a large reactive near-field quantity, mis-centring by half a step introduces errors of order $\omega \Delta t$ times the reactive power.

`DFTMonitor(frequencies, components, center, size)` stores the transform of any $E$, $H$ or $J$ component on a point, line, plane or volume, at the components' native staggered locations.

`FluxMonitor(frequencies, center, size)` computes the Poynting flux spectrum through a plane (one zero `size` entry; positive along $+\hat{n}$) or out of a closed box (all entries non-zero; positive outwards):

$$
\Phi(f) = \operatorname{Re}\sum_{\text{faces}} \sum_{\text{samples}} \left[E_b(f)\, H_c^*(f) - E_c(f)\, H_b^*(f)\right] \Delta A ,
$$

for a face normal to $a$ with tangential directions $(b, c)$ in cyclic order. The tangential $E$ samples lie on the grid plane nearest to the requested position. The $H$ components, which share their in-plane position with the $E$ component they multiply but are staggered half a cell along the normal, are averaged from both sides onto that plane. The area element of each product is the primary edge length of the $E$ component times the dual width in the other tangential direction, the same metric used by the curl operators. On a closed box, samples on the edges shared by two faces receive half weight (trapezoidal rule); without this, the energy is over-counted by several percent for small boxes.

By Parseval's theorem the energy crossing the surface is $2\int_0^\infty \Phi(f)\,df$. For a source with spectrum $S(f)$, the time-averaged power at frequency $f$ under harmonic excitation of unit amplitude is $P(f) = \Phi(f)/(2|S(f)|^2)$. The tests in `tests/test_018_flux_monitors.py` verify that:
- the flux out of a box around a dipole equals the work done by the source, $-\operatorname{Re}[E(f)\,I\ell^*(f)]$, to 0.1% (discrete Poynting theorem), independently of the box size;
- it matches the Hertzian dipole power to second order in the mesh size;
- slab transmission and reflection follow the Airy formula with $R+T=1$ to $10^{-5}$.

With MPI, each rank accumulates the samples in its own subdomain. `FluxMonitor.get_flux` sums over ranks, whereas `DFTMonitor.get_field` returns rank-local samples.

References:
- A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and S. G. Johnson, "MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method", *Comput. Phys. Commun.* **181**, 687 (2010), [doi:10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).

### 🔀 Magnetic currents, waveguide modes and mode ports

#### Magnetic current sources: `MagneticCurrentSource`

An impressed magnetic current density $\mathbf{M}$ [V/m²] enters Faraday's law as the dual of $\mathbf{J}$ in Ampère's law [Harrington, 1961]:

$$
\mu\,\partial_t \mathbf{H} = -\nabla\times\mathbf{E} - \mathbf{M},
\qquad
\widehat{\mathbf{h}}^{\,n+1/2} = \widehat{\mathbf{h}}^{\,n-1/2} - \Delta t\, \mathbf{M}_{\mu}^{-1}\left(\mathbf{C}\,\widehat{\mathbf{e}}^{\,n} + \mathbf{m}_s^{\,n}\right).
$$

The leapfrog H update from $(n-\tfrac12)\Delta t$ to $(n+\tfrac12)\Delta t$ is centred at $n\Delta t$, so $\mathbf{M}$ is sampled at $t = n\Delta t$ (the time passed to `source.update`), whereas $\mathbf{J}$ is sampled at $t + \Delta t/2$. $\mathbf{M}$ lives on the H samples (on the node along its own direction, at cell midpoints transversally). `MagneticCurrentSource` mirrors `CurrentSource`: the amplitude is integrated over the zero-size directions with the dual edge length along the current and the primary widths transversally, so a point, line, sheet and volume source have units of V m, V, V/m and V/m². `solver.M` is allocated by the first magnetic source; otherwise the H update is unchanged.

**Huygens sheets.** By the equivalence principle [Harrington, 1961; Taflove & Hagness, 2005, Ch. 5], the surface currents $\mathbf{K} = \hat{\mathbf{n}}\times\mathbf{H}_{inc}$ and $\mathbf{M}_s = -\hat{\mathbf{n}}\times\mathbf{E}_{inc}$ on a surface radiate the incident field on the side $\hat{\mathbf{n}}$ points to and cancel it on the other side. On the staggered grid this is exactly the total-field/scattered-field correction: $\mathbf{K}$ sits on the E plane $z_0$ and carries $\mathbf{H}_{inc}$ of the H plane $z_0 - \Delta z/2$ at time $t+\Delta t/2$, and $\mathbf{M}_s$ sits on that H plane and carries $\mathbf{E}_{inc}(z_0, t)$. For a plane wave $E_x = s(t - z/c)$ along $+z$ this gives $K_x(t) = -s(t + \Delta t/2 + \Delta z/2c)/\eta_0$ and $M_{s,y}(t) = -s(t)$. `tests/test_023_modes.py` checks that such a sheet radiates a broadband pulse with backward/forward flux below $10^{-3}$ (observed: $\sim10^{-11}$–$10^{-9}$ with CPML, $\le 3\times10^{-6}$ with Mur ABC, limited by the boundary reflection) for 35–170 cells per wavelength.

#### Waveguide mode solver: `solve_modes`

For a guide that is uniform along the normal $a$ of a grid plane (transverse axes $b, c$ with $\hat b\times\hat c = \hat a$), fields of the 3D scheme vary as $e^{i(\omega t - \beta a)}$. The forward difference of a sample on the $a$-nodes and the backward difference of a sample on the $a$-midpoints then both reduce to $-i\tilde\beta$, with

$$
\tilde\beta = \frac{2}{\Delta a}\sin\frac{\beta\Delta a}{2},\qquad
\tilde\omega = \frac{2}{\Delta t}\sin\frac{\omega\Delta t}{2},
$$

when the half-cell phase is absorbed in the amplitudes of the midpoint samples ($E_a$, $H_b$, $H_c$), and the leapfrog time derivative reduces to $i\tilde\omega$ [Taflove & Hagness, 2005, Ch. 4]. With $U_b, U_c$ the primary forward differences (node to midpoint, divided by $L$) and $V_b, V_c$ the dual backward differences (divided by $\tilde L$), exactly as in the curl matrices $\mathbf{C}$ and $\widetilde{\mathbf{C}}$, eliminating $E_a = (V_b H_c - V_c H_b)/(i\tilde\omega\varepsilon_a)$ and $H_a = -(U_b E_c - U_c E_b)/(i\tilde\omega\mu_a)$ gives

$$
\tilde\beta\begin{pmatrix}H_b\\H_c\end{pmatrix} = Q\begin{pmatrix}E_b\\E_c\end{pmatrix},\quad
Q = \frac{1}{\tilde\omega}\begin{pmatrix} V_b\mu_a^{-1}U_c & -\tilde\omega^2\varepsilon_c - V_b\mu_a^{-1}U_b\\ \tilde\omega^2\varepsilon_b + V_c\mu_a^{-1}U_c & -V_c\mu_a^{-1}U_b\end{pmatrix},
$$

$$
\tilde\beta\begin{pmatrix}E_b\\E_c\end{pmatrix} = P\begin{pmatrix}H_b\\H_c\end{pmatrix},\quad
P = \frac{1}{\tilde\omega}\begin{pmatrix} -U_b\varepsilon_a^{-1}V_c & \tilde\omega^2\mu_c + U_b\varepsilon_a^{-1}V_b\\ -\tilde\omega^2\mu_b - U_c\varepsilon_a^{-1}V_c & U_c\varepsilon_a^{-1}V_b\end{pmatrix},
$$

and the eigenproblem $\tilde\beta^2\,\mathbf{e} = PQ\,\mathbf{e}$ for the transverse E samples. This is the Yee-mesh finite-difference mode solver of Zhu & Brown [2002], here built from Wakis' own material tensors (`ieps`, `imu`), metric lengths ($L$, $\tilde L$) and boundary masks: PEC (and the PEC-terminated CPML/PML faces) removes the tangential E samples on the walls, PMC removes tangential H, periodic axes wrap, conductors with `ieps = 0` remove their E samples, and the ghost layer at the last index is excluded. A port `size` smaller than the domain puts electric walls on the port edges (e.g. to exclude CPML layers). The eigenvalues closest to $n_{max}^2\tilde\omega^2/c^2$ are found with shift-invert Arnoldi (`scipy.sparse.linalg.eigs`), and $\beta = (2/\Delta a)\arcsin(\tilde\beta\Delta a/2)$. Because the same discrete operators are used, a mode is an exact eigenmode of the 3D time stepping of a uniform guide: the TE10 propagation constant matches the discrete dispersion relation $(\tilde\omega/c)^2 = \tilde k_x^2 + \tilde\beta^2$ to $10^{-9}$ and the continuum $\sqrt{k^2-(\pi/a)^2}$ to grid-dispersion accuracy; the TE0 effective index of a dielectric slab converges at second order to the solution of $\tan(\kappa d/2) = \gamma/\kappa$ [Snyder & Love, 1983].

**Power normalization.** The `FluxMonitor` (and the discrete Poynting theorem of the Yee scheme) pairs the E plane with H averaged from the two neighbouring H planes, which for a mode is $\bar{\mathbf{h}} = \mathbf{h}\cos(\beta\Delta a/2)$. Modes are normalized to unit time-averaged power,

$$
\tfrac12\operatorname{Re}\sum \left(e_b\,\bar h_c^* - e_c\,\bar h_b^*\right)\Delta A = 1\ \text{W},
$$

with the `FluxMonitor` area elements. Note that, because of the ghost layer, PEC walls on a non-periodic axis sit on the nodes $0$ and $N-1$: a guide meshed with $N$ cells has width $(N-1)\Delta$.

Assumptions and limits: lossless isotropic materials uniform along $a$ within one cell of the plane, a locally uniform mesh along $a$, and serial (non-MPI) runs. Only propagating modes ($\tilde\beta^2 > 0$) are returned.

#### Mode source and mode decomposition: `ModeSource`, `ModeMonitor`

`ModeSource` is a Huygens sheet whose incident field is the mode $(\mathbf{e}, \mathbf{h})$ computed at the source frequency, with the time profile $s(t)$: $\mathbf{K} = \hat{\mathbf{a}}\times\mathbf{h}\,s(t + \Delta t/2 + \tau)$ on the E plane and $\mathbf{M}_s = -n\,\hat{\mathbf{a}}\times\mathbf{e}\,s(t)$ on the H plane behind it, with $n=\pm1$ the direction and $\tau = \beta\Delta a/(2\omega)$ the phase delay of the half cell between the planes [Oskooi & Johnson, 2013]. At the mode frequency the injection is exact to round-off; away from it the mode profile, the wave impedance and the phase velocity change, and a backward wave of relative amplitude $\sim |\Delta Z/2Z|$ leaks (e.g. TE10 with a mode at $1.5f_c$: $3\times10^{-5}$ at $1.5f_c$, $1.4\times10^{-2}$ at $1.45 f_c$, $8\times10^{-2}$ at $1.3f_c$). The launched transverse field is $\mathbf{E}_t = A\,s(t)\,\mathbf{e}$, so the forward mode amplitude is $a_+(f) = A\,S(f)$ and a time-harmonic source of amplitude $A$ [√W] carries $A^2$ W.

`ModeMonitor` projects the Fourier-transformed transverse fields on a plane ($\mathbf{H}$ averaged as in `FluxMonitor`) on the modes solved at each monitored frequency, using the orthogonality relation of lossless guides $\tfrac12\int \mathbf{e}_m\times\bar{\mathbf{h}}_n^*\cdot\hat{\mathbf{a}}\,dA = \delta_{mn}$ [Snyder & Love, 1983; Pozar, 2012]:

$$
a_\pm = \frac14\sum\left[\left(\mathbf{E}\times\bar{\mathbf{h}}^*\right)\pm\left(\mathbf{e}^*\times\bar{\mathbf{H}}\right)\right]\cdot\hat{\mathbf{a}}\,\Delta A,
$$

so that $\mathbf{E}_t = \sum (a_+ + a_-)\,\mathbf{e}$ and $\bar{\mathbf{H}}_t = \sum (a_+ - a_-)\,\bar{\mathbf{h}}$. The amplitudes have units of √W·s; $|a_\pm|^2/|S|^2$ is the time-harmonic power in the mode and a single forward mode gives the `FluxMonitor` flux $\Phi = 2|a_+|^2$ exactly. S-parameters between ports and modes follow as ratios of amplitudes, e.g. $S_{21} = a_+^{(2)}/a_+^{(1)}$.

Validation (`tests/test_023_modes.py`, PEC guide $a = 20\Delta$, CPML ends): downstream $|a_+|^2 = |S|^2$ within $0.7\%$ over $1.3$–$1.7 f_c$ ($5\times10^{-5}$ at the mode frequency), flux $= 2|a_+|^2$ to $10^{-8}$, residual backward amplitude (CPML reflection) $<1.3\times10^{-4}$; TE20 injection excites no TE10.

**Differentiability.** All terms are linear in the fields: the M term transposes like the J term ($\mathbf{h} \mathrel{-}= \Delta t\,\mathbf{M}_\mu^{-1}\mathbf{m}$, adjoint source on the adjoint H update), and $a_\pm$ are linear functionals of the DFT fields with fixed weights $\tfrac14\bar{\mathbf{h}}^*\Delta A$ and $\tfrac14\mathbf{e}^*\Delta A$, so $\partial|a_\pm|^2/\partial\mathbf{E}^* $ and $\partial|a_\pm|^2/\partial\mathbf{H}^*$ give the adjoint DFT sources directly. The dependence of the mode profile on the permittivity of the port plane is not differentiated.

References:
- R. F. Harrington, *Time-Harmonic Electromagnetic Fields*, McGraw-Hill (1961).
- Z. Zhu and T. G. Brown, "Full-vectorial finite-difference analysis of microstructured optical fibers", *Opt. Express* **10**, 853 (2002), [doi:10.1364/OE.10.000853](https://doi.org/10.1364/OE.10.000853).
- A. W. Snyder and J. D. Love, *Optical Waveguide Theory*, Chapman & Hall (1983).
- D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley (2012).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), Ch. 4 (numerical dispersion) and Ch. 5 (incident wave source conditions, TF/SF).
- A. Oskooi and S. G. Johnson, "Electromagnetic wave source conditions", arXiv:1301.5366 (2013).

### 🧊🔚 Boundary Conditions

Wakis supports several boundary condition (BC) types:
- **PEC (Perfect Electric Conductor)**: masks tangential electric-field degrees of freedom, enforcing $\vec{E}_{\parallel} = 0$ at the selected face.
- **PMC (Perfect Magnetic Conductor)**: masks tangential magnetic-field degrees of freedom, enforcing $\vec{H}_{\parallel} = 0$.
- **Periodic**: pairs the low and high faces of an axis and closes the corresponding FIT derivative stencil (`Px`, `Py`, or `Pz`) across that seam. The curl matrix is then rebuilt from the corrected derivative matrices and the periodic dual metrics are updated. A periodic face must be paired with the opposite face. Serial periodic boundaries are available in all three directions; longitudinal periodic MPI requires cyclic ghost exchange and is intentionally not enabled yet.
- **ABC**: a first-order Mur radiation condition for selected domain faces.
- **PML**: a finite, graded, electrically lossy layer terminated by the usual electric boundary mask.
- **CPML**: a convolutional absorbing layer that adds auxiliary curl-correction fields.

#### Mur ABC implementation

The `abc` boundary applies the first-order Mur radiation condition to the
tangential electric field on any selected low or high face. It approximates the
one-way wave equation normal to that face,

$$
\frac{\partial E_t}{\partial t} + v\frac{\partial E_t}{\partial n} = 0,
$$

where $E_t$ is a tangential electric-field component, $n$ is the outward face
normal, and the wave speed is taken from the homogeneous background material,

$$
v = \frac{1}{\sqrt{\varepsilon_{\mathrm{bg}}\mu_{\mathrm{bg}}}}.
$$

For a low face, Wakis updates the boundary plane after the usual electric and
magnetic field step as

$$
E_t^{n+1}(0) = E_t^n(1) + r_{\mathrm{lo}}
\left[E_t^{n+1}(1) - E_t^n(0)\right],
$$

and, for the corresponding high face,

$$
E_t^{n+1}(N_z-1) = E_t^n(N_z-2) + r_{\mathrm{hi}}
\left[E_t^{n+1}(N_z-2) - E_t^n(N_z-1)\right].
$$

The face-local coefficient uses the adjacent normal cell spacing, $\Delta x$,
$\Delta y$, or $\Delta z$ as appropriate,

$$
r = \frac{v\Delta t - \Delta z}{v\Delta t + \Delta z}.
$$

Only the previous boundary and adjacent-interior tangential-electric-field
planes are stored, so the additional memory scales with the active boundary
area rather than the domain volume. This implementation leaves the FIT curl
topology and metric operators unchanged. It assumes a homogeneous background
material at the absorbing boundary and cannot be combined with PML or CPML
faces. When ABC faces are selected on more than one axis, they share edges;
Wakis emits a warning because those edge values currently follow the explicit
`x`, `y`, then `z` face-update order. It is most appropriate for waves close to
normal incidence. CPML remains the preferred absorber for oblique, broadband,
low-frequency, or evanescent fields.

#### PML implementation

An ideal PML is impedance matched to the adjacent medium while attenuating outgoing waves. For an interface between media A and B, the reflection coefficient is

$$
\Gamma = \frac{\eta_B - \eta_A}{\eta_B + \eta_A},
$$

where $\eta = \sqrt{\mu/\varepsilon}$ is the wave impedance. In a matched electric-and-magnetic lossy medium, the damping rates satisfy

$$
\frac{\sigma_{\mathrm{el}}}{\varepsilon}
= \frac{\sigma_{\mathrm{mag}}}{\mu}.
$$

This preserves the impedance while introducing attenuation. Equivalently, a PML can be formulated through complex coordinate stretching [Gedney et al., 2000]:

$$
s = 1 + \frac{\sigma}{j\omega\varepsilon_0}.
$$

The current Wakis `pml` boundary is a deliberately simple adiabatic absorber rather than a complete matched, stretched-coordinate PML. It adds a geometrically graded artificial electric conductivity to the PML cells and uses the regular electric-current update. The inverse permittivity in those cells is set to the vacuum value, and no corresponding artificial magnetic conductivity is applied. Consequently, its effectiveness depends on layer thickness and the low-conductivity ramp; it is not expected to be reflectionless, especially for oblique incidence, broadband pulses, or non-vacuum material at the interface.

#### CPML implementation

Convolutional PML (CPML) improves absorption of grazing-incidence, low-frequency, and evanescent fields by using a complex-frequency-shifted coordinate stretch [Gedney et al., 2000]:

$$
s = \kappa + \frac{\sigma}{\alpha + j\omega\varepsilon_0}.
$$

Here $\kappa \geq 1$ scales the coordinate stretch, $\sigma$ is the graded conductivity, and $\alpha$ is a non-negative frequency-shift parameter. Wakis constructs $\sigma$, $\kappa$, and $\alpha$ on both primal and dual field locations with a polynomial distance profile. It converts them into recursion coefficients and stores only the CPML auxiliary fields needed to correct curl terms inside each boundary layer.

CPML is therefore distinct from the scalar `pml` path: it does not modify the bulk material conductivity tensor to absorb waves, and its auxiliary convolution fields make it the preferred absorbing boundary for the beam and TF/SF workflows. As with all finite layers, its practical reflection level still depends on the number of cells and the selected profile parameters.


### 📥🗿 Geometry Importing & Embedded Boundaries

Wakis integrates with [**PyVista**](https://docs.pyvista.org/) to import CAD geometries in `.STL`, `.STEP`, or `.OBJ` formats. The mesh is overlaid onto the simulation domain and mapped onto the Cartesian grid using:
- `pyvista`'s surface collision algorithm, based on VTK optimized ray-tracing, allows to detect where the input geometry intersects the primal and dual grids.
- Assignment of material properties ($\varepsilon_r$, $\mu_r$, $\sigma$) in $x$, $y$, and $z$ to the intersected cells using a first-order subpixel smoothing, inspired by the open-source solver MEEP (MIT).

| STL geometry | Imported material mask |
| ----- | ---- |
|  ![](img/STL_solid.png)     |  ![](img/STL_mask.png)    |

Future versions aim to include a more advanced meshing algorithm for improved fidelity near corners and edges.

### Surface impedance boundary condition (SIBC)

When the metal walls are very good conductors and the skin depth is much smaller than the grid spacing, it becomes inefficient (and inaccurate) to resolve the fields inside the conductor volume. Instead of using a volumetric conductivity, Wakis can model such metals with a **Leontovich surface impedance boundary condition (SIBC)** applied only on the surface cells. This can be disabled in the solver by turning `use_sibc=False`

**🔍 Algorithm explanation**

Starting from the imported solid mask of the full volume, our algorithm tags the **surface of solid regions** by computing the gradient magnitude of a scalar field (e.g. material mask) and converting it to a boolean mask. It uses [PyVista's Gradient filter](https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.compute_derivative):

The surface is identified where the gradient of the scalar field is non-zero:

$\|\nabla \phi\| = \sqrt{ (\partial_x \phi)^2 + (\partial_y \phi)^2 + (\partial_z \phi)^2 }$

Cells where $\|\nabla \phi\| > 0$ are marked as surface.

| SIBC  float  | SIBC bool                                                                             | No SIBC                                                                             |
| --- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
|   ![](img/sibc_float.png)  | ![](img/sibc_bool.png) | ![](img/no_sibc.png) |

**See also:**
<details>
<summary> Maximum conductivity resolvable w/o a Leontovich boundary</summary>

To accurately model a conducting material volumetrically in FIT/FDTD, the **skin depth**

$$\delta = \sqrt{\frac{2}{2\pi f_{\max}\,\mu_0\,\sigma}}$$

must be resolved by the grid. A common requirement is at least **3 cells per skin depth**, i.e.

$$\delta \gtrsim 3\Delta_n$$

where $(\Delta_n)$ is the grid spacing normal to the wall, considered as $\sqrt{2}\cdot \text{min}(\Delta x, \Delta y, \Delta z)$. Substituting the skin-depth expression and solving for $\sigma$ gives the maximum conductivity that can be volumetrically resolved at a target frequency $f_{\max}$:

$$\sigma_{\max} = \frac{10}{\pi f_{\max}\,\mu \Delta_n^2}$$

Materials with $\sigma > \sigma_{\max}$ have a skin depth too small to be captured by the mesh and should be modeled using a **Leontovich (surface impedance) boundary condition** instead of volumetric conductivity.
</details>


<details>
<summary> Surface parameters for the Leontovich (SIBC) boundary</summary>

For a good conductor, the Leontovich surface impedance has the form

$$Z_s = (1+j)\sqrt{\dfrac{\omega\mu}{2\sigma}}$$

meaning that the resistive and reactive parts are equal (a $45^\circ$ phase).
In the SIBC implementation, we approximate $Z_s$ by its magnitude-based form

$$Z_s \approx \sqrt{\pi f_{\max}\mu / \sigma}$$

and define an equivalent **surface admittance**

$$Y_s = 1/Z_s$$

To embed this into the time-domain FIT update, we convert $Y_s$ into effective
surface parameters:
- **surface conductivity:** $\sigma_s = 1/Z_s$
- **surface permittivity term:** $\varepsilon_s = 1/Z_s$

Both appear with the same magnitude because the $45^\circ$ impedance angle implies
equal real and imaginary parts in the admittance. These values are applied only to
boundary edges or cells marked as SIBC, replacing the volumetric response of the metal.

</details>


### 🚀 GPU and MPI Parallelization

Wakis supports heterogeneous architecture computing thanks to open-source packages like:
- **GPU acceleration** using [**CuPy**](https://cupy.dev/) and `cupyx.scipy.sparse`
- Drop-in replacement of NumPy/SciPy operations when `use_gpu=True`
- **MPI parallelization** using [**mpi4py**](https://mpi4py.readthedocs.io/)
- Efficient longitudinal domain decomposition with ghost-cell synchronization
- Seamless integration with **multi-GPU** setups using both `cupy` and `mpi4py` memory passing protocols

```{note}
#### 👩‍💻 Developer Notes about Wakis
- Fully open-source and available on [GitHub](https://github.com/ImpedanCEI/wakis)
- Packaged on [PyPI](https://pypi.org/project/wakis/)
- Documented with `Sphinx` and hosted on `ReadTheDocs`: [https://wakis.readthedocs.io/](https://wakis.readthedocs.io/)
- Includes **CI/CD**, with end-to-end tests running nightly on GitHub actions, tagged **versioned releases**, and numerous **ready-to-run examples** in both Python scripts and notebooks, inluding a dedicated [playground](https://github.com/ImpedanCEI/CEI-logo) repository.
```

## 4. Wake potential, impedance, and wake factors

Wakis computes beam coupling impedance from time-domain electromagnetic field simulations by evaluating the wakefields generated by a moving charged particle (or bunch) as it traverses an accelerator structure.

### 📚 Wake function and wake potential

Wakis uses $s>0$ for a test particle behind the source and $v=\beta c$ for
the beam velocity. The transverse source and test positions are
$\mathbf r_s=(x_s,y_s)$ and $\mathbf r_t=(x_t,y_t)$.

The **wake function** $\mathbf w(\mathbf r_s,\mathbf r_t,s)$ is the response
of the structure to a point source. It is a Green function normalized by the
source charge $q_s$:

$$
\mathbf w(\mathbf r_s,\mathbf r_t,s)
=\frac{1}{q_s}\int_{-\infty}^{\infty}
\left[\mathbf E(\mathbf r_t,z,t)
+v\,\mathbf e_z\times\mathbf B(\mathbf r_t,z,t)\right]
_{t=(z+s)/v}\,dz .
$$

Its longitudinal component contains only $E_z$,

$$
w_\parallel(\mathbf r_s,\mathbf r_t,s)
=\frac{1}{q_s}\int_{-\infty}^{\infty}
E_z(\mathbf r_t,z,t=(z+s)/v)\,dz ,
$$

while the transverse component contains the transverse Lorentz force. Both
are quoted in $\mathrm{V/C}$, or more commonly $\mathrm{V/pC}$. A test charge
$q_t$ receives an integrated voltage $q_s w$ and an energy change proportional
to $q_tq_s w$; the sign depends on the charge and wake conventions.

A simulation uses a finite bunch rather than a point source. Let
$\lambda(s)$ be its **charge-normalized longitudinal profile**,

$$
\int_{-\infty}^{\infty}\lambda(s)\,ds=1,
\qquad [\lambda]=\mathrm{m}^{-1}.
$$

The resulting **wake potential** is the convolution

$$
W_{\parallel,\perp}(s)
=\int_{-\infty}^{\infty}
w_{\parallel,\perp}(s-s')\lambda(s')\,ds'.
$$

Thus $w$ denotes the point-charge response and $W$ the response to the bunch
profile. `WakeSolver.WP`, `WPx`, and `WPy` store the latter in
$\mathrm{V/pC}$. This distinction follows the definitions in
[Teofili et al., *Phys. Rev. Accel. Beams* **24**, 041001 (2021)](https://doi.org/10.1103/PhysRevAccelBeams.24.041001).

Wakis obtains the transverse wake potential from the 3D longitudinal wake
potential using the Panofsky--Wenzel relation. With the $s$ and force signs
used in Wakis,

$$
\frac{\partial\mathbf W_\perp}{\partial s}
=-\nabla_\perp W_\parallel,
$$

and therefore

$$
W_{\perp,\alpha}(s)
=-\frac{\partial}{\partial\alpha}
\int_{-\infty}^{s}W_\parallel(s')\,ds',
\qquad \alpha=x,y.
$$

The transverse gradient is evaluated with second-order finite differences.

#### Transverse decomposition

Wakis supports transverse wake analysis:

$$
W_{\perp,\alpha}(\mathbf r_s,\mathbf r_t,s)
=W_{C,\alpha}(s)
+W_{D,\alpha}(s)\Delta\alpha_s
+W_{Q,\alpha}(s)\Delta\alpha_t
+\mathcal O(\lVert\mathbf r\rVert^2),
\qquad \alpha=x,y.
$$

- $W_D$: **dipolar wake**, linear in source offset
- $W_Q$: **quadrupolar wake**, linear in test offset
- $W_C$: **coherent term**, for asymmetric geometries

They can be separated by sampling field responses at multiple
$(x_s,y_s,x_t,y_t)$ combinations, either by displacing the beam source or the
integration path.


### 🔁 From Wake to Impedance

Given the bunch profile $\lambda(s)$ and wake potential $W(s)$, Wakis obtains
the point-charge impedance by deconvolution. Define the spatial Fourier
transform

$$
\widetilde g(f)=\int_{-\infty}^{\infty}
g(s)e^{-i2\pi f s/v}\,ds .
$$

The charge-normalized bunch spectrum $\widetilde\lambda(f)$ is dimensionless.
With the Wakis sign convention, the impedances are

- **Longitudinal impedance** in $\Omega$:

$$
Z_\parallel(f)=-\frac{\widetilde W_\parallel(f)}
{v\,\widetilde\lambda(f)}
$$

- **Transverse impedance of the simulated offset** in $\Omega$:

$$
Z_{\perp,\alpha}(f)=i\frac{\widetilde W_{\perp,\alpha}(f)}
{v\,\widetilde\lambda(f)},
\qquad \alpha=x,y.
$$

`WakeSolver.Zx` and `Zy` contain this offset-dependent transverse impedance.
For a purely dipolar wake, division by the corresponding nonzero source
offset gives the commonly quoted dipolar impedance,

$$
Z_{\perp,\alpha}^{\mathrm{dip}}(f)
=\frac{Z_{\perp,\alpha}(f)}{\Delta\alpha_s},
\qquad [Z_{\perp}^{\mathrm{dip}}]=\Omega/\mathrm m.
$$

Wakis uses `numpy.fft` and zero-padding. No window is applied automatically.

### ⚖️ Loss and kick factors

The bunch **loss factor** is the bunch-profile-weighted longitudinal wake
potential:

$$
k_\parallel
=\frac{\displaystyle\int W_\parallel(s)\lambda(s)\,ds}
{\displaystyle\int\lambda(s)\,ds}.
$$

Because `WakeSolver.lambdas` is normalized, the denominator is ideally one;
it is retained in the numerical implementation to account for sampling or a
truncated profile. `calc_loss_factor()` returns the signed value in
$\mathrm{V/pC}$.

For a source offset $\Delta\alpha_s\ne0$, the dipolar **kick factor** is

$$
k_{\perp,\alpha}
=\frac{1}{\Delta\alpha_s}
\frac{\displaystyle\int W_{\perp,\alpha}(s)\lambda(s)\,ds}
{\displaystyle\int\lambda(s)\,ds},
\qquad \alpha=x,y.
$$

`calc_kick_factors()` returns signed $k_x$ and $k_y$ in
$\mathrm{V/(pC\,m)}$. A kick factor is undefined for a zero source offset, so
the corresponding result is `None`. In asymmetric structures, a coherent or
quadrupolar contribution may also be present; a single-offset division should
then not be interpreted as a pure dipolar coefficient.


```{admonition} Modularity

#### 🧪 Open-Source Compatibility

The wake and impedance analysis is performed within the `WakeSolver` class in `Wakesolver.py`. Even if this module is tailored for Wakis, it is completely modular and can be used with other EM solvers besides Wakis' `SolverFIT3D` output:
- It has been tested with **WarpX** EM fields. WarpX is a powerful open-source PIC solver for full 3D EM fields
- It has been benchmarked with **CST** Wakefield solver, using both EM field ouput (time-domain field monitors) and calculated wake potential and impedance. > See the full wake analysis benchmark in [IPAC'23 proceedings, E. de la Fuente](https://doi.org/10.18429/JACoW-IPAC2023-WEPL170).
- **Interoperability**: Wakis can read field maps in HDF5, CSV, or NumPy format
- **Subvolume extraction** and interpolation are supported for field-based post-processing
```
