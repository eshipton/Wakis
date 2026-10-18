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

#### Grid conventions

**Staggering.** The primary grid has nodes $x_0 < x_1 < \dots$ (`grid.x`, likewise $y$, $z$), primary edge lengths $L_i = x_{i+1}-x_i$ and edge midpoints $x_{i+1/2}$ (`solver.x`). The curl matrix $\mathbf{C}$ is built from forward differences (`Px`, `Py`, `Pz`), which fixes where the sample $(i,j,k)$ of each component lives:

| sample $(i,j,k)$ | along $x$ | along $y$ | along $z$ |
|---|---|---|---|
| $E_x$, $J_x$ | $x_{i+1/2}$ | $y_j$ | $z_k$ |
| $E_y$, $J_y$ | $x_i$ | $y_{j+1/2}$ | $z_k$ |
| $E_z$, $J_z$ | $x_i$ | $y_j$ | $z_{k+1/2}$ |
| $H_x$ | $x_i$ | $y_{j+1/2}$ | $z_{k+1/2}$ |
| $H_y$ | $x_{i+1/2}$ | $y_j$ | $z_{k+1/2}$ |
| $H_z$ | $x_{i+1/2}$ | $y_{j+1/2}$ | $z_k$ |

i.e. $E_c$ sits at the edge midpoint along $c$ and on the nodes transversally, $H_c$ on the node along $c$ and at the midpoints transversally. Along one axis (here $x$, domain $[x_\text{min}, x_\text{max}]$ meshed with $N$ cells):

```text
 index i:     0         1               N-1         N      (ghost)
 nodes:    x_0=xmin    x_1    ...    x_{N-1}    x_N=xmax    x_{N+1}
             |----------|---- ... ------|----------|- - - - - -|
 E_y,E_z,H_x o          o               o          o
 E_x,H_y,H_z      *          *     ...       *           (*)
 dual length  L_0/2   (L_0+L_1)/2  (L_{N-2}+L_{N-1})/2  L_{N-1}/2
```

**Dual lengths.** The dual edge through node $i$ joins the neighbouring midpoints, $\tilde L_i = (L_{i-1}+L_i)/2$ (`grid.tL`), and the dual areas are products of dual and primary lengths (`grid.itA` stores their inverse). The first and last node of an axis get half a cell, $L_0/2$ and $L_{N-1}/2$: their dual cells end on the boundary. These are the lengths in Ampère's law $\widetilde{\mathbf{C}}\widetilde{\mathbf{D}}_s\mathbf{h}$, in the source normalization of `CurrentSource` and in the `FluxMonitor` area elements [Weiland, 1977; Weiland, 1996].

**Boundaries on the domain bounds.** By default (`GridFIT3D(..., boundary_at_domain_edge=True)`) one ghost cell is appended beyond $x_\text{max}$ on every axis, so the field arrays hold $N+1$ samples along each axis (`grid.Nx = N + 1`) and the last index is the boundary node $x_N = x_\text{max}$:
- **PEC** (and the PEC-terminated PML/CPML) zeroes the tangential E on the nodes $x_0$ and $x_N$: a PEC box meshed with $N$ cells is exactly $N\Delta x$ wide.
- **PMC** is the natural boundary of FIT and needs no masking: the tangential E on $x_0$ (or $x_N$) integrates H over the half dual cell $L/2$, and the tangential H just outside the domain is zero. This is exact by image symmetry ($H_\parallel$ odd about the wall), so the wall sits on $x_0$ (or $x_N$).
- **ABC** (Mur) updates the tangential E on $x_0$ and $x_N$.
- The ghost samples beyond $x_N$ ($E_x$, $H_y$, $H_z$ at $x_{N+1/2}$) have $1/\varepsilon = 1/\mu = 0$, so they stay zero and never couple to the domain; the forward difference of the last sample of a grid line does not wrap into the next line.
- On a **periodic/Bloch** axis `SolverFIT3D` removes the ghost cell from the grid (in place, `GridFIT3D.remove_edge_ghost`), so the axis keeps $N$ samples; the forward difference of sample $N-1$ wraps to sample 0, the period is $x_\text{max}-x_\text{min}$ and $\tilde L_0 = (L_{N-1}+L_0)/2$. Such a grid cannot be reused with non-periodic faces on that axis.
- With MPI only the last rank carries the $z$ ghost cell; `mpi_gather` returns the $N_z$ cell samples (the boundary node at $z_\text{max}$ is dropped).

Verification (`tests/test_026_grid_conventions.py`): the TM$_{110}$ frequency of an empty PEC box matches the leapfrog dispersion relation $\frac{2}{c\Delta t}\sin\frac{\omega\Delta t}{2} = \sqrt{\sum_a \left(\frac{2}{\Delta_a}\sin\frac{\pi\Delta_a}{2\ell_a}\right)^2}$ with $\ell_a$ the *domain* lengths to $10^{-5}$, and the continuum error converges at second order; PEC/PMC quarter-wave resonators ($f=c/4d$) and a box with PEC $x$/PMC $y$ walls ($f=c/2a$) match their discrete relations to $10^{-5}$; on a mesh graded by a factor 3 the TM$_{110}$ error converges at second order (−0.65 %, −0.16 %, −0.04 % at 12, 24, 48 cells), and the discrete energy $\tfrac12\sum\varepsilon V_e (E^n)^2 + \tfrac12\sum \mu V_h H^{n-1/2}H^{n+1/2}$ of the closed cavity is constant to round-off.

**Legacy convention** (`boundary_at_domain_edge=False`, used by the wake/meshing regression tests 001, 007, 013 and 014 and to reproduce results of earlier Wakis versions): $N$ samples per axis and the last index is a ghost layer, so the high-side PEC/ABC/CPML walls sit on $x_{N-1} = x_\text{max} - L_{N-1}$ (a PEC guide meshed with $N$ cells is $(N-1)\Delta x$ wide; a high-side PMC face also behaves as a PEC on $x_{N-1}$ because $\tilde L_{N-1} = 0$ freezes the tangential E there, and a low-side PMC sits on $x_{1/2}$); $\tilde L_i = (L_i + L_{i+1})/2$ is centred on node $i+1$, which makes the error of a graded mesh first order (+4.9 %, +2.5 %, +1.2 % for the case above). With MPI the legacy high $z$ wall sits on $z_\text{max}$ for one rank but one cell below it for several ranks. For the embedded-boundary cavity of test 001 (a 50 mm square box, $f_{110} = c/(\sqrt2\,a) = 4.240$ GHz without beam pipes) the dominant impedance peak moves from 4.333 GHz (legacy: the box is cut to 49 mm by the ghost layer, $c/(\sqrt2 \cdot 49\,\text{mm}) = 4.326$ GHz) to 4.268 GHz. Where the walls are embedded STL solids inside the domain (lossy cavity of tests 007/013) the domain-edge cell barely matters: $|Z|$ changes by 0.06 % and the wake potential by 3 % of its maximum.

References:
- T. Weiland, "A discretization method for the solution of Maxwell's equations for six-component fields", *Electronics and Communication (AEÜ)* **31**, 116–120 (1977).
- T. Weiland, "Time domain electromagnetic field computation with finite difference methods", *Int. J. Numer. Model.* **9**, 295–319 (1996).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), Ch. 4 (numerical dispersion).

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

### 🛰️ Near-to-far-field transformation: `Near2FarMonitor`

**Surface equivalence.** If a closed surface $S$ encloses all sources and scatterers, the field outside $S$ equals the field radiated in the unbounded background medium by the equivalent surface currents [Balanis 2016, Sec. 12.2; Taflove & Hagness 2005, Ch. 8]

$$
\mathbf{J}_s = \hat{n}\times\mathbf{H}, \qquad \mathbf{M}_s = -\hat{n}\times\mathbf{E},
$$

with $\hat{n}$ the outward normal. `Near2FarMonitor` is a `FluxMonitor` on a closed box, so the tangential DFT samples, the half-cell averaging of $H$ onto the $E$ plane and the trapezoidal area weights are exactly those of the flux computation. For a face normal to $a$ the sample of $E_b$ carries the product $E_b H_c$; at this point the currents are $J_b\,\Delta A = -w\,H_c$ and $M_c\,\Delta A = -w\,E_b$, where $w = \pm\Delta A$ is the signed weight of that product in $\Phi(f)$ (and similarly for $E_c H_b$). No further interpolation is needed.

**Radiation vectors.** With the $e^{+j\omega t}$ convention implied by the $e^{-2\pi i f t}$ DFT kernel, the far-field radiation vectors are [Balanis 2016, Sec. 12.3]

$$
\mathbf{N}(\hat{r}) = \sum_p \mathbf{J}_p\,\Delta A_p\, e^{+jk\,\mathbf{r}'_p\cdot\hat{r}}, \qquad
\mathbf{L}(\hat{r}) = \sum_p \mathbf{M}_p\,\Delta A_p\, e^{+jk\,\mathbf{r}'_p\cdot\hat{r}},
$$

with $k = \omega\sqrt{\varepsilon_r\mu_r}/c$ and $\eta = \eta_0\sqrt{\mu_r/\varepsilon_r}$ of the background (default: the solver background). The far field is $\mathbf{E} = \mathbf{F}(\hat r)\,e^{-jkR}/R$ with

$$
\mathbf{F} = -\frac{jk}{4\pi}\left[\eta\,(\mathbf{N} - (\mathbf{N}\cdot\hat r)\hat r) + \mathbf{L}\times\hat r\right], \quad
E_\theta = -\frac{jk e^{-jkR}}{4\pi R}(L_\phi + \eta N_\theta), \quad
E_\phi = \frac{jk e^{-jkR}}{4\pi R}(L_\theta - \eta N_\phi),
$$

and $\mathbf{H} = \hat r\times\mathbf{E}/\eta$. The sums are vectorized over frequencies, surface points and directions, chunked to bound memory. The continuum $k$ is used in the phase; the mismatch with the discrete dispersion relation is $O((k\Delta)^2)$ per unit of electrical box size.

**Ground planes.** For a box face lying on an infinite PEC plane (e.g. the $z_{min}$ boundary), the face is dropped and each remaining current element is accompanied by its image: tangential electric and normal magnetic currents are reversed [Balanis 2016, Sec. 4.7]. The result is valid in the half-space containing the box.

**Antenna metrics** [Balanis 2016, Ch. 2; IEEE Std 145-2013]:
- radiation intensity $U(\theta,\phi) = R^2|\mathbf{E}|^2/\eta = |\mathbf{F}|^2/\eta$;
- radiated power $P_{rad} = \oint U\,d\Omega$, evaluated with Gauss–Legendre quadrature in $\cos\theta$ and the uniform rule in $\phi$ (exact for the band-limited patterns of a finite source; default order $\max(24, kR_{max}+16)$), over the physical half-space when a ground plane is set;
- directivity $D = 4\pi U/P_{rad}$, gain $G = 4\pi U/P_{in}$ and radiation efficiency $e_{rad} = P_{rad}/P_{in} = G/D$, where the accepted power $P_{in}$ is supplied by the user, e.g. the source work $-\operatorname{Re}[E(f)I\ell^*(f)]$ or the flux through a box tightly around the feed;
- co-/cross-polarization with Ludwig's third definition: $E_{co} = E_\theta\cos\phi - E_\phi\sin\phi$, $E_{cross} = E_\theta\sin\phi + E_\phi\cos\phi$ for an $x$ reference [Ludwig 1973];
- circular components $E_{R,L} = (E_\theta \pm jE_\phi)/\sqrt2$ and axial ratio $AR = (|E_R|+|E_L|)/\big||E_R|-|E_L|\big|$;
- half-power beamwidth from a pattern cut by linear interpolation of the $-3$ dB points.

All quantities share the normalization of $\Phi(f)$ (no factor 1/2); ratios such as $D$, $G$ and $e_{rad}$ are normalization free.

**Validation** (`tests/test_021_near2far.py`, 20–30 cells per wavelength, CPML):
- Hertzian dipole: pattern $\propto\sin\theta$ to 0.6%, $D(90°)$ and $D_{max}$ equal to 1.5 within 1%, absolute $|E_\theta|$ to 1.2%, HPBW 90°, $P_{rad}$ equal to the box flux to 0.3%.
- Dipole at height $h$ above a PEC ground plane: pattern equals the free-space pattern times $|2\cos(kh\cos\theta)|$ (vertical) or $|2\sin(kh\cos\theta)|$ (horizontal) to 0.6%; $P_{rad}$ equals the source work to 0.6% and follows the image-theory ratio below to 2%. Ludwig-3 components of the horizontal dipole are checked analytically.
- The local density of states (source work, no near-to-far transformation) near the ground plane matches, for $u = 2kh$,

$$
\frac{P_\perp(h)}{P_0} = 1 + 3\left[\frac{\sin u}{u^3} - \frac{\cos u}{u^2}\right], \qquad
\frac{P_\parallel(h)}{P_0} = 1 - \frac32\left[\frac{\sin u}{u} + \frac{\cos u}{u^2} - \frac{\sin u}{u^3}\right],
$$

which follow from adding the field of the image dipole at the source, $P = -\tfrac12\operatorname{Re}[(E_{self}+E_{image})\,I\ell^*]$, using the exact Hertzian dipole fields [Balanis 2016, Sec. 4.2] (cf. the ground effects of Balanis 2016, Sec. 4.7).
- Purcell factor of a dipole midway between two PEC plates a gap $d$ apart. The images lie at distances $md$ ($m \neq 0$) with equal signs (perpendicular dipole) or signs $(-1)^m$ (parallel dipole). Summing the mutual terms above with the Fourier series of the Bernoulli polynomials, $\sum_m \cos(mx)/m^2 = \pi^2/6 - \pi x/2 + x^2/4$, $\sum_m \sin(mx)/m^3 = \pi^2x/6 - \pi x^2/4 + x^3/12$ and $\sum_m \sin(mx)/m = (\pi - x)/2$ for $0 < x < 2\pi$ [Abramowitz & Stegun 1964, Sec. 23.1], gives

$$
\frac{P_\perp}{P_0} = \frac{3\pi}{2kd}\quad (0<kd<2\pi), \qquad
\frac{P_\parallel}{P_0} = \frac{3\pi}{2kd}\left[1 + \left(\frac{\pi}{kd}\right)^2\right]\quad (\pi<kd<3\pi),
$$

and $P_\parallel = 0$ for $kd<\pi$, the single-mode (TEM, resp. TE$_1$+TM$_1$) results of the waveguide mode sum. The simulation agrees to 0.03 at $kd/\pi = 1.15$–$1.75$. Both PEC plates lie on the domain bounds (see *Grid conventions*).

**Differentiability.** The far field is linear in the recorded spectra: for each frequency, $\mathbf{F}(\hat r_i) = \sum_p G_{ip}\,[\,-w_p H_p,\ -w_p E_p\,]$ with $G_{ip}$ the dense kernel $-\tfrac{jk}{4\pi} e^{jk\mathbf{r}'_p\cdot\hat r_i}$ times the projections $\eta(\mathbb{1}-\hat r_i\hat r_i^T)\hat e_{J,p}$ and $\hat e_{M,p}\times\hat r_i$. `Near2FarMonitor.surface_currents()` returns the points, axes and current elements, so objectives such as $D(\hat r_0) = 4\pi|\mathbf F(\hat r_0)|^2 / \sum_i w_i|\mathbf F(\hat r_i)|^2$ can be written in JAX on top of the DFT data.

References:
- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley (2016), Ch. 2, 4, 6 and 12.
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), Ch. 8.
- A. C. Ludwig, "The definition of cross polarization", *IEEE Trans. Antennas Propag.* **21**(1), 116–119 (1973).
- IEEE Standard for Definitions of Terms for Antennas, IEEE Std 145-2013.
- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, NBS (1964), Sec. 23.1.

### 🌈 Dispersive materials (auxiliary differential equations)

Metals at optical frequencies, Lorentz dielectrics and lossy RF substrates have a frequency-dependent permittivity. `DispersiveMaterial` models it as a sum of poles plus a static conductivity:

$$
\varepsilon_r(\omega) = \varepsilon_\infty + \sum_p \chi_p(\omega) + \frac{\sigma}{i\omega\varepsilon_0},
\qquad
\chi^{\mathrm{L}}(\omega) = \frac{\Delta\varepsilon\,\omega_0^2}{\omega_0^2-\omega^2+i\gamma\omega},\quad
\chi^{\mathrm{D}}(\omega) = -\frac{\omega_p^2}{\omega^2-i\gamma\omega},\quad
\chi^{\mathrm{Deb}}(\omega) = \frac{\Delta\varepsilon}{1+i\omega\tau},
$$

for Lorentz (`Lorentz(delta_eps, omega_0, gamma)`), Drude (`Drude(omega_p, gamma)`, the Lorentz pole with $\omega_0 = 0$ and $\Delta\varepsilon\,\omega_0^2 \to \omega_p^2$) and Debye (`Debye(delta_eps, tau)`) poles. Angular frequencies are in rad/s (`wakis.materials.EV` converts eV to rad/s) and $\tau$ in s.

**Sign convention.** Wakis transforms time signals with the kernel $e^{-i\omega t}$ (`DFTMonitor`, `GaussianSource.fourier_transform`), so phasors follow $x(t) = \operatorname{Re}[X e^{+i\omega t}]$ and a passive medium has $\operatorname{Im}\varepsilon_r \le 0$. Optical tables quoting $\tilde n = n + i k$ ($k \ge 0$) use the opposite convention; `eps_from_nk(n, k)` returns $(n - ik)^2$.

**Time-domain model.** Each pole has a polarization $P_p$ obeying an auxiliary differential equation (ADE) [Taflove & Hagness, 2005, ch. 9],

$$
\ddot P_p + \gamma_p \dot P_p + \omega_{0,p}^2 P_p = \varepsilon_0 \Delta\varepsilon_p\,\omega_{0,p}^2 E \quad\text{(Lorentz/Drude)},\qquad
\tau_p \dot P_p + P_p = \varepsilon_0 \Delta\varepsilon_p E \quad\text{(Debye)},
$$

and enters Ampère's law as a polarization current:

$$
\varepsilon_0\varepsilon_\infty \dot E = \nabla\times H - J - \sigma E - \sum_p \dot P_p .
$$

**Discretization.** The polarizations live on the same edges and integer time levels as $E$. Second-order poles use central differences at $t^n$, which gives the explicit update

$$
P^{n+1} = \alpha P^n + \xi P^{n-1} + \zeta E^n,\quad
\alpha = \frac{2-\omega_0^2\Delta t^2}{1+\gamma\Delta t/2},\;
\xi = \frac{\gamma\Delta t/2-1}{1+\gamma\Delta t/2},\;
\zeta = \frac{\varepsilon_0\Delta\varepsilon\,\omega_0^2\Delta t^2}{1+\gamma\Delta t/2},
$$

the polarization form of the Lorentz ADE [Taflove & Hagness, 2005, ch. 9]. Debye poles and the conductivity use the trapezoidal (Crank–Nicolson) rule centred at $t^{n+1/2}$, $P^{n+1} = \kappa P^n + \beta(E^{n+1}+E^n)$ with $\kappa = (1-\Delta t/2\tau)/(1+\Delta t/2\tau)$ and $\beta = \varepsilon_0\Delta\varepsilon\,(\Delta t/2\tau)/(1+\Delta t/2\tau)$, so they impose no timestep restriction (unlike the explicit conductivity of non-dispersive materials, which requires $\Delta t < \varepsilon/\sigma$). Ampère's law at $t^{n+1/2}$,

$$
\varepsilon_0\varepsilon_\infty\frac{E^{n+1}-E^n}{\Delta t} = (\tilde C H)^{n+1/2} - J^n - \sigma\frac{E^{n+1}+E^n}{2} - \sum_p \frac{P_p^{n+1}-P_p^n}{\Delta t},
$$

is then solved locally for $E^{n+1}$ (a scalar division per edge). In Wakis this is implemented as a correction applied after the unchanged non-dispersive step (any backend: standard, CPML, MKL, GPU, MPI): from the provisional $E^\star$ computed with the solver's $\varepsilon_s$, the curl contribution is recovered exactly as $\Delta t(\tilde C H - J)/\varepsilon_0 = \varepsilon_{r,s}(E^\star - E^n)$, so CPML and TFSF terms are carried along. Only the dispersive edges are touched (index arrays and a few auxiliary arrays per pole); non-dispersive simulations are unaffected.

**Interfaces and mixtures.** An edge with fill fraction $w$ gets the arithmetic mean $\varepsilon_r^{\mathrm{eff}} = (1-w)\varepsilon_{r,\mathrm{bg}} + w\,\varepsilon_r(\omega)$, which is again a pole model: $\varepsilon_\infty \to (1-w)\varepsilon_{r,\mathrm{bg}} + w\varepsilon_\infty$ and all pole strengths and $\sigma$ scaled by $w$. `add_dispersive_material` uses $w = 1/2$ for tangential edges lying on a box face (second-order interface treatment, as in the slab test of the flux monitors); STL solids use the subpixel volume fraction already used for `ieps`.

**Discrete permittivity.** With $z = e^{i\omega\Delta t}$ the scheme behaves exactly like a Yee scheme with the vacuum operator $\tilde\Omega = 2\sin(\omega\Delta t/2)/\Delta t$ in a medium of permittivity

$$
\varepsilon_d(\omega) = \varepsilon_\infty + \sum_{\mathrm{L}} \frac{\Delta\varepsilon\,\omega_0^2}{\omega_0^2-\tilde\Omega^2+i\gamma\sin(\omega\Delta t)/\Delta t}
+ \sum_{\mathrm{Deb}} \frac{\Delta\varepsilon}{1+2i\tau\tan(\omega\Delta t/2)/\Delta t}
+ \frac{\sigma\cos(\omega\Delta t/2)}{i\tilde\Omega\varepsilon_0},
$$

returned by `DispersiveMaterial.epsilon_discrete(f, dt)`; $\varepsilon_d = \varepsilon_r + O(\Delta t^2)$.

**Stability.** For a lossless Drude pole the coupled Yee–ADE scheme is stable iff $(\omega_p\Delta t/2)^2 + C^2 \le \varepsilon_\infty$, and for a lossless Lorentz pole iff $(\omega_0\Delta t/2)^2 \le (\varepsilon_\infty - C^2)/(\varepsilon_s - C^2)$, with $C = c\Delta t\sqrt{\sum_i \Delta x_i^{-2}} \le 1$ the vacuum Courant number and $\varepsilon_s = \varepsilon_\infty+\Delta\varepsilon$ (Wakis derivation from the discrete dispersion relation $\tilde\Omega^2\varepsilon_d = c^2\tilde K^2$, requiring all roots on the unit circle for $c^2\tilde K^2\Delta t^2/4 \in [0, C^2]$). Damping only helps. For general multi-pole models, `DispersiveMaterial.is_stable(dt, courant)` evaluates the spectral radius of the single-mode amplification matrix over all grid wavenumbers; `add_dispersive_material` raises if the solver's $\Delta t$ is unstable. In practice the poles must be time-resolved ($\omega_0\Delta t, \omega_p\Delta t \lesssim 1$), which is always the case for optical metals meshed at a few nm; the default `cfln=0.5` leaves margin for strong poles (e.g. silicon's 0.3 µm Sellmeier pole requires $\Delta t \lesssim 8\times10^{-17}$ s).

**Material library and fitting.** `dispersive_lib` contains: gold, silver and aluminium with the Lorentz–Drude parameters of [Rakić et al., 1998] (valid 0.1–6 eV); lossless Sellmeier models of fused silica [Malitson, 1965] (0.21–6.7 µm) and silicon [Salzberg & Villa, 1957] (1.36–11 µm, transparent region only); and three-pole Debye fits (Wakis, 0.1–20 GHz) of Djordjevic–Sarkar wideband models [Djordjević et al., 2001] through nominal FR-4 ($\varepsilon' = 4.4$, $\tan\delta = 0.02$ at 1 GHz) and Rogers RO4350B ($\varepsilon' = 3.66$, $\tan\delta = 0.0037$ at 10 GHz) values. A Sellmeier term $B\lambda^2/(\lambda^2 - C^2)$ is a Lorentz pole with $\Delta\varepsilon = B$, $\omega_0 = 2\pi c/C$, $\gamma=0$. `fit_poles(f, eps, n_lorentz, n_drude, n_debye)` fits tabulated data by nonlinear least squares on the relative complex error; every parameter is optimized through its logarithm, so the fitted model is passive ($\operatorname{Im}\varepsilon_r\le 0$ for all $\omega$) and causal by construction.

**Validation** (`tests/test_025_dispersive.py`): Lorentz, Drude (across the plasma edge) and conductive Debye slabs reproduce the analytic Fabry–Pérot reflectance and transmittance with $\varepsilon(\omega)$ to $<10^{-2}$ (Drude: $2\times10^{-3}$) with second-order convergence in the cell size; a spatially uniform field reproduces $\varepsilon_d(\omega)$ to $10^{-8}$; the stability check matches the closed-form limits; a lossless Lorentz medium stays bounded for $4\times10^4$ steps; fits recover generating parameters to $10^{-6}$.

References:
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), ch. 9.
- A. D. Rakić, A. B. Djurišić, J. M. Elazar and M. L. Majewski, "Optical properties of metallic films for vertical-cavity optoelectronic devices", *Appl. Opt.* **37**, 5271–5283 (1998), [doi:10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).
- I. H. Malitson, "Interspecimen comparison of the refractive index of fused silica", *J. Opt. Soc. Am.* **55**, 1205–1208 (1965), [doi:10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
- C. D. Salzberg and J. J. Villa, "Infrared refractive indexes of silicon, germanium and modified selenium glass", *J. Opt. Soc. Am.* **47**, 244–246 (1957).
- A. R. Djordjević, R. M. Biljić, V. D. Likar-Smiljanić and T. K. Sarkar, "Wideband frequency-domain characterization of FR-4 and time-domain causality", *IEEE Trans. Electromagn. Compat.* **43**, 662–667 (2001).

### 🔋 Lumped elements, discrete ports and S-parameters

**Lumped elements.** `LumpedElement(component, center, size, R, L, C)` places a resistor, inductor and/or capacitor (in parallel) across a gap between two conductors. It occupies a box of primary edges along `component`: edges along the axis are in series, edges side by side are in parallel. With $\ell$ the gap length and $A$ the cross-section of the box (so that $\ell A = \sum_e \Delta l_e \Delta \tilde A_e$ over the edges, with $\Delta l_e$ the primary edge length and $\Delta \tilde A_e$ the dual face area), the element is the uniform block material

$$
\sigma = \frac{\ell}{R A}, \qquad \varepsilon_C = \frac{C \ell}{A}, \qquad \frac{\partial J_L}{\partial t} = \Gamma E, \quad \Gamma = \frac{\ell}{L A},
$$

whose voltage $V = -\sum_e g_e E_e$, with $g_e = \Delta l_e \Delta \tilde A_e / A$, and current reproduce $I = V/R + \frac{1}{L}\int V dt + C\, dV/dt$ [Piket-May et al., 1994; Taflove & Hagness, 2005]. $V$ is the potential of the high-coordinate end of the gap minus the low end. On each element edge the update of Ampère's law becomes

$$
(\varepsilon + \varepsilon_C)\frac{E^{n+1} - E^n}{\Delta t} = (\tilde{C} H)^{n+1/2} - J^{n+1/2} - \sigma \frac{E^{n+1} + E^n}{2} - J_L^{n+1/2}, \qquad J_L^{n+1/2} = J_L^{n-1/2} + \Delta t\, \Gamma E^n .
$$

The resistive current is averaged in time (semi-implicit), which is stable for any $R$; the explicit form $\sigma E^n$ is unstable once $\sigma \Delta t / \varepsilon > 2$, which is reached by ordinary resistors in millimetre cells. Solving for $E^{n+1}$ and writing $E^* = E^n + \Delta t\,\varepsilon^{-1}(\tilde{C}H - J)$ for the result of the unmodified solver step:

$$
E^{n+1} = a\, E^n + b\,(E^* - E^n), \qquad \alpha = \frac{\sigma \Delta t}{2(\varepsilon + \varepsilon_C)}, \quad a = \frac{1-\alpha}{1+\alpha}, \quad b = \frac{\varepsilon}{(\varepsilon + \varepsilon_C)(1+\alpha)} .
$$

Wakis implements this as a local, linear correction applied by the element right after `one_step` (`post_step`), while $J_L$ and any source current enter the solver's `J` before the step (`update`). The step functions themselves are untouched, so the element works with every backend (standard, MKL, CPML, GPU). The inductor update is explicit, like a Drude current; it is stable while $\Delta t \sqrt{\Gamma/\varepsilon} \ll 2$, which holds for nanohenry inductors in millimetre cells. The parallel elements are an adaptation to the FIT edge formulation; series RLC branches are not implemented.

**Discrete port.** `LumpedPort(src_time, component, center, size, impedance=Z0, amplitude=V_s)` is a resistor $Z_0$ in parallel with a soft current source $I_s(t) = V_s s(t)/Z_0$ (Norton form of a voltage source with internal impedance $Z_0$). The port records the time-centred voltage $\bar V^{n+1/2} = (V^n + V^{n+1})/2$ and the current delivered into the structure out of the positive terminal, $I^{n+1/2} = I_s^{n+1/2} - \bar V^{n+1/2}/Z_0$, both at $t_{n+1/2}$. These are the exact discrete charges flowing through the source and resistor, so charge is conserved and the displacement current of the gap cell is attributed to the structure; with this choice, the impedance of a uniform line seen from a port node is $Z/\cos(\beta\Delta z/2)$, i.e. second-order accurate. Their Fourier transforms $X(f) = \sum_n X^{n+1/2} e^{-2\pi i f t_{n+1/2}} \Delta t$ give

$$
Z_\text{in} = \frac{V}{I}, \qquad a = \frac{V + Z_0 I}{2\sqrt{Z_0}}, \qquad b = \frac{V - Z_0 I}{2\sqrt{Z_0}}, \qquad S_{11} = \frac{b}{a} = \frac{Z_\text{in} - Z_0}{Z_\text{in} + Z_0},
$$

the power waves of Kurokawa (1965) for a real reference impedance; $|a|^2 - |b|^2$ is the power delivered to the structure. By construction $a = \sqrt{Z_0}\, I_s / 2$, the wave launched by the source. For $N$ ports, `wakis.sparameters.s_matrix` runs $N$ simulations, each driving one port $j$ while the others are passive matched terminations, and returns $S_{ij} = b_i / a_j$. All quantities are plain arrays computed from the recorded time series by the formulas above.

**Touchstone.** `write_touchstone` and `read_touchstone` store S-parameters in the Touchstone v1 `.sNp` format (frequency in Hz, real/imaginary pairs, two-port order $S_{11}, S_{21}, S_{12}, S_{22}$, one matrix row per line for more ports); `to_skrf_network` converts to a scikit-rf `Network` when scikit-rf is installed.

**Energy-decay stopping criterion.** `emsolve(Nt, ..., stop_when_energy_decayed=1e-4)` stops once all sources have ended (`end_time`) and the field energy

$$
W = \frac12 \sum_e \varepsilon_e E_e^2 \Delta l_e \Delta\tilde A_e + \frac12 \sum_f \mu_f H_f^2 \Delta \tilde l_f \Delta A_f
$$

has decayed below the given fraction of its maximum (-40 dB here), as the steady-state energy criterion of commercial FIT solvers. Truncating the DFT then changes the spectra by a relative amount of order $\sqrt{W/W_\text{max}}$ or less.

**Validation** (`tests/test_022_lumped_ports.py`): a port in the middle of a matched parallel-plate line ($Z = \eta_0 d/w$) measures $Z_\text{in} = Z/2$ to 0.5% at 30 cells per wavelength and the discrete-line value $Z/(2\cos(\beta\Delta z/2))$ to 0.2%; shunt resistors from $Z/1000$ to $10Z$ give $\Gamma = (R\|Z - Z)/(R\|Z + Z)$ to $1.3\times10^{-3}$; de-embedded inductors and capacitors match $j\omega L$ and $1/(j\omega C)$ to 1%; a parallel LC tank is transparent at $1/(2\pi\sqrt{LC})$; a two-port S-matrix matches the ABCD-matrix result and is reciprocal; the -40 dB criterion stops after 1/8 of the steps with a $3\times10^{-5}$ change in $Z_\text{in}$.

References:
- M. Piket-May, A. Taflove and J. Baron, "FD-TD modeling of digital signal propagation in 3-D circuits with passive and active loads", *IEEE Trans. Microw. Theory Techn.* **42**(8), 1514 (1994), [doi:10.1109/22.297814](https://doi.org/10.1109/22.297814).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).
- K. Kurokawa, "Power waves and the scattering matrix", *IEEE Trans. Microw. Theory Techn.* **13**(2), 194 (1965), [IEEE Xplore 1125964](https://ieeexplore.ieee.org/document/1125964).
- Touchstone File Format Specification, Rev. 1.1, EIA/IBIS Open Forum (2002).

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

and the eigenproblem $\tilde\beta^2\,\mathbf{e} = PQ\,\mathbf{e}$ for the transverse E samples. This is the Yee-mesh finite-difference mode solver of Zhu & Brown [2002], here built from Wakis' own material tensors (`ieps`, `imu`), metric lengths ($L$, $\tilde L$) and boundary masks: PEC (and the PEC-terminated CPML/PML faces) removes the tangential E samples on the walls, PMC is the natural boundary (as in 3D, see *Grid conventions*), periodic axes wrap, conductors with `ieps = 0` remove their E samples, and the ghost samples beyond the last node are excluded. A port `size` smaller than the domain puts electric walls on the port edges (e.g. to exclude CPML layers). The eigenvalues closest to $n_{max}^2\tilde\omega^2/c^2$ are found with shift-invert Arnoldi (`scipy.sparse.linalg.eigs`), and $\beta = (2/\Delta a)\arcsin(\tilde\beta\Delta a/2)$. Because the same discrete operators are used, a mode is an exact eigenmode of the 3D time stepping of a uniform guide: the TE10 propagation constant matches the discrete dispersion relation $(\tilde\omega/c)^2 = \tilde k_x^2 + \tilde\beta^2$ to $10^{-9}$ and the continuum $\sqrt{k^2-(\pi/a)^2}$ to grid-dispersion accuracy; the TE0 effective index of a dielectric slab converges at second order to the solution of $\tan(\kappa d/2) = \gamma/\kappa$ [Snyder & Love, 1983].

**Power normalization.** The `FluxMonitor` (and the discrete Poynting theorem of the Yee scheme) pairs the E plane with H averaged from the two neighbouring H planes, which for a mode is $\bar{\mathbf{h}} = \mathbf{h}\cos(\beta\Delta a/2)$. Modes are normalized to unit time-averaged power,

$$
\tfrac12\operatorname{Re}\sum \left(e_b\,\bar h_c^* - e_c\,\bar h_b^*\right)\Delta A = 1\ \text{W},
$$

with the `FluxMonitor` area elements. The PEC walls sit on the domain bounds: a guide meshed with $N$ cells has width $N\Delta$ (legacy grids, `boundary_at_domain_edge=False`: $(N-1)\Delta$).

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
- **PEC (Perfect Electric Conductor)**: masks tangential electric-field degrees of freedom, enforcing $\vec{E}_{\parallel} = 0$ at the selected face. All non-periodic faces lie on the domain bounds.
- **PMC (Perfect Magnetic Conductor)**: enforces $\vec{H}_{\parallel} = 0$ on the face. It is the natural boundary of FIT (half dual cell on the boundary node, zero H outside), so no masking is needed; legacy grids mask the tangential H samples instead (see *Grid conventions*).
- **Periodic**: pairs the low and high faces of an axis and closes the corresponding FIT derivative stencil (`Px`, `Py`, or `Pz`) across that seam. The curl matrix is then rebuilt from the corrected derivative matrices and the periodic dual metrics are updated. A periodic face must be paired with the opposite face. Serial periodic boundaries are available in all three directions; longitudinal periodic MPI requires cyclic ghost exchange and is intentionally not enabled yet.
- **ABC**: a first-order Mur radiation condition for selected domain faces.
- **PML**: a finite, graded, electrically lossy layer terminated by the usual electric boundary mask.
- **CPML**: a convolutional absorbing layer that adds auxiliary curl-correction fields.
- **Bloch**: a periodic axis whose wrap-around coupling carries the phase $e^{ik_aL}$ of a prescribed Bloch wavevector (complex fields); see below.

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


#### Bloch-periodic (Floquet) boundaries

A structure that repeats with lattice vector $\mathbf{L}$ (a metasurface or grating unit cell, a photonic crystal) supports fields that repeat up to a phase, the Floquet-Bloch condition [Yeh, Yariv & Hong, 1977; Taflove & Hagness, 2005]:

$$
\mathbf{F}(\mathbf{r} + \mathbf{L}) = \mathbf{F}(\mathbf{r})\, e^{i\,\mathbf{k}\cdot\mathbf{L}} ,
$$

where $\mathbf{k}$ is the Bloch wavevector. A single unit cell then represents the infinite periodic structure for the chosen $\mathbf{k}$. In Wakis, `bc_low`/`bc_high = 'bloch'` on an axis together with `k_bloch=(kx, ky, kz)` [rad/m] applies this condition with $L$ the domain length along that axis. `k_bloch = 0` reduces to `'periodic'`.

**Discretization.** A periodic axis has no ghost layer: the forward difference of the last sample wraps to the first one. For a Bloch axis the wrapped sample carries the phase $p = e^{ik_aL}$; e.g. for $x$,

$$
(\mathbf{P}_x f)_{N_x-1} = p\, f_0 - f_{N_x-1}
$$

(for $N_x = 1$ the cell couples to itself, $(p-1) f_0$). The primary curl $\mathbf{C}$ is assembled from these complex $\mathbf{P}_a$. The dual curl, which acts on $H$ and needs the sample *before* the first one, $f_{-1} = f_{N_x-1}\,e^{-ik_aL}$, is exactly the Hermitian transpose:

$$
\widetilde{\mathbf{C}} = \mathbf{C}^{H} \quad(=\mathbf{C}^{T} \text{ for real } \mathbf{C}).
$$

Wakis always builds the dual operators with $\mathbf{C}^H$ (also in the CPML derivative operators), so real simulations are unchanged. The leapfrog scheme with $\mathbf{C}$ and $\mathbf{C}^H$ conserves the same discrete energy as the real scheme and has the same CFL limit. Using $\mathbf{C}^T$ instead would be inconsistent and unstable. This is a direct transcription of the continuous Bloch condition onto the FIT stencil, i.e. the "direct field" periodic boundary of FDTD [Taflove & Hagness, 2005].

**Complex fields.** For $\mathbf{k} \neq 0$ the phase is complex, so the fields must be complex: pass `dtype=np.complex128` (or `np.complex64`). Material tensors and $\Delta t$ keep the matching real precision. Complex fields cost about twice the memory and work of real ones. Real-field behaviour and performance are unchanged when Bloch faces are not used.

**Which part is physical.** The update operator has complex coefficients only through the Bloch phase. Its complex conjugate solves the same problem with $-\mathbf{k}$. Hence $\operatorname{Re}\mathbf{F}(t) = (\mathbf{F}+\mathbf{F}^*)/2$ is a real, physical field: the superposition of the $+\mathbf{k}$ and $-\mathbf{k}$ Bloch solutions of the infinite structure. It is *not* itself Bloch-periodic with $\mathbf{k}$. The cleaner reading is in the frequency domain. Wakis' monitors use $F(f) = \int F(t)e^{-2\pi i f t}dt$, i.e. phasors with time dependence $e^{+2\pi i f t}$. Each spectral component $\mathbf{F}(\mathbf{r}, f)$ of a complex Bloch run is the phasor of a physical time-harmonic field $\operatorname{Re}[\mathbf{F}(\mathbf{r}, f)e^{2\pi i f t}]$ that satisfies the Bloch condition with $\mathbf{k}$. For $f>0$ its tangential phase varies as $e^{i(2\pi f t + \mathbf{k}\cdot\mathbf{r})}$, so its phase fronts travel along $-\mathbf{k}$. For $f<0$ they travel along $+\mathbf{k}$. A real source time profile (e.g. `GaussianSource`) excites both signs of $f$ with conjugate-symmetric spectra, as in Meep's complex-field mode [Oskooi et al., 2010]. Positive and negative frequencies are two independent mirror-image waves with the same angle $|\sin\theta|$. For mirror-symmetric structures (or quantities like band frequencies, which satisfy $\omega(\mathbf{k}) = \omega(-\mathbf{k})$ by time reversal) they give identical results. Wakis' convention is therefore: **report results at $f > 0$, and interpret them as waves with tangential wavevector $-\mathbf{k}_\text{bloch}$** (or equivalently pass $-\mathbf{k}$ to describe a wave travelling along $+\mathbf{k}$). With complex fields the flux spectra at $\pm f$ are independent and the energy is $\int_{-\infty}^{\infty}\Phi(f)\,df$.

**Oblique incidence.** A plane wave hitting a periodic cell (normal $z$) at polar angle $\theta$ and azimuth $\phi$ has tangential wavevector $\mathbf{k}_\parallel = n k_0 \sin\theta\,(\cos\phi, \sin\phi, 0)$. It is launched by a sheet `CurrentSource` spanning the cell, with `amp_func = exp(1j k_par . r)`, in a complex simulation with `k_bloch = k_par` and absorbing (CPML) faces along $z$. Phase matching fixes $\mathbf{k}_\parallel$, not $\theta$. A broadband pulse therefore probes a different angle at each frequency:

$$
\theta(f) = \arcsin\frac{c\,|\mathbf{k}_\parallel|}{2\pi f n},
$$

and frequencies below the cutoff $f_c = c|\mathbf{k}_\parallel|/(2\pi n)$ are evanescent (`wakis.bloch.incidence_angle` and `k_parallel` implement these relations). Near $f_c$ the normal group velocity vanishes and these components linger in the cell. Use a source band that stays clear of $f_c$, or run long enough for them to decay; otherwise their truncation leaks into the DFT. Results at a fixed angle need one run per frequency, or interpolation over several $\mathbf{k}_\parallel$.

**Validation** (`tests/test_024_bloch.py`):
- A cell of length $L$ and a supercell of length $2L$ (phase $e^{2ikL}$) with Bloch-phased sources agree to round-off, with $\mathbf{F}(x+L) = \mathbf{F}(x)e^{ikL}$ (complex128: $10^{-12}$; complex64: $10^{-4}$). With $\mathbf{C}^T$ instead of $\mathbf{C}^H$ this fails.
- The normal wavenumber of a Bloch-phased sheet wave matches the scheme's discrete dispersion relation to $10^{-3}$. Its angle matches $\theta(f)$ to $0.1^\circ$.
- s- and p-polarized reflectance of an $n = 2$ half-space at $22^\circ$–$36^\circ$ (fixed $k_\parallel$) matches the Fresnel equations within 0.6% at $\geq 44$ cells per wavelength. The error is second order in the mesh size.
- The bands of a 1D photonic crystal ($n = 1.5/3$ half-period layers, 80 cells per period) match the Yeh–Yariv–Hong dispersion relation
$$\cos(kL) = \cos(k_1 d_1)\cos(k_2 d_2) - \tfrac12\left(\tfrac{n_1}{n_2}+\tfrac{n_2}{n_1}\right)\sin(k_1d_1)\sin(k_2d_2)$$
within 0.2% for the first four bands.

**Limitations.** Bloch faces along the MPI-decomposed $z$ axis are not supported (as for periodic). GPU and MKL backends receive complex sparse matrices but are untested. Plotting routines expect real fields: plot `np.real(...)` of the field arrays. The `'pml'` and Mur `'abc'` faces are not designed for oblique Bloch waves; use `'cpml'`.

**Adjoint.** The Bloch step is linear, with update matrices $\mathbf{A}_H = \Delta t\,\mathbf{M}_\mu^{-1}\mathbf{C}$ (for $H$) and $\mathbf{A}_E = \Delta t\,\mathbf{M}_\varepsilon^{-1}\mathbf{C}^H$ (for $E$), where the metric/material matrices are real and diagonal. For a real objective of complex fields, the discrete adjoint runs backwards in time with the Hermitian transposes $\mathbf{A}_H^H = \Delta t\,\mathbf{C}^H\mathbf{M}_\mu^{-1}$ and $\mathbf{A}_E^H = \Delta t\,\mathbf{C}\,\mathbf{M}_\varepsilon^{-1}$. These are the same Bloch curls with the same $\mathbf{k}$: the adjoint $E$ is driven by $\mathbf{C}^H$ (the dual curl, which carries the conjugate phase $e^{-ikL}$ on the wrap-around) and the adjoint $H$ by $\mathbf{C}$ (phase $e^{+ikL}$), with the diagonal metrics applied before rather than after the curl. The plain transpose $\mathbf{A}^T = \overline{\mathbf{A}^H}$, used by holomorphic/JAX-style complex transposes, corresponds instead to the scheme with $-\mathbf{k}$.

References:
- P. Yeh, A. Yariv and C.-S. Hong, "Electromagnetic propagation in periodic stratified media. I. General theory", *J. Opt. Soc. Am.* **67**, 423–438 (1977), [doi:10.1364/JOSA.67.000423](https://doi.org/10.1364/JOSA.67.000423).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), chapter on periodic structures.
- A. F. Oskooi et al., "MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method", *Comput. Phys. Commun.* **181**, 687 (2010), [doi:10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).


### 📥🗿 Geometry Importing & Embedded Boundaries

Wakis integrates with [**PyVista**](https://docs.pyvista.org/) to import CAD geometries in `.STL`, `.STEP`, or `.OBJ` formats. The mesh is overlaid onto the simulation domain and mapped onto the Cartesian grid using:
- `pyvista`'s surface collision algorithm, based on VTK optimized ray-tracing, allows to detect where the input geometry intersects the primal and dual grids.
- Assignment of material properties ($\varepsilon_r$, $\mu_r$, $\sigma$) in $x$, $y$, and $z$ to the intersected cells using a first-order subpixel smoothing, inspired by the open-source solver MEEP (MIT).

| STL geometry | Imported material mask |
| ----- | ---- |
|  ![](img/STL_solid.png)     |  ![](img/STL_mask.png)    |

Future versions aim to include a more advanced meshing algorithm for improved fidelity near corners and edges.

### 🟢 Conformal FIT: partially filled cells, subpixel dielectrics and thin sheets

Staircased cell masks move a curved or oblique boundary by up to a cell. The resulting error in resonant frequencies, phase and impedances is first order in the cell size $\Delta$. `GridFIT3D(conformal=...)` and `GridFIT3D(sheets=...)` instead keep the boundary inside the cells. They change only the metric and material coefficients of the FIT update, so the time step stays linear in the fields. All of it is opt-in, and the default staircase path and its regression data are unchanged. The implementation is in `wakis/conformal.py`.

**Geometry.** For every STL solid (triangulated, oriented outwards) Wakis computes exact intersections with the grid for the polygonal surface:

- the sections of the surface with the node planes $w=w_k$, as oriented segments with the solid on their left;
- for each primary edge, the length inside the solid along its grid line. With the crossings $c_m$ of the line and $s_m=\pm1$ for entering or leaving, the inside length up to $x$ is $F(x)=\sum_m s_m (x-c_m)_+$, evaluated with cumulative sums;
- for each primary face, the area inside the solid, from Green's theorem $A = \tfrac12\oint (u\,dv - v\,du)$ over the boundary of (section ∩ face). This boundary is the section contour split at the grid lines, plus the inside parts of the face sides, which contribute $\Delta u\,\ell_\text{right} + \Delta v\,\ell_\text{top}$ in face-local coordinates;
- for each E edge, the volume fraction of the solid in its dual cell. It is integrated exactly in $x,y$ (section areas) and with `conformal_subsamples` midpoint planes along $z$, which is exact for geometry extruded along $z$.

Degenerate positions, such as faces on node planes or vertices on grid lines, are resolved by evaluating the geometry with two opposite shifts of $10^{-7}\Delta$. The maximum of the two results (or the minimum, for holes in a PEC background) makes surfaces belong to the conductor. A 100³ grid with a sphere of $8\times10^4$ triangles takes about 1.3 s of geometry work plus 1.2 s for the stability estimate below.

**Conformal PEC (Dey–Mittra / PBA type).** Faraday's law on a face is integrated only over its part outside the conductor. With $\ell_e$ the open part of the edge length $L_e$ and $A_f$ the open part of the face area,

$$
\mu A_f \frac{d H_f}{dt} = -\sum_{e\in\partial f} \pm\, \ell_e E_e ,
$$

that is, `solver.L` holds $\ell_e$ and `solver.iA` holds $1/A_f$, while the dual (E) update is unchanged. Edges with $\ell_e=0$ get `ieps = 0`. With a PEC background (`bg='pec'`) the conductor is everything outside the non-PEC solids, so an accelerator cavity given as a vacuum STL is conformal too. This is the locally conformal FDTD of Dey and Mittra [Dey & Mittra, 1997], written in FIT form as in the Perfect Boundary Approximation [Krietenstein et al., 1998]. Overlapping PEC solids are combined with the maximum of their covered lengths and areas, which is an approximation of the union.

**Stability treatment (Wakis choice).** A small open area $A_f$ makes the face "stiff" and, with $\ell$ of order $\Delta$, reduces the stable time step without bound. The update can be written as $\mathbf H' = \mathbf H - \Delta t\,\mathbf A\mathbf E$, $\mathbf E' = \mathbf E + \Delta t\,\mathbf B\mathbf H'$ with $\mathbf B\mathbf A = \mathbf D_1\mathbf C^H\mathbf D_2\mathbf G\,\mathbf C\mathbf D_4$, where $\mathbf D_1 = \varepsilon^{-1}/\tilde A$, $\mathbf D_2 = \tilde L$, $\mathbf D_4 = \ell$ and $\mathbf G = \mu^{-1}\cdot$(inverse face areas). This product is similar to the Hermitian positive semi-definite $\mathbf R\mathbf C^H\mathbf D_2\mathbf G\mathbf C\mathbf R$, $\mathbf R=(\mathbf D_4\mathbf D_1)^{1/2}$, as long as $\mathbf D_2\mathbf G$ is symmetric positive semi-definite. The leapfrog is then stable for $\Delta t < 2/\sqrt{\lambda_\max}$, with a conserved discrete energy. Wakis keeps this structure and replaces $\operatorname{diag}(1/A_f)$ by

$$
\mathbf G = \mathbf T^T \operatorname{diag}(1/\hat A)\,\mathbf T,\qquad \hat A = \mathbf T A .
$$

Here a face is unstable when $A_f < \tfrac12\,\theta\,\max_e(\ell_e h_{\perp,e})$, with $h_{\perp,e}$ the face size perpendicular to edge $e$. This is the criterion of the legacy Wakis ECT solver, used here with $\theta=0.5$. An unstable face sends its flux to stable in-plane neighbours along its longest open edge, with weights $(2,-1)$ to the first and second neighbour. Its field is then the linear extrapolation $H_u = 2H_1 - H_2$, a uniform field is reproduced exactly, and the second-order convergence of Dey–Mittra is kept. Plain cell merging (weight 1) costs an $O(\Delta)$ error: the TE11 error of a circular cavity at 96 cells across went from $6\times10^{-5}$ to $1.2\times10^{-3}$. Faces without stable neighbours fall back to the area $\theta A_\text{stab}$.

The time step is then $\Delta t = \min(\Delta t_\text{default}, \texttt{cfln}\cdot 2/\sqrt{\lambda_\max})$. $\lambda_\max$ is computed with Lanczos iterations (`scipy.sparse.linalg.eigsh`, with a 1 % margin) and stored in `solver.conformal_dt_max`. A user-given `dt` above the limit triggers a warning. In the tests the limit is 0.85–1.0 times the Yee limit for curved PEC (0.86 for the sphere, independent of resolution), so with the default `cfln=0.5` the step is reduced by at most about 15 %. For the patch template it is 0.67 times the Yee limit (thin substrate cells and the sheet-edge correction of faces next to the edge, see below).

**Conformal dielectrics (`conformal=True` or `'all'`).** On each E edge the inverse permittivity is the diagonal of the anisotropic subpixel average [Farjadpour et al., 2006; Kottke et al., 2008]

$$
\varepsilon^{-1}_{cc} = n_c^2\,\langle\varepsilon^{-1}\rangle + (1-n_c^2)\,\langle\varepsilon\rangle^{-1},
$$

with $\langle\cdot\rangle$ the volume average over the dual cell of the edge and $\mathbf n$ the normal of the nearest surface triangle. Later solids replace earlier ones where they overlap: $f_j \leftarrow f_j(1-f_k)$. The conductivity is averaged arithmetically, and `imu` keeps the cell masks. FIT has one E component per edge, so the off-diagonal terms $n_xn_y(\langle\varepsilon^{-1}\rangle-\langle\varepsilon\rangle^{-1})$ are dropped. The method is therefore second order when $\mathbf E$ is parallel or normal to the interface (dielectric rod in a cavity, TM0: errors $1.2\times10^{-4}$, $3.5\times10^{-5}$, $3\times10^{-6}$ at 24/48/96 cells, against $2\times10^{-2}$, $5\times10^{-3}$, $1.6\times10^{-3}$ for the staircase). It is only first order for $\mathbf E$ crossing a curved interface obliquely: for the TE1 mode of the same cavity the error is $4.6\times10^{-3}$, $2.1\times10^{-3}$, $1.0\times10^{-3}$, comparable to the staircase. `conformal='pec'` keeps the conformal PEC and assigns each dielectric edge the material with the largest share of its dual cell.

**Thin PEC sheets.** A sheet is a planar zero-thickness polygon (`sheets={name: PolyData}`, `rectangle_sheet`, or an STL solid with material `'sheet'`). It must lie on a node plane normal to x, y or z; oblique sheets are not supported. The open lengths and areas of the edges and faces in its plane are computed from the polygon outline as above, so a sheet edge between nodes cuts its edges and faces partially. This coverage alone still leaves the edge $0.15$–$0.37\,\Delta$ too far out, depending on its position between the nodes, because the field at a knife edge is singular, $\propto r^{-1/2}$. The empirical one-third-cell rule of `patch_antenna` comes from this offset.

**Knife-edge correction (Wakis adaptation of the static-field corrections of [Shorthouse & Railton, 1992]).** Near a straight sheet edge along $t$, with $s$ pointing away from the metal and $w$ normal to the sheet, the static field is $F=\sqrt{\zeta}$, $\zeta = s + i w$, with the branch cut on the metal. $\operatorname{Re}F$ is the potential of the field with charge at the edge, and also $E_t$ of the field with current along the edge; $\operatorname{Im}F$ is its stream function. Every material coefficient within 1.5 cells of an axis-aligned sheet edge is scaled by the ratio of its exact voltage-to-flux relation for $F$ to the standard FIT one:

| quantity | exact relation used | factor on |
| --- | --- | --- |
| $E_s$, $E_w$ (charge at the edge) | $e = \Delta\operatorname{Re}F$ along the edge, $d = \Delta\operatorname{Im}F$ across its dual face | `ieps` |
| $E_t$ (current along the edge) | $e = \operatorname{Re}F$ at the node, $d = \iint_{\tilde A}\operatorname{Re}F$ (closed form) | `ieps` |
| $H_s$, $H_w$ faces | $b = \Delta\operatorname{Re}F$ along the face, $h = \Delta\operatorname{Im}F$ along its dual edge | inverse face area |

Dual faces and edges keep their FIT position. Where one crosses the sheet, the principal branch goes around the tip, so the discrete Gauss and Ampère laws exclude the charge and current on the sheet piece inside the dual cell. With this convention the static singular field solves the discrete equations exactly. Unstable faces next to the edge redistribute their flux with the ratio of the knife-edge fields instead of the linear extrapolation. Objects at the sheet corners (beyond the ends of an edge) are not corrected, except $E_t$ edges that reach beyond the corner: these are corrected on the part beside the sheet. The factors are kept in `solver.conformal_ieps_factor`, so that materials written later (templates) are corrected the same way. `sheet_edge_correction=False` disables the correction.

In a PEC box with a fin, the effective edge position is within $0.004\Delta$ of the physical one for any position of the edge between nodes when the charge is at the edge, against $0.15$–$0.37\Delta$ without the correction. When the current runs along the edge it is within $0.02$–$0.04\Delta$, against $0.12$–$0.33\Delta$. The correction is exact only for the singular term. Regular fields near the edge, such as a uniform $E_w$ through a sheet that does not perturb it, see $O(\Delta^2)$ perturbations of the order of $10^{-3}$ at 15 cells across a cavity. At patch corners the field has a stronger singularity, which is not modelled. In a 3D air patch (2 mm cells, sweep of the patch width) the corrected resonance still moves by up to 1 % when a non-radiating edge approaches a node from below. The sweep of the length, where the radiating edges carry the charge, is smooth. This residual edge-position dependence is the main limitation of the sheet model.

**Differentiability.** All conformal quantities are fixed coefficients: $\ell$ in `solver.Ds`, the non-diagonal $\mathbf G$ in `solver.iDa` (and in the CPML derivative operators), and the corrected `ieps`. The update stays linear. `AdjointStepper` transposes the solver's own operators, so the adjoint remains exact, including the non-diagonal redistribution (dot-product identity $\le 10^{-13}$ relative for PEC and CPML in `tests/test_028_conformal.py`). The geometry itself (STL, sheet outline) is not differentiated. A `MaterialGrid` design region overlapping conformal dielectrics or sheet edges would overwrite their `ieps`; keep design regions away from them.

**Limitations.** `use_mpi` is untested; each rank computes its own geometry. GPU is untested. SIBC and PML are rejected (CPML works). `save_to_h5` and `mpi_gather_asGrid` do not carry `conformal`/`sheets`. The permeability is not averaged. Merged faces assume a uniform $\mu$. The ±shift convention makes edges and faces lying exactly on a PEC surface closed.

**Validation** (`tests/test_028_conformal.py`, all in the time domain with a matrix-pencil fit of the ring-down and the leapfrog dispersion removed):

| Case | Mesh | Reference | Staircase / plain | Conformal |
| --- | --- | --- | --- | --- |
| PEC sphere, TM11 ($x=2.7437$) | 16/24/32 cells across $2a$ | analytic | −12 %, −10 %, −8 % (1st order) | $3.1\times10^{-3}$, $9.2\times10^{-4}$, $4.8\times10^{-4}$ (order 2.6) |
| PEC circular cavity TE11/TM01 (2D, eigenvalues) | 24/48/96 across $2.4b$ | analytic | 13 %, 6.6 %, 3.4 % | $1.1\times10^{-3}$, $2.5\times10^{-4}$, $6\times10^{-5}$ (order 2) |
| Rod-loaded cavity TM0, $\varepsilon_r=4$ | 24/48/96 | Bessel matching | $2\times10^{-2}$ … $1.6\times10^{-3}$ ('pec') | $1.2\times10^{-4}$ … $3\times10^{-6}$ ('all') |
| Fin in a PEC box, edge at 51 mm | 2 mm (edge mid-cell) / 1 mm (on node) | fine-mesh Richardson | $4.5$/$5.4\times10^{-3}$ and $2.1$/$3.0\times10^{-3}$ | $1.1\times10^{-4}$/$9\times10^{-6}$ and $6.2$/$1.9\times10^{-4}$ |
| Patch template, FR-4, 15/20/30 cells/λ (slow) | edges between nodes | Δ→0 extrapolation −2.3 % vs TLM | −7.3/−6.0 % at 20/30 (no correction), −2.5/−2.7 % (empirical $\Delta/3$) | −3.4/−2.4/−3.1 % (`thin_sheet=True`, no correction) |
| Stability: PEC ball + rod + sheet, 20 000 steps | 14×12×12 | bounded energy | — | no growth |

References:
- S. Dey and R. Mittra, "A locally conformal finite-difference time-domain (FDTD) algorithm for modeling three-dimensional perfectly conducting objects", *IEEE Microwave and Guided Wave Letters* **7**(9), 273–275 (1997).
- B. Krietenstein, R. Schuhmann, P. Thoma and T. Weiland, "The Perfect Boundary Approximation technique facing the big challenge of high precision field computation", *Proc. LINAC'98*, Chicago, pp. 860–862 (1998), [JACoW TH4041](https://epaper.kek.jp/l98/PAPERS/TH4041.PDF).
- A. Farjadpour, D. Roundy, A. Rodriguez, M. Ibanescu, P. Bermel, J. D. Joannopoulos, S. G. Johnson and G. W. Burr, "Improving accuracy by subpixel smoothing in the finite-difference time domain", *Opt. Lett.* **31**, 2972–2974 (2006), [doi:10.1364/OL.31.002972](https://doi.org/10.1364/OL.31.002972).
- C. Kottke, A. Farjadpour and S. G. Johnson, "Perturbation theory for anisotropic dielectric interfaces, and application to subpixel smoothing of discretized numerical methods", *Phys. Rev. E* **77**, 036611 (2008), [doi:10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
- D. B. Shorthouse and C. J. Railton, "The incorporation of static field solutions into the finite difference time domain algorithm", *IEEE Trans. Microwave Theory Tech.* **40**(5), 986–994 (1992).

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

### 🎯 Adjoint method and inverse design

Inverse design optimizes many parameters, here the permittivity of every edge of a design region, to maximize an objective $F$ of the simulated fields [Molesky et al., 2018]. Finite differences would need one simulation per parameter; the adjoint method gives the full gradient $dF/d\boldsymbol\rho$ from one forward and one adjoint (backward-in-time) simulation [Giles & Pierce, 2000]. As in automatic-differentiation frameworks for Maxwell solvers [Hughes et al., 2019], the solver gradient is exposed as a vector-Jacobian product that composes with automatically differentiated parametrizations and objectives. Wakis differentiates the *discrete* time stepping itself (discretize-then-differentiate), so the gradient is exact for the simulated fields up to rounding, not an approximation of the continuous gradient. This is a Wakis implementation of the standard discrete adjoint of a linear explicit time integrator.

**Forward step.** With the state $\mathbf x^n = (\widehat{\mathbf h}^{n-1/2}, \widehat{\mathbf e}^{n})$, $\mathbf A$ = `tDsiDmuiDaC`, $\mathbf B = \mathbf D_\varepsilon \mathbf K$ = `itDaiDepsDstC` ($\mathbf D_\varepsilon = \mathrm{diag}(\varepsilon^{-1})$ = `ieps`) and $\mathbf S = \mathrm{diag}(\Delta t\,\sigma\varepsilon^{-1})$ for ohmic loss, the leapfrog step of Section *Time-Stepping Routine* is

$$
\mathbf h' = \mathbf h - \Delta t\,\mathbf A\,\mathbf e, \qquad
\mathbf e' = (\mathbf 1 - \mathbf S)\,\mathbf e + \Delta t\,\mathbf B\,\mathbf h' - \Delta t\,\mathbf D_\varepsilon \mathbf j_s ,
$$

i.e. $\mathbf x^{n+1} = \mathbf U \mathbf x^n + \mathbf b^n$ with $\mathbf U$ independent of time and of the sources.

**Adjoint step.** For the Lagrangian $F - \sum_n \boldsymbol\lambda^{n+1}\cdot(\mathbf x^{n+1} - \mathbf U\mathbf x^n - \mathbf b^n)$, stationarity with respect to $\mathbf x^n$ gives $\boldsymbol\lambda^n = \mathbf U^T \boldsymbol\lambda^{n+1} + \partial F/\partial \mathbf x^n$, run backwards from $\boldsymbol\lambda^{N_t} = \partial F/\partial \mathbf x^{N_t}$. Transposing the two half steps in reverse order:

$$
\boldsymbol\lambda_h = \boldsymbol\lambda_h' + \Delta t\,\mathbf B^T \boldsymbol\lambda_e', \qquad
\boldsymbol\lambda_e = (\mathbf 1 - \mathbf S)\,\boldsymbol\lambda_e' - \Delta t\,\mathbf A^T \boldsymbol\lambda_h .
$$

The objective sensitivities are added to $\boldsymbol\lambda'$ *before* the transposed step that consumes them: those of $\mathbf e$ sampled at $(n+1)\Delta t$ and of $\mathbf h$ sampled at $(n+\tfrac12)\Delta t$ belong to the state after step $n$.

**Parameter gradient.** $\varepsilon^{-1}$ enters a step only as a factor of the electric increment, $\mathbf q^n = \mathbf e^{n+1} - \mathbf e^n = \Delta t\,\mathbf D_\varepsilon(\mathbf K \mathbf h' - \mathbf j_s - \sigma \mathbf e^n)$ (also with CPML, where the auxiliary terms are inside the same bracket). Hence, for each design edge $i$,

$$
\frac{dF}{d\varepsilon^{-1}_i} = \operatorname{Re}\sum_{n=0}^{N_t-1} \overline{\lambda_{e^*,i}^{n+1}}\, \frac{q_i^n}{\varepsilon^{-1}_i},
$$

where $\boldsymbol\lambda_{e^*}$ is the adjoint of the electric field produced by the solver step itself, before the lumped and dispersive corrections below (it equals $\boldsymbol\lambda_e^{n+1}$ on edges without such corrections, which is always the case on design edges). The conjugate and real part are only active for complex fields. The forward run stores $q^n_i/\varepsilon^{-1}_i$ on the design edges only ($N_t \times N_\text{design}$ values, optionally in float32), which avoids storing or recomputing the whole field history.

**Monitor objectives.** Every frequency-domain output is a running DFT of a fixed sparse linear functional of the state after each step,

$$
X_m(f) = \sum_{n=0}^{N_t-1} w_n(f)\, (\mathbf W \mathbf x^{n+1})_m, \qquad w_n(f) = \Delta t\, e^{-2\pi i f (n\Delta t + \tau)},
$$

with $\tau = \Delta t$ for $\mathbf e$ and $\Delta t/2$ for $\mathbf h$. `AdjointSimulation.forward` returns these accumulators ("raw outputs") and the adjoint run adds $\mathbf W^H\big(\sum_f \overline{w_n(f)}\, G_m(f)\big)$ to $\boldsymbol\lambda^{n+1}$, where $G = \partial F/\partial\operatorname{Re}X + i\,\partial F/\partial\operatorname{Im}X$ (the real part for real fields: $\partial F/\partial x^n = \operatorname{Re}(w_n \bar G)$). The sampling matrices $\mathbf W$ are: a selection of samples (`DFTMonitor`, the tangential E of `FluxMonitor`, `Near2FarMonitor` and `ModeMonitor`); the average $\tfrac12(\mathbf h_\text{above} + \mathbf h_\text{below})$ of the two H planes around the surface (their H accumulators); for a lumped element, the row $-\tfrac12 g_e$ of the voltage $\bar V^{n+1/2} = -\sum_e g_e (E_e^n + E_e^{n+1})/2$ sampled at $\tau = \Delta t/2$, where the $E^n$ half is a functional of the state after step $n-1$ ($\tau = 3\Delta t/2$, not sampled after the last step, $E^0 = 0$), and the row $-(A/\ell)\, g_e$ of the inductor current $I_L$ on the state $J_L^{n+1/2}$. Flux $\Phi = \sum \operatorname{Re}(E\bar H)\,w$, mode amplitudes $a_\pm = \tfrac14\sum(E\times\bar{\mathbf h} \pm \bar{\mathbf e}\times H)\cdot\hat a\,\Delta A$, the far field (linear in the surface currents $-wH$, $-wE$) and the port current $I = I_s - V/Z_0$ (or $V/R + I_L + 2\pi i f C V$ for an element) are explicit functions of the raw outputs, written in JAX (`flux`, `mode_amplitudes`, `farfield`, `power_waves`), so automatic differentiation provides $G$ for objectives such as mode transmission, directivity or $|S_{21}|^2$. JAX's cotangent of a complex output is $\bar G$, which the JAX binding conjugates. Mode profiles are not differentiated (port planes must lie outside the design).

**CPML.** The convolutional variables $\psi$ are part of the state. Each curl term of `_one_step_cpml` has the form $H_T \leftarrow H_T - \Delta t\,\mu_T^{-1} s\,(D E_S + \psi')$ with $\psi' = b\,\psi + c\,(D E_S)|_\text{face}$ (and analogously for $E$ with $\varepsilon^{-1}$ and the dual operators). With $\kappa = -\Delta t\,\mu^{-1}\boldsymbol\lambda_h$, the transposed term is

$$
\tilde\mu = \mu_\psi' + s\,\kappa_T|_\text{face}, \qquad
\mu_\psi = b\,\tilde\mu, \qquad
\boldsymbol\lambda_{e,S} \mathrel{+}= D^T\!\left(s\,\kappa_T + c\,\tilde\mu|^\text{scatter}_\text{face}\right),
$$

applied to the electric half step first (using $\nu = \Delta t\,\varepsilon^{-1}\boldsymbol\lambda_e'$) and then to the magnetic one.

**Complex (Bloch) fields.** With complex states the step is complex-linear, $\mathbf x' = \mathbf U\mathbf x$ with $\mathbf U$ built from the Bloch curls $\mathbf C$ and $\mathbf C^H$ and real diagonal metrics. A real objective is a function of the real pair $(\operatorname{Re}\mathbf x, \operatorname{Im}\mathbf x)$; with the multiplier $\boldsymbol\lambda = \partial F/\partial\operatorname{Re}\mathbf x + i\,\partial F/\partial\operatorname{Im}\mathbf x$, the real transpose of the pair map is the Hermitian transpose, $\boldsymbol\lambda^n = \mathbf U^H\boldsymbol\lambda^{n+1}$: every sparse operator is replaced by its conjugate transpose (`A.conj().T`), the real coefficients ($\varepsilon^{-1}$, $\sigma$, CPML $b$, $c$, lumped and ADE coefficients) are unchanged, and the dot-product identity holds for the complex inner product $\langle\mathbf a,\mathbf b\rangle = \sum\bar a_k b_k$. Since $\varepsilon^{-1}$ is real, $dF/d\varepsilon^{-1} = \operatorname{Re}\sum_n \bar\lambda\, q/\varepsilon^{-1}$ (above). Positive and negative frequencies are independent outputs.

**Impressed currents.** $\mathbf j_s$, the magnetic current $\mathbf m$ ($\mathbf h' = \mathbf h - \Delta t\,\mathbf A\mathbf e - \Delta t\,\mathbf D_\mu\mathbf m$) and the J/M sheets of `ModeSource` are in $\mathbf b^n$: they do not depend on the state and drop out of $\mathbf U^H$. Nothing design-dependent enters them as long as the mode-source plane lies outside the design region (the mode profile is solved once from the permittivity of that plane).

**Lumped elements.** Each element modifies its edges $e$ after the solver step (Section *Lumped elements*): $J_L' = J_L + \Delta t\,\Gamma E_e$ enters $\mathbf j$ before the step, $E^*_e$ is the solver result and $E'_e = a E_e + b (E^*_e - E_e)$. The inductor current $J_L$ is part of the state. Transposing in reverse order, with $\lambda'$ the multipliers after the step:

$$
\lambda_{e^*,e} = b\,\lambda'_{E,e}, \qquad
\lambda_{J_L} = \lambda'_{J_L} - \Delta t\,\varepsilon^{-1}_e\,\lambda_{e^*,e}, \qquad
\lambda_{E,e} \mathrel{+}= (a-b)\,\lambda'_{E,e} + \Delta t\,\Gamma\,\lambda_{J_L},
$$

where the solver-step transpose is applied to $(\lambda'_H, \lambda_{e^*})$; the second relation is the transpose of the $-\Delta t\,\varepsilon^{-1}\mathbf j$ term. The port source current is in $\mathbf b^n$. `AdjointSimulation` runs the forward loop of `emsolve` (`update`, `one_step`, `post_step`, monitors).

**Dispersive media.** On the edges of a dispersive group (Section *Dispersive materials*) the correction is $\mathrm{acc} = (E^*_e - E_n)\varepsilon_s + a_1 E_n - w\sum_L(p'_L - p_L) - w\sum_D(\kappa-1)p_D$, $p'_L = \alpha p_L + \xi p_{L,\text{old}} + \zeta E_n$, $p'_{L,\text{old}} = p_L$, $E'_e = \mathrm{acc}/d$, $p'_D = \kappa p_D + \beta(E'_e + E_n)$, with $E_n = E^n_e$ and the polarizations in the state. The transpose is

$$
\begin{aligned}
\lambda_\text{acc} &= \Big(\lambda'_{E,e} + \sum_D \beta\lambda'_{p_D}\Big)/d, \qquad \lambda_{e^*,e} = \varepsilon_s\,\lambda_\text{acc},\\
\lambda_{p_D} &= \kappa\lambda'_{p_D} - w(\kappa-1)\lambda_\text{acc}, \qquad \tilde\lambda_L = \lambda'_{p_L} - w\lambda_\text{acc},\\
\lambda_{p_L} &= \alpha\tilde\lambda_L + \lambda'_{p_{L,\text{old}}} + w\lambda_\text{acc}, \qquad \lambda_{p_{L,\text{old}}} = \xi\tilde\lambda_L,\\
\lambda_{E,e} &\mathrel{+}= (a_1 - \varepsilon_s)\lambda_\text{acc} + \sum_D\beta\lambda'_{p_D} + \sum_L\zeta\tilde\lambda_L .
\end{aligned}
$$

**Dispersive design.** `MaterialGrid(..., material=disp)` makes the fill fraction $w = \mathbf P\rho$ of a dispersive material over the background $\varepsilon_\text{min}$ the design variable (as a partially filled edge of `add_dispersive_material`): $\varepsilon_e = (1-w)\varepsilon_\text{min} + w\varepsilon_\infty$, $a_1 = \varepsilon_e - w s$, $d = \varepsilon_e + w s$ with $s = \sigma\Delta t/(2\varepsilon_0) + \sum_D\beta$, and the pole terms scale with $w$. $w$ enters only the correction, so

$$
\frac{dF}{dw_i} = \operatorname{Re}\sum_n \overline{\lambda_{\text{acc},i}^{n+1}}\Big[(\Delta\varepsilon - s)E^n_i - \sum_L (p'_L - p'_{L,\text{old}})_i - \sum_D(\kappa-1)p_{D,i}^n - (\Delta\varepsilon + s)E^{n+1}_i\Big],
$$

$\Delta\varepsilon = \varepsilon_\infty - \varepsilon_\text{min}$ (derivative of acc minus $E' \partial d/\partial w$); the bracket is stored per step on the design edges, and $dF/d\rho = \mathbf P^T dF/dw$. Stability is checked for the pure material; intermediate $w$ mixes two stable media, which is stable in all tested cases but not proven.

**Design parametrization.** `MaterialGrid` maps a density $\rho \in [0,1]$ on a regular design grid to $\varepsilon_r = \varepsilon_\text{min} + (\varepsilon_\text{max}-\varepsilon_\text{min})\,\mathbf P\rho$ on the design edges, with $\mathbf P$ the sparse trilinear interpolation matrix, so

$$
\frac{dF}{d\rho} = (\varepsilon_\text{max}-\varepsilon_\text{min})\,\mathbf P^T\!\left[\frac{dF}{d\varepsilon^{-1}} \cdot \left(-\frac{1}{\varepsilon_0\varepsilon_r^2}\right)\right].
$$

Filtering and projection are written in JAX: `conic_filter` [Bourdin, 2001], `gaussian_filter` and `tanh_projection` [Wang et al., 2011], so that jax.grad composes them with the simulation (`jax_simulation`, a `jax.custom_vjp` around the NumPy solver).

**Validation** (`tests/test_020_adjoint.py`): the dot-product identity $\langle \mathbf U\mathbf x, \mathbf y\rangle = \langle \mathbf x, \mathbf U^T\mathbf y\rangle$ holds to $\sim 10^{-17}$ relative to $\sum|U x||y|$ for PEC, periodic, lossy and CPML steps; adjoint gradients match central finite differences to $10^{-8}$–$10^{-10}$ with $h=10^{-4}$, and the difference scales as $h^2$, i.e. it is the finite-difference truncation error. `tests/test_026_adjoint_features.py` repeats the dot-product test against the real time loop (element `update`, `one_step`, `post_step`), including the auxiliary states, for magnetic currents, lumped R/L/C elements and ports (PEC and CPML), Lorentz + Drude + Debye media with conductivity (PEC and CPML) and complex Bloch fields (PEC and CPML): $\le 1.4\times10^{-16}$ relative. Finite-difference checks ($h = 10^{-4}$) of jax.grad through flux, far-field intensity, mode amplitudes ($a_+$, $a_-$, with a `ModeSource`), DFT fields with a magnetic source, $|S_{21}|^2$ plus a lumped RLC current, a dispersive medium beside the design, a dispersive design material and Bloch fields (including $-f_0$) agree to $10^{-10}$–$2\times10^{-8}$ relative.

**Limitations.** The adjoint supports CPU, non-MPI runs with PEC/PMC/periodic/Bloch/CPML boundaries, ohmic loss, dispersive media, lumped elements and ports, and electric, magnetic and mode sources. ABC (Mur), PML, TFSF injection, SIBC, GPU and MPI raise `NotImplementedError`, as do `J` components of `DFTMonitor`, lumped elements sharing edges with each other or with dispersive media, and design regions overlapping lumped elements or (other) dispersive media. Only isotropic $\varepsilon$ (or the fill fraction of one dispersive material) is designed; mode profiles and $\Delta t$ must not depend on the design.

References:
- M. B. Giles and N. A. Pierce, "An introduction to the adjoint approach to design", *Flow, Turbulence and Combustion* **65**, 393–415 (2000).
- S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vučković and A. W. Rodriguez, "Inverse design in nanophotonics", *Nat. Photonics* **12**, 659–670 (2018), [doi:10.1038/s41566-018-0246-9](https://doi.org/10.1038/s41566-018-0246-9).
- T. W. Hughes, I. A. D. Williamson, M. Minkov and S. Fan, "Forward-mode differentiation of Maxwell's equations", *ACS Photonics* **6**, 3010–3016 (2019), [arXiv:1908.10507](https://arxiv.org/abs/1908.10507).
- B. Bourdin, "Filters in topology optimization", *Int. J. Numer. Methods Eng.* **50**, 2143–2158 (2001), [doi:10.1002/nme.116](https://doi.org/10.1002/nme.116).
- F. Wang, B. S. Lazarov and O. Sigmund, "On projection methods, convergence and robust formulations in topology optimization", *Struct. Multidiscip. Optim.* **43**, 767–784 (2011), [doi:10.1007/s00158-010-0602-y](https://doi.org/10.1007/s00158-010-0602-y).

### 🧰 Design templates: setup rules and design formulas

`wakis.templates` (user guide: [templates.md](templates.md)) builds complete simulations from a declarative description. It adds no new field update; this section documents the automatic setup rules, the material assignment on the FIT edges and the closed-form design formulas used to configure and validate the templates.

**Setup rules.** The cell size in a material $(\varepsilon_r, \mu_r)$ is $\Delta \le c/(f_\text{max}\sqrt{\varepsilon_r\mu_r}\,N)$ with $N$ cells per wavelength; the leapfrog phase error per wavelength is $\approx(\pi/N)^2/6$ for small $\Delta$ [Taflove & Hagness, 2005, Ch. 4], 0.4 % for $N=20$. Mesh axes are piecewise uniform with nodes on all structure faces and lumped-port gaps (a tensor-product mesh): each interval between faces gets the size required by the densest material whose extent covers it, at least `min_cells_per_feature` cells, and the size field is limited to grow by a factor `grading` per cell. The distance to the CPML is $\lambda(f_\text{min})/4$ and the CPML has $\max(10, \lambda(f_\text{min})/(8\Delta))$ cells (polynomial grading of order 4, $R_0 = 10^{-8}$). The run stops with the energy criterion of `emsolve(stop_when_energy_decayed=...)`.

**Material assignment.** Solids are written directly into the inverse-permittivity tensor at the staggered E-edge positions (E along its edge midpoint, on the nodes transversally). PEC removes every edge in the closed solid (`ieps = 0`), so a zero-thickness box is a metal sheet and a zero-radius cylinder a wire along one line of edges. Dielectric edges lying on a face get the arithmetic mean of the two permittivities (tangential $\mathbf E$ continuous), as in `add_dispersive_material`; on periodic axes faces on the cell boundary are matched with their periodic image. Dispersive boxes use the ADE update of `add_dispersive_material` on their interior edges, while their face edges keep the averaged permittivity at the band centre. The STL import of `GridFIT3D` is not used because its cell masks attach a cell to the three edges leaving its lower corner, which cannot represent zero-thickness metal.

**Graded meshes (workaround).** The update of an E sample on node $i$ needs the dual edge length centred on that node, $\tilde L_i = (x_{i+1}-x_{i-1})/2$. `GridFIT3D` currently stores $\tilde L_i = (x_{i+2}-x_i)/2$, i.e. the dual length of node $i+1$, which is exact only on uniform meshes; on a graded mesh it acts as a local material error of the order of the grading ratio minus one. `Simulation.build` overwrites `grid.tL` and `grid.itA` with the node-centred values (unchanged on uniform meshes) and meshes periodic axes uniformly. With this correction a slab on a graded mesh reproduces the Airy transmittance to 0.016, as on a uniform mesh, instead of 0.059.

**Metal sheet edges.** The field of a zero-thickness PEC sheet ending on a mesh node fringes as if the sheet extended about a third of a cell further. For a patch antenna this lowers the resonance at first order in the cell size (observed: $\delta \approx 0.37\Delta$ per edge from a two-mesh study). `patch_antenna` therefore meshes the patch smaller by $\Delta/3$ per edge (`edge_correction`); this empirical correction is a Wakis choice validated by the mesh study below, not a published method. `patch_antenna(thin_sheet=True)` replaces it by a conformal sheet of the physical patch size with the knife-edge correction (see "Conformal FIT"), which needs no empirical parameter: −3.4 %, −2.4 % and −3.1 % vs TLM at 15, 20 and 30 cells/λ. This is within about 1 % of the Δ→0 extrapolation (−2.3 %), against −6 to −7 % for the uncorrected staircased patch, but not better than the empirical correction (−2.5 %, −2.7 %). The remaining scatter comes from the patch corners and from the edges carrying current along them (see "Conformal FIT"), so `thin_sheet` stays opt-in.

**Patch antenna (transmission-line model)** [Balanis, 2016, Sec. 14.2]. For substrate permittivity $\varepsilon_r$ and height $h$, design frequency $f_r$:

$$
W = \frac{c}{2f_r}\sqrt{\frac{2}{\varepsilon_r+1}},\qquad
\varepsilon_\text{eff} = \frac{\varepsilon_r+1}{2} + \frac{\varepsilon_r-1}{2}\left(1+12\frac{h}{W}\right)^{-1/2},
$$

$$
\frac{\Delta L}{h} = 0.412\,\frac{(\varepsilon_\text{eff}+0.3)(W/h+0.264)}{(\varepsilon_\text{eff}-0.258)(W/h+0.8)},\qquad
L = \frac{c}{2f_r\sqrt{\varepsilon_\text{eff}}} - 2\Delta L ,
$$

the fringing extension $\Delta L$ being Hammerstad's formula. The edge resistance is $R_\text{in}(0) = 1/(2(G_1+G_{12}))$ with

$$
G_1 = \frac{I_1}{120\pi^2},\quad I_1 = \int_0^\pi \left[\frac{\sin(\tfrac{k_0W}{2}\cos\theta)}{\cos\theta}\right]^2\sin^3\theta\,d\theta,\qquad
G_{12} = \frac{1}{120\pi^2}\int_0^\pi \left[\frac{\sin(\tfrac{k_0W}{2}\cos\theta)}{\cos\theta}\right]^2 J_0(k_0L\sin\theta)\sin^3\theta\,d\theta,
$$

and a probe or inset feed at distance $y_0$ from the radiating edge sees $R_\text{in}(y_0) = R_\text{in}(0)\cos^2(\pi y_0/L)$. The two-slot directivity is $D = D_0\,2/(1+g_{12})$ with $D_0 = (k_0W)^2/I_1$ and $g_{12} = G_{12}/G_1$. These are engineering approximations: the TLM resonance is typically within a few percent of measurements, and the edge resistance is less accurate.

**Dipole (induced EMF)** [Balanis, 2016, Sec. 8.5]. A centre-fed dipole of length $l$ and radius $a$ with sinusoidal current has $Z_\text{in} = (R_r + jX_m)/\sin^2(kl/2)$,

$$
R_r = \frac{\eta}{2\pi}\Big\{C + \ln kl - \text{Ci}(kl) + \tfrac12\sin kl\,[\text{Si}(2kl)-2\text{Si}(kl)] + \tfrac12\cos kl\,[C+\ln(kl/2)+\text{Ci}(2kl)-2\text{Ci}(kl)]\Big\},
$$

$$
X_m = \frac{\eta}{4\pi}\Big\{2\text{Si}(kl) + \cos kl\,[2\text{Si}(kl)-\text{Si}(2kl)] - \sin kl\,[2\text{Ci}(kl)-\text{Ci}(2kl)-\text{Ci}(2ka^2/l)]\Big\},
$$

with $C$ Euler's constant; $l=\lambda/2$ gives $73.1+42.5j\,\Omega$, and the reactance vanishes at $l\approx0.47$–$0.48\lambda$ for thin wires.

**Waveguides.** TE10 cutoff $f_c = c/(2a)$ and $\beta = \sqrt{k^2-(\pi/a)^2}$ [Pozar, 2012, Ch. 3]. A thin symmetric inductive diaphragm of opening $d$ is approximately a shunt susceptance $B/Y_0 = -(\lambda_g/a)\cot^2(\pi d/2a)$, so $S_{11} = -jb/(2+jb)$ [Marcuvitz, 1951; Pozar, 2012].

**Thin films.** Normal-incidence $R$ and $T$ of a slab of index $n$ and thickness $d$ between media of index $n_0$ follow from the characteristic matrix $\begin{pmatrix}\cos\delta & j\sin\delta/n\\ jn\sin\delta & \cos\delta\end{pmatrix}$, $\delta = k_0nd$ [Born & Wolf, 1999, Sec. 1.6]; for a lossless slab in vacuum $T = 1/(1+\tfrac14(n-1/n)^2\sin^2\delta)$ (Airy).

**Slab waveguide.** The fundamental TE mode of a symmetric slab solves $\tan(\kappa d/2) = \gamma/\kappa$, $\kappa = k_0\sqrt{n_1^2-n_\text{eff}^2}$, $\gamma = k_0\sqrt{n_\text{eff}^2-n_2^2}$ [Snyder & Love, 1983].

**Validation** (`tests/test_027_templates.py`; fast tests < 25 s each):

| Template | Mesh | Reference | Result |
| --- | --- | --- | --- |
| Dielectric slab cell ($\varepsilon_r=4$, 10 mm, 2–12 GHz) | 20 cells/λ, graded | Airy | $\lvert\Delta T\rvert, \lvert\Delta R\rvert < 0.017$, $\lvert R+T-1\rvert < 4\times10^{-3}$ |
| PEC patch array (p = 15 mm, 12 mm patches) | 20 cells/λ | qualitative | $R = 0.20$ at $0.2c/p$, $R = 0.995$ at $0.95c/p$, $\lvert R+T-1\rvert < 4\times10^{-4}$ |
| WR-90, empty, 1.25–1.9 $f_c$ | 20 cells across $a$ | $S_{21}=1$, TE10 $\beta$ | $\lvert\lvert S_{21}\rvert-1\rvert < 0.011$, $\lvert S_{11}\rvert < 3\times10^{-3}$, $\beta$ = discrete dispersion to $10^{-3}$ |
| WR-90, 12 mm inductive iris | same | lossless, $B/Y_0$ model | $\lvert S_{11}\rvert^2+\lvert S_{21}\rvert^2 = 1 \pm 0.015$, $\lvert S_{11}\rvert$ within 0.05 of the model |
| Si slab 220 nm in SiO2, 1395–1705 nm (Sellmeier Si) | 11 cells across the core | slab eigenvalue equation | $T = 1\pm1.3\times10^{-3}$, $n_\text{eff}$ within 0.25 % |
| Half-wave dipole, zero-radius wire | 20/30 and 40/60 cells/λ | 0.47–0.48 λ, 73 Ω, 2.15 dBi | resonant $L/\lambda$ = 0.434/0.445 → 0.466 and 0.452/0.460 → 0.476 extrapolated; $R_\text{in} = 72.6$–$72.9\,\Omega$; $D = 2.10$–$2.20$ dBi |
| Patch, $\varepsilon_r=2.2$, $h=3.2$ mm | 15 cells/λ, 2 across $h$ | TLM | resonance −3.1 %, $\lvert S_{11}\rvert = -12$ dB, $D = 8.0$ dBi |
| Patch, FR-4 4.4, $h=1.6$ mm (slow) | 20 / 30 cells/λ, 3 / 4 across $h$ | TLM (resonance, 50 Ω feed), 6.0 dBi | resonance −2.5 % / −2.7 %, $\lvert S_{11}\rvert = -11.0$ dB, $D = 6.4$ / $6.5$ dBi |

Without the edge correction the FR-4 patch resonates 7.3 % (20 cells/λ) and 6.0 % (30 cells/λ) below the TLM value, which extrapolates to −2.3 % for $\Delta\to0$, consistent with the corrected results.

**Differentiability.** The templates only configure existing features; their outputs ($S$-parameters, flux ratios, mode amplitudes, far-field directivity) are the plain-array outputs of `LumpedPort`, `FluxMonitor`, `ModeMonitor` and `Near2FarMonitor`, whose adjoint sources are documented in their sections. The graded-mesh dual lengths enter the operators as fixed metric coefficients. The optimizer wrapper is gradient-free (`scipy.optimize`); gradient-based optimization of a template geometry would go through `wakis.inverse` with a `MaterialGrid` design region.

References:
- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley (2016): Sec. 8.5 (induced-EMF dipole impedance), Sec. 14.2 (rectangular patch, transmission-line and cavity models).
- E. O. Hammerstad, "Equations for microstrip circuit design", *Proc. 5th European Microwave Conference*, 268–272 (1975), [doi:10.1109/EUMA.1975.332206](https://doi.org/10.1109/EUMA.1975.332206).
- D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley (2012).
- N. Marcuvitz, *Waveguide Handbook*, MIT Radiation Laboratory Series vol. 10, McGraw-Hill (1951).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press (1999).
- A. W. Snyder and J. D. Love, *Optical Waveguide Theory*, Chapman & Hall (1983).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).

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
