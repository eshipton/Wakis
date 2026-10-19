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
- With MPI only the last rank carries the $z$ ghost cell; `mpi_gather` and `mpi_gather_asField` return `grid.NZ_samples` $= N_z+1$ samples along $z$, the same layout as a serial run, and MPI runs reproduce the serial fields to round-off.

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

**Discrete Poynting theorem in the time domain.** Because the dual curl is the transpose of the primary curl, the leapfrog scheme conserves the energy $W^n = \tfrac12\sum \varepsilon V_e (E^n)^2 + \tfrac12\sum \mu V_h H^{n-1/2}H^{n+1/2}$ exactly. Restricted to a box (control volumes $V_e$, $V_h$ cut by the box faces, i.e. weights 1/2 on the faces and 1/4 on the edges), the balance per step is

$$
\frac{W^{n+1}-W^n}{\Delta t} + \Phi^{n+1/2} + \sum \sigma E^n \bar E\, V_e = -\sum J^{n+1/2}\bar E\, V_e, \qquad \bar E = \tfrac12(E^n + E^{n+1}),
$$

where $\Phi^{n+1/2}$ is exactly the surface sum used by `FluxMonitor` (tangential $\bar E$ on the face nodes times $H^{n+1/2}$ averaged across the face, trapezoidal weights on the box edges) and the ohmic term carries the explicit $\sigma E^n$ of the update. `tests/test_029_energy_balance.py` verifies the identity to $10^{-14}$ in a conducting medium where radiation and ohmic loss each take about half of the work done by a dipole, and the plane-wave attenuation $\alpha = -\operatorname{Im}k$, $k = \omega\sqrt{\mu_0\varepsilon_0(1 - i\sigma/\omega\varepsilon_0)}$, to 1 %.

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

### 🔔 Resonances by harmonic inversion: `harminv` and `ResonanceMonitor`

After a broadband excitation has ended, a field probe in a resonator or open cavity rings down as a sum of damped exponentials, one per resonance in the band (the eigenmode content of CST, or Harminv in Meep):

$$
x(t) = \sum_k a_k\, e^{2\pi i \tilde f_k t}, \qquad \tilde f_k = f_k + i g_k, \qquad Q_k = \frac{f_k}{2 g_k} .
$$

The phasor convention is that of the rest of Wakis ($e^{+i\omega t}$), so decaying modes have $\operatorname{Im}\tilde f_k = g_k > 0$; the field decays as $e^{-2\pi g_k t}$ (`Resonances.decay_rate` $= 2\pi g_k$) and the stored energy as $e^{-\omega_k t/Q_k}$. A real signal contains each mode at $\pm f_k$; only $f_k$ inside the window is returned, and the real signal is $2\operatorname{Re}[a_k e^{2\pi i\tilde f_k t}]$. Fourier transforming the ring-down resolves $f_k$ only to $\sim 1/T$ for a record of length $T$ and gives $Q$ only through a Lorentzian fit to a record much longer than the decay time; harmonic inversion fits the model directly and reaches round-off precision for noiseless signals, independently of $T$ as long as the modes are resolved by the information content of the signal.

**Filter diagonalization** (`method="fdm"`, default) [Wall & Neuhauser, 1995; Mandelshtam & Taylor, 1997]. With samples $c_n = x(t_0 + n\Delta t)$, $n = 0\dots 2M+2$, the signal is written as $c_n = \sum_k d_k u_k^n$, $u_k = e^{2\pi i\tilde f_k\Delta t}$, i.e. as an autocorrelation $c_n = (\Phi_0, \hat U^n \Phi_0)$ of a propagator $\hat U$ with eigenvalues $u_k$. In a basis of $K$ windowed Fourier vectors with $z_j = e^{2\pi i\varphi_j\Delta t}$, $\varphi_j$ uniform in the window (spacing $\approx 1/(1.1\,(M+1)\Delta t)$, a little finer than the Fourier resolution of the half record), the matrix elements

$$
U^{(p)}_{jj'} = \sum_{n,m=0}^{M} z_j^{-n} z_{j'}^{-m}\, c_{n+m+p}, \qquad p = 0, 1, 2,
$$

have closed forms costing $O(N)$ per basis vector [Mandelshtam & Taylor, 1997]. The generalized eigenproblem $U^{(1)}B_k = u_k U^{(0)}B_k$ gives the $u_k$; with $B_k^T U^{(0)} B_k = 1$ the amplitudes are $d_k = (\sum_j B_{jk} \sum_{n=0}^M c_n z_j^{-n})^2$, and the error estimate compares $u_k^2$ with $B_k^T U^{(2)} B_k$, reported as a frequency error $|B_k^TU^{(2)}B_k/u_k^2 - 1|/(4\pi\Delta t)$. $U^{(0)}$ is often near-singular (more basis vectors than modes), so Wakis solves the eigenproblem in the span of its singular vectors above `rcond` times the largest singular value (a regularization of the original method; set `rcond` near the relative noise level). The cost is $O(NK + K^3)$, so long, finely sampled FDTD records need no decimation. Modes outside the window leak into it; the basis extends 10 % beyond the window to keep modes near the edges accurate.

**Matrix pencil** (`method="pencil"`) [Hua & Sarkar, 1990]: the Hankel matrix $Y_{jl} = c_{j+l}$ with $L+1$ columns ($L \approx 0.4N$) has rank equal to the number of exponentials; the right singular vectors $V$ (rank set by `rcond`) satisfy $V_{2} = V_{1}T^{-1}\operatorname{diag}(u)T$ for the blocks without the last/first row, so the $u_k$ are the eigenvalues of $V_1^+ V_2$ and the amplitudes follow from least squares. The cost is $O(N^3)$: use it for short (decimated) records or as an independent cross-check. Its error column is the Cramér–Rao bound of a single undamped tone [Rife & Boorstyn, 1974] with the fit residual as noise, a rough guide only.

Modes outside $[f_\text{min}, f_\text{max}]$, with error above `max_rel_error` times the window width, or with amplitude below `min_rel_amplitude` times the largest are discarded (spurious fits of noise have large errors and small amplitudes).

**Usage in simulations.** `ResonanceMonitor(component, center, size, t_start)` records one component (or the sum over a small region) at its native time level after every step for $t \ge t_\text{start}$; start after the source has ended so that the record is a free ring-down. Place the probe away from nodal lines of the modes of interest and excite with a source whose spectrum covers the band. For high-$Q$ modes a record of a few periods of the beat between neighbouring modes suffices; low-$Q$ modes need the record to start before they have decayed below the noise.

**Validation** (`tests/test_028_resonances.py`):
- synthetic signals (two modes $0.3/T$ apart, i.e. below the Fourier resolution, plus a $Q=200$ mode): both methods recover $f$ to $10^{-10}$, $Q$ to $10^{-6}$ and the complex amplitudes to $10^{-8}$; with white noise of $10^{-5}$ of the peak the isolated mode keeps $f$ to $10^{-8}$ and $Q$ to $10^{-4}$, and the close pair stays separated;
- PEC box $100\times80\times60$ mm, 5 mm cells: all 8 modes with $E_z \ne 0$ in 2–5 GHz match the leapfrog dispersion relation (including $\Delta t$) to $10^{-9}$ (observed $\sim 10^{-12}$), and $Q > 10^9$;
- the same box filled with $\sigma = 10^{-3}$ S/m: each mode obeys the exact discrete relation $z^2 + (\Omega^2\Delta t^2 - 2 + s)z + 1 - s = 0$, $z = e^{2\pi i\tilde f\Delta t}$, $s = \sigma\Delta t/\varepsilon_0$, obtained by eliminating $H$ from the updates with the explicit loss term $\sigma E^n$ ($Q$ to $10^{-6}$), and the continuum $Q = \omega\varepsilon_0/\sigma$ to $O(s) = 5\cdot10^{-4}$;
- a square PEC cavity loaded with a dielectric slab whose face lies between grid nodes (exact transcendental reference): with subcell averaging of $\varepsilon$ the error falls at second order ($-0.23\,\%, -0.060\,\%, -0.017\,\%$ at 15, 30, 60 cells), staircasing is first order and ten times larger;
- radiation-limited TM$_{4}$ whispering-gallery mode of a 2D dielectric disk ($n=3$, $R = 100$ mm) against the exact complex root of $n J_4'(nkR)/J_4(nkR) = H_4^{(2)\prime}(kR)/H_4^{(2)}(kR)$ ($f = 0.9902$ GHz, $Q = 164.2$): with subcell averaging of $\varepsilon$ the errors are $-0.34\,\%, -0.13\,\%, -0.08\,\%$ in $f$ and $+2.6\,\%, +0.85\,\%, +0.5\,\%$ in $Q$ at 6, 4, 3 mm (second order), from a 60 ns record ($\approx$ one decay time); a Lorentzian fit to the spectrum of a 400 ns ring-down agrees with harminv to $5\cdot10^{-4}$ in $Q$.

**Differentiability.** The extracted $\tilde f_k$ are smooth functions of the recorded samples away from mode crossings, but the eigen-decomposition is not part of the adjoint layer. For inverse design of resonances, use objectives on DFT data instead (e.g. $|E(f_0)|^2$ near the target frequency, or the ratio of field energy at $f_0$ to the source spectrum) and use `harminv` to analyse the optimized structure.

References:
- M. R. Wall and D. Neuhauser, "Extraction, through filter-diagonalization, of general quantum eigenvalues or classical normal mode frequencies from a small number of residues or a short-time segment of a signal. I. Theory and application to a quantum-dynamics model", *J. Chem. Phys.* **102**, 8011–8022 (1995), [doi:10.1063/1.468999](https://doi.org/10.1063/1.468999).
- V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and its applications", *J. Chem. Phys.* **107**, 6756–6769 (1997), [doi:10.1063/1.475324](https://doi.org/10.1063/1.475324).
- Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise", *IEEE Trans. Acoust., Speech, Signal Process.* **38**(5), 814–824 (1990), [doi:10.1109/29.56027](https://doi.org/10.1109/29.56027).
- D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from discrete-time observations", *IEEE Trans. Inf. Theory* **20**(5), 591–598 (1974), [doi:10.1109/TIT.1974.1055282](https://doi.org/10.1109/TIT.1974.1055282).
- D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley (2012), Sec. 6.3 (rectangular cavity modes and $Q$ of a lossy filling).

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

### 🎯 Total-field/scattered-field plane-wave source: `TFSFSource`

`TFSFSource` launches an axis-aligned plane wave into a box. Inside the box the solver holds the total field $\psi = \psi^{inc} + \psi^{sca}$; outside it holds only the scattered field. The incident field is

$$
\mathbf{E}^{inc}(\mathbf{r},t) = A\, s\!\left(t - \frac{\hat{\mathbf{k}}\cdot(\mathbf{r}-\mathbf{r}_0)}{v}\right)\hat{\mathbf{p}},\qquad
\mathbf{H}^{inc} = \frac{\hat{\mathbf{k}}\times\mathbf{E}^{inc}}{\eta},\qquad v = \frac{c}{\sqrt{\varepsilon_r}},\ \eta = \sqrt{\frac{\mu_0}{\varepsilon_0\varepsilon_r}},
$$

with $\mathbf{r}_0$ on the box face where the wave enters. A scatterer of any material (conformal, tensor-averaged, dispersive, lossy) inside the box sees the plane wave. A closed `FluxMonitor` outside the box measures the scattered power, and one inside measures the extinction minus scattering, i.e. minus the absorbed power. So $\sigma_{sca}(f) = \Phi_{sca}(f)/\Phi_{inc}(f)$, with $\Phi_{inc}$ from `incident_flux_density(f)`.

```python
src = wakis.TFSFSource(wakis.sources.GaussianSource(f0, df), size=(Lx, Ly, Lz),
                       direction="+z", polarization="x")
src.prepare(solver)                       # snaps the box to grid nodes
(x0, x1), (y0, y1), (z0, z1) = src.box_bounds()
scat = wakis.FluxMonitor(f, size=(x1 - x0 + 6 * h, ...))   # box in the SF region
solver.emsolve(Nt, source=[src], monitors=[scat])
sigma_sca = scat.get_flux() / src.incident_flux_density(f)  # [m^2]
```

**The equivalence principle, continuous and discrete.** Oskooi & Johnson [2013, §4.2.1] write Maxwell's equations as $\mathcal{M}\psi = \partial_t(\psi + \chi*\psi) + \xi$ for the six-vector $\psi = (\mathbf{E},\mathbf{H})$ and currents $\xi = (\mathbf{J},\mathbf{K})$. The field that equals $\psi^{inc}$ in a region $\Omega$ and vanishes outside solves the source-free equations away from $\partial\Omega$. On $\partial\Omega$ its jump creates a surface delta function in $\mathcal{M}\psi$, which the surface currents of their Eq. (4.4) cancel:

$$
\mathbf{J}_s = \hat{\mathbf{n}}\times\mathbf{H}^{inc},\qquad \mathbf{K}_s = -\hat{\mathbf{n}}\times\mathbf{E}^{inc}\qquad(\hat{\mathbf{n}}\ \text{the inward normal}).
$$

They also note: "Essentially the same derivation applied to the discretized curl operator, M, yields the source terms in the total-field/scattered field approach [1], with the source currents being a discrete or “Kronecker” delta function" (p. 68; [1] is Taflove & Hagness). Repeating that derivation on the FIT matrices gives the form Wakis implements.

Write the solver step (see *Time-Stepping Routine*) as

$$
\mathbf{h}^{n+1/2} = \mathbf{h}^{n-1/2} - \Delta t\,\mathbf{A}\,\mathbf{e}^{n},\qquad
\mathbf{e}^{n+1} = \mathbf{e}^{n} + \Delta t\,\mathbf{B}\,\mathbf{h}^{n+1/2} - \Delta t\,\mathbf{M}_\varepsilon^{-1}\mathbf{j}^{n+1/2},
$$

with $\mathbf{A} = \mathbf{D}_A^{-1}\mathbf{M}_\mu^{-1}\mathbf{C}\mathbf{D}_s$ and $\mathbf{B} = \mathbf{M}_\varepsilon^{-1}\tilde{\mathbf{D}}_A^{-1}\mathbf{C}^{T}\tilde{\mathbf{D}}_s$. Let $\mathbf{P}_e$ and $\mathbf{P}_h$ be the diagonal 0/1 matrices that select the total-field samples: an E or H sample belongs to the box if its position (edge midpoint or node, see *Grid conventions*) lies in the closed box. The stored state is

$$
\mathbf{u} = \mathbf{P}\,\psi^{tot} + (\mathbf{1}-\mathbf{P})\,\psi^{sca} = \psi^{tot} - (\mathbf{1}-\mathbf{P})\,\psi^{inc}.
$$

Two assumptions are made:
1. $\psi^{tot}$ obeys the solver update everywhere.
2. $\psi^{inc}$ obeys the background update $\mathbf{A}_0, \mathbf{B}_0$, which equals $\mathbf{A}, \mathbf{B}$ on every row that couples across $\partial\Omega$ (homogeneous background at the box surface).

Then for H rows inside the box $\mathbf{u}$ must be updated with $\mathbf{A}\psi^{tot} = \mathbf{A}\mathbf{u} + \mathbf{A}(\mathbf{1}-\mathbf{P}_e)\psi^{inc}$. For rows outside, $\psi^{sca} = \psi^{tot}-\psi^{inc}$ gives $\mathbf{A}\mathbf{u} - \mathbf{A}\mathbf{P}_e\psi^{inc}$. Combining both cases:

$$
\boxed{\ \mathbf{h}^{n+1/2} \mathrel{+}= -\Delta t\,\big(\mathbf{P}_h\mathbf{A} - \mathbf{A}\mathbf{P}_e\big)\,\mathbf{e}^{inc,n},\qquad
\mathbf{e}^{n+1} \mathrel{+}= \Delta t\,\big(\mathbf{P}_e\mathbf{B} - \mathbf{B}\mathbf{P}_h\big)\,\mathbf{h}^{inc,n+1/2}\ }
$$

These are the commutators of the curl operators with the region masks. They are nonzero only where $\mathbf{A}$ or $\mathbf{B}$ couples a sample inside the box to one outside, i.e. on the two sample layers at the box surface. They are the discrete equivalent currents: $-\Delta t\,\mathbf{M}_\mu^{-1}\mathbf{k}_s$ with $\mathbf{k}_s = \mathbf{D}_A^{-1}(\mathbf{P}_h\mathbf{C}-\mathbf{C}\mathbf{P}_e)\mathbf{D}_s\,\mathbf{e}^{inc}$, and $-\Delta t\,\mathbf{M}_\varepsilon^{-1}\mathbf{j}_s$ with $\mathbf{j}_s = -\tilde{\mathbf{D}}_A^{-1}(\mathbf{P}_e\mathbf{C}^T-\mathbf{C}^T\mathbf{P}_h)\tilde{\mathbf{D}}_s\,\mathbf{h}^{inc}$. For a single face these reduce to $\mathbf{K}_s = -\hat{\mathbf{n}}\times\mathbf{E}^{inc}$ on the H plane half a cell outside the face and $\mathbf{J}_s = \hat{\mathbf{n}}\times\mathbf{H}^{inc}$ on the E plane of the face. The box edges and corners need no special treatment, because the commutator handles any mask.

**Exact discrete cancellation.** Substituting $\mathbf{u} = 0$ outside and $\mathbf{u} = \psi^{inc}$ inside shows that the scattered region stays exactly zero, and the total field stays exactly $\psi^{inc}$, provided $\psi^{inc}$ satisfies the discrete background update on the total-field rows. That is the condition Oskooi & Johnson state in §4.2.2: "even if the incident field ψ⁺ solves the exact Maxwell's equations, it will not solve the discretized Maxwell's equations without modification" (p. 69). `TFSFSource` therefore uses an auxiliary 1D FIT line (their "bootstrap" idea, done on the fly). Its construction:
- It runs along the propagation axis $a$, with nodes at the 3D nodes and with $\Delta l_i = x_{i+1}-x_i$ and $\Delta\tilde l_i = (\Delta l_{i-1}+\Delta l_i)/2$, the same arrays as the 3D grid. Graded meshes along $a$ are therefore reproduced bit-for-bit.
- It uses the 3D $\Delta t$ and the background $\varepsilon$, $\mu$.
- Its time levels are those of the 3D step: e at $n\Delta t$, h at $(n+\tfrac12)\Delta t$.

A field uniform in the transverse directions has zero transverse differences, so the 3D curl of $(E_p(a), H_q(a))$ reduces exactly to the 1D one:

$$
H_q^{n+1/2}\big|_{i+1/2} = H_q^{n-1/2}\big|_{i+1/2} - \frac{\Delta t\, s}{\mu\,\Delta l_i}\left(E_p^n|_{i+1} - E_p^n|_i\right),\qquad
E_p^{n+1}\big|_i = E_p^{n}\big|_i - \frac{\Delta t\, s}{\varepsilon\,\Delta\tilde l_i}\left(H_q^{n+1/2}|_{i+1/2} - H_q^{n+1/2}|_{i-1/2}\right),
$$

with $s = \epsilon_{qap}$ (Levi-Civita) and $H_q = \pm s E_p/\eta$ for propagation along $\pm a$. The line is driven by its own 1D TF/SF boundary one cell upstream of the box entry plane, using the analytic $s(t)$. One cell is the closest position at which the sample $H_q|_{i_0-1/2}$, which the 3D E correction reads, is still an ordinary total-field sample of the line. Both ends end in 96-cell graded matched absorbers ($\sigma_m/\mu = \sigma_e/\varepsilon$, cubic grading).

The corrections are precomputed as two sparse matrices, from line samples to the surface rows. They are built from the grid lengths, $\mathbf{C}$ and the solver's `imu`. Each step the source:
1. adds $-\Delta t(\mathbf{P}_h\mathbf{A}-\mathbf{A}\mathbf{P}_e)\mathbf{e}^{inc,n}$ to H before the step. This term is independent of the fields, so adding it before the curl update is identical to adding it after.
2. advances the line H to $n+\tfrac12$.
3. writes the equivalent current $\mathbf{j}_s(\mathbf{h}^{inc,n+1/2})$ into `solver.J` (only the change since the previous step, as `CurrentSource` does).
4. advances the line E to $n+1$.

The H correction must use $\mathbf{e}^{inc}$ at $n\Delta t$ and the E correction $\mathbf{h}^{inc}$ at $(n+\tfrac12)\Delta t$. Using $\mathbf{e}^{inc,n+1}$ instead is the classic half-step TF/SF leakage bug, and the tests check that it fails loudly.

**Outputs and normalization.** The entry-plane values are recorded every step: E at $(n+1)\Delta t$ on the entry node, and H at $(n+\tfrac12)\Delta t$ averaged from the two neighbouring dual samples, as `FluxMonitor` does. `incident_fields(f)` returns their DFTs with the monitor convention $F(f) = \sum_n F(t_n)e^{-2\pi i f t_n}\Delta t$. `incident_flux_density(f)` returns $\Phi_{inc} = \hat{\mathbf{k}}\cdot\operatorname{Re}(\mathbf{E}\times\mathbf{H}^*) = |E|^2\cos(k\Delta l/2)/\eta$ per m², in the units of `FluxMonitor.get_flux()` per m², so that cross-sections are plain ratios. The entry-plane spectrum differs from $A\,S(f)$ only by the grid dispersion over the one cell between injection and entry plane, $\sim(k\Delta l)^3/24$ rad. The 3D sees exactly the recorded field.

**Validation** (`tests/test_039_tfsf_planewave.py`):
- *Vacuum leakage.* In an empty domain, $\max|\psi_{SF}|/\max|\psi_{TF}|$ over the run (E and $\eta_0$H) is $6\times10^{-16}$–$1.2\times10^{-15}$ in float64. This holds for all six directions and both polarizations, on uniform grids and on grids graded along the axis (cell sizes $0.62$–$1.38\,h$). With CPML it is $2\times10^{-15}$ (float64) and $6\times10^{-7}$ (float32).
- *Mutation tests.* These show the leakage test can fail. Using $\mathbf{e}^{inc,n+1}$ in the H correction (half-step error) gives leakage 0.31. Shifting $E^{inc}$ one cell against $H^{inc}$ gives 0.92. Shifting both by one cell on a graded grid gives 0.98. On a uniform grid a common shift is only a translated discrete plane wave, and its leakage stays at $10^{-15}$.
- *Total field.* Inside the box the total field equals the line field to $10^{-12}$ relative.
- *Energy, lossless.* For an $\varepsilon_r = 4$ sphere, the net flux through a closed box inside the TF region is $<3\times10^{-8}$ of the incident power on a box face.
- *Energy, lossy.* For a lossy sphere ($\sigma = 0.05$ S/m), the absorbed power $-\Phi_{tot}$ equals the volume Ohmic loss $\sum\sigma|E(f)|^2\Delta V$ to $3\times10^{-4}$.
- *Mie.* The cross-section of a conformal sphere approaches Mie theory: +6.5–7.5 % at 3.5 cells per radius, +4.8–5.4 % at 4.9 cells.
- *Inverse design.* The adjoint gradient of the scattered flux with respect to a design block inside the box matches central finite differences to $10^{-10}$, because the source is an additive forcing independent of the fields.

The study `studies/tfsf/` compares this with Huygens face sheets driven by the continuum incident field, as in hand-built Meep TF/SF sources. With the exact staggered registration (J on the face carries $H^{inc}$ half a cell outside, K there carries $E^{inc}$ of the face node), the only error left is the dispersion mismatch: the leaked power falls as $(kh)^4$, $1.4\times10^{-4}$ of the face power at 20 cells per wavelength. Referencing both sheets to the face plane adds the half-cell offset: the power then falls as $(kh)^2$, $5\times10^{-2}$ at 20 cells per wavelength. The same sheets driven by the 1D-line values leak at round-off, which shows that the commutator corrections are exactly the discrete equivalent currents.

**Limits & Failure Modes.**
- *Axis-aligned incidence only.* Oblique incidence needs a discrete plane wave with the 3D grid's anisotropic dispersion; Tan & Potter [2010] and Ch. 3 of Taflove, Oskooi & Johnson (eds.) [2013], cited in Oskooi & Johnson §4.2.2, describe the discrete plane-wave line for this. Not implemented.
- *Homogeneous, lossless, non-magnetic background at the box surface.* Every sample in the correction stencil must carry the background `ieps`, `imu = 1/μ0` and `sigma = 0`; otherwise the source raises. Objects may cross the scattered-field region, but the incident field is not computed for them. For a layered background (substrate) or a waveguide, $\psi^{inc}$ is not a 1D plane wave.
- *Box inside the non-PML region.* Each box face must be at least one node from the domain edge, or $n_{pml}+1$ nodes from a (C)PML face. Along a periodic axis transverse to the propagation, a box covering the whole period has no faces on that axis (normal incidence on periodic structures). The propagation axis cannot be periodic, and Bloch phases are not supported.
- *Box snapping.* The faces snap to the nearest grid nodes; `box_bounds()` returns the faces used. Place flux monitors from these bounds, at least one node inside or outside. A monitor face that coincides with the TF/SF surface or cuts a staircased object gives meaningless net fluxes (observed: $2\times10^{-2}$ instead of $10^{-8}$).
- *Source truncation.* A `GaussianSource` cut at $5\sigma_t$ has a small DC component that leaves a static field of $\sim3\times10^{-7}$ in the line and the total-field region. This is a property of the incident pulse, not leakage; use `cutoff=8` for cleaner spectra.
- *Not supported.* MPI (raises `NotImplementedError`), complex Bloch fields with $k\neq0$, and the legacy beam TF/SF of `source_type='tfsf'` (independent).
- *Cost.* Two sparse products on the surface rows (O(box surface)) and a 1D line of $(N_a + 200)$ samples per step: about 1 % per step on a $64^3$ CPML grid with a $32^3$ box on CPU.

References:
- A. Oskooi and S. G. Johnson, "Electromagnetic wave source conditions", Ch. 4 in A. Taflove, A. Oskooi and S. G. Johnson (eds.), *Advances in FDTD Computational Electrodynamics: Photonics and Nanotechnology*, Artech House (2013), [arXiv:1301.5366](https://arxiv.org/abs/1301.5366): Eq. (4.4) and §4.2.1 (equivalence principle), §4.2.2 (discretization and dispersion of equivalent currents).
- D. E. Merewether, R. Fisher and F. W. Smith, "On implementing a numeric Huygens' source scheme in a finite difference program to illuminate scattering bodies", *IEEE Trans. Nucl. Sci.* **27**(6), 1829 (1980).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), Ch. 5.
- J. B. Schneider, "Plane waves in FDTD simulations and a nearly perfect total-field/scattered-field boundary", *IEEE Trans. Antennas Propag.* **52**(12), 3280 (2004).
- T. Tan and M. Potter, "FDTD discrete planewave (FDTD-DPW) formulation for a perfectly matched source in TFSF simulations", *IEEE Trans. Antennas Propag.* **58**(8), 2641 (2010).
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983) (Mie coefficients used in the tests).

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


### 🌈 Diffraction orders of periodic structures: `DiffractionMonitor`

**Floquet (Rayleigh) expansion.** In a homogeneous region above or below a structure that is periodic with periods $P_b, P_c$ along the transverse axes $(b, c)$ of a plane normal to $a$ ($x, y$ for a $z$-normal plane), the field of a Bloch-periodic solution is a discrete sum of plane waves, the diffraction orders [Petit 1980; Moharam & Gaylord 1981]:

$$
\mathbf{E}(\mathbf{r}) = \sum_{m,n} \left[\mathbf{E}^{\uparrow}_{mn} e^{i k_{a,mn} a} + \mathbf{E}^{\downarrow}_{mn} e^{-i k_{a,mn} a}\right] e^{i \mathbf{k}_{t,mn}\cdot\mathbf{r}}, \qquad
\mathbf{k}_{t,mn} = \mathbf{k}_\text{inc} + 2\pi\left(\tfrac{m}{P_b}, \tfrac{n}{P_c}\right),
$$

with $k_{a,mn} = \sqrt{n^2 k_0^2 - |\mathbf{k}_{t,mn}|^2}$. Orders with real $k_{a,mn}$ propagate at the polar angle $\sin\theta_{mn} = |\mathbf{k}_{t,mn}|/(n k_0)$ (grating equation); the others are evanescent. Order labels refer to physical waves $e^{i(\mathbf{k}\cdot\mathbf{r}-\omega t)}$, as in RCWA codes such as fmmax and S4. For real periodic fields $\mathbf{k}_\text{inc} = 0$. For complex Bloch fields, following the Bloch convention above, the spectral component at $f>0$ is the wave with $\mathbf{k}_\text{inc} = -\mathbf{k}_\text{bloch}$, and the one at $f<0$ its mirror image with $+\mathbf{k}_\text{bloch}$. Since the DFT phasors of Wakis have time dependence $e^{+2\pi i f t}$, the order $(m, n)$ appears in the monitor data with the spatial phase $e^{i\mathbf{q}\cdot\mathbf{r}}$, $\mathbf{q} = -\operatorname{sign}(f)\,\mathbf{k}_t$. The complex amplitudes returned keep the $e^{+2\pi i f t}$ convention of the other monitors: conjugate them to compare phases with an $e^{-i\omega t}$ code.

**Discrete transform and Parseval.** `DiffractionMonitor(frequencies, center, size)` is a `FluxMonitor` plane spanning the whole unit cell, so it accumulates the same samples: $E_b$ and $\bar H_c$ ($H_c$ averaged across the plane) at the $b$-edge midpoints, $E_c$ and $\bar H_b$ at the $c$-edge midpoints. Each sampled component $F$ is transformed at its own sample positions $\mathbf{r}_j$ (a uniform transverse mesh with $N_b \times N_c$ cells is required):

$$
\hat F_{mn} = \frac{1}{N_b N_c}\sum_j F(\mathbf{r}_j)\, e^{-i\mathbf{q}_{mn}\cdot\mathbf{r}_j}.
$$

The Bloch condition makes $F(\mathbf{r}_j)e^{-i\mathbf{q}_{00}\cdot\mathbf{r}_j}$ exactly periodic on the samples, so this is an exact DFT and the $N_bN_c$ orders of one DFT period represent the samples exactly. Because the area elements of `FluxMonitor` are uniform ($\Delta A = A/N_bN_c$, $A = P_bP_c$) and $E_b$, $\bar H_c$ share their positions, Parseval's theorem turns the monitor flux into a sum over orders,

$$
\Phi(f) = \sum_{m,n} \Phi_{mn}, \qquad \Phi_{mn} = A\,\operatorname{Re}\left[\hat E_{b}\hat{\bar H}_{c}^{*} - \hat E_{c}\hat{\bar H}_{b}^{*}\right]_{mn},
$$

exactly (`get_power()['net']`; verified to $10^{-15}$ relative).

**Up/down and TE/TM split with the discrete plane waves.** Let $\hat{\mathbf{k}}$ be the unit vector of the physical tangential wavevector of the order (the $b$ axis at normal incidence) and $\hat{\mathbf{s}} = \hat{\mathbf{a}}\times\hat{\mathbf{k}}$, and project the tangential transforms on them: $E_k, E_s, \bar H_k, \bar H_s$. In the uniform FIT mesh, a plane wave $e^{i(\mathbf{q}\cdot\mathbf{r} + \omega t)}$ is an exact solution when the continuous wavenumbers are replaced by the discrete ones [Taflove & Hagness 2005, Ch. 4]

$$
\Omega = \frac{2}{\Delta t}\sin\frac{\omega\Delta t}{2},\quad K_d = \frac{2}{\Delta_d}\sin\frac{q_d\Delta_d}{2},\quad K_a^2 = \Omega^2\varepsilon\mu - K_b^2 - K_c^2 ,
$$

with $k_a = (2/\Delta_a)\arcsin(K_a\Delta_a/2)$, and the transverse fields of each polarization are tied by real admittances: for the wave travelling along $\pm\hat{\mathbf{a}}$ ("up"/"down"), TE ($\mathbf{E}\parallel\hat{\mathbf{s}}$) has $H_k = \mp Y_\text{TE}E_s$ and TM ($\mathbf{H}\parallel\hat{\mathbf{s}}$) has $H_s = \pm Y_\text{TM}E_k$, with

$$
Y_\text{TE} = \frac{K_a}{|\Omega|\mu},\qquad Y_\text{TM} = \frac{|\Omega|\varepsilon}{K_a},
$$

independently of the sign of $f$. The average of $H$ across the plane (half a cell on each side) multiplies it by $\gamma = \cos(k_a\Delta_a/2) = \sqrt{1 - (K_a\Delta_a/2)^2}$. Writing the sampled fields as the sum of an up and a down wave and solving the $2\times2$ systems gives the power-normalized amplitudes

$$
a^{\uparrow\downarrow}_{s} = \tfrac12\sqrt{A\gamma Y_\text{TE}}\left(E_s \mp \frac{\bar H_k}{\gamma Y_\text{TE}}\right),\qquad
a^{\uparrow\downarrow}_{p} = \tfrac12\sqrt{A\gamma Y_\text{TM}}\left(E_k \pm \frac{\bar H_s}{\gamma Y_\text{TM}}\right).
$$

For a propagating order the cross terms between the up and down waves cancel because the admittances are real, so $\Phi_{mn} = |a^\uparrow_s|^2 + |a^\uparrow_p|^2 - |a^\downarrow_s|^2 - |a^\downarrow_p|^2$ exactly, and TE and TM carry orthogonal powers. The amplitudes are in √(W s²): $|a|^2$ is in the units of `FluxMonitor.get_flux` (for a harmonic source of amplitude $S(f)$, the time-averaged power is $|a|^2/2|S|^2$). Order efficiencies are ratios $|a|^2/\Phi_\text{inc}$, so the source spectrum cancels. Their phases are those of $E_s$ and of $E_k$ (the TM field along $\hat{\mathbf{p}}$, whose tangential part points along $+\hat{\mathbf{k}}$ for both directions), referred to the plane and to the transverse origin of the coordinates. Evanescent orders are not decomposed (zero amplitudes; their contribution remains in `'net'`). Orders are returned on the smallest box of $(m, n)$ containing every order propagating at some monitored frequency (`orders`, `propagating`); `order_angles()` gives $\theta_{mn}$ and the azimuth of $\mathbf{k}_t$ from the continuum relation.

**Discrete or continuum $k_a$.** The identity $\sum\Phi_{mn} = \Phi$ and the power balance up − down = net hold for any real admittance, but only the discrete $K_a$, $\Omega$ and $\gamma$ make the split exact for the waves of the scheme. With the continuum $k_a$, $\omega$ and no $\gamma$, a single up-going wave leaks a spurious down-going wave of relative power $\sim(k\Delta)^4$: measured above the lamellar grating below, $3.9\times10^{-5}$ ($0$ order) and $1.6\times10^{-5}$ ($\pm1$) at 20 cells per wavelength, $2.4\times10^{-6}$ and $1.0\times10^{-6}$ at 40, against $\leq 8\times10^{-9}$ (the CPML reflection) with the discrete relations. Wakis uses the discrete relations.

**Polarization conventions.** TE (s) has $\mathbf{E}\parallel\hat{\mathbf{s}} = \hat{\mathbf{a}}\times\hat{\mathbf{k}}$, perpendicular to the plane of incidence; TM (p) has $\mathbf{H}\parallel\hat{\mathbf{s}}$. At normal incidence the plane of incidence is degenerate and $\hat{\mathbf{k}} = \hat{\mathbf{b}}$ is used, so that for a $z$-normal plane TE is $E_y$ and TM is $E_x$, as in invrs-gym and fmmax at zero azimuth. (fmmax's "TE"/"TM" amplitudes are the $H_x$/$H_y$ components of each order. They coincide with s/p for planes of incidence containing $x$ and differ otherwise; invrs-gym's grating efficiencies sum the two output polarizations of each order, which is basis independent and equals `get_power()['up'/'down']`.) Circular amplitudes are

$$
a_L = \frac{a_p + i\,d\,\sigma\,a_s}{\sqrt2},\qquad a_R = \frac{a_p - i\,d\,\sigma\,a_s}{\sqrt2},
$$

with $d = +1$ (up) or $-1$ (down) and $\sigma = \operatorname{sign}(f)$ for the $e^{+2\pi i f t}$ phasors. $L$ is positive helicity (the field rotates counter-clockwise about the propagation direction, i.e. by the right-hand rule) and $R$ negative helicity: the optics convention [Born & Wolf 1999], which matches the RHCP/LHCP basis of the invrs-gym meta-atom library for its transmitted waves. $|a_L|^2 + |a_R|^2 = |a_s|^2 + |a_p|^2$.

**Normalization.** The incident flux is most simply the up-going $(0,0)$ power on a second monitor between the source and the structure, which also measures the reflected orders as down-going waves in the same run; alternatively use the flux of a reference run without the structure. Powers are fluxes, so transmission into a medium different from the incidence medium needs no impedance factor (invrs-gym's library rescales its amplitudes by $\sqrt{n_\text{sub}/n_\text{amb}}$ for the same reason). A sheet `CurrentSource` spanning the cell launches the incident plane wave in both directions; give it the phase profile $e^{i\mathbf{k}_\text{bloch}\cdot\mathbf{r}}$ at oblique incidence.

**Validation** (`tests/test_033_diffraction.py`; fmmax 1.7.1 references from `benchmarks/diffraction_fmmax_reference.py`, run in the invrs-gym environment, converged to $<6\times10^{-5}$ in the number of Fourier orders):
- Dielectric slab ($n=1.5$) in a 3D periodic cell with the $(\pm1, 0)$ orders open: only $(0,0)$ carries power ($<10^{-20}$), it is TM for an $E_x$ sheet, $T$ matches the Airy formula within $1.7\times10^{-3}$ at $\geq17$ cells per wavelength in the slab, $R+T = 1$ within $5\times10^{-6}$, $\sum_{mn}\Phi_{mn} = \Phi$ to $10^{-15}$, and the down-going wave above the slab (CPML reflection) is $<10^{-10}$ of the incident.
- Lamellar grating (period $1.5\lambda_0$, teeth $\varepsilon=4$, height $0.5\lambda_0$, fill 0.5, on $\varepsilon=2.25$, incidence from the substrate; 5 reflected and 3 transmitted orders at normal incidence). Maximum absolute efficiency error against fmmax at 20, 40, 80 cells per $\lambda_0$: TE $2.0\times10^{-2}$, $5.1\times10^{-3}$, $1.3\times10^{-3}$; TM $2.3\times10^{-2}$, $5.6\times10^{-3}$, $1.3\times10^{-3}$ (second order). At 80 cells/$\lambda_0$:

| | fmmax | Wakis |
|---|---|---|
| TE, $R_0$, $R_{\pm1}$, $R_{\pm2}$ | 0.21484, 0.10495, 0.02761 | 0.21612, 0.10393, 0.02777 |
| TE, $T_0$, $T_{\pm1}$ | 0.12414, 0.19795 | 0.12409, 0.19807 |
| TM, $R_0$, $R_{\pm1}$, $R_{\pm2}$ | 0.02369, 0.04904, 0.04949 | 0.02398, 0.04899, 0.05030 |
| TM, $T_0$, $T_{\pm1}$ | 0.07843, 0.35040 | 0.07709, 0.35012 |

- Same grating at $15^\circ$ incidence in the substrate (complex Bloch fields, $\mathbf{k}_\text{bloch} = -\mathbf{k}_\parallel$; reflected orders $-2\ldots1$, transmitted $-2\ldots0$): maximum error $2.2\times10^{-2}$, $5.1\times10^{-3}$, $1.2\times10^{-3}$ (TE) and $1.8\times10^{-2}$, $4.5\times10^{-3}$, $1.1\times10^{-3}$ (TM) at 20, 40, 80 cells/$\lambda_0$; e.g. TE $T_{-1}$ = 0.44775 vs 0.44821, TM $R_1$ = 0.18890 vs 0.18981. The $f<0$ spectrum reproduces the mirror-image incidence ($\eta_m(-f) = \eta_{-m}(f)$ to $10^{-9}$), and the specular order leaves at $15^\circ$.
- Half-wave plate ($n_x=1.5$, $n_y=1.2$): an $E_x - iE_y$ sheet in a complex run launches pure $L$ at $f>0$ and pure $R$ at $f<0$ ($<10^{-14}$ of the other handedness); the transmitted $L$/$R$ powers follow $|t_x \pm t_y|^2/4$ (Airy amplitudes) within $10^{-3}$, with 99.99 % conversion at the design frequency.

**Adjoint.** The amplitudes are linear functionals of the DFT accumulators already returned by `FluxMonitor` as raw outputs (`'E0'`, `'H0'`, `'E1'`, `'H1'` = $E_b$, $\bar H_c$, $E_c$, $\bar H_b$), with fixed weights (DFT matrices, projections and admittances, which depend only on the homogeneous medium at the plane). No new transposed update is needed: the adjoint sources are those of `FluxMonitor` driven by the cotangents of these arrays. `wakis.inverse.diffraction_orders(raw, monitor, polarization)` evaluates the same formulas in JAX, so efficiency objectives such as $|a^\uparrow_{s,(1,0)}|^2/\Phi_\text{inc}$ are differentiated exactly; the gradient agrees with finite differences to $\sim10^{-8}$ for real periodic and complex Bloch fields.

**Limitations.** The medium on and next to the plane must be homogeneous, isotropic and lossless (no dispersive material), the transverse mesh uniform and the mesh spacing along the normal equal on both sides of the plane. Place the plane far enough from the structure for the evanescent orders to decay if propagating-order sums are compared with the flux. With MPI only planes normal to $z$ are supported (untested), and the GPU path is untested.

References:
- R. Petit (ed.), *Electromagnetic Theory of Gratings*, Springer (1980).
- M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction", *J. Opt. Soc. Am.* **71**, 811–818 (1981), [doi:10.1364/JOSA.71.000811](https://doi.org/10.1364/JOSA.71.000811).
- V. Liu and S. Fan, "S4: A free electromagnetic solver for layered periodic structures", *Comput. Phys. Commun.* **183**, 2233–2244 (2012), [doi:10.1016/j.cpc.2012.04.026](https://doi.org/10.1016/j.cpc.2012.04.026).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005), Ch. 4 (numerical dispersion).
- M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press (1999).
- fmmax (Fourier modal method in JAX, MIT), https://github.com/facebookresearch/fmmax; invrs-gym, https://github.com/invrs-io/gym.

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

The time step is then $\Delta t = \min(\Delta t_\text{default}, \texttt{cfln}\cdot 2/\sqrt{\lambda_\max})$. $\lambda_\max$ is computed with Lanczos iterations (with a 1 % margin, see *Tensor subpixel averaging* for the recurrence) and stored in `solver.conformal_dt_max`. A user-given `dt` above the limit triggers a warning. In the tests the limit is 0.85–1.0 times the Yee limit for curved PEC (0.86 for the sphere, independent of resolution), so with the default `cfln=0.5` the step is reduced by at most about 15 %. For the patch template it is 0.67 times the Yee limit (thin substrate cells and the sheet-edge correction of faces next to the edge, see below).

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

### 🧊 Tensor subpixel averaging: `conformal='tensor'` and `MaterialGrid(smoothing='tensor')`

The diagonal average of `conformal='all'` is first order whenever $\mathbf E$ crosses a curved or oblique dielectric interface, because it drops the off-diagonal entries of the averaged tensor. `conformal='tensor'` keeps them. It is opt-in; without it every code path, operator and regression result is unchanged. The implementation is in `wakis/tensor_averaging.py`.

**Averaged tensor** [Farjadpour et al., 2006; Kottke et al., 2008]. For a pixel cut by a planar interface with unit normal $\mathbf n$ and fill fraction $f$, the effective inverse permittivity that makes the first-order perturbation of the eigenfrequencies vanish is

$$
\mathbf T = \langle\varepsilon^{-1}\rangle\,\mathbf n\mathbf n^T + \langle\varepsilon\rangle^{-1}(\mathbf I - \mathbf n\mathbf n^T)
= \langle\varepsilon\rangle^{-1}\mathbf I + a\,\mathbf n\mathbf n^T,\qquad a = \langle\varepsilon^{-1}\rangle - \langle\varepsilon\rangle^{-1} \ge 0 .
$$

Every E edge $e$ (component $c$) has its own pixel, the dual cell of `conformal='all'`, with $f_e$, $\mathbf n_e$ and $a_e$. Its diagonal entry $T_{cc}(e)$ is exactly the 'all' value, kept in `solver.ieps`.

**Symmetric corner (octant) operator (Wakis construction).** The off-diagonal entries couple $E_c$ to the other two D components, which live on other edges. Each primary cell is split into eight octants, one per corner. The octant at a corner contains one x, one y and one z edge: the three edges leaving that corner into the cell. With $\mathbf W = \operatorname{diag}(\ell\tilde A)$ the energy weights of the E edges (open length times dual area), the material operator is

$$
\mathbf E = \mathbf N\mathbf D,\qquad \mathbf N = \operatorname{diag}(\varepsilon^{-1}) + \mathbf W^{-1}\mathbf S,\qquad
S_{ab} = \sum_{o\ni a,b} \frac{V_o}{8}\,s_o\,\tfrac12\big(T_{ij}(a) + T_{ij}(b)\big),
$$

for the edges $a$ (component $i$) and $b$ (component $j\ne i$) of each octant $o$ of a cell of volume $V_o$. For a uniform $T_{ij}$ this is $E_i \mathrel{+}= T_{ij}\,\overline{D_j}$, with $\overline{D_j}$ the average of the four $D_j$ around the $E_i$ edge. That is the off-diagonal interpolation of Meep [Farjadpour et al., 2006; Oskooi et al., 2010]. Averaging $T_{ij}$ over the two edges makes $\mathbf S$, and so $\mathbf W\mathbf N$, symmetric, as in the symmetric FDTD schemes for anisotropic dielectrics [Werner & Cary, 2007]. The pixel tensors come from the conformal geometry: $f_e$ is the exact dual-cell fraction and $\mathbf n_e$ the normal of the nearest surface triangle.

**Positive definiteness and stability.** With the per-octant share $d_{o,e} = \varepsilon^{-1}_e W_e/\tilde V_e$ of the diagonal ($\tilde V_e = \sum_{o\ni e} V_o/8$),

$$
\mathbf W\mathbf N = \sum_o \frac{V_o}{8}\,\mathbf P_o^T\big(\mathbf D_o + s_o\mathbf t_o\big)\mathbf P_o + \operatorname{diag}\big(\mathbf W\varepsilon^{-1} - \textstyle\sum_o \tfrac{V_o}{8}\mathbf P_o^T\mathbf D_o\mathbf P_o\big),
$$

where $\mathbf P_o$ picks the three octant edges, $\mathbf D_o = \operatorname{diag}(d_o)$ and $\mathbf t_o$ is the off-diagonal $3\times3$ block. The last term is zero for edges inside the domain and non-negative at partial (conformal PEC) edges. The scale $s_o\le1$ is reduced only where $\lambda_\min(\mathbf D_o^{-1/2}\mathbf t_o\mathbf D_o^{-1/2}) < -0.99$, so every octant block is positive definite and $\mathbf W\mathbf N$ is SPD. Couplings to edges with $\varepsilon^{-1}=0$ (PEC, ghost cells) are dropped. In all tests $s_o = 1$ everywhere ($\lambda_\max \le 0.4$). Then the update $\mathbf H' = \mathbf H - \Delta t\mathbf A\mathbf E$, $\mathbf E' = \mathbf E + \Delta t\,\mathbf N\tilde{\mathbf A}^{-1}\mathbf C^H\tilde{\mathbf L}\mathbf H'$ conserves a discrete energy and is stable for $\Delta t < 2/\sqrt{\lambda_\max(\mathbf B\mathbf A)}$. $\mathbf B\mathbf A$ is similar to a Hermitian PSD matrix, so its eigenvalues are real. With $\mathbf N = \mathbf W^{-1}(\mathbf W\operatorname{diag}(\varepsilon^{-1}) + \mathbf S)$, $\mathbf S = \mathbf W\mathbf K$ Hermitian, the nonzero eigenvalues of $\mathbf B\mathbf A = \mathbf N\tilde{\mathbf A}^{-1}\mathbf C^H\tilde{\mathbf L}\mathbf G\mathbf C\mathbf L$ equal those of $\mathbf P\mathbf Q$ (cyclic permutation), $\mathbf P = \mathbf L\mathbf N\tilde{\mathbf A}^{-1} = \tilde{\mathbf A}^{-1}(\mathbf W\operatorname{diag}(\varepsilon^{-1}) + \mathbf S)\tilde{\mathbf A}^{-1}$ and $\mathbf Q = \mathbf C^H\tilde{\mathbf L}\mathbf G\mathbf C$, both Hermitian PSD ($\mathbf L\mathbf W^{-1} = \tilde{\mathbf A}^{-1}$). $\mathbf P\mathbf Q$ is self-adjoint in the $\mathbf P$-semi-inner product, and the Lanczos recurrence for $\mathbf P^{1/2}\mathbf Q\mathbf P^{1/2}$ needs only products with $\mathbf P$ and $\mathbf Q$: $\beta_{j+1}\mathbf r_{j+1} = \mathbf Q\mathbf z_j - \alpha_j\mathbf r_j - \beta_j\mathbf r_{j-1}$, $\mathbf z_j = \mathbf P\mathbf r_j$, $\alpha_j = \mathbf z_j^H\mathbf Q\mathbf z_j$, $\beta_{j+1}^2 = \tilde{\mathbf r}^H\mathbf P\tilde{\mathbf r}$ [Saad, *Numerical Methods for Large Eigenvalue Problems*, 2nd ed., SIAM 2011, ch. 6] (`wakis._lanczos`; for the diagonal case $\mathbf P$ is diagonal and this is the usual Lanczos for $\mathbf R\mathbf C^H\mathbf D_2\mathbf G\mathbf C\mathbf R$). It stops when the residual estimate of the largest Ritz pair is below $10^{-3}\theta$ (the ARPACK criterion), runs only on the edges with $P_{ee}>0$, stores five vectors and uses multi-threaded sparse products (or CuPy with `use_gpu`). Converged tightly it agrees with Arnoldi iterations on $\mathbf B\mathbf A$ itself to $10^{-9}$ (test); at the default tolerance both scatter by $\sim10^{-5}$. $\Delta t = \min(\Delta t_\text{default}, \texttt{cfln}\cdot 2/\sqrt{\lambda_\max})$ (`solver.conformal_dt_max`). A cheap a-priori bound also holds: each octant block is at most $(1+\mu_\max)\mathbf D_o$, $\mu_\max = \max_o \lambda_\max(s_o\mathbf D_o^{-1/2}\mathbf t_o\mathbf D_o^{-1/2}) < 2$, so the tensor step is stable at $\Delta t_\text{diag}/\sqrt{1+\mu_\max} > 0.57\,\Delta t_\text{diag}$. With the default `cfln = 0.5` the time step is therefore never reduced by the tensor; it was unchanged in all tests.

**Time step (Wakis construction).** $\mathbf N$ differs from the diagonal only on the coupled edges $R$, so the step is the solver's own diagonal step followed by a local correction:

$$
\mathbf E^* = \mathbf E^n + \Delta t\,\varepsilon^{-1}(\mathbf C^H\ldots\mathbf H - \mathbf J),\qquad
\mathbf E^{n+1}_R = \mathbf E^*_R + \mathbf K'(\mathbf E^*_R - \mathbf E^n_R),\qquad
K'_{ab} = \frac{S_{ab}}{W_a\,\varepsilon^{-1}_b} .
$$

This works for every backend and step variant (NumPy/MKL, CuPy with `cupyx` sparse, CPML, TFSF, conductivity, Bloch phases on wrapped couplings), costs one sparse product on $|R|$ edges per step and keeps the step linear. With MPI the couplings reach one $z$ layer into the ghost layers; the E ghost layers are exchanged once more after the correction. Couplings to edges of dispersive (ADE) groups are dropped, so dispersive solids keep the diagonal average and the corrections act on disjoint edges.

**Adjoint.** The correction is linear in $(\mathbf E^*, \mathbf E^n)$, so `AdjointStepper` adds its transpose:

$$
\boldsymbol\lambda_{E^*,R} = (\mathbf I + \mathbf K'^H)\boldsymbol\lambda_{E^{n+1},R},\qquad \boldsymbol\lambda_{E^n,R} \mathrel{+}= -\mathbf K'^H\boldsymbol\lambda_{E^{n+1},R} .
$$

The dot-product identity holds to $10^{-17}$ relative (PEC and CPML). For the gradient, the step is $\mathbf E^{n+1} = \mathbf E^n + \mathbf N\mathbf g^n$ with $\mathbf g^n = \Delta t(\ldots)$ the increment of $\mathbf D$, recorded on the design and coupled edges. Then $\partial F/\partial\varepsilon^{-1}_a = \sum_n \lambda^{n+1}_a g^n_a$ and $\partial F/\partial S_{ab} = \sum_n (\lambda^{n+1}_a g^n_b/W_a + \lambda^{n+1}_b g^n_a/W_b)$, with $\boldsymbol\lambda^{n+1}$ the adjoint of the corrected $\mathbf E^{n+1}$.

**Density designs: `MaterialGrid(..., smoothing='tensor')` (Wakis adaptation).** For a density $\rho$ (normally filtered and projected) the design edges take, instead of the arithmetic interpolation, the averaged tensor of a pixel with fill fraction $f_e = \operatorname{clip}(\mathbf P\rho, 0, 1)$ and normal from the density gradient $\mathbf g_e = \mathbf G\rho$ (trilinear interpolation of nodal central differences):

$$
\mathbf T_e = \frac{\mathbf I}{\langle\varepsilon\rangle} + a\,\mathbf m\mathbf m^T,\quad
\mathbf m = \frac{\mathbf g}{\sqrt{|\mathbf g|^2+\delta^2}},\quad
\langle\varepsilon\rangle = \varepsilon_\min + \Delta\varepsilon f,\quad
\langle\varepsilon^{-1}\rangle = \frac{1-f}{\varepsilon_\min} + \frac{f}{\varepsilon_\max},
$$

with $\delta = 10^{-2}/h_\text{design}$, so that $\mathbf m$ is the unit normal at an interface ($|\mathbf g|\sim 1/h$) and goes smoothly to zero (isotropic arithmetic mean) in uniform regions. Using the density as fill fraction and its gradient as normal is the idea of Meep's `MaterialGrid(do_averaging=True)` [Oskooi et al., 2010; Hammond et al., 2021]. The subpixel-smoothed projection of [Hammond et al., 2025] goes further and also derives the fill fraction from the filtered density and its gradient; it is not implemented here. $\mathbf T_e$ enters as above: diagonal in `ieps`, off-diagonals through the corner couplings. The map $\rho\mapsto(\varepsilon^{-1}, \mathbf S)$ is smooth, and `MaterialGrid.vjp` applies the exact chain rule through $f$ ($\partial\langle\varepsilon\rangle^{-1}/\partial f = -\Delta\varepsilon/\langle\varepsilon\rangle^2$, $\partial\langle\varepsilon^{-1}\rangle/\partial f = 1/\varepsilon_\max - 1/\varepsilon_\min$) and $\mathbf m$ ($\partial m_i/\partial g_k = \delta_{ik}/\sqrt q - g_ig_k/q^{3/2}$, $q = |\mathbf g|^2+\delta^2$). The octant scales $s_o$ are treated as constants, which is exact as long as no octant is clamped (`solver.tensor_operator.n_clamped == 0`). The finite-difference check agrees to $\le 10^{-6}$ relative. `apply` raises if $\Delta t$ exceeds the a-priori bound above (never with `cfln` $\le 0.5$).

**Limitations.** One E component per edge remains: the D components of the corner reconstruction lie on both sides of the interface in a cut cell. Some modes therefore still converge at first order, but with a much smaller constant. The prototype study found the same for every Cartesian variant; only interface-conforming meshes remove it. Dispersive pixels get the diagonal of the averaged tensor at every frequency, without couplings (see "Dispersive subpixel averaging" below for why). The MPI adjoint does not support the tensor correction. `field_energy` still uses the diagonal permittivity. The geometry (normals, fill fractions of STL solids) is not differentiated. Memory: $\mathbf K'$ has about 5.3 nonzeros per coupled edge and about 14 per primary cell touching an anisotropic edge (2.6 coupled edges per such cell; dielectric sphere at 16–32 cells across). In CSR that is about 0.2 kB per interface cell, plus one stored increment per coupled edge and step in an adjoint run.

**Validation** (`tests/test_035_tensor_averaging.py`; leapfrog dispersion removed, eigenvalues of the discrete operators as a cross-check):

| Case | Mesh | Reference | `'all'` (diagonal) | `'tensor'` |
| --- | --- | --- | --- | --- |
| 2D rod in a square PEC box, $\varepsilon_r=6$, TE dipole mode (eigenvalues) | 32/64/128/192 cells across | Delaunay FIT with interface nodes, Richardson | $2.7, 1.2, 0.43, 0.16\times10^{-3}$ | $1.3, 1.4, 0.48, 0.26\times10^{-4}$ |
| same, TE $n=0$ mode ($\mathbf E$ tangential, oblique to the grid) | 32/64/128/192 | same | $6.6, 3.9, 2.0, 1.35\times10^{-3}$ (order 0.8) | $2.1, 0.50, 0.17, 0.089\times10^{-3}$ (order 1.6) |
| same, TE mode 4 | 32/64/128/192 | same | $1.05, 0.53, 0.25, 0.16\times10^{-3}$ | $6.8, 2.4, 0.87, 0.47\times10^{-4}$ |
| same, TM (E$_z$) | 32 | same | $5.7\times10^{-5}$ | identical time series |
| 3D dielectric sphere ($\varepsilon_r=6$, radius $b/2$) in a PEC sphere, TM$_1$ (two probes, max) | 16/24/32/48/64 cells across $2b$ | Riccati–Bessel matching | $1.4, 0.96, 0.71, 0.48, 0.37\times10^{-2}$ (1st order) | $2.9, 0.82, 0.80, 0.39, 0.31\times10^{-4}$ (at the level of the PEC wall error and the grid splitting of the degenerate mode) |
| same, TE$_1$ ($\mathbf E$ tangential to the sphere, oblique to the grid) | 16/24/32/48 | same | $7.3, 6.8, 5.9, 4.3\times10^{-3}$ (order 0.5) | $7.2, 3.3, 1.8, 0.81\times10^{-3}$ (order 2) |
| Stability: dielectric ball and rod, PEC ball, sheet; 20 000 steps | 14×12×12 | bounded energy | — | no growth |
| Adjoint dot product, PEC and CPML | 14×12×12 | $\langle U x,y\rangle = \langle x,U^Ty\rangle$ | — | $\sim10^{-17}$ |
| `MaterialGrid(smoothing='tensor')` gradient vs central differences | 9×9 design, 2D TE | $h=10^{-5}$ | — | $\le 10^{-6}$ |

References:
- A. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and S. G. Johnson, "Meep: A flexible free-software package for electromagnetic simulations by the FDTD method", *Comput. Phys. Commun.* **181**, 687–702 (2010), [doi:10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).
- G. R. Werner and J. R. Cary, "A stable FDTD algorithm for non-diagonal, anisotropic dielectrics", *J. Comput. Phys.* **226**(1), 1085–1101 (2007).
- A. Hammond, A. Oskooi, S. G. Johnson and S. Ralph, "Photonic topology optimization with semiconductor-foundry design-rule constraints", *Opt. Express* **29**(15), 23916–23938 (2021), [doi:10.1364/OE.431188](https://doi.org/10.1364/OE.431188).
- A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph and S. G. Johnson, "Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection", [arXiv:2503.20189](https://arxiv.org/abs/2503.20189) (2025).
- Farjadpour et al. (2006) and Kottke et al. (2008): see "Conformal FIT" above.

### 🪙 Dispersive subpixel averaging: metals and dispersive dielectrics at interfaces

Kottke averaging (sections above) is derived for frequency-independent media. A cell cut by the surface of a metal or of a Lorentz/Debye dielectric needs the averaged permittivity at every frequency. Before this section was implemented, conformal modes averaged only $\varepsilon_\infty$ and put the poles on the staircased cell mask. This is also what Meep does: "Meep only does subpixel averaging of the *instantaneous* (i.e., infinite frequency) part of $\varepsilon$ and $\mu$. The dispersive part is not averaged at all" (Meep documentation, *Subpixel Smoothing*, "What about Dispersive Materials?"). For a Drude metal with $\varepsilon_\infty=1$ in vacuum, that is the staircase. The implementation is in `wakis/tensor_dispersive.py`. It is used automatically for STL solids with a `DispersiveMaterial` when `conformal='all'` or `'tensor'`, and by `MaterialGrid(material=..., smoothing='tensor')`.

**Averaged permittivity of a dispersive pixel.** A pixel with fill fractions $f_m$ (background included) and interface normal $\mathbf n$ is a laminate on the pixel scale. The tangential field sees the parallel combination and the normal field the series combination [Kottke et al., 2008]:

$$
\varepsilon_t(\omega) = \sum_m f_m\varepsilon_m(\omega),\qquad
\frac{1}{\varepsilon_n(\omega)} = \sum_m \frac{f_m}{\varepsilon_m(\omega)},\qquad
T_{cc}(\omega) = \frac{n_c^2}{\varepsilon_n(\omega)} + \frac{1-n_c^2}{\varepsilon_t(\omega)} .
$$

$\varepsilon_t$ is again a pole model, with the pole strengths scaled by $f_m$. $1/\varepsilon_n$ is not: its poles are the zeros of $\sum_m f_m/\varepsilon_m$, which depend on $f$. For a Drude metal ($\varepsilon_\infty$, $\omega_p$, $\gamma$) in a dielectric $\varepsilon_d$ it is a Lorentz pole (Wakis derivation):

$$
\varepsilon_n(\omega) = \frac{\varepsilon_d\varepsilon_\infty}{A} + \frac{f\,\varepsilon_d^2\omega_p^2/A^2}{\Omega^2-\omega^2+i\gamma\omega},\qquad
A = f\varepsilon_d + (1-f)\varepsilon_\infty,\quad \Omega^2 = \frac{(1-f)\,\omega_p^2}{A} .
$$

Here $f$ is the metal fraction. Re-fitting such effective poles per cell is the route of the dispersive contour-path method [Mohammadi et al., 2008], which uses Z-transforms. For multi-pole metals (gold: one Drude and five Lorentz poles) it needs a root search per cell, and the effective numerators are no longer of Lorentz form.

**Sub-capacitor network (Wakis construction).** Wakis instead realizes the series combination directly. This is the auxiliary-field idea of splitting the normal field per medium; Deinega and Valuev [2007] split $E$ into a parallel part and two normal parts, each driving its own polarization current (method as summarized by Hamm et al. [2014]). On edge $c$, $T_{cc}(\omega)$ is the elastance of capacitors in series that share the edge's displacement $D$:

- one series capacitor per dispersive material $m$, weight $c_k = n_c^2 f_m$, with the material's own model;
- one tangential capacitor, weight $c_k = 1-n_c^2$, with the mixed model $\varepsilon_t$ (static part $\sum_m f_m\varepsilon_{\infty,m}$, poles of material $m$ with multiplier $\mu = f_m$).

Each capacitor has a voltage $u_k$ and polarizations $p$ driven by $u_k$ through the unchanged ADEs of `wakis.dispersive`, so the edge field is the sum of the capacitor voltages:

$$
\frac{D}{\varepsilon_0} = \varepsilon_{\infty,k}\,u_k + \sum_p \mu_{kp}\,p_{kp},\qquad
E_c = \sum_k c_k u_k\ (+\ \text{static series terms of non-dispersive media}) .
$$

Inside a material ($f=1$, no normal) a single tangential capacitor remains, identical to the bulk ADE update. Series capacitors of non-dispersive media are static and stay in `ieps`. The instantaneous part $\sum_k c_k/\varepsilon_{\infty,k}$ is exactly the $\varepsilon_\infty$-averaged `ieps` that `wakis.conformal` already computes. The network therefore adds only the dynamic remainder, as a local correction after the unchanged solver step (any backend):

$$
\Delta Q_k = \sum_e b_{ke}\,\frac{D_e^{n+1}-D_e^n}{\varepsilon_0} = \sum_e b_{ke}\,\varepsilon_{r,s,e}(E^*_e-E^n_e),\qquad
E^{n+1}_e = E^*_e + \sum_k c_{ek}\Big(u_k^{n+1}-u_k^n-\frac{\Delta Q_k}{\varepsilon_{\infty,k}}\Big).
$$

$u^{n+1}$ follows from the discretization of Section "Dispersive materials" applied to each capacitor. Lorentz and Drude poles are explicit, $p^{n+1} = \alpha p^n+\xi p^{n-1}+\zeta u^n$. Debye poles and the conductivity use the trapezoidal rule, which gives one scalar division per capacitor:

$$
u^{n+1} = \frac{(\varepsilon_\infty-\mu\beta-\mu\hat\sigma)\,u^n + \Delta Q - \sum_L\mu(p_L^{n+1}-p_L^n) - \sum_D\mu(\kappa-1)p_D^n}{\varepsilon_\infty+\mu\beta+\mu\hat\sigma} .
$$

The discrete edge permittivity is then the exact Kottke combination of the *discrete* bulk permittivities $\varepsilon_d(\omega)$ of `DispersiveMaterial.epsilon_discrete`. For lossless Drude and Lorentz media the whole scheme equals the continuous-time one at the frequency $2\sin(\omega\Delta t/2)/\Delta t$, which the validation below uses to remove the leapfrog error exactly. The update is linear in $(E^*, E^n, u, p)$. Its transpose, in `AdjointStepper` through `NetworkTranspose`, is

$$
\lambda_{u'} = \lambda_{u}^{\mathrm{ext}} + \mathbf C^T\lambda_{E'} + \textstyle\sum_D\beta\lambda_{p_D'},\quad
\lambda_{\mathrm{acc}} = \lambda_{u'}/\mathrm{den},\quad
\lambda_{E^*} \mathrel{+}= \varepsilon_{r,s}\mathbf B^T\big(\lambda_{\mathrm{acc}} - \mathbf C^T\lambda_{E'}/\varepsilon_\infty\big),
$$

with the pole multipliers transposed as in `_DispersiveTranspose`. Memory: 1.3–1.4 capacitors per edge with a dispersive share (sphere at 16–24 cells across). That is about 65 B per such edge for a one-pole material and 170 B for gold (6 poles), and the run time goes up by 15 % (Drude) to 40 % (gold) for a sphere filling 6 % of the box.

**Passivity and stability (Wakis proof sketch).** Write the state as the displacements $\mathbf D$ (edges), $\mathbf H$, the capacitor polarizations $\mathbf P$ and velocities. The stored electric energy is

$$
U = \tfrac12\mathbf D^T\mathbf R\,\mathbf D + \sum_k \frac{V_k}{2}\Big[\frac{(q_k - \sum_p\mu_{kp}P_{kp})^2}{\varepsilon_0\varepsilon_{\infty,k}} + \sum_p\frac{\mu_{kp}\,\omega_{0,p}^2}{\varepsilon_0 S_p}P_{kp}^2\Big],\qquad q_k = \mathbf b_k\cdot\mathbf D,
$$

where $\mathbf R\succeq 0$ is the static elastance of the non-dispersive series parts (diagonal) and $V_k$ the capacitor's share of the dual volume. Every term is a positive semi-definite quadratic form, so the semi-discrete system is passive whenever every material is ($\gamma,\tau,\sigma\ge0$, $S_p = \Delta\varepsilon\,\omega_0^2$ or $\omega_p^2 \ge 0$). In the lossless case the update is exactly the leapfrog of $\ddot{\mathbf X} = -\mathbf M^{-1}\nabla^2U\,\mathbf X$ for $\mathbf X = (\mathbf D,\mathbf P)$: the D rows come from the curl–curl operator, and the P rows are the explicit central-difference ADEs. $\mathbf M^{-1} = \operatorname{diag}(\mathbf C^H\tilde{\mathbf D}\ldots\mathbf C,\ \varepsilon_0 S_p/\mu)$ is symmetric positive semi-definite, so $\mathbf M^{-1}\nabla^2 U$ has real non-negative eigenvalues. The step is stable iff $\Delta t^2\lambda_{\max}/4 \le 1$. A capacitor holds poles of total strength $\Omega_k^2 = \sum_p\mu_p S_p/\varepsilon_{\infty,k}$, and for the parallel mixture this is a weighted mediant bounded by the materials' bulk $\Omega_m^2 = \sum_p S_p/\varepsilon_{\infty,m}$. The bounds $(q-\sum P)^2 \le (1+\theta)q^2 + (1+\theta^{-1})(\sum\mu S)(\sum P^2/\mu S)$ split $U$ into a field part, which is at most $(1+\theta)$ times the static ($\varepsilon_\infty$-averaged) operator, and independent oscillators. Optimizing $\theta$ gives the sufficient condition

$$
r^2 + \frac{(\Delta t\,\Omega_{\max}/2)^2}{1-(\Delta t\,\omega_{0,\max}/2)^2} \le 1,\qquad r = \frac{\Delta t}{\Delta t_\text{static}},
$$

with $\Delta t_\text{static}$ the Lanczos/Arnoldi limit of the $\varepsilon_\infty$-averaged conformal operators (`solver.conformal_dt_max`). For a single lossless Drude pole in bulk this is exactly the sharp condition $(\omega_p\Delta t/2)^2 + C^2 \le \varepsilon_\infty$ of Section "Dispersive materials". The weights $c_k$ and fractions $f$ cancel, so subpixel averaging never tightens the time step beyond the bulk materials. `DispersiveNetwork.stability_bound` evaluates the condition, and the solver raises if it fails. Damping ($\gamma$, central differences) and the trapezoidal Debye and conductivity terms only remove energy. For Debye and $\sigma$ this is shown numerically (decay test below), not by the bound. The energy argument needs the network to be a sum of positive forms $\mathbf b_k\mathbf b_k^T$ ($\mathbf C = \mathbf W^{-1}\mathbf B^T\operatorname{diag}(V)$). The diagonal network has this form; the full tensor below does not.

**Why dispersive pixels get no off-diagonal couplings (Wakis analysis).** The full averaged tensor is $\mathbf T(\omega) = z_t(\mathbf I-\mathbf n\mathbf n^T) + z_n\mathbf n\mathbf n^T$, with $z_t = 1/\varepsilon_t$ and $z_n = 1/\varepsilon_n$. For one pixel it is positive real. The Cartesian operator of `conformal='tensor'`, however, takes the diagonal of each edge from that edge's own pixel and the off-diagonal entries from averages over neighbouring pixels. In an octant block the elastance $z_t(f_a)$ of pixel $a$ then multiplies the coefficient matrix $\begin{pmatrix}1-n_a^2 & -n_an_b/2\\ -n_an_b/2 & 0\end{pmatrix}$, which is indefinite. Where $z_t(f_a)$ has a pole, the block is not positive real. That happens at the mixture's epsilon-near-zero frequency $(1-f_a)\varepsilon_1 + f_a\varepsilon_2(\omega)=0$, which exists in some cut cell at every frequency where a metal has $\operatorname{Re}\varepsilon<0$. Equivalently, the off-diagonal coefficient $a = z_n - z_t$ alone has $\operatorname{Im}a = -f\operatorname{Im}\varepsilon_2\,(|\varepsilon_2|^{-2}-|\varepsilon_t|^{-2}) < 0$ (active) whenever $|\varepsilon_t|<|\varepsilon_2|$.

This was tested directly. `studies/tensor-dispersive/coupled.py` adds the dispersive off-diagonals $T_{ij}(\omega)$ to the corner couplings with the same capacitor engine (linear, exact transpose). On short runs it reproduces the static full-tensor accuracy for a Lorentz rod (second order: $-1.9\times10^{-3}$ at 32 cells to $-8.5\times10^{-5}$ at 192, against $+6.1\times10^{-3}$ to $+1.8\times10^{-3}$ for the diagonal). But in the 24–32 cell rod of the long-run tests, the field grows exponentially:

| Medium | Growth of $\lVert E\rVert$ |
| --- | --- |
| Lossless Drude | $10^{32}$ in $2\times10^4$ steps |
| Lossy Drude ($\gamma = 0.05\,\omega_p$) | $10^{56}$ in $5\times10^4$ steps |
| Lossless Lorentz | $10^{39}$ in $5\times10^4$ steps |

It is bounded for Debye and for a strongly damped Lorentz pole. Positive-real alternatives that share one tensor per octant are first order and less accurate than the diagonal. Two were tested on the frozen-$\varepsilon$ eigenvalues of the dielectric rod of `tests/test_035` (TE $n=0$): the exact octant fraction and normal ($1.0\times10^{-2}$, $6.4\times10^{-3}$, $3.5\times10^{-3}$, $1.7\times10^{-3}$ at 32–256 cells), and the average of the two edge tensors ($1.5\times10^{-2}$, $7.7\times10^{-3}$, $3.9\times10^{-3}$ at 32–128). The edge-pixel tensor gives $-2.1\times10^{-3}$ to $-5.8\times10^{-5}$ and the diagonal $6.7\times10^{-3}$ to $1.0\times10^{-3}$. The prototype reproduces the Wakis `'all'`/`'tensor'` numbers of `test_035` to two digits. A passive coupling needs the cross terms of the tangential part, $-z_t n_in_j$, to be carried by a positive form whose diagonal mixes the tangential elastances of pixels offset along the normal by half a cell. For a smooth field, full coupling forces a 50 % mixture, a series (harmonic) average across the interface instead of the parallel one, which is first order. Dispersive pixels therefore keep the provably passive diagonal, and non-dispersive pixels in `'tensor'` mode keep their full tensor.

**Accuracy for metals.** Kottke averaging rests on first-order perturbation theory, and its error constant contains $1/\varepsilon_t$ and $\varepsilon_n$. For $\operatorname{Re}\varepsilon_m<0$ these diverge in cells whose fraction is near the mixture's ENZ ($\varepsilon_t = 0$) or series pole ($1/\varepsilon_n = 0$). Such cells are local resonators near the frequencies of interest. The plasmon error therefore converges only at first order and not monotonically with the mesh. For dielectric-like media (Debye, Lorentz below resonance) the network has the accuracy of the static diagonal average. Okada and Cole [2012] report the same shift of surface-plasmon resonances by interpolated effective permittivities. For second-order plasmonics, use the interface-conforming `wakis.unstructured` solver ($1.3\times10^{-5}$ on the Drude rod below).

**Validation** (`tests/test_038_tensor_dispersive.py`; studies in `studies/tensor-dispersive/`, 2D on 8 CPU cores and 3D on the L4 GPU of the project VM). Frequencies are from harmonic inversion of the ring-down after a symmetric dipole pulse, with the leapfrog error removed. "Cells" means cells inside the cavity (rod) or in the box (sphere).

| Case | Reference | Staircase | Previous `'all'` ($\varepsilon_\infty$ only, = Meep) | Dispersive diagonal (this) |
| --- | --- | --- | --- | --- |
| 2D lossless Drude rod, $k_pa=4$, $b=a/2$, PEC cavity, TE $n=1$ plasmon ($\varepsilon=-0.20$); cells 1.5k / 6.0k / 10.6k / 23.9k / 42.5k / 95.7k | Bessel matching | $+0.13$ (spurious dominant mode), $-5.2\times10^{-3}$, $+2.5\times10^{-3}$, $+9.7\times10^{-4}$, $+8.8\times10^{-4}$, $-1.5\times10^{-4}$ | $-1.5\times10^{-2}$, $+1.4\times10^{-2}$, $+1.5\times10^{-2}$, $+1.0\times10^{-2}$, $+7.7\times10^{-3}$, $+6.5\times10^{-3}$; spurious modes of comparable amplitude | $-1.0\times10^{-2}$, $-3.9\times10^{-3}$, $-2.9\times10^{-3}$, $-2.1\times10^{-3}$, $-1.7\times10^{-3}$, $-1.2\times10^{-3}$; one dominant mode |
| 2D lossless Lorentz rod ($\varepsilon(\omega_\text{ref})=6$, test_035 geometry), TE $n=0$; 32/64/128/192 cells across | Delaunay FIT, Richardson | — | $-4.2\times10^{-2}$, $-2.1\times10^{-2}$, $-1.0\times10^{-2}$, $-6.7\times10^{-3}$ | $+6.1\times10^{-3}$, $+3.6\times10^{-3}$, $+1.8\times10^{-3}$, $+1.2\times10^{-3}$ (= static diagonal Kottke, order 1) |
| Conductive Debye slab, faces 0.3 cells off the nodes, $R(f)$ 0.6–1.4 GHz | Fabry–Pérot | $1.6\times10^{-2}$ | — | $2.9\times10^{-3}$ |
| 3D lossless Drude sphere ($b=a/2$, $k_pa=6$) in a PEC sphere; 16/24/32/48 cells across $2b$ (47k–1.2M cells), GPU | Riccati–Bessel matching | TM$_1$ ($ka=4.52995$): none at 16, $+9.2\times10^{-3}$ at 24, $-9.9\times10^{-3}$ at 48 (crowded spectrum, not run at 32) | TM$_1$: $-1.4\times10^{-3}$, —, $+1.4\times10^{-2}$, — (crowded) | TM$_1$ (weakly excited): $-1.7\times10^{-2}$, $-9.9\times10^{-4}$, $-6.0\times10^{-3}$, $+3.4\times10^{-4}$ (erratic); TM$_3$ ($ka=5.29458$, dominant): $-3.1\times10^{-3}$, $-1.8\times10^{-3}$, $-1.9\times10^{-3}$, $-8.5\times10^{-4}$ |
| 3D isolated lossless Drude sphere ($k_p=4$ m$^{-1}$, $b=0.2$ m), dipole LSPR with CPML, 16/24/32 cells across $2b$ (1.7M/5.8M/13.8M cells), GPU | complex Mie pole of $a_1$, $kb = 0.4287+0.0110i$ (quasi-static $0.462$) | 24 cells: $\operatorname{Re}k$ $+1.8\times10^{-2}$, $\operatorname{Im}k$ 86 % low (previous `'all'` identical) | — | $\operatorname{Re}k$: $+5.6\times10^{-3}$, $-9.1\times10^{-4}$, $-1.7\times10^{-2}$ (not monotone); $\operatorname{Im}k$ (radiative damping) 45–62 % low, independently of the domain size (0.75/1.5 m), the CPML thickness (10/30 cells) and the mesh (unresolved, see limitations) |
| Bulk (filled box), Drude + Lorentz + Debye + $\sigma$ | old ADE | — | — | identical to rounding |
| Long runs, $5\times10^4$ steps: lossless Drude, Lorentz, Drude + Lorentz | bounded | — | — | max/min of $\lVert E\rVert$ < 3 |
| Lossy (Drude, Lorentz, Debye, $\sigma$) | decaying | — | — | $<0.1$ of initial after $3\times10^4$ steps, never above |
| Adjoint dot product, `'all'`/`'tensor'`, PEC/CPML, all pole types | $\langle Ux,y\rangle = \langle x,U^Ty\rangle$ | — | — | $\le 5\times10^{-17}$; $>10^{-8}$ with one transposed coefficient perturbed by $10^{-3}$ |
| `MaterialGrid(material=Drude+Lorentz+Debye+σ, smoothing='tensor')`, 9×9 design, PEC/CPML | central differences, $h=10^{-5}$ | — | — | $2\times10^{-10}$ to $9\times10^{-9}$ |

For the Drude rod, the time-domain staircase results here differ from the "Wakis staircase" column of the unstructured-FIT table ($2.8\times10^{-3}$ at 42.5k). That table used a different source and probe and picked the strongest mode in a crowded spectrum. With the dipole filter used here, the staircase spectrum is dominated by spurious modes up to 64 cells (strongest mode $+13$ %), and from 96 cells on it settles at $\sim9\times10^{-4}$. The new diagonal converges monotonically, but slowly (order 0.7–1 in $h$): at 42.5k and 95.7k cells it is 4.6 and 5.2 times more accurate than the previous `'all'`, but 2 and 8 times less accurate than the staircase. The staircase error changes sign and is not monotone. For a lossless plasmon, Kottke averaging does not beat a fine staircase; its benefits are a clean spectrum at coarse meshes, monotone convergence, and smooth dependence on the geometry, which matters for gradients.

**`MaterialGrid(material=M, smoothing='tensor')`.** The design edges get the diagonal network of a pixel between `eps_min` and `M`, with fraction $f = \operatorname{clip}(\mathbf P\rho)$ and normal $\mathbf m$ from the density gradient (as in "Tensor subpixel averaging"). Each edge has a tangential capacitor ($c=1-m_c^2$, $\varepsilon_\infty = (1-f)\varepsilon_\min + f\varepsilon_{\infty,M}$, $\mu=f$) and a series capacitor ($c = m_c^2 f$). `ieps` holds the static Kottke value with $\varepsilon_{\infty,M}$. No off-diagonal couplings are created. The design parameters enter the step only through `ieps`, $c$, $\varepsilon_\infty$ and $\mu$, so `AdjointSimulation` records $\Delta Q$, $u^n$, $u^{n+1}$ and the pole sums $L=\sum_L(p'-p)$ and $D_s=\sum_D(\kappa-1)p$ per capacitor and step, and accumulates

$$
\frac{\partial F}{\partial c} = \sum_n\lambda_E\Big(u'-u-\frac{\Delta Q}{\varepsilon_\infty}\Big),\quad
\frac{\partial F}{\partial\varepsilon_\infty} = \sum_n\Big[\lambda_{u'}\frac{u-u'}{\mathrm{den}} + \lambda_E\frac{c\,\Delta Q}{\varepsilon_\infty^2}\Big],\quad
\frac{\partial F}{\partial\mu} = \sum_n\lambda_{u'}\frac{-L_s u - L - D_s - L_s u'}{\mathrm{den}},
$$

plus $\partial F/\partial\,\mathtt{ieps} = \sum_n\lambda_E\varepsilon_0\Delta Q$. Here $L_s = \sum_D\beta + \sigma\Delta t/2\varepsilon_0$ and $\lambda_{u'}$ is the total adjoint of the new capacitor voltage. The chain rule through $f$ and $\mathbf m$ follows `MaterialGrid._network_vjp`. The stored record takes $5\times2\times8$ B per design edge and step.

**Limitations and failure modes.**

- First order for metals, with a non-monotone error (resonant cut cells). No off-diagonal couplings on dispersive pixels (proved non-passive, observed unstable).
- One normal per edge (the last solid cutting the dual cell); several dispersive solids in one cell are mixed with the conformal overlap rule.
- Not covered:
  - MPI: each rank builds the network of its slab, as the ADE groups do.
  - The MPI adjoint.
  - Bloch phases (the network is real).
  - `field_energy`, which omits the capacitor energies.
  - Gradients of STL geometry.
- A dispersive tensor design cannot be combined with conformal dispersive solids in the same solver. Its design edges carry no couplings, even where $\rho$ selects the dielectric.
- The radiative damping of the open Drude sphere comes out ~45 % below the Mie pole for every setting tried, while its real part agrees to $10^{-3}$. The cause is not identified (a check against a non-dispersive dielectric sphere is the next step); use the real parts only.
- `conformal='pec'` and staircase grids keep the previous behaviour: poles on the cell mask, with the edges of the face weighted 1/2 by `add_dispersive_material`.

References:
- A. Deinega and I. Valuev, "Subpixel smoothing for conductive and dispersive media in the finite-difference time-domain method", *Opt. Lett.* **32**(23), 3429–3431 (2007), [doi:10.1364/OL.32.003429](https://doi.org/10.1364/OL.32.003429).
- A. Mohammadi, H. Nadgaran and M. Agio, "Contour-path effective permittivities for the two-dimensional finite-difference time-domain method", *Opt. Express* **13**(25), 10367–10381 (2005), [doi:10.1364/OPEX.13.010367](https://doi.org/10.1364/OPEX.13.010367).
- A. Mohammadi, T. Jalali and M. Agio, "Dispersive contour-path algorithm for the two-dimensional finite-difference time-domain method", *Opt. Express* **16**(10), 7397–7406 (2008), [doi:10.1364/OE.16.007397](https://doi.org/10.1364/OE.16.007397).
- J. Hamm, F. Renn and O. Hess, "Dispersive media subcell averaging in the FDTD method using corrective surface currents", *IEEE Trans. Antennas Propag.* **62**(2), 832–838 (2014), [doi:10.1109/TAP.2013.2292073](https://doi.org/10.1109/TAP.2013.2292073), [arXiv:1303.1403](https://arxiv.org/abs/1303.1403).
- A. Okada and D. Cole, "Effective permittivity for FDTD calculation of plasmonic materials", *Micromachines* **3**(1), 168–179 (2012), [doi:10.3390/mi3010168](https://doi.org/10.3390/mi3010168).
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983), ch. 4 (Mie coefficients of a sphere).
- Kottke et al. (2008): see "Conformal FIT"; the ADE discretization: see "Dispersive materials".

### 🔺 Unstructured FIT: triangles, tetrahedra and the Voronoi dual

The Cartesian solver represents a curved material interface either as a staircase (first order) or, with `conformal=True`, by partially filled cells. Its subpixel permittivity is diagonal, so it is only first order when $\mathbf E$ crosses a curved interface obliquely (TE modes of a dielectric rod, plasmonic surfaces). `wakis.unstructured` instead uses a mesh whose faces follow the interfaces: triangles in 2D and tetrahedra in 3D, each cell filled with one material. Body-fitted structured (nonorthogonal) FIT is the other route to curved boundaries [Schuhmann & Weiland, 2000; Cooke et al., 2006]; for curved interfaces a simplicial mesh keeps a diagonal material matrix and a Yee-like time step. `wakis.unstructured` is a separate solver core (`UnstructuredFIT2D`, `UnstructuredFIT3D`) and does not change `SolverFIT3D`.

**Grid doublet.** FIT is defined on any cell complex, not only on Cartesian grids [Clemens & Weiland, 2001]. The primal grid $G$ is the simplicial mesh (nodes, edges, faces, cells). The dual grid $\tilde G$ is its circumcentric (Voronoi) dual: dual nodes at the cell circumcentres, a dual edge through every primal face (circumcentre to circumcentre) and a dual face around every primal edge (the polygon of the circumcentres of the cells around that edge). Each dual edge is orthogonal to its primal face, and each dual face to its primal edge. This orthogonal "Delaunay–Voronoi grid doublet" is what keeps the material matrices diagonal [Clemens & Weiland, 2001]. Triangular FIT of this kind goes back to [van Rienen & Weiland, 1985]; on tetrahedra it is the Yee-like scheme of [Bossavit & Kettunen, 1999].

**Maxwell grid equations.** With $\mathbf e$ the voltages on the oriented primal edges, $\mathbf b$ the magnetic fluxes through the primal faces, $\mathbf C$ the face–edge incidence matrix (discrete curl), $\mathbf D$ the cell–face incidence matrix (discrete divergence) and $\mathbf G$ the edge–node incidence matrix (discrete gradient):

$$
\frac{d\mathbf b}{dt} = -\mathbf C\,\mathbf e,\qquad
\mathbf M_\varepsilon \frac{d\mathbf e}{dt} = \mathbf C^T\mathbf M_\nu\,\mathbf b - \mathbf j,\qquad
\mathbf D\,\mathbf C = 0,\quad \mathbf C\,\mathbf G = 0 .
$$

The incidence matrices are integer and mesh independent, so $\mathbf D\mathbf b = 0$ holds to round-off at every step. Only the Hodge (material) matrices carry metric and material:

$$
M_\varepsilon[e] = \frac{\varepsilon_0}{L_e}\sum_{T\ni e}\varepsilon_T\,\tilde A_{e,T},\qquad
M_\nu[f] = \frac{1}{A_f}\sum_{T\ni f}\frac{\tilde h_{f,T}}{\mu_0\mu_T}.
$$

Here $L_e$ and $A_f$ are the primal edge length and face area, $\tilde A_{e,T}$ is the part of the dual face of $e$ inside cell $T$ (two right triangles $m_e$–$c_f$–$c_T$ per face of $T$ through $e$), and $\tilde h_{f,T}$ is the part of the dual edge of $f$ inside $T$ (the signed distance from the face circumcentre $c_f$ to $c_T$). The permittivities add in parallel over the dual face, because tangential $\mathbf E$ is continuous. The reluctivities add in series along the dual edge, because normal $\mathbf B$ is continuous. Both are exact for interfaces made of mesh faces. When a circumcentre lies outside its cell (the cell is not well-centred), the pieces are signed. On edges where materials meet, the dual face is therefore split by clipping its pieces against the angular sectors of the cells around the edge in the bisector plane, which gives the exact area in each material (Wakis choice, `SimplexMesh._clip_interface_dual_areas`).

2D uses the same construction per unit length: TM has $E_z$ on the nodes with mass $\varepsilon_0\sum_T\varepsilon_T\tilde A_{n,T}$ and stiffness weights $\sum_T \tilde h_{e,T}/(\mu_0\mu_T L_e)$. TE has $H_z$ on the nodes, the electric flux $d_e$ through the primal edges, and the dual-edge voltage $v_e=\sum_T \tilde h_{e,T}\,E_{e,T}$ with $E_{e,T}=d_e/(L_e\varepsilon_0\varepsilon_T)$. The normal $D$ is continuous across the edge, so $1/\varepsilon$ is averaged in series along the dual edge, which is the correct interface condition when $\mathbf E$ crosses the interface. The weights $\tilde h_{e,T}/L_e = \tfrac12\cot\theta_T$ are the cotangent (P1 finite-element) weights.

**Delaunay requirement and fallbacks.** Summed over all cells, $\tilde A_e$ and $\tilde h_f$ are non-negative for interior objects of a Delaunay mesh (empty circumspheres). `SimplexMesh.diagnostics()` reports negative or tiny dual measures and cells that are not well-centred. The solvers refuse non-positive masses. Fallbacks:
- 2D, `hodge="barycentric"`: lumped barycentric node masses $A_T/3$ (positive on any mesh). The cotangent stiffness is the exact element stiffness (the symmetric Whitney Hodge [Schuhmann, Schmidt & Weiland, 2002] projected on node differences reduces to it) and is positive semi-definite on any mesh. The scheme stays stable for $\Delta t < 2/\sqrt{\lambda_\max(\mathbf M^{-1}\mathbf K)}$ and converges on non-Delaunay meshes (test with jittered nodes and flipped edges).
- 3D, `hodge="whitney"`: the consistent Whitney 1-form and 2-form mass matrices (exact integrals of products of barycentric coordinates; non-diagonal, positive definite on any mesh; the lowest-order Nédélec element, cf. the consistent tetrahedral material operators of [Cinalli et al., 2004]). The explicit update then needs one sparse solve with $\mathbf M_\varepsilon$ per step (factorized once).

**Degenerate Voronoi faces and the time step.** 3D Delaunay meshes contain edges whose Voronoi face is nearly degenerate (nearly co-spherical node sets), even when the cell quality is good. A tiny $\tilde A_e$ gives a tiny capacitance and collapses the explicit time step ($\Delta t\,c/h \approx 0.002$–$0.02$ in the tests). Edges with $\tilde A_e < \tau L_e^2$ (default $\tau = 10^{-2}$, `massless_tol`) are therefore treated as massless (Wakis choice). Their voltages $\mathbf e_s$ follow from the quasi-static constraint $(\mathbf C^T\mathbf M_\nu\mathbf b)_s = 0$, i.e. $\mathbf K_{ss}\mathbf e_s = -\mathbf K_{sf}\mathbf e_f$ with $\mathbf K = \mathbf C^T\mathbf M_\nu\mathbf C$ (sparse LU, factorized once). The eliminated system is a Schur complement, still symmetric positive semi-definite, so the leapfrog keeps a conserved energy. The time step rises to $\Delta t\,c/h\approx 0.13$–$0.17$ ($h$ = lattice spacing of `ball_mesh`; the Cartesian Yee limit is $0.58$). The neglected capacitance changes the dielectric-sphere modes by $4$–$7\times10^{-4}$ at $\tau = 10^{-2}$ ($h = a/7$ to $a/10$). This is small against the TM1 error but comparable to the TE1 error, whose field is tangential on the interface where most massless edges are. $\tau = 3\times10^{-3}$ halves this shift at a 40 % smaller time step.

**Time stepping and stability.** Leapfrog, $\mathbf b^{n+1/2} = \mathbf b^{n-1/2} - \Delta t\,\mathbf C\mathbf e^n$, then $\mathbf e^{n+1}$ from Ampère's law with the sources at $t^{n+1/2}$. $\Delta t = $ `cfl` $\cdot\,2/\sqrt{\lambda_\max}$, with $\lambda_\max$ of $\mathbf M_\varepsilon^{-1}\mathbf K$ from Lanczos iterations (`scipy.sparse.linalg.eigsh`). Conductivity and absorbing-boundary conductances are integrated semi-implicitly (average of $\mathbf e^n$ and $\mathbf e^{n+1}$). Every update is local and linear.

**Boundaries.** PEC: the voltages of all boundary edges are zero (removed from the unknowns). PMC is natural: the dual faces and edges end on the boundary. The first-order Silver–Müller condition $\mathbf H_t = \hat{\mathbf n}\times\mathbf E/\eta$ closes the open dual face of every boundary edge with its boundary segment $\tilde\ell_e$ (the in-face distances from $m_e$ to the circumcentres of the adjacent boundary faces). This adds a conductance $\tilde\ell_e/(\eta L_e)$ on boundary edges; in 2D TM it is $\ell_n/\eta$ on boundary nodes and in TE $\eta\,\ell_n$. It is exact at normal incidence and reflects a few percent on the curved outer boundaries tested. A CFS-PML is planned (`docs/unstructured_roadmap.md`).

**Sources and monitors.** A point dipole $I\ell(t)\,\hat{\mathbf d}\,\delta(\mathbf r-\mathbf r_0)$ is projected on the edges of its tetrahedron with the Whitney functions, $i_e = I\ell\,\hat{\mathbf d}\cdot\mathbf W_e(\mathbf r_0)$. This is the Galerkin projection, so the radiated field does not depend on the mesh size. In 2D the analogous projections use P1 hat functions (TM line current, TE magnetic line current) and Raviart–Thomas functions (TE in-plane dipole). Distributed currents use $i_e = \tilde A_e\,\mathbf J(m_e)\cdot\hat{\mathbf t}_e$. Probes interpolate with the same functions, and DFT monitors accumulate $\sum_n u^n e^{-i\omega t_n}\Delta t$ as `DFTMonitor` does.

**Dispersive media.** Drude, Lorentz and Debye poles use the ADEs and coefficients of `wakis.dispersive` (same conventions, $e^{+i\omega t}$). In 3D and in 2D TM the dispersive part of a dual face (fraction $\tilde A_{e,m}/\tilde A_e$ in material $m$) carries its own polarization in parallel with the rest. This is the edge update of `DispersiveMixin` with the fill fraction replaced by the exact dual-face fraction. In 2D TE every dual-edge segment inside a dispersive triangle carries its own field $E_{e,T}$ and polarization. The normal $D$ is shared, so $E_{e,T}^{n+1} = (d_e/(L_e\varepsilon_0) - \sum p^{n+1})/\varepsilon_\infty$ (series), which is exact for a field crossing a metal surface. `UnstructuredFIT2D.dispersive_eigenmode` solves the nonlinear eigenproblem $\mathbf K(\omega)\mathbf u = \omega^2\mathbf M\mathbf u$ by a secant iteration on the frequency.

**Eigenmodes.** `eigenmodes(n, f_target)` solves $\mathbf K\mathbf e=\omega^2\mathbf M_\varepsilon\mathbf e$ by shift-invert Lanczos (SuperLU of $\mathbf K-\sigma\mathbf M_\varepsilon$, $\sigma = (2\pi f_\text{target})^2$). The static null space $\mathbf e=\mathbf G\phi$ is projected out $\mathbf M_\varepsilon$-orthogonally inside the operator (discrete Poisson solve), so it is never returned. The `n` modes closest to $\sigma$ are returned; choose $\sigma$ so that $\omega^2 < 2\sigma$ for the wanted modes.

**Mesh generation.** `disk_mesh` and `ball_mesh` (pure Python) place nodes on every interface (rings, or Fibonacci points on spheres) and on a hexagonal or body-centred-cubic lattice elsewhere. They triangulate with `scipy.spatial.Delaunay`, smooth with optimal-Delaunay (ODT) sweeps [Chen & Xu, 2004], and remove lattice nodes inside the diametral ball of any shell triangle. The shell triangles are then Delaunay faces, so every interface is exactly a union of mesh faces. By default each interior interface is moved radially so that its polygon (polyhedron) encloses the exact area (volume) of its circle (sphere). This removes the leading geometric error of the inscribed polygon while keeping second order; the outer 3D boundary stays inscribed. For general geometry, `gmsh_mesh` drives the external generator gmsh (GPL; imported only when called, never a dependency) and reads its `.msh` file with meshio. Any meshio-readable file can be loaded with `SimplexMesh.read` (cell tags from `gmsh:physical`).

**Memory and run time** (measured on a 32-core VM): mesh, solver and time stepping use about 2.1 kB per tetrahedron. A leapfrog step costs about 45 ns per tetrahedron (single process, SciPy sparse). A shift-invert eigensolve adds the SuperLU fill-in, about 0.3 GB at 1.5×10⁴ tetrahedra and 1.7 GB at 4.5×10⁴. `wakis.unstructured.estimate_memory(mesh, eigen=True)` returns the calibrated estimate.

**Validation** (`tests/test_034_unstructured_*.py`; the convergence studies ran on the VM). "Cells" counts triangles or tetrahedra for the unstructured solver and Cartesian cells inside the cavity for Wakis (one Cartesian cell ≈ two triangles or about five tetrahedra in unknowns).

| Case | Unstructured FIT (cells: error) | Wakis staircase | Wakis conformal |
| --- | --- | --- | --- |
| 2D rod ($\varepsilon_r=6$, $b=a/2$) in PEC cavity, TM (E tangential) | 705: $1.2\times10^{-4}$; 2 860: $2.5\times10^{-5}$; 11 540: $5.6\times10^{-6}$; 46 269: $1.2\times10^{-6}$ (order 2) | 665: $7.5\times10^{-2}$ … 42 539: $9.8\times10^{-3}$ (order 1) | 665: $1.3\times10^{-4}$; 10 635: $1.2\times10^{-5}$; 42 539: $1.2\times10^{-5}$ |
| 2D rod, TE (E crosses the interface) | 705: $2.0\times10^{-4}$; 2 860: $4.7\times10^{-5}$; 11 540: $1.6\times10^{-5}$; 46 269: $4.4\times10^{-6}$ (order 2) | 665: 0.11 … 42 539: $1.5\times10^{-2}$ (order 1) | 665: $4.9\times10^{-3}$; 2 659: $2.2\times10^{-3}$; 10 635: $1.2\times10^{-3}$; 42 539: $5.7\times10^{-4}$ (order 1) |
| 2D rod with inscribed rings (as in the `nfit-study` prototype) | TM and TE: 2 860: $4.4\times10^{-4}$; 46 269: $2.7\times10^{-5}$ (order 2) | | |
| 2D rod, barycentric fallback | TM 46 269: $1.9\times10^{-6}$; TE: $4.3\times10^{-6}$ (order 2) | | |
| 2D lossless Drude rod ($k_p a = 4$), TE $n=1$ mode ($ka = 3.64691$, $\varepsilon(\omega)=-0.20$) | eigen: 705: $7.4\times10^{-4}$; 2 860: $2.2\times10^{-4}$; 11 540: $4.8\times10^{-5}$; 46 269: $1.3\times10^{-5}$ (order 2). Time domain + harmonic inversion: $9.4\times10^{-4}$, $2.1\times10^{-4}$, $5.8\times10^{-5}$ (0.5 s, 1.5 s, 5.4 s) | 665: $8.5\times10^{-3}$; 2 659: $3.7\times10^{-3}$; 10 635: $2.6\times10^{-3}$; 42 539: $2.8\times10^{-3}$ (stalls) | 665: $8.9\times10^{-3}$; 2 659: $1.2\times10^{-3}$; 10 635: $5.7\times10^{-3}$; 42 539: $6.3\times10^{-3}$ (no convergence) |
| 3D PEC sphere, TE11 ($ka=4.4934$, triplet mean) | 2 355: $-1.6\times10^{-2}$; 12 293: $-5.7\times10^{-3}$; 44 579: $-2.0\times10^{-3}$; 73 555: $-1.4\times10^{-3}$ (order 2.2 in $h$) | | |
| 3D PEC sphere, TM11 ($ka=2.7437$) | $\le 3.4\times10^{-4}$ from 12 293 tetrahedra on (dispersion and inscribed-wall errors cancel; $-2.7\times10^{-3}$ at 1 339) | | |
| 3D $\varepsilon_r=4$ sphere ($b=a/2$) in PEC sphere, TM1 (E crosses the surface, $ka=2.16342$) | 2 116: $-9.9\times10^{-3}$; 11 548: $-3.4\times10^{-3}$; 42 124: $-1.2\times10^{-3}$; 69 892: $-7.9\times10^{-4}$ (order 2.0 in $h$) | 5 438: 0.14; 43 506: $7.2\times10^{-2}$; 103 124: $5.6\times10^{-2}$ | 5 438: $5.9\times10^{-3}$; 43 506: $3.0\times10^{-3}$; 103 124: $2.3\times10^{-3}$ (order 1 in $h$) |
| same, TE1 ($ka=3.02433$) | $\le 2.5\times10^{-3}$ on all meshes, $1.5\times10^{-3}$ at 69 892 (not monotone: massless-edge capacitance and mesh noise; $6\times10^{-4}$ at 36 056 with `massless_tol=0`) | 12 891: $6.3\times10^{-3}$; 103 124: $4.4\times10^{-2}$ | 12 891: $3.4\times10^{-3}$; 103 124: $2.1\times10^{-3}$ |
| 3D cavity filled with a lossless Drude plasma ($k_pa=3$) | TM11 triplet at $\omega^2=\omega_c^2+\omega_p^2$ within $6\times10^{-4}$ (time domain, leapfrog error) | | |
| Discrete identities, stability | $\max|\mathbf D\mathbf b|/\max|\mathbf b| \approx 2\times10^{-14}$; energy bounded over $5\times10^4$ (2D) and $10^4$ (3D) steps; time-domain resonances match the eigen-solver to $2\times10^{-5}$ after the exact leapfrog correction $\sin(\omega_{td}\Delta t/2)=\omega\Delta t/2$ | | |
| 2D line source, Silver–Müller boundary at $3\lambda$ (16 nodes/λ) | $|E_z|$ within 8 % of the exact $H_0^{(2)}$ Green's function at 4 radii (PEC: errors up to 100 %) | | |

The unstructured mesh is second order in both polarizations. Where $\mathbf E$ crosses a curved interface it is more accurate than the Cartesian conformal method at equal cell counts, and the gap grows with refinement. At about $4.5\times10^4$ cells: 2D TE rod $4.4\times10^{-6}$ against $5.7\times10^{-4}$ (130×), Drude plasmon $1.3\times10^{-5}$ against $6\times10^{-3}$, and 3D TM1 $1.2\times10^{-3}$ against $3.0\times10^{-3}$ (2.5×). For tangential $\mathbf E$ (2D TM) the Cartesian conformal method is about as accurate per cell up to $10^4$ cells. A triangle carries about half the unknowns of a square cell (TM: one node per two triangles), and a tetrahedron about a fifth of the unknowns of a Cartesian cell (three edges). A 3D step costs 45 ns per tetrahedron. The time step is $\Delta t\,c/h\approx0.12$–$0.17$ ($h$ = lattice spacing), against $0.58$ for the Cartesian grid.

![Unstructured meshes and the plasmon mode](img/unstructured_meshes.png)
![2D convergence](img/unstructured_convergence_2d.png)
![3D convergence](img/unstructured_convergence_3d.png)

**Differentiability.** The update is linear with fixed sparse operators: $\mathbf C$, the diagonal $\mathbf M_\nu$, the diagonal coefficients of `CapacitiveUpdate` and, when present, the fixed Schur complement of the massless edges. The adjoint is the transposed leapfrog (see the roadmap), and frequency-domain outputs are plain arrays from the DFT data. The derivative of the Hodge with respect to a cell permittivity is the fixed sparse map $\partial M_\varepsilon[e]/\partial\varepsilon_T = \varepsilon_0\tilde A_{e,T}/L_e$ (`mesh.dual_area_partial`).

**Limitations.** No PML yet (first-order absorbing boundary only). No GPU or MPI. The 3D time step is about 4 times below the Cartesian Yee step at equal lattice spacing. Dispersive eigenmodes exist only in 2D TE; 3D dispersive cases run in the time domain. Polyhedral PEC walls stay inscribed. The eigensolver uses SuperLU, whose fill-in limits it to about $10^5$ tetrahedra.

References:
- M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique", *Progress In Electromagnetics Research* **32**, 65–87 (2001).
- U. van Rienen and T. Weiland, "Triangular discretization method for the evaluation of RF-fields in cylindrically symmetric cavities", *IEEE Trans. Magn.* **21**(6), 2317–2320 (1985).
- A. Bossavit and L. Kettunen, "Yee-like schemes on a tetrahedral mesh, with diagonal lumping", *Int. J. Numer. Model.* **12**(1/2), 129–142 (1999).
- R. Schuhmann and T. Weiland, "The nonorthogonal finite integration technique applied to 2D- and 3D-eigenvalue problems", *IEEE Trans. Magn.* **36**(4), 897–901 (2000).
- R. Schuhmann, P. Schmidt and T. Weiland, "A new Whitney-based material operator for the finite integration technique on triangular grids", *IEEE Trans. Magn.* **38**(2), 409–412 (2002).
- M. Cinalli, F. Edelvik, R. Schuhmann and T. Weiland, "Consistent material operators for tetrahedral grids based on geometric principles", *Int. J. Numer. Model.* **17**, 487–507 (2004).
- S. J. Cooke, R. Shtokhamer, A. A. Mondelli and B. Levush, "A finite integration method for conformal, structured-grid, electromagnetic simulation", *J. Comput. Phys.* **215**, 321–347 (2006).
- L. Chen and J. Xu, "Optimal Delaunay triangulations", *J. Comput. Math.* **22**(2), 299–308 (2004).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).

### ⏱️ Local time stepping (LTS) for the unstructured leapfrog

**Why.** An explicit leapfrog is stable for $\Delta t < 2/\sqrt{\lambda_\max(\mathbf M^{-1}\mathbf K)}$, and $\lambda_\max$ is set by the smallest (or most strongly coupled, or least massive) cells. On a graded mesh the whole domain then pays the time step of the finest region. In the graded Mie-cylinder meshes (`studies/mie-cylinder-graded`) the size field keeps equal cells per *local* wavelength: $h_\text{in}$ inside the $n=2$ cylinder, $h_\text{out}=2h_\text{in}$ in the far vacuum. Both regions have the same local step limit, about twice that of the thin vacuum band of size $h_\text{in}$ around the cylinder, which alone sets the global $\Delta t$. `UnstructuredFIT2D(..., lts=2)` lets that band take two half steps while everything else takes one step of $2\Delta t$.

**Second-order form.** Eliminating the edge fluxes from the 2D leapfrog (TM: $\mathbf v^{n+1/2}=\mathbf v^{n-1/2}-\Delta t\,\mathbf G\mathbf y^n$, $\mathbf M(\mathbf y^{n+1}-\mathbf y^n)=\Delta t\,\mathbf G^T\mathbf S\mathbf v^{n+1/2}$; TE is the same with the roles of the half and integer levels swapped) gives

$$
\mathbf y^{n+1}-2\mathbf y^n+\mathbf y^{n-1} = -\Delta t^2\,\mathbf A\,\mathbf y^n,\qquad \mathbf A=\mathbf M^{-1}\mathbf K,\quad \mathbf K=\mathbf G^T\mathbf S\mathbf G,
$$

with $\mathbf y$ = $E_z$ (TM) or $H_z$ (TE) on the free nodes, $\mathbf M$ the diagonal node mass and $\mathbf S$ the diagonal edge stiffness (`solver.stiff`).

**Diaz–Grote LTS** [Diaz & Grote, 2009]. Let $\mathbf P$ be the diagonal 0/1 projector on the fine nodes $F$. Exactly, $\mathbf y(t_n+\tau)+\mathbf y(t_n-\tau)$ is twice the even solution $\mathbf w(\tau)$ of $\mathbf w''=-\mathbf A\mathbf y$. Freeze the coarse part of the force at $t_n$ and keep the fine part live:

$$
\mathbf w''(\tau) = -\mathbf A(\mathbf I-\mathbf P)\,\mathbf y^n-\mathbf A\mathbf P\,\mathbf w(\tau),\qquad \mathbf w(0)=\mathbf y^n,\ \mathbf w'(0)=0,\qquad \mathbf y^{n+1}=-\mathbf y^{n-1}+2\mathbf w(\Delta t).
$$

Integrating $\mathbf w$ with $p$ leapfrog substeps of $\Delta t/p$ gives $\mathbf w(\Delta t)=\mathbf y^n-\tfrac{\Delta t^2}{2}\mathbf A\,r(\mathbf X)\,\mathbf y^n$ with $\mathbf X=\Delta t^2\mathbf P\mathbf A$, i.e.

$$
\boxed{\ \mathbf y^{n+1}-2\mathbf y^n+\mathbf y^{n-1}=-\Delta t^2\,\mathbf A\,r(\mathbf X)\,\mathbf y^n\ }
$$

with $r$ a polynomial of degree $p-1$, $r(0)=1$. The p-substep leapfrog for a scalar mode $\lambda$ inside $F$ gives $x\,r(x) = 2-2T_p(1-x/2p^2)$, $x=\Delta t^2\lambda$ ($T_p$ the Chebyshev polynomial). For $p=2$: $r(\mathbf X)=\mathbf I-\mathbf X/16$, one extra product with the rows of $\mathbf A$ on $F$ per coarse step.

**Stabilization.** For $\nu=0$, $x\,r(x)$ reaches the stability bound 4 at interior points ($x=8$ for $p=2$): modes inside $F$ at mid-spectrum are marginal, and coupled modes near the interface can exceed the bound. This is the instability "at certain time steps" that Grote, Michel & Sauter [2021] remove with a stabilized LTS. Wakis uses the damped Chebyshev polynomial of leapfrog–Chebyshev schemes [Carle, Hochbruck & Sturm, 2020] (Wakis choice of the exact form; the form used in [Grote, Michel & Sauter, 2021] was not checked against it):

$$
x\,r(x) = 2-2\,\frac{T_p(\delta-\beta x)}{T_p(\delta)},\qquad \delta=1+\frac{\nu}{p^2},\quad \beta=\frac{T_p(\delta)}{2T_p'(\delta)} .
$$

$\beta$ gives $r(0)=1$ (consistency). For $\delta-\beta x\in[-1,\delta]$, $0\le x\,r(x)\le 2+2/T_p(\delta)<4$, so modes inside $F$ keep a margin; the range is $x\le 2(1+\delta)T_p'(\delta)/T_p(\delta)$ ($16\to 14.94$ for $p=2$, $\nu=0.15$, i.e. the fine substep may use 96.6 % of its own limit). Default $\nu=0.15$: $\max x\,r(x)=3.735$, $\rho\ge0.966$. $r(\mathbf X)\mathbf y$ is evaluated with the three-term recurrence of the divided difference $D_k(s)=(T_k(\delta)-T_k(s))/(\delta-s)$: $\mathbf d_{k+1}=2\delta\mathbf d_k-2\beta\mathbf X\mathbf d_k+2T_k(\delta)\mathbf y-\mathbf d_{k-1}$, $\mathbf d_0=0$, $\mathbf d_1=\mathbf y$, $r(\mathbf X)\mathbf y=2\beta\,\mathbf d_p/T_p(\delta)$. Off $F$, $\mathbf X\mathbf d=0$, so $\mathbf d_k=D_k(\delta)\mathbf y$ in closed form and only the rows of $\mathbf A$ on $F$ (columns: $F$ and its first neighbours, the implicit buffer) are touched: $p-1$ sparse products of size $\approx 7|F|$.

**Energy and stability (proof).** $\mathbf K\,r(\mathbf X)=\sum_j c_j\Delta t^{2j}\mathbf K(\mathbf P\mathbf M^{-1}\mathbf K)^j$, and $\big(\mathbf K(\mathbf P\mathbf M^{-1}\mathbf K)^j\big)^T=(\mathbf K\mathbf M^{-1}\mathbf P)^j\mathbf K=\mathbf K(\mathbf P\mathbf M^{-1}\mathbf K)^j$ because $\mathbf P$ and $\mathbf M$ are diagonal. So $\mathbf K_p=\mathbf K\,r(\mathbf X)$ is symmetric, and the LTS is an ordinary leapfrog with the symmetric stiffness $\mathbf K_p$. Multiplying the scheme by $(\mathbf y^{n+1}-\mathbf y^{n-1})^T\mathbf M$ telescopes into the conserved quantity

$$
E^{n+1/2}=\tfrac12\Big[\mathbf w^T\big(\mathbf M-\tfrac{\Delta t^2}{4}\mathbf K_p\big)\mathbf w+\bar{\mathbf y}^T\mathbf K_p\bar{\mathbf y}\Big],\qquad \mathbf w=\frac{\mathbf y^{n+1}-\mathbf y^n}{\Delta t},\ \bar{\mathbf y}=\frac{\mathbf y^{n+1}+\mathbf y^n}{2}
$$

(`lts.modified_energy`; the energy of the time-differentiated field). It is a norm, and the scheme is stable, iff $\mathbf K_p\succeq0$ and $\rho=\tfrac{\Delta t}{2}\sqrt{\lambda_\max(\mathbf M^{-1}\mathbf K_p)}<1$. Wakis does not rely on an a-priori CFL bound: at setup it computes $\rho$ by Lanczos on the symmetric $\mathbf M^{-1/2}\mathbf K_p\mathbf M^{-1/2}$ (`lts.stability_ratio`) and accepts the partition only if $\rho\le\rho_\max=0.98$. Leapfrog dispersion then holds mode by mode, $\sin(\omega\Delta t/2)=\tfrac{\Delta t}2\sqrt{\lambda(\mathbf M^{-1}\mathbf K_p)}$, which is how the tests compare LTS and global eigenfrequencies.

**First-order form.** The solver keeps its (node field, edge flux) state; only the flux update changes: TM $\mathbf v\mathrel{-}=\Delta t\,\mathbf G\,r(\mathbf X)\mathbf u$, TE $\mathbf v\mathrel{+}=\Delta t\,(\mathbf G\,r(\mathbf X)\mathbf u-\dots)$, implemented as $\mathbf G\mathbf u+\mathbf G_{:,F}\,\delta\mathbf u_F$ (`_grad`). Sources, the scattered-field plane wave, node losses and monitors are unchanged and run at the coarse step.

**Partition.** Node rates $g_n=\sum_m|K_{nm}|/M_n$ (Gershgorin: $\max_n g_n\ge\lambda_\max$) are scaled so that $\max g_n=\lambda_\max$; node $n$ is fine if $\Delta t^2 g_n>4$. The node-local estimate cannot resolve a few % of CFL margin, so a repair loop follows: while $\rho>\rho_\max$, if the top Lanczos mode lives mostly on $F$ and its buffer, $F$ grows by one layer of neighbours; otherwise (a mode of the coarse region itself) $\Delta t$ shrinks by $0.995\,\rho_\max/\rho$. On the graded Mie meshes this gives $\Delta t=1.94$–$1.97\times$ the global step with 5–23 % of the nodes fine (`solver.lts_info`).

**Adjoint.** The step is linear and $r(\mathbf X)^T=r(\mathbf X^T)$, $\mathbf X^T=\Delta t^2\mathbf K\mathbf M^{-1}\mathbf P$, evaluated with the same recurrence (`LocalTimeStepping.apply_T`). `lts.linear_step` / `lts.linear_step_T` are the source-free step and its exact transpose; the dot-product test holds to $10^{-15}$ (tests).

**Cost.** Per coarse step: one full step plus $p-1$ products with $\mathbf A_{F,:}$ and one with $\mathbf G_{:,F}$. With fine fraction $f$ and $\Delta t_\text{LTS}=s\,\Delta t$, the work relative to global stepping is about $(1+c\,(p-1)f)/s$ with $c\approx0.5$.

**Cartesian graded grids (`SolverFIT3D`, not implemented).** The same polynomial applies to $\mathbf e''=-\mathbf M_\varepsilon^{-1}\mathbf C^T\mathbf M_\nu\mathbf C\,\mathbf e$ with $\mathbf P$ on edges: $\mathbf h^{n+1/2}=\mathbf h^{n-1/2}-\Delta t\,\mathbf M_\mu^{-1}\mathbf C\,r(\mathbf X)\mathbf e^n$. Needed: the rows of $\mathbf M_\varepsilon^{-1}\mathbf C^T\mathbf M_\nu\mathbf C$ on $F$ as a sparse matrix (CuPy for GPU), the Lanczos check on the 3D operator, CPML and dispersive/conformal ADE edges kept outside $F$ (or stepped at the fine rate), and the MPI halo extended by one layer for the extra product. On a rectilinear grid the local limit depends on $1/\Delta x^2+1/\Delta y^2+1/\Delta z^2$, so every cell on a fine strip is fine: only ~22 % of the cells of the graded Mie grid are coarse in both directions, so LTS would save about 10 % there. The larger Cartesian use case is conformal small cells (cut cells with small $\tilde A_e$), which LTS would place in $F$ instead of reducing the global step.

**Tetrahedra (`UnstructuredFIT3D`, not implemented).** With the diagonal Voronoi Hodge the construction is identical on edges ($\mathbf K=\mathbf C^T\mathbf M_\nu\mathbf C$). Two complications: the massless-edge Schur complement makes $\mathbf K$ non-local (each product needs the sparse LU solve), so $F$ must avoid edges coupled to massless edges or the Schur complement must be restricted to $F$; and the Whitney Hodge has a non-diagonal $\mathbf M_\varepsilon$, for which $\mathbf P\mathbf M^{-1}$ is not local and $\mathbf K_p$ loses its symmetry unless the mass is lumped on $F$.

**Limits & failure modes.**
- Two levels only: $p$ is any integer, but a single coarse/fine ratio. Meshes with a continuum of sizes would need the multi-level recursion [Diaz & Grote, 2015].
- Dispersive media: Lorentz and Drude poles are allowed, Debye poles are refused. With these poles the polarization at $t^{n+1}$ depends only on past fields, so in the second-order form it is a force frozen over the coarse step, like a source. The LTS operator is built from the $\varepsilon_\infty$ Hodge. The poles are stepped at the coarse $\Delta t$, so $\omega_p\Delta t$ must pass the von Neumann check of `DispersiveMaterial.is_stable` at the LTS ratio $\rho$; the solver enforces this. For Rakić gold at $h = 4$ nm this allows $p = 2$ but not $p = 3$. There is no energy proof for the coupled system. It is checked numerically instead: the dense one-step amplification matrix (fields and ADE states) of a Drude rod inside $F$ has spectral radius $1 + O(10^{-7})$ (round-off, the same as global stepping) for TE and TM, lossless and lossy. Removing the fine substeps gives 1.8. A too-fast pole is refused.
- Lossy nodes or edges in $F$ (ABC, absorber) are refused at setup, since the loss is not part of the energy proof. The radial CFS-PML is allowed only if $F$, its buffer and their edges lie outside the layer (checked); the PML layer itself steps at the coarse $\Delta t$.
- Sources inside $F$ are sampled at the coarse step: second order, but a pulse with spectral content near the fine Nyquist rate is aliased. Monitors also record at the coarse rate.
- The coarse step is $\le p\times$ the global step and the fine substep uses at most 96.6 % ($\nu=0.15$, $p=2$) of its limit; $\rho\ge0.966$ is the floor set by the damped polynomial.
- The time error in the coarse region grows with the larger step (about $(\Delta t_\text{LTS}/\Delta t)^2\approx4\times$ the global time error in the cavity tests, still second order). Where the spatial error dominates (the Mie meshes) the total error is unchanged; where the time error dominates, LTS loses accuracy.
- $\nu=0$ (the unstabilized Diaz–Grote polynomial) gives $\rho=1$ to Lanczos precision: the modified energy is still conserved, but it is no longer a norm and the field energy beats up to ~300× (study figure b).
- The partition is node based. A 2D triangle mesh with a few slivers produces small isolated $F$ sets; the repair loop handles their interface modes but each Lanczos check costs a few hundred operator products at setup.

References:
- J. Diaz and M. J. Grote, "Energy conserving explicit local time stepping for second-order wave equations", *SIAM J. Sci. Comput.* **31**(3), 1985–2014 (2009), doi:10.1137/070709414.
- M. J. Grote and T. Mitkova, "Explicit local time-stepping methods for Maxwell's equations", *J. Comput. Appl. Math.* **234**(12), 3283–3302 (2010), doi:10.1016/j.cam.2010.04.028.
- M. J. Grote and T. Mitkova, "High-order explicit local time-stepping methods for damped wave equations", *J. Comput. Appl. Math.* **239**, 270–289 (2013), doi:10.1016/j.cam.2012.09.046.
- J. Diaz and M. J. Grote, "Multi-level explicit local time-stepping methods for second-order wave equations", *Comput. Methods Appl. Mech. Engrg.* **291**, 240–265 (2015), doi:10.1016/j.cma.2015.03.027.
- M. J. Grote, M. Mehlin and S. A. Sauter, "Convergence analysis of energy conserving explicit local time-stepping methods for the wave equation", *SIAM J. Numer. Anal.* **56**(2), 994–1021 (2018), doi:10.1137/17M1121925.
- M. J. Grote, S. Michel and S. A. Sauter, "Stabilized leapfrog based local time-stepping method for the wave equation", *Math. Comp.* **90**(332), 2603–2643 (2021), doi:10.1090/mcom/3650.
- C. Carle, M. Hochbruck and A. Sturm, "On leapfrog-Chebyshev schemes", *SIAM J. Numer. Anal.* **58**(4), 2404–2433 (2020), doi:10.1137/18M1209453.
- Related lead (not read; authors per search result only): "Local time-integration for Friedrichs' systems", arXiv:2503.05282 (2025), leapfrog-based local integration with a scaled and shifted Chebyshev filter, covering Maxwell.
- S. Piperno, "Symplectic local time-stepping in non-dissipative DGTD methods applied to wave propagation problems", *ESAIM: M2AN* **40**(5), 815–841 (2006), doi:10.1051/m2an:2006035.
- E. Montseny, S. Pernet, X. Ferrières and G. Cohen, "Dissipative terms and local time-stepping improvements in a spatial high order Discontinuous Galerkin scheme for the time-domain Maxwell's equations", *J. Comput. Phys.* **227**(14), 6795–6820 (2008), doi:10.1016/j.jcp.2008.03.032.
- R. A. Chilton and R. Lee, "Conservative and provably stable FDTD subgridding", *IEEE Trans. Antennas Propag.* **55**(9), 2537–2549 (2007), doi:10.1109/TAP.2007.904092.
- J.-P. Bérenger, "A Huygens subgridding for the FDTD method", *IEEE Trans. Antennas Propag.* **54**(12), 3797–3804 (2006), doi:10.1109/TAP.2006.886519.

### 🔷 Polygonal FIT: hybrid square/triangle meshes and Voronoi cells

`wakis.unstructured.PolyMesh` and `PolygonFIT2D` extend the 2D unstructured FIT from triangles to cells that are arbitrary simple polygons: triangles, rectangles, Voronoi polygons, or any mix of them. In 2D every unknown lives on a node or an edge (TM: $E_z$ on nodes and $b_e$ through edges; TE: $H_z$ on nodes and $d_e$ through edges), so the topology needs only the edge–node incidence $\mathbf G$. Cells enter only through the Hodge (material) operators.

**Dual geometry.** Each cell $C$ has a dual point $c_C$. The dual edge of primal edge $e$ inside $C$ is the segment from $c_C$ to its foot $x_{e,C}$ on the line of $e$, with signed length $h_{e,C}$ (positive when $c_C$ is on the cell side). The foot position $s_{e,C}$ is measured along $e$ from its first node. The dual cell of node $n$ inside $C$ is the quadrilateral $(n, x_{e_1,C}, c_C, x_{e_2,C})$, of area $A_{n,C}=\tfrac12\sum_{e\ni n} s^{(n)}_{e,C}\,h_{e,C}$. These pieces tile each cell exactly, for any choice of $c_C$.

**Orthogonality.** The diagonal Hodge is consistent when the dual edge crosses $e$ as one straight segment perpendicular to it, i.e. when both cells of $e$ have the same foot ($s_{e,C_1}=s_{e,C_2}$). `PolyMesh.orthogonality_defect` $=|s_{e,C_1}-s_{e,C_2}|/L_e$ measures this. It is zero for:
- Delaunay triangles with circumcentres (feet at the edge midpoints): `PolygonFIT2D` then reproduces `UnstructuredFIT2D` to round-off (test);
- rectangles with their centres (midpoints);
- **hybrids of the two**: the shared edges between a square and a triangle have midpoint feet on both sides. So a square bulk joined to a Delaunay triangle patch is orthogonal everywhere, with no special interface treatment;
- **Voronoi polygons with their generators as dual points.** Neighbouring generators are mirror images across the shared Voronoi edge, so the dual edge is the Delaunay edge. Its foot is the midpoint of the generator pair, not of the Voronoi edge. Then $h_{e,C}=|p_i-p_j|/2>0$ and the node masses are Delaunay triangle areas, positive for every point set. The well-centredness condition of the triangle-primal mesh never arises.

**Maxwell grid equations** (per unit length, $w_C=1/(\mu_0\mu_C)$ for TM and $1/(\varepsilon_0\varepsilon_C)$ for TE):

$$
\dot{\mathbf b}=-\mathbf G\mathbf e_z,\quad \mathbf M_\varepsilon\dot{\mathbf e}_z=\mathbf G^T\mathbf S\,\mathbf b-\mathbf i;\qquad
\dot{\mathbf d}=\mathbf G\mathbf h_z-\mathbf i_e,\quad \mathbf M_\mu\dot{\mathbf h}_z=-\mathbf G^T\mathbf S\,\mathbf d-\mathbf k,
$$

with diagonal node masses $M_\varepsilon[n]=\varepsilon_0\sum_C\varepsilon_C A_{n,C}$ (and $M_\mu$ likewise). Only the node mass is inverted, so the edge Hodge $\mathbf S$ only needs to be symmetric positive (semi)definite, not diagonal:
- `hodge="orthogonal"`: $S_{ee}=\sum_C w_C h_{e,C}/L_e$ (series average along the dual edge, the exact interface condition for the normal flux);
- `hodge="mfd"` (any convex polygons): the mimetic inner product $\mathbf M_C=\mathbf R\mathbf R^T/|C|+\gamma(\mathbf I-\mathbf N(\mathbf N^T\mathbf N)^{-1}\mathbf N^T)$ per cell, with $N_e=L_e\mathbf n_e^T$, $R_e=(\mathbf m_e-\mathbf x_C)^T$ and $\gamma=\operatorname{tr}(\mathbf R\mathbf R^T)/(|C|\,n_C)$. Since $\mathbf R^T\mathbf N=|C|\mathbf I$ (divergence theorem), $\mathbf M_C\mathbf N=\mathbf R$: the matrix is exact for uniform fields and SPD [Brezzi, Lipnikov & Shashkov, 2005]. The node masses then use the median dual (centroid and edge midpoints). On triangles $\mathbf G\mathbf u$ is uniform in each cell, so the stability term never acts and the stiffness equals the cotangent one; only the masses differ (test). On randomly distorted quadrilaterals (where the diagonal Hodge is rejected as non-orthogonal) cavity modes converge at second order. This is the same family as the geometric/cell-method constitutive matrices of the Codecasa–Specogna–Trevisan group (see `notes/lit/fit_meshing_sota.md`); it is not new.

The leapfrog is stable for $\Delta t<2/\sqrt{\lambda_\max(\mathbf M^{-1}\mathbf G^T\mathbf S\mathbf G)}$, with $\lambda_\max$ from Lanczos.

**Hybrid meshes** (`hybrid_circles_mesh`). Axis-aligned squares fill the bulk ($h_\text{out}$) and the inside of every circular scatterer ($h_\text{in}$). Squares are removed where they come within the patch distance of an interface, leaving a staircase-bounded hole. The hole is filled with a Delaunay triangle patch built from the square nodes on the staircase (fixed), an area-preserving ring on each interface and staggered graded rings. Ring points stay outside the circumcircles of the squares, which keeps every staircase edge a Delaunay edge (verified by an exact area check). A few Laplacian sweeps then smooth the free points. Optionally (`gap_cells`), narrow gaps are refined: $h\le\max(h_\min,w/n_\text{gap})$ with $w$ the local gap width. With `outer_radius`, the domain is a disk with a thin $h_\text{out}$ triangle annulus at the boundary, so that a first-order ABC can be used on the same domain as the triangle arms. This is the mesh-level analogue of the hybrid FDTD/FETD methods [Rylander & Bondeson, 2000]. Here the junction needs no special treatment: the same FIT operators apply on both sides and the dual is orthogonal across the shared edges. A related FE construction that reduces to FIT on rectangles is [Egger & Radu, arXiv:2209.10432].

**Voronoi-primal meshes** (`voronoi_disk_mesh`). The generators sit on staggered rings, which gives hexagonal cells, and on a hexagonal lattice in the core. Each interface circle is straddled by a mirrored pair of rings at $r_k\mp\delta$ with equal angles. The Voronoi edge of each pair lies on the circle's tangent, so the cells conform to the interface (tangent polygon, area-preserving radius $\rho=r\sqrt{\pi/(n\tan(\pi/n))}$). Mirrored seeding is the idea behind conforming Voronoi meshers such as VoroCrust [Abdelkader et al., 2020]. The outer boundary is a mirrored pair whose outer cells are discarded. Nearly cocircular generators produce very short Voronoi edges with a huge $h/L$, which set the global time step. Options:
- `lloyd`: Lloyd sweeps of the free generators;
- `collapse`: edges shorter than `collapse`$\cdot|p_i-p_j|$ are collapsed to a point. This is the $L\to0$ limit of their stiffness (equal node values). It moves the merged vertex by at most $L/2$ and leaves the neighbouring edges orthogonal only to $O$(`collapse`). The study uses `collapse=0.1`, `lloyd=10`; 6–17 % of the edges then have an orthogonality defect.

**Absorbing boundaries.** First-order Silver–Müller on boundary nodes. On a square region of squares there is also a split-field (Berenger) PML: the node field splits as $u=u_x+u_y$, where $u_x$ collects the contributions of the $x$-directed edges ($G_x^T$) and is damped by $s_x(x)$, and the flux of an $x$-directed edge is damped by the same $s_x$ (matched, $\sigma/\varepsilon=\sigma^*/\mu$). The profile is cubic, with $s_\max=(m+1)\,c\ln(1/R_0)/(2d)$. On a point-source pulse, a 12-cell layer reflects 0.4 % of the direct signal; without the magnetic (edge) loss it reflects 24 % (test).

**Plane-wave source** (scattered field, as `UnstructuredFIT2D.add_plane_wave`). TM uses the node current $(M_\varepsilon-\varepsilon_0\varepsilon_bA_n)\,\dot E_\text{inc}$. TE puts $-(1-\varepsilon_b/\varepsilon_C)\,\mathbf g_{e,C}\cdot\mathbf E_\text{inc}$ on each dual-edge segment of a scatterer cell. Here $\mathbf g_{e,C}$ is the segment vector, $c_C\to$ foot ($=h_{e,C}\mathbf n_e$, orthogonal Hodge) or centroid $\to$ midpoint (the vector that makes the mimetic matrix exact, $\mathbf M_C\mathbf N=\mathbf R$). **Dispersive and lossy scatterers** (new): the auxiliary equations must see the total field.
- In TM the nodes touching such cells are advanced in total-field form ($u_\text{tot}=u_s+E_\text{inc}$, with the background current $\varepsilon_0\varepsilon_bA_n\dot E_\text{inc}$).
- In TE each dispersive segment is driven by $D_n=d_e/L_e+\varepsilon_0\varepsilon_bE_\text{inc}\cdot\mathbf n_e$ and contributes $h_{e,C}(E_{e,C}-E_\text{inc}\cdot\mathbf n_e)$.

A Drude cylinder ($\varepsilon_\infty=4$, $\omega_p=1.37\times10^{16}$ s⁻¹, $a=30$ nm) reproduces the series $Q_\text{sca}$ to 1.4 % RMS (TE, through the localized surface plasmon resonance (LSPR) at 320 nm) and 0.8 % (TM). Dropping the $\varepsilon_b E_\text{inc}$ term gives > 10 % error (test).

**Interpolation.** Probes use mean-value coordinates [Floater, 2003], which reproduce linear fields on any polygon. Points within round-off of an edge use the 1D edge weights; this avoids a 0/0 that turned results into NaN on the cloud machines.

**Numerical dispersion at equal unknowns** (`studies/mie-cylinder-polyfit/dispersion.py`, Bloch analysis, derived). The table gives the worst-angle phase-velocity error at 3000 unknowns per $\lambda^2$:

| lattice (unknowns per area) | semi-discrete | leapfrog at 0.95 CFL | $c\Delta t/h$ |
|---|---|---|---|
| squares, $3/h^2$ | $1.6\times10^{-3}$ | $9.0\times10^{-4}$ | 0.672 |
| equilateral triangles, $4.62/h^2$ | $1.9\times10^{-3}$ | $3.8\times10^{-4}$ | 0.776 |
| regular hexagons (Voronoi primal), $1.92/h^2$ | $7.9\times10^{-4}$ | $3.2\times10^{-4}$ | 0.672 |

At equal unknowns the honeycomb (hexagonal primal) has half the semi-discrete error of squares or triangles. In the leapfrog, the temporal error partly cancels the spatial one on triangles and hexagons, but less so on squares.

**Mie benchmark** ($n=2$, $x=2$–10, results in `studies/mie-cylinder-polyfit/README.md`): at 1 % RMS Q_sca error, on the same domain (disk + ABC) as graded triangles, the hybrid needs 1.40×/1.34× fewer unknowns and 1.22×/1.13× less work (TM/TE). On the same square + PML domain as graded conformal Cartesian it needs 1.40×/1.25× fewer unknowns and 1.28×/1.06× less work. Its time per unknown-step is 0.73–0.84× that of the triangle solver on the same machine. Against the best triangle baseline (compact disk with radial CFS-PML) it is equal in TM work and 1.14× worse in TE, because the split PML forces a square domain. The Voronoi mesh needs 1.95× fewer unknowns and 1.87× less work than graded triangles in TM, but misses 1 % in TE (1.02 % at res 40). On the plasmonic cases (B2, B3; `studies/benchmark-polygon`) the hybrid matches graded triangles in TE and is cheapest to 0.1 % on the Au TM case. Its gap refinement reaches a 5 % gap-enhancement error on the 2 nm dimer with at least 2.6× fewer unknowns than the triangle baseline's first converged rung.

#### Limits & Failure Modes
- **Non-orthogonal meshes.** The diagonal Hodge is only consistent on orthogonal pairs, and the solver refuses meshes with defect > `ortho_tol`. The Voronoi arm deliberately overrides this (`ortho_tol=inf`) after edge collapse. The local $O$(`collapse`) inconsistency is the most likely cause of its slower convergence at fine resolution (see the study).
- **Mimetic Hodge.** The constant $\gamma$ is a choice. On rectangles the mimetic matrix is about 1.7× less accurate than the diagonal Hodge (cavity test), so prefer `orthogonal` whenever the mesh allows it. Dispersive TE needs the orthogonal Hodge. Non-convex cells are not supported, and the median-dual masses can be negative on them.
- **Global time step.** The hybrid inner squares have $h_\text{in}$, so $\Delta t$ equals that of the graded Cartesian grid. Gap refinement lowers $\Delta t$ for the whole mesh (local time stepping is a separate branch).
- **PML.** Split-field only, and only on axis-aligned edges (the square bulk). It is not a CFS-PML, so evanescent and grazing waves at low frequency are absorbed less well (a 10-cell layer reflects ≈ 2 % at $x=2$ in the study). Use `bc='pec'` behind it; the ABC loss is not applied on PML nodes.
- **Mesh generators** handle circles only (any number, including narrow gaps). The Voronoi generator handles concentric circles only. Edge collapse can degenerate a cell when `collapse` > 0.3 (it raises).
- **Plane wave:** PEC and magnetic scatterers are not supported. Conductivity is supported in TM only (TE: use a Drude pole).
- **Fast structured path:** not used. A NumPy stencil update on squares costs 3.3–4.5 ns per unknown-step against 4.9–6.3 ns for the sparse incidence update on the same grid (1.4–1.5×, three n2 machines; 1.5× on a laptop). It would apply only to the square part of a hybrid run, beside the source, DFT and PML costs, so the expected gain is ≈ 1.2×. That did not justify a second code path with interface bookkeeping (decision recorded).
- **Gap refinement:** `gap_cells` fixes the cells across a gap, independent of the bulk h. A convergence ladder must scale it, otherwise the gap error stalls (B3, 2 nm gap: 1.1 % then 2.9 %).

References:
- T. Rylander and A. Bondeson, "Stable FEM-FDTD hybrid method for Maxwell's equations", *Comput. Phys. Commun.* **125**, 75–82 (2000).
- H. Egger and B. Radu, "A Yee-like finite element scheme for Maxwell's equations on hybrid grids", arXiv:2209.10432 (2022).
- F. Brezzi, K. Lipnikov and M. Shashkov, "Convergence of mimetic finite difference methods for diffusion problems on polyhedral meshes", *SIAM J. Numer. Anal.* (2005).
- M. S. Floater, "Mean value coordinates", *Comput. Aided Geom. Des.* **20**(1), 19–27 (2003).
- A. Abdelkader et al., "VoroCrust: Voronoi meshing without clipping", *ACM Trans. Graph.* **39**(3) (2020), arXiv:1902.08767.
- J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves", *J. Comput. Phys.* **114**, 185–200 (1994).

### 🌀 Curvilinear FIT: body-fitted O-grids and multi-block meshes (time domain)

`wakis.curvilinear` is a 2D time-domain FIT solver (`CurvilinearFIT2D`, TM and TE) on structured, body-fitted quadrilateral meshes: O-grids around round objects, optionally joined to a Cartesian outer block. It is the nonorthogonal FIT (NFIT) of [Holland, 1983; Schuhmann & Weiland, 2000] in matrix form: the topological operators are those of every FIT grid, and only the material (Hodge) matrix sees the metric.

**Unknowns and grid equations.** Per unit length in $z$, as in `UnstructuredFIT2D`: TM has $E_z$ on the nodes and the magnetic flux $b_e$ through every primal edge (times $\hat z$); TE has $H_z$ on the nodes and the electric flux $d_e$:

$$
\text{TM: } \dot{\mathbf b} = -\mathbf G\mathbf e,\quad \mathbf M\dot{\mathbf e} = \mathbf G^T\mathbf W\mathbf b;\qquad
\text{TE: } \dot{\mathbf d} = \mathbf G\mathbf h,\quad \mathbf M\dot{\mathbf h} = -\mathbf G^T\mathbf W\mathbf d .
$$

$\mathbf G$ is the integer edge–node incidence. $\mathbf M$ is diagonal (lumped dual areas times $\varepsilon$ or $\mu$). $\mathbf W$ maps fluxes to dual-edge voltages. Because $\mathbf W$ is assembled in the flux→voltage direction, the explicit leapfrog never inverts a non-diagonal matrix: the only "inverse" in the scheme is the diagonal $\mathbf M^{-1}$. This is the observation of [Schuhmann & Weiland, 2000] that the NFIT interpolation formulas *are* the inverse material matrices.

**The Hodge as a sum of field components.** Each cell contributes components $q$ with a reconstruction row $\mathbf r_q$ (one Cartesian or normal component of $\mathbf B$ or $\mathbf D$ from the fluxes of the cell's sides), a weight $w_q$ (area) and a material $p_q$ ($\mu$ for TM, $\varepsilon$ for TE):

$$
\mathbf W=\sum_q \frac{w_q}{p_q}\,\mathbf r_q^T\mathbf r_q,\qquad
\tfrac12\mathbf b^T\mathbf W\mathbf b=\sum_q \frac{w_q}{2p_q}(\mathbf r_q\mathbf b)^2 \approx \int\frac{|\mathbf B|^2}{2\mu}\,dA .
$$

$\mathbf W$ is therefore symmetric positive semi-definite *by construction*, which is the stability property [Schuhmann & Weiland, 2000] attribute to the "symmetric evaluation" of the metric. Two cell types:

- *General quadrilateral* (straight sides; core of the O-grid, circle-to-square transition block). At each of the 4 corners the in-plane vector is reconstructed from the fluxes of the two sides meeting there: with length-weighted side normals $\mathbf N_1,\mathbf N_2$, $\mathbf B_c = [\mathbf N_1\ \mathbf N_2]^{-T}(b_1, b_2)$, exact for a uniform field; weight $A/4$. $\mathbf W$ then couples the two edges at each corner: about 7 non-zeros per row (banded on a structured block). On rectangles the coupling vanishes and $\mathbf W$ is the diagonal standard-FIT matrix (tested).
- *Polar sector* $r_0<r<r_1$, $\theta_0<\theta<\theta_1$ (exact arcs). The cell is orthogonal, so each side contributes one normal component $b_e/L_e$ on its half dual edge: diagonal, i.e. standard FIT in cylindrical coordinates with exact arc lengths and sector areas. Node areas are the exact quarter-sector shares.

The flux through an edge depends only on its end points ($b_e = A_z(j)-A_z(i)$ in TM; likewise $d_e$ from $H_z$ in TE), so a chord in one cell and an arc in the neighbouring cell describe the same unknown. Polar and general cells can therefore be mixed freely. Material interfaces are rings of the mesh: a cell holds one material, normal $B$ ($D$) is continuous across the shared edge, and $1/p$ is averaged in series along the dual edge, which is the correct interface condition for TE where $\mathbf E$ crosses the interface (no subpixel averaging needed).

**Time stepping and stability.** Standard leapfrog. With a symmetric $\mathbf W$ the quantity $\tfrac12\mathbf b^{n+1/2}\!\cdot\!\mathbf W\mathbf b^{n+1/2}+\tfrac12\mathbf e^{n+1}\!\cdot\!\mathbf M\mathbf e^{n}$ (TM; TE analogous) is conserved exactly; it is positive for $\Delta t<2/\sqrt{\lambda_\max}$ with $\lambda_\max$ of $\mathbf M^{-1}\mathbf G^T\mathbf W\mathbf G$ (Lanczos). Measured: relative drift $7\times10^{-16}$ over 1500 steps; with an asymmetric $\mathbf W$ (mutation) the energy drifts or the run blows up.

**Mesh generation** (`ogrid_mesh`, `ogrid_cartesian_mesh`, `radial_nodes`). The O-grid has a "butterfly" core inside $r_t$: a square of $n_\theta/4\times n_\theta/4$ cells and four transition blocks (linear transfinite interpolation) to the circle $r_t$, then polar rings whose radii are graded and contain every interface, extraction circle and PML boundary exactly. The four singular nodes of the core (valence 3) have 135° corners. Laplacian smoothing moves them towards 120° but shrinks the nearby cells and *lowers* the time step (0.72 → 0.39 $c\Delta t/h$ at 300 sweeps), so it is off by default. The number of transition layers is chosen so that the layer thickness on the square's diagonal is about one core cell. For the Mie cylinder the core sits inside the dielectric ($r_t=0.9a$), where the wave is slower, so it does not limit $\Delta t$: $c\Delta t/h_\text{in}=0.68$–$0.71$ against the 2D Yee limit $0.95/\sqrt2=0.67$ of the Cartesian grid at the same $h_\text{in}$. In a vacuum core the core cells limit $\Delta t$ to about $0.4\,h/c$.

The multi-block mesh adds a transition block from the outer polar circle to a square (straight-sided general cells) and a Cartesian frame of rectangles outside it, conforming node by node at the square.

**Radial PML.** Complex coordinate stretching [Chew & Weedon, 1994] in polar coordinates, $\tilde r=r+\frac1{i\omega}\int_{r_0}^r\sigma\,dr'$, gives $s_r = 1+\sigma/(i\omega)$ along $r$ and $s_\varphi=\tilde r/r = 1+\bar\sigma/(i\omega)$, $\bar\sigma=\frac1r\int_{r_0}^r\sigma$, along $\varphi$ [Teixeira & Chew, 1997]. In FIT language stretching only rescales lengths, so on the diagonal (polar) cells of the layer a node mass gets the factor $s_r s_\varphi$ and an edge Hodge entry $s_\perp/s_\parallel$ ($s_\parallel$ along the edge). With $e^{+i\omega t}$ this gives local auxiliary updates: per PML edge $\dot h+\sigma_\parallel h = c_e(\dot b+\sigma_\perp b)$ (Crank–Nicolson), per PML node $M(\dot e + (\sigma_r+\bar\sigma)e + \sigma_r\bar\sigma\,\psi)=(\mathbf G^T\mathbf h)$ with $\dot\psi=e$. The Cartesian PML of the multi-block mesh is the same with $(s_x,s_y)$. Profile $\sigma=\sigma_\max\xi^3$, $\sigma_\max = 4c\ln(10^6)/(2d)$. PML edges must have diagonal Hodge rows (checked). Reflection of a point-source pulse against a large-domain reference: $2\times10^{-4}$ (10 cells of $\lambda/20$), $1.2\times10^{-5}$ (20 cells), $5\times10^{-5}$ (12 cells of $\lambda/12$, test); without the $s_\varphi$ factor (a Cartesian-style stretch on a curved layer) $6$–$7\times10^{-3}$.

**Plane-wave source** (scattered field). The incident field enters only through the constitutive relation of non-background cells, $D_s = D-\varepsilon_0\varepsilon_bE_\text{inc}$. TM: the node update runs on the total field $e_s+E_\text{inc}$ at nodes touching the scatterer, with the source term $\varepsilon_0\varepsilon_bA_n\dot E_\text{inc}$. TE: every Hodge component $q$ in the scatterer computes $E_{\text{tot},q}$ from $D_{s,q}+\varepsilon_0\varepsilon_b\,\mathbf E_\text{inc}\cdot\hat{\mathbf n}_q$ at its quadrature point and contributes $E_{\text{tot},q}-\mathbf E_\text{inc}\cdot\hat{\mathbf n}_q$ to the voltages. Both forms hold unchanged for dispersive scatterers: the ADE simply acts on the total field.

**Dispersive media.** Drude/Lorentz/Debye poles with the ADE coefficients of `wakis.dispersive`. TM: the node update `CapacitiveUpdate` with the exact dispersive fraction of each node area (as in `UnstructuredFIT2D`). TE: each Hodge component of a dispersive cell carries its own $E_q$ and polarization, $E_q^{n+1}=(D_q/\varepsilon_0-\sum p^{n+1})/\varepsilon_\infty$, and $\mathbf W$ for the stability estimate uses $\varepsilon_\infty$. Lorentz cylinder ($\varepsilon=4.05$–$6.73$, lossy, $x=2$–$10$): Q_sca RMS error 1.9 % / 0.9 % (TM, res 14/20) and 2.4 % / 1.1 % (TE) against the series with the same $\varepsilon(\omega)$.

**Validation and cost** (`tests/test_046_curvilinear_nfit.py`, study `studies/mie-cylinder-nfit/`, one Batch run per n2-standard-2):

| Case | Curvilinear FIT (O-grid) | Best baseline |
| --- | --- | --- |
| Rod ($\varepsilon_r=6$, $b=a/2$) in a PEC cavity, first 4 modes, TM/TE | RMS error $7.4\times10^{-4}$ / $9.1\times10^{-4}$ at $n_\theta=128$, $1.3\times10^{-4}$ / $1.5\times10^{-4}$ at 256 (orders 1.1 → 2.5, pre-asymptotic) | |
| Empty square cavity on the multi-block mesh (O-grid + transition + Cartesian), TM | first 6 modes within 1 % at 708 cells | analytic |
| Energy, 1500 steps, symmetric $\mathbf W$ / asymmetric mutant | drift $7\times10^{-16}$ / $>10^{-4}$ or blow-up | |
| Radial PML, 10 cells of $\lambda/20$; 20 cells; no $s_\varphi$ | $2\times10^{-4}$; $1.2\times10^{-5}$; $6.3\times10^{-3}$ | |
| B1 Mie cylinder $n=2$, x = 2–10, 1 % RMS Q_sca (3000 fs window), TM | 158k unknowns, $1.7\times10^{10}$ work, 319 s | graded triangles + PML: 244k, $3.3\times10^{10}$, 561 s; Cartesian graded conformal: 366k, $5.1\times10^{10}$, 1406 s |
| same, TE | 176k, $2.0\times10^{10}$, 297 s | triangles: 215k, $2.7\times10^{10}$, 470 s; Cartesian: 307k, $3.9\times10^{10}$, 1138 s |
| B2 Au nanocylinder TE (LSPR), 0.1 % Q_ext NRMS | 19.6k unknowns, $1.05\times10^{9}$ work, 31 s (radial $h/2$, outer $\le 2h$; tuned on B2) | uniform triangles: 88k, $2.6\times10^{9}$, 64–72 s |
| B3 Au dimer, 5 % gap enhancement (9-point stencil) | 14k / 23k / 56k unknowns (g = 10/5/2 nm) | graded triangles: below 5 % from 6k / 7k / 35k |
| Coated cylinder (20 nm, $n=3.5$ shell), 1 % RMS | TM 165k, TE 248k | triangles: TM 191k, TE 221k; Cartesian sub-cell: stalls at 1.1 % / 3.4 % |

A structured body-fitted mesh wins where the geometry is radial: about 1.2–1.6× fewer unknowns and 1.3–1.9× less work than graded triangles on the dielectric cylinder, and 4.5× fewer unknowns with 2.5× less work than uniform triangles on the plasmonic cylinder. It loses where a feature needs local refinement in the angular direction (dimer gap: 3–8× worse than graded triangles), because refinement in a structured block runs along whole grid lines.

![Curvilinear meshes](../studies/mie-cylinder-nfit/figures/a_meshes.png)
![B1 cost to accuracy](../studies/mie-cylinder-nfit/figures/g_error_vs_cost_3000fs.png)

**3D extension (not implemented).** Everything above carries over to curvilinear hexahedra: the topological curl and divergence of the structured block are unchanged; the edge and face Hodges are sums over the 8 corners of each hexahedron of $\mathbf r_q^T\mathbf r_q\,w_q/p_q$ with the field vector reconstructed from the 3 edges (or 3 faces) meeting at the corner (Schuhmann–Weiland's banded NFIT matrices, ~19 non-zeros per row); a hexahedron with orthogonal corners reduces to standard FIT. Directly relevant to Wakis's users: body-fitted O-grids of round beam pipes, tapers and elliptical cavity cells (accelerator cavities), where the staircase or cut-cell wall is the dominant wake-impedance error; extruding the 2D O-grid along $z$ (prisms of polar sectors and butterfly cores) gives a beam-pipe mesh whose $z$ lines stay orthogonal, so only the transverse Hodge is non-diagonal and a moving-window/wake solver needs no change in $z$. Open questions: the butterfly core on the beam axis (the beam passes through it; its 135° cells cost a factor ~2 in $\Delta t$ in vacuum unless the core is enlarged or the axis region uses smaller cells with local time stepping), conforming joins to the Cartesian Wakis grid (the multi-block construction above), and GPU kernels for the banded Hodge.

**Limits & failure modes.**
- *Geometry class.* An O-grid is natural for round (or smoothly star-shaped) bodies, coatings and concentric layers; clusters of round bodies need one box per body (B3 dimers: boxes must tile a rectangle, implemented for equal cylinders on a line). Corners, re-entrant shapes and arbitrary CAD need an unstructured mesher (triangles) or block decomposition by hand; there is no automatic multi-block mesher.
- *Time step.* Cells shrink towards the core: a vacuum core limits $c\Delta t/h$ to about 0.4; Laplacian smoothing of the core makes it worse. Thin radial layers (a 20 nm shell, a 0.5 nm gap layer) reduce $\Delta t$ in proportion to their optical thickness. The azimuthal spacing grows with $r$ (fixed $n_\theta$), so a large outer O-grid is under-resolved azimuthally ($2.5\,h_\text{in}$ at $r = 2.5a$ in the baseline domain) — use a compact domain or the multi-block Cartesian outer block.
- *Area bookkeeping.* A ring where exact arcs (polar cells) meet straight chords (core or transition cells) double-counts or misses the circular segments, $O((h/r)^2)$ relative (0.13 % of the cylinder area at $h=a/10$ in the dimer mesh). Second order is kept, but the constant can be lowered by giving those cells curved sides.
- *Corner Hodge on skewed cells* is consistent (exact for uniform fields) but its error constant grows with skew; the rod-in-cavity convergence is pre-asymptotic (orders 1.1, 1.5, 2.5 from $n_\theta=32$ to 256).
- *PML* only on diagonal-Hodge cells (polar rings or rectangles); the radial PML needs the $s_\varphi$ factor (without it the reflection is 100× larger). Evanescent fields are not absorbed (standard PML limit).
- *Not implemented:* PEC objects with the plane-wave source, magnetic media with the plane wave, TE conductivity (use a Drude pole), local time stepping, GPU, adjoint (the update is linear with fixed sparse operators, so the transposed leapfrog applies as for `UnstructuredFIT2D`).

References:
- R. Holland, "Finite-difference solution of Maxwell's equations in generalized nonorthogonal coordinates", *IEEE Trans. Nucl. Sci.* **30**(6), 4589–4591 (1983).
- R. Schuhmann and T. Weiland, "The nonorthogonal finite integration technique applied to 2D- and 3D-eigenvalue problems", *IEEE Trans. Magn.* **36**(4), 897–901 (2000).
- S. J. Cooke, R. Shtokhamer, A. A. Mondelli and B. Levush, "A finite integration method for conformal, structured-grid, electromagnetic simulation", *J. Comput. Phys.* **215**, 321–347 (2006).
- W. C. Chew and W. H. Weedon, "A 3D perfectly matched medium from modified Maxwell's equations with stretched coordinates", *Microw. Opt. Technol. Lett.* **7**(13), 599–604 (1994).
- F. L. Teixeira and W. C. Chew, "Systematic derivation of anisotropic PML absorbing media in cylindrical and spherical coordinates", *IEEE Microw. Guided Wave Lett.* **7**(11), 371–373 (1997).

### 🔲 Local Cartesian refinement (subgridding): `wakis.subgrid`

A rectilinear grid that must resolve a small feature (a thin fin, a nanogap, a metal surface) carries the fine spacing along every row and column that crosses it. In the graded Mie study, 78 % of the cells of the graded grid were fine along at least one axis. Subgridding refines only the cells near the feature: a coarse Cartesian grid holds nested patches refined by an integer ratio $r$ (2 or 3), recursively (quadtree, `max_level` levels). `wakis.subgrid` is a 2D prototype (TM and TE). The 3D plan is at the end of this section.

**Mesh** (`QuadMesh`). Every leaf cell is an axis-aligned rectangle. Its corners lie on the lattice that splits each coarse interval into $r^{L}$ parts, so the topology is exact integer arithmetic, and the coarse grid may itself be graded. Where a coarse leaf borders finer leaves, the fine corners inside a coarse side are *hanging nodes*. They are not unknowns. Each takes the linear interpolant of the two end nodes of that side, which is the trace of the coarse cell's bilinear field. This keeps the scalar field continuous. The constraints form the prolongation $\mathbf P$ (all nodes × free nodes, rows sum to 1, $\mathbf u_\text{all} = \mathbf P\mathbf u$). Edges are the sides of the leaf cells. Along an interface, the coarse side and the $r$ collinear fine sides are *different* edges, each belonging to its own cell. Balance (no side touches leaves two levels finer) is enforced by refining offending cells.

**Grid equations.** Both polarizations use one scalar node field $u$ and one edge flux $f$ per unit length. In TM, $u=E_z$ and $f_e = L_e\mu(\mathbf H\times\hat{\mathbf t})_z$; the node mass comes from $\varepsilon$ and the edge Hodge from $\mu$. In TE, $u=H_z$ and $f_e = L_e\varepsilon(\hat{\mathbf t}\times\mathbf E)_z$; the node mass comes from $\mu$ and the edge Hodge from $\varepsilon$. With $\mathbf G$ the edge–node incidence of the composite mesh and $\mathbf K = \mathbf G\mathbf P$:

$$
\mathbf f^{n+1/2} = \mathbf f^{n-1/2} - \Delta t\,\mathbf K\mathbf u^n,\qquad
\mathbf D(\mathbf u^{n+1}-\mathbf u^n) = \Delta t\,\mathbf K^T\mathbf W\mathbf f^{n+1/2},
$$

with $\mathbf D = \mathbf P^T\mathbf m$ and $\mathbf W$ diagonal. Here $\mathbf m$ holds the dual-cell masses of all nodes (a quarter of each adjacent cell), and $\mathbf D$ is the row-sum lumping of $\mathbf P^T\,\mathrm{diag}(\mathbf m)\,\mathbf P$, which uses $\mathbf P\mathbf 1 = \mathbf 1$. The edge Hodge is $W_e = L^*_e/(\mu L_e)$, with $L^*_e$ the sum of the half widths of the cells that own the edge. On a uniform grid this is exactly Yee/FIT. On the composite mesh it is the conforming bilinear ($Q_1$) finite-element scheme with trapezoidal lumping of mass and stiffness and hanging-node constraints. The coarse-to-fine map $\mathbf P$ and the fine-to-coarse map $\mathbf P^T$ are therefore exact transposes, a Galerkin interface operator in the spirit of [Venkatarayalu et al., 2007] and [Chilton & Lee, 2007]. Three properties follow:

- **Stability.** $\mathbf D^{-1}\mathbf K^T\mathbf W\mathbf K$ is self-adjoint and positive semi-definite in the $\mathbf D$ inner product. The leapfrog conserves $\mathcal E = \tfrac12\mathbf f^{n+1/2}\cdot\mathbf W\mathbf f^{n+1/2}+\tfrac12\mathbf u^n\cdot\mathbf D\mathbf u^{n+1}$ exactly. It is stable iff $\Delta t < 2/\sqrt{\lambda_\max}$, and $\lambda_\max$ is computed at setup by Lanczos. Over $10^5$ steps on a 3-level mesh with dielectrics crossing both interfaces, $\mathcal E$ stays constant to $2\times10^{-15}$. Two mutations break it. A non-transposed restriction, where hanging nodes inject into the coarse update, grows by $10^{12}$ within 4 000 steps even at half the CFL limit. A time step 2 % above the computed limit also diverges.
- **Second order.** Cavity eigenfrequencies with an off-centre patch converge at order 2.0, and their errors lie between those of the coarse and the fine grid. The interface reflection falls as $N^{-2}$ ($N$ = coarse cells per wavelength) at every angle, for both 2:1 and 3:1. At 2:1 it is 0.47 % at normal incidence, 0.19 % at 30° and 0.69 % at 60° for $N = 20$, and 5 % at 75°. The mutation in which a hanging node copies one neighbour instead of interpolating is first order: the reflection falls only as $N^{-1}$, and the eigenvalue convergence order drops below 1.
- **Exact adjoint.** Every update is linear with fixed sparse matrices. `linear_step_T` is the exact transpose of a step and passes the dot-product test to $10^{-13}$; a transpose that omits $\mathbf P$ fails it.

The reflection is measured in the frequency domain. The leapfrog steady state satisfies $(\mathbf K^H\mathbf W\mathbf K - \tilde\omega^2\mathbf D)\mathbf u = \mathbf b$ exactly, with $\tilde\omega = 2\sin(\omega\Delta t/2)/\Delta t$. A strip two coarse cells high with Bloch phase $e^{-ik_y L_y}$ represents an infinite straight interface (`wakis.subgrid.interface.bloch_reflection`). The fit uses the exact discrete wavenumber of the coarse grid, so the result contains no coarse-grid dispersion, and the floor of the measurement without an interface is below $10^{-5}$.

**Materials (sub-pixel).** Cells cut by an interface are sampled with $n_\text{sub}\times n_\text{sub}$ points per quarter cell and per half strip. TM node masses average $\varepsilon$ arithmetically, because $E_z$ is tangential to every interface. For the TE edge Hodge, $D_n$ is continuous across a strip and $E_t$ along it. Each column $k$ across the edge therefore gives $R_k=\int dn/\varepsilon$ (series), and the columns combine in parallel: $W_e = 1/(\varepsilon_0 L_e\,\overline{R_k^{-1}})$. The background is a lossless dielectric $\varepsilon_b$ (vacuum by default).

**Sources, PML, dispersion.** The plane wave enters in scattered-field form, so no TF/SF box is needed.
- TM: $-\mathbf P^T[(\mathbf m-\mathbf m_b)\circ(\mathbf u_\text{inc}^{n+1}-\mathbf u_\text{inc}^n)]$, the discrete time derivative.
- TE: the constitutive relation $E_s = D_s/\varepsilon - (1-\varepsilon_b/\varepsilon)E_\text{inc}$ integrated over each column, $g = W f - c_e E_{\text{inc},n}$.

A Bérenger split-field PML (cubic grading) fills the coarse frame. Drude and Lorentz poles use the ADEs of `wakis.dispersive`, driven by the total field. In TM they sit on the nodes, mixed by the area fraction of the dual cell (exact, since $E_z$ is tangential). In TE they sit on the edges, with the series-per-column rule given in the limits below. A Lorentz cylinder reproduces the dispersive series to 9.5 % RMS at res 10 and 5 % at res 14 (`test_052`); without the pole dynamics the error is more than 3× larger.

**Local time stepping.** The interface operator is symmetric, so the stabilized leapfrog–Chebyshev LTS of `wakis.unstructured.lts` (sibling branch `feature/unstructured-lts`; [Diaz & Grote, 2009], with damping) applies without change: `SubgridFIT2D(..., lts=p)` feeds the flux update with $r(\mathbf X)\mathbf u$. With $p=2$ on the Mie benchmark, only the fine *vacuum* band takes two sub-steps (4–11 % of the nodes). The dielectric (where $v=c/2$) and the coarse grid take the coarse step. The step count halves, and the stability ratio is 0.975–0.98 (computed by Lanczos at setup). The dielectric then runs at about twice its previous Courant number, and the leapfrog's temporal dispersion cancels more of the spatial dispersion. This *also* halves the Mie error. The effect is generic: the triangle LTS study sees the same, and global-$\Delta t$ runs at Courant 0.5, 0.7 and 0.95 give 2.36 %, 2.31 % and 2.20 %. It is not specific to subgridding.

**Benchmark (B1, Mie cylinder n = 2, x = 2–10).** Global $\Delta t$, a 2:1 patch on the coarse cells within $a+2h_\text{in}$, and $h_\text{out}=2h_\text{in}$: the same fine band and coarse size as the graded arms. To reach 1 % RMS error in TM (TE), the compact subgrid needs:
- 175k (164k) unknowns and $7.2\times10^9$ ($6.5\times10^9$) unknown-steps;
- against graded triangles with the radial PML, 240k (245k) unknowns and $1.07\times10^{10}$ ($1.11\times10^{10}$): 1.4–1.5× fewer unknowns and 1.5–1.7× less work;
- against the compact graded rectilinear grid, 362k (347k) unknowns and $1.67\times10^{10}$ ($1.56\times10^{10}$): 2.1× fewer unknowns and 2.3–2.4× less work.

With LTS ($p=2$) the subgrid reaches 1 % with 88k (105k) unknowns and $1.4\times10^9$ ($1.8\times10^9$) unknown-steps, in 16 s (22 s) on one core. Tables, wall times and the LTS caveat are in `studies/mie-cylinder-subgrid`.

**Plasmonic and gap benchmarks** (`studies/subgrid-benchmarks`, shared store arms `subgrid-2to1*`). Against graded conformal Cartesian, the subgrid needs:
- 1.8–2× fewer unknowns for 1 % Q_ext error on the Au/Ag nanocylinder in TE (B2);
- 1.5–3.8× fewer for a 10 % gap-enhancement error on the Au dimer (B3);
- 16–52× fewer on a Si nanogap dimer with a 2–5 nm gap (S1), with LTS 50–200× less time.

Against graded triangles it is on par on S1 at 5 % but 4–5× more expensive on B2/B3 TE. There E crosses curved metal surfaces, and the diagonal sub-pixel Hodge converges at first order in h.

**Limits & failure modes.**
- **Spurious interface reflection.** It is second order but not zero: 2 % at 10 coarse cells per wavelength at normal incidence, rising toward grazing incidence (14 % at 75° for $N=10$). Keep strong fields and resonant features at least a few fine cells away from level jumps, and keep at least about 10 coarse cells per local wavelength.
- **Ratio.** Ratios above 3 were not tested; reflection grows mildly from 2:1 to 3:1 (+20 %). Multi-level meshes are balanced automatically; unbalanced meshes are resolved recursively but are not validated.
- **Time step.** With global $\Delta t$ the smallest cell sets the step everywhere. Without LTS, subgridding reduces unknowns but not steps.
- **LTS with dispersive media.** The ADE runs at the coarse step. With the Rakić Au/Ag models, whose Lorentz poles reach $\omega_0 \approx 2\times10^{16}$ rad/s, the coarse step of $p=4$ fails the ADE stability check or, in some runs, diverges, even though the Lanczos check of the lossless operator passes. LTS is therefore not usable for the plasmonic benchmarks until the ADE of the fine set is sub-stepped as well.
- **Local time stepping.** The LTS is two-level (one ratio $p$). A deep multi-level patch puts all levels below the coarse CFL into $F$. Its stability proof covers the lossless, source-free system: the PML must stay outside $F$ and its buffer (checked), and with ADE media inside $F$ stability is only tested, not proven.
- **Materials.** The TE Hodge is diagonal: the series/parallel strip rule is exact for interfaces aligned with the grid and first order for oblique ones, as with the diagonal conformal averaging of `SolverFIT3D`. For dispersive TE media every column of a cut edge carries its own metal-segment field and polarization, in series with the host: $E_m = ((g + b\,p/\varepsilon_\infty)/R - p)/\varepsilon_\infty$ and $g = Wf - W\varepsilon_0\sum_k w_k b_k p_k/(\varepsilon_\infty R_k) - c_e E_\text{inc}$. This is exact for a flat metal surface aligned with the grid and passive by construction. A setup guard checks $b_k/\varepsilon_\infty \le R_k$, because inconsistent column data made the semi-discrete system active during development. At curved metal surfaces the error is still first order. On B2-Au-TE the error falls from 5.1 % to 1.6 % as h goes from 4 nm to 1 nm, about as accurate per h as the conformal Cartesian arm, while body-fitted triangles reach 0.2 % at 1 nm.
- **PML.** The split-field PML is not covered by the energy proof. It runs only on the coarse frame and is not included in `linear_step_T`, so its adjoint, and those of the ADE and the sources, remain to be written.
- **Performance.** This is a SciPy-sparse prototype: about 15–25 ns per unknown-step on one core, including the per-step DFT on the extraction circles. There is no GPU or MPI path.

**Extension to 3D (`SolverFIT3D`, plan).**
- **Mesh.** `GridFIT3D` keeps its non-uniform coarse tensor grid and gains an octree of refined boxes, nested and balanced, of ratio 2 or 3.
- **Unknowns.** In 3D, $\mathbf e$ lives on edges. Hanging *edges* on a coarse face take the Nédélec trace of the coarse face: fine edge voltages are fixed fractions of the parallel coarse edge, and of the coarse face's neighbouring edges for transverse fine edges inside the face. This yields $\mathbf P_e$ (all fine and coarse edges × free edges). The curl becomes $\mathbf C\mathbf P_e$, Ampère uses $\mathbf P_e^T$, and the masses are row-sum lumped, $\mathbf M_\varepsilon = \mathbf P_e^T\mathbf m_\varepsilon$.
- **Properties.** The same proof gives energy conservation, CFL stability and an exact transpose, as in the hexahedral hanging-variable FETD of Venkatarayalu et al. $\mathbf D\mathbf b = 0$ holds for the fine faces because $\mathbf b$ is updated from $\mathbf C\mathbf P_e\mathbf e$.
- **Data layout.** Keep each patch as a dense Cartesian block (one Yee update kernel per block) and apply $\mathbf P_e$ and $\mathbf P_e^T$ as thin sparse coupling operators on the block surfaces only. This keeps the GPU kernels and the existing conformal/tensor material code per block.
- **MPI.** Assign whole blocks to ranks (the coarse slab decomposition plus patch ownership by centre). Only the surface couplings and the coarse halos are exchanged, and patches that straddle a slab boundary are split on coarse-cell lines.
- **LTS.** Two-level Chebyshev LTS per patch level (substeps $p=r^\ell$) needs one extra product of the fine block operator per substep, which also maps onto block kernels.
- **Cost.** The leading cost is the coarse-fine surface (about $6 r^2$ edges per coarse face).
- **Validation.** Repeat the cavity energy, dot-product and Bloch-reflection tests of this section in 3D before any physics runs.

References:
- P. Thoma and T. Weiland, "A consistent subgridding scheme for the finite difference time domain method", *Int. J. Numer. Model.* **9**(5), 359–374 (1996).
- W. Podebrad, M. Clemens and T. Weiland, "New flexible subgridding scheme for the finite integration technique", *IEEE Trans. Magn.* **39**(3), 1662–1665 (2003).
- J.-P. Bérenger, "A Huygens subgridding for the FDTD method", *IEEE Trans. Antennas Propag.* **54**(12), 3797–3804 (2006).
- R. A. Chilton and R. Lee, "Conservative and provably stable FDTD subgridding", *IEEE Trans. Antennas Propag.* **55**(9), 2537–2549 (2007).
- N. V. Venkatarayalu, R. Lee, Y.-B. Gan and L.-W. Li, "A stable FDTD subgridding method based on finite element formulation with hanging variable", *IEEE Trans. Antennas Propag.* **55**(3), 907–915 (2007).
- F. Collino, T. Fouquet and P. Joly, "Conservative space-time mesh refinement methods for the FDTD solution of Maxwell's equations", *J. Comput. Phys.* **211**(1), 9–35 (2006).
- F. Bekmambetova, X. Zhang and P. Triverio, "A dissipative systems theory for FDTD with application to stability analysis and subgridding", *IEEE Trans. Antennas Propag.* **65**(2), 751–762 (2017); 3D: **66**(12), 7156–7170 (2018).
- Y. Cheng et al., "A stable FDTD subgridding scheme with SBP-SAT for transient TM analysis", *J. Comput. Phys.* **494**, 112510 (2023).
- J. Diaz and M. J. Grote, "Energy conserving explicit local time stepping for second-order wave equations", *SIAM J. Sci. Comput.* **31**(3), 1985–2014 (2009).

### 🎓 High-order time-domain schemes: nodal DGTD on curved triangles and the Cartesian (2,4) scheme

**Why.** Yee/FIT is second order in space and time. Its phase velocity error is $\delta v/v \approx -(kh)^2/24$ per axis (the leapfrog adds $+(\omega\Delta t)^2/24$). For a resonator, the phase error accumulates over $Q$ round trips and shows up as a shifted resonance. In the Mie studies, for $x = ka > 6$ the error of both conformal Cartesian and triangular FIT is dominated by these bulk shifts, not by the interface. Halving the error costs $2^{d/2}\cdot 2^{1/2}$ more work in $d = 2$ dimensions (unknowns × steps $\propto h^{-3}$). A scheme of order $q$ needs $h \propto \varepsilon^{1/q}$, so its cost to reach a small target $\varepsilon$ falls like $\varepsilon^{-3/q}$ instead of $\varepsilon^{-3/2}$. `wakis.highorder` holds two 2D prototypes that test this. Both are separate solver cores and do not change FIT.

**Nodal DGTD (`DGTD2D`).** Both polarizations are one system for $(u, v_x, v_y)$ with coefficients $(a, b)$:

$$
a\,\partial_t u = \partial_x v_y - \partial_y v_x,\qquad
b\,\partial_t v_x = -\partial_y u,\qquad
b\,\partial_t v_y = \partial_x u ,
$$

with TM: $u = E_z$, $\mathbf v = \mathbf H$, $a = \varepsilon$, $b = \mu$, and TE: $u = H_z$, $\mathbf v = -\mathbf E$, $a = \mu$, $b = \varepsilon$ (duality).

- **Discretization.** On every triangle, the fields are polynomials of degree $p$ in a nodal (Lagrange) basis on Warburton's warp-and-blend nodes [Warburton, 2006]. The derivatives are exact on the polynomial space. Elements couple only through a numerical flux on their faces. This is the strong form of [Hesthaven & Warburton, 2002; 2008]:

$$
\mathbf M_J \frac{d q}{dt} = \mathbf M_J\,\nabla\!\cdot\!F(q) + \oint \ell\,\mathbf n\cdot(F^* - F^-)\,ds .
$$

- **Upwind flux.** Projected on the face normal, the system is the 1D pair $a\,\partial_t u = \partial_\xi H_\tau$, $b\,\partial_t H_\tau = \partial_\xi u$, with $H_\tau = n_x v_y - n_y v_x$ the tangential component. Its characteristics are $u \mp Z H_\tau$ with $Z = \sqrt{b/a}$. Solving the Riemann problem between the media $(-)$ and $(+)$ (continuity of $u^*$ and $H_\tau^*$) gives

$$
H_\tau^* - H_\tau^- = \frac{Z^+\,[H_\tau] + \alpha\,[u]}{Z^+ + Z^-},\qquad
u^* - u^- = \frac{Y^+\,[u] + \alpha\,[H_\tau]}{Y^+ + Y^-},
$$

  with $[f] = f^+ - f^-$ and $Y = 1/Z$. $\alpha = 1$ is the exact upwind flux for piecewise-constant media; it dissipates only unresolved modes. $\alpha = 0$ is the impedance-weighted central flux, which conserves energy. The flux couples only *tangential* fields, which are continuous across every interface. The normal-field jump of TE at a dielectric or metal interface is therefore free inside the discretization, with no averaging.
- **Curved elements.** Faces that are chords of a mesh circle are curved isoparametrically. The element nodes move by the Szabó–Babuška blend $\mathbf d = 4\lambda_i\lambda_j\,\boldsymbol\delta(\xi)/(1-\xi^2)$, with $\xi = \lambda_j - \lambda_i$ and $\boldsymbol\delta$ the arc-minus-chord displacement of the edge. The blend is smooth, so the polynomial interpolant of the map converges at order $p+1$. The metric terms come from the interpolated map at the nodes. The $J$-weighted mass matrix and the face mass matrices with the varying line Jacobian come from Gauss cubature, so every curved element has its own lift matrix.
  - The first version blended radially from the opposite vertex: $\mathbf d = (\lambda_i+\lambda_j)\,\Delta(t)$, $t = \lambda_j/(\lambda_i+\lambda_j)$. That map is only Lipschitz at the opposite vertex, and it cut the circular-cavity convergence of $p = 4$ from about 5 to about 2. `test_curved_elements_restore_high_order` guards this.
- **Time stepping.** The 5-stage, 4th-order, 2N-storage Runge–Kutta of [Carpenter & Kennedy, 1994]. $\Delta t$ is set from the spectral radius of the semi-discrete operator, computed by ARPACK on the matrix-free operator, times the scheme's imaginary-axis limit $\beta \approx 3.34$ and a CFL factor. On the same mesh $\Delta t \propto h/p^2$, so high $p$ pays in steps what it saves in unknowns.
- **Absorbing boundaries.** The first-order Silver–Müller condition (zero exterior state in the flux) is used on a disk, with the reflections de-embedded by the harmonic extraction. The other option is a complex-frequency-shifted stretched-coordinate PML in a rectangular frame:
  - $1/s_q = 1 - \sigma_q/(s+\alpha+\sigma_q)$ acts on each weak derivative $\partial_q f$ through an auxiliary field $\Psi$ with $\dot\Psi = \sigma_q\,\partial_q f - (\sigma_q+\alpha)\Psi$;
  - the weak derivative includes the lifted flux term, split by components: $v^* - v^- = \hat{\mathbf t}\,(H_\tau^*-H_\tau^-)$;
  - four auxiliary fields live on the PML elements only.

  A first version without the CFS shift and with direct $\sigma$ damping of $\mathbf v$ gave an $O(1)$ low-frequency error in a 1000 fs Mie run.
- **Plane-wave source.** The scattered-field formulation $a\,\partial_t u_s = \partial_x v_{s,y} - \partial_y v_{s,x} - (a - a_b)\,\partial_t u_{inc}$, plus the analogous source for $\mathbf v$ in TE. It is a volume source on the nodes of the non-background elements. It is exact for curved elements because the flux only sees continuous tangential components.
- **Dispersive media.** `DispersiveMaterial` with Lorentz/Drude ($\dot J = -\gamma J - \omega_0^2 P + \varepsilon_0 S\,E$, $\dot P = J$), Debye and conductivity. The poles are driven by the *total* field $E_s + E_{inc}$ and are integrated by the same RK scheme. The ADE is solved without a time-discrete permittivity (no $\sin(\omega\Delta t/2)$ factors): the time error is RK4's $O(\Delta t^4)$.
- **Adjoint compatibility.** The semi-discrete system $\dot X = A X + f(t)$ is linear. `operator_matrix()` assembles $A$ exactly as a sparse matrix by probing the matrix-free operator with a distance-2 colouring of the elements: about $n_{colours}\cdot N_p\cdot n_{fields}$ applications, independent of the mesh size. The discrete adjoint of an LSRK step is the same scheme applied to $A^T$ in reverse. `test_operator_matrix_and_adjoint_dot_product` checks $\langle y, S x\rangle = \langle S^T y, x\rangle$ to $10^{-12}$; a mutation (using $A$ instead of $A^T$) fails. This is a different structure from FIT's leapfrog adjoint (it needs the stage vectors, i.e. 5 states per step to checkpoint), but it is equally exact.
- **GPU suitability.** All work is dense element-local GEMMs ($[D_r; D_s]\,(N_p\times 3K)$, lift $N_p\times 3N_{fp}$) plus gathers, which is the textbook GPU pattern. Arithmetic intensity grows with $p$, so DG is *more* GPU-friendly than FIT's memory-bound sparse updates. The numpy prototype runs on one CPU core.

**Cartesian (2,4) (`FDTD24`).** The Yee staggering with the fourth-order staggered difference

$$
\partial_x f\big|_{i+1/2} \approx \frac{1}{h}\Big[\tfrac98\,(f_{i+1}-f_i) - \tfrac1{24}\,(f_{i+2}-f_{i-1})\Big],
$$

[Yefet & Petropoulos, 2001; Hwang & Cangellaris, 2003] gives a spatial phase error of $-\tfrac{3}{640}(kh)^4$. The leapfrog stability limit drops by $6/7$ ($\max|\text{symbol}| = 9/8 + 1/24$).

- **Time error.** With leapfrog, the time error $+(\omega\Delta t)^2/24$ dominates the (2,4) spatial error at practical CFL numbers. At 14 cells per dielectric wavelength it is about 7 times larger. Option `time_scheme='lsrk4'` integrates the same semi-discrete operator, including the CPML auxiliaries in ADE form, with LSRK4.
  - Yoshida's 4th-order symplectic composition [Yoshida, 1990] was tried and abandoned: its negative sub-step turns the CPML into an amplifier, and the run went unstable.
- **Interfaces.** Sub-pixel averaging is the Hwang–Cangellaris/Farjadpour type. TM uses the area-averaged $\varepsilon$ ($E_z$ is tangential). TE uses the diagonal of the averaged inverse tensor, $(\varepsilon^{-1})_{ii} = n_i^2\langle\varepsilon^{-1}\rangle + (1-n_i^2)/\langle\varepsilon\rangle$. Both are at best second order, so the interface caps the (2,4) scheme at second order once the bulk error is gone. The study measures where that happens.
- **Source and loss.** The scattered-field source is written in constitutive form: $E_s = \kappa D_s/\varepsilon_0 + (\kappa - 1)E_{inc}$ with $\kappa = \varepsilon_{\rm eff}^{-1}$. A conductivity is handled by Crank–Nicolson on $D_s$, driven by the total field. PEC walls use image ghosts (integer-point fields odd, half-point fields even), so the wide stencil keeps 4th order at a PEC wall.
- **Adjoint and GPU.** Every update is a sum of difference operators and diagonal scalings, so it is linear and transposable like Yee, and it is as GPU-friendly as Yee: the stencil is twice as wide, the memory traffic per cell is the same, and there are about 1.5 times as many flops.

**Results** (study `studies/highorder-mie/`, benchmark arms in `studies/highorder-bench/`). The numbers are in the study README: unknowns, work and uncontended time to reach 2 %, 1 % and 0.1 % RMS error in $Q_{sca}$, against the best second-order baselines (graded triangle FIT with PML, graded conformal Cartesian FIT).

#### Limits & Failure Modes

- **Straight-sided DG elements are second order at curved interfaces.** With polygonal interfaces, DG at $p = 4$ stalls at about 3 % RMS on the Mie cylinder (B1), the same as a coarse FIT mesh. High order needs the geometry to high order too. Curved elements are supported for circles only (any number, including the outer boundary). General curves need a CAD projection.
- **Cost per unknown.** A DG unknown costs 5 RHS evaluations per step and dense element operations, with $\Delta t \propto h/p^2$. Measured on one n2 core, it is 150–300 ns per unknown-step on one n2 core vs 17–35 ns for the numpy Yee/(2,4) leapfrog and ~19 ns for triangle FIT. DG wins only where its error per unknown falls much faster, at tight tolerances or electrically large problems; the study quantifies the crossover. At loose tolerances (a few %), graded FIT is cheaper.
- **Time step set by the smallest element.** Global RK has no local time stepping. A small element at a gap or a thin feature sets $\Delta t$ for the whole mesh (B3 gaps).
- **Upwind dissipation.** $\alpha = 1$ damps unresolved modes, which is welcome, but it also damps resolved high-Q resonances slightly. $\alpha = 0$ conserves energy but leaves spurious modes undamped.
- **Mesh quality.** The force-equilibrium mesher (`circles_mesh`) keeps circle chords Delaunay through a node clearance. It does not handle circles that touch or overlap. A gap narrower than the local $h$ is forced to one element across.
- **PML.** It is rectangular and needs vacuum or background in the frame. Its reflection is set by the element resolution of the $\sigma$ profile (about $10^{-3}$ field error for 2 elements of $p = 3$; under $10^{-4}$ for 4). No curved elements may lie inside it.
- PML late-time growth: fine-mesh runs with the rectangular CFS-PML (p = 3 at λ_d/6, p = 4 at λ_d/4) grow after several hundred fs; halving the CFL does not help and the same meshes without PML are stable, so the instability is in the discrete PML (open question; use the de-embedded ABC disk for long windows).
- **(2,4) scheme.** The sub-pixel averaging is second order at best. The diagonal TE tensor drops the off-diagonal terms, so TE is first order near oblique interfaces. With leapfrog the scheme is second order in time; it needs LSRK4 or a small CFL to be 4th order. Its benefit disappears once the interface error dominates. Metals were not implemented in `FDTD24`; the mixed-cell averaging problem of conformal Cartesian metals is unchanged.

References:
- J. S. Hesthaven and T. Warburton, "Nodal high-order methods on unstructured grids I. Time-domain solution of Maxwell's equations", *J. Comput. Phys.* **181**, 186–221 (2002), doi:10.1006/jcph.2002.7118.
- J. S. Hesthaven and T. Warburton, *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*, Springer (2008), doi:10.1007/978-0-387-72067-8.
- T. Warburton, "An explicit construction of interpolation nodes on the simplex", *J. Eng. Math.* **56**, 247–262 (2006), doi:10.1007/s10665-006-9086-6.
- M. H. Carpenter and C. A. Kennedy, "Fourth-order 2N-storage Runge–Kutta schemes", NASA TM-109112 (1994).
- M. Ainsworth, "Dispersive and dissipative behaviour of high order discontinuous Galerkin finite element methods", *J. Comput. Phys.* **198**, 106–130 (2004), doi:10.1016/j.jcp.2004.01.004.
- K. Busch, M. König and J. Niegemann, "Discontinuous Galerkin methods in nanophotonics", *Laser Photonics Rev.* **5**, 773–809 (2011), doi:10.1002/lpor.201000045.
- W. J. Gordon and C. A. Hall, "Construction of curvilinear co-ordinate systems and applications to mesh generation", *Int. J. Numer. Methods Eng.* **7**, 461–477 (1973), doi:10.1002/nme.1620070405.
- A. Yefet and P. G. Petropoulos, "A staggered fourth-order accurate explicit finite difference scheme for the time-domain Maxwell's equations", *J. Comput. Phys.* **168**, 286–315 (2001), doi:10.1006/jcph.2001.6691.
- K.-P. Hwang and A. C. Cangellaris, "Computational efficiency of Fang's fourth-order FDTD schemes", *Electromagnetics* **23**, 89–102 (2003), doi:10.1080/02726340390159450.
- K.-P. Hwang and A. C. Cangellaris, "Effective permittivities for second-order accurate FDTD equations at dielectric interfaces", *IEEE Microw. Wireless Compon. Lett.* **11**, 158–160 (2001), doi:10.1109/7260.916329.
- Z. Zhao and G. W. Wei, "High-order FDTD methods via derivative matching for Maxwell's equations with material interfaces", *J. Comput. Phys.* **200**, 60–103 (2004), doi:10.1016/j.jcp.2004.03.008.
- H. Yoshida, "Construction of higher order symplectic integrators", *Phys. Lett. A* **150**, 262–268 (1990), doi:10.1016/0375-9601(90)90092-3.
- J. A. Roden and S. D. Gedney, "Convolution PML (CPML)", *Microw. Opt. Technol. Lett.* **27**, 334–339 (2000).
- F. Rapetti and A. Bossavit, "Whitney forms of higher degree", *SIAM J. Numer. Anal.* **47**, 2369–2386 (2009), doi:10.1137/070705489 (the high-order FIT route, not implemented; see the study README).

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

**GPU and MPI.** The adjoint runs on the backend of the forward solver. With `use_gpu=True` the adjoint variables, the transposed operators $\mathbf A^H$, $\mathbf B^H$ (CuPy CSR matrices), the CPML, lumped and dispersive coefficients, the monitor adjoint sources and the stored increments $q^n/\varepsilon^{-1}$ all stay on the GPU, in float64 like the forward run, so the GPU gradient is as exact as the CPU one. When $N_t N_\text{design}$ stored values would exceed half of the free GPU memory, they are kept in host memory instead (one row copied per step; `store_on_host` forces either choice), and `store_dtype=np.float32` halves them.

With MPI, each of the $P$ ranks holds a slab of $z$-planes plus one ghost plane on each interior side, and the forward step is, on the stacked local arrays,

$$
\mathbf x' = \mathbf R\,\mathbf X_E\,\mathbf L_E\,\mathbf X_H\,\mathbf L_H\,\mathbf Q\,\mathbf x ,
$$

where $\mathbf Q$ copies each owned plane into the ghost plane of the neighbouring rank, $\mathbf L_H$ and $\mathbf L_E$ are the local half steps (they read the ghost planes), $\mathbf X_H$ and $\mathbf X_E$ are the ghost exchanges of `SolverFIT3D._mpi_communicate` (the ghost values computed locally are discarded and replaced by the owner's values) and $\mathbf R$ keeps the owned planes. The output is ghost-consistent, so this is the exact global step $\mathbf U$, and

$$
\mathbf U^T = \mathbf Q^T\,\mathbf L_H^T\,\mathbf X_H^T\,\mathbf L_E^T\,\mathbf X_E^T\,\mathbf R^T .
$$

$\mathbf R^T$ extends by zeros, $\mathbf L^T$ are the local transposed half steps above, and $\mathbf Q^T$ and $\mathbf X^T$ are both the reverse exchange: the adjoint held on each ghost plane is sent to the rank that owns the plane, added there and zeroed. The transposed step therefore performs, in reverse order of the forward exchanges, a reverse exchange of $\boldsymbol\lambda_h$ after $\mathbf B^H$ ($\mathbf X_H^T$) and of $\boldsymbol\lambda_e$ after $\mathbf A^H$ ($\mathbf Q^T$); a reverse exchange of the incoming $\boldsymbol\lambda$ also moves to their owners the adjoint sources of monitor samples read on ghost planes (e.g. the H average of a flux plane on a subdomain interface). Each rank stores the increments of the design edges it owns, so $dF/d\rho = \sum_r \mathbf P_r^T (\cdot)$ is summed over the ranks, and the monitor outputs of all ranks are concatenated along their sample axis (in rank order, i.e. along $z$ for a `DFTMonitor`) so that every rank evaluates the same objective. This is the standard transposition of a halo exchange in a distributed linear operator, written out for the Wakis update order.

**Fused GPU kernels.** On a solver built with `fused=True` (`wakis/gpu_fused.py`, see `docs/performance.md`) the forward step evaluates, per edge $i$ of component $T$, the two derivative terms $p_a = (\mathbf D_a \mathbf e_{S_a})_i$ and $p_b = (\mathbf D_b \mathbf e_{S_b})_i$ from one merged sparse row, updates the $\psi$ entries owned by that edge and the field. The transposed step uses the same split: for the electric half step, with $\nu = \Delta t\,\varepsilon^{-1}\boldsymbol\lambda_e$,

$$
\tilde\mu_a = \mu_{\psi_a}' + \nu_i,\quad \tilde\mu_b = \mu_{\psi_b}' - \nu_i,\qquad
\beta_{a,i} = \nu_i + c_a\tilde\mu_a,\quad \beta_{b,i} = -\nu_i + c_b\tilde\mu_b,\qquad
\mu_{\psi} = b\,\tilde\mu,
$$

then $\boldsymbol\lambda_h \leftarrow \boldsymbol\lambda_h' + [\mathbf D_a^H\;\mathbf D_b^H]\,[\boldsymbol\beta_a;\boldsymbol\beta_b]$ with the stored transpose of the merged rows (and the same for the magnetic half step with $\kappa = -\Delta t\,\mu^{-1}\boldsymbol\lambda_h$ and the loss factor on $\boldsymbol\lambda_e$). This is algebraically the transposed step above (only the order of floating-point additions differs), so the fused adjoint is exact to rounding: the dot-product test gives $<10^{-12}$ relative on PEC, lossy, periodic, CPML, mixed, Bloch, tensor-averaged and dispersive solvers, and fails for a $10^{-3}$ change of one transposed entry (`tests/test_049_gpu_fused.py`).

**Validation** (`tests/test_020_adjoint.py`): the dot-product identity $\langle \mathbf U\mathbf x, \mathbf y\rangle = \langle \mathbf x, \mathbf U^T\mathbf y\rangle$ holds to $\sim 10^{-17}$ relative to $\sum|U x||y|$ for PEC, periodic, lossy and CPML steps; adjoint gradients match central finite differences to $10^{-8}$–$10^{-10}$ with $h=10^{-4}$, and the difference scales as $h^2$, i.e. it is the finite-difference truncation error. `tests/test_026_adjoint_features.py` repeats the dot-product test against the real time loop (element `update`, `one_step`, `post_step`), including the auxiliary states, for magnetic currents, lumped R/L/C elements and ports (PEC and CPML), Lorentz + Drude + Debye media with conductivity (PEC and CPML) and complex Bloch fields (PEC and CPML): $\le 1.4\times10^{-16}$ relative. Finite-difference checks ($h = 10^{-4}$) of jax.grad through flux, far-field intensity, mode amplitudes ($a_+$, $a_-$, with a `ModeSource`), DFT fields with a magnetic source, $|S_{21}|^2$ plus a lumped RLC current, a dispersive medium beside the design, a dispersive design material and Bloch fields (including $-f_0$) agree to $10^{-10}$–$2\times10^{-8}$ relative. Both files pass unchanged with `--gpu` (L4 GPU, same float64 tolerances). `tests/test_032_mpi_adjoint.py` (run under `mpiexec -n 2`, 4 and 8) checks the distributed transposed step with the global inner product ($\le 10^{-17}$ relative, PEC with loss and periodic $x$, $y$) and an `AdjointSimulation` whose design region, DFT line and flux planes cross the subdomain interfaces: outputs, objective and gradient equal the serial run to $\le 5\times10^{-15}$ relative and a central finite difference to $1.3\times10^{-10}$. Without the reverse exchanges the same tests fail by $1.5\times10^{-3}$ (dot product) and 35% (gradient).

**Limitations.** The adjoint supports CPU and GPU runs with PEC/PMC/periodic/Bloch/CPML boundaries, ohmic loss, dispersive media, lumped elements and ports, and electric, magnetic and mode sources. With MPI it supports the standard step (tested with PEC and periodic $x$, $y$ boundaries and ohmic loss) with `DFTMonitor` and `FluxMonitor` objectives; CPML, lumped elements, dispersive media, `ModeMonitor`, `Near2FarMonitor` and `LumpedReadout` raise `NotImplementedError` under MPI. ABC (Mur), PML, TFSF injection and SIBC raise `NotImplementedError`, as do `J` components of `DFTMonitor`, lumped elements sharing edges with each other or with dispersive media, and design regions overlapping lumped elements or (other) dispersive media. Only isotropic $\varepsilon$ (or the fill fraction of one dispersive material) is designed; mode profiles and $\Delta t$ must not depend on the design.

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

### 🌀 Linear plasma wakefield in a plasma-filled pipe

A rigid bunch that crosses a plasma displaces the plasma electrons, which then oscillate at the plasma frequency behind it: a plasma wakefield [Chen et al., 1985]. In the linear regime this is an ordinary dispersive-medium wakefield problem, so Wakis can compute it with the beam source of the wake solver and a Drude material (`wakis.plasma`, example `examples/007_linear_plasma_wakefield.py`, test `tests/test_037_plasma_wake.py`).

**Plasma model.** A cold, collisionless, unmagnetized electron fluid with immobile ions, linearized around the uniform density $n_p$, carries the current $\partial_t \mathbf J_p = \varepsilon_0\omega_p^2\mathbf E - \nu \mathbf J_p$ with $\omega_p = \sqrt{n_p e^2/(\varepsilon_0 m_e)}$ (`plasma_frequency(n)`), i.e. the Drude permittivity $\varepsilon(\omega) = 1 - \omega_p^2/(\omega^2 - i\nu\omega)$ of Section *Dispersive materials*. `cold_plasma(omega_p, collision_rate=0)` returns this `DispersiveMaterial`; its ADE is exactly the linearized fluid momentum equation, integrated with the scheme and stability limit given there ($(\omega_p\Delta t/2)^2 + C^2 \le 1$).

**Linear wake of a rigid bunch.** Let a bunch with line density $\lambda(\xi)$ (charge $q = \int\lambda\,d\xi$) move at $v = c$ along $z$ at the transverse position $\mathbf r_0$ inside a perfectly conducting circular pipe of radius $a$ filled with plasma; all fields depend on $\mathbf r_\perp$ and $\xi = z - ct$. With $e^{i(kz - \omega t)}$, $\omega = kc$, Maxwell's equations with $\nabla\cdot(\varepsilon\mathbf E) = \rho_b/\varepsilon_0$ give for $E_z$ (a short Wakis derivation of the result of [Keinigs & Jones, 1987], whose unbounded-plasma form is also reviewed in [Esarey et al., 2009])

$$
\left(\nabla_\perp^2 - k_p^2\right)\tilde E_z = \frac{i k\, k_p^2}{k^2 - k_p^2}\,\frac{\tilde\lambda(k)\,\delta(\mathbf r_\perp - \mathbf r_0)}{\varepsilon_0},
\qquad k_p = \frac{\omega_p}{c},
$$

because $\partial_z^2 + \varepsilon\,\omega^2/c^2 = -k_p^2$ for every $k$ when $v = c$. The transverse Green function with $g = 0$ on the wall, $(\nabla_\perp^2 - k_p^2) g = -2\pi\delta(\mathbf r_\perp - \mathbf r_0)$, follows from Graf's addition theorem for $K_0$ [DLMF §10.44(ii)]:

$$
g(\mathbf r_\perp;\mathbf r_0) = K_0(k_p|\mathbf r_\perp - \mathbf r_0|) - \sum_{m\ge0} \epsilon_m \cos m(\phi-\phi_0)\, \frac{K_m(k_p a)}{I_m(k_p a)}\, I_m(k_p r_0)\, I_m(k_p r),
\qquad \epsilon_0 = 1,\ \epsilon_{m>0} = 2
$$

(`pipe_green_function`; $g = K_0$ for $a\to\infty$). Inverting the Fourier transform with the causal prescription $\omega \to \omega + i0$ (poles at $k = \pm k_p$ below the real axis, so nothing ahead of the bunch) gives

$$
E_z(\mathbf r_\perp, \xi) = -\frac{k_p^2\, g}{2\pi\varepsilon_0}\int_\xi^\infty \lambda(\xi')\cos k_p(\xi - \xi')\,d\xi' ,
$$

and Faraday's law, $\partial_\xi(E_x - cB_y) = \partial_x E_z$ (Panofsky–Wenzel for $v=c$), gives the transverse force per unit charge on a test particle moving at $c$,

$$
\mathbf F_\perp = \mathbf E_\perp + c\,\hat{\mathbf z}\times\mathbf B = \frac{k_p\,\nabla_\perp g}{2\pi\varepsilon_0}\int_\xi^\infty \lambda(\xi')\sin k_p(\xi' - \xi)\,d\xi' .
$$

For a Gaussian bunch centred at $\xi = 0$ both integrals are closed-form, $\int_\xi^\infty\lambda e^{-ik_p\xi'}d\xi' = \tfrac q2 e^{-k_p^2\sigma_z^2/2}\operatorname{erfc}\!\big((\xi + ik_p\sigma_z^2)/\sqrt2\sigma_z\big)$ (`linear_wake`). Far behind the bunch the wake is an undamped cosine,

$$
E_z \to -\frac{q\,k_p^2\, g}{2\pi\varepsilon_0}\, e^{-k_p^2\sigma_z^2/2}\cos k_p\xi ,
$$

(`wake_amplitude`). For a given bunch length and a density scan at fixed $k_p a$, $k_p r$, the amplitude $\propto (k_p\sigma_z)^2 e^{-k_p^2\sigma_z^2/2}$ is maximal at $k_p\sigma_z = \sqrt2$ [Esarey et al., 2009]. A positive bunch is decelerated ($E_z<0$ at its centre). In impedance language the point-charge wake per unit length is that of an undamped resonator at $\omega_p$ (a Wakis restatement): $w'_\parallel(\xi) = (k_p^2 g/2\pi\varepsilon_0)\cos k_p\xi$ behind the charge, i.e. $R'/Q = Z_0 k_p g/(2\pi)$ per unit length, with $\operatorname{Re}Z_\parallel'(\omega)$ a delta function at $\omega_p$ for an infinitely long pipe.

**Simulation setup and discretization.** `Beam` drives a line current $J_z$ in one column of grid nodes, so the simulated field is the response of the lattice transverse operator: it is singular on the beam line and differs from $K_0$ by $O(h^2/r^2)$ near it; compare at $r \gtrsim a/4$ or at fixed physical radii under refinement. The pipe is a vacuum STL cylinder in a PEC background (`bg='pec'`, `GridFIT3D(conformal='pec')` for cut-cell walls) and the plasma is added to every open edge with `solver.add_dispersive_material(cold_plasma(omega_p), center, size)` (box faces placed between E samples so that no edge is half-filled). Two transients must be kept out of the comparison window:
* *entrance*: a bunch with $v=c$ emerging from the PEC end wall into vacuum does not radiate, but at the vacuum–plasma boundary its unscreened field reorganizes into the screened one and emits transition radiation in TM modes of the plasma-filled pipe (cut-off $\sqrt{\omega_p^2 + (c\,j_{0n}/a)^2}$). These travel with group velocity $<c$ and fall behind the bunch as it advances; at a distance $D$ from the entrance only frequencies $\omega \gtrsim \omega_c\sqrt{D/2|\xi|}$ arrive within $|\xi|$ of the bunch, and they are suppressed by the bunch spectrum. Short bunches have more spectrum above cut-off and are the worst case: for $k_p\sigma_z = 0.5$ and a window of $3\lambda_p$, a 12 $\lambda_p$ pipe shows 8–17 % RMS deviation, 24 and 36 $\lambda_p$ pipes about 2 %, and the residual is a chirped transient that starts at $\xi\approx-2\lambda_p$ (over $2\lambda_p$ behind the bunch the deviation is 0.4–0.6 % at 48 cells per $\lambda_p$, but 2.3 % at 32 cells per $\lambda_p$, where the transient reaches closer to the bunch; see `docs/img/plasma_wake_ez.png`). The tests use a vacuum entrance of one $\lambda_p$, $D \ge 15\lambda_p$ and windows of 2–3 $\lambda_p$. An adiabatic density up-ramp (2 $\lambda_p$) did not reduce the transient.
* *exit*: stop when the bunch is about one $\lambda_p$ before the exit wall; nothing propagates ahead of a bunch moving at $c$.
The field is then read along $z$ at one time, $\xi = z - z_\mathrm{bunch}(t)$ with $z_\mathrm{bunch} = z_\min + c(t - t_i)$ (`Beam.ti`), $E_z$ on the $z$ midpoints at $t^{n+1}$ and $H$ half a step earlier. The Drude ADE shifts the oscillation frequency to $2\arcsin(\omega_p\Delta t/2)/\Delta t$ and the grid dispersion changes the effective $k_p$ by $O((k_p h)^2)$; both are below $10^{-3}$ at 20 cells per $\lambda_p$.

**Validation** (`tests/test_037_plasma_wake.py`, $k_p a = 2$, $\lambda_p = 1$ mm, cubic cells; slow cases run on an NVIDIA L4):

Errors of $E_z(\xi)$ behind an on-axis bunch, $k_p\sigma_z = 1$, 24 $\lambda_p$ pipe, window $-3\lambda_p<\xi<-3\sigma_z$: relative error of the fitted amplitude of $A\cos(k\xi+\phi)$ (the wavenumber error is $\le 2\times10^{-4}$ in all runs, decreasing with $h$):

| cells per $\lambda_p$ (per $a$) | staircase $r\approx a/2$ | staircase $r\approx 3a/4$ | conformal $r\approx a/4$ | conformal $r\approx a/2$ | conformal $r\approx 3a/4$ |
|---|---|---|---|---|---|
| 24 (7.6) | +4.9 % | +16.8 % | +2.1 % | +0.73 % | +0.26 % |
| 32 (10.2) | +2.5 % | +10.1 % | +1.0 % | +0.40 % | +0.54 % |
| 48 (15.3) | +2.3 % | +5.7 % | +0.47 % | +0.15 % | +0.11 % |
| 64 (20.4) | +1.3 % | +4.5 % | +0.26 % | +0.08 % | +0.05 % |

The staircased wall converges at first order (effective radius error $\sim h$, largest near the wall); with conformal PEC the error converges at second order and is dominated near the beam line by the lattice Green function ($r = a/4$ is only 2–5 cells). At 48 cells per $\lambda_p$ with conformal walls, $k_p\sigma_z = 0.5,\ \sqrt2,\ 2$ give amplitude errors of 0.1–0.7 % and RMS deviations of 0.1–0.6 % over two plasma periods; the wake is undamped as predicted. For an offset bunch ($x_0 = \lambda_p/8$), $F_x = E_x - cB_y$ at 6.5 cells on either side agrees to 0.8 % (48 cells per $\lambda_p$; at $\lambda_p/16$ from the beam: 5.4 %, 2.9 %, 1.6 % at 32, 48, 64 cells per $\lambda_p$). The CI test (20 cells per $\lambda_p$, < 15 s on a laptop CPU) checks 2.5 % in amplitude at $a/2$ and the staircase–conformal difference near the wall.

![Linear plasma wake behind a Gaussian bunch, Wakis vs theory](img/plasma_wake_ez.png)

![Wake amplitude vs bunch length and radius](img/plasma_wake_amplitude.png)

**Limits.** The model is the *linear, cold, rigid-beam* response: (i) linear plasma response, valid when the perturbed density is small, $|\delta n| \ll n_p$ (for a narrow bunch this requires a bunch density well below $n_p$ over the plasma skin depth); the results scale linearly with $q$; (ii) cold plasma: no thermal pressure, so the wake does not propagate and has no Landau damping (add $\nu>0$ for collisional damping); (iii) immobile ions, valid for times short compared with the ion plasma period; (iv) a rigid bunch at exactly $v = c$ (`Beam(beta=1)`, the default; the formulas above assume $v=c$): no beam evolution, no self-modulation, hosing, or beam loading by a witness bunch (other than as an additional prescribed source). The nonlinear blowout regime ($n_b \gtrsim n_p$, electron cavitation, wave breaking) needs a particle-in-cell or quasi-static code [Esarey et al., 2009].

**Relation to proton-driven (AWAKE-type) plasma wakefields.** Proton-driven plasma wakefield acceleration was proposed by [Caldwell et al., 2009]; the AWAKE experiment at CERN drives a wake with 400 GeV SPS proton bunches in a rubidium plasma about 10 m long and has accelerated externally injected electrons up to 2 GeV [Adli et al., 2018]. The linear formula above shows why the proton bunch cannot drive the wake directly: its length is many plasma wavelengths, $k_p\sigma_z \gg 1$, so $e^{-k_p^2\sigma_z^2/2}$ cancels the wake of the smooth bunch. AWAKE therefore relies on the seeded self-modulation of the long bunch into micro-bunches spaced by $\lambda_p$ that drive the wake resonantly [Kumar, Pukhov & Lotov, 2010]; the growth of the resulting transverse wakefields from a seed level below 15 MV/m to above 300 MV/m was observed at plasma densities of $2.1\times10^{14}$ and $7.7\times10^{14}\ \mathrm{cm^{-3}}$ [Turner et al., 2019]. Self-modulation is an instability of a *deformable* beam and is outside this rigid-beam model; Wakis can however compute the linear wake of any *prescribed* bunch train or truncated profile with the same setup (e.g. several `Beam` sources), which is the seed and the linear drive term of that process.

References:
- P. Chen, J. M. Dawson, R. W. Huff and T. Katsouleas, "Acceleration of electrons by the interaction of a bunched electron beam with a plasma", *Phys. Rev. Lett.* **54**, 693–696 (1985), [doi:10.1103/PhysRevLett.54.693](https://doi.org/10.1103/PhysRevLett.54.693).
- R. Keinigs and M. E. Jones, "Two-dimensional dynamics of the plasma wakefield accelerator", *Phys. Fluids* **30**, 252–263 (1987).
- E. Esarey, C. B. Schroeder and W. P. Leemans, "Physics of laser-driven plasma-based electron accelerators", *Rev. Mod. Phys.* **81**, 1229–1285 (2009), [doi:10.1103/RevModPhys.81.1229](https://doi.org/10.1103/RevModPhys.81.1229).
- NIST Digital Library of Mathematical Functions, §10.44(ii), Addition theorems for modified Bessel functions, [dlmf.nist.gov/10.44](https://dlmf.nist.gov/10.44).
- A. Caldwell, K. Lotov, A. Pukhov and F. Simon, "Proton-driven plasma-wakefield acceleration", *Nature Phys.* **5**, 363–367 (2009).
- N. Kumar, A. Pukhov and K. Lotov, "Self-modulation instability of a long proton bunch in plasmas", *Phys. Rev. Lett.* **104**, 255003 (2010), [doi:10.1103/PhysRevLett.104.255003](https://doi.org/10.1103/PhysRevLett.104.255003).
- E. Adli et al. (AWAKE Collaboration), "Acceleration of electrons in the plasma wakefield of a proton bunch", *Nature* **561**, 363–367 (2018), [doi:10.1038/s41586-018-0485-4](https://doi.org/10.1038/s41586-018-0485-4).
- M. Turner et al. (AWAKE Collaboration), "Experimental observation of plasma wakefield growth driven by the seeded self-modulation of a proton bunch", *Phys. Rev. Lett.* **122**, 054801 (2019), [doi:10.1103/PhysRevLett.122.054801](https://doi.org/10.1103/PhysRevLett.122.054801).


```{admonition} Modularity

#### 🧪 Open-Source Compatibility

The wake and impedance analysis is performed within the `WakeSolver` class in `Wakesolver.py`. Even if this module is tailored for Wakis, it is completely modular and can be used with other EM solvers besides Wakis' `SolverFIT3D` output:
- It has been tested with **WarpX** EM fields. WarpX is a powerful open-source PIC solver for full 3D EM fields
- It has been benchmarked with **CST** Wakefield solver, using both EM field ouput (time-domain field monitors) and calculated wake potential and impedance. > See the full wake analysis benchmark in [IPAC'23 proceedings, E. de la Fuente](https://doi.org/10.18429/JACoW-IPAC2023-WEPL170).
- **Interoperability**: Wakis can read field maps in HDF5, CSV, or NumPy format
- **Subvolume extraction** and interpolation are supported for field-based post-processing
```
