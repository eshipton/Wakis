# 🕹️ User's Guide

This section shows how to use `wakis` in a simulation script.

Since `wakis` has been developed for computing beam-coupling impedance for particle accelerator components, the example that will serve as a conductive thread for the explanation is a **pillbox cavity with a passing proton beam**.

The guide will go into detailed step-by-step on how to write the simulation script and visualize or access the computed data.

```{contents}
:depth: 3
```

## 🪜 Simulation step-by-step
### Import modules

The first part of a python script always includes importing external sources of code e.g., `packages` or `modules`. In `wakis`, we use:

* `numpy`: Used for numerical operations, especially for matrix operations.
* `scipy.constants`: to import physical constants easily like vacuum permittivity `eps_0` or the speed of light `c`
* `matplotlib`: Used for 1d and 2d plotting and visualization.
* `h5py`: To store data in the memory-efficient format HDF5
* `tqdm`: This package is used for displaying progress bars in loops.
* `pyvista`: For handling and visualizing 3D CAD geometries and vtk-based 3D plotting.

Optionally, one can use `pathlib` to work with file paths. The first part of a simulation script could look similar to:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.constants import c as c_light
from tqdm import tqdm
```

Next step is to import the `wakis` classes that will allow to run the electromagnetic simulations:

```python
from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.sources import Beam
```

### Simulation domain, geometry and materials setup

`wakis` is a numerical electromagnetic solver that uses the Finite Integration Technique. The grid used is a structured grid composed by rectangular cells. The simulation domain is a rectangular box that will be broken into cells where the electromagnetic fields will be computed. Notice that all the input parameters for Wakis must be in SI Base Units.

#### Number of mesh cells

 In every `wakis` simulation example, the script starts specifying the number of mesh cells per direction the user wants to use:

```python
# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 100
```

```{note}
Note that the number of cells will heavily affect the simulation time (in particular, the timestep due to CFL condition), and also the memory requirements.
```

### `STL` geometry importing
In beam-coupling impedance simulations one is usually interested in the geometric impedance, together with the impedance coming from material properties. In `wakis`, the geometry to simulate (sometimes referred as Embedded boundaries) can be imported from a `.stl` file containing a CAD model, and the units should be meters [m].

```python
# Run from the root of a cloned Wakis repository.
data_dir = Path('notebooks/data')
stl_cavity = data_dir / '001_vacuum_cavity.stl'
stl_shell = data_dir / '001_lossymetal_shell.stl'
stl_solids = {'cavity': str(stl_cavity), 'shell': str(stl_shell)}

# Optional: plot the geometry imported using PyVista
geometry = pv.read(stl_shell) + pv.read(stl_cavity)
geometry.plot() # add pyvista **kwargs to make a fancier plot
```
![geometry plot example using a few pyvista **kwargs](img/geometry_plot.png)

Some [example `.stl` files](https://github.com/ImpedanCEI/wakis/tree/main/notebooks/data) are available in the GitHub repository for the `notebooks/` and `examples` folders. However, `PyVista` allows to generate importable geometry really easy through Constructive Solid Geometry (CSG). CSG is a modeling technique where complex geometries are built by combining simple primitives (like cubes, spheres, cylinders, cones) using Boolean operations: Union (+) combines two solids, Intersection (*) keeps only the overlapping volume or Difference (−): subtracts one solid from another. An example on how to generate a pillbox cavity using CSG is shown bellow:

```
import pyvista as pv

# create 2 cylinders oriented in the z-direction
pipe = pv.Cylinder(direction=(0,0,1.), radius=0.12, height=1)
cavity = pv.Cylinder(direction=(0,0,1.), radius=0.5, height=0.3)

# combine them and plot
surf = pipe + cavity
surf.plot()

# save
surf.save('geometry.stl', binary=False)
```


```{seealso}
Check [PyVista's documentation](https://docs.pyvista.org/version/stable/user-guide/simple.html#plotting) for more advanced 3d plotting
```

Imported `.stl` files can also be transformed with the `stl_scale`, `stl_rotate`, and `stl_translate` arguments to `GridFIT3D`. Each accepts a value applied to all solids or a dictionary keyed by solid name. When transforming solids, choose domain bounds that enclose the transformed geometry.

### Associating a material to each solid
Each `stl` solid can be associated with a material by indicating `[eps, mu, conductivity]`. In `materials.py`, the most used materials are available in the material library. The user can specify a value for the relative permittivity $\varepsilon$, the relative permeability $\mu$ and the value for the conductivity $\sigma$ in [S/m] or use the material name from the library:

```python
# Indicate relative permittivity, relative permeability and conductivity [S/m]
# [eps_r, mu_r, conductivity]
# or the material name from the library:
stl_materials = {'cavity': 'vacuum', 'shell': 'pec'}
```

### Defining the `grid` object
After this, the user needs to indicate the domain bounds to simulate, in [m]. They can be calculated from the imported geometry or just hardcoded with `float` values:

``` python
# Domain bounds
geometry = pv.read(stl_shell) + pv.read(stl_cavity)
xmin, xmax, ymin, ymax, zmin, zmax = geometry.bounds
```

Finally, we can build our simulation grid using `GridFIT3D` class including all the information above:

```python
# initialize grid object
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax,
                Nx, Ny, Nz,
                stl_solids=stl_solids,
                stl_materials=stl_materials,
                )
```

Optionally, the `grid` of the simulation can be inspected interactively in 3D (thanks to PyVista):

```python
grid.inspect(
    add_stl=['cavity', 'shell'],
    stl_opacity=0.5,
    stl_colors=['blue', 'red'],
)
```

 ![Gif showing how the grid.inspect() method works](img/grid_inspect.gif)

The grid coordinates and cell sizes are available as attributes. For example, `grid.dx` contains the cell widths in $x$, while `grid.Nx * grid.Ny * grid.Nz` gives the total cell count.

```{tip}
Thanks to a fully exposed python API, all (relevant) class attributes can be checked by typing the class instance (e.g., `grid` or `solver`) followed a dot `.` and the name of the attribute.

When running on `ipython` all attributes and functions() can be viewed by typing the name of the class instance + `.` and then pressing `TAB`. Similarly, when writing code on a IDE (e.g., Visual Studio), a list with all the attributes and functions will be shown when typing e.g., `solver.`.
```

### Setting up the electromagnetic solver

Once the simulation domain has been defined through the geometry and the grid, the electromagnetic (EM) solver can be created. The solver implemented in `wakis` uses the Finite Integration Technique (FIT) [^1]. The recipe on how to instantiate the `SolverFIT3D` class is given below:

```python
bc_low = ['pec', 'pec', 'pec']
bc_high = ['pec', 'pec', 'pec']
solver = SolverFIT3D(
    grid=grid,
    cfln=0.5,
    bc_low=bc_low,
    bc_high=bc_high,
    use_stl=True,
    bg='vacuum',
)

```

#### Simulation timestep
The solver calculates `solver.dt` from the smallest cell width in each direction and the CFL number. Conductivity can impose a smaller timestep. To specify a timestep explicitly, pass `dt=...` when constructing `SolverFIT3D`.

### Boundary conditions
The required parameters `bc_low` and `bc_high` allow to choose the 6 boundary conditions for our simulation box. For the lower-end boundaries bc_low = [x-, y-, z-] and for the high-end boundaries bc_high = [x+, y+, z+] is used. The supported values to give for -/+ boundaries are:

* `pec` stands for Perfect Electric Conductor: it is a Dirichlet boundary conditions that forces the tangent electric $E$ field at the boundary to be 0.
* `pmc` stands for Perfect Magnetic Conductor: similarly to pec, it is a Dirichlet boundary conditions that forces the tangent magnetic $H$ field at the boundary to be 0.
* `periodic`: The field values of the high-end boundary are passed to the lower-end boundary, simulating a periodic structure where the fields re-enter the simulation domain.
* `abc`: First-order Mur absorbing boundary condition (ABC)[^2], suited to waves approaching the boundary near normal incidence.
* `pml`: Graded conductive absorbing layer. Its reflection depends on layer thickness and the conductivity profile; see the [physics guide](physicsguide.md).
* `cpml`: Convolutional perfectly matched layer, the preferred absorbing boundary for broadband and oblique waves [^3].
* `bloch`: Bloch-periodic (Floquet) boundary: like `periodic`, but the fields re-enter with the phase $e^{ik_aL}$ set by `k_bloch=(kx, ky, kz)` [rad/m]. Needs `dtype=np.complex128` for nonzero `k_bloch`; see the unit-cell example below.

An example of the boundary conditions that are typically used for the pillbox cavity example are:

```python
# To absorb waves at both longitudinal ends instead of using PEC:
bc_low = ['pec', 'pec', 'cpml']
bc_high = ['pec', 'pec', 'cpml']
# Pass these lists when constructing SolverFIT3D.
```
### Accessing `solver`'s fields, matrices and simulation parameters
Similarly to `grid`, the solver exposes its fields and parameters as attributes. Inspect the calculated timestep with:

```python
print(solver.dt)  # [s]
```

#### Electromagnetic fields `E`, `J`, `H`
Once the solver has been instantiated, the electromagnetic fields electric $E$, magnetic $H$ and current $J$ can be accessed and modified to e.g., add initial conditions to the simulation. The fields are 3D vectorial matrices of sizes [Nx, Ny, Nz]x3. The times 3 comes from their vectorial nature, since there are values for each simulation cell in $x$, $y$, and $z$ direction. Below an example on how to access the electric field component $E_z$ to add an initial condition to the field:

```python
# modify the lower left bottom corner (cell [0,0,0]) of the simulation domain.
solver.E[0,0,0,'z'] = c_light

# modify the x component of H for a XY plane at a particular z
iz = Nz//3
solver.H[:, :, iz, 'x'] = np.ones((Nx, Ny))

# access the y component of J along z at a particular x, y
ix, iy = Nx//2, Ny//2
solver.J[ix, iy, :,'y']

# get the absolute value of the electric field
E_abs = solver.E.get_abs() #size [Nx, Ny, Nz]
```

The `inspect()` and `inspect3D()` methods allow quick visualization of field values:
``` python
solver.E.inspect(plane='YZ',            # 2d plane, cut at the domain center
                 cmap='bwr',            # colormap
                 dpi=100,               # plot DPI value (pixel resolution)
                 # also, the plane can be specified as a 2D slice in x,y,z
                 # e.g., for the YZ plane at x = Nx//2
                 x=Nx//2,
                 y=slice(0,Ny),
                 z=slice(0,Nz)
                )

```

#### Material tensors `ieps`, `imu`, `sigma`

The same applies for the material tensors for permittivity $\varepsilon$, permeability $\mu$, and conductivity $\sigma$. For computational cost reasons, the values saved in memory correspond to $\varepsilon^{-1}$ and $\mu^{-1}$.  To access a specific value or a slice of values (and modify them if desired):

```python
# permittivity(^-1) tensor in x direction
solver.ieps[:, :, :, 'x']

# permeability(^-1) tensor in y direction
solver.imu[:, :, :, 'y']

# Check the conductivity along one component:
print(np.max(solver.sigma[:, :, :, 'z']))
```

Since `wakis` supports anisotropy, each material tensor stores $x$, $y$, and $z$ components on the 3D grid. Use `solver.sigma.inspect()` to view a slice. Set material properties through `stl_materials` or `bg` when constructing the solver.


```{tip}
Material tensors and electromagnetic fields are `Field` objects. Use their `inspect()` method to check a 2D slice before running the simulation.
```
### Running a simulation

With an initial field or a source, advance the simulation one timestep at a time with `solver.one_step()`:
```python
for step in tqdm(range(100)):
    solver.one_step()
```

The `emsolve()` routine runs the loop and can optionally save fields to HDF5 or plot them:

```python
source = Beam(sigmaz=0.02)
solver.emsolve(Nt=100, source=source, save=False, plot=False)
```

To save fields, pass `save=True`, `fields=['E']`, `components=['z']`, and `save_every=10`. The optional `subdomain` argument selects an $x$, $y$, $z$ slice.

```{tip}
All the functions inside `wakis` are documented with `docstrings` explaining each input parameter. To access it (after instantiating the class) simply type a question mark at the end, e.g.:
`solver.emsolve?` or `solver.plot3D?`
```

### Adding other time-dependent sources

In most time-domain simulations, when the field is not excited by initial conditions, a time-dependent source is used that to update the field values at a particular region in the simulation domain. In `wakis`, several time-dependent sources are available inside `sources.py`. They can be passed to the `solver` object in order for them to be added to the time-stepping routine.

Sources in `wakis` can be:
* Point sources: only modifying the field or field component at one specific location $(x_s, y_s, z_s)$
* Line sources: modifying the field over a line $(x_s, y_s, \forall z)$
* Plane or port sources: modifying the field on a 2d plane e.g., $z=z_s \forall x,y$
* Volume sources: modify the field in a 3d subdomain: e.g., $z=slice(0, Nz-30) \forall x,y$.

The sources can modify components of the $E$ and $H$ fields or the current $J$. Each source updates the solver once per timestep.

To add a time-dependent source, one can simply setup a time-loop and run the routine `solver.one_step()` after the source has been applied (see [Running a simulation](#running-a-simulation) section). However, a more optimized way is to pass a `source` to the EM solver. The `source` objects available inside `sources.py` are:
* `Beam`: a line source for $J_z$ that adds a Gaussian current traversing the domain from z- to z+. Its longitudinal size $\sigma_z$, total charge $q$, and transverse position are constructor arguments.
* `PlaneWave`: a port source that excites a sinusoidal plane wave in the +$z$ direction by modifying $E_x$ and $H_y$ in an $xy$ plane.
* `WavePacket`: a port source for a Gaussian wave packet travelling in +$z$, with configurable frequency or wavelength, longitudinal size, transverse size, and speed.
* `Dipole`: Updates the user-defined field and component every timestep to introduce a dipole-like sinusoidal excitation
* `Pulse`: Injects an electromagnetic pulse at the given source point (xs, ys, zs), with the selected shape {"Harris", "Gaussian", "Rectangular"}, length, and amplitude
* `CurrentSource`: a *soft* point, line, sheet or volume source of impressed current $J$, placed in physical coordinates and driven by a `GaussianSource`, `ContinuousSource` or `CustomSource` time profile (see below).

#### Soft sources for antennas and scattering

Except for `Beam` and `CurrentSource`, the sources above are *hard*: they overwrite field values, so scattered waves reflect off the source cells. `CurrentSource` instead adds an impressed current to Ampère's law. It is transparent to incoming fields, superposes with other sources and with conduction currents, and is normalized to be independent of the mesh. A point source has amplitude in units of current moment $I\ell$ [A m]:

```python
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource

# Broadband Hertzian dipole along z at the origin, 1 GHz +/- 0.3 GHz
dipole = CurrentSource(
    GaussianSource(frequency=1e9, fwidth=0.3e9),
    component="z",
    center=(0.0, 0.0, 0.0),
    amplitude=1.0,  # [A m]
)

# CW sheet of surface current Kx [A/m] at z = -0.1 m spanning the domain in x, y
sheet = CurrentSource(
    ContinuousSource(frequency=2e9),
    component="x",
    center=(0.0, 0.0, -0.1),
    size=(xmax - xmin, ymax - ymin, 0.0),
)

solver.emsolve(Nt=2000, source=[dipole, sheet])
```

`GaussianSource.fourier_transform(f)` returns the analytic source spectrum, so field spectra can be normalized by the excitation, e.g. to obtain the radiation resistance of an antenna.

#### Frequency-domain monitors

`DFTMonitor` and `FluxMonitor` accumulate Fourier transforms during the run, so a single broadband simulation gives fields and power flow at many frequencies. Pass them to `emsolve`, or call `monitor.update(solver, t)` after each `solver.one_step()` in a manual loop, with `t` the time at which the step started:

```python
import numpy as np
import wakis

freqs = np.linspace(0.8e9, 1.2e9, 41)

# Fields at the feed point and on a plane
feed = wakis.DFTMonitor(freqs, components=["Ez"], center=(0.0, 0.0, 0.0))
plane = wakis.DFTMonitor(freqs, components=["Ex", "Hy"], center=(0.0, 0.0, 0.1), size=(0.2, 0.2, 0.0))

# Power radiated out of a closed box and transmitted through a plane (+z)
box = wakis.FluxMonitor(freqs, size=(0.1, 0.1, 0.1))
transmitted = wakis.FluxMonitor(freqs, center=(0.0, 0.0, 0.2), size=(0.2, 0.2, 0.0))

solver.emsolve(Nt, source=dipole, monitors=[feed, plane, box, transmitted])

Ez_f, (x, y, z) = feed.get_field("Ez")  # shape (n_freq, nx, ny, nz)
S = dipole.src_time.fourier_transform(freqs)
P_rad = box.get_flux() / (2 * np.abs(S) ** 2)  # time-averaged power for |Il| = 1 A m [W]
```

Transmission and reflection spectra are obtained, as usual, by normalizing the flux to a reference run without the structure.

#### Inverse design with the adjoint method

`wakis.inverse` computes the exact gradient of an objective of `DFTMonitor` fields with respect to the permittivity of a design region, with one forward and one adjoint run (see the physics guide). The NumPy core needs no extra dependency; the JAX binding needs `pip install wakis[inverse]`. Sources and monitors are templates, copied for every run, and `make_solver` must build a fresh solver each time:

```python
import jax
import jax.numpy as jnp
import numpy as np
import wakis
from wakis.inverse import AdjointSimulation, MaterialGrid, conic_filter, jax_simulation, tanh_projection
from wakis.sources import CurrentSource, GaussianSource

jax.config.update("jax_enable_x64", True)  # keep float64 gradients
f0, dx = 3e9, 7.5e-3

def make_solver():  # 2D TM domain: one periodic cell along z, CPML in x and y
    grid = wakis.GridFIT3D(-0.15, 0.15, -0.15, 0.15, -dx / 2, dx / 2, 40, 40, 1, verbose=0)
    return wakis.SolverFIT3D(grid, bc_low=["cpml", "cpml", "periodic"],
                             bc_high=["cpml", "cpml", "periodic"], n_pml=8, verbose=0)

source = CurrentSource(GaussianSource(f0, 1e9), component="z", center=(-0.06, 0, 0))
probe = wakis.DFTMonitor([f0], components=["Ez"], center=(0.07, 0, 0))
design = MaterialGrid(center=(0, 0, 0), size=(0.06, 0.06, 0), shape=(9, 9, 1), eps_min=1, eps_max=4)
sim = AdjointSimulation(make_solver, [source], {"probe": probe}, design, Nt=400)

# NumPy: F = |Ez(f0)|^2, G = dF/dRe(Ez) + 1j dF/dIm(Ez) = 2 Ez
rho = np.full(design.shape, 0.5)
Ez = sim.forward(rho)["probe"]["Ez"]
grad = sim.vjp({"probe": {"Ez": 2 * Ez}})  # dF/drho, shape (9, 9, 1)

# JAX: filters, projection and objective are differentiated automatically
simulate = jax_simulation(sim)

def objective(rho_raw, beta=8.0):
    rho = tanh_projection(conic_filter(rho_raw, radius=2.0), beta)
    return 1e20 * jnp.sum(jnp.abs(simulate(rho)["probe"]["Ez"]) ** 2)

rho_raw = jnp.full(design.shape, 0.5)
for it in range(20):  # plain normalized gradient ascent
    value, g = jax.value_and_grad(objective)(rho_raw)
    rho_raw = jnp.clip(rho_raw + 0.05 * g / jnp.abs(g).max(), 0, 1)
```

The adjoint supports CPU, non-MPI runs with PEC, PMC, periodic and CPML boundaries, ohmic loss and soft `CurrentSource` excitations; ABC, PML, TFSF, SIBC, GPU and MPI raise `NotImplementedError`. `AdjointSimulation(..., store_dtype=np.float32)` halves the memory of the stored E increments ($N_t \times N_\text{design}$ values).

#### Far-field patterns and antenna metrics

`Near2FarMonitor` records the tangential fields on a closed box around the antenna and computes the far field, radiation intensity, directivity and gain at any angle after the run (CST's far-field monitor). The box must enclose all sources and scatterers and lie in a homogeneous region outside the CPML. The example below computes the pattern of a Hertzian dipole in free space (theory: $D = 1.5\sin^2\theta$, $D_{max} = 1.76$ dBi):

#### Periodic unit cells at oblique incidence (Bloch boundaries)

Metasurfaces, gratings and photonic crystals are simulated with one unit cell and Bloch-periodic boundaries. These are CST's "unit cell" boundaries, or Meep's `k_point`. Put `'bloch'` on the periodic axes and pass the Bloch wavevector `k_bloch` [rad/m]. A nonzero `k_bloch` needs complex fields (`dtype=np.complex128`). An oblique plane wave is launched by a sheet current with the matching phase profile $e^{i\mathbf{k}_\parallel\cdot\mathbf{r}}$:

```python
import numpy as np
import wakis
from wakis.near2far import half_power_beamwidth, plot_pattern_cut
from wakis.sources import CurrentSource, GaussianSource

grid = wakis.GridFIT3D(-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 40, 40, 40)
solver = wakis.SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=10)

src_time = GaussianSource(frequency=1e9, fwidth=0.3e9)
dipole = CurrentSource(src_time, component="z", center=(0.0, 0.0, 0.0))
freqs = np.array([0.9e9, 1.0e9, 1.1e9])
feed = wakis.DFTMonitor(freqs, components=["Ez"])
n2f = wakis.Near2FarMonitor(freqs, size=(0.15, 0.15, 0.15))

solver.emsolve(int((src_time.end_time + 6e-9) / solver.dt), source=dipole, monitors=[feed, n2f])

theta = np.linspace(0, np.pi, 181)
D = n2f.directivity(theta, 0.0)            # shape (n_freq, n_theta), linear
D_max, direction = n2f.max_directivity()   # ~1.5 at theta = 90 deg
print(10 * np.log10(D_max), "dBi")
print(np.degrees(half_power_beamwidth(theta, D[1])), "deg HPBW")  # ~90

ff = n2f.get_farfield(theta, 0.0, r=10.0)  # E_theta, E_phi, H_theta, H_phi at 10 m
# Gain and efficiency need the accepted power, here the work done by the source
S = src_time.fourier_transform(freqs)
P_in = -np.real(feed.get_field("Ez")[0].ravel() * np.conj(S))
G = n2f.gain(theta, 0.0, P_in)
print(n2f.radiation_efficiency(P_in))      # ~1 for this lossless antenna

fig, ax = plot_pattern_cut(theta, D[1], label="E-plane, 1 GHz")
```

All outputs share the normalization of `FluxMonitor.get_flux`; for a harmonic source of complex amplitude $S(f)$ divide fields by $S(f)$ and powers by $2|S(f)|^2$. For an antenna on an infinite ground plane formed by a PEC domain boundary, place the corresponding box face on that boundary and pass e.g. `ground_plane="-z"`: the face is dropped, image currents are added and `radiated_power` integrates the upper half-space only. `wakis.near2far` also provides `ludwig3` (co-/cross-polarization), `axial_ratio` and `plot_pattern_3d`.

from wakis.bloch import incidence_angle, k_parallel
from wakis.sources import CurrentSource, GaussianSource

px, py = 0.02, 0.02                      # unit-cell period [m]
grid = wakis.GridFIT3D(0, px, 0, py, -0.3, 0.3, 20, 20, 240)  # add the cell's STL solids here
k_par = k_parallel(10e9, np.radians(30.0), phi=0.0)  # 30 deg at 10 GHz, plane of incidence xz

solver = wakis.SolverFIT3D(
    grid,
    bc_low=["bloch", "bloch", "cpml"],
    bc_high=["bloch", "bloch", "cpml"],
    k_bloch=k_par,
    dtype=np.complex128,
    use_stl=True,
)

src_time = GaussianSource(frequency=10e9, fwidth=1.5e9)
sheet = CurrentSource(                   # s-polarized: current along y
    src_time, component="y", center=(px / 2, py / 2, 0.2), size=(px, py, 0.0),
    amp_func=lambda x, y, z: np.exp(1j * (k_par[0] * x + k_par[1] * y)),
)
freqs = np.linspace(8e9, 12e9, 41)
trans = wakis.FluxMonitor(freqs, center=(px / 2, py / 2, -0.2), size=(px, py, 0.0))
solver.emsolve(Nt, source=sheet, monitors=[trans])

T = trans.get_flux() / flux_reference      # reference: same run without the structure
theta = incidence_angle(k_par, freqs)      # the angle changes with frequency!
```

Points to keep in mind (details in the [physics guide](physicsguide.md)):
* `k_par` is fixed, so each frequency is hit at a different angle $\theta(f) = \arcsin(c|k_\parallel|/2\pi f)$. Below the cutoff $c|k_\parallel|/2\pi$ the wave is evanescent. Keep the source band clear of the cutoff, where waves barely leave the cell.
* Use frequencies $f>0$. With Wakis' Fourier convention, the $f>0$ wave has tangential wavevector $-\mathbf{k}_\parallel$ (its mirror image, at $-f$, is excited too). Reflectance, transmittance and band frequencies are the same for both.
* The physical time-domain field is `np.real(solver.E.toarray())`. Plotting routines expect real arrays.
* `tests/test_024_bloch.py` shows complete, validated set-ups: oblique Fresnel reflection, and photonic-crystal bands from a broadband pulse with spectral peak finding.


#### Dispersive materials: gold film reflectance

Frequency-dependent materials (`DispersiveMaterial`: Drude, Lorentz and Debye poles plus conductivity, see the physics guide) are added to box regions with `solver.add_dispersive_material` or assigned to STL solids directly in `stl_materials`. `dispersive_lib` provides gold, silver, aluminium (Rakić Lorentz–Drude), fused silica, silicon (Sellmeier), FR-4 and RO4350B (Debye). The reflectance of an optically thick gold film at normal incidence, in a quasi-1D periodic cell, normalized by a run without the film:

#### Waveguide modes and mode ports

`solve_modes` computes the propagating modes of a waveguide cross-section with the solver's own grid, materials and boundaries. `ModeSource` launches one mode in one direction (an electric plus a `MagneticCurrentSource`-type sheet), and `ModeMonitor` measures forward/backward mode amplitudes, from which S-parameters follow. A port is a plane: `size` has one zero entry (the normal) and `np.inf` (or the domain size) transversally; a smaller transverse size puts electric walls on the port edges, e.g. to exclude CPML layers.

```python
import numpy as np
from scipy.constants import c
import wakis
from wakis.materials import dispersive_lib
from wakis.sources import CurrentSource, GaussianSource

gold = dispersive_lib["gold"]
freqs = c / np.linspace(450e-9, 800e-9, 8)

def reflected_flux(material=None, dz=2e-9, thickness=100e-9):
    grid = wakis.GridFIT3D(-5e-9, 5e-9, -5e-9, 5e-9, -1e-6, 0.6e-6, 1, 1, int(1.6e-6 / dz))
    solver = wakis.SolverFIT3D(grid, bc_low=["periodic", "periodic", "cpml"],
                               bc_high=["periodic", "periodic", "cpml"], n_pml=20)
    if material is not None:  # film between z = 0 and z = thickness
        solver.add_dispersive_material(material, center=(0, 0, thickness / 2),
                                       size=(1, 1, thickness))
    pulse = GaussianSource(frequency=c / 600e-9, fwidth=0.6 * c / 600e-9)
    sheet = CurrentSource(pulse, component="x", center=(0, 0, -0.6e-6), size=(10e-9, 10e-9, 0))
    plane = wakis.FluxMonitor(freqs, center=(0, 0, -0.3e-6), size=(10e-9, 10e-9, 0))
    solver.emsolve(int((pulse.end_time + 30e-15) / solver.dt), source=sheet, monitors=[plane])
    return plane.get_flux()

R = 1 - reflected_flux(gold) / reflected_flux()

n = np.sqrt(gold.epsilon(freqs))         # exp(+iwt) convention: Im(eps) <= 0
R_fresnel = np.abs((1 - n) / (1 + n)) ** 2  # agrees with R to < 1% (2 nm cells)
```

Tabulated data can be turned into a passive pole model with `fit_poles`, e.g. `fit_poles(f, eps_from_nk(n, k), n_lorentz=2, n_drude=1)`; `material.epsilon(f)` evaluates the model and `material.epsilon_discrete(f, solver.dt)` the permittivity actually realised by the time stepping.

#### Lumped ports, lumped elements and S-parameters

A `LumpedPort` is a discrete port: a voltage source with internal impedance $Z_0$ (default 50 Ω) across a gap between two conductors, like the discrete port of CST. A `LumpedElement` is a resistor, inductor and/or capacitor across such a gap. Both are given by a box of edges in physical coordinates: along `component` the box must span the gap exactly, from one conductor to the other; transversally, a zero size gives a single line of edges and a finite size a sheet. The positive port terminal is at the high-coordinate end of the gap. Pass them to `emsolve` with the other sources.

The example computes S11 and the input impedance of a parallel-plate line with a 10 mm gap and 40 mm width ($Z = \eta_0 d/w = 94$ Ω), fed in the middle (so the port sees $Z/2 = 47$ Ω) and loaded with a 50 Ω resistor 1 cm away, then exports a Touchstone file:
```python
import numpy as np
import wakis
from wakis.sources import GaussianSource
from wakis.sparameters import write_touchstone

# PEC plates at x = 0 and x = 10 mm (the high PEC wall sits on the last x node)
grid = wakis.GridFIT3D(0, 12e-3, -20e-3, 20e-3, -0.06, 0.06, 6, 2, 60)
solver = wakis.SolverFIT3D(grid, bc_low=["pec", "periodic", "cpml"],
                           bc_high=["pec", "periodic", "cpml"])

gap = dict(component="x", size=(10e-3, 40e-3, 0.0))  # full gap, full width
port = wakis.LumpedPort(GaussianSource(3e9, 1.5e9), center=(5e-3, 0, 0), impedance=50, **gap)
load = wakis.LumpedElement(center=(5e-3, 0, 0.01), R=50, **gap)

# Stop when the field energy has decayed by 40 dB (at most 10000 steps)
solver.emsolve(10000, source=[port, load], stop_when_energy_decayed=1e-4)

freqs = np.linspace(1e9, 5e9, 41)
s11 = port.s11(freqs)               # (Zin - Z0)/(Zin + Z0)
z_in = port.impedance_in(freqs)     # V/I [Ohm]
a, b = port.power_waves(freqs)      # incident/reflected power waves
write_touchstone("line.s1p", freqs, s11, z0=50)
```

Use a Gaussian pulse whose spectrum covers the band of interest and at least about 20 cells per wavelength at the highest frequency. For a multi-port S-matrix, `wakis.sparameters.s_matrix` runs one simulation per port from a function that builds the solver, driving one port at a time and terminating the others in $Z_0$:
```python
from wakis.sparameters import s_matrix, to_skrf_network

def make_solver():
    return wakis.SolverFIT3D(grid, bc_low=["pec", "periodic", "cpml"],
                             bc_high=["pec", "periodic", "cpml"], verbose=0)

ports = [wakis.LumpedPort(center=(5e-3, 0, -0.01), **gap),
         wakis.LumpedPort(center=(5e-3, 0, 0.01), **gap)]
S = s_matrix(make_solver, ports, GaussianSource(3e9, 1.5e9), 10000, freqs,
             stop_when_energy_decayed=1e-4)   # shape (n_freq, 2, 2)
write_touchstone("line.s2p", freqs, S)
network = to_skrf_network(freqs, S)          # requires scikit-rf
```

The lumped elements correct their edges after each step through `post_step(solver, t)`, which `emsolve` calls automatically; in a manual loop call `element.update(solver, t)` before and `element.post_step(solver, t)` after `solver.one_step()`.



import wakis
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.sources import GaussianSource

# WR-like PEC guide along z with CPML ends: 21 x 11 cells -> a = 20 mm, b = 10 mm
# (PEC walls sit on the first and last field node: width = (N - 1) * dx)
dx = 1e-3
grid = wakis.GridFIT3D(0, 21 * dx, 0, 11 * dx, 0, 0.16, 21, 11, 160)
solver = wakis.SolverFIT3D(grid, bc_low=["pec", "pec", "cpml"], bc_high=["pec", "pec", "cpml"])

fc = c / (2 * 20 * dx)  # TE10 cutoff
port = dict(size=(np.inf, np.inf, 0))
te10 = solve_modes(solver, 1.5 * fc, center=(0, 0, 0.05), n_modes=2, **port)[0]
print(te10.neff, te10.beta)  # te10.e / te10.h: unit-power transverse profiles

pulse = GaussianSource(frequency=1.5 * fc, fwidth=0.15 * fc)
source = ModeSource(pulse, center=(0, 0, 0.05), direction="+", mode_index=0, **port)
freqs = np.linspace(1.3 * fc, 1.7 * fc, 21)
port1 = ModeMonitor(freqs, center=(0, 0, 0.03), **port)   # behind the source
port2 = ModeMonitor(freqs, center=(0, 0, 0.11), **port)   # downstream

solver.emsolve(int((pulse.end_time + 3e-9) / solver.dt), source=source, monitors=[port1, port2])

S = pulse.fourier_transform(freqs)
a_fwd, a_bwd = port2.get_amplitudes(mode_index=0)  # [sqrt(W) s]
P_mode = np.abs(a_fwd) ** 2 / np.abs(S) ** 2       # ~1 W: power launched for amplitude 1
```

With a device between the ports, $S_{21}(f) = a_+^{(2)}/(A\,S(f))$ and $S_{11}(f) = a_-^{(1)}/(A\,S(f))$ (or ratios to a reference run). The source is exactly unidirectional at its mode frequency; with broadband pulses in dispersive guides a small backward wave leaks away from it (a few % at $\pm 10\%$ of the frequency for TE10 at $1.5f_c$), so measure reflections with `ModeMonitor` rather than assuming a perfect launch.

Magnetic current sources can also be used on their own, e.g. a Huygens plane-wave sheet radiating only along $+z$ (the magnetic sheet sits on the H plane half a cell behind the electric sheet, and the incident H is advanced by $\Delta z/2c$):

```python
from scipy.constants import epsilon_0, mu_0
from wakis.sources import CurrentSource, CustomSource, MagneticCurrentSource

eta0 = np.sqrt(mu_0 / epsilon_0)
pulse = GaussianSource(frequency=1e9, fwidth=0.4e9)
sheet = dict(size=(xmax - xmin, ymax - ymin, 0.0))
K = CurrentSource(CustomSource(lambda t: pulse(t + dz / (2 * c))), component="x",
                  center=(0, 0, 0), amplitude=-1 / eta0, **sheet)        # K = z x H_inc
M = MagneticCurrentSource(pulse, component="y", center=(0, 0, -dz / 2),
                          amplitude=-1.0, **sheet)                       # M = -z x E_inc
solver.emsolve(Nt, source=[K, M])
```

Custom sources provide an `update(solver, t)` method. For example:
```python
class MySource:
    def __init__(self, frequency):
        self.frequency = frequency

    def update(self, solver, t):
        solver.J[solver.Nx // 2, solver.Ny // 2, :, 'z'] = np.sin(
            2 * np.pi * self.frequency * t
        )

solver.emsolve(Nt=100, source=MySource(frequency=1e9))
```

Combining `wakis` sources, geometry capabilities and material tensors, many different physical phenomena can be simulated with `wakis`: laser pulses, interaction with plasma, waveguides... For particle accelerators, the `Beam` class was created but the profile and trajectory can be easily modified to simulate advanced impedance effects.


## 🕹️ Using `wakis` as a Wakefield solver for beam-coupling impedance

### A few words about beam-coupling impedance and wakefields

The determination of electromagnetic wakefields and their impact on accelerator performance is a significant issue in current accelerator components. These wakefields, which are generated within the accelerator vacuum chamber as a result of the interaction between the structure and a passing beam, can have significant effects on the machine.

These effects can be characterized through the beam coupling impedance in the frequency domain, and wake potential in the time domain. Accurate evaluation of these properties is essential for predicting dissipated power and maintaining beam stability. `wakis` project was conceived at CERN, in the ABP-CEI group, to provide the accelerator community with a python based, open-source tool able to compute the beam-coupling impedance for present and future accelerator components.

### Setting up the wake solver

`wakis` can compute wake potential and impedance for both longitudinal and transverse planes for general 3D structures. The wake computation is performed right after the electromagnetic simulation, and the dedicated routines are encapsulated in the class `WakeSolver`, inside `wakeSolver.py`. The recipe on how to instantiate the `WakeSolver`class is:

```python
wake = WakeSolver(
    q=1e-9,              # total beam charge [C]
    sigmaz=0.02,         # longitudinal beam size [m]
    beta=1.0,
    xsource=0.0,
    ysource=0.0,
    xtest=0.0,
    ytest=0.0,
    skip_cells=10,
    save=True,
    results_folder='results/',
)
```

As can be deduced from the instantiation, the `wake` object contains the information of the beam source. When `wake` is passed to the `solver` object, a `Beam` source will be automatically added using `wake` attributes.

```{tip}
The `skip_cells` parameter excludes the specified number of cells at each end of the wake integration path. For example, `skip_cells=10` can avoid perturbations near the beam injection or reflections from the domain boundaries. The older `add_space` argument to `wakesolve` remains available for compatibility.
```

### Running a wakefield simulation
A wakefield simulation can be run using `solver.wakesolve()`. It advances the electromagnetic fields until the requested `wakelength` is reached, saves the longitudinal electric field to the wake solver's `Ez_file`, and computes wake potentials and impedances. Reset the fields if reusing the solver from the preceding examples:

```python
solver.reset_fields()
solver.wakesolve(
    wakelength=1.0,  # simulated wake length [m]
    wake=wake,
    plot=False,
)
```
### Recomputing some magnitudes
Once the simulation is finished and the data is safely stored in the HDF5 file, any result of wake potential and/or impedance can be re-computed and optimized. For instance, the longitudinal or transverse impedance can be recomputed using a higher number of samples or a different maximum frequency:

```python
# re-calculate longitudinal impedance from the wake.WP result
wake.calc_long_Z(samples=10001, fmax=2e9)
plt.plot(wake.f, np.real(wake.Z)) #plot real part
plt.plot(wake.f, np.imag(wake.Z)) #plot imaginary part
plt.plot(wake.f, np.abs(wake.Z))  #plot absolute

# re-calculate transverse impedance from the wake.WPx and WPy result
wake.calc_trans_Z(samples=10001, fmax=2e9)
plt.plot(wake.f, np.abs(wake.Zx)) #plot absolute of Zx
plt.plot(wake.f, np.abs(wake.Zy)) #plot absolute of Zy
```

## 📈 Extrapolate a partially decayed wake with [`IDDEFIX`](https://github.com/ImpedanCEI/IDDEFIX)
IDDEFIX is a physics-informed evolutionary optimization framework that fits a resonator-based model (parameterized by R, f, Q) to wakefield simulation data. It leverages Differential Evolution to optimize these parameters, enabling efficient classification and extrapolation of electromagnetic wakefield behavior. This allows for reduced simulation time while maintaining long-term accuracy, akin to time-series forecasting in machine learning. Developed by Sebastien Joly and Malthe Raschke, it is part of the Wakis ecosystem of python packages developed at CERN, and available in the [ImpedanCEI](https://github.com/ImpedanCEI/) organization.

```{seealso}
Specific documentation for [`IDDEFIX`](https://github.com/ImpedanCEI/IDDEFIX) is available at [http://iddefix.readthedocs.io/](http://iddefix.readthedocs.io/)
```

### Import simulation results
This example assumes a separate 30 m wake run was saved to `results_wl30/`. Import IDDEFIX and load those Wakis results:
```python
import wakis
import iddefix

# Load partially decayed wake results
wake30 = wakis.WakeSolver()
wake30.load_results('results_wl30/')
wake_length = 30  # [m]

# Plot imported results
fig, ax = plt.subplots()
ax.plot(wake30.s, wake30.WP, c='tab:red', label='Wake length = 30 m')
ax.set_xlabel('s [m]')
```

One can recompute impedance from the partially decayed wakes using `iddefix.compute_fft()` and `compute_deconvolution()` routines. Note that `Wakis` simulations compute the **Wake potential**, since the excitation is not a delta but a distributed current source (gaussian bunch). Therefore, to get the impedance, a deconvolution is needed:
```python
fig, ax = plt.subplots()

# Plot wakis impedance
ax.plot(wake30.f, np.abs(wake30.Z), c='k', lw=2, label='Impedance from Wakis')

# Plot iddefix FFT and deconvolution results
f, WP_fft = iddefix.compute_fft(wake30.s/c_light, wake30.WP*1e12/c_light, fmax=1.5e9)
ax.plot(f, np.abs(WP_fft), c='tab:red', alpha=0.7, label='Impedance from FFT')
f, Z = iddefix.compute_deconvolution(wake30.s/c_light, wake30.WP*1e12/c_light, fmax=1.5e9, sigma=10e-2/c_light)
ax.plot(f, np.abs(Z), c='tab:green', alpha=0.7, label='Impedance from deconvolution')
ax.set_xlabel('frequency [Hz]')
ax.legend()
```

### Parameter bounds determination
To estimate the parameter bounds of the resonators in the impedance data, one can use the `SmartBoundDetermination` class:
```python
# Compute impedance through deconvolution
# To improve algorithm speed and convergence, it is advised to keep the data to about 1000 samples
f,Z = iddefix.compute_deconvolution(wake30.s/c_light, wake30.WP*1e12/c_light, samples=1000, fmax=1.2e9, sigma=10e-2/c_light)
Z *= -1.0 # longitudinal impedance normalization

# Control the heigths to be passed to the peak finder routine
heights = np.zeros_like(Z)
heights[:] = 450
heights[np.logical_and(f>0.70e9,f<0.8e9)] = 3000

bounds = iddefix.SmartBoundDetermination(f, np.real(Z),
                                        Rs_bounds=[0.8, 10],    # bound multipliers for peak Rs
                                        Q_bounds=[0.5, 5],      # bound multipliers for estimated Q
                                        fres_bounds=[-0.01e9, +0.01e9]) # bound margins for estimated fres

bounds.find(minimum_peak_height=heights, distance=10 )
bounds.inspect()        # plot the found peaks and estimated Q
bounds.to_table()       # print as a table the estimated bounds
```

### Run differential evolution (DE) and minimization
Now it is time to pass the data to the `EvolutionaryAlgorithm` class. The available fit founctions use the Broadband Resonator Formalism, and the evolutionary algorithm find the list of parameters (Rs, Q, fr) that better describe the impedance. To run the Differential Evolution algorithm, follow:
```python
DE_model = iddefix.EvolutionaryAlgorithm(f,
                                         Z.real,
                                         N_resonators=bounds.N_resonators,
                                         parameterBounds=bounds.parameterBounds,
                                         plane='longitudinal',
                                         fitFunction='impedance',
                                         wake_length=wake_length, # in [m]
                                         objectiveFunction=iddefix.ObjectiveFunctions.sumOfSquaredErrorReal
                                         )

# Run the differential evolution
DE_model.run_differential_evolution(maxiter=30000,
                                    popsize=150,
                                    tol=0.001,
                                    mutation=(0.3, 0.8),
                                    crossover_rate=0.5)
print(DE_model.warning)
```

Additionally, we can ran a second-step optimization using the Nelder-Mead minimization for local refinement:
```python
DE_model.run_minimization_algorithm()
```

### Asses the DE fitting
To asses the fitting, one can compare the partially decayed with the analytically fitted one:
```python
#%matplotlib ipympl

# Retrieve partially decayed wake portential (30 m)
WP_pd = DE_model.get_wake_potential(wake30.s/c_light)

# Retreieve partially decayed  fittted impedance
f_pd = np.linspace(0, 1.2e9, 10000)
Z_pd = DE_model.get_impedance_from_fitFunction(f_pd, use_minimization=False)  # use only evolutionary parameters
Z_pd_min = DE_model.get_impedance_from_fitFunction(f_pd, use_minimization=True) # use minimization parameters

# Plot comparison
fig1, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
ax[0].plot(wake30.s, wake30.WP, c='k', alpha=0.8,label='Wakis wl=30 m')
ax[0].plot(wake30.s, -WP_pd*1e-12, c='tab:red', lw=1.5, label='iddefix')
ax[0].set_xlabel('s [m]')
ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='tab:red')
ax[0].legend()

#ax[1].plot(wake30.f*1e-9, np.abs(wake30.Z), c='k', label='Wakis wl=30 m')
ax[1].plot(wake30.f*1e-9, np.real(wake30.Z), ls='-', c='k', lw=1.5, label=' Wakis wl=30 m Real')
#ax[1].plot(wake30.f*1e-9, np.imag(wake30.Z), ls=':', c='k', lw=1.5, label='Imag')

ax[1].plot(f_pd*1e-9, np.real(Z_pd), ls='-', c='tab:blue', alpha=0.8, lw=1.5, label='DE wl=30 m Real')
#ax[1].plot(f_pd*1e-9, np.imag(Z_pd), ls=':', c='tab:blue', alpha=0.6, lw=1.5, label='DE Imag')

ax[1].plot(f_pd*1e-9, np.real(Z_pd_min), ls='-', c='tab:red', alpha=0.6, lw=1.5, label='DE+min wl=30 m Real')
#ax[1].plot(f_pd*1e-9, np.imag(Z_pd_min), ls=':', c='tab:red', alpha=0.6, lw=1.5, label='DE+min Imag')

ax[1].set_xlabel('f [GHz]')
ax[1].set_ylabel('Longitudinal impedance [Ω]', color='tab:blue')
ax[1].legend()

fig1.tight_layout()
```

### Fully decayed impedance
Once the fitting is satisfactory, the fully decayed impedance can be computed via de Broadband resonator formalism using the (Rs, Q, fr) parameters fitted with the differential evolution:

```python
# Fully decayed wake, wakelength 1000 m
t_fd = np.linspace(wake30.s[0]/c_light, 1000/c_light, 10000)
WP_fd = DE_model.get_wake_potential(t_fd, sigma=1e-2/c_light)

# Fully decayed wake, wakelength inf m
f_fd = np.linspace(0, 1.5e9, 10000)
Z_fd = DE_model.get_impedance(f_fd)

# Plot
fig1, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
ax[0].plot(t_fd*c_light, WP_fd*1e-12*c_light, c='tab:red', lw=1.5, label='iddefix')
ax[0].set_xlabel('s [m]')
ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='tab:red')
ax[0].legend()

ax[1].plot(f_fd*1e-9, np.abs(Z_fd), c='tab:blue', alpha=0.8, lw=2, label='Fully decayed, Abs')
ax[1].plot(f_fd*1e-9, np.real(Z_fd), ls='--', c='tab:blue', lw=1.5, label='Fully decayed, Real')
ax[1].plot(f_fd*1e-9, np.imag(Z_fd), ls=':', c='tab:blue', lw=1.5, label='Fully decayed, Imag')
ax[1].set_xlabel('f [GHz]')
ax[1].set_ylabel('Longitudinal impedance [Ω]', color='tab:blue')
ax[1].legend()

fig1.tight_layout()
```

## 🧮 Compute non-equidistant Fourier Transforms with [`neffint`](https://github.com/ImpedanCEI/neffint)
Neffint is an acronym for Non-equidistant Filon Fourier integration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing. Developed by Eskil Vik and Nicolas Mounet, it is part of the Wakis ecosystem of python packages developed at CERN, and available in the [ImpedanCEI](https://github.com/ImpedanCEI/) organization.

`neffint` has been integrated in `IDDEFIX` as an alternative method to compute Fourier Transforms:
```python
time, wake_function = iddefix.compute_ineffint(wake30.f, wake30.Z,
                                 times=np.linspace(1e-11, 50e-9, 1000), #avoid starting at zero
                                 plane='longitudinal',
                                 adaptative=True, # refines the sampling, but can be slow/unstable
                                 )

frequency, wake_spectrum = iddefix.compute_neffint(
                                 wake30.s / c_light, wake30.WP * 1e12 / c_light,
                                 frequencies=np.linspace(0, 5e9, 1000),
                                 plane='longitudinal',
                                 adaptative=True, # refines the sampling, but can be slow/unstable
                                 )
```

```{seealso}
Check out `IDDEFIX`'s [test_001](https://github.com/ImpedanCEI/IDDEFIX/blob/main/tests/001_test_fft.py) for a comparison of all the different FFT methods available
```

## 🌡️ Estimate the beam induced heating of the simulated impedance with [`BIHC`](https://github.com/ImpedanCEI/BIHC)
Beam Induced Heating Computation (BIHC) tool is a package that allows the estimation of the dissipated power due to the passage of a particle beam inside an accelerator component. The dissipated power value depends on the characteristics of the particle beam (**beam spectrum and intensity**) and on the characteristics of the consdiered accelerator component (**beam-coupling impedance**). BIHC helps generating different beam filling schemes, bunch profiles, bunch intensities, and load machine parameters for the different accelerators at CERN.

```{seealso}
Specific documentation for [`BIHC`](https://github.com/ImpedanCEI/BIHC) is available at [https://bihc.readthedocs.io/](https://bihc.readthedocs.io/)
```

### Generate beam parameters
To generate the beam spectrum, we need the filling scheme used in the machine, containing the number and spacing between bunches. An example of a CERN LHC filling scheme is given bellow:

```python
def fillingSchemeLHC(ninj, ntrain=5, nbunches=36):
    '''
    Returns the filling scheme for the LHC
    using the standard pattern

    Parameters
    ----------
    ninj: number of injections (batches)
    '''

    # Define filling scheme: parameters
    #ninj = 11 # Defining number of injections
    nslots = 3564 # Defining total number of slots for LHC
    #ntrain = 5 # Defining the number of trains
    #nbunches = 36 # Defining a number of bunchs e.g. 18, 36, 72..
    batchS = 7 # Batch spacing in 25 ns slots
    injspacing = 37 # Injection spacing in 25 ns slots

    # Defining the trains as lists of True/Falses
    bt = [True]*nbunches
    st = [False]*batchS
    stt = [False]*injspacing
    sc = [False]*(nslots-(ntrain*nbunches*ninj+((ntrain-1)*(batchS)*ninj)+((1)*injspacing*(ninj))))
    an1 = bt+ st +bt+ st+ bt+ st+ bt+ stt
    an = an1 * ninj + sc # This is the final true false sequence that is the beam distribution

    return an
```
With this information, we can use `BIHC` to fill the `Beam` class:
```python
import bihc

# Create beam object
fillingScheme = fillingSchemeLHC(ninj=9, ntrain=4, nbunches=72)
bl = 1.2e-9                 # bunch length [s]
Np = 2.3e11                 # bunch intensity [protons/bunch]
bunchShape = 'q-GAUSSIAN'   # bunch profile shape in time
qvalue = 3/5                # value of q parameter in the q-gaussian distribution
fillMode = 'FLATTOP'        # Energy
fmax = 2e9                  # Maximum frequency of the beam spectrum [Hz]

beam = bihc.Beam(Np=Np, bunchLength=bl, fillingScheme=fillingScheme,
                bunchShape=bunchShape, qvalue=qvalue,
                machine='LHC', fillMode=fillMode, spectrum='numeric', fmax=fmax)

print(f'* Number of bunches used: {np.sum(fillingScheme)}')
print(f'* Total intensity: {np.sum(fillingScheme)*Np:.2e} protons')
```
And plot the beam longitudinal profile (time-domain) and beam spectrum (frequency domain):
```python
fig, ax = plt.subplots(1,2, figsize=[14,6])

t, prof = beam.longitudinalProfile
ax[0].plot(t*1e6, prof*beam.Np,)
ax[0].set_xlabel('Time [µs]')
ax[0].set_ylabel('Profile Intensity [protons]')

f, spectrum = beam.spectrum
ax[1].plot(f*1e-9, spectrum*beam.Np*np.sum(fillingScheme), c='r')
ax[1].set_xlabel('Frequency [GHz]')
ax[1].set_ylabel('Spectrum Intensity [protons]')
ax[1].set_xlim((0, 2.0))
```

### The impedance object
To compute the power loss, we need to fill the `Impedance` class with the impedance data of the accelerator device under study:
```python
Z = bihc.Impedance(f=wake30.f, Z=wake30.Z)

# To load a CST file instead, use:
# Z = bihc.Impedance(CST_file='impedance.txt')
```

### Power loss calculation, 1 beam case
With `BIHC` we can simply calculate the power loss by `beam.getPloss(Z)`. However, due to inacuracies in the wakefield simulation or the CAD model, or to account for changes in the revolution frequency during operation, `BIHC` also performs a statistical analysis by rigidly shifting the impedance curve in piece-wise shifts `beam.getShiftedPloss(Z, shift=shift)` to account for different overlaps with the beam spectral lines.
A basic **power loss calculation** can be done by:
```python
print('Calculate beam-induced power loss')
print('---------------------------------')
# Get unshifted ploss
ploss, ploss_density = beam.getPloss(Z)
print(f'Dissipated power (no-shift): {ploss:.3} W')

# Get min/max power loss with rigid shift
shift = 20e6  # distance between shift steps [Hz]
shifts, power = beam.getShiftedPloss(Z, shift=shift)

print(f'Minimum dissipated power: P_min = {np.min(power):.3} W, at step {shifts[np.argmin(power)]}')
print(f'Maximum dissipated power: P_max = {np.max(power):.3} W, at step {shifts[np.argmax(power)]}')
print(f'Average dissipated power: P_mean = {np.mean(power):.3} W')

# Retrieve impedance that gave the maximum Ploss
Z_max = beam.Zmax
```
One can also plot the **power loss density** across the frequencies of interest:
```python
# Unshifted impedance
ploss, ploss_density = beam.getPloss(Z)

# Shifted impedance
ploss_max, ploss_density_max = beam.getPloss(Z_max)

fig, ax = plt.subplots(figsize=[10,7])
l1, = ax.plot(np.linspace(0, Z_max.f.max()/1e9, len(ploss_density_max )), ploss_density_max , color='r', marker='v', lw=3, alpha=0.8)
l0, = ax.plot(np.linspace(0, Z.f.max()/1e9, len(ploss_density )), ploss_density , color='k', marker='v', lw=3, alpha=0.8)

ax.set_ylabel('Power by frequency [W]', color='k')
ax.set_yscale('log')
ax.set_xlabel('Frequency [GHz]')
ax.set_xlim((0, 1.5))
ax.set_ylim(ymin=1e-1, ymax=1e4)
ax.grid(which='minor', axis='y', alpha=0.8, ls=':')
ax.legend([l0, l1], ['Power loss', 'Maximum shifted power loss'], loc=1)
```

### Power loss calculation, 2 counter-rotating beams
If the accelerator component of interested was placed in a collider's common-beam chamber, it would see the effect of **2 beam power loss**. The beam-induced heating in this case is a function of the distance with the interaction point (IP) and can be greater than a factor 2 of the 1-beam case. We can compute this with `BIHC` too:

```python
#       2 beam case
# ----------------------
# Defining the phase shift array for LHC
c = 299792458 # Speed of light in vacuum [m/s]
ring_circumference =  26658.883   #[m]

start = -3.5 #m
stop = 3.5 #m
resolution = 0.001 #m power2b

s = np.arange(start, stop, resolution)
tau_s = 2*s/c # Phase shift array [s]

power2b = beam.get2BeamPloss(Z, tau_s=tau_s)
power2b_max = beam.get2BeamPloss(Z_max, tau_s=tau_s)

# Plot power los vs distance from IP
fig, ax = plt.subplots(figsize=[12,5])

ax.plot(s, power2b_max, label="2-b Ploss Max.", c='b', ls='-', alpha=0.7)
ax.plot(s, power2b, label="2-b Ploss ", c='deepskyblue', ls='-', alpha=0.7)

ax.set_ylabel('Dissipated power [W]')
ax.set_xlabel('s Distance from IP [m]')

ax.axhline(np.max(power), c='b', ls='--', alpha=0.5, label='Max. 1-b power')
ax.axhline(np.mean(power), c='deepskyblue', ls='--', alpha=0.5, label='1-b power')

ax.set_ylim(ymin=0)
ax.set_title(f'2 beam power loss vs s')
fig.legend(bbox_to_anchor=(0.55, 0.0), fontsize=14, loc='lower center', ncol=2)
fig.tight_layout(rect=[0, 0.2, 1, 1])
```
