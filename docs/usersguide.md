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

Non-periodic boundaries (PEC, PMC, ABC, CPML) sit on the domain bounds: a PEC box from `xmin` to `xmax` meshed with `Nx` cells is exactly `xmax - xmin` wide. For this, `GridFIT3D` appends one ghost cell beyond each max bound, so the field arrays hold `Nx + 1` samples per axis (`grid.Nx` is the number of samples) and index `-1` is the boundary node at `xmax`. Pass `boundary_at_domain_edge=False` to recover the legacy convention of earlier Wakis versions (high-side walls one cell inside the domain), e.g. to reproduce published wake results; see *Grid conventions* in the physics guide.

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

The adjoint supports PEC, PMC, periodic, Bloch (complex fields) and CPML boundaries, ohmic loss, dispersive materials, lumped elements and ports, and electric, magnetic and mode sources (`CurrentSource`, `MagneticCurrentSource`, `ModeSource`, `LumpedPort`); ABC, PML, TFSF and SIBC raise `NotImplementedError`. The design region must not share edges with lumped elements or dispersive materials. `AdjointSimulation(..., store_dtype=np.float32)` halves the memory of the stored E increments ($N_t \times N_\text{design}$ values).

**GPU and MPI.** The adjoint runs wherever the forward solver runs: build the solver with `use_gpu=True` in `make_solver` and both runs, the stored increments and the gradient accumulation stay on the GPU (`store_on_host=True` keeps the increments in host memory for long runs on large designs; by default this happens when they would take more than half of the free GPU memory). For MPI, build grid and solver with `use_mpi=True` and launch the whole optimization script with `mpiexec -n 4 python design.py`: every rank receives the same gathered outputs and the same gradient, so the optimizer loop needs no MPI calls. Under MPI the adjoint supports the standard time step (no CPML, lumped elements or dispersive media) with `DFTMonitor` and `FluxMonitor` objectives:

```python
def make_solver():  # z split into slabs over the MPI ranks
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.15, 0.15, 64, 64, 96, use_mpi=True, verbose=0)
    return wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_mpi=True, verbose=0)

sim = AdjointSimulation(make_solver, [source], {"flux": wakis.FluxMonitor([f0], size=(0.15, 0.15, 0))},
                        MaterialGrid(center=(0, 0, 0), size=(0.06, 0.06, 0.18), shape=(7, 7, 21)), Nt=2000)
sim.output_shapes()  # gathers the flux weights into sim.readouts (for wakis.inverse.flux)
```

See [Performance](performance.md) for forward and adjoint timings on CPU, GPU and MPI.

**Objectives on any monitor.** `forward` returns the *raw* DFT accumulators of each monitor, and the helpers of `wakis.inverse` turn them into physical quantities inside JAX, identical to the monitors' own post-processing:

| Monitor | raw outputs | helper |
|---|---|---|
| `DFTMonitor` | one array per component | – |
| `FluxMonitor` | `'E0'`, `'H0'`, ... per surface term | `flux(raw, sim.readouts[name])` = `get_flux()` |
| `Near2FarMonitor` | as `FluxMonitor` | `farfield(raw, sim.readouts[name], theta, phi)` |
| `ModeMonitor` | `'E'`, `'H'` on the port plane | `mode_amplitudes(raw, sim.readouts[name], mode_index)` = `get_amplitudes()` |
| `LumpedReadout(element, freqs)` | `'V'`, `'I'` of a lumped element or port | `power_waves(V, I, z0)` |

`sim.readouts` holds monitor copies initialized on a solver without the design (created by `jax_simulation` or `sim.output_shapes()`); mode profiles are computed there once and are not differentiated, so mode planes must lie outside the design region. A `LumpedReadout` refers to an element that is also passed in `sources`; for example $|S_{21}|^2$ of a two-port is `jnp.abs(b2 / a1) ** 2` with `a1, _ = power_waves(out["p1"]["V"], out["p1"]["I"], 50)` and `_, b2 = power_waves(out["p2"]["V"], out["p2"]["I"], 50)`. `MaterialGrid(..., material=disp)` designs the fill fraction of a `DispersiveMaterial` over the background `eps_min` instead of a permittivity.

**Example: waveguide offset coupler.** Two parallel slab waveguides ($\varepsilon_r = 6$, three cells wide) are offset by 4 cm; an 8 × 8 cm design region between them is optimized to maximize the power transmitted into the fundamental mode of the output guide. `ModeSource` launches the unit-power mode with spectrum $S(f)$, so $T = |a_+|^2/|S|^2$ is the power transmission:

```python
import jax
import jax.numpy as jnp
import numpy as np
import wakis
from wakis.inverse import (AdjointSimulation, MaterialGrid, conic_filter, jax_simulation,
                           mode_amplitudes, tanh_projection)
from wakis.mode_solver import ModeMonitor, ModeSource
from wakis.sources import GaussianSource

jax.config.update("jax_enable_x64", True)
f0, dx, eps_core, offset = 3e9, 5e-3, 6.0, 0.02  # 20 cells per vacuum wavelength

def make_solver():  # 2D: 60 x 40 cells, one periodic cell along z
    grid = wakis.GridFIT3D(-0.15, 0.15, -0.1, 0.1, -dx / 2, dx / 2, 60, 40, 1, verbose=0)
    solver = wakis.SolverFIT3D(grid, bc_low=["cpml", "cpml", "periodic"],
                               bc_high=["cpml", "cpml", "periodic"], n_pml=10, verbose=0)
    for x0, y0 in ((-0.1, -offset), (0.1, offset)):  # fixed cores: MaterialGrid at rho = 1
        core = MaterialGrid(center=(x0, y0, 0), size=(0.12, 3 * dx, 0), shape=(1, 1, 1),
                            eps_min=eps_core, eps_max=eps_core)
        core.apply(solver, np.ones((1, 1, 1)))
    return solver

window = (0, 0.12, np.inf)  # port planes, excluding the CPML layers
source = ModeSource(GaussianSource(f0, 1e9), center=(-0.1, -offset, 0), size=window)
monitors = {"out": ModeMonitor([f0], center=(0.1, offset, 0), size=window)}
design = MaterialGrid(center=(0, 0, 0), size=(0.08, 0.08, 0), shape=(17, 17, 1),
                      eps_min=1.0, eps_max=eps_core)
sim = AdjointSimulation(make_solver, [source], monitors, design, Nt=1200)
simulate = jax_simulation(sim)
S = source.src_time.fourier_transform(f0)

def transmission(rho_raw, beta=4.0):
    rho = tanh_projection(conic_filter(rho_raw, radius=1.5), beta)
    a_plus, _ = mode_amplitudes(simulate(rho)["out"], sim.readouts["out"])
    return jnp.sum(jnp.abs(a_plus) ** 2) / jnp.abs(S) ** 2

rho = np.full(design.shape, 0.5)
for it in range(10):
    T, g = jax.value_and_grad(transmission)(rho)
    print(f"iteration {it}: T = {float(T):.3f}")
    rho = np.clip(rho + 0.1 * g / np.abs(g).max(), 0, 1)
```

Each iteration (forward + adjoint, 2400 cells × 1200 steps) takes ~0.5 s on a laptop CPU. The transmission rises from $T = 0.22$ for the uniform start to $T = 0.95$ after 10 iterations (0.22, 0.50, 0.71, 0.84, 0.91, 0.91, 0.92, 0.94, 0.94, 0.95); the fixed normalized step causes small oscillations near the optimum. For a manufacturable design, continue with increasing projection steepness `beta` (e.g. 8, 16, 32).

#### Far-field patterns and antenna metrics

`Near2FarMonitor` records the tangential fields on a closed box around the antenna and computes the far field, radiation intensity, directivity and gain at any angle after the run (CST's far-field monitor). The box must enclose all sources and scatterers and lie in a homogeneous region outside the CPML. The example below computes the pattern of a Hertzian dipole in free space (theory: $D = 1.5\sin^2\theta$, $D_{max} = 1.76$ dBi):

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

#### Periodic unit cells at oblique incidence (Bloch boundaries)

Metasurfaces, gratings and photonic crystals are simulated with one unit cell and Bloch-periodic boundaries. These are CST's "unit cell" boundaries, or Meep's `k_point`. Put `'bloch'` on the periodic axes and pass the Bloch wavevector `k_bloch` [rad/m]. A nonzero `k_bloch` needs complex fields (`dtype=np.complex128`). An oblique plane wave is launched by a sheet current with the matching phase profile $e^{i\mathbf{k}_\parallel\cdot\mathbf{r}}$:

```python
import numpy as np
import wakis
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


#### Triangular and hexagonal photonic crystals: `wakis.lattices`

`wakis.lattices` represents 2D triangular (Bravais) and honeycomb (two-sublattice) photonic crystals on Cartesian FIT meshes via their centred-rectangular $a \times \sqrt{3}a$ supercell, unfolds supercell bands onto the primitive hexagonal Brillouin zone along $\Gamma\text{–}M\text{–}K\text{–}\Gamma$, provides an independent 2D plane-wave-expansion (`PWE2D`) solver, and computes gauge-invariant Berry curvature and valley Chern numbers (physics guide: *Triangular and hexagonal lattices on Cartesian FIT grids*; full script: `examples/016_triangular_honeycomb_lattices.py`; tutorial: `tutorials/06_periodic_structures/06-06_valley_hall_edge_states/`):

```python
import numpy as np
from wakis.lattices import (
    HexagonalLattice,
    PWE2D,
    fit_berry_curvature_patch,
    fit_supercell_bands_at_k,
)

lat = HexagonalLattice(a=1.0)
path = lat.bz_path(["Gamma", "M", "K", "Gamma"], n_per_segment=12)

# Valley-Hall honeycomb crystal (r_A != r_B breaks inversion symmetry at K/K')
r0, delta, eps_rod = 0.18, 0.12, 12.0
r_a, r_b = r0 * (1.0 + delta), r0 * (1.0 - delta)
A1, A2 = lat.supercell_vectors
inc_sup = lat.honeycomb_supercell_sites(r_a, r_b, eps_in=eps_rod)

# 1. Supercell PWE bands unfolded onto the primitive hexagonal Brillouin zone
pwe_sup = PWE2D(A1, A2, inc_sup, eps_bg=1.0, n_pw=5, polarization="TM")
sup_bands = pwe_sup.solve_path(path.k_points, n_bands=8, unfold_supercell=True, lattice_a=1.0)
f_unfolded = sup_bands["unfolded_frequencies"]  # shape (n_k, 4)

# 2. 2D FIT + harminv supercell eigenfrequencies at K with tau-translation unfolding
k_K = lat.high_symmetry_points()["K"]
fit_k = fit_supercell_bands_at_k(
    lat, inc_sup, k_K, polarization="TM", nx=24, ny=42, eps_bg=1.0, f_min=0.22, f_max=0.35
)
print("Unfolded FIT bands at K:", fit_k["unfolded_frequencies"][:2])

# 3. Gauge-invariant Berry curvature & valley Chern sign from PWE or 2D FIT fields
q = np.linspace(-0.22 * 2 * np.pi, 0.22 * 2 * np.pi, 9)
omega_pwe, c_patch_pwe = pwe_sup.berry_curvature(k_K[0] + q, k_K[1] + q, band_index=0)
fit_berry = fit_berry_curvature_patch(
    lat, inc_sup, k_center=k_K, dk_half=0.18 * 2 * np.pi, n_k=3, f_target=0.271, nx=16, ny=28
)
print("Patch valley Chern at K:", c_patch_pwe, fit_berry["c_patch"])
```


#### Diffraction efficiencies of gratings and metasurfaces: `DiffractionMonitor`

`DiffractionMonitor(frequencies, center, size)` decomposes the field on a plane of a periodic (or Bloch) unit cell into its diffraction orders: complex amplitudes and powers of each propagating order $(m, n)$, split into waves going up (along $+$normal) and down, and into TE/TM or circular polarizations. With a monitor between the source and the structure (incident wave = up $(0,0)$, reflected orders = down) and one beyond it (transmitted orders = up), one run gives all reflection and transmission efficiencies. A lamellar grating (period $1.5\lambda_0$, $\varepsilon = 4$ teeth on glass) lit from the substrate at normal incidence:

```python
import numpy as np
from scipy.constants import c, epsilon_0
import wakis
from wakis.sources import CurrentSource, GaussianSource

lam = 0.3                       # free-space wavelength at f0 [m]
f0, P, H = c / lam, 1.5 * lam, 0.5 * lam
N = 120                         # cells per period: 80 per wavelength
dx = P / N
nz = round(2.7 * lam / dx)
grid = wakis.GridFIT3D(0, P, 0, dx, -1.1 * lam, -1.1 * lam + nz * dx, N, 1, nz)
solver = wakis.SolverFIT3D(grid, bc_low=["periodic", "periodic", "cpml"],
                           bc_high=["periodic", "periodic", "cpml"], n_pml=15)

def eps(x, z):                  # glass below z = 0, teeth for x < P/2, 0 < z < H
    tooth = (np.mod(x, P) < P / 2) & (z > 0) & (z < H)  # periodic in x
    return np.where(z < 0, 2.25, np.where(tooth, 4.0, 1.0))

# Edge permittivity: mean of the media touching each edge (edges on nodes see
# both sides), i.e. tests/test_033_diffraction.py::_set_grating
xn, zn = grid.x[:-1], grid.z[:-1]
xm, zm = xn + dx / 2, zn + np.diff(grid.z) / 2
d = 1e-9
for comp, (x, z) in {"x": (xm, zn), "y": (xn, zn), "z": (xn, zm)}.items():
    X, Z = np.meshgrid(x, z, indexing="ij")
    ox = [0.0] if comp == "x" else [-d, d]
    oz = [0.0] if comp == "z" else [-d, d]
    e = np.mean([eps(X + a, Z + b) for a in ox for b in oz], axis=0)
    solver.ieps[:, :, :, comp] = 1 / (epsilon_0 * e[:, None, :])

src = GaussianSource(frequency=f0, fwidth=0.15 * f0)
sheet = CurrentSource(src, component="y",            # TE: E along y ("x" for TM)
                      center=(P / 2, dx / 2, -0.7 * lam), size=(P, dx, 0))
refl = wakis.DiffractionMonitor([f0], center=(0, 0, -0.4 * lam), size=(P, dx, 0))
trans = wakis.DiffractionMonitor([f0], center=(0, 0, lam), size=(P, dx, 0))
solver.emsolve(int((src.end_time + 40 / f0) / solver.dt), source=sheet,
               monitors=[refl, trans])

i0 = list(refl.orders[0]).index(0)
incident = refl.get_power()["up"][:, i0, 0]          # incident flux (0, 0)
R = refl.get_power(incident)["down"][0, :, 0]        # efficiency per order m
T = trans.get_power(incident)["up"][0, :, 0]
print(dict(zip(refl.orders[0].tolist(), R.round(4).tolist())))   # {-2: 0.0278, -1: 0.1039, 0: 0.2161, ...}
print(dict(zip(trans.orders[0].tolist(), T.round(4).tolist())))  # {-1: 0.1981, 0: 0.1241, 1: 0.1981}
print(R.sum() + T.sum())                       # 1 - 2e-4
theta, phi = trans.order_angles()              # diffraction angles [rad]
amps = trans.get_amplitudes()                  # 'up_s', 'up_p', ... complex [sqrt(W) s]
circ = trans.get_amplitudes("circular")        # 'up_R', 'up_L', ...
```

fmmax (RCWA) gives $R = \{0.0276, 0.1050, 0.2148\}$ and $T = \{0.1980, 0.1241\}$ for this grating; the error decreases at second order with the mesh (physics guide, 'Diffraction orders'). Things to know:
* Order $(m,n)$ has the physical tangential wavevector $\mathbf{k}_\text{inc} + 2\pi(m/P_b, n/P_c)$, with $(b, c) = (x, y)$ for a $z$-normal plane. Arrays have shape `(n_freq, len(orders[0]), len(orders[1]))`; `propagating` flags the propagating orders (others are zero).
* At normal incidence TE means $E_y$ and TM $E_x$ (as in invrs-gym); otherwise TE is perpendicular to each order's plane of incidence. `'R'`/`'L'` are negative/positive helicity (optics convention, as invrs-gym's meta-atom library).
* Amplitude phases use Wakis' $e^{+2\pi i f t}$ phasors; conjugate them to compare with RCWA codes ($e^{-i\omega t}$), e.g. for meta-atom phase libraries.
* At oblique incidence use complex Bloch fields as above; the $f>0$ spectrum describes incidence with tangential wavevector $-\mathbf{k}_\text{bloch}$, so pass `k_bloch = -k_par` (and the source profile $e^{i\mathbf{k}_\text{bloch}\cdot\mathbf{r}}$) to describe $+\mathbf{k}_\parallel$.
* The medium at the monitor plane must be homogeneous and lossless, and the transverse mesh uniform.
* For inverse design, use the monitor in an `AdjointSimulation` and `wakis.inverse.diffraction_orders(raw, sim.readouts[name])` in the JAX objective, e.g. `jnp.abs(diffraction_orders(out["trans"], mon)["up_s"][:, i1, 0])**2 / incident`.

#### Dispersive materials: gold film reflectance

Frequency-dependent materials (`DispersiveMaterial`: Drude, Lorentz and Debye poles plus conductivity, see the physics guide) are added to box regions with `solver.add_dispersive_material` or assigned to STL solids directly in `stl_materials`. `dispersive_lib` provides gold, silver, aluminium (Rakić Lorentz–Drude), fused silica, silicon (Sellmeier), FR-4 and RO4350B (Debye). The reflectance of an optically thick gold film at normal incidence, in a quasi-1D periodic cell, normalized by a run without the film:

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

# PEC plates at x = 0 and x = 10 mm (the domain bounds)
grid = wakis.GridFIT3D(0, 10e-3, -20e-3, 20e-3, -0.06, 0.06, 5, 2, 60)
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

#### Waveguide modes and mode ports

`solve_modes` computes the propagating modes of a waveguide cross-section with the solver's own grid, materials and boundaries. `ModeSource` launches one mode in one direction (an electric plus a `MagneticCurrentSource`-type sheet), and `ModeMonitor` measures forward/backward mode amplitudes, from which S-parameters follow. A port is a plane: `size` has one zero entry (the normal) and `np.inf` (or the domain size) transversally; a smaller transverse size puts electric walls on the port edges, e.g. to exclude CPML layers.

```python
import numpy as np
from scipy.constants import c

import wakis
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.sources import GaussianSource

# WR-like PEC guide along z with CPML ends: 20 x 10 cells -> a = 20 mm, b = 10 mm
# (PEC walls sit on the domain bounds)
dx = 1e-3
grid = wakis.GridFIT3D(0, 20 * dx, 0, 10 * dx, 0, 0.16, 20, 10, 160)
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

#### Resonant frequencies and Q factors: `ResonanceMonitor` and `harminv`

Eigenmodes of a cavity are found from a single broadband run: excite with a pulse, record a field component after the pulse has ended, and fit the ring-down with harmonic inversion. The example computes the modes of a 100 × 80 × 60 mm box filled with a slightly conducting medium; each mode has $Q = \omega\varepsilon_0/\sigma$:
```python
import numpy as np
from scipy.constants import epsilon_0
import wakis
from wakis.sources import CurrentSource, GaussianSource

sigma = 1e-3  # S/m
grid = wakis.GridFIT3D(0, 0.10, 0, 0.08, 0, 0.06, 20, 16, 12)
solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3,
                           bg=[1.0, 1.0, sigma])

pulse = GaussianSource(frequency=3.5e9, fwidth=1.5e9)
dipole = CurrentSource(pulse, component="z", center=(0.031, 0.023, 0.017))
# Probe at a generic point, recording only the free ring-down
probe = wakis.ResonanceMonitor("Ez", center=(0.067, 0.051, 0.039), t_start=pulse.end_time)
solver.emsolve(int((pulse.end_time + 30e-9) / solver.dt), source=dipole, monitors=[probe])

modes = probe.harminv(2e9, 5e9)       # filter diagonalization in the 2-5 GHz window
for f, Q, err in zip(modes.frequency, modes.Q, modes.error):
    print(f"f = {f / 1e9:.6f} GHz  Q = {Q:.1f}  (omega eps0/sigma = "
          f"{2 * np.pi * f * epsilon_0 / sigma:.1f}, error ~ {err:.1e} Hz)")
```

`modes.complex_frequency` holds $\tilde f = f + i g$ with $Q = f/(2g)$ and `modes.amplitude` the complex amplitudes of $x(t) = \sum_k a_k e^{2\pi i\tilde f_k t}$. For open resonators (radiation $Q$), surround the structure with CPML and record for at least a few periods; a probe away from nodal lines and a source whose spectrum covers the window are needed to see every mode. `wakis.harminv(signal, dt, fmin, fmax, method="fdm" | "pencil")` works on any uniformly sampled signal, e.g. a wake potential or a port voltage.

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


#### Conformal geometry: curved PEC, dielectric interfaces and thin metal sheets

By default STL solids are staircased: each cell is inside or outside a solid, which costs first-order accuracy on curved or oblique boundaries. With `conformal=True` the solver instead uses the length of every primary edge and the area of every primary face that lie outside PEC (partially filled cells), and an anisotropic average of the permittivity on the edges cut by a dielectric interface. Sheets (`sheets=` or the STL material `'sheet'`) model zero-thickness metal, such as patches and strips, whose edges may lie between mesh nodes. The physics guide section "Conformal FIT" explains the method.

Lowest TM mode of a spherical cavity (a vacuum sphere in a PEC background):

```python
import numpy as np
import pyvista as pv
from scipy.constants import c
import wakis
from wakis.sources import CurrentSource, GaussianSource

a, N = 0.1, 24  # radius [m], cells across the diameter
L = (N + 4) * 2 * a / N
sphere = pv.Sphere(radius=a, theta_resolution=300, phi_resolution=300)
grid = wakis.GridFIT3D(-L/2, L/2, -L/2, L/2, -L/2, L/2, N + 4, N + 4, N + 4,
                       stl_solids={'cavity': sphere},
                       stl_materials={'cavity': 'vacuum'},
                       conformal=True)      # False: staircase
solver = wakis.SolverFIT3D(grid, bc_low=['pec']*3, bc_high=['pec']*3,
                           bg='pec', use_stl=True)
f0 = 2.7437 * c / (2 * np.pi * a)           # TM11: zero of [x j1(x)]'
pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
src = CurrentSource(pulse, component='z', center=(0.02, 0.01, 0.0))
monitor = wakis.DFTMonitor(np.linspace(0.85, 1.1, 501) * f0, components=['Ez'],
                           center=(-0.015, 0.025, 0.01))
solver.emsolve(int((pulse.end_time + 40 / f0) / solver.dt), source=src,
               monitors=[monitor])
```

The peak of `abs(monitor.get_field('Ez')[0])` lies within 0.1 % of `f0` at 24 cells across the diameter. The staircased sphere is about 10 % low. The solver chooses the time step from the Lanczos stability limit of the conformal operators (`solver.conformal_dt_max`; usually 0.85–1 times the Yee limit). A zero-thickness sheet:

```python
from wakis.conformal import rectangle_sheet

grid = wakis.GridFIT3D(..., sheets={'patch': rectangle_sheet(center=(0, 0, 1.6e-3),
                                                             size=(28.5e-3, 37e-3, 0))})
```

The sheet must lie on a grid plane normal to x, y or z. Its edges can be anywhere. A knife-edge correction (`sheet_edge_correction=True`) places the edge at its physical position. For the patch template use `patch_antenna(..., thin_sheet=True)`, which needs no empirical `edge_correction`.

`conformal='tensor'` also keeps the off-diagonal entries of the averaged permittivity tensor. Use it when the electric field crosses a curved or oblique dielectric interface: dielectric spheres and rods with E in the cross-section, photonic resonators, lenses, gratings. For the TM$_1$ mode of a dielectric sphere ($\varepsilon_r=6$) in a spherical cavity at 32 cells across, the error drops from $7\times10^{-3}$ ('all') to below $10^{-4}$, at the same time step and 10–20 % more run time:

```python
core = pv.Sphere(radius=a / 2, theta_resolution=300, phi_resolution=300)
grid = wakis.GridFIT3D(-L/2, L/2, -L/2, L/2, -L/2, L/2, N + 4, N + 4, N + 4,
                       stl_solids={'cavity': sphere, 'core': core},
                       stl_materials={'cavity': 'vacuum', 'core': [6.0, 1.0, 0.0]},
                       conformal='tensor')
solver = wakis.SolverFIT3D(grid, bc_low=['pec']*3, bc_high=['pec']*3,
                           bg='pec', use_stl=True)
print(solver.tensor_operator.rows.size, 'coupled E edges')
```

The same smoothing is available for density-based inverse design: `MaterialGrid(..., smoothing='tensor')` uses the (filtered, projected) density as the fill fraction of each edge's pixel and its gradient as the interface normal. The boundaries of nearly binary designs then move continuously within the cells, and the gradient stays exact:

```python
design = MaterialGrid(center=(0, 0, 0), size=(0.06, 0.06, 0), shape=(31, 31, 1),
                      eps_min=1.0, eps_max=12.0, smoothing='tensor')
```

Dispersive materials keep the diagonal of the averaged tensor (see below); no couplings are created on their edges. Keep the default `cfln=0.5` or smaller: the tensor step is then always stable. See "Tensor subpixel averaging" in the physics guide.

#### Plasmonic particles: dispersive subpixel averaging

With `conformal='all'` or `'tensor'`, STL solids made of a `DispersiveMaterial` (Drude, Lorentz, Debye, `dispersive_lib['gold']`, ...) get the Kottke average at every frequency. Each cut edge sees the parallel (tangential) and series (normal) combination of the media in its cell. This is done by a small network of sub-capacitors on the cut edges, each running the materials' own pole equations. It is passive by construction, needs no change to the time step and has an exact adjoint (physics guide, "Dispersive subpixel averaging"). The localized surface-plasmon (dipole) resonance of a Drude sphere in vacuum, as a complex frequency from the ring-down:

```python
import numpy as np
import pyvista as pv
from scipy.constants import c
import wakis
from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource

b, kp = 0.2, 4.0                     # sphere radius [m], plasma wavenumber [1/m]
metal = DispersiveMaterial(1.0, [Drude(kp * c, 0.0)])
h = 2 * b / 24                       # 24 cells across the diameter
m = int(np.ceil(1.5 / h))
grid = wakis.GridFIT3D(*np.repeat([[-m * h / 2, m * h / 2]], 3, axis=0).ravel(), m, m, m,
                       stl_solids={'ball': pv.Sphere(radius=b, theta_resolution=300,
                                                     phi_resolution=300)},
                       stl_materials={'ball': metal}, conformal='all')
solver = wakis.SolverFIT3D(grid, bc_low=['cpml'] * 3, bc_high=['cpml'] * 3,
                           n_pml=10, use_stl=True)          # use_gpu=True on a GPU
f0 = kp / np.sqrt(3) * c / (2 * np.pi)                      # quasi-static (eps = -2)
pulse = GaussianSource(frequency=f0, fwidth=0.3 * f0)
src = CurrentSource(pulse, component='z', center=(0, 0, 1.3 * b))
iz = (m // 2, m // 2, int(np.argmin(abs(solver.grid.z[:-1] + h / 2 + 1.3 * b))), 'z')
trace = []

class Probe:
    def update(self, solver, t):
        trace.append(float(solver.E[iz]))

solver.emsolve(int((pulse.end_time + 25 / f0) / solver.dt), source=src, monitors=[Probe()])
start = int(np.ceil(pulse.end_time / solver.dt))
res = harminv(np.array(trace[start:]), solver.dt, 0.75 * f0, 1.0 * f0)
F = 2 / solver.dt * np.sin(np.pi * res.complex_frequency * solver.dt) / (2 * np.pi)
k = 2 * np.pi * F[np.argmax(abs(res.amplitude))] / c       # ~ (2.144 + 0.055i) / m
```

The exact Mie pole of the dipole coefficient $a_1$ is $k = 2.1436 + 0.0549i$ m$^{-1}$, against the quasi-static $k_p/\sqrt3 = 2.309$ m$^{-1}$. The difference is the dynamic depolarization; see `studies/tensor-dispersive/mie.py` for the reference and `sphere3d.py` for the convergence study. This example (24 cells across, 0.73M cells, 17k steps, 3 min on an L4 GPU) gives $k = 2.1419 + 0.029i$: the resonance is within $8\times10^{-4}$ of Mie (staircase: $+1.8$ % at the same mesh), but the error is not monotone in the mesh ($-1.7$ % at 32 cells across), and the radiative linewidth comes out 45–60 % low for every mesh, domain and CPML tried, so only the real part should be trusted for now. For metals the averaging converges at first order and not always monotonically: it removes the spurious modes and the half-cell surface shift of the staircase, but a fine staircase can be as accurate. Use `wakis.unstructured` for high-accuracy plasmonics. `MaterialGrid(material=gold, eps_min=2.1, smoothing='tensor')` applies the same averaging to density designs between a dielectric and a dispersive material, with exact gradients.

#### Unstructured meshes: curved dielectrics and metals with `wakis.unstructured`

For curved interfaces where $\mathbf E$ crosses the surface (TE modes of rods and spheres, plasmonic particles), the tetrahedral/triangular FIT solver of `wakis.unstructured` uses a mesh that conforms to the interfaces and stays second order in both polarizations (physics guide, "Unstructured FIT"). The example meshes a dielectric sphere ($\varepsilon_r = 4$, radius 0.5 m) inside a spherical PEC cavity (radius 1 m), computes its resonances and compares them with the exact spherical-Bessel solution:
```python
import numpy as np
from scipy.constants import c
from wakis.unstructured import UnstructuredFIT3D, analytic, ball_mesh, estimate_memory

a, b, eps = 1.0, 0.5, 4.0
mesh = ball_mesh([b, a], h=0.15)        # tag 0: r < b, tag 1: b < r < a
print(mesh.n_cells, "tetrahedra", mesh.diagnostics())
print(f"~{estimate_memory(mesh, eigen=True) / 1e9:.2f} GB")

solver = UnstructuredFIT3D(mesh, materials={0: [eps, 1.0]}, bc="pec")
k_tm = analytic.sphere_in_cavity(a, b, eps, "TM", 3)   # E crosses the surface
k_te = analytic.sphere_in_cavity(a, b, eps, "TE", 3)
f, modes = solver.eigenmodes(6, f_target=2.65 * c / (2 * np.pi))
print("k a =", np.round(2 * np.pi * f / c, 4))
print("exact", np.round(np.r_[k_tm, k_te], 4))
```
The six modes are the TM and TE dipole triplets (relative errors about $3\times10^{-3}$ at $h = 0.15$, i.e. $1.2\times10^4$ tetrahedra, decreasing as $h^2$). `modes` holds the edge voltages; `mesh.edges_to_cell_vectors(modes[:, 0])` gives the field at the cell centres for plotting.

The same solver runs in the time domain. The 2D example below excites a lossless Drude rod ($\omega_p = 4c/a$) in a PEC cavity with an in-plane dipole (TE, $H_z$ on the nodes) and finds its surface-plasmon resonance with harmonic inversion:
```python
from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.unstructured import UnstructuredFIT2D, disk_mesh

metal = DispersiveMaterial(1.0, [Drude(4.0 * c, 0.0)])
solver = UnstructuredFIT2D(disk_mesh([0.5, 1.0], 0.05), {0: metal}, polarization="TE", bc="pec")
f0 = 3.6469 * c / (2 * np.pi)             # exact n = 1 plasmon, k a = 3.6469
s = 1 / (2 * np.pi * 0.4 * f0)
pulse = lambda t: np.exp(-((t - 5 * s) / s) ** 2 / 2) * np.sin(2 * np.pi * f0 * (t - 5 * s))
solver.add_point_source([0.6, 0.1], pulse, direction=[1, 0])
probe = solver.add_probe([[0.27, -0.43]])
solver.run(4000)
signal = solver.probe_signal(probe)[1000:, 0]
modes = harminv(signal, solver.dt, 0.9 * f0, 1.1 * f0, t0=1001 * solver.dt)
print(2 * np.pi * modes.frequency / c)     # [3.6482, 3.7689]: n = 1 (error 3.5e-4) and n = 2 modes
```
`solver.dispersive_eigenmode(f0)` gives the same mode from the frequency domain. Meshes from other tools are loaded with `SimplexMesh.read("part.msh")` (any meshio format; material tags from the gmsh physical groups), and `wakis.unstructured.gmsh_mesh` drives gmsh if it is installed (it is an optional external tool, not a Wakis dependency). `bc="abc"` gives a first-order absorbing outer boundary; a PML is on the roadmap (`docs/unstructured_roadmap.md`).

#### Design templates: ready-made antennas, waveguides and unit cells

`wakis.templates` sets up complete simulations from a declarative description: it chooses the mesh from a target number of cells per wavelength in the densest material, the padding to the CPML, the CPML thickness and the run length (energy-decay criterion), and returns the usual engineering results. See the [templates tutorial](templates.md) for all templates and pitfalls.

```python
from wakis.templates import patch_antenna, patch_results

sim = patch_antenna(frequency=2.45e9, substrate=4.4, height=1.6e-3)
sim.run(energy_decay=1e-4)
res = patch_results(sim)  # resonance, |S11| dip, Zin, bandwidth, directivity
print(res["resonance"] / 1e9, res["s11_min_db"], res["directivity_dbi"])
```

For multi-port photonic integrated circuits (waveguides, bends, tapers, couplers, MMIs, and ring resonators), `wakis.templates.photonics` provides `PhotonicCircuit` (and `wakis.subgrid.SubgridPhotonicCircuit` for 2D quadtree subgridding via `backend="subgrid"`), extruded `PolySlab` geometry primitives, and automated $N$-port `run_s_parameters()`.

> Naming: `PhotonicCircuit` (formerly `PICSimulation`) — 'PIC' is avoided in Wakis because it usually means particle-in-cell.

#### Semiconductor gain media and diode lasers: `SemiconductorGain`

`wakis.gain_semiconductor` provides `SemiconductorGain` (and the frequency-chirp helper `adiabatic_chirp`), coupling carrier density $N(\mathbf{r}, t)$ [$\text{m}^{-3}$] (driven by current injection $\eta_i J/(q d)$, SRH/radiative/Auger recombination $A N + B N^2 + C N^3$, stimulated emission work $-\mathbf{E}\cdot\partial_t\mathbf{P}/(\hbar\omega_0)$, and optional carrier diffusion $D\nabla^2 N$) to one or more Lorentzian ADE gain poles and Henry's linewidth enhancement factor $\alpha_H$ (`solver.add_gain_material`). See *Semiconductor gain media* in the [physics guide](physicsguide.md) and the full example script `examples/015_semiconductor_fp_laser.py`:

```python
import numpy as np
from scipy.constants import c
import wakis
from wakis.gain_semiconductor import SemiconductorGain, adiabatic_chirp
from wakis.sources import CurrentSource, GaussianSource

f0 = c / 1.55e-6  # 193.41 THz (1.55 um telecom band)
gain = SemiconductorGain(
    omega_0=2 * np.pi * f0,
    gamma=2 * np.pi * 25.0e12,
    N_tr=1.2e24,                # transparency carrier density [1/m^3]
    differential_gain=4.0e-20,  # dg/dN [m^2]
    gain_model="linear",        # or "logarithmic" with g0=...
    alpha_H=4.0,                # Henry linewidth enhancement factor [-]
    J=2.5e7,                    # injection current density [A/m^2] (or callable J(t))
    eta_i=0.90,
    active_thickness=0.12e-6,   # active-layer thickness d [m]
    A=5.0e8,                    # SRH [1/s]
    B=1.0e-16,                  # radiative [m^3/s]
    C=3.0e-41,                  # Auger [m^6/s]
    diffusion_coeff=1.0e-3,     # ambipolar diffusion D [m^2/s]
    eps_inf=12.25,
    initial_carriers="steady_state",
)

# Analytic Fabry-Perot diode laser operating point (with optional DBR effective length L_eff),
# small-signal modulation transfer function, and Koch-Bowers chirp helper
ss = gain.steady_state_laser(length=300e-6, R1=0.32, R2=0.32, L_eff=300e-6)
H_tf = gain.modulation_transfer_function(np.linspace(0.1e9, 20e9, 200), length=300e-6, R1=0.32)

# Register on SolverFIT3D and inspect carrier density / particle & Poynting budget (including dchi_work_joules)
grid = wakis.GridFIT3D(-1e-6, 1e-6, -1e-6, 1e-6, -1.6e-6, 1.6e-6, 2, 2, 160, verbose=0)
solver = wakis.SolverFIT3D(grid, cfln=0.5, bc_low=["periodic", "periodic", "cpml"],
                           bc_high=["periodic", "periodic", "cpml"], n_pml=10, verbose=0)
solver.add_gain_material(gain, center=(0, 0, 0), size=(4e-6, 4e-6, 1.2e-6))
N_avg = solver.get_gain_carriers(cell_average=True)
budget = solver.get_gain_bookkeeping()
```

#### Active gain media: spontaneous-emission noise and lifetime acceleration

`wakis.gain` (`FourLevelGain`, `SaturableGain`, `solver.add_gain_material`) couples four-level rate equations and Lorentzian dipole polarizations to `SolverFIT3D`. By default (`spontaneous_emission=False`), the equations are deterministic and require an external seed pulse to start lasing; passing `spontaneous_emission=True` (with optional `seed=...`) injects Fluctuation–Dissipation-Theorem (FDT) Langevin polarization noise proportional to $\sqrt{N_2}$ (evaluated at the Crank–Nicolson half-step midpoint $N_2^{n+1/2}$) so amplified spontaneous emission (ASE) and laser oscillation start from rest. Passing `time_scale=s` (alias `lifetime_scale=s`) scales $\tau_{ij} \to \tau_{ij}/s$ and $W_p \to s W_p$ (while holding $\kappa_a$ at the physical unscaled $\tau_{21}$) to compress nanosecond population build-up by a factor of $s$ (for $s \ll s_{\max,\text{cav}} = \tau_{\text{sat}}/\tau_p$ in the Class-B underdamped regime) without altering the threshold pump rate, steady-state inversion, saturation intensity, or steady-state laser output:

```python
from wakis.gain import (
    FourLevelGain,
    ase_spectrum_1d,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)

f0 = c / 1.2e-6
R_dbr = dbr_reflectance(2.0, 1.0, n_pairs=1, extra_high=True)  # 3-layer (H L H) DBR: 1 (H,L) pair + 1 extra H
g_th = fabry_perot_threshold_gain(length=1.2e-6, R1=R_dbr, R2=R_dbr)

gain = FourLevelGain(
    omega_a=2 * np.pi * f0,
    gamma_a=1.0e14,
    tau_21=15.0e-12,
    tau_10=20.0e-15,
    tau_32=10.0e-15,
    N_total=6.0e25,
    spontaneous_emission=True,  # FDT Langevin noise (~ sqrt(N_2)); False by default
    seed=42,                    # reproducible per-group PCG64 RNG
    time_scale=100.0,           # 100x lifetime acceleration; steady state unchanged
    spatial_averaging=True,
)
gain.W_p = gain.threshold_pump_rate(2.5 * g_th)
solver.add_gain_material(gain, center=(0.0, 0.0, 0.6e-6), size=(4e-6, 4e-6, 1.2e-6))

# Analytic helpers for ASE gain narrowing, beta-rounded L-I curve, and validity bounds
freqs = np.linspace(200e12, 300e12, 201)
S_ase = ase_spectrum_1d(freqs, length=2.4e-6, material=gain)
I_out = fabry_perot_laser_output(
    2.5, gain.saturation_intensity(), R_dbr, R_dbr,
    spatial_hole_burning=False, beta_sp=1e-2,
)
limits = gain.time_scaling_limits(tau_p=22.0e-15)  # checks s/tau_sat << kappa_cav, gamma_a
```
See `examples/012_fabry_perot_laser.py` and `examples/014_laser_noise_and_time_scaling.py` for full Fabry–Pérot laser, ASE gain-narrowing, multi-mode competition, and lifetime-acceleration benchmarks.

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

### Linear plasma wakefield in a plasma-filled pipe

A cold plasma is a Drude medium, so the linear plasma wakefield of a rigid bunch moving at $c$ can be computed with a `Beam` source and `wakis.plasma` (physics guide: *Linear plasma wakefield in a plasma-filled pipe*; full script: `examples/007_linear_plasma_wakefield.py`):

```python
import numpy as np
import pyvista as pv
from scipy.constants import c
from wakis import GridFIT3D, SolverFIT3D
from wakis.plasma import cold_plasma, linear_wake, plasma_frequency
from wakis.sources import Beam

omega_p = plasma_frequency(1.1e21)              # n_e [1/m^3] -> lambda_p ~ 1 mm
k_p = omega_p / c
a, L, h = 2 / k_p, 24 * 2 * np.pi / k_p, 2 * np.pi / k_p / 32
n = int(np.ceil(a / h)) + 2
pipe = pv.Cylinder(center=(0, 0, L / 2), direction=(0, 0, 1), radius=a,
                   height=3 * L, resolution=720).triangulate()
grid = GridFIT3D(-n * h, n * h, -n * h, n * h, 0, L, 2 * n, 2 * n, int(L / h),
                 stl_solids={"pipe": pipe}, stl_materials={"pipe": "vacuum"},
                 conformal="pec")                 # cut-cell pipe wall
solver = SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3,
                     bg="pec", use_stl=True)
lam_p = 2 * np.pi / k_p
solver.add_dispersive_material(cold_plasma(omega_p),   # plasma from z = lambda_p
                               center=(0, 0, lam_p - h / 4 + 2 * L), size=(4 * L,) * 3)
beam = Beam(q=1e-9, sigmaz=1 / k_p)
Nt = int(((L - lam_p - solver.z.min()) / c + beam.ti) / solver.dt)
solver.emsolve(Nt, source=[beam])

xi = solver.z - (solver.z.min() + c * (Nt * solver.dt - beam.ti))  # z - z_bunch
i0, di = np.argmin(abs(grid.x)), int(round(a / 2 / h))
ez_wakis = solver.E[i0 + di, i0, :, "z"]
ez_theory, _, _ = linear_wake(grid.x[i0 + di], 0.0, xi, k_p, 1 / k_p, q=1e-9, radius=a)
```

Compare behind the bunch, a few $\lambda_p$ long, far from the plasma entrance (its transition radiation trails the bunch), and at radii of several cells from the beam line.

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

## 🔮 Nonlinear Kerr ($\chi^{(3)}$) Media and Soliton Microcombs (`wakis.nonlinear`)

Wakis supports sub-cycle third-order ($\chi^{(3)}$) Kerr and delayed vibrational Raman nonlinear media in `SolverFIT3D` via `KerrMaterial` and `solver.add_nonlinear_material`, together with analytical helpers for self-phase modulation (SPM), third-harmonic generation (THG), Kerr cavity perturbation theory, and a split-step Fourier Lugiato–Lefever Equation (LLE) solver for dissipative Kerr soliton microcombs:

```python
import numpy as np
import wakis
from wakis.materials import Lorentz

# 1. Define a dispersion-engineered Kerr material (instantaneous chi^(3) + optional Raman/Lorentz poles)
kerr_mat = wakis.KerrMaterial(
    eps_inf=2.25,
    chi3=2.0e-18,            # [m^2/V^2] (or specify n2=... [m^2/W])
    poles=[Lorentz(omega_0=2 * np.pi * 80e12, delta_eps=0.15, gamma=0.0)],
    raman_fraction=0.18,     # optional delayed Raman fraction f_R in [0, 1]
    tau1=12.2e-15,           # Raman vibrational period parameter [s]
    tau2=32.0e-15,           # Raman damping time [s]
    method="newton",         # 'newton' (default), 'fixed_point', or 'explicit_ade'
)

# Convert between chi^(3) [m^2/V^2] and n2 [m^2/W] (with the 3/4 real-field factor)
n2_val = wakis.chi3_to_n2(kerr_mat.chi3, eps_r=2.25)
phi_nl = wakis.spm_phase_shift(n2=n2_val, intensity=1.0e14, length=6.0e-6, wavelength=1.5e-6)

# 2. Attach KerrMaterial to a box region in SolverFIT3D
grid = wakis.GridFIT3D(-0.1e-6, 0.1e-6, -0.1e-6, 0.1e-6, 0.0, 10.0e-6, 2, 2, 400, verbose=0)
solver = wakis.SolverFIT3D(
    grid, cfln=0.5, bc_low=["periodic", "periodic", "cpml"], bc_high=["periodic", "periodic", "cpml"], verbose=0
)
solver.add_nonlinear_material(kerr_mat, center=(0.0, 0.0, 5.0e-6), size=(0.2e-6, 0.2e-6, 6.0e-6))

# 3. Solve the FIT-informed Lugiato-Lefever Equation (LLE) for a soliton microcomb
phi = np.linspace(-np.pi, np.pi, 256, endpoint=False)
sol_ref = wakis.lle_soliton_analytic(phi, zeta_0=4.0, d2=0.04, f_pump=2.3)
lle_out = wakis.solve_lle(
    d2=0.04, f_pump=2.3, zeta_0=4.0, n_modes=256, dtau=0.01, n_steps=1500, psi_init=sol_ref["psi"]
)
```

```{seealso}
See `examples/013_kerr_nonlinear_optics.py` and [Tutorial 5.11 · Kerr ($\chi^{(3)}$) Nonlinear Media in FIT and FIT-Informed Soliton Microcombs](../tutorials/05_guided_waves/05-11_kerr_microcomb/kerr_microcomb.ipynb) for complete worked examples.
```

## ⚡ Relativistic Particle-in-Cell (`wakis.particles`)

`wakis.particles` couples relativistic charged macroparticles (`Species`, `ParticleSolver`, `BeamInjector`, `Collector`) self-consistently to `SolverFIT3D` via a leapfrog `mid_step` hook between the magnetic and electric field updates, with exact 3D Esirkepov charge-conserving current deposition on uniform and graded meshes:

```python
import numpy as np
from scipy import constants as const
import wakis
from wakis.particles import (
    BeamInjector,
    Collector,
    ParticleSolver,
    Species,
    cold_plasma_dispersion_discrete,
    gauss_residual,
)

# 1. Create a 3D FIT grid and solver
grid = wakis.GridFIT3D(0.0, 0.02, 0.0, 0.02, 0.0, 0.08, 6, 6, 32, verbose=0)
solver = wakis.SolverFIT3D(
    grid, cfln=0.5, bc_low=["periodic", "periodic", "pec"], bc_high=["periodic", "periodic", "pec"], verbose=0
)

# 2. Define an electron species (q = -e, m = m_e by default; u = gamma * v [m/s])
electrons = Species(name="e-", charge=-const.e, mass=const.m_e)

# 3. Optional continuous beam injector at the z = zmin PEC cathode and downstream collector
injector = BeamInjector(
    species=electrons,
    z0=0.0,                             # cathode emitter plane [m]
    current=5.0,                        # beam current [A]
    energy_eV=250.0e3,                  # relativistic kinetic energy [eV]
    x0=0.01,
    y0=0.01,
    radius=0.004,                       # beam radius [m]
    direction=1,                        # +1 along +z
    n_macro_per_step=16,
)
collector = Collector(name="anode_cup", bounds=(0.0, 0.02, 0.0, 0.02, 0.035, 0.08))

# 4. Attach ParticleSolver with external axial magnetic guiding field B_z = 0.25 T
ps = ParticleSolver(
    solver,
    species=[electrons],
    injectors=[injector],
    collectors=[collector],
    pusher="higuera_cary",              # 'boris' (default), 'vay', or 'higuera_cary'
    shape=2,                            # 1 (CIC, default) or 2 (TSC)
    gather="galerkin",                  # 'galerkin' (energy-consistent) or 'momentum'
    B_ext=lambda x, y, z, t: (np.zeros_like(x), np.zeros_like(y), np.full_like(z, 0.25)),
    profile=False,                      # opt-in per-step CPU stage timing (default False, zero overhead)
).attach()

# If solver.E or particle positions/weights are modified in place after attach(),
# call ps.reset_gauss_reference() before stepping to refresh the Gauss reference state.

# 5. Step the coupled Vlasov-Maxwell system and inspect conservation diagnostics
for _ in range(80):
    solver.one_step()

print("Active macroparticles:", electrons.n_particles)
print("Gauss law relative residual:", gauss_residual(solver, [electrons]))
print("Charge balance [C]:", ps.injected_charge(), ps.total_charge() + ps.absorbed_charge())
print("Collector absorbed energy [J]:", collector.absorbed_energy)
```

### GPU-Accelerated Execution (`use_gpu=True`)

When `SolverFIT3D` is initialised with `use_gpu=True` (either with the standard CuPy sparse step or the fused `wakis.gpu_fused` CUDA step `fused=True`), setting `use_gpu=True` on each `Species` keeps all particle struct-of-arrays buffers on the device (`xp = cupy`) and executes the coupled gather-push, Esirkepov `float64` `atomicAdd` current deposition, boundary compaction, and periodic cell sorting via compiled CUDA `RawKernel` kernels asynchronously on the GPU stream without device-to-host synchronization on the standard Esirkepov path:

```python
import cupy as cp
from scipy import constants as const
import wakis
from wakis.particles import ParticleSolver, Species, gauss_residual

# 1. Create a GPU-accelerated 3D FIT solver (standard or fused=True)
grid_gpu = wakis.GridFIT3D(0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 16, 16, 16, verbose=0)
solver_gpu = wakis.SolverFIT3D(
    grid_gpu,
    cfln=0.45,
    bc_low=["periodic", "periodic", "periodic"],
    bc_high=["periodic", "periodic", "periodic"],
    use_gpu=True,
    fused=True,
    verbose=0,
)

# 2. Allocate device-resident electron species (xp = cupy)
electrons_gpu = Species(
    name="e-", charge=-const.e, mass=const.m_e, capacity=4096, use_gpu=True
)
rs = cp.random.RandomState(42)
electrons_gpu.add_particles(
    x=rs.uniform(0.002, 0.018, size=1024),
    y=rs.uniform(0.002, 0.018, size=1024),
    z=rs.uniform(0.002, 0.018, size=1024),
    ux=rs.normal(0.0, 0.05 * const.c, size=1024),
    uy=rs.normal(0.0, 0.05 * const.c, size=1024),
    uz=rs.normal(0.20 * const.c, 0.02 * const.c, size=1024),
    w=cp.full(1024, 1.0e3, dtype=cp.float64),
)

# 3. Attach ParticleSolver with periodic cell sorting and external B_z field
ps_gpu = ParticleSolver(
    solver_gpu,
    species=[electrons_gpu],
    pusher="boris",
    shape=2,
    gather="galerkin",
    sort_every=20,
    B_ext=(0.0, 0.0, 0.15),
).attach()

# 4. Advance asynchronously on the GPU stream (no host synchronization inside one_step)
for _ in range(40):
    solver_gpu.one_step()

print("GPU Gauss law relative residual:", gauss_residual(solver_gpu, [electrons_gpu]))
```

```{seealso}
See `examples/017_particles_plasma_oscillation_and_gyration.py`, `examples/018_particles_gpu_throughput.py`, and the [Physics Guide section on Electromagnetic Particle-in-Cell](physicsguide.md#electromagnetic-particle-in-cell-wakisparticles) for discrete charge-conservation theory, relativistic pusher benchmarks, cold-plasma dispersion analysis, and GPU throughput scaling.
```
