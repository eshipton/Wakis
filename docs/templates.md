# 🧰 Design templates

`wakis.templates` is a thin workflow layer on top of the solver for engineers
who want a working simulation of a common device in a few lines, in the spirit
of the component libraries of commercial tools, but scriptable. Each template
is a function with physically meaningful parameters and sensible defaults that
returns a configured `Simulation`; a matching `*_results` helper extracts the
usual engineering numbers. Everything is built from the public Wakis API
(`GridFIT3D`, `SolverFIT3D`, `LumpedPort`, `ModeSource`/`ModeMonitor`,
`Near2FarMonitor`, `FluxMonitor`, dispersive materials), so you can always
open the box: `sim.grid`, `sim.solver` and the monitors are ordinary Wakis
objects.

```{contents}
:depth: 2
```

## Quick start

```python
from wakis.templates import patch_antenna, patch_results
from wakis.templates.analysis import plot_s_parameters, plot_impedance

sim = patch_antenna(frequency=2.45e9, substrate=4.4, height=1.6e-3)
print(sim.design)          # TLM length, width, feed offset, predicted f_r, D
sim.run(energy_decay=1e-4) # -40 dB energy criterion
print(sim.summary())       # cells, cell sizes, CPML, dt, number of steps
res = patch_results(sim)   # resonance, |S11| dip, Zin, bandwidth, directivity

f = sim.frequencies
plot_s_parameters(f, {"S11": sim.s11()})
plot_impedance(f, sim.input_impedance(), z0=50)
```

## The `Simulation` object

```python
from wakis.templates import Simulation, Box, Cylinder
from wakis.lumped import LumpedPort
import wakis

sim = Simulation(
    f_min=2e9, f_max=3e9,
    structures=[
        Box(center=(0, 0, 0.8e-3), size=(60e-3, 60e-3, 1.6e-3), material=4.4),
        Box(center=(0, 0, 0), size=(60e-3, 60e-3, 0), material="pec"),   # sheet
        Cylinder(center=(0, 0, 10e-3), radius=0, length=20e-3,
                 material="pec", axis="z"),                              # thin wire
    ],
    cells_per_wavelength=20,
)
sim.add_source(LumpedPort(sim.pulse, component="z",
                          center=(5e-3, 0, 0.8e-3), size=(0, 0, 1.6e-3)))
sim.add_monitor("box", wakis.FluxMonitor(sim.frequencies, size=(0.1, 0.1, 0.05)))
sim.run()
flux = sim.monitor("box").get_flux()
```

What it does for you:

| Setting | Rule | Override |
| --- | --- | --- |
| Excitation | `sim.pulse`: Gaussian pulse centred in the band, spectrum 0.6 of its peak at `f_min`/`f_max` | build your own source |
| Cell size | `cells_per_wavelength` cells per wavelength at `f_max` **in each material** (projected on the axes, tensor mesh) | `cells_per_wavelength`, `max_cell` |
| Snapping | every structure face and lumped-port gap is a mesh node | — |
| Thin layers | at least `min_cells_per_feature` cells between consecutive faces (e.g. across a substrate) | `min_cells_per_feature` |
| Grading | neighbouring cells differ by at most ~`grading` (1.4) | `grading` |
| Padding | λ/4 at `f_min` between the structures and the CPML | `padding`, `domain` |
| CPML | max(10, λ(f_min)/(8Δ)) cells (≤ 30) | `n_pml` |
| Run length | stop when the field energy has decayed to `energy_decay` (default 1e-4 = -40 dB) of its peak, after the pulse has ended | `run(energy_decay=..., max_time=...)` |

Materials are `"pec"`, a relative permittivity (number), a key of
`wakis.materials.dispersive_lib` (`"fr4"`, `"ro4350b"`, `"silicon"`,
`"fused silica"`, `"gold"`, ...) or a `DispersiveMaterial`. Later structures
override earlier ones, so put the substrate first and the metal on top. A box
with a zero size is a metal sheet; a cylinder of zero radius is a wire along
one line of mesh edges.

`run(reference=True)` additionally runs the empty domain (same mesh, sources
and monitors) for normalizing transmission spectra (`sim.reference_monitor`).
Every `run()` uses fresh copies of the sources and monitors, so a
`Simulation` can be re-run (sweeps).

## Templates

### Rectangular patch antenna: `patch_antenna`

Probe-fed rectangular patch on a finite grounded substrate. Defaults: FR-4
(ε<sub>r</sub> = 4.4, lossless; pass `substrate="fr4"` or `"ro4350b"` for the
dispersive, lossy laminate models), h = 1.6 mm, 2.45 GHz, 50 Ω.
The template designs W and L with the transmission-line model (TLM) and
places the probe where the cavity-model resistance R(0)cos²(πy₀/L) equals
50 Ω; `sim.design` lists all design values.

Reading the results (`patch_results`):

- **Resonance**: the |S11| minimum. The TLM is an approximation good to a few
  percent; with the default mesh expect the simulated resonance 2-4 % below
  the TLM value. A coarser mesh shifts it further down (first-order staircase
  error of the patch edges; see the `edge_correction` parameter). With
  `thin_sheet=True` the patch is a conformal `Sheet` of its physical size
  (edges between mesh nodes, knife-edge correction) and needs no
  empirical correction; at 20 cells per wavelength the FR-4 patch then
  resonates 2.4 % below the TLM value, the converged (Δ→0) result.
- **Matching**: |S11| < -10 dB means more than 90 % of the available power is
  accepted. The TLM edge resistance is only approximate, so the dip may be
  shallower than designed; tune `feed_offset` (see the optimizer below).
- **Input impedance**: R should be close to 50 Ω at resonance; a positive
  reactance comes from the probe inductance (larger for thick substrates).
- **Directivity**: patches on a finite ground plane reach 6-8 dBi (the TLM
  two-slot model over an infinite ground gives 6.0 dBi for FR-4 and 7.0 dBi
  for ε<sub>r</sub> = 2.2 at 2.45 GHz).

Validated (tests/test_027_templates.py): ε<sub>r</sub> = 2.2, h = 3.2 mm,
15 cells/λ: resonance -3.1 % from TLM, |S11| = -12 dB, D = 8.0 dBi (15 s);
FR-4 at 20 cells/λ (3 cells across h): -2.5 %, -11.0 dB, 6.4 dBi (≈2 min);
FR-4 at 30 cells/λ (4 cells across h): -2.7 %, -11.0 dB, 6.5 dBi (≈5 min).

### Half-wave dipole: `dipole_antenna`

Centre-fed straight wire along z with a one-cell lumped gap port (Z₀ = 73 Ω).
`dipole_results` returns the first resonance (Im Z<sub>in</sub> = 0), the
length in wavelengths, Z<sub>in</sub> and the directivity. Expect
L ≈ 0.47-0.48 λ at resonance, Z<sub>in</sub> ≈ 70 Ω and D = 1.64 (2.15 dBi).
The zero-radius wire has an effective radius and gap proportional to the cell
size, so the resonance converges at first order: 0.434 λ at 20 cells/λ,
0.445 at 30, 0.452 at 40, 0.460 at 60, extrapolating to 0.47-0.48 λ. Use
40+ cells/λ, or extrapolate from two meshes as the tests do. The input
resistance at resonance is 72.6-72.9 Ω and D = 2.10-2.20 dBi on all meshes.

### Rectangular waveguide: `rectangular_waveguide`

WR-90 (default) PEC guide along z with a TE10 `ModeSource` and two
`ModeMonitor` ports, optionally with a thin symmetric inductive iris
(`iris_width`). `waveguide_s_parameters` gives S11 and S21 normalized by the
injected mode. Empty guide: |S21| = 1 within 1.1 %, |S11| < 3e-3 over
1.25-1.9 f<sub>c</sub>; the phase advance matches the TE10 propagation
constant. Iris (12 mm): |S11|²+|S21|² = 1 within 1.5 %, |S11| within 0.05 of
the shunt-susceptance model.

### Periodic unit cells: `unit_cell`, `dielectric_slab_cell`, `patch_array_cell`

Unit cell periodic in x and y, CPML along z, x-polarized plane wave at
normal incidence from a sheet current. `reflection_transmission(sim)` returns
R and T normalized by the empty-cell reference run (call
`sim.run(reference=True)`). These are the specular values, meaningful below
the grating threshold f < c/p. Validated: a dielectric slab against the Airy
formulas (|ΔT| < 0.016 at ≥ 20 cells/λ); a square patch array (p = 15 mm,
12 mm patches) is transparent at low frequency (R = 0.2 at 0.2 c/p) and
totally reflecting just below c/p (R = 0.995), with R + T = 1 within 3e-4.

### Silicon photonics: `slab_waveguide`

220 nm silicon slab (Sellmeier silicon, dispersive) in SiO2 at 1550 nm,
2D-like (one periodic cell along y), or a 3D strip with `width=...`. A TE0
`ModeSource` and two `ModeMonitor`s; `slab_results` returns the transmission,
the effective index from the phase advance and the mode-solver index.
Validated: T = 1 within 1.3e-3, n<sub>eff</sub> within 0.25 % of the
dispersive slab eigenvalue equation over 1395-1705 nm (11 cells across the
core).

## Sweeps and optimization

```python
from wakis.templates import parameter_sweep, optimize, patch_antenna
from wakis.templates.analysis import to_db
import numpy as np

def build(length):                       # module-level for multiprocessing
    return patch_antenna(length=length)

def s11_at_target(sim):
    return float(np.interp(2.45e9, sim.frequencies, to_db(sim.s11())))

sweep = parameter_sweep(build, s11_at_target,
                        {"length": [27e-3, 28e-3, 29e-3]}, processes=3)
best = optimize(build, s11_at_target, x0={"length": 28e-3},
                bounds={"length": (26e-3, 30e-3)}, method="bounded", xatol=1e-5)
print(best.params, best.fun, len(best.history))
```

`optimize` wraps `scipy.optimize` (Nelder-Mead, Powell, bounded scalar or
differential evolution); every evaluation is a full simulation, so keep the
number of parameters small (1-4) and the mesh as coarse as the accuracy you
need allows.

## Typical pitfalls

- **Too little padding.** If the CPML is closer than ~λ/4 to the structure
  it absorbs the reactive near field: resonances shift and Q drops. Keep the
  default or increase `padding` for low frequencies.
- **Too coarse a mesh.** Below ~15 cells per wavelength in the densest
  material the phase error exceeds 1 % and resonances move. Thin layers need
  at least 2-3 cells across (`min_cells_per_feature`); metal edges are
  staircased, which shifts resonances at first order in the cell size. Check
  convergence by repeating the run with 1.5× `cells_per_wavelength`.
- **Not waiting for the energy to decay.** Truncating a ringing resonator
  smears its spectrum (spurious ripples, wrong Q, |S11| > 1). Resonant
  antennas and filters need -40 dB (`energy_decay=1e-4`) or lower; fields
  near the cutoff of a guide or near the grating threshold of a periodic
  structure decay slowly, use 1e-5 to 1e-6 there.
- **Frequencies outside the pulse band.** Results are normalized by the pulse
  spectrum; outside [`f_min`, `f_max`] it is small and the results are noisy.
- **Periodic results above c/p.** Higher diffraction orders appear; the
  specular R and T no longer add up to the total.
- **Design formulas are approximate.** TLM resonance and edge resistance,
  induced-EMF dipole impedance and the iris susceptance are textbook
  estimates (a few to ~10 %), not exact references.

The formulas and their references are listed in the physics guide, section
*Design templates*.
