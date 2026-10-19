# Wakis tutorials

Worked examples that teach the Finite Integration Technique (FIT) with Wakis
and validate it against theory. Each tutorial is a runnable script and a
Colab notebook generated from the same source, with figures committed in its
`figures/` folder.

<table>
<tr>
<td width="42%"><a href="01_mie_scattering/mie_scattering.ipynb"><img src="01_mie_scattering/figures/qsca_vs_mie.png" width="100%"></a></td>
<td>

### [01 &mdash; Mie scattering of a lossless dielectric sphere](01_mie_scattering/mie_scattering.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/01_mie_scattering/mie_scattering.ipynb)

Plane-wave scattering by a sphere (n = 2, r = 1 µm, x = 2&ndash;10) with a
total-field/scattered-field source: equivalence principle, PyVista geometry,
staircase vs conformal meshes, memory preflight, near and far fields,
Q<sub>sca</sub> versus an exact Mie series, and a resolution study.

</td>
</tr>
<tr>
<td width="42%"><i>coming next</i></td>
<td>

### 02 &mdash; Plasmonic spheres (Ag, Au), 10 nm to 1 µm
Dispersive (Drude-Lorentz) metals with the same TF/SF set-up, absorption
from the inner flux box, comparison with complex-index Mie theory.

</td>
</tr>
</table>

## Running a tutorial

```bash
# local, from the repository root (virtual environment with `uv pip install -e .`)
python tutorials/01_mie_scattering/mie_scattering.py --quick          # CPU, minutes
python tutorials/01_mie_scattering/mie_scattering.py --full --gpu     # GPU, accurate
```

On Colab, open the notebook with the badge: its first cell installs the fork
from GitHub (`pip install git+https://github.com/eshipton/Wakis.git@main`)
and `xvfb` for off-screen PyVista rendering, and the run mode is chosen from
the hardware (GPU &rarr; full, CPU &rarr; quick).

## Conventions shared by all tutorials

* **One source, two formats.** `NN_name/name.py` is a
  [Jupytext](https://jupytext.readthedocs.io) *percent* script (`# %%` code
  cells, `# %% [markdown]` text cells). It runs as a headless script and is
  the only file to edit. The notebook is generated from it:

  ```bash
  python tutorials/build.py                 # regenerate all notebooks (no outputs)
  python tutorials/build.py --check         # CI: fail if a notebook is stale
  python tutorials/build.py 01 --execute --args "--full --gpu"   # run, keep outputs
  ```

  Executed notebooks keep only static outputs (PNG images, text), so they
  render on GitHub and stay below ~5 MB.
* **Run modes.** `--quick` (CPU, &le; 1&ndash;2 Mcells, minutes),
  `--full` (GPU, converged), `--gpu`/`--cpu` to force the backend. In a
  notebook the options come from the environment variable
  `WAKIS_TUTORIAL_ARGS`.
* **Preflight first.** Every tutorial calls `wakis.estimate.estimate` and
  `preflight` before allocating a 3D grid.
* **Validation.** Every tutorial compares with an analytic or independent
  reference (`wakis.analytic`) and states its error and resolution.
* **Visualization.** `wakis.visualize` provides the shared helpers:
  `setup_rendering` (off-screen PyVista, static notebook backend),
  `plot_scene` (solids and labelled boxes), `plot_material_slice`
  (permittivity per edge: staircase vs conformal), `field_slice` /
  `plot_field_slice`, `plot_isosurfaces`, `SliceRecorder` + `save_animation`
  (GIF movies).
* **Units.** SI throughout (metres, seconds, Hz); plots label µm and fs.
