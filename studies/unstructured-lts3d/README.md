# Local time stepping on 3D tetrahedral meshes (`wakis.unstructured.lts3d`)

Study scripts for Brief W4-1: explicit multi-level Chebyshev local time stepping (LTS) on 3D tetrahedral finite integration meshes (`UnstructuredFIT3D`) with both circumcentric Voronoi and consistent Whitney Hodge operators.

The derivation, stability analysis, verification table, and failure modes are in [`docs/physicsguide_sections/unstructured_lts3d.md`](../../docs/physicsguide_sections/unstructured_lts3d.md); figures are catalogued in [`docs/figures.md`](../../docs/figures.md).

## Question

Does tetrahedral LTS reduce wall time at equal accuracy against existing global-step fixes (`improve_dt` node relocation and `massless_tol` Schur elimination), and how much of the element time-step spread can it use?

## Cases

* **C1 (`graded_box`):** PEC cavity ($1.0 \times 0.75 \times 0.5\text{ m}$), raw Delaunay `box_mesh` with $x$-coordinates refined geometrically towards $x = 0$ down to $h/4$, interior jitter $0.2$, seeds $\{1, 2, 3\}$, 4-point $h$ ladder, analytic $\text{TM}_{110}$ mode.
* **C2 (`uniform_box`):** Same PEC cavity and $h$ ladder on a uniform tensor grid with interior jitter $0.2$ (sliver-limited only), seeds $\{1, 2, 3\}$, analytic $\text{TM}_{110}$ mode.
* **C3 (`pillbox_delaunay`):** CAD PEC pillbox (`make_mesh("pillbox", cpw, optimize="none")` and `pillbox_exact` from `studies/unstructured-cadmesh/cadmesh_study.py`), $\text{cpw} \in \{10, 15, 20, 25\}$, analytic $\text{TM}_{010}$ mode.
* **C4 (`pillbox_gmsh`):** Same CAD pillbox meshed with `optimize="gmsh"` (non-Delaunay; Whitney Hodge arms only).

## Arms and Named Baselines

* **Voronoi Hodge:**
  * **G:** Global leapfrog ($\text{cfl} = 0.95$, `massless_tol=1e-2`).
  * **G-imp:** `improve_dt` node-relocated mesh with global leapfrog (wave-1 CAD baseline).
  * **G-ml3e-2, G-ml1e-1:** Global leapfrog with `massless_tol=3e-2` and `1e-1` (massless Schur elimination as a competitor).
  * **L:** Automatic LTS (`lts=dict(p=2, dt="auto")`).
  * **L-dt1.25, L-dt1.5, L-dt2.0, L-dt3.0:** LTS at fixed coarse steps $\Delta t \in \{1.25, 1.5, 2.0, 3.0\}\,\Delta t_{\text{global}}$.
  * **L-imp:** Automatic LTS on the `improve_dt` mesh.
* **Whitney Hodge:**
  * **GW:** Global leapfrog (`hodge="whitney"`).
  * **LW-element:** LTS with `indicator="element"`.
  * **LW-edge_rate:** LTS with `indicator="edge_rate"`.

## Reproduce

```bash
# Quick local smoke check (writes untracked results/*_quick.json and results/*_quick.png):
python studies/unstructured-lts3d/run_study.py --quick
python studies/unstructured-lts3d/plot_figures.py --quick
python studies/unstructured-lts3d/verify_numbers.py --check

# Full study parts on Cloud Batch:
python studies/unstructured-lts3d/run_study.py --part preflight
python studies/unstructured-lts3d/run_study.py --part bounds --procs 8
python studies/unstructured-lts3d/run_study.py --part accuracy --procs 8
OMP_NUM_THREADS=1 python studies/unstructured-lts3d/run_study.py --part timing --procs 1
python studies/unstructured-lts3d/run_study.py --part stability --procs 4
python studies/unstructured-lts3d/plot_figures.py
python studies/unstructured-lts3d/verify_numbers.py --emit-markdown
python studies/unstructured-lts3d/verify_numbers.py --check
```
