# Nonorthogonal FIT (NFIT) study: circular PEC cavity

Test of a body-fitted, nonorthogonal FIT mesh against Cartesian meshing, following
Schuhmann & Weiland, IEEE Trans. Magn. 36, 897 (2000), Clemens & Weiland, PIER 32, 65
(2001) and Cooke et al., J. Comput. Phys. 215, 321 (2006).

## Method (`nfit2d.py`)
2D TMz FIT on an arbitrary quadrilateral mesh: Ez on primary nodes, magnetic flux on
primary edges, H on dual edges.
- Topological operators (curl = node→edge incidence G) are mesh-independent, exactly
  as in FIT.
- Only the magnetic material matrix M_nu depends on the metric. It is assembled from
  the magnetic energy with B reconstructed at each cell corner from the two edges meeting
  there (exact for constant B). It is symmetric positive definite; on rectangles it is
  diagonal and identical to standard FIT, and on skewed cells it couples the edges at each
  corner, the NFIT structure (about 5 nonzeros per row).
- Eigenproblem G^T M_nu G ez = k^2 D_A ez (scipy eigsh). Explicit leapfrog is stable
  (200k steps at 0.95 x CFL, bounded energy).

## Results (`make_figures.py` → `meshes_and_modes.png`, `convergence.png`)
TM01 relative frequency error vs cells inside the cavity:

| Method | ~1.3–2.7k cells | ~5k–10k cells | Order | Stable dt·c/h |
|---|---|---|---|---|
| Wakis 3D, staircase | 3.4e-2 | 1.7e-2 | 1 | ~0.7 |
| prototype staircase | 1.8e-2 | 5.1e-3 | 1 | 0.80 |
| **prototype NFIT O-grid** | 6.6e-4 | 1.7e-4 (5k), 4.3e-5 (20k) | **2** | **0.30** |
| **Wakis 3D, conformal (PBA-style)** | 1.6e-4 | 3.0e-5 | ~2.3 | 0.85–1 × Yee |

Max error over the first 8 modes at ~5k cells: staircase 1.2e-2, NFIT 5.4e-4.

## Triangular FIT (`trifit2d.py`) and the four-grid comparison (`make_figures2.py`)
Delaunay triangulation of the disk (hexagonal interior lattice + boundary ring) with its
Voronoi dual: dual edges are perpendicular to primary edges, so the material matrix is
diagonal again (cotangent weights (cot a + cot b)/2, non-negative for a Delaunay mesh).

| Grid | quad-equivalent cells | TM01 error | Order | Stable dt·c/h |
|---|---|---|---|---|
| Delaunay triangles | 357 / 1438 / 5784 / 23168 | 2.0e-3 / 4.9e-4 / 1.2e-4 / 2.9e-5 | 2 | 0.83 |
| O-grid NFIT | 320 / 1280 / 5120 / 20480 | 2.4e-3 / 6.6e-4 / 1.7e-4 / 4.3e-5 | 2 | 0.30 |

`grids_same_geometry.png`: the same cavity on the Wakis staircase grid, the Wakis conformal
cut cells (116 partially filled cells), the O-grid and the triangles (+ Voronoi dual).
`convergence_all.png`: TM01 error of all grid types.

## Takeaways
- Body-fitted NFIT turns the 1st-order staircase error into 2nd-order convergence:
  ~20x more accurate at equal cells, with the same topological FIT operators.
- On a PEC circle, Wakis's existing Cartesian conformal method (cut cells, the CST
  Perfect Boundary Approximation idea) is still 2–4x more accurate than this first NFIT
  prototype and keeps a ~2.5x larger time step. This mirrors the historical move of
  CST from NFIT to PBA.
- The prototype is deliberately simple: linear-blended O-grid with straight boundary
  chords and small skewed corner cells (min edge 0.37 h). Mesh smoothing/orthogonalization
  and curved boundary edges would improve both accuracy and the time step.
- Triangles (Voronoi dual) match the O-grid accuracy with a diagonal material matrix and a
  Yee-like time step (0.83 vs 0.30), so for body-fitted meshes they are the better choice here.
- Where NFIT should win: curved *dielectric* interfaces. Wakis's conformal dielectric
  averaging is diagonal-only and only first order when E crosses a curved interface
  obliquely; a body-fitted mesh aligns cell faces with the interface. Next test: a
  dielectric rod/disk resonator (TE and TM), where this comparison matters.

## Dielectric rod (eps_r = 6, radius a/2) in a PEC cavity (`dielectric2d.py`, `make_figures3.py`)
Exact TM (Ez) and TE (Hz) modes from Bessel matching. TE is the hard case: E crosses the
curved interface. Lowest-mode relative error (`rod_convergence.png`, grids in `rod_grids.png`):

| Grid | TM, ~3k / ~11k cells | TE, ~3k / ~11k cells | TE order |
|---|---|---|---|
| Wakis staircase | 3.7e-2 / 1.9e-2 | 5.9e-2 / 3.1e-2 | 1 |
| prototype Cartesian staircase | 7.4e-3 / 3.6e-3 | 5.2e-3 / 2.4e-3 | 1 |
| **Wakis conformal (Kottke, diagonal)** | 7.2e-5 / **1.1e-5** | 2.2e-3 / **1.2e-3** | **1** |
| body-fitted O-grid (NFIT) | 4.2e-5 / 1.1e-5 | 2.5e-3 / 6.1e-4 | 2 |
| **Delaunay triangles** | 4.4e-4 (1.4k) / 1.1e-4 (5.8k) | 4.4e-4 (1.4k) / **1.1e-4 (5.8k)** | **2** |

- TM (E tangential): Wakis conformal is excellent, as good as the body-fitted grids.
- TE (E crossing the interface): Wakis conformal is only first order (diagonal Kottke
  averaging on staggered Cartesian edges). Body-fitted triangles are ~10x more accurate at
  half the cells and 2nd order, so the gap grows with refinement. This is the case that
  matters for optics (nanophotonics, plasmonics: fields peak at curved material surfaces).

## Option 1 prototype: full-tensor averaging on a Cartesian grid (`tensor2d.py`)
Rod (eps 6, r = 0.5, off-grid centre) in a square PEC box aligned with the grid (only the
dielectric interface is curved). Reference: Delaunay triangles, Richardson from h = 0.01/0.005
(stable to ~1e-6). Interface cells: inverse-permittivity tensor T = <1/eps> n n^T + (I - n n^T)/<eps>
applied through the symmetric corner Hodge (off-diagonal x-y coupling at each cell corner).

TE relative error of the first four modes at 4k / 16k / 65k / 262k cells:

| Scheme | mode 1 | mode 2 | mode 3 (n=0, E tangential) | mode 4 |
|---|---|---|---|---|
| diagonal (Wakis today) | 1.3e-3 / 7.0e-4 / 3.7e-4 / 1.9e-4 | 1.3e-3 / … / 1.9e-4 | 8.2e-3 / 4.1e-3 / 2.0e-3 / 1.0e-3 | 3.2e-4 / … / 4.2e-5 |
| **full tensor** | 1.1e-4 / 3.9e-5 / 1.2e-5 / **4.9e-6** | 1.3e-4 / … / **5.1e-6** | 4.5e-3 / 2.2e-3 / 1.1e-3 / 5.5e-4 | 4.6e-6 / … / 1.1e-5 |

- Modes 1, 2, 4: ~2nd order, 25–40x more accurate than diagonal averaging.
- Mode 3 (n = 0, E purely tangential, D_t jumps): still 1st order with every Cartesian variant
  (staircase, diagonal, cell- or quarter-cell tensor, exact fill fractions), improved 2x by the
  tensor. Cause: the corner reconstruction combines D fluxes of edges on both sides of the
  interface in cut cells (O(h) local error). Interface-conforming triangles remove it (2nd order).
- TM unchanged (tangential E, arithmetic mean already correct).
