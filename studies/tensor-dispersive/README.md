# Dispersive subpixel averaging study

Scripts behind the physics-guide section "Dispersive subpixel averaging".

| Script | Case |
| --- | --- |
| `rod2d.py` | 2D lossless Drude rod in a circular PEC cavity, TE n = 1 surface plasmon vs Bessel matching |
| `rod_lorentz.py` | 2D Lorentz rod (eps(w_ref) = 6, geometry of `tests/test_035`), TE n = 0, vs the Delaunay FIT reference |
| `sphere3d.py` | 3D Drude sphere: TM_1 mode in a PEC sphere (`cavity`) and the dipole LSPR in vacuum with CPML (`open`, complex Mie pole) |
| `mie.py` | exact references: Riccati-Bessel cavity modes, complex Mie a_1 pole, quasi-static limit |
| `coupled.py` | RESEARCH ONLY: the edge-pixel full tensor for dispersive pixels (non-passive, unstable for metals) |
| `analyze.py` | tables and the convergence figure from `runs/*.jsonl` |

Each run script appends one JSON line per run:
`python rod2d.py 96 all runs/rod_x.jsonl`. The conformal mode is `False`
(staircase), `pec` (conformal PEC wall, staircased poles: the previous
behaviour of dispersive solids), `all` or `tensor` (dispersive diagonal
Kottke average), or `coupled` (research variant). The sweeps ran on the
project VM: 2D rods on 8 CPU cores and 3D on the L4 GPU. The previous
`all` behaviour came from a checkout of `develop` at c7a4c6d.

`frozen_proto2d.py` and `frozen_eigs.py` hold the frozen-permittivity
eigenvalue prototype: a 2D TE Yee/FIT operator for a rod in a square PEC box
(the geometry of `tests/test_035`, scaled by 10). It compares averaging
schemes (staircase, diagonal, edge-pixel tensor, octant-pixel tensors) at a
fixed real eps. It reproduces the `test_035` numbers of Wakis 'all' and
'tensor' for eps = 6. Usage:
`EPS2=6 KREF=1.9746767234 python frozen_eigs.py 32,64,128 diag,edge_tensor,quad_exact,quad_Tavg`.
Without EPS2 it uses the Drude rod (k_p = 4) at its self-consistent
frequency, k = 3.7492892653 from `wakis.unstructured` (Richardson,
h = 0.02/0.01).
