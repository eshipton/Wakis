"""Benchmark cases (a)-(e) of the GPU time-stepping study.

Each builder returns ``(solver, sources, monitors)``; the time loop is that of
`SolverFIT3D.emsolve` (sources, `one_step`, element post-steps, monitors).

  a  plain3d    3D Cartesian, PEC box, point current source
  b  cpml3d     3D, CPML on all six faces
  c  tensor2d   2D dielectric rod, conformal='tensor' (subpixel tensor averaging)
  c  ade3d      3D CPML box with a Lorentz + Drude + Debye block (ADE)
  d  tfsf2d     2D TF/SF plane wave, two 81-frequency flux boxes + DFT plane
  e  adjoint2d  AdjointSimulation (forward + vjp) on a 2D CPML problem with a
                DFT monitor and a MaterialGrid (see bench.py)

`fused` is passed to SolverFIT3D only when it is not None (the baseline
branch does not have the keyword).
"""

import numpy as np

import wakis
from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.sources import CurrentSource, GaussianSource

C0 = 299792458.0


def _solver(grid, use_gpu, fused, **kw):
    if fused is not None:
        kw["fused"] = fused
    return wakis.SolverFIT3D(grid, use_gpu=use_gpu, verbose=0, **kw)


def _dipole(f0, component="z", center=(0.0, 0.0, 0.0)):
    return CurrentSource(
        GaussianSource(f0, 0.5 * f0), component=component, center=center
    )


def plain3d(use_gpu, fused=None, n=64):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, n, n, n, verbose=0)
    solver = _solver(grid, use_gpu, fused, bc_low=["pec"] * 3, bc_high=["pec"] * 3)
    return solver, [_dipole(3e9)], []


def cpml3d(use_gpu, fused=None, n=64):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, n, n, n, verbose=0)
    solver = _solver(
        grid, use_gpu, fused, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=8
    )
    return solver, [_dipole(3e9)], []


def tensor2d(use_gpu, fused=None, n=256):
    import pyvista as pv

    r = 0.1
    h = 2 * r / n
    rod = pv.Cylinder(
        center=(0.0031, -0.0017, 0), direction=(0, 0, 1), radius=r / 2,
        height=20 * h, resolution=720,
    ).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -r, r, -r, r, -h / 2, h / 2, n, n, 1,
        stl_solids={"rod": rod}, stl_materials={"rod": [6.0, 1.0, 0.0]},
        conformal="tensor", verbose=0,
    )  # fmt: skip
    solver = _solver(
        grid, use_gpu, fused,
        bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        use_stl=True,
    )  # fmt: skip
    return solver, [_dipole(1e9, center=(0.07, 0.07, 0.0))], []


MATERIAL = DispersiveMaterial(
    2.0,
    [
        Lorentz(1.5, 2 * np.pi * 3.5e9, 2 * np.pi * 0.4e9),
        Drude(2 * np.pi * 2e9, 2 * np.pi * 0.5e9),
        Debye(0.8, 1e-10),
    ],
    sigma=0.02,
)


def ade3d(use_gpu, fused=None, n=64):
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, n, n, n, verbose=0)
    solver = _solver(
        grid, use_gpu, fused, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=8
    )
    solver.add_dispersive_material(MATERIAL, center=(0.03, 0, 0), size=(0.06, 0.1, 0.1))
    return solver, [_dipole(3e9, center=(-0.04, 0, 0))], []


TFSF_FREQS = np.linspace(100e12, 400e12, 81)


def tfsf2d(use_gpu, fused=None, n=104, nz=4):
    """Tutorial-01-like 2D TF/SF run (no scatterer), ~43 kcells."""
    dx = 50e-9
    h = n / 2 * dx
    hz = nz / 2 * dx
    grid = wakis.GridFIT3D(-h, h, -h, h, -hz, hz, n, n, nz, verbose=0)
    solver = _solver(
        grid, use_gpu, fused, bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"], n_pml=10,
    )  # fmt: skip
    pulse = GaussianSource(frequency=250e12, fwidth=100e12)
    tf = 25 * dx
    src = wakis.TFSFSource(
        pulse, center=(0, 0, 0), size=(2 * tf, 2 * tf, None), direction="+x",
        polarization="z", amplitude=1.0,
    )  # fmt: skip
    lz = 2 * dx
    scat = wakis.FluxMonitor(TFSF_FREQS, size=(2 * (tf + 4 * dx),) * 2 + (lz,))
    inner = wakis.FluxMonitor(TFSF_FREQS, size=(2 * (tf - 6 * dx),) * 2 + (lz,))
    span = 2 * (h - 12 * dx)
    plane = wakis.DFTMonitor(
        TFSF_FREQS[::27], components=("Ez",), size=(span, span, 0.0)
    )
    return solver, [src], [scat, inner, plane]


CASES = {
    "plain3d": plain3d,
    "plain3d_big": lambda g, f=None: plain3d(g, f, n=128),
    "cpml3d": cpml3d,
    "tensor2d": tensor2d,
    "ade3d": ade3d,
    "tfsf2d": tfsf2d,
}


# ------------------------------------------------------------ adjoint case


def adjoint2d(use_gpu, fused=None, n=160, Nt=2000):
    """AdjointSimulation of a 2D CPML problem (dipole, DFT probe, 2D design)."""
    from wakis.inverse import AdjointSimulation, MaterialGrid

    L = 0.3
    dx = L / n

    def make_solver():
        grid = wakis.GridFIT3D(
            -L / 2, L / 2, -L / 2, L / 2, -dx / 2, dx / 2, n, n, 1, verbose=0
        )
        return _solver(
            grid, use_gpu, fused, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=10,
        )  # fmt: skip

    src = _dipole(3e9, center=(-0.08, 0.0, 0.0))
    freqs = np.linspace(2e9, 4e9, 6)
    mon = wakis.DFTMonitor(
        freqs, components=("Ez", "Hx"), center=(0.08, 0.0, 0.0), size=(0.0, 0.1, 0.0)
    )
    design = MaterialGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.0), (20, 20, 1), 1.0, 4.0)
    return AdjointSimulation(make_solver, [src], {"probe": mon}, design, Nt)
