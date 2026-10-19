"""
End-to-end Mie validation (tutorial 01 set-up on a coarse grid): a plane wave
from a TF/SF box scatters off a dielectric sphere (n = 2, r = 1 um); the
scattered flux box gives Q_sca, compared with wakis.analytic.mie.

At 10 cells/um the low size parameters (x = 2-3.5, 9 to 15 cells per
internal wavelength) are resolved; the tolerance is set from the measured
error of the conformal run (a few %), well below the staircase error, so the
test fails if the TF/SF normalization, the flux sign, the conformal materials
or the time centring of the monitors break. Also checks Q_abs ~ 0 (lossless).
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.analytic import mie_efficiencies
from wakis.sources import GaussianSource

R, N = 1e-6, 2.0


def _run(conformal, res=10, use_gpu=False):
    dx = 1e-6 / res
    n_half = 32
    h = n_half * dx
    sphere = pv.Sphere(radius=R, theta_resolution=120, phi_resolution=120)
    grid = wakis.GridFIT3D(
        -h, h, -h, h, -h, h, 2 * n_half, 2 * n_half, 2 * n_half,
        stl_solids={"s": sphere}, stl_materials={"s": [N**2, 1.0]},
        conformal=conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=10,
        use_stl=True, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    x = np.linspace(2.0, 3.5, 7)
    f = x * c / (2 * np.pi * R)
    pulse = GaussianSource(frequency=f.mean(), fwidth=0.35 * f.mean())
    src = wakis.TFSFSource(pulse, size=(2 * 13 * dx,) * 3, direction="+z",
                           polarization="x")  # fmt: skip
    scat = wakis.FluxMonitor(f, size=(2 * 16 * dx,) * 3)
    inner = wakis.FluxMonitor(f, size=(2 * 11 * dx,) * 3)
    solver.emsolve(int(200e-15 / solver.dt), source=[src], monitors=[scat, inner])
    phi = src.incident_flux_density(f) * np.pi * R**2
    return x, scat.get_flux() / phi, -inner.get_flux() / phi


@pytest.mark.slow
def test_mie_sphere_tfsf(use_gpu):
    x, q_sca, q_abs = _run("all", use_gpu=use_gpu)
    q_mie = mie_efficiencies(N, x)[1]
    err = np.abs(q_sca / q_mie - 1)
    assert err.max() < 0.05, err
    assert np.abs(q_abs).max() < 0.05 * q_mie.max()
