# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Benchmark case definitions (geometry, materials, band, source, extraction
settings, reference solution and metrics). See ``wakis/benchmarks/README.md``.

2D scattering cases (`Scattering2DCase`), cylinders along z at normal
incidence; u = E_z (TM) or H_z (TE):

* ``B1``  dielectric cylinder, n = 2, a = 1 um, x = k a = 2..10 (the shared
  mesh benchmark of the earlier studies).
* ``B2-Au`` gold nanocylinder, a = 20 nm, in a lossless host of index 2.0,
  lambda = 450..900 nm; ``B2-Ag`` silver nanocylinder, a = 20 nm, vacuum,
  lambda = 300..500 nm. Rakic et al. (1998) Lorentz-Drude models of
  `wakis.dispersive.dispersive_lib` in both solver and reference.
* ``B3-g<gap>`` gold dimer, a = 20 nm, gaps 2, 5 or 10 nm, host n = 2.0,
  TE with E along the dimer axis, lambda = 500..1000 nm; reference
  `wakis.analytic.multi_cylinder.CylinderCluster`; gap-centre field
  enhancement |E / E0|^2.
* ``B5`` PEC cylinder, a = 1 um, x = 0.5..10 (series in the PEC limit).

3D antenna case (`DipoleCase`): ``B4`` centre-fed thin-wire half-wave
dipole, see `wakis.benchmarks.dipole`.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.constants import c as c_light
from scipy.special import h1vp, hankel1, jv, jvp

from ..analytic.mie_cylinder import cylinder_coefficients
from ..analytic.multi_cylinder import CylinderCluster
from ..dispersive import DispersiveMaterial, dispersive_lib
from . import metrics as M
from .extraction import cross_sections_from_circles

__all__ = ["Scattering2DCase", "get_case", "list_cases", "CASES"]


def _eps(material, f):
    """Complex relative permittivity (Wakis exp(+i w t)) of a case material."""
    if isinstance(material, DispersiveMaterial):
        return material.epsilon(f)
    if isinstance(material, str):
        raise ValueError("PEC has no permittivity")
    return np.full(np.shape(f), complex(material[0]))


def pec_cylinder_coefficients(x, polarization, n_terms):
    """
    Scattering coefficients of a PEC cylinder (the |m| -> infinity limit).

    TM (E_z = 0 on the surface): c_n = J_n(x) / H_n(x);
    TE (d H_z / d r = 0): c_n = J_n'(x) / H_n'(x).
    """
    n = np.arange(n_terms + 1)
    if polarization.upper() == "TM":
        return jv(n, x) / hankel1(n, x)
    return jvp(n, x) / h1vp(n, x)


@dataclass
class Scattering2DCase:
    """
    A 2D plane-wave scattering benchmark.

    Attributes
    ----------
    name : str
    description : str
    cylinders : list of dict
        ``dict(center=(x, y), radius=a, material=key)`` [m].
    materials : dict
        ``{key: DispersiveMaterial | [eps_r, mu_r] | "pec"}``.
    n_background : float
        Index of the lossless host (fills the whole domain).
    polarization : {"TE", "TM"}
    direction : float
        Propagation angle [rad] (0: +x). TE: E_inc = (-sin, cos) direction.
    wavelengths : ndarray
        Vacuum wavelengths of the metric band [m] (increasing).
    circles : tuple of float
        Recommended extraction radii [m] (must enclose the scatterers and,
        for TF/SF arms, the total-field box).
    n_points : int
        Points per extraction circle.
    r_absorber : float
        Recommended inner radius of the absorber [m] (arms may use more).
    t_run : float
        Recommended DFT window [s] (the fields decay to < 1e-4 of their peak).
    norm_width : float
        Width used for the efficiencies Q = C / norm_width [m].
    gap_point : tuple, optional
        Centre of the field-enhancement metric (TE only).
    gap_width : float
        Gap g [m]; the enhancement ``enh`` is the mean of |E/E0|^2 over
        `gap_points` (the centre and 8 points on a circle of radius g/4),
        which is insensitive to where the centre falls inside a cell;
        ``enh_centre`` is the pointwise centre value (diagnostic).
    deembed : bool
        Rotationally symmetric scatterer centred on the origin.
    targets : dict
        Accuracy targets used for the cost-to-target tables.
    primary : str
        Headline metric name.
    """

    name: str
    description: str
    cylinders: list
    materials: dict
    n_background: float
    polarization: str
    direction: float
    wavelengths: np.ndarray
    circles: tuple
    n_points: int = 64
    r_absorber: float = 0.0
    t_run: float = 100e-15
    norm_width: float = 1.0
    gap_point: tuple = None
    gap_width: float = 0.0
    deembed: bool = True
    targets: dict = field(default_factory=dict)
    primary: str = "q_ext_nrms"

    # ---------------------------------------------------------------- band
    @property
    def freqs(self):
        """Frequencies of the metric band [Hz] (same order as wavelengths)."""
        return c_light / np.asarray(self.wavelengths)

    @property
    def k_background(self):
        return 2 * np.pi * self.n_background / np.asarray(self.wavelengths)

    def source(self):
        """
        Recommended Gaussian pulse (`wakis.sources.GaussianSource`) covering
        the band: centre at the band middle, spectral sigma = band / 4
        (amplitude >= e^-2 at the band edges).
        """
        from ..sources import GaussianSource

        f = self.freqs
        return GaussianSource(0.5 * (f.min() + f.max()), 0.25 * (f.max() - f.min()))

    @property
    def gap_points(self):
        """Stencil of the gap metric: centre + 8 points at radius g/4 [m]."""
        if self.gap_point is None:
            return None
        c = np.asarray(self.gap_point, float)
        th = 2 * np.pi * np.arange(8) / 8
        ring = c + 0.25 * self.gap_width * np.stack([np.cos(th), np.sin(th)], 1)
        return np.vstack([c, ring])

    @property
    def bounding_radius(self):
        return max(np.hypot(*cy["center"]) + cy["radius"] for cy in self.cylinders)

    # ----------------------------------------------------------- reference
    @cached_property
    def reference(self):
        """
        Exact reference spectra (dict of arrays over `wavelengths`).

        Keys: ``c_ext``, ``c_sca``, ``c_abs`` [m], ``q_ext``, ``q_sca``,
        ``q_abs`` (= C / norm_width) and, with a gap point, ``enh``
        (|E / E0|^2 at the gap point, TE).
        """
        out = {k: [] for k in ("c_ext", "c_sca", "c_abs", "enh", "enh_centre")}
        for f, k in zip(self.freqs, self.k_background):
            if len(self.cylinders) == 1 and np.allclose(self.cylinders[0]["center"], 0):
                cy = self.cylinders[0]
                mat = self.materials[cy["material"]]
                x = k * cy["radius"]
                n_t = int(np.ceil(x + 4 * x ** (1 / 3) + 10))
                if isinstance(mat, str):
                    cn = pec_cylinder_coefficients(x, self.polarization, n_t)
                else:
                    m = np.sqrt(complex(_eps(mat, f))) / self.n_background
                    cn = cylinder_coefficients(m, x, self.polarization, n_terms=n_t)
                w = np.where(np.arange(cn.size) > 0, 2.0, 1.0)
                ce = 4 / k * np.sum(w * cn.real)
                cs = 4 / k * np.sum(w * np.abs(cn) ** 2)
                out["c_ext"].append(ce)
                out["c_sca"].append(cs)
                out["c_abs"].append(ce - cs)
                if self.gap_point is not None:
                    raise NotImplementedError("gap point needs a cylinder cluster")
                continue
            cl = self.cluster(f)
            ce, cs, ca = cl.cross_sections()
            out["c_ext"].append(ce)
            out["c_sca"].append(cs)
            out["c_abs"].append(ca)
            if self.gap_point is not None:
                e2 = np.sum(np.abs(cl.in_plane_e(self.gap_points)) ** 2, 1)
                out["enh"].append(float(e2.mean()))
                out["enh_centre"].append(float(e2[0]))
        res = {k: np.asarray(v) for k, v in out.items() if v}
        for key in ("ext", "sca", "abs"):
            res["q_" + key] = res["c_" + key] / self.norm_width
        return res

    def cluster(self, f, n_max=None):
        """`CylinderCluster` at frequency `f` [Hz] (non-PEC cylinders)."""
        k = 2 * np.pi * f * self.n_background / c_light
        m = [
            np.sqrt(complex(_eps(self.materials[cy["material"]], f)))
            / self.n_background
            for cy in self.cylinders
        ]
        return CylinderCluster(
            [cy["center"] for cy in self.cylinders],
            [cy["radius"] for cy in self.cylinders],
            m, k, self.polarization, n_max=n_max, direction=self.direction,
        )  # fmt: skip

    # -------------------------------------------------------------- metrics
    def spectra_from_result(self, result):
        """
        Spectra (C and Q) of an arm result.

        `result` holds either ``samples`` (n_freq, n_radii, n_points)
        scattered-field phasors on circles of radii ``radii`` and ``u0`` (see
        `cross_sections_from_circles`), or the widths ``c_ext``, ``c_sca``
        directly. Gap fields: ``e_gap`` (n_freq, n_gap_points, 2) total
        in-plane E phasors at `gap_points` and ``e0_gap`` (n_freq,) incident
        amplitude phasors.
        """
        if "samples" in result:
            sp = cross_sections_from_circles(
                result["samples"], result["u0"], self.freqs, result["radii"],
                self.n_background, self.direction, deembed=self.deembed,
            )  # fmt: skip
        else:
            sp = dict(
                c_ext=np.asarray(result["c_ext"]), c_sca=np.asarray(result["c_sca"])
            )
            sp["c_abs"] = sp["c_ext"] - sp["c_sca"]
        for key in ("ext", "sca", "abs"):
            sp["q_" + key] = sp["c_" + key] / self.norm_width
        if result.get("e_gap") is not None:
            e = np.asarray(result["e_gap"])
            e0 = np.asarray(result["e0_gap"])
            if e.ndim == 2:  # legacy: centre only
                e = e[:, None, :]
            e2 = np.sum(np.abs(e) ** 2, 2) / np.abs(e0[:, None]) ** 2
            sp["enh"] = e2.mean(1)
            sp["enh_centre"] = e2[:, 0]
            if result.get("e_gap_rt") is not None:
                ert = np.asarray(result["e_gap_rt"])
                if ert.ndim == 2:
                    ert = ert[:, None, :]
                e2_rt = np.sum(np.abs(ert) ** 2, 2) / np.abs(e0[:, None]) ** 2
                sp["enh_rt"] = e2_rt.mean(1)
        return sp

    def evaluate(self, result):
        """
        All metrics of an arm result against the reference.

        Returns
        -------
        dict
            ``q_ext_nrms``, ``q_sca_nrms``, ``q_abs_nrms``, ``q_sca_rms_rel``,
            peak errors of Q_ext (``lam_peak``, ``height``, ``fwhm``) and,
            with a gap point, ``enh_nrms``, ``enh_peak`` (relative error of
            the maximum enhancement), ``enh_lam`` (its wavelength error),
            plus ``enh_rt_peak`` when ``e_gap_rt`` is provided;
            plus the spectra themselves under ``spectra``.
        """
        ref = self.reference
        sp = self.spectra_from_result(result)
        lam = np.asarray(self.wavelengths)
        out = {}
        for key in ("ext", "sca", "abs"):
            out[f"q_{key}_nrms"] = M.nrms(sp["q_" + key], ref["q_" + key])
        out["q_sca_rms_rel"] = M.rms_rel(sp["q_sca"], ref["q_sca"])
        pe = M.peak_errors(lam, sp["q_ext"], ref["q_ext"])
        out.update({f"q_ext_{k}": v for k, v in pe.items()})
        if "enh" in sp and "enh" in ref:
            out["enh_nrms"] = M.nrms(sp["enh"], ref["enh"])
            pe = M.peak_errors(lam, sp["enh"], ref["enh"])
            out["enh_peak"] = pe["height"]
            out["enh_lam"] = pe["lam_peak"]
            if "enh_centre" in sp:
                pc = M.peak_errors(lam, sp["enh_centre"], ref["enh_centre"])
                out["enh_centre_peak"] = pc["height"]
            if "enh_rt" in sp:
                prt = M.peak_errors(lam, sp["enh_rt"], ref["enh"])
                out["enh_rt_peak"] = prt["height"]
        if "refl" in sp:
            out["boundary_reflection"] = float(np.max(sp["refl"]))
        out["spectra"] = {
            k: np.asarray(v).tolist()
            for k, v in sp.items()
            if k in ("q_ext", "q_sca", "q_abs", "enh", "enh_centre", "enh_rt", "refl")
        }
        return out


# --------------------------------------------------------------------- cases
NM = 1e-9
UM = 1e-6


def _b1(pol):
    a = 1 * UM
    x = np.linspace(2.0, 10.0, 41)
    lam = np.sort(2 * np.pi * a / x)
    return Scattering2DCase(
        name=f"B1-{pol}",
        description="Dielectric cylinder n = 2, a = 1 um, x = 2..10 (shared Mie case)",
        cylinders=[dict(center=(0.0, 0.0), radius=a, material="diel")],
        materials={"diel": [4.0, 1.0]},
        n_background=1.0, polarization=pol, direction=0.0, wavelengths=lam,
        circles=(1.9 * UM, 2.2 * UM, 2.5 * UM), n_points=256, r_absorber=2.7 * UM,
        # 3000 fs: a high-Q TE resonance near x = 6.4 leaves a 0.87 % (TE) /
        # 0.14 % (TM) window floor at 1000 fs (converged DGTD, high-order
        # agent); 0.036 % / 0.012 % at 3000 fs
        t_run=3000e-15, norm_width=2 * a, deembed=True,
        targets=dict(q_sca_rms_rel=(0.02, 0.01)), primary="q_sca_rms_rel",
    )  # fmt: skip


def _b2(metal, pol):
    a = 20 * NM
    if metal == "Au":
        nb, lam = 2.0, np.linspace(450, 900, 91) * NM
    else:
        nb, lam = 1.0, np.linspace(300, 500, 81) * NM
    mat = dispersive_lib["gold" if metal == "Au" else "silver"]
    return Scattering2DCase(
        name=f"B2-{metal}-{pol}",
        description=(
            f"{metal} nanocylinder a = 20 nm (Rakic 1998 Lorentz-Drude) in a host "
            f"n = {nb}, {pol}"
        ),
        cylinders=[dict(center=(0.0, 0.0), radius=a, material=metal)],
        materials={metal: mat},
        n_background=nb, polarization=pol, direction=0.0, wavelengths=lam,
        circles=(50 * NM, 60 * NM, 70 * NM), n_points=64, r_absorber=90 * NM,
        t_run=80e-15, norm_width=2 * a, deembed=True,
        targets=dict(q_ext_nrms=(0.02, 0.01, 0.005), q_ext_lam_peak=(0.005, 0.002),
                     q_ext_height=(0.02, 0.01), q_ext_fwhm=(0.05, 0.02)),
        primary="q_ext_nrms",
    )  # fmt: skip


def _b3(gap_nm):
    a = 20 * NM
    g = gap_nm * NM
    xc = a + g / 2
    return Scattering2DCase(
        name=f"B3-g{gap_nm:g}",
        description=(
            f"Au dimer a = 20 nm, gap {gap_nm:g} nm, host n = 2.0, TE, E along the "
            "dimer axis (x), incidence along +y"
        ),
        cylinders=[dict(center=(-xc, 0.0), radius=a, material="Au"),
                   dict(center=(xc, 0.0), radius=a, material="Au")],
        materials={"Au": dispersive_lib["gold"]},
        n_background=2.0, polarization="TE", direction=np.pi / 2,
        wavelengths=np.linspace(500, 1000, 101) * NM,
        circles=(65 * NM, 80 * NM, 95 * NM), n_points=64, r_absorber=110 * NM,
        t_run=100e-15, norm_width=4 * a, gap_point=(0.0, 0.0), gap_width=g,
        deembed=False,
        targets=dict(q_ext_nrms=(0.02, 0.01), q_ext_lam_peak=(0.005, 0.002),
                     enh_peak=(0.05, 0.02), enh_lam=(0.005, 0.002)),
        primary="enh_peak",
    )  # fmt: skip


def _b5(pol):
    a = 1 * UM
    x = np.linspace(0.5, 10.0, 39)
    return Scattering2DCase(
        name=f"B5-{pol}",
        description="PEC cylinder a = 1 um, x = 0.5..10 (series in the PEC limit)",
        cylinders=[dict(center=(0.0, 0.0), radius=a, material="pec")],
        materials={"pec": "pec"},
        n_background=1.0, polarization=pol, direction=0.0,
        wavelengths=np.sort(2 * np.pi * a / x),
        circles=(1.5 * UM, 1.8 * UM, 2.1 * UM), n_points=128, r_absorber=2.3 * UM,
        t_run=400e-15, norm_width=2 * a, deembed=True,
        targets=dict(q_ext_nrms=(0.02, 0.01)), primary="q_ext_nrms",
    )  # fmt: skip


def _registry():
    reg = {}
    for pol in ("TM", "TE"):
        reg[f"B1-{pol}"] = lambda pol=pol: _b1(pol)
        for metal in ("Au", "Ag"):
            reg[f"B2-{metal}-{pol}"] = lambda m=metal, pol=pol: _b2(m, pol)
        reg[f"B5-{pol}"] = lambda pol=pol: _b5(pol)
    for g in (2, 5, 10):
        reg[f"B3-g{g}"] = lambda g=g: _b3(g)
    for name, gap in (("B4", 1e-3), ("B4-g2", 2e-3), ("B4-g4", 4e-3)):
        reg[name] = lambda name=name, gap=gap: __import__(
            "wakis.benchmarks.dipole", fromlist=["DipoleCase"]
        ).DipoleCase(name=name, gap=gap)
    return reg


CASES = _registry()


def list_cases():
    """Names of all registered cases."""
    return sorted(CASES)


def get_case(name):
    """Build the case `name` (see `list_cases`)."""
    try:
        return CASES[name]()
    except KeyError:
        raise KeyError(f"unknown case {name!r}; available: {list_cases()}") from None
