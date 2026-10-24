# Wikimedia Commons & Physics Diagram Attributions — Tutorial 2.1 (`02-01_dispersion_physics`)

## 1. Audited Wikimedia Commons Candidate Files

The following candidate figures on Wikimedia Commons (linked from the English Wikipedia articles [*Dielectric spectroscopy*](https://en.wikipedia.org/wiki/Dielectric_spectroscopy), [*Permittivity*](https://en.wikipedia.org/wiki/Permittivity), and [*Dielectric loss*](https://en.wikipedia.org/wiki/Dielectric_loss)) were inspected via their Commons file description pages:

| Commons File | Title | Author / Uploader | Licence | Commons URL | Disposition in Tutorial 2.1 |
|---|---|---|---|---|---|
| `File:Dielectric_responses.svg` | *Dielectric responses* | Vectorized by [User:Pieter Kuiper](https://commons.wikimedia.org/wiki/User:Pieter_Kuiper) after [User:Archimerged](https://en.wikipedia.org/wiki/User:Archimerged) (adapted from Hewlett Packard Co. Application Note 1217-1, *Basics of Measuring the Dielectric Properties of Materials*, 1992) | Wikimedia Commons `Attribution` (`Category:Attribution only license`; adapted from 1992 Hewlett Packard Application Note 1217-1) | <https://commons.wikimedia.org/wiki/File:Dielectric_responses.svg> | **Redrawn from Wakis physics models (`dielectric_responses_wakis.png`)**: the Commons file uses a custom attribution tag derived from a proprietary 1992 application note and shows qualitative hand-drawn curves. We compute the full broadband spectrum ($\varepsilon'(f), \varepsilon''(f)$ from $1\;\text{MHz}$ to $30\;\text{PHz}$) directly from Wakis's `Debye`, `Lorentz`, and `Drude` classes and credit the layout as *"after File:Dielectric_responses.svg on Wikimedia Commons"*. |
| `File:Loss_tangent_phasors_1.svg` | *Loss tangent phasors 1* | [User:Sakurambo](https://commons.wikimedia.org/wiki/User:Sakurambo) (vectorized from `File:Loss_tangent_phasors_1.png` by User:LSTech) | **Public Domain** (`PD-self`, [Public Domain](https://en.wikipedia.org/wiki/Public_domain)) | <https://commons.wikimedia.org/wiki/File:Loss_tangent_phasors_1.svg> | **Redrawn in `wakis.figstyle` (`dielectric_responses_wakis.png`, Panel b)**: complex current-density phasor diagram $\hat{\mathbf{J}}_{\text{tot}} = \omega\varepsilon_0\varepsilon''\hat{\mathbf{E}} + i\omega\varepsilon_0\varepsilon'\hat{\mathbf{E}}$ with loss angle $\delta$ and $\tan\delta = \varepsilon''/\varepsilon'$ in Wakis's $\exp(+i\omega t)$ convention, credited as *"after File:Loss_tangent_phasors_1.svg (User:Sakurambo / User:LSTech, Public Domain)"*. |
| `File:Diel.png` | *Diel* | HyperPhysics (uploaded by [User:GifTagger](https://commons.wikimedia.org/wiki/User:GifTagger)) | `PD ineligible` ([Public Domain](https://en.wikipedia.org/wiki/Public_domain)) | <https://commons.wikimedia.org/wiki/File:Diel.png> | Inspected; dipole alignment mechanism incorporated into the orientational `Debye` band annotation of `dielectric_responses_wakis.png`. |

---

## 2. Generated Figure in `figures/wikimedia/`

- **File name**: `dielectric_responses_wakis.png`
- **Title**: *Broadband dielectric dispersion mechanisms ($\varepsilon'(f), \varepsilon''(f)$ from RF to extreme UV) and loss-tangent phasor diagram mapped to Wakis `Debye`, `Lorentz`, and `Drude` poles*
- **Author**: Wakis contributors (generated programmatically by `dispersion_physics.py` using `wakis.dispersive` and `wakis.figstyle`)
- **Licence**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / CERN Wakis package licence
- **Source & Attribution**:
  - Panel (a) layout after [`File:Dielectric_responses.svg`](https://commons.wikimedia.org/wiki/File:Dielectric_responses.svg) (vectorized by Pieter Kuiper after User:Archimerged, Wikimedia Commons); curves computed analytically from `wakis.dispersive.Debye`, `wakis.dispersive.Lorentz`, and `wakis.dispersive.Drude` following C. Kittel, *Introduction to Solid State Physics*, 8th ed. (Wiley, 2005), ch. 16, and J. D. Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, 1999), ch. 7.
  - Panel (b) layout after [`File:Loss_tangent_phasors_1.svg`](https://commons.wikimedia.org/wiki/File:Loss_tangent_phasors_1.svg) by User:Sakurambo / User:LSTech ([Public Domain](https://en.wikipedia.org/wiki/Public_domain), Wikimedia Commons).
- **Modifications**: Original numerical redraw in `wakis.figstyle` using Wakis's $\exp(+i\omega t)$ sign convention ($\varepsilon = \varepsilon' - i\varepsilon''$, $\varepsilon'' = -\operatorname{Im}\varepsilon \ge 0$).
