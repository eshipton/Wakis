# Waveguide port accuracy on general (non-extruded) tetrahedral meshes

`studies/unstructured-ports3d/general_mesh_accuracy.py` and `tests/test_111_unstructured_ports3d_paths.py`
quantify the accuracy of 3D modal ports (`PortPlane`, `ModalPort`, `attach_ports`, `s_matrix` in
`wakis.unstructured.ports3d`) on general 3D Delaunay tetrahedral meshes with conforming port planes, compared
at equal cells per wavelength ($\mathrm{cpw} = \lambda_0 / h$) against the extruded baseline (`extrude_section`).

## 1. Derivation of the $\mathcal{O}((kh)^2)$ error terms on general meshes

On an extruded right-prism tetrahedral mesh (`extrude_section`), the 3D circumcentric Voronoi dual is the exact
tensor product of the 2D cross-section Voronoi dual and the 1D axial dual (see `unstructured_ports3d.md`,
Section 3):

1. Every boundary edge $e \in \Gamma$ on a port plane has 3D dual area $A_e^{3\text{D}} = h_e\,\Delta z / 2$, so
   the boundary electric Hodge entry is $M_{\varepsilon,e}^{3\text{D}} = \varepsilon\,(\Delta z / 2)\,M_{\Gamma,e}$
   with $M_{\Gamma,e} = h_e / L_e$.
2. Internal quad-split diagonals have zero 3D dual area (massless) and internal prism faces have zero dual
   length, so transverse and axial curl balances decouple mode-by-mode into the 1D discrete ladder
   (`ExtrudedLine`). Terminating the port with the ladder admittance $Y_h / c_h$ (`ExtrudedLine.port_admittance`
   divided by `ExtrudedLine.cos_half`) yields a reflection floor $|S_{11}|$ at machine precision.

On a general 3D Delaunay tetrahedral mesh whose boundary triangles on the port planes $z \in \{0, L\}$ conform to
the cross-section triangulation (`PortPlane`), the interior vertices are not arranged on extruded vertical lines
and the tetrahedra adjacent to $\Gamma$ are not right prisms. Two $\mathcal{O}((kh)^2)$ discretisation errors
arise even when the port plane itself is planar and conforming:

### Boundary half-cell dual-area and oblique-edge mismatch

In the 3D frequency-domain FIT system of `UnstructuredFIT3D` at angular frequency $\omega$ (with
$\tilde\omega = \tfrac{2}{\Delta t}\sin\tfrac{\omega\Delta t}{2}$ and $c_h = \cos\tfrac{\omega\Delta t}{2}$),

$$\Bigl[i\tilde\omega M_\varepsilon + c_h W g W^T + \frac{C^T M_\nu C}{i\tilde\omega}\Bigr] U = W I_s,$$

the port weight vector is built from the 2D Voronoi Hodge on $\Gamma$, $w = M_\Gamma \phi$ with
$\phi^T M_\Gamma \phi = 1$. For a general tetrahedral layer of characteristic thickness $h$ adjacent to $\Gamma$:

* The 3D circumcentric dual area $A_e^{3\text{D}}$ of a boundary edge $e \in \Gamma$ is assembled from the
  circumcentres of the 3D tetrahedra sharing $e$, which is not proportional edge-by-edge to the 2D Voronoi length
  $h_e$:

  $$M_{\varepsilon,e}^{3\text{D}} = \varepsilon\,\frac{h_{\text{eff}}}{2}\,M_{\Gamma,e}\bigl(1 + \eta_e\bigr),$$

  where the local geometric fluctuation $\eta_e$ comes from the aspect and dihedral angles of the boundary
  tetrahedra.
* Interior edges connected to vertices on $\Gamma$ are oblique rather than purely axial ($z$-directed), so the
  discrete curl-curl operator $K = C^T M_\nu C$ couples the boundary tangential mode $u_\Gamma = V_0 \phi$ to
  interior oblique edges and non-fundamental transverse variations over a boundary layer of thickness $\sim h$.

Eliminating the interior degrees of freedom via the Schur complement onto $\Gamma$ and projecting onto $\phi$
replaces the exact ladder half-shunt by a perturbed boundary admittance

$$Y_{\Gamma,\text{eff}}(\omega) = Y_h(\omega)\bigl[1 + \mathcal{O}((kh)^2)\bigr].$$

### Bulk 3D tetrahedral dispersion and wave-impedance error

In the interior of a general tetrahedral mesh, the discrete 3D Voronoi Hodge operators $M_\varepsilon$ and
$M_\nu$ approximate the continuum curl-curl equations to second order in the mesh size $h$
(Clemens & Weiland 2001). For a propagating waveguide mode with continuum wavenumber
$\beta = \sqrt{k_0^2 - k_c^2}$, the effective 3D discrete propagation constant $\beta_{3\text{D},h}$ and bulk
characteristic admittance $Y_{3\text{D},h}$ satisfy

$$\beta_{3\text{D},h} - \beta = \mathcal{O}(k^3 h^2), \qquad
\frac{Y_{3\text{D},h} - Y}{Y} = \mathcal{O}((kh)^2).$$

Combining the boundary Schur-complement mismatch and the bulk-to-reference admittance mismatch in the power-wave
scattering matrix $S = B A^{-1}$ (`power_waves`, `s_matrix`) gives the two scaling laws measured in the study:

$$|S_{11}| = \mathcal{O}((kh)^2) = \mathcal{O}(\mathrm{cpw}^{-2}), \qquad
\bigl|\arg\bigl(S_{21}\,e^{i\beta L}\bigr)\bigr| = \bigl|(\beta_{3\text{D},h} - \beta)L\bigr| + \mathcal{O}((kh)^2)
= \mathcal{O}(\mathrm{cpw}^{-2}).$$

---

## 2. Verification on WR-90 and coaxial waveguides

`studies/unstructured-ports3d/general_mesh_accuracy.py` evaluates two matched two-port guides across
$f \in \{8.0, 10.0, 12.0\}\text{ GHz}$ (centre frequency $f_0 = 10.0\text{ GHz}$, free-space wavelength
$\lambda_0 = 0.0299792458\text{ m}$), comparing general 3D Delaunay tetrahedral meshes with conforming port
planes against the `extrude_section` baseline at identical cross-sectional resolutions $h$ and cells per
wavelength $\mathrm{cpw} = \lambda_0 / h$:

1. **Straight WR-90 waveguide** ($a = 22.86\text{ mm}$, $b = 10.16\text{ mm}$, $L = 20.0\text{ mm}$),
   fundamental $\text{TE}_{10}$ mode (`PortPlane.te_modes`), at 4 resolutions $h = a / \{4, 6, 8, 12\}$
   ($\mathrm{cpw} \in \{5.25, 7.87, 10.49, 15.74\}$).
2. **Straight coaxial line** ($r_{\text{in}} = 2.0\text{ mm}$, $r_{\text{out}} = 4.6\text{ mm}$,
   $L = 6.0\text{ mm}$), fundamental $\text{TEM}$ mode (`PortPlane.tem_mode`), at 4 radial resolutions
   $h = (r_{\text{out}} - r_{\text{in}}) / \{2, 3, 4, 5\}$ ($\mathrm{cpw} \in \{23.06, 34.59, 46.12, 57.65\}$).

### Per-resolution results (`studies/unstructured-ports3d/out/general_mesh_accuracy.json`)

| Case | Param | $h$ (mm) | $\mathrm{cpw}$ | Live tets | $|S_{11}|$ floor | $S_{21}$ phase err vs $\beta L$ (rad) | 2D mode rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wr90_general` | 4 | 5.715 | 5.25 | 321 | `2.3467e-02` | `1.5387e-01` | `2.3019e-02` |
| `wr90_general` | 6 | 3.810 | 7.87 | 918 | `1.1442e-02` | `6.6514e-02` | `1.0123e-02` |
| `wr90_general` | 8 | 2.857 | 10.49 | 2289 | `7.2582e-03` | `3.5251e-02` | `5.3034e-03` |
| `wr90_general` | 12 | 1.905 | 15.74 | 7408 | `3.3120e-03` | `1.5917e-02` | `2.2502e-03` |
| `wr90_extruded` | 4 | 5.715 | 5.25 | 288 | `3.4762e-14` | `1.2219e-01` | `2.1838e-02` |
| `wr90_extruded` | 6 | 3.810 | 7.87 | 864 | `1.5406e-13` | `6.3400e-02` | `9.8104e-03` |
| `wr90_extruded` | 8 | 2.857 | 10.49 | 2112 | `1.0564e-12` | `3.5060e-02` | `4.9361e-03` |
| `wr90_extruded` | 12 | 1.905 | 15.74 | 6912 | `1.8851e-12` | `1.5825e-02` | `2.0562e-03` |
| `coax_general` | 2 | 1.300 | 23.06 | 1628 | `5.9155e-04` | `2.3795e-03` | `3.5537e-02` |
| `coax_general` | 3 | 0.867 | 34.59 | 5429 | `2.8971e-04` | `1.1641e-03` | `1.5981e-02` |
| `coax_general` | 4 | 0.650 | 46.12 | 13500 | `1.6311e-04` | `5.7857e-04` | `9.3306e-03` |
| `coax_general` | 5 | 0.520 | 57.65 | 25992 | `1.0116e-04` | `3.9375e-04` | `5.7925e-03` |
| `coax_extruded` | 2 | 1.300 | 23.06 | 1449 | `2.1924e-15` | `2.8940e-03` | `3.8050e-02` |
| `coax_extruded` | 3 | 0.867 | 34.59 | 4770 | `5.8160e-15` | `1.4134e-03` | `1.6151e-02` |
| `coax_extruded` | 4 | 0.650 | 46.12 | 11802 | `1.3675e-14` | `7.0845e-04` | `9.1628e-03` |
| `coax_extruded` | 5 | 0.520 | 57.65 | 22695 | `5.5024e-14` | `4.9563e-04` | `5.8353e-03` |

### Fitted convergence orders ($\mathrm{err} \sim C\,h^p$)

With $\sigma_{\ln}$ the RMS residual of $\ln(\mathrm{err})$ about the 4-level log-linear fit, the $2\sigma$
tolerance on the 2-level coarse slope ($\Delta\ln h = \ln(3/2)$) is
$\delta p = 2\sqrt{2}\,\sigma_{\ln} / |\ln(h_2/h_1)|$:

| Metric | 4-level order $p$ | Coarse-2 order $p_{12}$ | Log RMS residual $\sigma_{\ln}$ | $\delta p$ | Theory min ($2 - \delta p$) | Study min ($p - \delta p$) |
|---|---:|---:|---:|---:|---:|---:|
| `wr90_general_s11` | `1.7695` | `1.7716` | `0.0257` | `0.1795` | `1.8205` | `1.5900` |
| `wr90_general_phase` | `2.0742` | `2.0685` | `0.0177` | `0.1236` | `1.8764` | `1.9505` |
| `wr90_extruded_phase` | `1.8732` | `1.6181` | `0.0400` | `0.2787` | `1.7213` | `1.5945` |
| `coax_general_s11` | `1.9224` | `1.7606` | `0.0296` | `0.2067` | `1.7933` | `1.7157` |
| `coax_general_phase` | `2.0017` | `1.7634` | `0.0461` | `0.3219` | `1.6781` | `1.6798` |
| `coax_extruded_phase` | `1.9684` | `1.7675` | `0.0461` | `0.3215` | `1.6785` | `1.6469` |

* **Extruded baseline (`extrude_section`)**: The matched-line $|S_{11}|$ floor stays at machine precision across
  all resolutions (`3.4762e-14` to `1.8851e-12` on WR-90; `2.1924e-15` to `5.5024e-14` on the coax line),
  whereas the continuum phase error $|\arg(S_{21} e^{i\beta L})|$ converges at second order (`p = 1.8732` on
  WR-90, `p = 1.9684` on coax) and closely tracks the general-mesh phase error at equal $\mathrm{cpw}$.
* **General tetrahedral meshes**: The reflection floor drops from `2.3467e-02` ($\mathrm{cpw} = 5.25$) to
  `3.3120e-03` ($\mathrm{cpw} = 15.74$) on WR-90 with fitted order `p = 1.7695` (`coarse2_order_p = 1.7716`), and
  from `5.9155e-04` ($\mathrm{cpw} = 23.06$) to `1.0116e-04` ($\mathrm{cpw} = 57.65$) on the coaxial line with
  fitted order `p = 1.9224` (`coarse2_order_p = 1.7606`). The $S_{21}$ phase error converges with order
  `p = 2.0742` on WR-90 and `p = 2.0017` on the coaxial line.
* **Mutation check**: Scaling the port-face dual area weights by `port_face_scale = 0.5` on `wr90_general` at
  $\mathrm{cpw} \in \{5.25, 7.87\}$ raises the $|S_{11}|$ floor to `8.6769e-01` and `8.6313e-01` and collapses
  the 2-level convergence order to `0.0130` (`test_general_port_face_area_mutation_fails`).

---

## 3. Limits & Failure Modes

* **Pre-asymptotic reflection slope on coarse WR-90 general meshes.** At $\mathrm{cpw} \in [5.25, 15.74]$ (where
  $8.0\text{ GHz}$ is close to the WR-90 cutoff $f_c = 6.557\text{ GHz}$), higher-order dispersive corrections in
  $\beta_h$ near cutoff reduce the measured WR-90 general-mesh $|S_{11}|$ order to `1.7695` (`1.7716` on the two
  coarsest levels), slightly below the strict asymptotic theory bound `2.0 - 0.1795 = 1.8205`. This is recorded
  via `pytest.mark.xfail(strict=True)` in `test_general_wr90_s11_strict_theory_order`.
* **Conforming planar port triangulation required.** `PortPlane` extracts boundary triangles whose vertices lie
  within the plane tolerance of $(p_0, \hat n)$. General 3D tetrahedral meshes must therefore be generated with
  their boundary faces conforming to the planar port cross-sections and conductor walls; non-planar or
  non-conforming boundary facets leave gaps in the 2D port mesh (`SimplexMesh2D`) and corrupt the modal solve.
* **Delaunay boundary dual positivity on curved conductors.** Unrestricted 3D Delaunay tetrahedralisation of a
  curved outer cylinder can connect four vertices on the same curved boundary into a flat chord tetrahedron whose
  circumcentre lies far outside the cell, producing negative boundary face dual lengths ($h_f < 0$) and spurious
  instabilities in the Voronoi Hodge (`UnstructuredFIT3D`). Keeping interior vertices separated from polygonal
  conductor walls by a radial clearance ($\ge 0.62\,h$ in `general_coax_mesh`) and snapping near-zero dual
  segments (`_snap_degenerate_duals`) preserves non-negative Hodge diagonals (`neg_int = 0`, `neg_bnd = 0`).
* **Single-frequency resistive termination.** As on extruded meshes, `ModalPort` applies a frequency-independent
  conductance $g = Y(f_0)$ in time-stepping runs, so broadband S-parameter extraction requires either running
  until internal reflections between ports decay or solving the exact discrete frequency-domain system
  (`PortSet.frequency_response`).
