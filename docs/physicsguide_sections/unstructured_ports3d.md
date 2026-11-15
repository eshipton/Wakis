# Lumped and waveguide ports for tetrahedral FIT

`wakis.unstructured.ports3d` adds discrete (lumped) ports on edge paths and waveguide (modal) ports on planar
boundary triangle sets to `UnstructuredFIT3D` (Voronoi Hodge), with S-parameter extraction, a frequency-domain
solve of the same discrete equations, and an exact transpose of the ported step. Tests:
`tests/test_101_unstructured_ports3d.py`; study: `studies/unstructured-ports3d/`.

```python
from wakis.unstructured.ports3d import (rectangle_section, extrude_section, PortPlane, ModalPort,
                                        LumpedPort, edge_path, attach_ports, ExtrudedLine, power_waves)
nodes, tris, tags = rectangle_section(a, b, h, wall=h / 2)          # tag 0: PEC frame, tag 1: inside
mesh = extrude_section(nodes, tris, tags, z)                        # right prisms -> 3 tets each
s = UnstructuredFIT3D(mesh, {0: "pec"}, bc="pmc", massless_tol=1e-6) # end faces stay open (natural)
P1 = PortPlane(s, (0, 0, z[0]), (0, 0, 1)); kc, phi = P1.te_modes(1)
line = ExtrudedLine(z, 1.0, 1.0, kc[0], s.dt)                        # discrete mode admittance
Y = lambda f: line.port_admittance(f).real / line.cos_half(f)
p1 = ModalPort(P1, phi[:, 0], Y, f0=10e9, i_src=pulse)              # g = Y(f0) termination
attach_ports(s, [p1, ...]); s.run(n)
V, I = p1.spectra(freqs); a, b = power_waves(V, I, Y(freqs))
```

## 1. A port is a rank-one boundary admittance

FIT unknowns of `UnstructuredFIT3D`: edge voltages $u_e=\int_e E\cdot dl$ and face fluxes $b_f$, with

$$b' = -C u,\qquad M_\varepsilon u' + G u = C^T M_\nu b - j,$$

$M_\varepsilon$, $M_\nu$ the diagonal circumcentric Hodges (Clemens & Weiland 2001), $G$ the conductance
(conductivity, absorbing boundary). A port adds an edge current $j$.

**Waveguide port.** On a boundary surface $\Gamma$ a tangential surface current $K$ enters the Galerkin
(Whitney) form of Ampère's law as $j_e=\int_\Gamma K\cdot W_e\,dA$ (Bossavit 1988). The tangential trace of the 3D
Whitney edge function $W_e$ of an edge of $\Gamma$ is the 2D Whitney function of the port triangle mesh, so the
3D voltages of the port edges *are* the 2D edge-element dofs (same edges, same orientation; `PortPlane` builds the
2D mesh with the node order preserved). Let $E_t\approx\sum_e u_e W_e$ on $\Gamma$ and let the port mode have
dofs $\phi$ normalized by the port metric, $\phi^T M_\Gamma\phi=1$, with
$M_\Gamma[e,e']=\int_\Gamma W_e\cdot W_{e'}\,dA$ (dimensionless). The modal amplitude is
$V=\int_\Gamma E_t\cdot e_m\,dA=\phi^T M_\Gamma u$. A modal impedance boundary $K = (g V - I_s)\,e_m$
(conductance $g$, impressed current $I_s$) gives

$$j = w\,(g\,V - I_s),\qquad V = w^T u,\qquad w = M_\Gamma\phi.$$

With the diagonal (Voronoi) Hodge, the lumped version of $M_\Gamma$ is $\mathrm{diag}(h_e/L_e)$ with $h_e$ the 2D
dual (circumcentric) length of edge $e$ in $\Gamma$: on an extruded mesh this is exactly the boundary half-cell
of the 3D Hodge, $M_{\varepsilon,e}=\varepsilon\,h_e\,(\Delta/2)/L_e$ (section 3). Both are available
(`PortPlane(hodge="voronoi" | "whitney")`).

**Lumped port.** An edge path from the + to the − terminal with orientation signs $s$; $w=s$ makes
$V=\sum s_e u_e=\int_+^- E\cdot dl=\varphi_+-\varphi_-$ and $j=s\,(V/R - I_s)$: every path edge carries the same
current, a resistor $R=1/g$ in parallel with a Norton current $I_s = v_s/R$ (an ideal source $v_s$ in series
with $R$; Piket-May et al. 1994 distribute $R$ edge by edge, here the path is one branch). The current
delivered into the structure is $I = I_s - gV$ for both kinds of port.

$w w^T$ is symmetric positive semi-definite, so every port is passive.

## 2. Time integration: trapezoidal conductance, solved exactly

The solver's update is $D_+ u^{n+1} = a_1 u^n + \Delta t\,r^{n+1/2}$, $D_\pm=M_\varepsilon \pm G\Delta t/2$,
$r=C^TM_\nu b^{n+1/2}-j_{src}$. The port conductance is integrated with the same trapezoidal rule, i.e. at the
time level $n+\tfrac12$ of the curl term and of the source:

$$\bigl(D_+ + \tfrac{\Delta t}{2}WgW^T\bigr)u^{n+1}=\bigl(a_1-\tfrac{\Delta t}{2}WgW^T\bigr)u^n+\Delta t\,(r+W I_s^{n+1/2}),$$

$W=[w_1\dots w_P]$ (free edges), $g=\mathrm{diag}(g_p)$. The rank-$P$ implicit term is solved exactly with the
Woodbury identity (Hager 1989): with $u^*$ the solver's own update of the right-hand side
and $Q=W^T\mathrm{diag}(\Delta t/D_+)W$,

$$u^{n+1}=u^*-\mathrm{diag}(\Delta t/D_+)\,W\alpha,\qquad (I+\tfrac g2 Q)\,\alpha=\tfrac g2\,W^Tu^*.$$

Cost: $P$ sparse dot products per step. **Energy.** Multiplying by $\bar u=(u^n+u^{n+1})/2$, the leapfrog's
conserved (modified) energy changes by $-\Delta t(\bar u^TG\bar u+\sum_p g_p V_p^2) + \Delta t\sum_p V_p I_{s,p}$
with $V_p = w_p^T\bar u$: the ports are dissipative for any $g\ge 0$ and do not change the CFL limit.
**Monitors** are recorded at the same level: $V^{n+1/2}=w^T\bar u$ and $I_s^{n+1/2}$, so the delivered power is
exactly $V^{n+1/2}I^{n+1/2}$ per step.

**Frequency response of the scheme.** With $u^n=Ue^{i\omega n\Delta t}$, $b^{n+1/2}=Be^{i\omega(n+1/2)\Delta t}$,
$\tilde\omega=\tfrac{2}{\Delta t}\sin\tfrac{\omega\Delta t}{2}$ and $c_h=\cos\tfrac{\omega\Delta t}{2}$ (time
convention $e^{+i\omega t}$, as `dtft`):

$$\Bigl[i\tilde\omega M_\varepsilon + c_h(G+WgW^T)+\frac{K}{i\tilde\omega}\Bigr]U = W I_s,\qquad V^{n+1/2}\to c_h W^T U,$$

$K=C^TM_\nu C$ with the massless edges kept as zero-capacitance rows (`PortSet.frequency_response`). The DTFT of
a time-domain run that has decayed reproduces it to the truncation level; this is the independent check of the
lumped port (a $V^n$ monitor would differ by $O(\omega\Delta t)$, a non-trapezoidal conductance by the
admittance factor $c_h$).

## 3. Exact mode separation on extruded meshes: the discrete ladder

`extrude_section` extrudes a triangulated cross-section along $z$ (levels $z_k$) into right prisms and splits
each into three tetrahedra with the min-index diagonal rule (Dompierre et al. 1999). The six vertices of a right
prism lie on one sphere, so the three tetrahedra share the circumcentre: the quad diagonals have zero dual area
(massless, eliminated by the solver), the inner prism faces zero dual length, and the circumcentric dual is the
tensor product of the 2D Voronoi dual of the section and the 1D dual of the $z$ levels (checked in
`test_extruded_dual_is_tensor_product`):

| object | dual measure |
|---|---|
| transverse edge $e$ at level $k$ | area $h_e\,(\Delta_{k-1/2}+\Delta_{k+1/2})/2$ (half at the end levels) |
| $z$ edge at node $i$ | area $A_i$ (2D Voronoi area) |
| triangle face at level $k$ | length $(\Delta_{k-1/2}+\Delta_{k+1/2})/2$ |
| side face of transverse edge $e$ between levels | length $h_e$ |

Let $\phi$ be a TE mode of the section, $K_2\phi = k_c^2 M_\Gamma\phi$ with $K_2 = C_2^T\mathrm{diag}(1/A_t)C_2$
(2D curl-curl, geometry only) and $M_\Gamma=\mathrm{diag}(h_e/L_e)$, homogeneous layers $(\varepsilon_l,\mu_l)$. Insert
$u=\sum_k V_k\,\phi$ (transverse edges of level $k$), zero on $z$ edges. Faraday gives the triangle fluxes
$b_k=-C_2\phi\int V_k$ and the side-face fluxes $\psi_{k+1/2}=-\phi\int(V_{k+1}-V_k)$; projecting Ampère's law on
$\phi$ (using $K_2\phi=k_c^2M_\Gamma\phi$, $\phi^TM_\Gamma\phi=1$) gives a ladder network with node voltages
$V_k$ (integer steps) and branch currents at the half steps,

$$Z_l = i\tilde\omega\,\mu_l\Delta_l,\qquad
Y_k=\sum_{l\,\mathrm{adj}\,k}\frac{\Delta_l}{2}\Bigl(i\tilde\omega\,\varepsilon_l+\frac{k_c^2}{i\tilde\omega\,\mu_l}\Bigr).$$

The $z$-edge equations are satisfied identically: their right-hand side is the discrete divergence
$G_2^TM_\Gamma\phi$ (times a layer factor), zero for $k_c\neq0$; for the TEM mode
$\phi=-G_2\psi$ of a two-conductor section ($\psi$ the $M_\Gamma$-harmonic potential, $k_c=0$) it vanishes at the
interior nodes and the conductor nodes carry PEC (fixed) $z$ edges. The reduction is therefore **exact** for the
time-stepped scheme, not an approximation (`ExtrudedLine`).

**Discrete dispersion** (uniform section): $V_k\propto e^{-i\theta k}$ gives $Y Z = -4\sin^2(\theta/2)$, i.e.

$$\Bigl(\frac{2}{\Delta}\sin\frac{\beta_h\Delta}{2}\Bigr)^2 = \tilde\omega^2\varepsilon\mu - k_{c,h}^2 ,$$

the Yee-type relation in $z$ and $t$ with the discrete cutoff $k_{c,h}$ of the section.

**Discrete mode admittance at a port plane.** The end node carries a half shunt. For a pure outgoing wave
$I_{1/2}=V_0(1-e^{-i\theta})/Z$ and $Y/2=-(1-\cos\theta)/Z$, so the current that the port must deliver is

$$Y_h = \frac{I}{V_0} = \frac{\sin\beta_h\Delta}{\tilde\omega\,\mu\,\Delta}
\quad\xrightarrow{\ \Delta,\Delta t\to0\ }\quad \frac{\beta}{\omega\mu}=Y_{TE}.$$

Referred to the recorded voltage $V^{n+1/2}=c_hV_0$ the admittance is $Y_h/c_h$ (`port_admittance / cos_half`).
For TEM ports the physical line impedance is $Z_{0,h} = v^2 c_h/Y_h$ with $v=1/\sqrt{\phi_1^TM_\Gamma\phi_1}$
the volts per modal unit ($\phi_1$: 1 V between the conductors).

## 4. S-parameters

Power waves (Kurokawa 1965) for a real reference admittance $Y$ (the mode admittance, *not* the terminating
conductance): $a = (\sqrt Y V + I/\sqrt Y)/2$, $b=(\sqrt Y V - I/\sqrt Y)/2$, $|a|^2-|b|^2=\mathrm{Re}(VI^*)$.
With $P$ ports and $P$ runs (port $k$ excited in run $k$), $S = BA^{-1}$ (`s_matrix`). The terminating
conductance $g$ only changes $A$; it need not be matched. A constant $g = Y_h(f_0)/c_h$ is matched at $f_0$
only (a broadband mode-matched termination needs the convolution with $Y_h(\omega)$, not implemented), so the
runs last until the multiply reflected waves have decayed; the tests check the residual energy ratio.

Because the ladder is exact, a uniform guide referred to $Y_h/c_h$ has $S_{11}=0$ and
$S_{21}=e^{-i N\theta}$ *to the DFT truncation level*; any mismatch in the port mode, the time levels or the
admittance shows up as a reflection floor. Against the continuum the errors are the derived $O(h^2)$ ones:
$k_{c,h}$ (2D), $\beta_h$ (dispersion relation above), and $Z_{0,h}$ for TEM.

## 5. Transpose (adjoint use)

The ported step $x^n=(u^n,b^{n-1/2}, I_s^{n+1/2})\mapsto(u^{n+1},b^{n+1/2})$ is linear;
`PortSet.linear_step` / `linear_step_T` implement it and its exact transpose ($A=D_++\tfrac{\Delta t}{2}WgW^T$ is
symmetric, so the Woodbury solve is its own transpose; the massless elimination contributes
$\Delta t\,K_{sf}^TK_{ss}^{-T}C_s^T\bar b$). Dot-product test to $10^{-15}$; a reverse-mode gradient of
$J=\sum_n (V_2^{n+1/2})^2$ with respect to the source samples matches central differences. The ported step is
not yet wired into `UnstructuredAdjointStepper` (material gradients with ports).

## 6. Verification

Numbers: Cloud Batch `w2ports-tests-*` (tests and mutations) and `w2ports-study-*` (convergence study, results
in `studies/unstructured-ports3d/out/results.json`).

Tests (`w2ports-tests-1`, n2-standard-8): 15 passed in 35 s. Study (`w2ports-study-1`, n2-standard-16, all
cases in parallel). WR-90 ($a\times b$ = 22.86 × 10.16 mm, $L$ = 40 mm, $\Delta = h/1.5$, jittered Delaunay
section, 8–12 GHz, 41 frequencies):

| h | tets | $k_{c,h}/k_c-1$ | guide $\max|S_{11}|$ | guide $|S_{21}-e^{-iN\theta}|$ | guide phase error vs $\beta L$ (rad) | slab $\max|S-S_{ladder}|$ | slab $\max|S_{11}-S_{11,c}|$ |
|---|---|---|---|---|---|---|---|
| a/4 | 1320 | −2.2e-2 | 9.4e-14 | 1.9e-13 | 0.28 | 4.0e-10 | 0.143 |
| a/6 | 3456 | −9.8e-3 | 1.5e-13 | 6.9e-13 | 0.13 | 5.6e-10 | 0.066 |
| a/8 | 7056 | −4.9e-3 | 8.9e-13 | 4.5e-12 | 0.075 | 7.3e-10 | 0.037 |
| a/12 | 20736 | −2.1e-3 | 1.5e-12 | 7.9e-12 | 0.032 | 8.2e-10 | 0.015 |
| a/16 | 44352 | −1.3e-3 | 1.6e-12 | 8.5e-12 | 0.019 | 7.5e-10 | 0.0091 |
| a/24 | 126252 | −5.3e-4 | 6.8e-13 | 3.6e-12 | 0.0082 | 8.7e-10 | 0.0040 |

* The guide phase error vs the continuum equals the ladder prediction $-(\beta_h-\beta)L$ to < 1e-11 rad at
  every resolution, and falls as $h^2$ (0.28 → 0.0082 rad for $h$ ÷ 6: ratio 34 vs 36). Slab: reciprocity
  ≤ 1.6e-15, unitarity ≤ 1.4e-9 (truncation; residual energy ≤ 6e-18 of the peak).
* Coax (2 / 4.6 mm, 100 Ω sheet load, 20 mm, 1–8 GHz): $|\Gamma-\Gamma_{ladder}|$ ≤ 1.1e-12;
  $Z_{0,h}/Z_0-1$ = −3.0e-2, −1.4e-2, −7.8e-3, −3.4e-3 and $\max|\Gamma-\Gamma_c|$ = 0.0135, 0.0064, 0.0035,
  0.0015 at $h$ = 1.2, 0.8, 0.6, 0.4 mm (ratio 8.9 for $h$ ÷ 3: second order).
* Lumped radial paths on a 10 mm coax (R = 50 Ω both ends, 6300 tets): $\max|S_{11}^{TD}-S_{11}^{FD}|$ =
  1.0, 1.2e-3, 1.3e-6, 1.3e-9 after 1, 2, 3, 4 ns of run (pure truncation, decaying with the residual
  field). DC limit with $R_L$ = 75 Ω: $\mathrm{Re}Z_{in}-R_L$ = 7.8e-8, 7.8e-7, 7.8e-6 Ω at 0.1, 0.32, 1 MHz
  ($\propto f^2$) and $\mathrm{Im}Z_{in}\propto f$ (4.2 nH loop inductance).
* Mutations (`studies/unstructured-ports3d/mutations.py`, all caught): $V^n$ monitor (6 tests fail), explicit
  conductance (7), backward-Euler conductance (6), continuum port admittance without the stencil factors (2),
  transpose without the massless term (2), unnormalized TEM mode (2). Control: 15 passed.
* Negative control in the suite: the Whitney-mass port mode on the Voronoi solver raises the straight-guide
  floor to between 1e-4 and 1e-1 (vs ~1e-12 matched).

## 7. Limits & Failure Modes

* **Exactness needs extruded meshes.** The discrete ladder, $Y_h$ and the $10^{-10}$ reflection floor hold for
  `extrude_section` meshes (right prisms, uniform material per $z$ layer, homogeneous cross-section for TE
  modes). On a general tetrahedral mesh the port mode from the 2D problem is not an exact discrete 3D mode near
  the port: expect an $O((kh)^2)$ reflection floor and use the continuum (or the ladder-predicted) admittance
  as reference. The Whitney-mass port mode on the Voronoi solver shows this floor already on extruded meshes
  (`test_mode_mismatch_raises_s11_floor`).
* **TE modes of homogeneous sections only.** `te_modes` solves the transverse curl-curl problem, which is exact
  for TE modes of hollow or homogeneously filled guides; TM modes ($E_z\neq0$) and inhomogeneous sections
  (dielectric waveguides, microstrip quasi-TEM) need the full $(E_t,E_z)$ formulation of Lee, Sun & Cendes
  (1991) and a frequency-dependent mode; not implemented. TEM: two conductors only.
* **Termination is matched at one frequency.** The constant modal conductance reflects
  $(Y_h/c_h-g)/(Y_h/c_h+g)$ away from $f_0$ (about 0.14 at 8 GHz for WR-90 matched at 10 GHz); the extraction
  is exact regardless, but run times grow near cutoff where $Y_h\to0$. Below the discrete cutoff $Y_h$ is
  imaginary and the power waves are not defined (evanescent ports).
* **Massless section edges.** `extrude_section(massless_rtol>0)` zeroes small section capacitances (larger
  $\Delta t$) but keeps their side-face reluctance, so the separation is no longer exact (measured $1.8\times10^{-3}$
  in $|\Gamma|$ on the test coax at 0.05). Zeroing the side faces too leaves the quad diagonals without any
  reluctance (singular massless block).
* **Not supported:** `hodge="whitney"` solvers (the Whitney update has no conductance path), the spherical PML,
  dispersive port edges, GPU (`wakis.unstructured.gpu`) and MPI paths; `edge_path` picks a shortest path of free
  edges, which may leave the intended plane.
* **Lumped feeds are not TEM sources.** A single radial path in a coax excites evanescent higher-order modes; its
  reactance is part of the discrete model (and of $S_{11}$), so a lumped port is validated against the same
  discrete equations (frequency-domain solve) and against the DC limit $Z_{in}\to R_L$, not against
  transmission-line theory.
* **Adjoint**: only the source-sample gradient is exercised; port conductances and material gradients through
  `UnstructuredAdjointStepper` with ports are open.

## References

* M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite Integration Technique," Prog.
  Electromagn. Res. 32, 65–87 (2001). doi:10.2528/PIER00080103
* A. Bossavit, "Whitney forms: a class of finite elements for three-dimensional computations in
  electromagnetism," IEE Proc. A 135(8), 493–500 (1988). doi:10.1049/ip-a-1.1988.0077
* K. Kurokawa, "Power waves and the scattering matrix," IEEE Trans. Microw. Theory Techn. 13(2), 194–202 (1965).
  doi:10.1109/TMTT.1965.1125964
* M. Piket-May, A. Taflove and J. Baron, "FD-TD modeling of digital signal propagation in 3-D circuits with
  passive and active loads," IEEE Trans. Microw. Theory Techn. 42(8), 1514–1523 (1994). doi:10.1109/22.297814
* J.-F. Lee, D.-K. Sun and Z. J. Cendes, "Full-wave analysis of dielectric waveguides using tangential vector
  finite elements," IEEE Trans. Microw. Theory Techn. 39(8), 1262–1271 (1991). doi:10.1109/22.85399
* W. W. Hager, "Updating the inverse of a matrix," SIAM Rev. 31(2), 221–239 (1989). doi:10.1137/1031049
* J. Dompierre, P. Labbé, M.-G. Vallet and R. Camarero, "How to subdivide pyramids, prisms and hexahedra into
  tetrahedra," Proc. 8th Int. Meshing Roundtable, 195–204 (1999).
* D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley (2012): TE10 and transmission-line formulas used as the
  continuum references.
