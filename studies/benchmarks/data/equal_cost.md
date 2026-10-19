## B2-Au-TE: q_ext_nrms <= 0.01

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00576 | 45,783 | 2.02e+09 | 86 |
| cart-conformal-all-uniform | 7 | 0.00552 | 170,358 | 7.4e+09 | 193 |
| cart-staircase-graded | 7 | 0.0275 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0271 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00166 | 5,607 | 4.83e+07 | 3 |
| nfit-ogrid | 7 | 0.000762 | 2,927 | 5.68e+07 | 3 |
| nfit-ogrid-r0.5 | 7 | 0.000612 | <2,817 | <4.98e+07 | <3 |
| subgrid-2to1 | 7 | 0.00883 | 24,852 | 1.56e+09 | 162 |
| subgrid-2to1-lts4 | 4 | 0.0088 | 24,782 | 3.87e+08 | 50 |
| subgrid-2to1-ppw40 | 7 | 0.00824 | 33,477 | 2.03e+09 | 200 |
| tri-graded | 7 | 0.00114 | 4,845 | 6.47e+07 | 5 |
| tri-graded-pml | 7 | 0.00114 | 4,845 | 6.47e+07 | 5 |
| tri-graded-pml-lts2 | 7 | 0.00113 | 4,792 | 3.18e+07 | 3 |
| tri-graded-repro | 7 | 0.00114 | 4,845 | 6.47e+07 | 5 |
| tri-uniform | 7 | 0.00014 | <13,195 | <1.67e+08 | <9 |
| voronoi-poly | 7 | 0.00502 | 7,711 | 3.77e+08 | 22 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 9.45x / 31.20x / 15.87x
- tri-graded-pml: 9.45x / 31.20x / 16.75x
- tri-graded-pml-lts2: 9.55x / 63.44x / 24.81x
- tri-graded-repro: 9.45x / 31.20x / 15.85x
- tri-uniform: 3.47x / 12.12x / 9.17x

## B2-Au-TE: q_ext_nrms <= 0.001

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00576 | nan | nan | nan |
| cart-conformal-all-uniform | 7 | 0.00552 | nan | nan | nan |
| cart-staircase-graded | 7 | 0.0275 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0271 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00166 | nan | nan | nan |
| nfit-ogrid | 7 | 0.000762 | 42,176 | 4.56e+09 | 261 |
| nfit-ogrid-r0.5 | 7 | 0.000612 | 35,333 | 3.21e+09 | 143 |
| subgrid-2to1 | 7 | 0.00883 | nan | nan | nan |
| subgrid-2to1-lts4 | 4 | 0.0088 | nan | nan | nan |
| subgrid-2to1-ppw40 | 7 | 0.00824 | nan | nan | nan |
| tri-graded | 7 | 0.00114 | nan | nan | nan |
| tri-graded-pml | 7 | 0.00114 | nan | nan | nan |
| tri-graded-pml-lts2 | 7 | 0.00113 | nan | nan | nan |
| tri-graded-repro | 7 | 0.00114 | nan | nan | nan |
| tri-uniform | 7 | 0.00014 | 87,971 | 2.58e+09 | 72 |
| voronoi-poly | 7 | 0.00502 | nan | nan | nan |

## B2-Au-TM: q_ext_nrms <= 0.01

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00012 | <8,112 | <7.26e+07 | <5 |
| cart-conformal-all-uniform | 7 | 1.06e-05 | <15,552 | <1.39e+08 | <10 |
| cart-staircase-graded | 7 | 0.0296 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0294 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.000138 | <5,445 | <4.66e+07 | <2 |
| nfit-ogrid | 7 | 2.19e-05 | <2,401 | <4.24e+07 | <3 |
| nfit-ogrid-r0.5 | 7 | 3.14e-05 | <2,785 | <4.92e+07 | <4 |
| subgrid-2to1 | 7 | 0.00143 | <5,957 | <5.24e+07 | <3 |
| subgrid-2to1-lts4 | 4 | 0.000548 | <11,209 | <6.99e+07 | <3 |
| subgrid-2to1-ppw40 | 7 | 0.000297 | <11,285 | <9.93e+07 | <3 |
| tri-graded | 7 | 0.000297 | 6,905 | 1.14e+08 | 6 |
| tri-graded-pml | 7 | 0.000297 | 6,905 | 1.14e+08 | 7 |
| tri-graded-pml-lts2 | 7 | 0.000328 | 6,805 | 5.57e+07 | 4 |
| tri-graded-repro | 7 | 0.000297 | 6,905 | 1.14e+08 | 9 |
| tri-uniform | 7 | 3.7e-05 | <13,053 | <1.65e+08 | <6 |
| voronoi-poly | 7 | 0.000993 | <2,142 | <2.79e+07 | <3 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 1.17x / 0.64x / 0.80x
- tri-graded-pml: 1.17x / 0.64x / 0.72x
- tri-graded-pml-lts2: 1.19x / 1.30x / 1.31x
- tri-graded-repro: 1.17x / 0.64x / 0.52x
- tri-uniform: 0.62x / 0.44x / 0.78x

## B2-Au-TM: q_ext_nrms <= 0.001

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00012 | 20,944 | 5.39e+08 | 29 |
| cart-conformal-all-uniform | 7 | 1.06e-05 | <15,552 | <1.39e+08 | <10 |
| cart-staircase-graded | 7 | 0.0296 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0294 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.000138 | 15,594 | 4.71e+08 | 13 |
| nfit-ogrid | 7 | 2.19e-05 | 7,004 | 2.44e+08 | 9 |
| nfit-ogrid-r0.5 | 7 | 3.14e-05 | 9,225 | 3.39e+08 | 13 |
| subgrid-2to1 | 7 | 0.00143 | nan | nan | nan |
| subgrid-2to1-lts4 | 4 | 0.000548 | <11,209 | <6.99e+07 | <3 |
| subgrid-2to1-ppw40 | 7 | 0.000297 | <11,285 | <9.93e+07 | <3 |
| tri-graded | 7 | 0.000297 | 12,358 | 3.18e+08 | 12 |
| tri-graded-pml | 7 | 0.000297 | 12,358 | 3.18e+08 | 12 |
| tri-graded-pml-lts2 | 7 | 0.000328 | 11,617 | 1.42e+08 | 6 |
| tri-graded-repro | 7 | 0.000297 | 12,358 | 3.18e+08 | 12 |
| tri-uniform | 7 | 3.7e-05 | 21,029 | 2.89e+08 | 9 |
| voronoi-poly | 7 | 0.000993 | 8,752 | 4.65e+08 | 22 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-uniform): unknowns / work / time
- tri-graded: 1.26x / 0.44x / 0.80x
- tri-graded-pml: 1.26x / 0.44x / 0.85x
- tri-graded-pml-lts2: 1.34x / 0.98x / 1.64x
- tri-graded-repro: 1.26x / 0.44x / 0.82x
- tri-uniform: 0.74x / 0.48x / 1.05x

## B2-Ag-TE: q_ext_nrms <= 0.01

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00538 | 42,258 | 1.77e+09 | 78 |
| cart-conformal-all-uniform | 7 | 0.00516 | 153,753 | 6.32e+09 | 202 |
| cart-staircase-graded | 7 | 0.0243 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0239 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00149 | 5,335 | 5.47e+07 | 4 |
| nfit-ogrid | 7 | 0.000698 | 2,789 | 5.28e+07 | 3 |
| nfit-ogrid-r0.5 | 7 | 0.000567 | <2,817 | <4.98e+07 | <3 |
| subgrid-2to1 | 7 | 0.00746 | 20,898 | 1.21e+09 | 188 |
| subgrid-2to1-lts4 | 4 | 0.00742 | 20,761 | 2.99e+08 | 35 |
| subgrid-2to1-ppw40 | 7 | 0.00703 | 21,445 | 1.11e+09 | 101 |
| tri-graded | 7 | 0.00104 | 4,796 | 6.47e+07 | 5 |
| tri-graded-pml | 7 | 0.00104 | 4,796 | 6.47e+07 | 6 |
| tri-graded-pml-lts2 | 6 | 0.00104 | 5,043 | 3.63e+07 | 3 |
| tri-graded-repro | 7 | 0.00104 | 4,796 | 6.47e+07 | 5 |
| tri-uniform | 7 | 0.000132 | <13,195 | <1.67e+08 | <6 |
| voronoi-poly | 7 | 0.00439 | 7,348 | 3.48e+08 | 26 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 8.81x / 27.27x / 15.22x
- tri-graded-pml: 8.81x / 27.27x / 12.57x
- tri-graded-pml-lts2: 8.38x / 48.63x / 24.63x
- tri-graded-repro: 8.81x / 27.27x / 14.73x
- tri-uniform: 3.20x / 10.60x / 12.26x

## B2-Ag-TE: q_ext_nrms <= 0.001

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.00538 | nan | nan | nan |
| cart-conformal-all-uniform | 7 | 0.00516 | nan | nan | nan |
| cart-staircase-graded | 7 | 0.0243 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0239 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00149 | nan | nan | nan |
| nfit-ogrid | 7 | 0.000698 | 37,907 | 3.85e+09 | 160 |
| nfit-ogrid-r0.5 | 7 | 0.000567 | 33,527 | 2.94e+09 | 119 |
| subgrid-2to1 | 7 | 0.00746 | nan | nan | nan |
| subgrid-2to1-lts4 | 4 | 0.00742 | nan | nan | nan |
| subgrid-2to1-ppw40 | 7 | 0.00703 | nan | nan | nan |
| tri-graded | 7 | 0.00104 | nan | nan | nan |
| tri-graded-pml | 7 | 0.00104 | nan | nan | nan |
| tri-graded-pml-lts2 | 6 | 0.00104 | nan | nan | nan |
| tri-graded-repro | 7 | 0.00104 | nan | nan | nan |
| tri-uniform | 7 | 0.000132 | 82,680 | 2.33e+09 | 44 |
| voronoi-poly | 7 | 0.00439 | nan | nan | nan |

## B2-Ag-TM: q_ext_nrms <= 0.01

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 6.71e-05 | <8,112 | <7.26e+07 | <7 |
| cart-conformal-all-uniform | 7 | 1.33e-05 | <18,252 | <1.64e+08 | <6 |
| cart-staircase-graded | 7 | 0.0192 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0191 | nan | nan | nan |
| hybrid-quadtri | 7 | 7.61e-05 | <4,653 | <3.98e+07 | <2 |
| nfit-ogrid | 7 | 3.09e-05 | <2,401 | <4.24e+07 | <3 |
| nfit-ogrid-r0.5 | 7 | 3.77e-05 | <2,785 | <4.92e+07 | <5 |
| subgrid-2to1 | 7 | 0.000724 | <5,881 | <5.3e+07 | <2 |
| subgrid-2to1-lts4 | 4 | 0.000975 | <8,661 | <5.41e+07 | <3 |
| subgrid-2to1-ppw40 | 7 | 0.000156 | <8,429 | <1.01e+08 | <3 |
| tri-graded | 7 | 0.000187 | 5,340 | 7.55e+07 | 9 |
| tri-graded-pml | 7 | 0.000187 | 5,340 | 7.55e+07 | 5 |
| tri-graded-pml-lts2 | 6 | 0.000207 | 5,586 | 4.25e+07 | 4 |
| tri-graded-repro | 7 | 0.000187 | 5,340 | 7.55e+07 | 7 |
| tri-uniform | 7 | 2.25e-05 | <13,053 | <1.65e+08 | <6 |
| voronoi-poly | 7 | 0.000344 | <1,782 | <2.32e+07 | <2 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 1.52x / 0.96x / 0.87x
- tri-graded-pml: 1.52x / 0.96x / 1.42x
- tri-graded-pml-lts2: 1.45x / 1.71x / 2.02x
- tri-graded-repro: 1.52x / 0.96x / 0.99x
- tri-uniform: 0.62x / 0.44x / 1.20x

## B2-Ag-TM: q_ext_nrms <= 0.001

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 6.71e-05 | 16,395 | 3.41e+08 | 16 |
| cart-conformal-all-uniform | 7 | 1.33e-05 | <18,252 | <1.64e+08 | <6 |
| cart-staircase-graded | 7 | 0.0192 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.0191 | nan | nan | nan |
| hybrid-quadtri | 7 | 7.61e-05 | 13,088 | 3.69e+08 | 10 |
| nfit-ogrid | 7 | 3.09e-05 | 6,301 | 2.07e+08 | 8 |
| nfit-ogrid-r0.5 | 7 | 3.77e-05 | 7,184 | 2.26e+08 | 8 |
| subgrid-2to1 | 7 | 0.000724 | 5,950 | 5.3e+07 | 2 |
| subgrid-2to1-lts4 | 4 | 0.000975 | 26,883 | 4.77e+08 | 24 |
| subgrid-2to1-ppw40 | 7 | 0.000156 | 10,912 | 1.01e+08 | 3 |
| tri-graded | 7 | 0.000187 | 10,872 | 2.53e+08 | 12 |
| tri-graded-pml | 7 | 0.000187 | 10,872 | 2.53e+08 | 12 |
| tri-graded-pml-lts2 | 6 | 0.000207 | 10,331 | 1.15e+08 | 8 |
| tri-graded-repro | 7 | 0.000187 | 10,872 | 2.53e+08 | 11 |
| tri-uniform | 7 | 2.25e-05 | 15,464 | 2.01e+08 | 7 |
| voronoi-poly | 7 | 0.000344 | 6,263 | 2.66e+08 | 11 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-uniform): unknowns / work / time
- tri-graded: 1.68x / 0.65x / 0.53x
- tri-graded-pml: 1.68x / 0.65x / 0.54x
- tri-graded-pml-lts2: 1.77x / 1.42x / 0.81x
- tri-graded-repro: 1.68x / 0.65x / 0.56x
- tri-uniform: 1.18x / 0.81x / 0.94x

## B3-g10: enh_peak <= 0.05

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 6 | 0.046 | 130,573 | 1.02e+10 | 413 |
| cart-conformal-all-uniform | 6 | 0.0457 | 475,910 | 3.67e+10 | 1,004 |
| cart-staircase-graded | 6 | 0.163 | nan | nan | nan |
| cart-staircase-uniform | 6 | 0.164 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00386 | 8,837 | 1.43e+08 | 15 |
| nfit-boxed | 7 | 0.00587 | 23,428 | 1.2e+09 | 38 |
| subgrid-2to1 | 7 | 0.0535 | nan | nan | nan |
| subgrid-2to1-gap4 | 7 | 0.0535 | nan | nan | nan |
| subgrid-2to1-gap4-lts8 | 2 | 0.0524 | nan | nan | nan |
| subgrid-2to1-lts4 | 4 | 0.0532 | nan | nan | nan |
| tri-graded | 7 | 0.00913 | <6,048 | <9.54e+07 | <8 |
| tri-graded-pml | 7 | 0.00913 | <6,048 | <9.54e+07 | <9 |
| tri-graded-pml-lts2 | 7 | 0.00917 | <6,048 | <4.77e+07 | <5 |
| tri-graded-repro | 7 | 0.00913 | <6,048 | <9.54e+07 | <10 |
| tri-uniform | 7 | 0.000156 | <18,024 | <2.84e+08 | <16 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 21.59x / 106.56x / 48.75x
- tri-graded-pml: 21.59x / 106.56x / 47.17x
- tri-graded-pml-lts2: 21.59x / 213.12x / 87.65x
- tri-graded-repro: 21.59x / 106.56x / 40.56x
- tri-uniform: 7.24x / 35.76x / 25.19x

## B3-g5: enh_peak <= 0.05

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 7 | 0.041 | 94,581 | 6.15e+09 | 251 |
| cart-conformal-all-uniform | 7 | 0.0409 | 343,517 | 2.23e+10 | 846 |
| cart-staircase-graded | 7 | 0.145 | nan | nan | nan |
| cart-staircase-uniform | 7 | 0.145 | nan | nan | nan |
| hybrid-quadtri | 7 | 0.00188 | 7,093 | 1.61e+08 | 12 |
| nfit-boxed | 7 | 0.0123 | 39,877 | 2.96e+09 | 90 |
| subgrid-2to1 | 7 | 0.0738 | nan | nan | nan |
| subgrid-2to1-gap4 | 7 | 0.0738 | nan | nan | nan |
| subgrid-2to1-gap4-lts8 | 4 | 0.0721 | nan | nan | nan |
| subgrid-2to1-lts4 | 4 | 0.0734 | nan | nan | nan |
| tri-graded | 7 | 0.00386 | 8,350 | 2.19e+08 | 16 |
| tri-graded-pml | 7 | 0.00386 | 8,350 | 2.19e+08 | 13 |
| tri-graded-pml-lts2 | 7 | 0.00399 | 8,318 | 1.09e+08 | 8 |
| tri-graded-repro | 7 | 0.00386 | 8,350 | 2.19e+08 | 17 |
| tri-uniform | 7 | 0.000482 | 35,290 | 9.16e+08 | 27 |

Ratios vs the cheapest Cartesian arm by work (cart-conformal-all-graded): unknowns / work / time
- tri-graded: 11.33x / 28.08x / 15.34x
- tri-graded-pml: 11.33x / 28.08x / 19.20x
- tri-graded-pml-lts2: 11.37x / 56.29x / 32.18x
- tri-graded-repro: 11.33x / 28.08x / 14.64x
- tri-uniform: 2.68x / 6.72x / 9.31x

## B3-g2: enh_peak <= 0.05

| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |
|---|---|---|---|---|---|
| cart-conformal-all-graded | 3 | 0.0361 | 112,095 | 8.17e+09 | 342 |
| cart-conformal-all-uniform | 3 | 0.036 | 427,132 | 3.11e+10 | 954 |
| cart-staircase-graded | 3 | 0.041 | 83,698 | 5.11e+09 | 169 |
| cart-staircase-uniform | 3 | 0.0405 | 306,668 | 1.87e+10 | 618 |
| hybrid-quadtri | 7 | 0.0112 | 13,238 | 7.55e+08 | 52 |
| nfit-boxed | 7 | 0.0366 | 95,578 | 1.27e+10 | 742 |
| subgrid-2to1 | 3 | 0.0807 | nan | nan | nan |
| subgrid-2to1-gap4 | 3 | 0.0798 | nan | nan | nan |
| subgrid-2to1-gap4-lts8 | 3 | 0.0783 | nan | nan | nan |
| subgrid-2to1-lts4 | 3 | 0.0803 | nan | nan | nan |
| tri-graded | 3 | 0.00375 | <34,780 | <1.69e+09 | <76 |
| tri-graded-pml | 3 | 0.00375 | <34,780 | <1.69e+09 | <85 |
| tri-graded-pml-lts2 | 3 | 0.00392 | <34,780 | <8.44e+08 | <74 |
| tri-graded-repro | 3 | 0.00375 | <34,780 | <1.69e+09 | <124 |
| tri-uniform | 2 | 0.00247 | <203,733 | <9.88e+09 | <289 |

Ratios vs the cheapest Cartesian arm by work (cart-staircase-graded): unknowns / work / time
- tri-graded: 2.41x / 3.03x / 2.22x
- tri-graded-pml: 2.41x / 3.03x / 1.98x
- tri-graded-pml-lts2: 2.41x / 6.05x / 2.29x
- tri-graded-repro: 2.41x / 3.03x / 1.36x
- tri-uniform: 0.41x / 0.52x / 0.58x

