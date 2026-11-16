# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Mutation verification script for `wakis.unstructured.adapt` (§6 of Brief W4-2).

Monkeypatches private helpers of `wakis.unstructured.adapt` at call time and
runs the corresponding target tests in `tests/test_115_unstructured_adapt.py`
in-process via `pytest.main`. Prints a summary table
`mutation | target tests | predicted | observed` and exits with code 1 if any
mutation is not caught (i.e. if any target test unexpectedly passes).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys

import numpy as np
import pytest

from wakis.unstructured import adapt, analytic, solver3d

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_FILE = os.path.join(ROOT, "tests", "test_115_unstructured_adapt.py")


def _mu1_no_closure(cells, split_code_arr):
    """MU1: Disable conformity closure (single bisection round)."""
    return np.empty(0, dtype=np.int64)


def _mu2_local_tiebreak(nodes, active_cells, l2_ref):
    """MU2: Key = rounded squared length only; tie-break by local edge index 0..5."""
    pairs = active_cells[:, adapt._LOC_E]
    d = nodes[pairs[..., 1]] - nodes[pairs[..., 0]]
    l2_round = np.round(np.einsum("nci,nci->nc", d, d) / l2_ref, 10)
    return np.argmax(l2_round, axis=1)


_ORIG_FIND_NONCONFORMING = adapt._find_nonconforming_cells
_MU3_SPLIT_CODES = np.empty(0, dtype=np.int64)


def _mu3_track_nonconforming(cells, split_code_arr):
    """Track already-split edges during closure rounds for MU3."""
    global _MU3_SPLIT_CODES
    active = _ORIG_FIND_NONCONFORMING(cells, split_code_arr)
    _MU3_SPLIT_CODES = (
        split_code_arr.copy() if active.size > 0 else np.empty(0, dtype=np.int64)
    )
    return active


def _mu3_shortest_edge(nodes, active_cells, l2_ref):
    """MU3: Bisect the shortest local edge instead of the longest."""
    pairs = active_cells[:, adapt._LOC_E]
    lo = np.minimum(pairs[..., 0], pairs[..., 1])
    hi = np.maximum(pairs[..., 0], pairs[..., 1])
    codes = lo * adapt._EDGE_CODE_STRIDE + hi
    d = nodes[pairs[..., 1]] - nodes[pairs[..., 0]]
    l2 = np.einsum("nci,nci->nc", d, d)
    if _MU3_SPLIT_CODES.size > 0:
        l2 = np.where(np.isin(codes, _MU3_SPLIT_CODES), l2 - 1e12, l2)
    return np.argmin(l2, axis=1)


_ORIG_SPLIT_CELLS = adapt._split_cells


def _mu4_drop_second_child(
    active_cells, active_tags, active_parent, active_orig_face, loc_edge, mid_nodes
):
    """MU4: Return only the first child tetrahedron of each bisection."""
    cells, tags, parent, orig_face = _ORIG_SPLIT_CELLS(
        active_cells, active_tags, active_parent, active_orig_face, loc_edge, mid_nodes
    )
    n = len(active_cells)
    return cells[:n], tags[:n], parent[:n], orig_face[:n]


def _mu5_identity_projector(project, pts, patches):
    """MU5: Identity boundary projector (leave new boundary nodes on chord midpoints)."""
    return np.asarray(pts, dtype=float).copy()


_ORIG_LOCAL_EDGE_SIGNS = adapt._local_edge_signs


def _mu7_flip_edge0_sign(cells):
    """MU7: Flip orientation sign s_p of local edge 0 in whitney_fields."""
    sgn = _ORIG_LOCAL_EDGE_SIGNS(cells).copy()
    sgn[:, 0] *= -1.0
    return sgn


def _mu8_lam_h_squared_normal_jump(h_F, mu_F, lam_h, eps_F):
    """MU8: Use lam_h^2 instead of lam_h in the normal-jump face weight."""
    return h_F * mu_F, h_F * ((lam_h**2) / eps_F)


def _mu9_drop_h_f(h_F, mu_F, lam_h, eps_F):
    """MU9: Drop face diameter factor h_F from face jump weights."""
    ones = np.ones_like(h_F)
    return ones * mu_F, ones * (lam_h / eps_F)


def _mu10_drop_mu_t_element(h_T, mu_T, lam_h, eps_T, u_l2_T):
    """MU10: Drop mu_T factor in the element residual term."""
    return (h_T**2) * ((lam_h * eps_T) ** 2) * u_l2_T


def _mu11_no_zz_averaging(mesh, B, live):
    """MU11: Set recovered curl R = B_T without nodal averaging (eta_ZZ = 0)."""
    return np.broadcast_to(B[:, None, :], (mesh.n_cells, 4, 3)).copy()


def _mu12_ascending_dorfler(eta2):
    """MU12: Sort error indicators ascending instead of descending in dorfler_mark."""
    return np.argsort(eta2, kind="stable")


def _mu13_unmapped_mps_coeffs(R, v):
    """MU13: Return the right singular vector v without mapping through R^-1."""
    return np.asarray(v, dtype=float).copy()


_ORIG_ONE_STEP = solver3d.UnstructuredFIT3D.one_step


def _mu14_unconditional_one_step(self):
    """MU14: Bypass the positive-finite dt check in UnstructuredFIT3D.one_step."""
    self.n += 1


MUTATIONS = [
    (
        "MU1",
        (adapt, "_find_nonconforming_cells"),
        _mu1_no_closure,
        ["test_r1_conformity"],
        "predicted R1(iii)/(iv); killed earlier at R1(ii) face conformity",
    ),
    (
        "MU2",
        (adapt, "_pick_bisect_edge"),
        _mu2_local_tiebreak,
        ["test_r1_conformity"],
        "predicted R1(iii)/(iv); killed earlier at R1(ii) face conformity",
    ),
    (
        "MU3",
        (adapt, "_pick_bisect_edge"),
        _mu3_shortest_edge,
        ["test_r3_kuhn_similarity", "test_r4_kuhn_classes"],
        "R3 node set / volume and R4 class count fail",
    ),
    (
        "MU4",
        (adapt, "_split_cells"),
        _mu4_drop_second_child,
        ["test_r2_volume_conservation"],
        "R2 volume conservation fails by half parent volume",
    ),
    (
        "MU5",
        (adapt, "_apply_projector"),
        _mu5_identity_projector,
        ["test_r7_sphere_projection"],
        "R7 radial error ~ L^2/(8R) >> 1e-12 R fails",
    ),
    (
        "MU7",
        (adapt, "_local_edge_signs"),
        _mu7_flip_edge0_sign,
        ["test_s1_stokes", "test_s2_galerkin_identity"],
        "S1 Stokes and S2 Galerkin identity fail by O(1)",
    ),
    (
        "MU8",
        (adapt, "_res_face_jump_weight"),
        _mu8_lam_h_squared_normal_jump,
        ["test_s3_scale_invariance"],
        "S3 scale invariance fails by s^-2",
    ),
    (
        "MU9",
        (adapt, "_res_face_jump_weight"),
        _mu9_drop_h_f,
        ["test_s3_scale_invariance"],
        "S3 scale invariance fails by s^-1",
    ),
    (
        "MU10",
        (adapt, "_res_element_term"),
        _mu10_drop_mu_t_element,
        ["test_s4_material_homogeneity"],
        "S4 material homogeneity fails by factor mu_r=1.7",
    ),
    (
        "MU11",
        (adapt, "_zz_recover_curl"),
        _mu11_no_zz_averaging,
        ["test_e2_e3_e4_box_effectivity_fast"],
        "eta_ZZ = 0 -> E3 fast band [0.5, 2] fails",
    ),
    (
        "MU12",
        (adapt, "_dorfler_order"),
        _mu12_ascending_dorfler,
        ["test_s5_dorfler_mark"],
        "S5 Dorfler smallest-set / largest-values check fails",
    ),
    (
        "MU13",
        (analytic, "_mps_solve_coeffs"),
        _mu13_unmapped_mps_coeffs,
        ["test_e5_mps_reference"],
        "unmapped right singular vector -> E5 field/Green check fails",
    ),
    (
        "MU14",
        (solver3d.UnstructuredFIT3D, "one_step"),
        _mu14_unconditional_one_step,
        ["test_solver3d_stability_false_dt_none_raises"],
        "ValueError not raised when stability=False and dt=None",
    ),
]


def _extract_failure_summary(output: str) -> str:
    """Extract a concise one-line failure reason from pytest output."""
    for line in output.splitlines():
        s = line.strip()
        if s.startswith(("E   AssertionError:", "E   ValueError:")):
            msg = re.sub(r"^E\s+", "", s)
            return msg[:90]
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("E   "):
            return re.sub(r"^E\s+", "", s)[:90]
    return "non-zero exit"


def main() -> int:
    rows = []
    all_caught = True

    for mu_id, (target_obj, attr_name), patch_fn, tests, predicted in MUTATIONS:
        global _MU3_SPLIT_CODES
        _MU3_SPLIT_CODES = np.empty(0, dtype=np.int64)
        orig_fn = getattr(target_obj, attr_name)
        setattr(target_obj, attr_name, patch_fn)
        if mu_id == "MU3":
            adapt._find_nonconforming_cells = _mu3_track_nonconforming
        observed_parts = []
        caught_this_mu = True
        try:
            for tname in tests:
                _MU3_SPLIT_CODES = np.empty(0, dtype=np.int64)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    code = int(
                        pytest.main(
                            [
                                f"{TEST_FILE}::{tname}",
                                "-q",
                                "--tb=short",
                                "-p",
                                "no:cacheprovider",
                            ]
                        )
                    )
                out = buf.getvalue()
                if code == 0:
                    caught_this_mu = False
                    observed_parts.append(f"{tname}: SURVIVED (exit 0)")
                else:
                    reason = _extract_failure_summary(out)
                    observed_parts.append(f"{tname}: KILLED ({reason})")
        finally:
            setattr(target_obj, attr_name, orig_fn)
            if mu_id == "MU3":
                adapt._find_nonconforming_cells = _ORIG_FIND_NONCONFORMING
            _MU3_SPLIT_CODES = np.empty(0, dtype=np.int64)

        if not caught_this_mu:
            all_caught = False
        rows.append(
            (
                mu_id,
                ", ".join(tests),
                predicted,
                "; ".join(observed_parts),
            )
        )

    header = ("mutation", "target tests", "predicted", "observed")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(4)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(*r))

    if not all_caught:
        print("\nERROR: One or more mutations survived!", file=sys.stderr)
        return 1
    print(f"\nAll {len(rows)} mutations caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
