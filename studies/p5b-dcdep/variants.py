"""Run dc_decomp.py with one numerical ingredient switched off, to locate the interior gamma excess.

Usage: python studies/p5b-dcdep/variants.py --grid coarse --variant {base,nokick,noclosure,halfcfl}
  nokick    : wakis.particles.solver._ENABLE_PEC_WALL_KICK = False
  noclosure : wakis.particles.solver._ENABLE_PEC_WALL_CLOSURE = False
  halfcfl   : examples._space_charge_waves_common.CFL_NUMBER halved (dt / 2, same physics)
The DCDEC line's gamma_diff_mean / per_line.gamma_diff is the quantity to compare across variants.
"""

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=("coarse", "fine"))
    ap.add_argument("--variant", required=True, choices=("base", "nokick", "noclosure", "halfcfl"))
    a = ap.parse_args()
    import examples._space_charge_waves_common as scw
    import wakis.particles.solver as ps

    if a.variant == "nokick":
        ps._ENABLE_PEC_WALL_KICK = False
    elif a.variant == "noclosure":
        ps._ENABLE_PEC_WALL_CLOSURE = False
    elif a.variant == "halfcfl":
        scw.CFL_NUMBER = 0.5 * scw.CFL_NUMBER
    print(f"VARIANT {a.variant} grid={a.grid} kick={ps._ENABLE_PEC_WALL_KICK} "
          f"closure={ps._ENABLE_PEC_WALL_CLOSURE} cfl={scw.CFL_NUMBER}", flush=True)
    spec = importlib.util.spec_from_file_location("dc_decomp", pathlib.Path(__file__).with_name("dc_decomp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(["--grid", a.grid]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
