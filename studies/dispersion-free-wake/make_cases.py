"""
Write the case list (one shell command per line) of the study.

Usage: python make_cases.py (writes cases_small.txt, cases_large.txt); run
with `run_queue.sh cases_small.txt 4` and `run_queue.sh cases_large.txt 1`.
Every line runs `run_case.py` on the GPU and writes results/<name>.json.
"""

import itertools

PY = "python run_case.py --gpu"
SCHEMES = [("tete", 0.99), ("yee", 0.99), ("yee", 0.5)]


def line(name, **kw):
    opts = " ".join(f"--{k} {v}" for k, v in kw.items() if v is not True)
    flags = " ".join(f"--{k}" for k, v in kw.items() if v is True)
    return f"{PY} {opts} {flags} --out results/{name}"


lines = []
# A. smooth pipes: spurious wake vs length (exact: 0)
for L, sig, (sch, cfl) in itertools.product(
    [0.05, 0.1, 0.2, 0.4, 0.8], [1.6e-3, 0.8e-3], SCHEMES
):
    lines.append(line(f"pipe_sq_L{L}_s{sig}_{sch}{cfl}", case="pipe", scheme=sch,
                      cfln=cfl, dz=4e-4, sigma=sig, a=0.016, length=L,
                      wakelength=0.1))  # fmt: skip
for L, (sch, cfl) in itertools.product([0.1, 0.4], SCHEMES):
    lines.append(line(f"pipe_rd_L{L}_s0.0008_{sch}{cfl}", case="pipe", scheme=sch,
                      cfln=cfl, dz=4e-4, sigma=8e-4, a=0.016, rp=0.006,
                      length=L, wakelength=0.1, round=True))  # fmt: skip
# B. closed box cavity (analytic reference)
for dz, sig, (sch, cfl) in itertools.product(
    [8e-4, 4e-4, 2.6667e-4, 2e-4, 1.3333e-4], [3.2e-3, 1.6e-3, 0.8e-3], SCHEMES
):
    lines.append(line(f"box_dz{dz}_s{sig}_{sch}{cfl}", case="box", scheme=sch,
                      cfln=cfl, dz=dz, sigma=sig, a=0.032, length=0.096,
                      wakelength=0.3))  # fmt: skip
# C. pillbox between beam pipes (fine-mesh references)
for dz, sig, (sch, cfl) in itertools.product(
    [8e-4, 4e-4, 2e-4, 1e-4], [3.2e-3, 1.6e-3, 0.8e-3, 0.4e-3], SCHEMES[:2]
):
    lines.append(line(f"pillbox_dz{dz}_s{sig}_{sch}{cfl}", case="pillbox",
                      scheme=sch, cfln=cfl, dz=dz, sigma=sig, wakelength=0.3))  # fmt: skip

# Grids above ~3 M cells run one at a time (GPU memory), the rest 4 at once


def cells(cmd):
    opt = dict(zip(cmd.split()[::1][:-1], cmd.split()[1:]))
    dz = float(opt["--dz"])
    dx = 2 * dz
    if "--case box" in cmd:
        return (0.032 / dx) ** 2 * 0.096 / dz
    if "--case pipe" in cmd:
        return (0.016 / dx) ** 2 * (float(opt["--length"]) / dz + 20)
    return (2 * 0.016 * 1.06 / dx) ** 2 * (0.056 / dz + 20)


large = [c for c in lines if cells(c) > 3e6]
small = [c for c in lines if cells(c) <= 3e6]
with open("cases_small.txt", "w") as f:
    f.write("\n".join(small) + "\n")
with open("cases_large.txt", "w") as f:
    f.write("\n".join(large) + "\n")
print(len(small), "small,", len(large), "large")
