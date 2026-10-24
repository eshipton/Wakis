#!/usr/bin/env python3
# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Download completed Cloud Batch PIC tutorial outputs from GCS into ``<tutorial>/data/``.

Usage
-----
    python tutorials/05_guided_waves/cloud/collect.py RUN_ID tutorials/05_guided_waves/05-01_y_junction
    python tutorials/05_guided_waves/cloud/collect.py RUN_ID tutorials/05_guided_waves/05-01_y_junction --plot
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
BUCKET = os.environ.get("BUCKET", "cloud-code-511302-autoresearch")


def resolve_tutorial_dir(arg):
    p = Path(arg).resolve()
    if p.is_file():
        return p.parent
    return p


def find_tutorial_script(tdir):
    scripts = sorted(tdir.glob("*.py"))
    return scripts[0] if scripts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_id", help="Cloud Batch run identifier (e.g. pic-20261021-120000)"
    )
    ap.add_argument(
        "tutorials",
        nargs="+",
        help="Tutorial directory or script path(s) (e.g. tutorials/05_guided_waves/05-01_y_junction)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Run `<tutorial>.py --plot` after downloading `data/`",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gcloud storage cp commands without executing them",
    )
    args = ap.parse_args()

    for item in args.tutorials:
        tdir = resolve_tutorial_dir(item)
        tutorial_name = tdir.name
        data_dir = tdir / "data"
        gcs_src = f"gs://{BUCKET}/pic-tutorials/{args.run_id}/{tutorial_name}/*"
        cmd = ["gcloud", "storage", "cp", "-r", gcs_src, str(data_dir) + "/"]
        if args.dry_run:
            print(" ".join(cmd))
            continue

        data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        print(f"Downloaded {gcs_src} -> {data_dir}")

        if args.plot:
            script = find_tutorial_script(tdir)
            if script is not None:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO)
                env["MPLBACKEND"] = "Agg"
                subprocess.run(
                    [sys.executable, str(script), "--plot"],
                    cwd=REPO,
                    env=env,
                    check=True,
                )


if __name__ == "__main__":
    main()
