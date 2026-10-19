"""
Build the tutorial notebooks from their single source of truth.

Each tutorial is written once, as a Jupytext "percent" script
(``tutorials/NN_name/name.py``: ``# %%`` code cells and ``# %% [markdown]``
text cells). The script runs headless from the command line, and this tool
generates the Colab notebook ``name.ipynb`` next to it.

Usage
-----
    python tutorials/build.py                 # regenerate all notebooks (no outputs)
    python tutorials/build.py --check         # fail if a notebook is out of sync
    python tutorials/build.py 01 --execute \\
        --args "--quick --cpu" --timeout 7200 # run it and store the outputs

``--execute`` runs the notebook with nbclient in its own folder (figures are
written to ``figures/``), passes ``--args`` to the tutorial through the
environment variable WAKIS_TUTORIAL_ARGS, and keeps only static outputs
(PNG images, text): widget and HTML outputs are dropped so the committed
notebook stays small and renders on GitHub and Colab.

Requires ``jupytext`` (and ``nbclient``, ``ipykernel`` for ``--execute``).
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_NOTEBOOK_MB = 5.0
KEEP_MIME = ("image/png", "image/gif", "text/plain", "text/markdown", "text/latex")


def sources(selection):
    """Percent-format tutorial sources, optionally filtered by folder prefix."""
    out = []
    for folder in sorted(
        p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()
    ):
        if selection and not any(folder.name.startswith(s) for s in selection):
            continue
        for py in sorted(folder.glob("*.py")):
            head = py.read_text().splitlines()[:12]
            if any("format_name: percent" in line for line in head):
                out.append(py)
    return out


_PROGRESS = re.compile(r"\d+%\|.*\|\s*\d+/\d+")  # tqdm progress-bar lines


def _strip_outputs(nb):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # tqdm without widgets writes one stream output per update: keep the last
        outs = cell.get("outputs", [])
        bars = [i for i, o in enumerate(outs)
                if o.get("output_type") == "stream" and _PROGRESS.search(o.get("text", ""))]  # fmt: skip
        cell["outputs"] = [o for i, o in enumerate(outs) if i not in bars[:-1]]
        kept = []
        for o in cell.get("outputs", []):
            if o.get("output_type") in ("display_data", "execute_result"):
                data = {k: v for k, v in o.get("data", {}).items() if k in KEEP_MIME}
                if not data:
                    continue
                o["data"] = data
                o.get("metadata", {}).pop(
                    "application/vnd.jupyter.widget-view+json", None
                )
            elif o.get("output_type") == "stream":
                # tqdm progress bars: keep the last line of carriage returns;
                # drop VTK logger noise
                text = o.get("text", "")
                lines = [line.split("\r")[-1] for line in text.split("\n")]
                lines = [ln for ln in lines if "WARN|" not in ln]
                o["text"] = "\n".join(lines)
                if not o["text"].strip():
                    continue
            kept.append(o)
        cell["outputs"] = kept
    return nb


def build(py, execute=False, args="", timeout=7200, kernel="python3"):
    import jupytext

    nb = jupytext.read(py)
    target = py.with_suffix(".ipynb")
    if execute:
        from nbclient import NotebookClient

        os.environ["WAKIS_TUTORIAL_ARGS"] = args
        os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel,
            resources={"metadata": {"path": str(py.parent)}},
        )
        client.execute()
        nb = _strip_outputs(nb)
    else:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell["outputs"], cell["execution_count"] = [], None
    nb.metadata.setdefault("accelerator", "GPU")
    nb.metadata.setdefault("colab", {"provenance": [], "gpuType": "T4"})
    jupytext.write(nb, target, fmt="ipynb")
    size = target.stat().st_size / 1e6
    print(f"{target.relative_to(ROOT.parent)}: {len(nb.cells)} cells, {size:.2f} MB")
    if size > MAX_NOTEBOOK_MB:
        print(f"  warning: larger than {MAX_NOTEBOOK_MB} MB")
    return target


def check(py):
    """True when the notebook's cell sources match the script."""
    import jupytext

    target = py.with_suffix(".ipynb")
    if not target.exists():
        print(f"missing {target}")
        return False
    a = [(c.cell_type, c.source) for c in jupytext.read(py).cells]
    b = [(c.cell_type, c.source) for c in jupytext.read(target).cells]
    if a != b:
        print(f"out of sync: {target} (run tutorials/build.py)")
        return False
    return True


def restrip(path):
    """Apply the output hygiene of --execute to an executed notebook."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    nbformat.write(_strip_outputs(nb), path)
    print(f"{path}: {Path(path).stat().st_size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tutorials", nargs="*", help="folder prefixes, e.g. 01")
    ap.add_argument("--execute", action="store_true", help="run and keep outputs")
    ap.add_argument("--args", default="", help="options passed to the tutorial")
    ap.add_argument("--timeout", type=int, default=7200, help="per cell [s]")
    ap.add_argument("--check", action="store_true", help="verify notebooks in sync")
    ap.add_argument("--restrip", metavar="IPYNB", help="clean outputs of a notebook")
    a = ap.parse_args()
    if a.restrip:
        return restrip(a.restrip)
    files = sources(a.tutorials)
    if a.check:
        sys.exit(0 if all(check(py) for py in files) else 1)
    for py in files:
        build(py, a.execute, a.args, a.timeout)


if __name__ == "__main__":
    main()
