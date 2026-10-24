# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Validate that every figure and GIF referenced in tutorial scripts exists and is tracked by git.

Checks across all tutorial folders ``tutorials/NN_*/NN-MM_*/``:
1. Markdown image links ``![...](figures/...)``.
2. HTML image tags ``<img src="figures/...">``.
3. ``Image(filename=...)`` / ``display(Image(...))`` calls with literal image paths.
4. ``FIG_DIR / "..."`` (and ``fig_dir / "..."``, ``target_dir / "..."``, ``wiki_dir / "..."``)
   path expressions.
5. Backtick-quoted figure/GIF filenames in Markdown comment cells (``# ... `...gif` ``).
6. Negative-control mutation checks verifying that missing and untracked references fail.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXT_RE = r"(?:png|gif|svg|jpg|jpeg)"
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_RE = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\']')
IMAGE_CALL_RE = re.compile(r"\bImage\s*\(([^;\n]+)\)")
STRING_IMG_RE = re.compile(rf'["\']([^"\']+\.{IMAGE_EXT_RE})["\']')
FIG_DIR_DIV_RE = re.compile(
    rf'\b(FIG_DIR|fig_dir|target_dir|wiki_dir)\s*/\s*([fF]?["\'])([^"\']+\.{IMAGE_EXT_RE})["\']'
)
MD_BACKTICK_IMG_RE = re.compile(
    rf"`((?:figures/)?[A-Za-z0-9_./-]+\.{IMAGE_EXT_RE})`"
)


def get_git_tracked_files(repo_root: Path = REPO_ROOT) -> set[str]:
    """Return the set of repository-relative POSIX paths tracked by git.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root directory. Default is ``REPO_ROOT``.

    Returns
    -------
    tracked : set of str
        Tracked file paths relative to ``repo_root`` using forward slashes.
    """
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True
    )
    return {line.strip() for line in out.splitlines() if line.strip()}


def discover_tutorial_scripts(
    repo_root: Path = REPO_ROOT,
) -> list[tuple[Path, Path]]:
    """Discover all Python scripts inside built tutorial directories.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root directory. Default is ``REPO_ROOT``.

    Returns
    -------
    scripts : list of tuple[Path, Path]
        Pairs of ``(tutorial_dir, script_path)`` sorted by chapter and tutorial.
    """
    tutorials_dir = repo_root / "tutorials"
    results: list[tuple[Path, Path]] = []
    for chap_dir in sorted(tutorials_dir.glob("[0-9][0-9]_*")):
        if not chap_dir.is_dir():
            continue
        for tut_dir in sorted(chap_dir.glob("[0-9][0-9]-[0-9][0-9]_*")):
            if not tut_dir.is_dir():
                continue
            for py_path in sorted(tut_dir.glob("*.py")):
                results.append((tut_dir, py_path))
    return results


def extract_script_figure_refs(
    text: str, tut_dir: Path
) -> list[tuple[str, str, Path]]:
    """Extract all figure and GIF references from a tutorial Python script.

    Parameters
    ----------
    text : str
        Source text of the tutorial ``.py`` file.
    tut_dir : Path
        Directory of the tutorial containing ``figures/``.

    Returns
    -------
    refs : list of tuple[str, str, Path]
        Tuples of ``(ref_kind, raw_reference, resolved_path)``.
    """
    refs: list[tuple[str, str, Path]] = []

    # 1. Markdown image links: ![alt](target)
    for m in MD_IMAGE_RE.finditer(text):
        target = m.group(1).strip().split()[0]
        if target.startswith(("http://", "https://")):
            continue
        refs.append(("md_image", target, tut_dir / target))

    # 2. HTML <img src="..."> tags
    for m in HTML_IMG_RE.finditer(text):
        target = m.group(1).strip()
        if target.startswith(("http://", "https://")):
            continue
        refs.append(("html_img", target, tut_dir / target))

    # 3. Image(...) display calls with string literal paths
    for m in IMAGE_CALL_RE.finditer(text):
        call_body = m.group(1)
        if any(
            v in call_body
            for v in ("FIG_DIR", "fig_dir", "target_dir", "wiki_dir")
        ):
            continue
        for sm in STRING_IMG_RE.finditer(call_body):
            raw = sm.group(1)
            if "{" in raw or raw.startswith(("http://", "https://")):
                continue
            cand = raw if raw.startswith("figures/") else f"figures/{raw}"
            refs.append(("image_call", raw, tut_dir / cand))

    # 4. (FIG_DIR | fig_dir | target_dir | wiki_dir) / "..." path expressions
    for m in FIG_DIR_DIV_RE.finditer(text):
        var_name, quote_prefix, name = m.group(1), m.group(2), m.group(3)
        if quote_prefix.lower().startswith("f") and "{" in name:
            continue
        if "{" in name:
            continue
        sub = "figures/wikimedia" if var_name == "wiki_dir" else "figures"
        cand = name if name.startswith("figures/") else f"{sub}/{name}"
        refs.append(("fig_dir_path", f"{var_name}/{name}", tut_dir / cand))

    # 5. Backtick-quoted figure/GIF filenames in Markdown comment lines
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        for m in MD_BACKTICK_IMG_RE.finditer(line):
            raw = m.group(1)
            if raw.startswith("File:") or "{" in raw:
                continue
            cand = raw if raw.startswith("figures/") else f"figures/{raw}"
            refs.append(("md_backtick", raw, tut_dir / cand))

    return refs


def validate_script_figure_refs(
    text: str,
    tut_dir: Path,
    repo_root: Path,
    tracked_files: set[str],
    source_label: str | None = None,
) -> list[str]:
    """Validate figure/GIF references in a single tutorial script text.

    Parameters
    ----------
    text : str
        Source text of the tutorial script.
    tut_dir : Path
        Tutorial folder path.
    repo_root : Path
        Repository root directory.
    tracked_files : set of str
        Set of repository-relative POSIX paths tracked by git.
    source_label : str or None, optional
        Label for error reporting. Defaults to ``tut_dir`` relative to ``repo_root``.

    Returns
    -------
    errors : list of str
        Validation error messages (empty when all referenced files exist and are tracked).
    """
    label = source_label or tut_dir.relative_to(repo_root).as_posix()
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for kind, raw, full_path in extract_script_figure_refs(text, tut_dir):
        try:
            rel_posix = full_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel_posix = full_path.as_posix()
        key = (kind, rel_posix)
        if key in seen:
            continue
        seen.add(key)

        if not full_path.is_file():
            errors.append(
                f"{label} [{kind}]: missing figure file '{rel_posix}' (referenced as '{raw}')"
            )
        elif rel_posix not in tracked_files:
            errors.append(
                f"{label} [{kind}]: untracked figure file '{rel_posix}' (referenced as '{raw}')"
            )
    return errors


def validate_all_tutorial_figures(
    repo_root: Path = REPO_ROOT,
    tracked_files: set[str] | None = None,
) -> list[str]:
    """Validate all figure/GIF references across every tutorial script in the repository.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root directory. Default is ``REPO_ROOT``.
    tracked_files : set of str or None, optional
        Pre-fetched set of git-tracked POSIX paths. Fetched via ``git ls-files`` when ``None``.

    Returns
    -------
    errors : list of str
        Human-readable validation errors (empty when valid).
    """
    if tracked_files is None:
        tracked_files = get_git_tracked_files(repo_root)

    scripts = discover_tutorial_scripts(repo_root)
    if not scripts:
        return ["No tutorial scripts discovered under tutorials/"]

    errors: list[str] = []
    for tut_dir, py_path in scripts:
        rel_label = py_path.relative_to(repo_root).as_posix()
        text = py_path.read_text(encoding="utf-8")
        errors.extend(
            validate_script_figure_refs(
                text, tut_dir, repo_root, tracked_files, source_label=rel_label
            )
        )
    return errors


def run_mutation_checks(
    repo_root: Path = REPO_ROOT,
    tracked_files: set[str] | None = None,
) -> None:
    """Run negative-control mutation checks to ensure the validator can fail.

    Parameters
    ----------
    repo_root : Path, optional
        Repository root directory. Default is ``REPO_ROOT``.
    tracked_files : set of str or None, optional
        Set of git-tracked POSIX paths. Fetched when ``None``.
    """
    if tracked_files is None:
        tracked_files = get_git_tracked_files(repo_root)

    scripts = discover_tutorial_scripts(repo_root)
    assert scripts, "Mutation setup failed: no tutorial scripts found"
    sample_tut_dir, sample_py = scripts[0]
    base_text = sample_py.read_text(encoding="utf-8")

    # Mutation 1: broken Markdown image link ![...](figures/nonexistent.gif)
    mut_md = base_text + "\n# ![Broken anim](figures/nonexistent_mutation_1.gif)\n"
    err_md = validate_script_figure_refs(
        mut_md, sample_tut_dir, repo_root, tracked_files
    )
    assert any(
        "missing figure file" in e and "nonexistent_mutation_1.gif" in e
        for e in err_md
    ), "Mutation check 1 failed: broken Markdown image link was not detected"

    # Mutation 2: broken Image(filename=...) display call
    mut_img = (
        base_text
        + '\ndisplay(Image(filename="figures/nonexistent_mutation_2.gif"))\n'
    )
    err_img = validate_script_figure_refs(
        mut_img, sample_tut_dir, repo_root, tracked_files
    )
    assert any(
        "missing figure file" in e and "nonexistent_mutation_2.gif" in e
        for e in err_img
    ), "Mutation check 2 failed: broken Image(filename=...) call was not detected"

    # Mutation 3: broken FIG_DIR / "..." expression
    mut_fdir = base_text + '\n_ = FIG_DIR / "nonexistent_mutation_3.gif"\n'
    err_fdir = validate_script_figure_refs(
        mut_fdir, sample_tut_dir, repo_root, tracked_files
    )
    assert any(
        "missing figure file" in e and "nonexistent_mutation_3.gif" in e
        for e in err_fdir
    ), "Mutation check 3 failed: broken FIG_DIR / '...' reference was not detected"

    # Mutation 4: broken backtick figure reference in Markdown comment
    mut_tick = base_text + "\n# See `figures/nonexistent_mutation_4.gif`.\n"
    err_tick = validate_script_figure_refs(
        mut_tick, sample_tut_dir, repo_root, tracked_files
    )
    assert any(
        "missing figure file" in e and "nonexistent_mutation_4.gif" in e
        for e in err_tick
    ), "Mutation check 4 failed: broken Markdown backtick reference was not detected"

    # Mutation 5: existing figure file on disk that is not tracked by git
    refs = extract_script_figure_refs(base_text, sample_tut_dir)
    assert refs, f"Mutation setup failed: no figure refs in {sample_py}"
    existing = [r for r in refs if r[2].is_file()]
    assert existing, f"Mutation setup failed: no existing figure refs in {sample_py}"
    _, _, existing_fig = existing[0]
    rel_existing = (
        existing_fig.resolve().relative_to(repo_root.resolve()).as_posix()
    )
    reduced_tracked = tracked_files - {rel_existing}
    err_untracked = validate_script_figure_refs(
        base_text, sample_tut_dir, repo_root, reduced_tracked
    )
    assert any(
        "untracked figure file" in e and rel_existing in e
        for e in err_untracked
    ), "Mutation check 5 failed: untracked figure file was not detected"


def main() -> int:
    """Run all tutorial figure-reference validations and mutation self-checks."""
    tracked_files = get_git_tracked_files(REPO_ROOT)
    errors = validate_all_tutorial_figures(REPO_ROOT, tracked_files)
    if errors:
        print(
            f"FIGURE REFERENCE VALIDATION FAILED ({len(errors)} error(s)):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    run_mutation_checks(REPO_ROOT, tracked_files)

    scripts = discover_tutorial_scripts(REPO_ROOT)
    n_refs = sum(
        len(extract_script_figure_refs(py.read_text(encoding="utf-8"), tut))
        for tut, py in scripts
    )
    print(
        f"OK: {n_refs} figure/GIF references across {len(scripts)} tutorial scripts "
        "exist and are tracked by git (5/5 mutation checks passed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
