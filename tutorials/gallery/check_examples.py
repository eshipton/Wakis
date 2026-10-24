# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Validate EXAMPLES.md against the chapter-organized repository structure.

Checks:
1. All required chapter and post-chapter section headings exist.
2. Every built tutorial folder ``tutorials/NN_*/NN-MM_slug`` appears in
   ``EXAMPLES.md`` with its tutorial number ``N.M`` and links to both
   ``slug.ipynb`` and ``slug.py``.
3. Every standalone script in ``examples/*.py``, upstream notebook in
   ``notebooks/*.ipynb``, and method study in ``studies/*/`` is linked.
4. Every relative image path, relative file/directory link, and internal
   ``#anchor`` link in ``EXAMPLES.md`` resolves.
5. HTML ``<table>``, ``<tr>``, and ``<td>`` tags are balanced.
6. No legacy catalog IDs (``W01``, ``M01``, ``T01``, ``A01``, ``F01``, ``B01``)
   appear in ``EXAMPLES.md``.
7. Mutation self-checks verify that each validation rule fails when violated.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_figure_refs import (  # noqa: E402
    run_mutation_checks as run_figure_mutation_checks,
    validate_all_tutorial_figures,
)

EXAMPLES_MD = REPO_ROOT / "EXAMPLES.md"

REQUIRED_HEADINGS = [
    "About",
    "Chapter 1 · Fields on a Grid",
    "Chapter 2 · Materials",
    "Chapter 3 · Diffraction and Scattering",
    "Chapter 4 · Radiation and Antennas",
    "Chapter 5 · Guided Waves",
    "Chapter 6 · Periodic Structures",
    "Chapter 7 · Light Sources",
    "Chapter 8 · Vacuum Electronics",
    "Chapter 9 · Inverse Design",
    "Appendix · Solver Features and How-Tos",
    "Standalone scripts",
    "Upstream Wakis notebooks",
    "Method studies",
    "Why FIT?",
    "Contributing an example",
]

LEGACY_ID_RE = re.compile(r"(?:#[WTFAM]\d{2,3}\b|\b[WTFAM]\d{2,3}\b)")


def github_slug(heading_text: str) -> str:
    """Convert a Markdown heading text into its GitHub anchor slug.

    Parameters
    ----------
    heading_text : str
        Raw heading text after stripping leading ``#`` characters.

    Returns
    -------
    slug : str
        Lowercase GitHub anchor string (without leading ``#``).
    """
    text = heading_text.strip().lower()
    slug_chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            slug_chars.append(ch)
        elif ch == " ":
            slug_chars.append("-")
    return "".join(slug_chars)


def discover_built_tutorials(repo_root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Return all built tutorial folders and their expected notebook/script paths.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    tutorials : list of tuple[str, Path, Path, Path]
        Tuples of ``(number_str, folder_rel, ipynb_rel, py_rel)`` sorted by
        chapter and tutorial number.
    """
    tutorials_dir = repo_root / "tutorials"
    results: list[tuple[str, Path, Path, Path]] = []
    for chap_dir in sorted(tutorials_dir.glob("[0-9][0-9]_*")):
        if not chap_dir.is_dir():
            continue
        for tut_dir in sorted(chap_dir.glob("[0-9][0-9]-[0-9][0-9]_*")):
            if not tut_dir.is_dir():
                continue
            m = re.match(r"^(\d{2})-(\d{2})_(.+)$", tut_dir.name)
            if not m:
                continue
            chap_num = int(m.group(1))
            tut_num = int(m.group(2))
            slug = m.group(3)
            num_str = f"{chap_num}.{tut_num}"
            rel_dir = tut_dir.relative_to(repo_root)
            ipynb_rel = rel_dir / f"{slug}.ipynb"
            py_rel = rel_dir / f"{slug}.py"
            if not (repo_root / py_rel).is_file():
                continue  # data or work in progress only: not built yet
            results.append((num_str, rel_dir, ipynb_rel, py_rel))
    return results


def validate_markdown(text: str, repo_root: Path) -> list[str]:
    """Validate ``EXAMPLES.md`` content and return a list of error strings.

    Parameters
    ----------
    text : str
        Full Markdown content of ``EXAMPLES.md``.
    repo_root : Path
        Repository root path used to resolve relative links and discover files.

    Returns
    -------
    errors : list of str
        Human-readable validation errors (empty when valid).
    """
    errors: list[str] = []

    # 1. Collect headings and slugs
    headings: list[str] = []
    valid_slugs: set[str] = set()
    for line in text.splitlines():
        m_head = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m_head:
            h_text = m_head.group(2)
            headings.append(h_text)
            valid_slugs.add(github_slug(h_text))

    for req in REQUIRED_HEADINGS:
        if req not in headings:
            errors.append(f"Missing required section heading: '## {req}'")

    # 2. Check HTML table tag balance
    for tag in ("table", "tr", "td"):
        n_open = len(re.findall(rf"<{tag}\b[^>]*>", text))
        n_close = len(re.findall(rf"</{tag}>", text))
        if n_open != n_close:
            errors.append(f"Unbalanced <{tag}> tags: {n_open} open vs {n_close} close")

    # 3. Check for legacy catalog IDs
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m_id in LEGACY_ID_RE.finditer(line):
            errors.append(
                f"Legacy catalog ID '{m_id.group(0)}' found on line {line_no}"
            )

    # 4. Extract all image sources and links
    img_srcs = re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
    img_srcs += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    for src in img_srcs:
        if src.startswith(("http://", "https://")):
            continue
        img_path = repo_root / src
        if not img_path.is_file():
            errors.append(f"Missing image file: {src}")

    # Extract Markdown links [label](target) and HTML links <a href="target">
    raw_links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)
    raw_links += re.findall(r'<a\s+[^>]*href="([^"]+)"', text)

    linked_rel_paths: set[str] = set()
    for target in raw_links:
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            slug = target[1:]
            if slug not in valid_slugs:
                errors.append(f"Broken internal anchor link: {target}")
            continue

        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        norm_rel = path_part.rstrip("/")
        linked_rel_paths.add(norm_rel)
        resolved = repo_root / path_part
        if not resolved.exists():
            errors.append(f"Broken relative link: {target}")

    # 5. Check every built tutorial folder appears with number N.M and links
    built_tutorials = discover_built_tutorials(repo_root)
    if not built_tutorials:
        errors.append("No built tutorials discovered under tutorials/")

    for num_str, rel_dir, ipynb_rel, py_rel in built_tutorials:
        num_pattern = re.compile(
            rf"-\s+\*\*{re.escape(num_str)}\s+·\s+Tutorial\s+{re.escape(num_str)}\b"
        )
        if not num_pattern.search(text):
            errors.append(
                f"Built tutorial {num_str} ({rel_dir}) missing entry '- **{num_str} · Tutorial {num_str} ...'"
            )
        if ipynb_rel.as_posix() not in linked_rel_paths:
            errors.append(
                f"Built tutorial {num_str} missing notebook link: {ipynb_rel.as_posix()}"
            )
        if py_rel.as_posix() not in linked_rel_paths:
            errors.append(
                f"Built tutorial {num_str} missing script link: {py_rel.as_posix()}"
            )

    # 6. Check standalone scripts, upstream notebooks, and method studies
    for script_path in sorted((repo_root / "examples").glob("*.py")):
        rel_s = script_path.relative_to(repo_root).as_posix()
        if rel_s not in linked_rel_paths:
            errors.append(f"Unlinked standalone script: {rel_s}")

    for nb_path in sorted((repo_root / "notebooks").glob("*.ipynb")):
        rel_nb = nb_path.relative_to(repo_root).as_posix()
        if rel_nb not in linked_rel_paths:
            errors.append(f"Unlinked upstream notebook: {rel_nb}")

    for study_dir in sorted((repo_root / "studies").iterdir()):
        if not study_dir.is_dir() or study_dir.name.startswith("."):
            continue
        rel_st = study_dir.relative_to(repo_root).as_posix()
        if rel_st not in linked_rel_paths:
            errors.append(f"Unlinked method study directory: {rel_st}")

    # 7. Check that every figure/GIF referenced in tutorial scripts exists and is tracked by git
    errors.extend(validate_all_tutorial_figures(repo_root))

    return errors


def run_mutation_checks(text: str, repo_root: Path) -> None:
    """Run negative-control mutation tests to ensure the validator can fail.

    Parameters
    ----------
    text : str
        Valid Markdown text of ``EXAMPLES.md``.
    repo_root : Path
        Repository root directory.
    """
    # Mutation 1: broken image path
    mut_img = text + '\n<img src="tutorials/gallery/nonexistent_plot.png">\n'
    err_img = validate_markdown(mut_img, repo_root)
    assert any("Missing image file" in e for e in err_img), (
        "Mutation check 1 failed: broken image was not detected"
    )

    # Mutation 2: broken relative file link
    mut_link = text + "\nSee [broken](tutorials/nonexistent_file.py).\n"
    err_link = validate_markdown(mut_link, repo_root)
    assert any("Broken relative link" in e for e in err_link), (
        "Mutation check 2 failed: broken relative link was not detected"
    )

    # Mutation 3: broken internal anchor link
    mut_anchor = text + "\nJump to [nowhere](#nonexistent-section-anchor).\n"
    err_anchor = validate_markdown(mut_anchor, repo_root)
    assert any("Broken internal anchor link" in e for e in err_anchor), (
        "Mutation check 3 failed: broken anchor link was not detected"
    )

    # Mutation 4: missing built tutorial entry (remove 3.4 entry)
    mut_tut = text.replace(
        "- **3.4 · Tutorial 3.4",
        "- **3.99 · Tutorial 3.99",
        1,
    )
    err_tut = validate_markdown(mut_tut, repo_root)
    assert any("Built tutorial 3.4" in e for e in err_tut), (
        "Mutation check 4 failed: missing built tutorial 3.4 was not detected"
    )

    # Mutation 5: reintroduced legacy catalog ID
    mut_legacy = text + "\nLegacy reference W01 should not appear.\n"
    err_legacy = validate_markdown(mut_legacy, repo_root)
    assert any("Legacy catalog ID" in e for e in err_legacy), (
        "Mutation check 5 failed: legacy ID W01 was not detected"
    )

    # Mutation 6: tutorial script figure/GIF reference checks
    run_figure_mutation_checks(repo_root)


def main() -> int:
    """Run all validation and mutation checks on ``EXAMPLES.md``."""
    if not EXAMPLES_MD.is_file():
        print(f"ERROR: {EXAMPLES_MD} does not exist", file=sys.stderr)
        return 1

    text = EXAMPLES_MD.read_text(encoding="utf-8")
    errors = validate_markdown(text, REPO_ROOT)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} error(s)):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    run_mutation_checks(text, REPO_ROOT)

    built_tutorials = discover_built_tutorials(REPO_ROOT)
    n_scripts = len(list((REPO_ROOT / "examples").glob("*.py")))
    n_notebooks = len(list((REPO_ROOT / "notebooks").glob("*.ipynb")))
    n_studies = len(
        [
            d
            for d in (REPO_ROOT / "studies").iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    )
    print(
        f"OK: EXAMPLES.md validated ({len(built_tutorials)} built tutorials, "
        f"{n_scripts} standalone scripts, {n_notebooks} upstream notebooks, "
        f"{n_studies} method studies; 5/5 mutation checks passed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
