"""Audit DOIs in the repository against Crossref and DataCite metadata.

Scans tracked ``*.py`` and ``*.md`` files in the repository for Digital Object
Identifiers (DOIs), queries Crossref (with DataCite fallback for arXiv, Zenodo,
and JACoW prefixes), and compares the surrounding citation context against the
registered title, author list, and publication year.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches standard DOIs as well as legacy Wiley SICI DOIs containing
# parentheses, angle brackets (raw or percent-encoded), and ';2-X' suffixes.
DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/(?:[^\s\"<>\[\]{}|`',;]|<[^>\s]+>|;[0-9A-Za-z-]+|\((?=[^)]*\)))+"
)


def normalize_doi(raw_doi: str) -> str:
    """Normalize an extracted DOI string for API lookup.

    Parameters
    ----------
    raw_doi : str
        Raw DOI substring matched from source text or URLs.

    Returns
    -------
    str
        Lowercase, URL-decoded DOI with trailing punctuation and badge
        extensions (such as ``.svg``) stripped.
    """
    doi = urllib.parse.unquote(raw_doi).rstrip(".:)*,;]>")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    if doi.lower().endswith(".svg"):
        doi = doi[:-4]
    return doi.lower()


def extract_words(text: str) -> set[str]:
    """Extract lowercase alphabetic tokens of length >= 4.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    set of str
        Set of normalized word tokens.
    """
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def fetch_crossref(doi: str, timeout: float = 30.0) -> dict | None:
    """Query the Crossref Works API for a DOI.

    Parameters
    ----------
    doi : str
        Normalized DOI string.
    timeout : float, optional
        HTTP request timeout in seconds [s]. Default is 30.0.

    Returns
    -------
    dict or None
        Normalized metadata dictionary with keys ``title``, ``authors``,
        ``year``, and ``source="crossref"``, or ``None`` if not found.
    """
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "wakis-citation-audit/0.2 (https://github.com/ImpedanCEI/Wakis)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            msg = json.load(resp)["message"]
    except Exception:
        return None

    title = (msg.get("title") or [""])[0]
    authors = [a.get("family", "") for a in msg.get("author", []) if a.get("family")]
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "source": "crossref",
    }


def fetch_datacite(doi: str, timeout: float = 30.0) -> dict | None:
    """Query the DataCite API as a fallback for arXiv, Zenodo, and JACoW DOIs.

    Parameters
    ----------
    doi : str
        Normalized DOI string.
    timeout : float, optional
        HTTP request timeout in seconds [s]. Default is 30.0.

    Returns
    -------
    dict or None
        Normalized metadata dictionary with keys ``title``, ``authors``,
        ``year``, and ``source="datacite"``, or ``None`` if not found.
    """
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "wakis-citation-audit/0.2 (https://github.com/ImpedanCEI/Wakis)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            attrs = json.load(resp)["data"]["attributes"]
    except Exception:
        return None

    titles = attrs.get("titles") or []
    title = titles[0].get("title", "") if titles else ""
    authors = []
    for creator in attrs.get("creators", []):
        fam = creator.get("familyName")
        if not fam and creator.get("name"):
            fam = creator["name"].split(",")[0].strip()
        if fam:
            authors.append(fam)
    year = attrs.get("publicationYear")
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "source": "datacite",
    }


def classify_occurrence(
    context: str,
    title: str,
    authors: list[str],
) -> tuple[str, float]:
    """Classify a single DOI occurrence against registered metadata.

    Parameters
    ----------
    context : str
        Text preceding the DOI occurrence (up to 450 characters).
    title : str
        Registered work title.
    authors : list of str
        Registered author family names.

    Returns
    -------
    status : {"OK", "WEAK", "MISMATCH"}
        Match classification status.
    score : float
        Fraction of title words (length >= 4) appearing in the local context
        [dimensionless].
    """
    title_words = extract_words(title)
    context_words = extract_words(context[-350:])
    score = len(title_words & context_words) / max(1, len(title_words))
    author_hit = any(a and a.lower() in context.lower() for a in authors[:3])
    if score >= 0.5 or (author_hit and score >= 0.25):
        status = "OK"
    elif author_hit or score >= 0.25:
        status = "WEAK"
    else:
        status = "MISMATCH"
    return status, round(score, 2)


def collect_occurrences(
    repo_root: Path,
    files: list[str] | None = None,
) -> dict[str, list[tuple[str, int, str]]]:
    """Scan repository files for DOIs and their preceding text context.

    Parameters
    ----------
    repo_root : Path
        Root directory of the git worktree.
    files : list of str or None, optional
        Explicit relative file paths to scan. When ``None``, all git-tracked
        ``*.py`` and ``*.md`` files are scanned.

    Returns
    -------
    dict
        Mapping from normalized DOI to list of ``(rel_path, line_no, context)``
        tuples.
    """
    if files is None:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "*.py", "*.md"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = proc.stdout.split()

    occurrences: dict[str, list[tuple[str, int, str]]] = {}
    for rel_path in files:
        fpath = repo_root / rel_path
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in DOI_PATTERN.finditer(text):
            doi = normalize_doi(match.group(0))
            ctx = text[max(0, match.start() - 450) : match.start()]
            line_no = text.count("\n", 0, match.start()) + 1
            occurrences.setdefault(doi, []).append((rel_path, line_no, ctx))
    return occurrences


def audit_repository(
    repo_root: Path = REPO_ROOT,
    files: list[str] | None = None,
    delay_s: float = 0.12,
    use_datacite: bool = True,
) -> list[dict]:
    """Audit all DOIs in the repository against Crossref (and DataCite).

    Parameters
    ----------
    repo_root : Path, optional
        Root directory of the git worktree.
    files : list of str or None, optional
        Subset of relative file paths to audit.
    delay_s : float, optional
        Polite delay between API requests in seconds [s]. Default is 0.12.
    use_datacite : bool, optional
        Whether to query DataCite when Crossref returns 404. Default is True.

    Returns
    -------
    list of dict
        Audit result records, one per DOI occurrence (or per ``NOT_FOUND`` DOI).
    """
    occurrences = collect_occurrences(repo_root, files=files)
    rows: list[dict] = []

    for doi, uses in sorted(occurrences.items()):
        meta = fetch_crossref(doi)
        if meta is None and use_datacite:
            meta = fetch_datacite(doi)
        if delay_s > 0:
            time.sleep(delay_s)

        if meta is None:
            rows.append(
                {
                    "doi": doi,
                    "status": "NOT_FOUND",
                    "uses": [[f, line] for f, line, _ in uses],
                }
            )
            continue

        title = meta["title"]
        authors = meta["authors"]
        year = meta["year"]
        for rel_path, line_no, ctx in uses:
            status, score = classify_occurrence(ctx, title, authors)
            rows.append(
                {
                    "doi": doi,
                    "status": status,
                    "file": rel_path,
                    "line": line_no,
                    "crossref_title": title[:150],
                    "crossref_authors": authors[:4],
                    "year": year,
                    "title_overlap": score,
                    "context": ctx[-260:].replace("\n", " "),
                }
            )
    return rows


def format_report(rows: list[dict]) -> str:
    """Format a human-readable Markdown summary and table of flagged citations.

    Parameters
    ----------
    rows : list of dict
        Audit records returned by :func:`audit_repository`.

    Returns
    -------
    str
        Formatted Markdown report string.
    """
    counts = Counter(r["status"] for r in rows)
    lines = [
        "# Citation Audit Report",
        "",
        f"- **OK**: {counts.get('OK', 0)}",
        f"- **WEAK**: {counts.get('WEAK', 0)}",
        f"- **MISMATCH**: {counts.get('MISMATCH', 0)}",
        f"- **NOT_FOUND**: {counts.get('NOT_FOUND', 0)}",
        "",
    ]
    flagged = [r for r in rows if r["status"] != "OK"]
    if not flagged:
        lines.append("All audited DOIs matched Crossref/DataCite metadata.")
        return "\n".join(lines)

    lines.extend(
        [
            "| # | Status | DOI | Location(s) | Registered Title / Authors | Overlap |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for idx, rec in enumerate(flagged, 1):
        status = rec["status"]
        doi = rec["doi"]
        if status == "NOT_FOUND":
            locs = ", ".join(f"`{f}:{ln}`" for f, ln in rec.get("uses", []))
            meta_str = "*(not found in Crossref/DataCite)*"
            overlap_str = "—"
        else:
            locs = f"`{rec['file']}:{rec['line']}`"
            auths = ", ".join(rec.get("crossref_authors") or [])
            yr = rec.get("year") or "?"
            title = " ".join((rec.get("crossref_title") or "").split())
            meta_str = f"{auths} ({yr}): {title}"
            overlap_str = f"{rec.get('title_overlap', 0.0):.2f}"
        lines.append(
            f"| {idx} | `{status}` | `{doi}` | {locs} | {meta_str} | {overlap_str} |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line argument list.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Audit DOIs in tracked Python and Markdown files against Crossref "
            "and DataCite metadata."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="Path to the git repository root (default: worktree root).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional subset of repo-relative files to scan.",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "Print a Markdown report table of non-OK citations to stdout (when "
            "given without a path or as '-') or write the report to PATH "
            "('.json' writes JSON records; any other extension writes Markdown)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write raw audit JSON records to PATH.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.12,
        help="Delay in seconds between API lookups (default: 0.12).",
    )
    parser.add_argument(
        "--no-datacite",
        action="store_true",
        help="Disable DataCite fallback for non-Crossref DOIs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the citation audit CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit code (0 on completion).
    """
    args = parse_args(argv)
    rows = audit_repository(
        repo_root=args.repo.resolve(),
        files=args.files,
        delay_s=args.delay,
        use_datacite=not args.no_datacite,
    )
    counts = Counter(r["status"] for r in rows)
    print(dict(counts))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")

    if args.report is not None:
        if args.report == "-":
            print()
            print(format_report(rows))
        else:
            out_path = Path(args.report)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() == ".json":
                out_path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
            else:
                out_path.write_text(format_report(rows) + "\n", encoding="utf-8")
            print(format_report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
