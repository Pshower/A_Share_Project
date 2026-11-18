"""Lint the documentation: dead links, missing titles, references to code that no longer exists.

The doc set has been reorganised twice and each move left something behind: a link to a file that
moved, a page without an ``# H1``, a runbook telling the reader to run a tool that the branch cut
deleted.  A one-off cleanup does not stop that from recurring, so this file is the check that runs
after every change.

Checks (each is a separate function so a failure says which rule broke):

``links``      a relative ``.md`` link whose target is not on disk
``titles``     a document without a level-one heading
``tools``      a reference to ``tools/<name>.py`` that is not in the repository
``duplicates`` two live documents sharing a title
``index``      a live document that the index (``doc/README.md``) never mentions
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)]*)?\)")
TOOL = re.compile(r"\btools[/\\]([a-z0-9_]+)\.py")
# ``python -B -m tools.<name>`` names a module, not a file: the same check applies, but only to
# the explicit command form -- prose like "tools.shadow_run 的构造参数" is not a command.
COMMAND = re.compile(r"python[^\n]{0,40}?-m\s+tools\.([a-z0-9_]+)")
TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def documents(root, include_archive=True):
    root = Path(root)
    files = list(root.glob("*.md"))
    if include_archive:
        files += list((root / "archive").glob("*.md"))
    return sorted(files)


def check_links(path):
    issues = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for target in LINK.findall(line):
            if target.startswith(("http", "#")):
                continue
            if not (path.parent / target).resolve().exists():
                issues.append(dict(kind="links", doc=str(path), line=number, detail=target))
    return issues


def check_title(path):
    text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    if not TITLE.search(text):
        return [dict(kind="titles", doc=str(path), line=1, detail="no level-one heading")]
    return []


# Files whose tool references are deliberately historical: an audit log, an index that quotes
# archived reports, and a pre-registration written before the branch cut.  Listing them here is a
# decision (they keep the record readable), not an oversight.
HISTORICAL = {"implementation_plan.md", "README.md", "market_leverage_protocol_20261114.md"}
# Reviews quote the commands they are complaining about; treating those quotes as instructions
# would make the linter report the audit itself.
RECORD_PREFIX = ("repository_review_", "investment_advice_review_", "delivery_")
PLANNED = {"data_health"}          # now written; kept so archived mentions stay readable


def _is_record(path) -> bool:
    return path.name in HISTORICAL or path.name.startswith(RECORD_PREFIX)


def check_tools(path, root):
    issues = []
    if path.parent.name == "archive" or path.name in HISTORICAL:
        return issues
    alive = {item.stem for item in (Path(root) / "tools").glob("*.py")} | PLANNED
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for name in TOOL.findall(line):
            if name not in alive:
                issues.append(dict(kind="tools", doc=str(path), line=number, detail=name))
    return issues


def check_commands(path, root):
    """``python -B -m tools.<name>`` in a runbook must name a tool that still exists.

    ``check_tools`` only sees the ``tools/<name>.py`` path form, so a deleted module cited as a
    shell command used to sail through -- which is how a runbook kept telling people to run a tool
    that had been removed months earlier.
    """
    issues = []
    if path.parent.name == "archive" or _is_record(path):
        return issues
    alive = {item.stem for item in (Path(root) / "tools").glob("*.py")} | PLANNED
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue                      # a commented-out command is a record, not an instruction
        for name in COMMAND.findall(line):
            if name not in alive:
                issues.append(dict(kind="commands", doc=str(path), line=number,
                                   detail="tools.%s is not a tool" % name))
    return issues


def check_duplicates(paths):
    seen, issues = {}, []
    for path in paths:
        match = TITLE.search(path.read_text(encoding="utf-8").lstrip("\ufeff"))
        if not match:
            continue
        title = match.group(1).strip()
        if title in seen:
            issues.append(dict(kind="duplicates", doc=str(path), line=1,
                               detail="title also used by %s" % Path(seen[title]).name))
        else:
            seen[title] = path
    return issues


def check_index(root, paths):
    index = Path(root) / "README.md"
    if not index.exists():
        return []
    text = index.read_text(encoding="utf-8")
    issues = []
    for path in paths:
        if path.name == "README.md":
            continue
        if path.name not in text:
            issues.append(dict(kind="index", doc=str(path), line=1,
                               detail="not listed in doc/README.md"))
    return issues


def scan(root=PROJECT_ROOT / "doc", include_archive=True):
    root = Path(root)
    files = documents(root, include_archive=include_archive)
    issues = []
    for path in files:
        issues += check_links(path)
        issues += check_tools(path, Path(root).parent)
        issues += check_commands(path, Path(root).parent)
        if path.parent == root:                     # titles and index apply to the live set
            issues += check_title(path)
    issues += check_duplicates([path for path in files if path.parent == root])
    issues += check_index(root, [path for path in files if path.parent == root])
    return issues


def summarise(issues):
    counts = {}
    for issue in issues:
        counts[issue["kind"]] = counts.get(issue["kind"], 0) + 1
    return counts


def run(root=PROJECT_ROOT / "doc", include_archive=True, log=print):
    issues = scan(root, include_archive=include_archive)
    for issue in issues:
        log("%-11s %-42s line %-4d %s" % (
            issue["kind"], Path(issue["doc"]).name, issue["line"], issue["detail"]))
    summary = summarise(issues)
    log("%d issue(s): %s" % (len(issues), json.dumps(summary, ensure_ascii=False)))
    return dict(issues=issues, summary=summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lint the documentation set.")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "doc"))
    parser.add_argument("--live-only", action="store_true", help="Skip doc/archive.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.root, include_archive=not args.live_only)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
