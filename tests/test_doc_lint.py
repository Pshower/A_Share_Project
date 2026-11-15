"""The lint is the thing that keeps the doc set from rotting again, so its rules are pinned."""
from __future__ import annotations

import textwrap

from tools import doc_lint


def tree(root, files):
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")


def kinds(issues):
    return sorted({issue["kind"] for issue in issues})


def test_a_dead_link_is_reported(tmp_path):
    tree(tmp_path, {"a.md": "# A\n\n[gone](b.md)\n", "README.md": "# index\n\n| [a](a.md) |\n"})
    assert "links" in kinds(doc_lint.scan(tmp_path))


def test_a_live_link_is_fine(tmp_path):
    tree(tmp_path, {"a.md": "# A\n\n[b](b.md)\n", "b.md": "# B\n", "README.md": "# i\n\na.md b.md\n"})
    assert doc_lint.scan(tmp_path) == []


def test_a_missing_title_is_reported(tmp_path):
    tree(tmp_path, {"a.md": "just text\n", "README.md": "# i\n\na.md\n"})
    assert "titles" in kinds(doc_lint.scan(tmp_path))


def test_a_bom_does_not_hide_the_title(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("\ufeff# A\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# i\n\na.md\n", encoding="utf-8")
    assert doc_lint.scan(tmp_path) == []


def test_a_reference_to_a_missing_tool_is_reported(tmp_path):
    (tmp_path / "tools").mkdir()
    tree(tmp_path, {"a.md": "# A\n\nrun `tools/nope.py`\n", "README.md": "# i\n\na.md\n"})
    assert "tools" in kinds(doc_lint.scan(tmp_path))


def test_archive_and_historical_files_may_name_deleted_tools(tmp_path):
    (tmp_path / "tools").mkdir()
    tree(tmp_path, {
        "implementation_plan.md": "# plan\n\nrun `tools/nope.py`\n",
        "archive/old.md": "# old\n\nrun `tools/nope.py`\n",
        "README.md": "# i\n\nimplementation_plan.md\n",
    })
    issues = doc_lint.scan(tmp_path)
    assert [issue for issue in issues if issue["kind"] == "tools"] == []


def test_a_document_missing_from_the_index_is_reported(tmp_path):
    tree(tmp_path, {"a.md": "# A\n", "README.md": "# index\n"})
    assert "index" in kinds(doc_lint.scan(tmp_path))


def test_duplicate_titles_are_reported(tmp_path):
    tree(tmp_path, {"a.md": "# Same\n", "b.md": "# Same\n", "README.md": "# i\n\na.md b.md\n"})
    assert "duplicates" in kinds(doc_lint.scan(tmp_path))


def test_summarise_counts_by_rule():
    issues = [dict(kind="links"), dict(kind="links"), dict(kind="titles")]
    assert doc_lint.summarise(issues) == {"links": 2, "titles": 1}
