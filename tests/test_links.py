"""Relative links in the README and the full report resolve to something in the tree (seventh post-run change).

A reader is pointed to docs/sql.md, sql/ and docs/deviations.md by clickable links; this fails if a link would lead
nowhere: a missing file or directory, a path outside the repository, or a heading its target does not have."""
import re
from pathlib import Path

import pytest

from judge import config as C

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
DOCS = ("README.md", "results/report.md")


def _links(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "", text, flags=re.S)     # fenced code
    text = re.sub(r"`[^`\n]*`", "", text)                  # code spans (a link's text may be one; its target stays)
    return LINK.findall(text)


def _slugs(md: str) -> set[str]:
    """The anchors GitHub gives the headings of a Markdown file."""
    out = set()
    for h in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", re.sub(r"```.*?```", "", md, flags=re.S)):
        out.add(re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-"))
    return out


def _relative(doc: str) -> list[str]:
    return [t for t in _links((C.ROOT / doc).read_text()) if not re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I)]


@pytest.mark.parametrize("doc", DOCS)
def test_every_relative_link_resolves_to_a_file_or_directory_in_the_tree(doc):
    root = C.ROOT.resolve()
    targets = _relative(doc)
    assert targets, f"{doc} has no relative links"
    bad = []
    for t in targets:
        path, _, anchor = t.partition("#")
        dest = ((C.ROOT / doc).parent / path).resolve() if path else (C.ROOT / doc).resolve()
        if root != dest and root not in dest.parents:
            bad.append(f"{t}: outside the repository")
        elif not dest.exists():
            bad.append(f"{t}: no such file or directory")
        elif path.endswith("/") and not dest.is_dir():
            bad.append(f"{t}: not a directory")
        elif anchor and dest.suffix == ".md" and anchor not in _slugs(dest.read_text()):
            bad.append(f"{t}: no heading #{anchor} in {Path(dest).relative_to(root)}")
    assert not bad, f"{doc}:\n" + "\n".join(bad)


def test_the_readme_links_the_sql_its_walk_through_and_the_deviations_list():
    targets = set(_relative("README.md"))
    for t in ("sql/", "docs/sql.md", "docs/deviations.md"):
        assert t in targets, t
    report = set(_relative("results/report.md"))
    for t in ("../sql/", "../docs/sql.md", "../docs/deviations.md"):     # the same blocks, one level down
        assert t in report, t
