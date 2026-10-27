"""PazBrain wiki → mini export (pazbrain/scripts/export_for_mini.py) on a synthetic wiki with
a fake ssh: which pages are exported, how they render, deterministic archives, the guards
that keep a broken export from emptying the mini's index, and that the wiki is only read."""
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "pazbrain" / "scripts" / "export_for_mini.py"
spec = importlib.util.spec_from_file_location("export_for_mini", SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

PRIVATE_TIER = """---
title: AskJetson private tier
slug: askjetson-private-tier
category: decisions
tags: [askjetson, homelab]
updated: 2026-10-20
confidence: high
summary: The mini runs qwen3 and bge-m3.
---
# AskJetson private tier

## Decision
- Mini runs [[qwen3]] and bge-m3
  - 8 GB is enough
"""
TODO = """---
title: Homelab todo
slug: homelab-todo
category: projects
tags: []
summary: ''
---
- Replace the UPS battery
"""


def write_wiki(root: Path, pages: dict[str, str]) -> None:
    for rel, text in pages.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


WIKI = {
    "decisions/askjetson-private-tier.md": PRIVATE_TIER,
    "projects/homelab-todo.md": TODO,
    "inbox/no-frontmatter.md": "Just a note about the router\n",
    "index.md": "generated index",
    "log.md": "generated log",
    "_archive/old-page.md": "---\ntitle: Old\n---\nretired",
    ".git/notes.md": "never",
    "people/broken.md": "---\ntitle: [unclosed\n---\nbody",
}


class FakeSsh:
    def __init__(self, rc=0):
        self.rc, self.pushes = rc, []

    def run(self, argv, **kw):
        assert argv[0] == "ssh", argv           # the only program this script runs
        self.pushes.append((argv, kw.get("input")))
        return subprocess.CompletedProcess(argv, self.rc, b'{"received": {"pages": 3}}', b"refused: boom")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    root = tmp_path / "wiki"
    write_wiki(root, WIKI)
    ssh = FakeSsh()
    monkeypatch.setattr(exp.subprocess, "run", ssh.run)
    monkeypatch.setattr(exp, "WIKI", root)
    monkeypatch.setattr(exp, "STATE", tmp_path / "state.json")
    ssh.root = root
    return ssh


def members(data: bytes) -> dict[str, str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {m.name: tar.extractfile(m).read().decode() for m in tar.getmembers()}


def test_same_pages_as_the_logseq_mirror(fake):
    pages, skipped = exp.export()
    assert sorted(pages) == ["AskJetson private tier.md", "Homelab todo.md", "no frontmatter.md"]
    assert skipped == 1                                        # unreadable frontmatter


def test_pages_render_title_meta_summary_and_full_body(fake):
    md = exp.export()[0]["AskJetson private tier.md"].decode()
    assert md == ("# AskJetson private tier\n\n"
                  "Category: decisions · Tags: askjetson, homelab · Updated: 2026-10-20 · Confidence: high\n\n"
                  "Summary: The mini runs qwen3 and bge-m3.\n\n"
                  "## Decision\n- Mini runs [[qwen3]] and bge-m3\n  - 8 GB is enough\n")
    todo = exp.export()[0]["Homelab todo.md"].decode()
    assert todo == "# Homelab todo\n\nCategory: projects\n\n- Replace the UPS battery\n"   # empty fields left out
    assert exp.export()[0]["no frontmatter.md"].decode() == "# no frontmatter\n\nJust a note about the router\n"


def test_file_names_are_flat_and_unique():
    used: set[str] = set()
    assert exp.file_name("Notes", used) == "Notes.md"
    assert exp.file_name("notes", used) == "notes (2).md"
    assert exp.file_name("../../etc/passwd", used) == "_.._etc_passwd.md"
    assert exp.file_name("...", used) == "untitled.md"


def test_archive_is_deterministic_with_a_manifest(fake):
    a, b = exp.build_tar(exp.export()[0]), exp.build_tar(exp.export()[0])
    assert a == b
    assert json.loads(members(a)[exp.MANIFEST]) == {"source": "wiki", "pages": 3}


def test_push_uses_the_restricted_key_and_is_skipped_when_unchanged(fake):
    assert exp.main([]) == 0
    argv, data = fake.pushes[0]
    assert argv[:3] == ["ssh", "-i", str(exp.KEY)] and "StrictHostKeyChecking=yes" in argv
    assert members(data)["Homelab todo.md"].startswith("# Homelab todo")
    assert exp.main([]) == 0 and len(fake.pushes) == 1          # unchanged: nothing sent
    (fake.root / "projects/homelab-todo.md").write_text(TODO.replace("UPS battery", "router"))
    assert exp.main([]) == 0 and len(fake.pushes) == 2


def test_dry_run_never_pushes(fake):
    assert exp.main(["--dry-run"]) == 0 and fake.pushes == []


def test_a_much_smaller_export_is_not_pushed(fake):
    write_wiki(fake.root, {f"topics/page-{i}.md": f"---\ntitle: Page {i}\n---\nnote {i}" for i in range(20)})
    assert exp.main([]) == 0
    for i in range(16):
        (fake.root / f"topics/page-{i}.md").unlink()
    assert exp.main([]) == 1 and len(fake.pushes) == 1
    assert exp.main(["--force"]) == 0 and len(fake.pushes) == 2


def test_empty_or_missing_wiki_is_not_pushed(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "WIKI", tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    assert exp.main([]) == 1
    monkeypatch.setattr(exp, "WIKI", tmp_path / "missing")
    assert exp.main([]) == 1 and fake.pushes == []


def test_failed_push_is_retried_next_run(fake):
    fake.rc = 2
    assert exp.main([]) == 1
    fake.rc = 0
    assert exp.main([]) == 0 and len(fake.pushes) == 2           # state was not saved on failure


def test_the_wiki_is_only_read(fake):
    def digest():
        return hashlib.sha256(b"".join(p.read_bytes() + str(p.stat().st_mtime_ns).encode()
                                       for p in sorted(fake.root.rglob("*")) if p.is_file())).hexdigest()
    before = digest()
    exp.main([])
    exp.main(["--force"])
    assert digest() == before


def test_logs_carry_no_titles_or_text(fake, capsys):
    exp.main([])
    err = capsys.readouterr().err
    for secret in ("AskJetson", "Homelab", "UPS", "qwen3"):
        assert secret not in err
