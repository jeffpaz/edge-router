"""PazBrain → mini export (pazbrain/scripts/export_for_mini.py) with a fake Logseq CLI and a
fake ssh: read-only commands only, markdown rendering, deterministic archives, and the
guards that keep a broken export from emptying the mini's index."""
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

PAGES = {
    "AskJetson private tier": [
        {"block/title": "Decisions", "logseq.property/heading": 2, "block/order": "a0"},
        {"block/title": "Mini runs [[qwen3]] and bge-m3", "block/order": "a1",
         "block/children": [{"block/title": "8 GB is enough", "block/order": "a0"}]},
    ],
    "Homelab todo": [{"block/title": "Replace the UPS battery", "block/order": "a0"}],
    "a/b: odd name": [{"block/title": "x", "block/order": "a0"}],
}


class FakeSystem:
    def __init__(self, pages=PAGES, reply=b'{"received": {"pages": 3}}', push_rc=0):
        self.pages, self.reply, self.push_rc = dict(pages), reply, push_rc
        self.cli_calls, self.pushes = [], []

    def run(self, argv, **kw):
        if argv[0] == "ssh":
            self.pushes.append((argv, kw.get("input")))
            return subprocess.CompletedProcess(argv, self.push_rc, self.reply, b"refused: boom")
        self.cli_calls.append(argv[1:])
        if argv[1:3] == ["list", "page"]:
            items = [{"block/title": t, "db/id": i} for i, t in enumerate(self.pages)]
            items.append({"block/title": "Tags", "db/ident": "logseq.class/Tag"})
            data = {"items": items}
        elif argv[1] == "show":
            title = next(a for a in argv if a.startswith("--page="))[len("--page="):]
            data = {"root": {"block/title": title, "block/children": self.pages[title]}}
        else:
            raise AssertionError(f"unexpected CLI call {argv}")
        return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "ok", "data": data}), "")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    fs = FakeSystem()
    monkeypatch.setattr(exp.subprocess, "run", fs.run)
    monkeypatch.setattr(exp, "STATE", tmp_path / "state.json")
    return fs


def members(data: bytes) -> dict[str, str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {m.name: tar.extractfile(m).read().decode() for m in tar.getmembers()}


def test_only_read_only_cli_commands_run(fake):
    exp.export()
    assert {tuple(c[:2]) if c[0] == "list" else (c[0],) for c in fake.cli_calls} <= exp.READ_ONLY
    for bad in (("upsert", "page"), ("remove", "page"), ("graph", "import"), ("list", "task"), ("query",)):
        with pytest.raises(exp.ExportError, match="non-read-only"):
            exp.cli(*bad)


def test_pages_render_to_markdown(fake):
    pages = exp.export()
    assert sorted(pages) == ["AskJetson private tier.md", "Homelab todo.md", "a_b_ odd name.md"]  # no built-ins
    md = pages["AskJetson private tier.md"].decode()
    assert md == ("# AskJetson private tier\n\n\n### Decisions\n\n- Mini runs [[qwen3]] and bge-m3\n"
                  "  - 8 GB is enough\n")


def test_file_names_are_flat_and_unique():
    used: set[str] = set()
    assert exp.file_name("Notes", used) == "Notes.md"
    assert exp.file_name("notes", used) == "notes (2).md"
    assert exp.file_name("../../etc/passwd", used) == "_.._etc_passwd.md"
    assert exp.file_name("...", used) == "untitled.md"


def test_archive_is_deterministic_with_a_manifest(fake):
    a, b = exp.build_tar(exp.export()), exp.build_tar(exp.export())
    assert a == b
    m = members(a)
    assert json.loads(m[exp.MANIFEST]) == {"graph": "pazbrain", "pages": 3}


def test_push_goes_to_the_restricted_key_and_is_skipped_when_unchanged(fake):
    assert exp.main([]) == 0
    argv, data = fake.pushes[0]
    assert argv[:3] == ["ssh", "-i", str(exp.KEY)] and "StrictHostKeyChecking=yes" in argv
    assert members(data)["Homelab todo.md"].startswith("# Homelab todo")
    assert exp.main([]) == 0 and len(fake.pushes) == 1          # unchanged: nothing sent
    fake.pages["Homelab todo"] = [{"block/title": "Battery replaced", "block/order": "a0"}]
    assert exp.main([]) == 0 and len(fake.pushes) == 2


def test_dry_run_never_pushes(fake):
    assert exp.main(["--dry-run"]) == 0 and fake.pushes == []


def test_a_much_smaller_export_is_not_pushed(fake):
    fake.pages = {f"Page {i}": [{"block/title": "x", "block/order": "a0"}] for i in range(20)}
    assert exp.main([]) == 0
    fake.pages = dict(list(fake.pages.items())[:4])
    assert exp.main([]) == 1 and len(fake.pushes) == 1
    assert exp.main(["--force"]) == 0 and len(fake.pushes) == 2


def test_empty_graph_is_not_pushed(fake):
    fake.pages = {}
    assert exp.main([]) == 1 and fake.pushes == []


def test_failed_push_is_retried_next_run(fake):
    fake.push_rc = 2
    assert exp.main([]) == 1
    fake.push_rc = 0
    assert exp.main([]) == 0 and len(fake.pushes) == 2           # state was not saved on failure


def test_logs_carry_no_titles_or_text(fake, capsys):
    exp.main([])
    err = capsys.readouterr().err
    for secret in ("AskJetson", "Homelab", "UPS", "qwen3"):
        assert secret not in err
