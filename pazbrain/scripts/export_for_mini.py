#!/usr/bin/env python3
"""Export the PazBrain Logseq graph to markdown and push it to the Mac mini's private index.

One way only: Jetson → mini. The graph is read with two read-only CLI commands
(`list page`, `show`); no upsert/remove/import is ever built. The pages go to the mini as
one tar stream over SSH with a dedicated key that, on the mini, can only run PazRAG's
scripts/pazbrain_receive.sh (it takes the export and re-indexes; nothing comes back but
counts). The mini holds no key for this host.

Each run renders every page (one `show` per page, ~0.5 s each) into a deterministic tar.
When it is identical to the last pushed export, nothing is sent. An export with no pages,
or with fewer than half the pages of the last push, is not sent (a broken graph or CLI
must not empty the index); --force overrides.

    export_for_mini.py              # export and push if changed (pazbrain-mini-export.service)
    export_for_mini.py --dry-run    # export only; print counts
    export_for_mini.py --force      # push even if unchanged or much smaller

Logs carry counts, sizes and timings only — never page titles or text.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

LOGSEQ = str(Path.home() / ".local" / "bin" / "logseq")
GRAPH = "pazbrain"
CLI_TIMEOUT_MS = 30_000
PROC_TIMEOUT_S = 60
SHOW_LEVEL = 10
MANIFEST = ".manifest.json"

MINI = os.environ.get("PAZBRAIN_MINI_SSH", "jeffpaz@192.168.11.60")
KEY = Path(os.environ.get("PAZBRAIN_MINI_KEY", str(Path.home() / ".ssh" / "pazbrain_to_mini")))
STATE = Path(os.environ.get("PAZBRAIN_MINI_STATE", "/opt/pazbrain/run/mini-export.json"))
PUSH_TIMEOUT_S = 900  # the mini re-indexes changed pages before answering

# The only CLI commands this script may run. Anything else is a bug, not a feature.
READ_ONLY = {("list", "page"), ("show",)}
ENV = dict(os.environ, NODE_NO_WARNINGS="1")

_UNSAFE = re.compile(r"[/\\\x00-\x1f\x7f:]")


class ExportError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cli(*args: str) -> dict:
    if args[:2] not in READ_ONLY and args[:1] not in READ_ONLY:
        raise ExportError(f"refusing non-read-only CLI command {args[:2]}")
    argv = [LOGSEQ, *args, f"--graph={GRAPH}", "--output=json", f"--timeout-ms={CLI_TIMEOUT_MS}"]
    proc = subprocess.run(argv, capture_output=True, text=True, env=ENV, timeout=PROC_TIMEOUT_S)
    if proc.returncode != 0:
        raise ExportError(f"logseq {args[0]} exited {proc.returncode}")
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        raise ExportError(f"logseq {args[0]} returned non-JSON") from None
    if out.get("status") not in (None, "ok"):
        raise ExportError(f"logseq {args[0]} status {out.get('status')}")
    return out.get("data", out)


def user_pages() -> list[dict]:
    """Pages you wrote: built-in property / class pages carry a db/ident and are skipped."""
    data = cli("list", "page", "--limit=100000")
    items = data.get("items", data) if isinstance(data, dict) else data
    return [p for p in items if not p.get("db/ident") and (p.get("block/title") or "").strip()]


def _ordered(blocks: list[dict]) -> list[dict]:
    return sorted(blocks or [], key=lambda b: b.get("block/order") or "")


def render(title: str, root: dict) -> str:
    """Markdown: `# title`, heading blocks as headings, other blocks as nested bullets."""
    lines = [f"# {title}", ""]

    def walk(blocks: list[dict], depth: int) -> None:
        for b in _ordered(blocks):
            text = (b.get("block/title") or "").strip()
            heading = b.get("logseq.property/heading")
            if text:
                if isinstance(heading, int) and heading > 0 and depth == 0:
                    lines.extend(["", "#" * min(6, heading + 1) + " " + text.replace("\n", " "), ""])
                else:
                    first, *rest = text.splitlines()
                    lines.append("  " * depth + "- " + first)
                    lines.extend("  " * (depth + 1) + r for r in rest)
            walk(b.get("block/children") or [], depth + 1)

    walk(root.get("block/children") or [], 0)
    return "\n".join(lines).rstrip() + "\n"


def file_name(title: str, used: set[str]) -> str:
    base = _UNSAFE.sub("_", title).strip().strip(".")[:150] or "untitled"
    name, n = f"{base}.md", 2
    while name.casefold() in used:
        name, n = f"{base} ({n}).md", n + 1
    used.add(name.casefold())
    return name


def export() -> dict[str, bytes]:
    pages: dict[str, bytes] = {}
    used: set[str] = set()
    for p in sorted(user_pages(), key=lambda p: p["block/title"].casefold()):
        title = p["block/title"].strip()
        root = cli("show", f"--page={title}", f"--level={SHOW_LEVEL}").get("root") or {}
        pages[file_name(title, used)] = render(title, root).encode()
    return pages


def build_tar(pages: dict[str, bytes]) -> bytes:
    """Deterministic: same pages, same bytes (sorted, fixed mtime/owner/mode)."""
    files = {**pages, MANIFEST: json.dumps({"graph": GRAPH, "pages": len(pages)}, sort_keys=True).encode()}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(files[name]), 0, 0o644, 0, 0
            tar.addfile(info, io.BytesIO(files[name]))
    return buf.getvalue()


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE)


def push(data: bytes) -> str:
    argv = ["ssh", "-i", str(KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", MINI, "pazbrain-receive"]
    proc = subprocess.run(argv, input=data, capture_output=True, timeout=PUSH_TIMEOUT_S)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:] or [""]
        raise ExportError(f"push exited {proc.returncode}: {detail[0][:200]}")
    return proc.stdout.decode(errors="replace").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    try:
        pages = export()
    except (ExportError, subprocess.TimeoutExpired, KeyError) as exc:
        log(f"export failed: {type(exc).__name__}: {exc}")
        return 1
    data = build_tar(pages)
    digest = hashlib.sha256(data).hexdigest()
    state = load_state()
    log(f"exported pages={len(pages)} bytes={len(data)} seconds={time.monotonic() - t0:.1f}")
    if args.dry_run:
        return 0
    if not args.force:
        if not pages:
            log("not pushed: the export has no pages")
            return 1
        last = state.get("pages") or 0
        if last >= 10 and len(pages) < last / 2:
            log(f"not pushed: {len(pages)} pages vs {last} last time (use --force if intended)")
            return 1
        if state.get("sha256") == digest:
            log("unchanged since the last push: nothing sent")
            return 0
    try:
        reply = push(data)
    except (ExportError, subprocess.TimeoutExpired) as exc:
        log(f"push failed: {type(exc).__name__}: {exc}")
        return 1
    save_state({"sha256": digest, "pages": len(pages), "pushed_at": int(time.time())})
    log(f"pushed; mini replied {reply[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
