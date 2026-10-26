#!/usr/bin/env python3
"""PazBrain Drive sync: Google Drive brain folder pazbrain/ -> local mirror -> PazBrain API.

One way only. Drive is read through the rclone remote "pazbrain-drive" (scope
drive.readonly) and copied into /opt/pazbrain/drive-mirror/; nothing is ever written to
Drive. Each new or changed <category>/<slug>.md is PUT to the PazBrain API as
updated_by=grok-app. Files deleted from Drive are never archived automatically: they are
listed on the inbox page "sync-deletions" for review. Writes the API refuses (secret
scrubber or schema) are listed on "sync-rejections" (rule + filename only).

    pazbrain_sync.py              # one run (the pazbrain-sync systemd timer runs this every 15 min)
    pazbrain_sync.py --dry-run    # sync the mirror, print planned API calls, send no writes

State: /opt/pazbrain/sync_state.db. Run summaries: /opt/pazbrain/logs/sync.log.
Page bodies and tokens are never printed or logged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
import yaml
from dotenv import load_dotenv

ENV_FILE = "/etc/pazbrain.env"
REMOTE = "pazbrain-drive:"
REMOTE_DIR = "pazbrain"
MIRROR = Path("/opt/pazbrain/drive-mirror")
STATE_DB = Path("/opt/pazbrain/sync_state.db")
LOG_FILE = Path("/opt/pazbrain/logs/sync.log")
LOCK_FILE = Path("/opt/pazbrain/sync.lock")
LOGSEQ_INGEST = Path("/opt/pazbrain/logseq_ingest.py")
LOGSEQ_TIMEOUT_S = 600

UPDATED_BY = "grok-app"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")  # same rule as pazbrain/store.py
HTTP_TIMEOUT_S = 30
REJECTIONS_PAGE = "sync-rejections"
DELETIONS_PAGE = "sync-deletions"
# Google Docs come through exported as Markdown; real .md files are copied unchanged.
RCLONE_SYNC_FLAGS = ["--drive-export-formats", "md,txt", "--include", "*.md", "--include", "*.txt"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(value):
    """YAML turns unquoted dates into date objects; the API wants strings."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def stem_of(path: Path) -> str:
    """sync-test.md -> sync-test; a Google Doc titled "sync-test.md" exports as sync-test.md.md."""
    name = path.name
    while name.endswith((".md", ".txt")):
        name = name[: -3] if name.endswith(".md") else name[: -4]
    return name


def parse_page(text: str) -> tuple[dict, str]:
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        raise ValueError("no YAML frontmatter")
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)(.*)$", text, re.S)
    if not m:
        raise ValueError("unterminated YAML frontmatter")
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter ({type(exc).__name__})") from None
    if not isinstance(fm, dict):
        raise ValueError("frontmatter is not a mapping")
    return _plain(fm), m.group(2)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def open_state() -> sqlite3.Connection:
    conn = sqlite3.connect(STATE_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS files (
               path TEXT PRIMARY KEY,
               sha256 TEXT,
               slug TEXT,
               result TEXT,
               detail TEXT,
               updated_at TEXT
           )"""
    )
    os.chmod(STATE_DB, 0o600)
    return conn


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class API:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {token}"

    def get_page(self, slug: str) -> tuple[int, dict]:
        r = self.http.get(f"{self.base}/pages/{quote(slug, safe='')}", timeout=HTTP_TIMEOUT_S)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})

    def put_page(self, slug: str, payload: dict) -> tuple[int, dict]:
        r = self.http.put(f"{self.base}/pages/{quote(slug, safe='')}", json=payload, timeout=HTTP_TIMEOUT_S)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})

    def patch_page(self, slug: str, payload: dict) -> tuple[int, dict]:
        r = self.http.patch(f"{self.base}/pages/{quote(slug, safe='')}", json=payload, timeout=HTTP_TIMEOUT_S)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})

    def note(self, slug: str, title: str, summary: str, lines: list[str]) -> str:
        """Append lines to an inbox review page, creating it if needed. Returns a status label."""
        if not lines:
            return "none"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = "\n".join(f"- {stamp}: {line}" for line in lines)
        status, page = self.get_page(slug)
        if status == 404:
            st, _ = self.put_page(slug, {
                "title": title, "category": "inbox", "confidence": "stated", "summary": summary,
                "tags": ["sync"], "sources": [{"type": "file", "ref": "pazbrain_sync.py"}],
                "body": f"## Log\n\n{content}\n", "updated_by": UPDATED_BY,
            })
            return f"create {st}"
        if status != 200:
            return f"get {status}"
        for attempt in range(2):
            st, body = self.patch_page(slug, {
                "updated_by": UPDATED_BY, "base_updated": page["updated"],
                "ops": [{"op": "append_section", "heading": "Log", "content": content}],
            })
            if st == 409 and attempt == 0:
                page = body.get("current") or self.get_page(slug)[1]
                continue
            return f"patch {st}"
        return "patch 409"


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def rclone_sync() -> tuple[bool, str]:
    """Mirror Drive pazbrain/ locally. Returns (source_present, problem)."""
    dirs = subprocess.run(["rclone", "lsf", REMOTE, "--dirs-only", "--max-depth", "1"],
                          capture_output=True, text=True, timeout=300)
    if dirs.returncode != 0:
        return False, f"rclone lsf failed (exit {dirs.returncode})"
    if f"{REMOTE_DIR}/" not in dirs.stdout.splitlines():
        return False, ""
    MIRROR.mkdir(mode=0o700, parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rclone", "sync", f"{REMOTE}{REMOTE_DIR}/", str(MIRROR), *RCLONE_SYNC_FLAGS],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        last = [l for l in proc.stderr.splitlines() if "ERROR" in l][-1:] or [f"exit {proc.returncode}"]
        return True, "rclone sync failed: " + last[0][-300:]
    return True, ""


def process_file(api: API | None, rel: str, path: Path, dry_run: bool) -> tuple[str, str, str]:
    """Returns (result, slug, detail). result: created|updated|rejected|error|planned."""
    try:
        fm, body = parse_page(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        return "rejected", "", str(exc)
    missing = [k for k in ("title", "slug", "category") if not fm.get(k)]
    if missing:
        return "rejected", str(fm.get("slug", "")), "missing " + ", ".join(missing)
    slug = str(fm["slug"])
    if not SLUG_RE.fullmatch(slug):
        return "rejected", slug, "slug fails ^[a-z0-9][a-z0-9-]{0,80}$"
    if stem_of(path) != slug:
        return "rejected", slug, f"slug does not match filename ({stem_of(path)})"
    parent = Path(rel).parent.name
    note = f" (folder {parent}/ but category {fm['category']})" if parent and parent != fm["category"] else ""

    payload = {
        "title": fm.get("title"), "category": fm.get("category"),
        "tags": fm.get("tags") or [], "sources": fm.get("sources") or [],
        "confidence": fm.get("confidence"), "summary": fm.get("summary"),
        "body": body.strip(), "updated_by": UPDATED_BY,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    status, current = api.get_page(slug)
    if status not in (200, 404):
        return "error", slug, f"GET {status}"
    if status == 200:
        payload["base_updated"] = current.get("updated")
    action = "update" if status == 200 else "create"
    if dry_run:
        print(f"  PLAN PUT /pages/{slug} ({action}{', base_updated=' + str(payload['base_updated']) if status == 200 else ''}) from {rel}{note}")
        return "planned", slug, action

    for attempt in range(2):
        st, resp = api.put_page(slug, payload)
        if st == 409 and attempt == 0:
            status, current = api.get_page(slug)
            if status == 200:
                payload["base_updated"] = current.get("updated")
            continue
        break
    if st == 200:
        return ("created" if resp.get("action") == "create" else "updated"), slug, (resp.get("commit") or "")[:7] + note
    if st == 422 and resp.get("error") == "secret_detected":
        rules = sorted({m.get("rule", "?") for m in resp.get("matches", [])})
        return "rejected", slug, "secret scrubber: " + ", ".join(rules)
    if st == 422:
        fields = sorted({".".join(str(x) for x in e.get("loc", [])[1:]) or "body" for e in resp.get("detail", [])})
        return "rejected", slug, "schema: invalid " + ", ".join(fields)
    if st in (400, 413):
        return "rejected", slug, f"API {st}: {resp.get('error', '')}"
    return "error", slug, f"PUT {st} {resp.get('error', '')}".strip()


def logseq_ingest() -> str:
    """Mirror the wiki into the Logseq "pazbrain" graph. Never fails the sync; returns a status label.
    logseq_ingest.py stops the db-worker-node daemon at the end if it started it."""
    if not LOGSEQ_INGEST.exists():
        return "absent"
    try:
        proc = subprocess.run([sys.executable, str(LOGSEQ_INGEST)], capture_output=True, text=True,
                              timeout=LOGSEQ_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"timeout after {LOGSEQ_TIMEOUT_S}s"
    except Exception as exc:  # the wiki sync must never fail because of Logseq
        return f"failed to run ({type(exc).__name__})"
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    return (last.removeprefix("logseq_ingest: ") or "no summary") + ("" if proc.returncode == 0 else f" (exit {proc.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(prog="pazbrain_sync.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="sync the mirror and print planned API calls; send no writes")
    args = ap.parse_args()

    LOG_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("pazbrain_sync: another run is in progress; exiting")
        return 0

    load_dotenv(ENV_FILE)
    token, base = os.getenv("PAZBRAIN_WRITE_TOKEN", ""), os.getenv("PAZBRAIN_API", "")
    if not token or not base:
        sys.exit(f"pazbrain_sync: PAZBRAIN_WRITE_TOKEN / PAZBRAIN_API not set in {ENV_FILE}")
    api = API(base, token)

    summary = {"ts": now_iso(), "dry_run": args.dry_run, "seen": 0, "created": 0, "updated": 0,
               "skipped": 0, "rejected": 0, "errors": 0, "deleted": 0, "planned": 0, "note": ""}
    files_log: list[dict] = []
    exit_code = 0
    try:
        present, problem = rclone_sync()
        if problem:
            summary["note"] = problem
            summary["errors"] += 1
            exit_code = 1
            return exit_code
        if not present:
            summary["note"] = f"{REMOTE_DIR}/ not in Drive yet; nothing to sync"
            print(f"pazbrain_sync: {summary['note']}")
            return 0

        conn = open_state()
        known = {row[0]: row for row in conn.execute("SELECT path, sha256, slug, result FROM files")}
        seen: set[str] = set()
        rejections: list[str] = []
        for path in sorted(p for p in MIRROR.rglob("*") if p.is_file() and p.suffix in (".md", ".txt")):
            rel = str(path.relative_to(MIRROR))
            seen.add(rel)
            summary["seen"] += 1
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            prev = known.get(rel)
            if prev and prev[1] == sha and prev[3] != "deleted":
                summary["skipped"] += 1
                continue
            try:
                result, slug, detail = process_file(api, rel, path, args.dry_run)
            except requests.RequestException as exc:
                result, slug, detail = "error", "", f"HTTP {type(exc).__name__}"
            summary[result if result in summary else "errors"] += 1
            files_log.append({"file": rel, "slug": slug, "result": result, "detail": detail})
            print(f"  {result:8} {rel} {('-> ' + slug) if slug else ''} {detail}".rstrip(), file=sys.stderr)
            if args.dry_run:
                continue
            if result == "rejected":
                rejections.append(f"`{rel}` — {detail}")
            # Remember the hash only for outcomes that won't change on retry; errors retry next run.
            if result != "error":
                conn.execute(
                    "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
                    (rel, sha, slug, result, detail, now_iso()),
                )
            else:
                exit_code = 1
        conn.commit()

        # Deletions: only after a successful rclone sync, never archived automatically.
        gone = [r for p, r in known.items() if p not in seen and r[3] != "deleted"]
        deletions = [f"`{r[0]}`" + (f" (page [[{r[2]}]])" if r[2] else "") for r in gone]
        summary["deleted"] = len(gone)
        for r in gone:
            print(f"  deleted  {r[0]} (was {r[2] or 'no page'}) — listed for review, not archived", file=sys.stderr)
            files_log.append({"file": r[0], "slug": r[2], "result": "deleted_in_drive"})
        if not args.dry_run:
            for r in gone:
                conn.execute("UPDATE files SET result='deleted', detail='deleted in Drive', updated_at=? WHERE path=?",
                             (now_iso(), r[0]))
            conn.commit()
            summary["rejections_page"] = api.note(
                REJECTIONS_PAGE, "Sync rejections",
                "Drive files the sync could not import (rule + filename)", rejections)
            summary["deletions_page"] = api.note(
                DELETIONS_PAGE, "Sync deletions",
                "Pages whose Drive file was deleted; review and archive by hand", deletions)
            # Every run, not only when this sync wrote: the wiki also changes through other API writers.
            summary["logseq"] = logseq_ingest()
            print(f"pazbrain_sync: logseq {summary['logseq']}", file=sys.stderr)
        conn.close()
        return exit_code
    except Exception as exc:  # summarise, never dump content
        summary["errors"] += 1
        summary["note"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"pazbrain_sync: {summary['note']}", file=sys.stderr)
        return 1
    finally:
        summary["files"] = files_log
        fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print("pazbrain_sync: " + ", ".join(f"{k}={summary[k]}" for k in
              ("seen", "created", "updated", "skipped", "rejected", "errors", "deleted", "planned")) +
              (f" ({summary['note']})" if summary["note"] else ""))


if __name__ == "__main__":
    sys.exit(main())
