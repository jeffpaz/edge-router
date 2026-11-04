"""PazBrain wiki store (read-only) plus the raw/inbox drop box.

Layout under ROOT (/opt/pazbrain, one git repo; see ROOT/AGENTS.md):
  wiki/<folder>/<slug>.md     pages — YAML frontmatter + Markdown body, written only by the librarian
  wiki/projects/<room>/AGENTS.md  project rooms (not pages; ignored here)
  wiki/_archive/              retired pages (ignored)
  index.md, log.md            maintained by the librarian; log.md feeds /recent
  raw/inbox/                  the only place this module writes (add_to_inbox)

INDEX_DB is derived data (pages, FTS5, wiki-link graph). It is rebuilt automatically whenever the
page files change (checked on every read: one stat per page), or by `python -m pazbrain.reindex`.

Visibility: pages marked `visibility: local-only` are left out of every read made with
include_local=False (routes.py passes that for requests that came through the Cloudflare Tunnel).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import threading
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

WIKI_DIR = Path(os.getenv("PAZBRAIN_WIKI_DIR", "/opt/pazbrain/wiki"))
ROOT = Path(os.getenv("PAZBRAIN_ROOT", str(WIKI_DIR.parent)))
INDEX_DB = Path(os.getenv("PAZBRAIN_INDEX_DB", "/opt/pazbrain/index.db"))
INBOX_DIR = ROOT / "raw" / "inbox"
LOG_MD = ROOT / "log.md"

# type -> folder under wiki/
FOLDERS = {"person": "people", "project": "projects", "area": "areas", "topic": "topics",
           "source": "sources", "concept": "concepts"}
TYPES = tuple(FOLDERS)
VISIBILITY = ("cloud", "local-only")
ARCHIVE = "_archive"
MAX_INBOX_BYTES = 200 * 1024
SCHEMA_VERSION = 2

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
_LINK_RE = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|[^\[\]\n]*)?\]\]")
_LOG_RE = re.compile(r"^## \[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\] (.+?) \| (.+?) \| (.*)$")
# The Jetson's clock and log.md are Phoenix time (no DST).
_LOG_TZ = "-07:00"

_lock = threading.RLock()
_signature: str | None = None


# ---------------------------------------------------------------------------
# Errors — routes.py maps .status onto the HTTP response
# ---------------------------------------------------------------------------


class StoreError(Exception):
    status = 400
    kind = "invalid"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidInput(StoreError):
    status, kind = 400, "invalid"


class NotFound(StoreError):
    status, kind = 404, "not_found"


class TooLarge(StoreError):
    status, kind = 413, "too_large"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def slugify(text: str) -> str:
    """Same rule the frontend uses to resolve [[Some Page]] to a slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:81]


def check_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise InvalidInput("slug must match ^[a-z0-9][a-z0-9-]{0,80}$")
    return slug


def _one_line(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


def _plain(value: Any) -> Any:
    """YAML may hand back datetimes for unquoted timestamps; keep everything as strings."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def extract_links(text: str, own_slug: str = "") -> list[str]:
    links: list[str] = []
    for m in _LINK_RE.finditer(text or ""):
        target = slugify(m.group(1))
        if SLUG_RE.fullmatch(target) and target != own_slug and target not in links:
            links.append(target)
    return links


# ---------------------------------------------------------------------------
# Page files
# ---------------------------------------------------------------------------


def _parse(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    fm = _plain(yaml.safe_load(text[4:end]) or {})
    if not isinstance(fm, dict):
        raise ValueError("frontmatter is not a mapping")
    return fm, text[end + 5 :].lstrip("\n")


def _page_files() -> list[Path]:
    """Every page file: wiki/<folder>/<slug>.md for the six type folders (rooms and _archive excluded)."""
    files = []
    for folder in FOLDERS.values():
        files += sorted((WIKI_DIR / folder).glob("*.md"))
    return [p for p in files if SLUG_RE.fullmatch(p.stem)]


def _find(slug: str) -> Path | None:
    for folder in FOLDERS.values():
        path = WIKI_DIR / folder / f"{slug}.md"
        if path.is_file():
            return path
    return None


def _read(path: Path) -> tuple[dict[str, Any], str]:
    fm, body = _parse(path.read_text(encoding="utf-8"))
    fm["slug"] = path.stem
    fm["category"] = path.parent.name  # folder; kept for older API clients
    fm.setdefault("type", next((t for t, f in FOLDERS.items() if f == path.parent.name), ""))
    if fm.get("visibility") not in VISIBILITY:
        fm["visibility"] = "local-only"  # unknown or missing: fail closed
    return fm, body


def _related(fm: dict[str, Any]) -> list[str]:
    return extract_links(" ".join(str(r) for r in fm.get("related") or []))


def _git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(WIKI_DIR), *args], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Index DB (derived; rebuilt when the page files change)
# ---------------------------------------------------------------------------

_SCHEMA = """
DROP TABLE IF EXISTS pages;
DROP TABLE IF EXISTS pages_fts;
DROP TABLE IF EXISTS links;
CREATE TABLE pages (
    slug TEXT PRIMARY KEY, title TEXT, type TEXT, category TEXT, visibility TEXT, tags TEXT,
    summary TEXT, created TEXT, updated TEXT, confidence TEXT
);
CREATE VIRTUAL TABLE pages_fts USING fts5(
    slug UNINDEXED, title, summary, tags, body, tokenize = 'porter unicode61'
);
CREATE TABLE links (src TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY (src, dst));
CREATE INDEX links_dst ON links (dst);
"""


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(INDEX_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _files_signature() -> str:
    h = hashlib.sha256()
    for p in _page_files():
        st = p.stat()
        h.update(f"{p.relative_to(WIKI_DIR)}\0{st.st_mtime_ns}\0{st.st_size}\n".encode())
    return h.hexdigest()


def rebuild_index() -> int:
    """Rebuild INDEX_DB from the page files. Returns the page count. Unreadable pages are skipped."""
    global _signature
    with _lock:
        sig = _files_signature()
        count = 0
        with closing(_db()) as conn, conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            for path in _page_files():
                try:
                    fm, body = _read(path)
                except (ValueError, yaml.YAMLError, OSError):
                    continue
                tags = [str(t) for t in fm.get("tags") or []]
                conn.execute(
                    "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fm["slug"], str(fm.get("title", "")), fm["type"], fm["category"], fm["visibility"],
                     json.dumps(tags), str(fm.get("summary", "")), str(fm.get("created", "")),
                     str(fm.get("updated", "")), str(fm.get("confidence", ""))),
                )
                conn.execute(
                    "INSERT INTO pages_fts (slug, title, summary, tags, body) VALUES (?, ?, ?, ?, ?)",
                    (fm["slug"], str(fm.get("title", "")), str(fm.get("summary", "")), " ".join(tags), body),
                )
                dsts = extract_links(body, fm["slug"])
                dsts += [d for d in _related(fm) if d not in dsts and d != fm["slug"]]
                conn.executemany("INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)",
                                 [(fm["slug"], d) for d in dsts])
                count += 1
        _signature = sig
        return count


def ensure_ready() -> None:
    """Raise if the wiki is missing; rebuild the index if the page files changed since the last build."""
    if not WIKI_DIR.is_dir():
        raise RuntimeError(f"no wiki at {WIKI_DIR}")
    with _lock:
        if _signature is None:
            with closing(_db()) as conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                rebuild_index()
                return
        if _files_signature() != _signature:
            rebuild_index()


def _hidden(conn: sqlite3.Connection, include_local: bool) -> set[str]:
    if include_local:
        return set()
    return {r["slug"] for r in conn.execute("SELECT slug FROM pages WHERE visibility != 'cloud'")}


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_page(slug: str, include_local: bool = False) -> dict[str, Any]:
    check_slug(slug)
    ensure_ready()
    path = _find(slug)
    if path is None:
        raise NotFound(f"no page {slug!r}")
    fm, body = _read(path)
    if fm["visibility"] != "cloud" and not include_local:
        raise NotFound(f"no page {slug!r}")  # same answer as a missing page: reveal nothing
    page = {k: fm.get(k) for k in (
        "title", "slug", "type", "category", "visibility", "summary", "created", "updated", "confidence",
    )}
    page["tags"] = [str(t) for t in fm.get("tags") or []]
    page["sources"] = [str(s) for s in fm.get("sources") or []]
    page["related"] = _related(fm)
    page["updated_by"] = None  # not tracked in frontmatter any more; see /history
    page["body"] = body
    with closing(_db()) as conn:
        hidden = _hidden(conn, include_local)
        rows = conn.execute(
            "SELECT l.src AS slug, p.title FROM links l JOIN pages p ON p.slug = l.src "
            "WHERE l.dst = ? ORDER BY lower(p.title)", (slug,),
        ).fetchall()
        page["backlinks"] = [dict(r) for r in rows if r["slug"] not in hidden]
        # Which outgoing [[links]] point at pages this caller can see, so the UI can mark the rest red.
        out = extract_links(body, slug)
        existing = {
            r["slug"] for r in conn.execute(
                f"SELECT slug FROM pages WHERE slug IN ({','.join('?' * len(out))})", out
            )
        } - hidden if out else set()
        page["links"] = [{"slug": s, "exists": s in existing} for s in out]
    return page


def list_pages(type_: str | None = None, tag: str | None = None, limit: int = 100,
               include_local: bool = False) -> list[dict[str, Any]]:
    ensure_ready()
    sql = "SELECT slug, title, type, category, visibility, summary, updated, tags FROM pages WHERE 1=1"
    args: list[Any] = []
    if not include_local:
        sql += " AND visibility = 'cloud'"
    if type_:
        sql += " AND type = ?"
        args.append(type_)
    if tag:
        sql += " AND EXISTS (SELECT 1 FROM json_each(pages.tags) WHERE value = ?)"
        args.append(tag.strip().lower())
    sql += " ORDER BY updated DESC LIMIT ?"
    args.append(limit)
    with closing(_db()) as conn:
        return [{**dict(r), "tags": json.loads(r["tags"] or "[]")} for r in conn.execute(sql, args)]


def search(q: str, limit: int = 20, include_local: bool = False) -> list[dict[str, Any]]:
    ensure_ready()
    terms = re.findall(r"\w+", q or "", flags=re.UNICODE)[:12]
    if not terms:
        return []
    # Quote every term so user input can never be parsed as FTS5 syntax; prefix-match the last.
    expr = " ".join(f'"{t}"' for t in terms[:-1]) + f' "{terms[-1]}"*'
    sql = (
        "SELECT p.slug, p.title, p.type, p.category, p.visibility, p.summary, p.updated, "
        "snippet(pages_fts, -1, char(2), char(3), '…', 16) AS snippet "
        "FROM pages_fts JOIN pages p ON p.slug = pages_fts.slug "
        "WHERE pages_fts MATCH ?" + ("" if include_local else " AND p.visibility = 'cloud'") +
        " ORDER BY bm25(pages_fts, 0.0, 10.0, 5.0, 3.0, 1.0) LIMIT ?"
    )
    with closing(_db()) as conn:
        rows = conn.execute(sql, (expr.strip(), limit)).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        # Escape the page text; only our own <mark> tags survive.
        item["snippet"] = html.escape(item["snippet"] or "").replace("\x02", "<mark>").replace("\x03", "</mark>")
        out.append(item)
    return out


def history(slug: str, limit: int = 50, include_local: bool = False) -> list[dict[str, Any]]:
    check_slug(slug)
    ensure_ready()
    path = _find(slug)
    if path is not None and not include_local and _read(path)[0]["visibility"] != "cloud":
        raise NotFound(f"no page {slug!r}")
    # Paths are relative to wiki/; history from before the move (decisions/, journal/) still matches.
    out = _git(
        "log", f"-n{limit}", "--format=%H%x1f%an%x1f%aI%x1f%s", "--",
        f":(glob)*/{slug}.md", f":(glob){ARCHIVE}/{slug}.*.md",
    )
    entries = []
    for line in out.splitlines():
        commit, author, when, subject = line.split("\x1f", 3)
        entries.append({"commit": commit, "author": author, "date": when, "message": subject})
    if not entries and path is None:
        raise NotFound(f"no page {slug!r}")
    return entries


def recent(limit: int = 50, include_local: bool = False) -> list[dict[str, Any]]:
    """Entries from ROOT/log.md (`## [YYYY-MM-DD HH:MM] op | agent | summary`), newest first.
    slug is the first [[link]] in the summary, or ""."""
    ensure_ready()
    try:
        lines = LOG_MD.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    with closing(_db()) as conn:
        hidden = _hidden(conn, include_local)
    out = []
    for line in reversed(lines):
        m = _LOG_RE.match(line)
        if not m:
            continue
        day, hm, action, by, summary = m.groups()
        links = extract_links(summary)
        if hidden & set(links):
            continue
        out.append({"date": f"{day}T{hm}:00{_LOG_TZ}", "updated_by": by.strip(), "action": action.strip(),
                    "slug": links[0] if links else "", "summary": summary.strip()})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# The one write: drop material into raw/inbox for the librarian
# ---------------------------------------------------------------------------


def add_to_inbox(*, title: str, content: str, source_agent: str) -> dict[str, Any]:
    """Write raw/inbox/YYYY-MM-DD-HHMMSS-<slug>.md. Never touches wiki/. Returns the relative path."""
    title = _one_line(title, 200)
    if not title:
        raise InvalidInput("title is required")
    if not content.strip():
        raise InvalidInput("content is required")
    received = datetime.now().astimezone()
    meta = {"title": title, "source_agent": source_agent, "received": received.isoformat(timespec="seconds")}
    text = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000) + "---\n\n" + content.strip() + "\n"
    if len(text.encode("utf-8")) > MAX_INBOX_BYTES:
        raise TooLarge(f"inbox item exceeds {MAX_INBOX_BYTES} bytes")
    base = f"{received:%Y-%m-%d-%H%M%S}-{slugify(title)[:60] or 'item'}"
    with _lock:
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        path, n = INBOX_DIR / f"{base}.md", 2
        while path.exists():
            path, n = INBOX_DIR / f"{base}-{n}.md", n + 1
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    return {"path": str(path.relative_to(ROOT)), "received": meta["received"]}
