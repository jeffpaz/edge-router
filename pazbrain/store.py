"""PazBrain wiki store: Markdown pages in a git repo, plus a SQLite FTS5 index.

Layout under WIKI_DIR (one git repo, one commit per write):
  <category>/<slug>.md   live pages — YAML frontmatter + Markdown body
  _archive/<slug>.md     soft-deleted pages
  index.md               generated table of contents
  log.md                 append-only change log, one line per write

INDEX_DB is derived data (pages, FTS5, wiki-link graph); `python -m pazbrain.reindex`
rebuilds it from the files at any time.

All writes are serialised by one lock: the edge-router runs a single uvicorn worker,
and sync route handlers run in its threadpool.
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import subprocess
import threading
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

WIKI_DIR = Path(os.getenv("PAZBRAIN_WIKI_DIR", "/opt/pazbrain/wiki"))
INDEX_DB = Path(os.getenv("PAZBRAIN_INDEX_DB", "/opt/pazbrain/index.db"))

CATEGORIES = ("people", "projects", "areas", "topics", "decisions", "journal", "inbox")
CONFIDENCE = ("stated", "inferred")
SOURCE_TYPES = ("email", "file", "chat")
ARCHIVE = "_archive"
MAX_PAGE_BYTES = 100 * 1024

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_./-]{0,49}$")
AUTHOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_LINK_RE = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|[^\[\]\n]*)?\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

_lock = threading.RLock()
_ready = False


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


class Conflict(StoreError):
    status, kind = 409, "conflict"

    def __init__(self, detail: str, current: dict[str, Any]) -> None:
        super().__init__(detail)
        self.current = current


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _next_updated(prev: str | None) -> str:
    """A fresh `updated` stamp, strictly later than prev so base_updated checks can't collide."""
    stamp = now_iso()
    if prev and stamp <= prev:
        try:
            bumped = datetime.fromisoformat(prev.replace("Z", "+00:00")) + timedelta(milliseconds=1)
            stamp = bumped.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except ValueError:
            pass
    return stamp


def slugify(text: str) -> str:
    """Same rule the frontend uses to resolve [[Some Page]] to a slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:81]


def check_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise InvalidInput("slug must match ^[a-z0-9][a-z0-9-]{0,80}$")
    return slug


def _inside_wiki(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != WIKI_DIR.resolve() and WIKI_DIR.resolve() not in resolved.parents:
        raise InvalidInput("path escapes the wiki")
    return resolved


def _one_line(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


def _norm_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for raw in tags or []:
        tag = re.sub(r"\s+", "-", str(raw).strip().lower())
        if not TAG_RE.fullmatch(tag):
            raise InvalidInput(f"invalid tag {tag[:60]!r}")
        if tag not in out:
            out.append(tag)
    if len(out) > 50:
        raise InvalidInput("at most 50 tags")
    return out


def _norm_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for src in sources or []:
        stype = str(src.get("type", ""))
        ref = _one_line(src.get("ref", ""), 500)
        if stype not in SOURCE_TYPES or not ref:
            raise InvalidInput("each source needs type (email|file|chat) and ref")
        if (stype, ref) in seen:
            continue
        seen.add((stype, ref))
        item: dict[str, Any] = {"type": stype, "ref": ref}
        if src.get("date"):
            item["date"] = _one_line(src["date"], 40)
        out.append(item)
    if len(out) > 200:
        raise InvalidInput("at most 200 sources")
    return out


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


def extract_links(body: str, own_slug: str = "") -> list[str]:
    links: list[str] = []
    for m in _LINK_RE.finditer(body or ""):
        target = slugify(m.group(1))
        if SLUG_RE.fullmatch(target) and target != own_slug and target not in links:
            links.append(target)
    return links


# ---------------------------------------------------------------------------
# Page files
# ---------------------------------------------------------------------------


def _serialize(fm: dict[str, Any], body: str) -> str:
    dumped = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=1000)
    return f"---\n{dumped}---\n\n{body.strip()}\n"


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


def _find(slug: str) -> Path | None:
    for cat in CATEGORIES:
        path = WIKI_DIR / cat / f"{slug}.md"
        if path.is_file():
            return path
    return None


def _read(path: Path) -> tuple[dict[str, Any], str]:
    return _parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(WIKI_DIR), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _commit(message: str, author: str) -> str:
    _git("add", "-A")
    _git(
        "-c", f"user.name={author}",
        "-c", f"user.email={author.lower()}@pazbrain.local",
        "commit", "-q", "-m", message,
    )
    return _git("rev-parse", "HEAD").strip()


# ---------------------------------------------------------------------------
# Index DB
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    slug TEXT PRIMARY KEY, title TEXT, category TEXT, tags TEXT,
    summary TEXT, updated TEXT, updated_by TEXT, confidence TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    slug UNINDEXED, title, summary, tags, body, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS links (src TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY (src, dst));
CREATE INDEX IF NOT EXISTS links_dst ON links (dst);
"""


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(INDEX_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _unindex(conn: sqlite3.Connection, slug: str) -> None:
    conn.execute("DELETE FROM pages WHERE slug = ?", (slug,))
    conn.execute("DELETE FROM pages_fts WHERE slug = ?", (slug,))
    conn.execute("DELETE FROM links WHERE src = ?", (slug,))


def _index(conn: sqlite3.Connection, fm: dict[str, Any], body: str) -> None:
    slug = fm["slug"]
    _unindex(conn, slug)
    tags = fm.get("tags") or []
    conn.execute(
        "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (slug, fm.get("title", ""), fm.get("category", ""), json.dumps(tags),
         fm.get("summary", ""), fm.get("updated", ""), fm.get("updated_by", ""),
         fm.get("confidence", "")),
    )
    conn.execute(
        "INSERT INTO pages_fts (slug, title, summary, tags, body) VALUES (?, ?, ?, ?, ?)",
        (slug, fm.get("title", ""), fm.get("summary", ""), " ".join(tags), body),
    )
    conn.executemany(
        "INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)",
        [(slug, dst) for dst in extract_links(body, slug)],
    )


def _write_index_md(conn: sqlite3.Connection) -> None:
    lines = ["# PazBrain index", "", "_Generated on every write. Do not edit by hand._", ""]
    for cat in CATEGORIES:
        rows = conn.execute(
            "SELECT slug, title, summary FROM pages WHERE category = ? ORDER BY lower(title)", (cat,)
        ).fetchall()
        lines += [f"## {cat.capitalize()} ({len(rows)})", ""]
        lines += [f"- [[{r['slug']}|{r['title']}]] — {r['summary']}" for r in rows] or ["_none_"]
        lines.append("")
    (WIKI_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _append_log(action: str, slug: str, updated_by: str, summary: str) -> None:
    line = f"- {now_iso()} | {updated_by} | {action} | {slug} | {_one_line(summary, 300).replace('|', '/')}\n"
    with open(WIKI_DIR / "log.md", "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def init_repo() -> None:
    """Create the wiki repo and index schema if missing. Idempotent."""
    global _ready
    with _lock:
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        for sub in (*CATEGORIES, ARCHIVE):
            (WIKI_DIR / sub).mkdir(exist_ok=True)
            keep = WIKI_DIR / sub / ".gitkeep"
            if not keep.exists():
                keep.touch()
        with closing(_db()) as conn, conn:
            conn.executescript(_SCHEMA)
        if not (WIKI_DIR / ".git").exists():
            _git("init", "-q", "-b", "main")
            (WIKI_DIR / "log.md").write_text("# PazBrain change log\n\n", encoding="utf-8")
            with closing(_db()) as conn, conn:
                _write_index_md(conn)
            _commit("[system] init wiki", "system")
        _ready = True


def ensure_ready() -> None:
    if not _ready:
        init_repo()


def rebuild_index() -> int:
    """Rebuild INDEX_DB from the page files; commit index.md if it changed. Returns page count."""
    with _lock:
        ensure_ready()
        count = 0
        with closing(_db()) as conn, conn:
            conn.execute("DELETE FROM pages")
            conn.execute("DELETE FROM pages_fts")
            conn.execute("DELETE FROM links")
            for cat in CATEGORIES:
                for path in sorted((WIKI_DIR / cat).glob("*.md")):
                    fm, body = _read(path)
                    fm["slug"], fm["category"] = path.stem, cat
                    _index(conn, fm, body)
                    count += 1
            _write_index_md(conn)
        if _git("status", "--porcelain").strip():
            _commit("[system] reindex", "system")
        return count


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _page_dict(fm: dict[str, Any], body: str) -> dict[str, Any]:
    page = {k: fm.get(k) for k in (
        "title", "slug", "category", "tags", "sources", "created", "updated",
        "updated_by", "confidence", "summary",
    )}
    page["tags"] = page["tags"] or []
    page["sources"] = page["sources"] or []
    page["body"] = body
    return page


def get_page(slug: str) -> dict[str, Any]:
    check_slug(slug)
    ensure_ready()
    path = _find(slug)
    if path is None:
        raise NotFound(f"no page {slug!r}")
    fm, body = _read(path)
    page = _page_dict(fm, body)
    with closing(_db()) as conn:
        rows = conn.execute(
            "SELECT l.src AS slug, p.title FROM links l JOIN pages p ON p.slug = l.src "
            "WHERE l.dst = ? ORDER BY lower(p.title)",
            (slug,),
        ).fetchall()
        page["backlinks"] = [dict(r) for r in rows]
        # Which outgoing [[links]] point at pages that exist, so the UI can mark the rest red.
        out = extract_links(body, slug)
        existing = {
            r["slug"] for r in conn.execute(
                f"SELECT slug FROM pages WHERE slug IN ({','.join('?' * len(out))})", out
            )
        } if out else set()
        page["links"] = [{"slug": s, "exists": s in existing} for s in out]
    return page


def list_pages(category: str | None = None, tag: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    ensure_ready()
    sql = "SELECT slug, title, category, summary, updated, tags FROM pages WHERE 1=1"
    args: list[Any] = []
    if category:
        sql += " AND category = ?"
        args.append(category)
    if tag:
        sql += " AND EXISTS (SELECT 1 FROM json_each(pages.tags) WHERE value = ?)"
        args.append(tag.strip().lower())
    sql += " ORDER BY updated DESC LIMIT ?"
    args.append(limit)
    with closing(_db()) as conn:
        return [{**dict(r), "tags": json.loads(r["tags"] or "[]")} for r in conn.execute(sql, args)]


def search(q: str, limit: int = 20) -> list[dict[str, Any]]:
    ensure_ready()
    terms = re.findall(r"\w+", q or "", flags=re.UNICODE)[:12]
    if not terms:
        return []
    # Quote every term so user input can never be parsed as FTS5 syntax; prefix-match the last.
    expr = " ".join(f'"{t}"' for t in terms[:-1]) + f' "{terms[-1]}"*'
    sql = (
        "SELECT p.slug, p.title, p.category, p.summary, p.updated, "
        "snippet(pages_fts, -1, char(2), char(3), '…', 16) AS snippet "
        "FROM pages_fts JOIN pages p ON p.slug = pages_fts.slug "
        "WHERE pages_fts MATCH ? ORDER BY bm25(pages_fts, 0.0, 10.0, 5.0, 3.0, 1.0) LIMIT ?"
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


def history(slug: str, limit: int = 50) -> list[dict[str, Any]]:
    check_slug(slug)
    ensure_ready()
    out = _git(
        "log", f"-n{limit}", "--format=%H%x1f%an%x1f%aI%x1f%s", "--",
        f":(glob)*/{slug}.md", f":(glob){ARCHIVE}/{slug}.*.md",
    )
    entries = []
    for line in out.splitlines():
        commit, author, when, subject = line.split("\x1f", 3)
        entries.append({"commit": commit, "author": author, "date": when, "message": subject})
    if not entries and _find(slug) is None:
        raise NotFound(f"no page {slug!r}")
    return entries


def recent(limit: int = 50) -> list[dict[str, Any]]:
    ensure_ready()
    lines = (WIKI_DIR / "log.md").read_text(encoding="utf-8").splitlines()
    out = []
    for line in reversed(lines):
        if not line.startswith("- "):
            continue
        parts = line[2:].split(" | ", 4)
        if len(parts) != 5:
            continue
        when, by, action, slug, summary = parts
        out.append({"date": when, "updated_by": by, "action": action, "slug": slug, "summary": summary})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _transaction(apply, message: str, author: str) -> str:
    """Run apply() (file + index changes), then commit. On any failure restore the
    working tree to HEAD and rebuild the index so files, index and git stay in step."""
    try:
        apply()
        return _commit(message, author)
    except Exception:
        try:
            _git("reset", "-q", "--hard", "HEAD")
            _git("clean", "-qfd")
        finally:
            rebuild_index()
        raise


def _check_base(fm: dict[str, Any], base_updated: str | None, body: str) -> None:
    if base_updated != fm.get("updated"):
        raise Conflict(
            "base_updated does not match the page's current updated; re-read and retry",
            _page_dict(fm, body),
        )


def _check_author(updated_by: str) -> str:
    if not AUTHOR_RE.fullmatch(updated_by or ""):
        raise InvalidInput("updated_by must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
    return updated_by


def _save(slug: str, fm: dict[str, Any], body: str, old_path: Path | None, action: str) -> dict[str, Any]:
    text = _serialize(fm, body)
    if len(text.encode("utf-8")) > MAX_PAGE_BYTES:
        raise TooLarge(f"page exceeds {MAX_PAGE_BYTES} bytes")
    new_path = _inside_wiki(WIKI_DIR / fm["category"] / f"{slug}.md")

    def apply() -> None:
        if old_path is not None and old_path.resolve() != new_path:
            old_path.unlink()
        new_path.write_text(text, encoding="utf-8")
        _append_log(action, slug, fm["updated_by"], fm["summary"])
        with closing(_db()) as conn, conn:
            _index(conn, fm, body.strip() + "\n")
            _write_index_md(conn)

    commit = _transaction(
        apply, f"[{fm['updated_by']}] {action} {slug}: {_one_line(fm['summary'], 200)}", fm["updated_by"]
    )
    return {"action": action, "commit": commit, "page": get_page(slug)}


def put_page(
    slug: str,
    *,
    title: str,
    category: str,
    tags: list[str],
    sources: list[dict[str, Any]],
    confidence: str,
    summary: str,
    body: str,
    updated_by: str,
    base_updated: str | None,
) -> dict[str, Any]:
    check_slug(slug)
    _check_author(updated_by)
    if category not in CATEGORIES:
        raise InvalidInput(f"category must be one of {', '.join(CATEGORIES)}")
    if confidence not in CONFIDENCE:
        raise InvalidInput("confidence must be stated or inferred")
    title = _one_line(title, 200)
    if not title:
        raise InvalidInput("title is required")

    with _lock:
        ensure_ready()
        old_path = _find(slug)
        old_fm: dict[str, Any] | None = None
        if old_path is not None:
            old_fm, old_body = _read(old_path)
            _check_base(old_fm, base_updated, old_body)
        fm = {
            "title": title,
            "slug": slug,
            "category": category,
            "tags": _norm_tags(tags),
            "sources": _norm_sources(sources),
            "created": (old_fm or {}).get("created") or now_iso(),
            "updated": _next_updated((old_fm or {}).get("updated")),
            "updated_by": updated_by,
            "confidence": confidence,
            "summary": _one_line(summary, 300),
        }
        return _save(slug, fm, body or "", old_path, "update" if old_path else "create")


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    want = heading.strip().lstrip("#").strip().casefold()
    in_fence = False
    start = level = None
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        if start is not None and len(m.group(1)) <= level:
            return start, i
        if start is None and m.group(2).casefold() == want:
            start, level = i, len(m.group(1))
    return (start, len(lines)) if start is not None else None


def _apply_section(body: str, heading: str, content: str, replace: bool) -> str:
    heading = heading.strip()
    if not heading or "\n" in heading or len(heading) > 200:
        raise InvalidInput("heading must be one line, 1-200 characters")
    content = content.strip("\n")
    lines = body.rstrip("\n").split("\n") if body.strip() else []
    bounds = _section_bounds(lines, heading)
    if bounds is None:
        new = "\n".join(lines).rstrip()
        return (new + "\n\n" if new else "") + f"## {heading.lstrip('#').strip()}\n\n{content}\n"
    start, end = bounds
    if replace:
        section = ["", *content.split("\n"), ""]
    else:
        section = lines[start + 1 : end]
        while section and not section[-1].strip():
            section.pop()
        section += ["", *content.split("\n"), ""]
    if end == len(lines):
        while section and not section[-1].strip():
            section.pop()
    return "\n".join(lines[: start + 1] + section + lines[end:]).rstrip("\n") + "\n"


def patch_page(
    slug: str,
    *,
    ops: list[dict[str, Any]],
    updated_by: str,
    base_updated: str | None,
    summary: str | None = None,
) -> dict[str, Any]:
    check_slug(slug)
    _check_author(updated_by)
    if not ops:
        raise InvalidInput("ops must not be empty")

    with _lock:
        ensure_ready()
        path = _find(slug)
        if path is None:
            raise NotFound(f"no page {slug!r}")
        fm, body = _read(path)
        _check_base(fm, base_updated, body)
        for op in ops:
            kind = op.get("op")
            if kind in ("append_section", "replace_section"):
                body = _apply_section(body, op.get("heading", ""), op.get("content", ""), kind == "replace_section")
            elif kind == "add_tags":
                fm["tags"] = _norm_tags([*(fm.get("tags") or []), *op.get("tags", [])])
            elif kind == "add_sources":
                fm["sources"] = _norm_sources([*(fm.get("sources") or []), *op.get("sources", [])])
            else:
                raise InvalidInput(f"unknown op {str(kind)[:40]!r}")
        fm["updated"] = _next_updated(fm.get("updated"))
        fm["updated_by"] = updated_by
        if summary is not None:
            fm["summary"] = _one_line(summary, 300)
        fm["slug"], fm["category"] = slug, path.parent.name
        return _save(slug, fm, body, path, "patch")


def archive_page(slug: str, *, updated_by: str) -> dict[str, Any]:
    check_slug(slug)
    _check_author(updated_by)
    with _lock:
        ensure_ready()
        path = _find(slug)
        if path is None:
            raise NotFound(f"no page {slug!r}")
        fm, _ = _read(path)
        dest = WIKI_DIR / ARCHIVE / f"{slug}.md"
        if dest.exists():
            dest = WIKI_DIR / ARCHIVE / f"{slug}.{now_iso().replace(':', '').replace('.', '')}.md"
        dest = _inside_wiki(dest)
        summary = fm.get("summary") or fm.get("title") or slug

        def apply() -> None:
            os.replace(path, dest)
            _append_log("archive", slug, updated_by, summary)
            with closing(_db()) as conn, conn:
                _unindex(conn, slug)
                _write_index_md(conn)

        commit = _transaction(apply, f"[{updated_by}] archive {slug}: {_one_line(summary, 200)}", updated_by)
        return {"action": "archive", "commit": commit, "archived_to": str(dest.relative_to(WIKI_DIR.resolve()))}
