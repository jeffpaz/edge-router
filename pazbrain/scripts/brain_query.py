#!/usr/bin/env python3
"""PazBrain query sidecar: read-only Logseq CLI behind a unix socket, for edge-router /brain/query.

The edge-router container has no Node or logseq CLI, and a second CLI there would start its own
db-worker-node on the same graph, so the router forwards validated requests here instead, over
/opt/pazbrain/run/brain-query.sock (this host's ~/pazbrain/run, bind-mounted into the container).

Only the operations in OPS are reachable. Each re-validates its inputs and builds a fixed CLI
argument list for the "pazbrain" graph: no shell, values passed as --opt=value so they can never be
read as flags. No upsert/remove/move/import or raw query is ever constructed, so nothing reachable
from HTTP can write. Results are capped (items, text length, tree size, response bytes).

The CLI reuses one db-worker-node daemon across requests; this process never stops it.

    brain_query.py                      # serve (pazbrain-query.service)
    brain_query.py --check              # run every op once against the graph and exit
    brain_query.py --socket /tmp/x.sock # serve on another socket (testing)

Request bodies and results are never logged; only op, status and timing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

SOCKET = Path("/opt/pazbrain/run/brain-query.sock")
LOGSEQ = str(Path.home() / ".local" / "bin" / "logseq")
GRAPH = "pazbrain"
CLI_TIMEOUT_MS = 10_000
PROC_TIMEOUT_S = 15
MAX_BODY_BYTES = 4096
MAX_RESPONSE_BYTES = 256 * 1024
MAX_ITEMS = 50
MAX_TEXT = 1000
MAX_TREE_NODES = 300

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")  # same rule as pazbrain/store.py
TEXT_RE = re.compile(r"^[^-\x00-\x1f\x7f][^\x00-\x1f\x7f]{0,99}$")  # 1-100 chars, no control chars, no leading '-'
STATUSES = ("backlog", "canceled", "doing", "done", "in-review", "todo")
ENV = dict(os.environ, NODE_NO_WARNINGS="1")


class BadRequest(ValueError):
    pass


class CliError(RuntimeError):
    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _text(p: dict, key: str, required: bool = True) -> str:
    val = p.get(key, "")
    if not isinstance(val, str):
        raise BadRequest(f"{key} must be a string")
    val = val.strip()
    if not val and not required:
        return ""
    if not TEXT_RE.fullmatch(val):
        raise BadRequest(f"{key}: 1-100 characters, no control characters, must not start with '-'")
    return val


def _int(p: dict, key: str, lo: int, hi: int, default: int) -> int:
    val = p.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int) or not lo <= val <= hi:
        raise BadRequest(f"{key} must be an integer {lo}-{hi}")
    return val


def _status(p: dict, required: bool) -> str | None:
    val = p.get("status")
    if val is None and not required:
        return None
    if val not in STATUSES:
        raise BadRequest("status must be one of " + ", ".join(STATUSES))
    return val


def _edn_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # control chars are rejected, so only \" and \\ escapes: valid EDN


# ---------------------------------------------------------------------------
# Allow-listed operations: (validated params) -> (CLI args, result shape)
# ---------------------------------------------------------------------------


def op_search_page(p):
    return ["search", "page", f"--content={_text(p, 'q')}"], "items"


def op_search_block(p):
    return ["search", "block", f"--content={_text(p, 'q')}"], "items"


def op_show_page(p):
    slug = p.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise BadRequest("slug must match ^[a-z0-9][a-z0-9-]{0,80}$")
    # level 1 = the page alone, 2 = its top-level blocks, 3 = one more level of nesting
    return ["show", f"--page={slug}", f"--level={_int(p, 'level', 1, 3, 2)}"], "tree"


def op_list_task(p):
    args = ["list", "task", f"--limit={_int(p, 'limit', 1, MAX_ITEMS, 20)}"]
    status = _status(p, required=False)
    return args + ([f"--status={status}"] if status else []), "items"


def op_recent_updated(p):
    return ["query", "--name=recent-updated", f"--inputs=[{_int(p, 'days', 1, 90, 7)}]"], "result"


def op_task_search(p):
    status, title, days = _status(p, required=True), _text(p, "title", required=False), _int(p, "days", 0, 365, 0)
    return ["query", "--name=task-search", f"--inputs=[{_edn_str(status)} {_edn_str(title)} {days}]"], "result"


OPS = {
    "search_page": op_search_page,
    "search_block": op_search_block,
    "show_page": op_show_page,
    "list_task": op_list_task,
    "recent_updated": op_recent_updated,
    "task_search": op_task_search,
}


# ---------------------------------------------------------------------------
# CLI + result shaping
# ---------------------------------------------------------------------------


def run_cli(args: list[str]) -> dict:
    argv = [LOGSEQ, *args, f"--graph={GRAPH}", "--output=json", f"--timeout-ms={CLI_TIMEOUT_MS}"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, env=ENV, timeout=PROC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise CliError(504, "timeout", f"query took longer than {PROC_TIMEOUT_S}s") from None
    try:
        resp = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise CliError(502, "cli_error", f"logseq exited {proc.returncode} without JSON") from None
    if resp.get("status") != "ok":
        err = resp.get("error") or {}
        code = str(err.get("code", "cli_error"))
        status = 404 if code.endswith("not-found") else 400 if code.startswith("invalid") else 502
        raise CliError(status, code, str(err.get("message", ""))[:300])
    return resp.get("data") or {}


def _clip(v):
    return v[:MAX_TEXT] + "…" if isinstance(v, str) and len(v) > MAX_TEXT else v


def _flat(item: dict) -> dict:
    """Scalar fields only (nested entity refs are internal ids), long strings clipped."""
    return {k: _clip(v) for k, v in item.items() if isinstance(v, (str, int, float, bool)) and v is not None}


def shape(kind: str, data: dict) -> dict:
    if kind == "tree":
        budget = [MAX_TREE_NODES]

        def node(b: dict) -> dict:
            budget[0] -= 1
            out = {"id": b.get("db/id"), "title": _clip(b.get("block/title", ""))}
            kids = []
            for c in b.get("block/children", []):
                if budget[0] <= 0:
                    break
                kids.append(node(c))
            if kids:
                out["children"] = kids
            return out

        root = data.get("root") or {}
        tree = node(root)
        return {"page": root.get("block/name"), "tree": tree, "truncated": budget[0] <= 0,
                "linked_references": (data.get("linked-references") or {}).get("count", 0)}
    rows = data.get(kind) or []
    return {"count": min(len(rows), MAX_ITEMS), "truncated": len(rows) > MAX_ITEMS,
            "items": [_flat(r) for r in rows[:MAX_ITEMS] if isinstance(r, dict)]}


def handle(payload: dict) -> dict:
    op = payload.get("op")
    if op not in OPS:
        raise BadRequest("op must be one of " + ", ".join(OPS))
    args, kind = OPS[op]({k: v for k, v in payload.items() if k != "op"})
    return {"op": op, **shape(kind, run_cli(args))}


# ---------------------------------------------------------------------------
# HTTP over a unix socket
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "brain-query/1"

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        if len(raw) > MAX_RESPONSE_BYTES:
            status, raw = 502, json.dumps({"error": "too_large", "detail": "result exceeds size cap"}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not_found"})

    def do_POST(self):
        start = time.monotonic()
        if self.path != "/query":
            return self._send(404, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise BadRequest(f"body must be 1-{MAX_BODY_BYTES} bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise BadRequest("body must be a JSON object")
            status, body = 200, handle(payload)
        except (BadRequest, json.JSONDecodeError, ValueError) as exc:
            status, body = 400, {"error": "bad_request", "detail": str(exc)[:300]}
        except CliError as exc:
            status, body = exc.status, {"error": exc.code, "detail": exc.detail}
        except Exception as exc:  # never leak a traceback to the router
            status, body = 500, {"error": "internal", "detail": type(exc).__name__}
        self._send(status, body)
        op = body.get("op") or "-"
        print(f"brain_query op={op} status={status} ms={int((time.monotonic() - start) * 1000)}", file=sys.stderr, flush=True)

    def log_message(self, *args):  # request lines can carry nothing sensitive, but keep the journal quiet
        pass


class UnixHTTPServer(socketserver.UnixStreamServer):
    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)  # BaseHTTPRequestHandler expects a (host, port) client address


def serve(sock: Path) -> None:
    sock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if sock.exists() or sock.is_socket():
        sock.unlink()
    old = os.umask(0o177)
    try:
        server = UnixHTTPServer(str(sock), Handler)
    finally:
        os.umask(old)
    os.chmod(sock, 0o600)
    try:
        run_cli(["graph", "info"])  # start (or reuse) the daemon now rather than on the first request
    except CliError as exc:
        print(f"brain_query: warm-up failed ({exc.code}); continuing", file=sys.stderr)
    print(f"brain_query: serving on {sock}", file=sys.stderr, flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # systemd stop: unwind so the socket is removed
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sock.unlink(missing_ok=True)


def check() -> int:
    samples = [
        {"op": "search_page", "q": "a"},
        {"op": "search_block", "q": "the"},
        {"op": "show_page", "level": 2},  # slug: the first page search_page found
        {"op": "list_task", "limit": 5},
        {"op": "recent_updated", "days": 7},
        {"op": "task_search", "status": "todo"},
    ]
    failed = 0
    first_page = None
    for s in samples:
        if s["op"] == "show_page":
            if not first_page:
                print(f"  skip {s['op']:15} (search_page found no page to show)")
                continue
            s = {**s, "slug": first_page}
        try:
            r = handle(s)
            if s["op"] == "search_page":
                names = (i.get("block/title") for i in r.get("items", []))
                first_page = next((n for n in names if isinstance(n, str) and SLUG_RE.fullmatch(n)), None)
            print(f"  ok   {s['op']:15} count={r.get('count', len(r.get('tree', {}).get('children', [])))} truncated={r.get('truncated')}")
        except (BadRequest, CliError) as exc:
            failed += 1
            print(f"  FAIL {s['op']:15} {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="brain_query.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="run every op once and exit")
    ap.add_argument("--socket", type=Path, default=SOCKET)
    a = ap.parse_args()
    sys.exit(check() if a.check else serve(a.socket))
