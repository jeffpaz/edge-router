#!/usr/bin/env python3
"""PazBrain writer: send notes/emails/files to Grok with the PazBrain tools, run Grok's
tool calls against the PazBrain API, and print Grok's changelog.

    pazbrain notes.txt mail1.eml            # process files
    cat notes.txt | pazbrain                # or stdin
    pazbrain --dry-run notes.txt            # reads run for real; writes are only printed

Grok never sees a token. It gets the tool definitions (/opt/pazbrain/grok_tools.json) and
the system prompt (/opt/pazbrain/grok_prompt.txt); this script attaches
PAZBRAIN_WRITE_TOKEN when it makes each HTTP call and returns only {status, body} to Grok.

Per tool call, one line goes to stderr; Grok's final changelog goes to stdout. A run
record (no page content, no tokens) is appended to /opt/pazbrain/logs/writer.log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from openai import OpenAI

ENV_FILE = "/etc/pazbrain.env"
PROMPT_FILE = Path("/opt/pazbrain/grok_prompt.txt")
TOOLS_FILE = Path("/opt/pazbrain/grok_tools.json")
LOG_FILE = Path("/opt/pazbrain/logs/writer.log")
XAI_BASE_URL = "https://api.x.ai/v1"  # same endpoint the edge router's openai client uses

MAX_TURNS = 25
HTTP_TIMEOUT_S = 30
MAX_TOOL_RESULT = 20 * 1024
MAX_INPUT_BYTES = 400 * 1024
WRITE_TOOLS = {"put_page", "patch_page", "archive_page"}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


def _redactor(*secrets: str):
    """Strip secret values out of any text before it is printed or logged."""
    secrets = tuple(s for s in secrets if s and len(s) >= 8)

    def redact(text: str) -> str:
        for s in secrets:
            text = text.replace(s, "[REDACTED]")
        return text

    return redact


def _drop_none(value):
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


class PazBrainAPI:
    """Maps each Grok tool to its PazBrain endpoint (see pazbrain/routes.py)."""

    def __init__(self, base: str, write_token: str, dry_run: bool):
        self.base = base.rstrip("/")
        self.dry_run = dry_run
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {write_token}"

    def _send(self, method: str, path: str, params=None, body=None) -> tuple[int, object]:
        resp = self.http.request(
            method, self.base + path, params=params, json=body, timeout=HTTP_TIMEOUT_S
        )
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, resp.text

    @staticmethod
    def _slug_path(args: dict) -> str:
        slug = str(args.get("slug", ""))
        if not SLUG_RE.fullmatch(slug):
            raise ValueError("slug must match ^[a-z0-9][a-z0-9-]{0,80}$")
        return "/pages/" + quote(slug, safe="")

    @staticmethod
    def _missing_sources(name: str, args: dict) -> str | None:
        """Every fact needs a source in the `sources` field. Models tend to write sources as
        body text instead, so refuse those writes and tell Grok where they go."""
        if name == "put_page" and not args.get("sources"):
            return ("put_page needs a non-empty `sources` argument: a list of "
                    "{type: email|file|chat, ref, date}. Put sources in that field, not in the body.")
        if name == "patch_page":
            ops = args.get("ops") or []
            adds_content = any(o.get("op") in ("append_section", "replace_section") for o in ops)
            has_sources = any(o.get("op") == "add_sources" and o.get("sources") for o in ops)
            if adds_content and not has_sources:
                return ("patch_page that adds content needs an add_sources op in the same call: "
                        "{op: add_sources, sources: [{type, ref, date}]}. Put sources there, not in the body.")
        return None

    def call(self, name: str, args: dict) -> tuple[int, object, bool]:
        """Returns (status, body, not_sent). not_sent is 'dry-run' or 'rejected' when no HTTP call was made."""
        args = _drop_none(args)
        if name in ("put_page", "patch_page", "archive_page"):
            args.setdefault("updated_by", "grok")
        problem = self._missing_sources(name, args)
        if problem:
            return 400, {"error": "sources_required", "detail": problem}, "rejected"

        if name == "search_pages":
            req = ("GET", "/search", {k: args[k] for k in ("q", "limit") if k in args}, None)
        elif name == "get_page":
            req = ("GET", self._slug_path(args), None, None)
        elif name == "list_pages":
            req = ("GET", "/pages", {k: args[k] for k in ("category", "tag", "limit") if k in args}, None)
        elif name == "recent_changes":
            req = ("GET", "/recent", {k: args[k] for k in ("limit",) if k in args}, None)
        elif name == "put_page":
            req = ("PUT", self._slug_path(args), None, {k: v for k, v in args.items() if k != "slug"})
        elif name == "patch_page":
            req = ("PATCH", self._slug_path(args), None, {k: v for k, v in args.items() if k != "slug"})
        elif name == "archive_page":
            req = ("DELETE", self._slug_path(args), {"updated_by": args["updated_by"]}, None)
        else:
            return 0, {"error": "unknown_tool", "detail": f"No tool named {name!r}."}, "rejected"

        method, path, params, body = req
        if self.dry_run and name in WRITE_TOOLS:
            print(f"\n--- DRY RUN: would send {method} {path}"
                  + (f"?{'&'.join(f'{k}={v}' for k, v in params.items())}" if params else ""))
            if body is not None:
                print(json.dumps(body, indent=2, ensure_ascii=False))
            return 200, {"dry_run": True, "detail": "Not written (dry run). Continue as if it succeeded."}, "dry-run"
        status, data = self._send(method, path, params, body)
        return status, data, None


def _read_inputs(paths: list[str]) -> tuple[str, list[str]]:
    if paths:
        parts, names = [], []
        for p in paths:
            path = Path(p)
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"[source file: {path.name}]\n{text.strip()}")
            names.append(path.name)
        content = "\n\n".join(parts)
    else:
        if sys.stdin.isatty():
            sys.exit("pazbrain: no input files and nothing on stdin (see pazbrain --help)")
        content, names = sys.stdin.read().strip(), ["<stdin>"]
    if not content:
        sys.exit("pazbrain: input is empty")
    if len(content.encode("utf-8")) > MAX_INPUT_BYTES:
        sys.exit(f"pazbrain: input exceeds {MAX_INPUT_BYTES // 1024} KB; split it into smaller batches")
    return content, names


def _append_log(record: dict) -> None:
    LOG_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="pazbrain",
        description="Send files (or stdin) to Grok, which updates the PazBrain wiki through its API.",
        epilog="Logs: /opt/pazbrain/logs/writer.log (run records only; no content or tokens).",
    )
    ap.add_argument("files", nargs="*", help="files to process; reads stdin when none are given")
    ap.add_argument("--dry-run", action="store_true",
                    help="run read tools for real, but only print PUT/PATCH/DELETE instead of sending them")
    args = ap.parse_args()

    load_dotenv(ENV_FILE)
    write_token = os.getenv("PAZBRAIN_WRITE_TOKEN", "")
    xai_key = os.getenv("XAI_API_KEY", "")
    model = os.getenv("XAI_MODEL", "grok-3-mini")
    api_base = os.getenv("PAZBRAIN_API", "http://127.0.0.1:8000/brain/api")
    missing = [n for n, v in (("PAZBRAIN_WRITE_TOKEN", write_token), ("XAI_API_KEY", xai_key)) if not v]
    if missing:
        sys.exit(f"pazbrain: {', '.join(missing)} not set in {ENV_FILE}")
    redact = _redactor(write_token, xai_key, os.getenv("PAZBRAIN_READ_TOKEN", ""))

    content, input_names = _read_inputs(args.files)
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    tools = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    api = PazBrainAPI(api_base, write_token, args.dry_run)
    grok = OpenAI(api_key=xai_key, base_url=XAI_BASE_URL, timeout=180, max_retries=2)

    today = datetime.now().strftime("%Y-%m-%d")
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Today is {today}. Process these items:\n\n{content}"},
    ]
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": input_names, "dry_run": args.dry_run, "model": model,
        "turns": 0, "tools": [], "result": "error",
    }
    changelog = ""
    exit_code = 1
    try:
        for turn in range(1, MAX_TURNS + 1):
            record["turns"] = turn
            resp = grok.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto"
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                changelog = (msg.content or "").strip()
                record["result"] = "ok"
                exit_code = 0
                break
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                slug = "-"
                not_sent = None
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                    if not isinstance(targs, dict):
                        raise ValueError("arguments must be a JSON object")
                    slug = str(targs.get("slug", "-"))
                    status, body, not_sent = api.call(name, targs)
                except (ValueError, KeyError) as exc:
                    status, body, not_sent = 0, {"error": "bad_arguments", "detail": redact(str(exc))}, "rejected"
                except requests.RequestException as exc:
                    status, body = 0, {"error": "http_error", "detail": redact(str(exc))}
                text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
                if len(text) > MAX_TOOL_RESULT:
                    text = text[:MAX_TOOL_RESULT] + " …[truncated]"
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"status": status, "body": text}, ensure_ascii=False),
                })
                tag = {"dry-run": " (dry-run, not sent)", "rejected": " (rejected by writer, not sent)"}.get(not_sent, "")
                print(f"[tool] {name} slug={slug} -> {status}{tag}", file=sys.stderr)
                record["tools"].append({"tool": name, "slug": slug, "status": status, "not_sent": not_sent})
        else:
            record["result"] = "max_turns"
            print(f"pazbrain: stopped after {MAX_TURNS} turns without a final changelog", file=sys.stderr)
    except Exception as exc:  # report, log, never leak secrets
        record["result"] = "error"
        record["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
        print(f"pazbrain: {record['error']}", file=sys.stderr)
    finally:
        _append_log(record)

    if changelog:
        print(changelog)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
