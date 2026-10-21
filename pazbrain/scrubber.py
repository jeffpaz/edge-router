"""Secret scrubber: refuse wiki writes that contain credentials or financial identifiers.

scan() returns the names of the rules that matched — never the matched text, so a
rejection can be logged and returned to the caller without re-leaking the secret.
"""

from __future__ import annotations

import re
from typing import Any, Callable


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _aba_ok(digits: str) -> bool:
    w = (3, 7, 1) * 3
    return sum(int(c) * k for c, k in zip(digits, w)) % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_BARE9_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b")


def _cards(text: str) -> bool:
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and digits[0] in "3456" and len(set(digits)) > 1 and _luhn_ok(digits):
            return True
    return False


def _routing(text: str) -> bool:
    # A 9-digit number passing the ABA checksum, near a banking keyword.
    for m in _BARE9_RE.finditer(text):
        window = text[max(0, m.start() - 40) : m.start()].lower()
        if _aba_ok(m.group()) and re.search(r"routing|\baba\b|\brtn\b|transit|wire|ach", window):
            return True
    return False


def _iban(text: str) -> bool:
    for m in _IBAN_RE.finditer(text):
        s = m.group().replace(" ", "")
        if 15 <= len(s) <= 34 and _iban_ok(s):
            return True
    return False


_RULES: list[tuple[str, Callable[[str], bool]]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----").search),
    ("ssn", re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)").search),
    ("ssn", re.compile(r"(?i)\b(?:ssn|social security(?: number| no\.?)?)\b\W{0,6}\d{3}\W?\d{2}\W?\d{4}\b").search),
    ("card_number", _cards),
    ("bank_routing_number", _routing),
    ("bank_routing_number", re.compile(r"(?i)\b(?:routing|aba)\s*(?:number|no\.?|#)?\s*[:=#]?\s*\d{9}\b").search),
    ("bank_account_number", re.compile(
        r"(?i)\b(?:bank\s+)?(?:account|acct)\s*(?:number|no\.?|num|#)\s*[:=#]?\s*[\dX*-]{6,20}\b").search),
    ("bank_account_number", _iban),
    ("password", re.compile(
        r"(?i)\b(?:password|passwd|pwd|passphrase|passcode)\b\s*(?:[:=]|\bis\b|\bwas\b)\s*\S{3,}").search),
    ("api_key", re.compile(
        r"(?<![A-Za-z0-9])(?:"
        r"sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_-]{16,}"
        r"|xai-[A-Za-z0-9]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
        r"|AIza[0-9A-Za-z_-]{35}"
        r"|xox[abposr]-[A-Za-z0-9-]{10,}"
        r"|glpat-[A-Za-z0-9_-]{20}"
        r"|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"
        r"|pb[wr]_[A-Za-z0-9_-]{40,}"
        r")").search),
    ("api_key", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}").search),
    ("api_key", re.compile(
        r"(?i)\b(?:api[_ -]?key|secret(?:[_ -]?key)?|access[_ -]?token|auth[_ -]?token|bearer)\b"
        r"\s*[:=]?\s*['\"]?[A-Za-z0-9_\-./+=]{20,}").search),
]


def _strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")


def scan(payload: Any) -> list[dict[str, str]]:
    """Every (rule, field) that matched anywhere in payload; empty list when clean."""
    hits: list[dict[str, str]] = []
    for field, text in _strings(payload):
        for name, test in _RULES:
            if test(text) and not any(h["rule"] == name and h["field"] == field for h in hits):
                hits.append({"rule": name, "field": field})
    return hits
