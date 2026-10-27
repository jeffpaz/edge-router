"""Personal-documents intent: questions about the user's own records (W-2s, tax returns,
statements, policies, leases ...) or explicit requests to search their documents.

A match forces the request private and routes it to the mini's document index
(MINI_RAG_URL) — never to a cloud provider, whatever the checkbox says.

It needs either an explicit "in my documents / files" request, or a possessive
("my", "our") in front of a document noun. General questions about a form ("what is a
W-2?") and how-to questions ("how do I file my taxes?") stay with normal routing.
"""
import re

_DOC_NOUNS = (
    r"w-?2s?|1099(?:-[a-z]+)?s?|1040s?|tax returns?|tax documents?|tax forms?|"
    r"pay ?stubs?|paychecks?|earnings statements?|"
    r"closing (?:statements?|disclosures?|documents?|costs)|settlement statements?|"
    r"mortgage(?: statements?| rate| balance| payments?)?|escrow|deed|"
    r"bank statements?|brokerage statements?|account statements?|statements?|"
    r"insurance(?: polic(?:y|ies)| cards?)?|polic(?:y|ies)|declarations? page|"
    r"leases?|rental agreements?|contracts?|agreements?|receipts?|invoices?|bills?|"
    r"loan(?: documents?| agreements?)?|warranty|warranties"
)

# "my 2023 W-2", "our homeowners insurance policy", "my last pay stub"
_POSSESSIVE = re.compile(rf"\b(?:my|our)\s+(?:[\w'-]+\s+){{0,3}}?(?:{_DOC_NOUNS})\b")
# "in my documents", "search my files", "look through our paperwork"
_EXPLICIT = re.compile(
    r"\b(?:in|from|search|check|look (?:in|through)|find (?:in|it in))\s+(?:my|our)\s+"
    r"(?:personal\s+)?(?:documents?|docs|files|paperwork|records|pdfs)\b"
)
_HOW_TO = re.compile(r"^\s*(?:how (?:do|can|should) (?:i|we)|how to|should (?:i|we)|what happens if)\b")


def is_personal_docs(query: str) -> bool:
    q = query.lower()
    if _EXPLICIT.search(q):
        return True
    if _HOW_TO.search(q):
        return False
    return bool(_POSSESSIVE.search(q))
