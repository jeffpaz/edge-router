"""Personal-documents intent: questions about the user's own records (W-2s, tax returns,
statements, policies, leases, pay, payments ...), their PazBrain notes, or explicit
requests to search their documents or notes.

A match forces the request private and routes it to the mini's document index
(MINI_RAG_URL) — never to a cloud provider, whatever the checkbox says. (With the
checkbox on, every question consults the index first anyway; this detector is what
keeps document questions off the cloud when the checkbox is off.)

It matches:
  - an explicit request: "in my documents", "search my files", "in my notes"
  - PazBrain by name: "what does PazBrain say about the homelab?"
  - "my/our" + a document or personal-finance noun: "my 2023 W-2", "our mortgage rate",
    "my car insurance deductible", "my salary"
  - first-person money or records verbs: "what did Acme pay me", "how much did I pay in
    property tax", "did we sign the lease"
General questions about a form ("what is a W-2?") and how-to questions ("how do I file
my taxes?") keep normal routing.
"""
import re

_DOC_NOUNS = (
    # documents
    r"w-?2s?|1099(?:-[a-z]+)?s?|1040s?|tax returns?|tax documents?|tax forms?|"
    r"pay ?stubs?|paychecks?|earnings statements?|"
    r"closing (?:statements?|disclosures?|documents?|costs)|settlement statements?|"
    r"mortgage(?: statements?| rate| balance| payments?)?|escrow|deed|"
    r"bank statements?|brokerage statements?|account statements?|statements?|"
    r"insurance(?: polic(?:y|ies)| cards?)?|polic(?:y|ies)|declarations? page|"
    r"leases?|rental agreements?|contracts?|agreements?|receipts?|invoices?|bills?|"
    r"loan(?: documents?| agreements?)?|warranty|warranties|"
    # personal finance and records
    r"salary|wages|income|earnings|bonus(?:es)?|refunds?|withholding|deductions?|"
    r"taxes|property tax(?:es)?|tax bill|employer|landlord|lender|"
    r"bank accounts?|brokerage accounts?|account (?:balance|number)|balance|"
    r"premiums?|deductibles?|copays?|claims?|rent|utilities|utility bills?|"
    r"401\(?k\)?|ira|hsa|pension|dividends?|interest paid"
)

# "my 2023 W-2", "our homeowners insurance policy", "my car insurance deductible"
_POSSESSIVE = re.compile(rf"\b(?:my|our)\s+(?:[\w'-]+\s+){{0,3}}?(?:{_DOC_NOUNS})\b")
# "in my documents", "search my files", "look through our paperwork"
_EXPLICIT = re.compile(
    r"\b(?:in|from|search|check|look (?:in|through)|find (?:in|it in))\s+(?:my|our)\s+"
    r"(?:personal\s+)?(?:documents?|docs|files|paperwork|records|pdfs|notes|wiki|journal|pazbrain|logseq)\b"
)
# "what does PazBrain say", "my PazBrain notes on X", "in my Logseq"
_PAZBRAIN = re.compile(r"\bpaz ?brain\b|\b(?:my|our) (?:logseq|wiki|journal|notes?)\b")
# "what did Acme pay me", "how much did they refund us", "who withheld from me"
_PAID_ME = re.compile(
    r"\b(?:pay|paid|owe[sd]?|charged?|billed?|refund(?:ed)?|reimburs\w*|withh[eo]ld|"
    r"sen[dt]|deposit(?:ed)?)\s+(?:me|us)\b"
)
# "how much did I pay in property tax", "what did we earn in 2022", "when did I sign"
_I_DID = re.compile(
    r"\b(?:did|do|have|had)\s+(?:i|we)\s+(?:\w+\s+)?(?:pay|paid|earn(?:ed)?|make|made|"
    r"receive[d]?|get|got|owe[d]?|spend|spent|sign(?:ed)?|buy|bought|sell|sold|"
    r"contribute[d]?|deduct(?:ed)?|claim(?:ed)?|file[d]?|write|wrote|note[d]?|decide[d]?|plan(?:ned)?)\b"
)
_HOW_TO = re.compile(r"^\s*(?:how (?:do|can|should) (?:i|we)|how to|should (?:i|we)|what happens if)\b")


def is_personal_docs(query: str) -> bool:
    q = query.lower()
    if _EXPLICIT.search(q) or _PAZBRAIN.search(q):
        return True
    if _HOW_TO.search(q):
        return False
    return bool(_POSSESSIVE.search(q) or _PAID_ME.search(q) or _I_DID.search(q))
