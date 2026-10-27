"""Live-data intent detector.

A question flagged here goes to Grok with server-side web search, a costly call
(~12k input tokens of search results), so it needs real evidence, not one loose word:

  - a strong phrase on its own ("latest news", "who won", "bitcoin price",
    "weather in ...", "NBA standings"), or
  - two signals of different kinds: a time word plus a live topic, or a team /
    league plus a time word or a sports / live topic.

Every term matches on word boundaries, so "win" never fires inside "window" and
"live" never fires inside "deliver".
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class IntentResult:
    is_realtime: bool
    confidence: float        # 0.0 - 1.0
    signals: list[str]       # which signals triggered
    preferred_provider: Optional[str]  # "grok" or None


# Phrases that need current data on their own.
STRONG_PATTERNS: list[tuple[str, str]] = [
    ("breaking_news",   r"\bbreaking news\b"),
    ("latest_news",     r"\b(?:latest|recent|today'?s|current|top) (?:news|headlines)\b"),
    ("news_now",        r"\bnews (?:today|tonight|this (?:morning|week)|right now)\b"),
    ("headlines",       r"\bheadlines\b"),
    ("who_won",         r"\bwho (?:won|is winning|leads|scored)\b"),
    ("score_question",  r"\bwhat(?:'s| is| was) the (?:final )?score\b"),
    ("live_score",      r"\b(?:live|final|box) scores?\b"),
    ("results_list",    r"\bwhat (?:are|were) the \w+(?: \w+)? (?:results|scores|standings)\b"),
    ("did_team",        r"\bdid (?:the )?\w+(?: \w+)? (?:win|lose|beat)\b"),
    ("asset_price",     r"\b(?:stock|share|bitcoin|btc|ethereum|eth|crypto|gold|oil|gas) prices?\b"),
    ("price_of_asset",  r"\b(?:price|value) of (?:bitcoin|btc|ethereum|eth|gold|oil|gas)\b"),
    ("ticker",          r"(?<![\w$])\$[a-z]{1,5}\b"),
    ("market_now",      r"\b(?:stock market|nasdaq|dow jones|the dow|s&p 500) (?:today|now|right now|this week)\b"),
    ("weather",         r"\b(?:what'?s|what is|how'?s|how is) the weather\b"
                        r"|\bweather (?:in|for|today|tomorrow|tonight|this)\b"
                        r"|\bforecast (?:for|in|today|tomorrow)\b"),
    ("election_now",    r"\belection results?\b|\bwho is leading the\b|\bpolls? (?:show|today)\b"),
]

# Time words: on their own they don't mean live data ("the latest version of Python").
TEMPORAL_SIGNALS = [
    "today", "tonight", "yesterday", "last night", "this morning", "right now",
    "currently", "at the moment", "just happened", "this season", "so far this season",
    "this week", "latest", "as of now",
]

# Topics that are live when paired with a time word.
LIVE_TOPICS = [
    "score", "scores", "standings", "playoffs", "playoff", "injured", "injury",
    "traded", "stock", "stocks", "stock market", "crypto", "bitcoin", "earnings",
    "ipo", "nasdaq", "dow jones", "s&p 500", "weather", "forecast", "news",
    "election", "polls",
]

# Sports words that only count next to a team or league ("write a story about a game").
SPORTS_TOPICS = ["game", "games", "match", "won", "win", "lost", "lose", "beat", "roster", "draft"]

SPORTS_ENTITIES = [
    "nba", "nfl", "nhl", "mlb", "mls", "ufc", "wnba", "ncaa", "premier league",
    "knicks", "lakers", "celtics", "warriors", "heat", "suns",
    "yankees", "dodgers", "cubs", "diamondbacks",
    "chiefs", "patriots", "packers", "cowboys", "cardinals",
]


def _found(terms: list[str], text: str) -> list[str]:
    """Terms present as whole words (apostrophes count as a boundary: "won't" ≠ "won")."""
    return [t for t in terms if re.search(rf"(?<![\w&']){re.escape(t)}(?![\w&'])", text)]


def classify(query: str) -> IntentResult:
    q = query.lower().strip()

    strong = [f"strong:{name}" for name, pattern in STRONG_PATTERNS if re.search(pattern, q)]
    temporal = [f"temporal:{t}" for t in _found(TEMPORAL_SIGNALS, q)]
    topics = [f"topic:{t}" for t in _found(LIVE_TOPICS, q)]
    sports = [f"sports:{t}" for t in _found(SPORTS_ENTITIES, q)]
    sports_topics = [f"sports_topic:{t}" for t in _found(SPORTS_TOPICS, q)] if sports else []

    is_realtime = bool(
        strong
        or (temporal and topics)
        or (sports and (temporal or topics or sports_topics))
    )
    signals = strong + temporal + topics + sports + sports_topics if is_realtime else []
    confidence = min(1.0, 0.5 + 0.25 * len(signals)) if is_realtime else 0.0

    return IntentResult(
        is_realtime=is_realtime,
        confidence=confidence,
        signals=signals,
        # Grok is the only configured provider with live search (web + X).
        preferred_provider="grok" if is_realtime else None,
    )
