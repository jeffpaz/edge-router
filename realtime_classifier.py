import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class IntentResult:
    is_realtime: bool
    confidence: float        # 0.0 - 1.0
    signals: list[str]       # which signals triggered
    preferred_provider: Optional[str]  # "grok" or "openai" or None

# Temporal keywords that imply current/live data needed
TEMPORAL_SIGNALS = [
    "last night", "last week", "yesterday", "today", "tonight",
    "this morning", "right now", "currently", "at the moment",
    "latest", "recent", "just happened", "breaking", "live",
    "this year", "this season", "this week", "so far this",
]

# Topic keywords that almost always need real-time data
REALTIME_TOPICS = [
    # Sports
    "score", "scores", "won", "win", "lost", "lose", "game",
    "match", "standings", "playoffs", "championship", "draft",
    "traded", "injured", "roster",
    # Finance
    "stock", "price", "market", "crypto", "bitcoin", "etf",
    "earnings", "ipo", "nasdaq", "dow", "s&p",
    # News / Weather
    "weather", "forecast", "news", "headlines", "election",
    "poll", "vote", "passed", "signed", "announced",
    # General currency
    "who is the current", "who is now", "what is the current",
    "how much does", "what does it cost",
]

# Named sports leagues/teams (high signal for score queries)
SPORTS_ENTITIES = [
    "nba", "nfl", "nhl", "mlb", "nhl", "mls", "ufc",
    "knicks", "lakers", "celtics", "warriors", "heat",
    "yankees", "dodgers", "cubs", "chiefs", "patriots",
    "packers", "cowboys",
]

def classify(query: str) -> IntentResult:
    q = query.lower().strip()
    signals = []

    # Check temporal signals
    for signal in TEMPORAL_SIGNALS:
        if signal in q:
            signals.append(f"temporal:{signal}")

    # Check realtime topics
    for topic in REALTIME_TOPICS:
        if topic in q:
            signals.append(f"topic:{topic}")

    # Check sports entities
    for entity in SPORTS_ENTITIES:
        if entity in q:
            signals.append(f"sports:{entity}")

    # Check for question patterns implying current state
    current_patterns = [
        r"\bwho (won|is winning|leads|scored)\b",
        r"\bwhat (is the score|happened|is going on)\b",
        r"\bdid .+ (win|lose|beat|score)\b",
        r"\bhow did .+ (do|play|perform)\b",
        r"\bwhat (are|were) the .+(results|scores|standings)\b",
        r"\bis .+ still\b",
        r"\bhas .+ (been|gone|happened)\b",
    ]
    for pattern in current_patterns:
        if re.search(pattern, q):
            signals.append(f"pattern:{pattern}")

    is_realtime = len(signals) > 0
    confidence = min(1.0, len(signals) * 0.25 + (0.5 if is_realtime else 0))

    # Prefer Grok for sports/news (has live X data), OpenAI as fallback
    preferred = None
    if is_realtime:
        has_sports = any("sports:" in s or "topic:score" in s
                         or "topic:won" in s or "topic:game" in s
                         for s in signals)
        preferred = "grok" if has_sports else "openai"

    return IntentResult(
        is_realtime=is_realtime,
        confidence=confidence,
        signals=signals,
        preferred_provider=preferred
    )
