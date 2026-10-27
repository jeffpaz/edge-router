"""Live-data detector: real live questions flag, everyday questions don't (each flag is
a costly Grok web-search call)."""
import pytest

from realtime_classifier import classify

LIVE = [
    "What's the latest news about the election today?",
    "Any breaking news?",
    "Top headlines this morning",
    "Who won the Lakers game last night?",
    "Did the Chiefs win yesterday?",
    "What's the score of the Suns game?",
    "What are the NBA standings?",
    "How are the Yankees doing this season?",
    "Lakers score",
    "What is the bitcoin price right now?",
    "Price of gold today",
    "How is $aapl doing?",
    "Is the Nasdaq up today?",
    "What's the weather in Phoenix tomorrow?",
    "Weather forecast for Sedona",
    "Stock market today",
    "Election results",
]

NOT_LIVE = [
    "What is a haiku?",
    "Why won't my window close?",
    "Can you deliver a live demo for my team?",
    "What is the price of a Big Mac?",
    "Write a story about a soccer game",
    "What game should I play today?",
    "What's the latest version of Python?",
    "Explain how the stock market works",
    "I want to go to the farmers market today",
    "Is it still worth learning Java?",
    "How did you do that?",
    "Has the meeting been moved?",
    "What does it cost to build a deck?",
    "Help me win an argument with my brother",
    "Summarize this news article for me",
    "What causes weather patterns?",
    "Who is the current president of France?",
]


@pytest.mark.parametrize("query", LIVE)
def test_live_questions_flag(query):
    result = classify(query)
    assert result.is_realtime, query
    assert result.preferred_provider == "grok"
    assert result.signals


@pytest.mark.parametrize("query", NOT_LIVE)
def test_everyday_questions_do_not_flag(query):
    result = classify(query)
    assert not result.is_realtime, (query, result.signals)
    assert result.preferred_provider is None and result.signals == []
