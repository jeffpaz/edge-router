"""tripBuddy sub-classifier: transport words route to directions only when asked as a question."""
import pytest

from trip_routes import classify_trip


@pytest.mark.parametrize("text,expected", [
    # Plain mentions are chat → plan_edit (local model).
    ("I bought walking specific shoes. I've done my part. Your guys job is to get me up and out.", "plan_edit"),
    ("I'll drive us Saturday", "plan_edit"),
    ("We took an uber last night lol", "plan_edit"),
    # Real route questions still go to directions.
    ("Can we walk to the Alamo from the house?", "directions"),
    ("Should we uber or drive downtown?", "directions"),
    ("is the river walk walkable from the Pearl?", "directions"),
    ("How far is the Pearl from the Airbnb", "directions"),
    ("directions to Tucker's", "directions"),
    ("is there parking at the market", "directions"),
    # Other routes unchanged.
    ("any good tacos near the house?", "place"),
    ("add a dive bar Friday night", "plan_edit"),
])
def test_classify_trip(text, expected):
    assert classify_trip(text)[0] == expected
