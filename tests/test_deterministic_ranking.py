"""Deterministic Cheapest / Comfortable / Fastest ranking tests."""

from __future__ import annotations

from app.ml.deterministic_ranker import rank_all, rank_cheapest, rank_comfortable, rank_fastest
from app.services.route_feasibility import validate_candidate


def _c(**kwargs):
    base = {
        "legs": [
            {
                "mode": kwargs.get("modes", ["bus"])[0],
                "from_location": "A",
                "to_location": "B",
                "from_coordinates": (12.82, 80.04),
                "to_coordinates": (13.05, 80.28),
                "estimated_duration_minutes": kwargs.get("estimated_duration_minutes", 60),
                "estimated_cost_inr": kwargs.get("estimated_cost_inr", 40),
                "distance_meters": 20000,
                "data_source": "test",
            }
        ],
        "walking_meters": 0,
        "waiting_minutes": 5,
        "transfers": 0,
        "modes": ["bus"],
        "preference_scores": {"cheapest": 0.5, "comfortable": 0.5, "fastest": 0.5},
    }
    base.update(kwargs)
    return base


def test_cheapest_is_lowest_fare():
    cheap = _c(name="cheap-bus", estimated_cost_inr=30, estimated_duration_minutes=80, modes=["bus"])
    mid = _c(name="train", estimated_cost_inr=90, estimated_duration_minutes=55, modes=["train"])
    pricey = _c(name="cab", estimated_cost_inr=600, estimated_duration_minutes=40, modes=["car"])
    winner, _, _ = rank_cheapest([pricey, mid, cheap])
    assert winner["name"] == "cheap-bus"


def test_fastest_is_lowest_duration():
    slow = _c(name="bus", estimated_duration_minutes=90, estimated_cost_inr=20, modes=["bus"])
    fast = _c(name="cab", estimated_duration_minutes=42, estimated_cost_inr=600, modes=["car"])
    mid = _c(name="rapido", estimated_duration_minutes=50, estimated_cost_inr=300, modes=["rapido"])
    winner, _, _ = rank_fastest([slow, mid, fast])
    assert winner["name"] == "cab"


def test_comfortable_prefers_fewer_transfers():
    hard = _c(
        name="many",
        transfers=3,
        walking_meters=1200,
        estimated_duration_minutes=70,
        estimated_cost_inr=50,
        modes=["walk", "bus", "train", "walk"],
    )
    direct = _c(
        name="direct-bus",
        transfers=0,
        walking_meters=200,
        estimated_duration_minutes=85,
        estimated_cost_inr=40,
        modes=["bus"],
    )
    cab = _c(
        name="cab",
        transfers=0,
        walking_meters=80,
        estimated_duration_minutes=45,
        estimated_cost_inr=650,
        modes=["car"],
    )
    winner, _, _ = rank_comfortable([hard, direct, cab])
    # Both direct-bus and cab have 0 transfers; lower walking wins among them, then waiting/duration/cost.
    assert winner["transfers"] == 0
    assert winner["name"] in {"direct-bus", "cab"}


def test_rank_all_three_prefs_only():
    candidates = [
        _c(name="bus", estimated_cost_inr=35, estimated_duration_minutes=80, transfers=0, modes=["bus"]),
        _c(name="car", estimated_cost_inr=650, estimated_duration_minutes=40, transfers=0, modes=["car"]),
        _c(
            name="train-auto",
            estimated_cost_inr=95,
            estimated_duration_minutes=70,
            transfers=1,
            walking_meters=400,
            modes=["walk", "train", "auto"],
        ),
    ]
    results = rank_all(candidates)
    assert set(results) == {"cheapest", "comfortable", "fastest"}
    assert "safest" not in results
    assert results["cheapest"]["name"] == "bus"
    assert results["fastest"]["name"] == "car"


def test_impossible_long_walk_rejected():
    bad = {
        "name": "bad-walk",
        "modes": ["walk"],
        "legs": [
            {
                "mode": "walk",
                "from_location": "Potheri",
                "to_location": "Thiruvallur",
                "from_coordinates": (12.8250, 80.0455),
                "to_coordinates": (13.1439, 79.9080),
                "estimated_duration_minutes": 480,
                "estimated_cost_inr": 0,
                "distance_meters": 40000,
                "data_source": "test",
            }
        ],
        "estimated_duration_minutes": 480,
        "estimated_cost_inr": 0,
        "transfers": 0,
        "walking_meters": 40000,
    }
    ok, _ = validate_candidate(bad)
    assert ok is False


def test_missing_ml_does_not_break_ranking():
    from app.ml.route_predictor import RoutePreferencePredictor

    predictor = RoutePreferencePredictor(model_path="/tmp/does-not-exist-raaha.joblib")
    candidates = [
        _c(name="a", estimated_cost_inr=40, estimated_duration_minutes=70, modes=["bus"]),
        _c(name="b", estimated_cost_inr=500, estimated_duration_minutes=40, modes=["car"]),
    ]
    ranked = predictor.rank_recommendations(candidates)
    assert set(ranked["recommendations"]) == {"cheapest", "comfortable", "fastest"}
    assert ranked["recommendations"]["cheapest"]["name"] == "a"
    assert ranked["recommendations"]["fastest"]["name"] == "b"
