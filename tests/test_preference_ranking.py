"""Deterministic preference ranking scenarios (Cheapest / Comfortable / Fastest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.deterministic_ranker import rank_all, rank_cheapest, rank_comfortable, rank_fastest
from app.ml.route_predictor import RoutePreferencePredictor
from app.ml.train_model import train_and_save


def _candidate(**kwargs):
    base = {
        "legs": [],
        "description": "test",
        "preference_scores": {
            "fastest": 0.5,
            "cheapest": 0.5,
            "comfortable": 0.5,
        },
        "transfers": 0,
        "walking_meters": 200,
        "waiting_minutes": 5,
        "comfort_index": 0.5,
        "crowding_estimate": 0.5,
        "contains_simulated_data": True,
    }
    base.update(kwargs)
    if not base["legs"]:
        modes = base.get("modes") or ["bus"]
        total = float(base.get("estimated_cost_inr") or 0)
        paid = [m for m in modes if m != "walk"] or modes
        share = round(total / max(len(paid), 1), 2)
        base["legs"] = [
            {
                "mode": mode,
                "from_location": "A",
                "to_location": "B",
                "estimated_duration_minutes": int(base.get("estimated_duration_minutes") or 30)
                // max(len(modes), 1),
                "estimated_cost_inr": 0.0 if mode == "walk" else share,
                "data_source": "test",
            }
            for mode in modes
        ]
    return base


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("model")
    path = root / "route_preference_model.joblib"
    metrics = root / "training_metrics.json"
    train_and_save(model_path=path, metrics_path=metrics, n_samples=200, seed=7)
    return path


def test_cheapest_is_lowest_fare():
    bus = _candidate(name="bus", modes=["bus"], estimated_cost_inr=40, estimated_duration_minutes=70)
    car = _candidate(name="car", modes=["car"], estimated_cost_inr=600, estimated_duration_minutes=40)
    winner, _, _ = rank_cheapest([car, bus])
    assert winner["name"] == "bus"


def test_fastest_is_lowest_duration():
    bus = _candidate(name="bus", modes=["bus"], estimated_cost_inr=40, estimated_duration_minutes=70)
    car = _candidate(name="car", modes=["car"], estimated_cost_inr=600, estimated_duration_minutes=40)
    winner, _, _ = rank_fastest([bus, car])
    assert winner["name"] == "car"


def test_comfortable_prefers_fewer_transfers():
    hard = _candidate(
        name="hard",
        modes=["walk", "bus", "train", "walk"],
        transfers=3,
        walking_meters=1200,
        estimated_duration_minutes=70,
        estimated_cost_inr=50,
    )
    direct = _candidate(
        name="direct",
        modes=["bus"],
        transfers=0,
        walking_meters=250,
        estimated_duration_minutes=85,
        estimated_cost_inr=40,
    )
    winner, _, _ = rank_comfortable([hard, direct])
    assert winner["name"] == "direct"


def test_no_safest_in_rank_all():
    results = rank_all(
        [
            _candidate(name="a", modes=["bus"], estimated_cost_inr=30, estimated_duration_minutes=80),
            _candidate(name="b", modes=["car"], estimated_cost_inr=500, estimated_duration_minutes=40),
        ]
    )
    assert set(results) == {"cheapest", "comfortable", "fastest"}
    assert "safest" not in results


def test_predictor_without_model_still_ranks(tmp_path: Path):
    predictor = RoutePreferencePredictor(model_path=tmp_path / "missing.joblib")
    ranked = predictor.rank_recommendations(
        [
            _candidate(name="bus", modes=["bus"], estimated_cost_inr=35, estimated_duration_minutes=70),
            _candidate(name="car", modes=["car"], estimated_cost_inr=600, estimated_duration_minutes=40),
        ]
    )
    assert set(ranked["recommendations"]) == {"cheapest", "comfortable", "fastest"}
    assert ranked["recommendations"]["cheapest"]["name"] == "bus"
    assert ranked["recommendations"]["fastest"]["name"] == "car"


def test_predictor_with_optional_model(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    ranked = predictor.rank_recommendations(
        [
            _candidate(name="bus", modes=["bus"], estimated_cost_inr=35, estimated_duration_minutes=70),
            _candidate(name="car", modes=["car"], estimated_cost_inr=600, estimated_duration_minutes=40),
            _candidate(
                name="train",
                modes=["walk", "train", "auto"],
                transfers=1,
                walking_meters=400,
                estimated_cost_inr=90,
                estimated_duration_minutes=65,
            ),
        ]
    )
    assert set(ranked["recommendations"]) == {"cheapest", "comfortable", "fastest"}
    assert ranked["recommendations"]["cheapest"]["name"] == "bus"
