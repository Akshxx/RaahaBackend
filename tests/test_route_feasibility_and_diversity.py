"""Feasibility and connectivity tests (SAFEST removed from product ranking)."""

from __future__ import annotations

from app.ml.deterministic_ranker import rank_all, rank_comfortable
from app.services.chennai_network import (
    corridor_candidates_for_endpoints,
    impossible_potheri_thiruvallur_walk_candidate,
    path_to_candidate,
)
from app.services.route_feasibility import filter_feasible, validate_candidate


def test_rejects_potheri_thiruvallur_walk():
    bad = impossible_potheri_thiruvallur_walk_candidate()
    ok, reasons = validate_candidate(bad)
    assert ok is False
    assert any("exceeds" in r.lower() or "above preferred" in r.lower() for r in reasons)


def test_rejects_long_walking_connections():
    candidate = {
        "name": "long-walk",
        "modes": ["walk"],
        "legs": [
            {
                "mode": "walk",
                "from_location": "A",
                "to_location": "B",
                "from_coordinates": (12.82, 80.04),
                "to_coordinates": (12.90, 80.10),
                "estimated_duration_minutes": 90,
                "estimated_cost_inr": 0,
                "distance_meters": 9000,
                "data_source": "test",
            }
        ],
        "estimated_duration_minutes": 90,
        "estimated_cost_inr": 0,
        "transfers": 0,
        "walking_meters": 9000,
    }
    ok, _ = validate_candidate(candidate)
    assert ok is False


def test_potheri_velachery_train_path_builds_when_network_supports_it():
    candidate = path_to_candidate(
        name="train-test",
        hub_path=["srm_kattankulathur", "potheri_railway", "velachery_railway", "phoenix_marketcity"],
        mode_sequence=["walk", "train", "auto"],
        notes="test",
    )
    assert candidate is not None
    assert "train" in candidate["modes"]
    ok, reasons = validate_candidate(candidate)
    assert ok is True, reasons


def test_disconnected_legs_rejected():
    candidate = {
        "name": "disconnected",
        "modes": ["train", "auto"],
        "legs": [
            {
                "mode": "train",
                "from_location": "Potheri",
                "to_location": "Velachery",
                "from_coordinates": (12.8250, 80.0455),
                "to_coordinates": (12.9758, 80.2209),
                "estimated_duration_minutes": 45,
                "estimated_cost_inr": 15,
                "distance_km": 32,
                "data_source": "t",
            },
            {
                "mode": "auto",
                "from_location": "Somewhere else",
                "to_location": "Phoenix",
                "from_coordinates": (13.08, 80.27),
                "to_coordinates": (12.9916, 80.2170),
                "estimated_duration_minutes": 12,
                "estimated_cost_inr": 80,
                "distance_km": 2,
                "data_source": "t",
            },
        ],
        "estimated_duration_minutes": 57,
        "estimated_cost_inr": 95,
        "transfers": 1,
        "walking_meters": 0,
    }
    ok, reasons = validate_candidate(candidate)
    assert ok is False
    assert any("Disconnected" in r for r in reasons)


def test_comfortable_prefers_fewer_transfers():
    hard = {
        "name": "many-transfers",
        "modes": ["walk", "bus", "train", "walk"],
        "legs": [],
        "estimated_duration_minutes": 70,
        "estimated_cost_inr": 60,
        "transfers": 3,
        "walking_meters": 1400,
        "waiting_minutes": 16,
    }
    easy = {
        "name": "direct-bus",
        "modes": ["bus"],
        "legs": [],
        "estimated_duration_minutes": 80,
        "estimated_cost_inr": 40,
        "transfers": 0,
        "walking_meters": 200,
        "waiting_minutes": 8,
    }
    winner, _, _ = rank_comfortable([hard, easy])
    assert winner["name"] == "direct-bus"


def test_invalid_candidates_not_selected():
    bad = impossible_potheri_thiruvallur_walk_candidate()
    good = {
        "name": "metro",
        "modes": ["metro"],
        "legs": [
            {
                "mode": "metro",
                "from_location": "A",
                "to_location": "B",
                "from_coordinates": (12.97, 80.16),
                "to_coordinates": (13.00, 80.21),
                "estimated_duration_minutes": 60,
                "estimated_cost_inr": 50,
                "distance_meters": 12000,
                "data_source": "t",
            }
        ],
        "estimated_duration_minutes": 60,
        "estimated_cost_inr": 50,
        "transfers": 0,
        "walking_meters": 200,
        "waiting_minutes": 5,
    }
    accepted, _ = filter_feasible([bad, good])
    assert all(c["name"] != bad["name"] for c in accepted)
    results = rank_all(accepted or [good])
    assert set(results) == {"cheapest", "comfortable", "fastest"}
    assert "safest" not in results


def test_single_feasible_candidate_not_fabricated():
    only = {
        "name": "only",
        "modes": ["bus"],
        "legs": [],
        "estimated_duration_minutes": 60,
        "estimated_cost_inr": 40,
        "transfers": 0,
        "walking_meters": 200,
        "waiting_minutes": 8,
    }
    results = rank_all([only])
    assert {results[p]["name"] for p in results} == {"only"}


def test_corridor_candidates_still_parse_for_legacy_network():
    origin = {"name": "SRM", "latitude": 12.8231, "longitude": 80.0442}
    destination = {"name": "Phoenix", "latitude": 12.9916, "longitude": 80.2170}
    candidates = corridor_candidates_for_endpoints(origin, destination)
    assert len(candidates) >= 1
