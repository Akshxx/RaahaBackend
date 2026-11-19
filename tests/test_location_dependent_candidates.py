"""Location-dependent candidate generation smoke tests (network optional)."""

from __future__ import annotations

import pytest

from app.ml.deterministic_ranker import rank_all
from app.services.candidate_generator import generate_candidates


OD_PAIRS = [
    (
        "srm_marina",
        {"name": "SRM", "latitude": 12.8231, "longitude": 80.0442},
        {"name": "Marina Beach", "latitude": 13.0500, "longitude": 80.2824},
    ),
    (
        "srm_central",
        {"name": "SRM", "latitude": 12.8231, "longitude": 80.0442},
        {"name": "Chennai Central", "latitude": 13.0827, "longitude": 80.2707},
    ),
    (
        "tnagar_marina",
        {"name": "T Nagar", "latitude": 13.0418, "longitude": 80.2341},
        {"name": "Marina Beach", "latitude": 13.0500, "longitude": 80.2824},
    ),
]


@pytest.mark.parametrize("label,origin,destination", OD_PAIRS)
def test_candidates_depend_on_locations(label, origin, destination):
    candidates, quality = generate_candidates(origin, destination)
    assert quality["candidate_count"] == len(candidates)
    assert candidates, f"No candidates for {label}: {quality}"
    # Ranking objectives must hold on the returned pool.
    ranked = rank_all(candidates)
    assert set(ranked) == {"cheapest", "comfortable", "fastest"}
    cheap = ranked["cheapest"]
    fast = ranked["fastest"]
    comfort = ranked["comfortable"]
    min_cost = min(float(c["estimated_cost_inr"]) for c in candidates)
    min_eta = min(int(c["estimated_duration_minutes"]) for c in candidates)
    min_xfers = min(int(c["transfers"]) for c in candidates)
    assert float(cheap["estimated_cost_inr"]) <= min_cost + 1e-6 or cheap.get("fare_unavailable")
    assert int(fast["estimated_duration_minutes"]) <= min_eta + 1
    assert int(comfort["transfers"]) <= min_xfers


def test_different_destinations_change_candidate_metrics():
    origin = {"name": "SRM", "latitude": 12.8231, "longitude": 80.0442}
    marina = {"name": "Marina", "latitude": 13.0500, "longitude": 80.2824}
    airport = {"name": "Airport", "latitude": 12.9941, "longitude": 80.1709}
    a, qa = generate_candidates(origin, marina)
    b, qb = generate_candidates(origin, airport)
    assert a and b
    # Durations/costs should not be identical across distant destinations.
    a_signature = sorted(
        (round(c["estimated_duration_minutes"], 0), round(float(c["estimated_cost_inr"]), 0), tuple(c.get("modes") or []))
        for c in a
    )
    b_signature = sorted(
        (round(c["estimated_duration_minutes"], 0), round(float(c["estimated_cost_inr"]), 0), tuple(c.get("modes") or []))
        for c in b
    )
    assert a_signature != b_signature
