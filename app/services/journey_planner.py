"""Journey planning orchestration: geocode → providers → candidates → rank."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from app.ml.route_predictor import PREFERENCE_KEYS, get_predictor
from app.models.journey import JourneyLeg, JourneyOption, JourneyRequest, MLJourneyRequest
from app.providers.provider_registry import get_registry
from app.services.candidate_generator import generate_candidates
from app.services.location_service import search_location

logger = logging.getLogger("raaha.planner")


def plan_journey(request: JourneyRequest) -> dict[str, Any]:
    """Legacy endpoint helper — returns ranked routes."""
    ml_request = MLJourneyRequest(
        origin=request.origin,
        destination=request.destination,
    )
    result = plan_journey_ml(ml_request)

    return {
        "status": "success",
        "origin": result["origin"],
        "destination": result["destination"],
        "preference": request.preference,
        "routes": result["routes"],
        "data_source": "Nominatim + provider registry + deterministic ranker",
        "model_used": result.get("ml_assist_enabled", False),
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "data_quality": result["data_quality"],
        "provider_status": result.get("provider_status"),
    }


def plan_journey_ml(request: MLJourneyRequest) -> dict[str, Any]:
    logger.info(
        "Journey request received origin=%r destination=%r",
        request.origin,
        request.destination,
    )

    try:
        origin = search_location(request.origin)
        destination = search_location(request.destination)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Geocoding failed")
        raise HTTPException(status_code=502, detail=f"Location lookup failed: {exc}") from exc

    kwargs = {}
    if request.departure_time:
        kwargs["departure_time"] = request.departure_time

    candidates, quality = generate_candidates(origin, destination, **kwargs)
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "No feasible route candidates could be generated.",
                "provider_report": quality.get("provider_report"),
                "rejected_examples": quality.get("rejected_examples"),
            },
        )

    predictor = get_predictor()
    predictor.ensure_loaded()  # optional ML — never hard-fails planning

    # Always rank all three preferences so every route can be fully labelled.
    all_preferences = list(PREFERENCE_KEYS)
    ranked = predictor.rank_recommendations(candidates, preferences=all_preferences)
    recommendations_raw = ranked["recommendations"]   # {pref: raw_dict}
    candidate_routes_raw = ranked["candidate_routes"]

    provider_status = quality.get("provider_status") or get_registry().status_summary()

    # Build a label map: route name → list of labels it won
    _LABEL_MAP = {
        "cheapest": "Cheapest",
        "comfortable": "Most Comfortable",
        "fastest": "Fastest",
    }
    label_lookup: dict[str, list[str]] = {}
    score_lookup: dict[str, dict[str, float]] = {}
    for pref, raw in recommendations_raw.items():
        rname = raw["name"]
        label_lookup.setdefault(rname, []).append(_LABEL_MAP.get(pref, pref.capitalize()))
        # Collect per-preference scores from the raw dict
        pscores = raw.get("preference_scores") or {}
        score_lookup.setdefault(rname, {}).update(pscores)
        if raw.get("suitability_score") is not None:
            score_lookup[rname][pref] = float(raw["suitability_score"])

    # Build de-duplicated flat route list (winners first, then remaining candidates)
    seen: set[str] = set()
    routes: list[JourneyOption] = []

    # Add winners in a consistent order: cheapest → fastest → comfortable
    for pref in all_preferences:
        raw = recommendations_raw.get(pref)
        if raw and raw["name"] not in seen:
            seen.add(raw["name"])
            opt = _to_option(raw)
            opt.labels = label_lookup.get(raw["name"], [])
            opt.scores = score_lookup.get(raw["name"])
            routes.append(opt)

    # Append remaining candidates (not already a winner) with their own scoring
    for raw in candidate_routes_raw:
        if raw["name"] not in seen:
            seen.add(raw["name"])
            opt = _to_option(raw)
            opt.labels = label_lookup.get(raw["name"], [])
            opt.scores = raw.get("preference_scores")
            routes.append(opt)

    logger.info(
        "Final routes returned total=%s winners=%s",
        len(routes),
        len(recommendations_raw),
    )

    return {
        "status": "success",
        "model_used": bool(ranked.get("ml_assist_enabled")),
        "model_name": ranked["model_name"],
        "model_version": ranked["model_version"],
        "ranking_layer": ranked.get("ranking_layer"),
        "ml_assist_enabled": ranked.get("ml_assist_enabled"),
        "origin": origin["name"],
        "destination": destination["name"],
        "origin_coordinates": {
            "latitude": origin["latitude"],
            "longitude": origin["longitude"],
        },
        "destination_coordinates": {
            "latitude": destination["latitude"],
            "longitude": destination["longitude"],
        },
        "routes": routes,
        "data_quality": quality,
        "provider_status": provider_status,
    }


def _to_option(raw: dict[str, Any]) -> JourneyOption:
    legs = []
    for leg in raw.get("legs") or []:
        mode = str(leg.get("mode", "walk")).lower()
        if mode not in {
            "car",
            "bus",
            "bike",
            "train",
            "rapido",
            "auto",
            "metro",
            "walk",
        }:
            mode = "walk"
        legs.append(
            JourneyLeg(
                mode=mode,  # type: ignore[arg-type]
                from_location=str(leg.get("from_location") or ""),
                to_location=str(leg.get("to_location") or ""),
                estimated_duration_minutes=int(leg.get("estimated_duration_minutes") or 0),
                estimated_cost_inr=float(leg.get("estimated_cost_inr") or 0),
                data_source=str(leg.get("data_source") or leg.get("provider") or "unknown"),
                distance_km=leg.get("distance_km"),
                distance_meters=leg.get("distance_meters"),
                waiting_minutes=leg.get("waiting_minutes"),
                provider=leg.get("provider"),
                fare_status=leg.get("fare_status"),
                duration_status=leg.get("duration_status"),
                fare_unavailable=leg.get("fare_unavailable"),
                confidence=leg.get("confidence"),
                instructions=leg.get("instructions"),
                departure_time=leg.get("departure_time"),
                arrival_time=leg.get("arrival_time"),
            )
        )

    return JourneyOption(
        name=raw["name"],
        legs=legs,
        estimated_duration_minutes=int(raw["estimated_duration_minutes"]),
        estimated_cost_inr=float(raw.get("estimated_cost_inr") or 0),
        transfers=int(raw.get("transfers", 0)),
        description=raw.get("description", ""),
        modes=list(raw.get("modes") or []),
        walking_meters=raw.get("walking_meters"),
        waiting_minutes=raw.get("waiting_minutes"),
        distance_km=raw.get("distance_km"),
        suitability_score=raw.get("suitability_score")
        or (raw.get("preference_scores") or {}).get(raw.get("preference", ""), None),
        preference_scores=raw.get("preference_scores"),
        contains_simulated_data=bool(raw.get("contains_simulated_data")),
        data_sources=list(raw.get("data_sources") or []),
        feature_snapshot=raw.get("feature_snapshot"),
        recommendation_reason=raw.get("recommendation_reason") or raw.get("ranking_note"),
        ranking_note=raw.get("ranking_note") or raw.get("recommendation_reason"),
        ranking_method=raw.get("ranking_method"),
        provider=raw.get("provider"),
        fare_unavailable=raw.get("fare_unavailable"),
        fare_partial=raw.get("fare_partial"),
        confidence=raw.get("confidence"),
        data_freshness=raw.get("data_freshness"),
        preference=raw.get("preference"),
    )
