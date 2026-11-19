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
    preferences = list(PREFERENCE_KEYS)
    if request.preference in PREFERENCE_KEYS:
        preferences = [request.preference] + [p for p in PREFERENCE_KEYS if p != request.preference]
    elif request.preference == "balanced":
        preferences = list(PREFERENCE_KEYS)

    ml_request = MLJourneyRequest(
        origin=request.origin,
        destination=request.destination,
        preferences=preferences,  # type: ignore[arg-type]
    )
    result = plan_journey_ml(ml_request)

    preferred = request.preference if request.preference in result["recommendations"] else "fastest"
    ordered = []
    if preferred in result["recommendations"]:
        ordered.append(result["recommendations"][preferred])
    for key in PREFERENCE_KEYS:
        option = result["recommendations"].get(key)
        if option and option not in ordered:
            ordered.append(option)

    return {
        "status": "success",
        "origin": result["origin"],
        "destination": result["destination"],
        "preference": request.preference,
        "routes": ordered,
        "data_source": "Nominatim + provider registry + deterministic ranker",
        "model_used": result.get("ml_assist_enabled", False),
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "recommendations": result["recommendations"],
        "candidate_routes": result["candidate_routes"],
        "data_quality": result["data_quality"],
        "provider_status": result.get("provider_status"),
    }


def plan_journey_ml(request: MLJourneyRequest) -> dict[str, Any]:
    logger.info(
        "Journey request received origin=%r destination=%r preferences=%s",
        request.origin,
        request.destination,
        request.preferences,
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

    preferences = [p for p in request.preferences if p in PREFERENCE_KEYS] or list(PREFERENCE_KEYS)
    ranked = predictor.rank_recommendations(candidates, preferences=preferences)
    recommendations = {
        key: _to_option(value) for key, value in ranked["recommendations"].items()
    }
    candidate_routes = [_to_option(item) for item in ranked["candidate_routes"]]
    provider_status = quality.get("provider_status") or get_registry().status_summary()

    logger.info(
        "Final recommendations returned prefs=%s candidates=%s",
        list(recommendations.keys()),
        len(candidate_routes),
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
        "recommendations": recommendations,
        "candidate_routes": candidate_routes,
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
