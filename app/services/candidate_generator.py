"""Multimodal candidate generation from real providers + feasibility filter.

Pipeline:
1. Collect normalized routes from OSRM / Google / GTFS / RailRadar
2. Convert to candidate dicts
3. Validate geographical feasibility
4. Dedupe
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.providers.provider_registry import get_registry
from app.services.route_feasibility import filter_feasible

logger = logging.getLogger("raaha.candidates")


def generate_candidates(
    origin: dict[str, Any],
    destination: dict[str, Any],
    **kwargs: Any,
) -> Tuple[List[dict], dict]:
    registry = get_registry()
    normalized, provider_report = registry.collect_routes(origin, destination, **kwargs)

    raw: List[dict] = [route.to_candidate() for route in normalized]
    has_road = any(set(c.get("modes") or []) & {"car", "auto", "rapido", "bike"} for c in raw)
    if not has_road:
        logger.warning("No road candidates from OSRM/Google — adding labelled distance fallback")
        raw.extend(_distance_fallback_candidates(origin, destination))
        provider_report.setdefault("calls", []).append(
            {
                "provider": "haversine_fallback",
                "ok": True,
                "count": 2,
                "note": "Used only because live road routing was unavailable",
            }
        )
    has_transit = any(
        set(c.get("modes") or []) & {"bus", "metro", "train"} for c in raw
    )
    if not has_transit:
        logger.info("No transit candidates from providers; OD may be outside feed coverage")

    accepted, rejected = filter_feasible(raw)
    accepted = _dedupe(accepted)

    real_available = any(
        not c.get("contains_simulated_data") for c in accepted
    ) or any(
        call.get("ok") and call.get("provider") in {"osrm", "google_routes", "chennai_gtfs", "railradar"}
        for call in provider_report.get("calls", [])
    )
    simulated_used = any(c.get("contains_simulated_data") for c in accepted)

    quality = {
        "real_data_available": bool(accepted) and real_available,
        "simulated_data_used": simulated_used,
        "candidate_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected_examples": [
            {"name": r.get("name"), "reasons": r.get("reject_reasons")}
            for r in rejected[:8]
        ],
        "provider_report": provider_report,
        "provider_status": registry.status_summary(),
        "notes": (
            "Candidates are built from configured providers (OSRM, optional Google Routes, "
            "Chennai GTFS, optional RailRadar). Estimated fares are labelled. "
            "Synthetic Chennai hub templates are no longer used as live evidence."
        ),
    }
    logger.info(
        "Candidates accepted=%s rejected=%s providers=%s",
        len(accepted),
        len(rejected),
        [c.get("provider") for c in provider_report.get("calls", [])],
    )
    return accepted, quality


def _distance_fallback_candidates(origin: dict, destination: dict) -> List[dict]:
    """Labelled estimate when OSRM/Google road routing is unavailable."""
    import math

    from app.providers.base_provider import DataStatus, NormalizedLeg, NormalizedRoute

    lat1, lon1 = math.radians(origin["latitude"]), math.radians(origin["longitude"])
    lat2, lon2 = math.radians(destination["latitude"]), math.radians(destination["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    km = 6371.0 * 2 * math.asin(math.sqrt(a))
    road_km = km * 1.35
    drive_min = max(12.0, road_km / 0.45)
    out = []
    for mode, name, mult_time, per_km, base, comfort in [
        ("car", "Driving estimate (fallback)", 1.0, 18.0, 40.0, 0.75),
        ("auto", "Auto estimate (fallback)", 1.2, 14.0, 30.0, 0.55),
        ("rapido", "Two-wheeler estimate (fallback)", 1.05, 10.0, 25.0, 0.4),
    ]:
        leg = NormalizedLeg(
            mode=mode,
            from_location=origin["name"],
            to_location=destination["name"],
            from_coordinates=(origin["latitude"], origin["longitude"]),
            to_coordinates=(destination["latitude"], destination["longitude"]),
            duration_minutes=drive_min * mult_time,
            distance_meters=road_km * 1000,
            fare_inr=round(base + road_km * per_km, 2),
            waiting_minutes=3,
            provider="haversine_fallback",
            data_source="haversine_fallback_estimate",
            duration_status=DataStatus.ESTIMATED,
            fare_status=DataStatus.ESTIMATED,
            confidence=0.35,
        )
        out.append(
            NormalizedRoute(
                name=name,
                legs=[leg],
                provider="haversine_fallback",
                contains_simulated_data=False,
                data_sources=["haversine_fallback_estimate"],
                metadata={
                    "description": (
                        "Estimated road journey because live OSRM/Google routing was unavailable. "
                        "Not a verified live ETA or fare."
                    ),
                    "comfort_index": comfort,
                    "crowding_estimate": 0.25,
                    "confidence": 0.35,
                    "data_freshness": "estimate_only",
                },
            ).to_candidate()
        )
    return out


def _dedupe(candidates: List[dict]) -> List[dict]:
    best: Dict[tuple, dict] = {}
    order: List[tuple] = []
    for candidate in candidates:
        modes = tuple(str(m).lower() for m in (candidate.get("modes") or []))
        key = (
            modes,
            int(candidate.get("estimated_duration_minutes") or 0) // 5,
            int(float(candidate.get("estimated_cost_inr") or 0) // 30),
            tuple(
                (str(leg.get("from_location")), str(leg.get("to_location")))
                for leg in (candidate.get("legs") or [])[:4]
            ),
        )
        existing = best.get(key)
        if existing is None:
            best[key] = candidate
            order.append(key)
            continue
        cur = (
            float(candidate.get("estimated_duration_minutes") or 1e9),
            float(candidate.get("estimated_cost_inr") or 1e9),
        )
        prev = (
            float(existing.get("estimated_duration_minutes") or 1e9),
            float(existing.get("estimated_cost_inr") or 1e9),
        )
        if cur < prev:
            best[key] = candidate
    return [best[key] for key in order]
