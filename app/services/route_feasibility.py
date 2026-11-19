"""Geographical and physical feasibility validation for multimodal candidates."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.services.feasibility_config import DEFAULT_FEASIBILITY_CONFIG, FeasibilityConfig

logger = logging.getLogger("raaha.feasibility")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def validate_candidate(
    candidate: Dict[str, Any],
    config: Optional[FeasibilityConfig] = None,
) -> Tuple[bool, List[str]]:
    """Return (is_feasible, reasons). Reasons explain reject or soft warnings."""
    cfg = config or DEFAULT_FEASIBILITY_CONFIG
    reasons: List[str] = []
    legs = candidate.get("legs") or []
    if not legs:
        return False, ["Candidate has no legs."]

    total_walk = 0.0
    prev_end = None

    for index, leg in enumerate(legs):
        mode = str(leg.get("mode", "")).lower()
        start = leg.get("from_coordinates") or leg.get("start_coordinates")
        end = leg.get("to_coordinates") or leg.get("end_coordinates")
        declared = leg.get("distance_meters")
        if declared is None and leg.get("distance_km") is not None:
            declared = float(leg["distance_km"]) * 1000.0

        # Continuity: next leg should start near previous end when coords exist.
        if prev_end and start:
            gap = haversine_meters(prev_end[0], prev_end[1], start[0], start[1])
            if gap > cfg.max_transfer_gap_km * 1000.0:
                return False, [
                    f"Disconnected transfer before leg {index + 1}: "
                    f"{gap:.0f} m gap exceeds {cfg.max_transfer_gap_km} km."
                ]

        if mode == "walk":
            walk_m = None
            approx = False
            if start and end:
                walk_m = haversine_meters(start[0], start[1], end[0], end[1])
                # Haversine underestimates pedestrian path; inflate slightly.
                walk_m *= 1.25
                approx = True
            elif declared is not None:
                walk_m = float(declared)
            else:
                # Unknown walk length — reject rather than invent connectivity.
                return False, [
                    f"Walking leg {index + 1} lacks distance/coordinates; rejected."
                ]

            total_walk += walk_m
            limit = (
                cfg.preferred_transfer_walk_meters
                if 0 < index < len(legs) - 1
                else cfg.preferred_last_mile_walk_meters
            )
            if walk_m > cfg.max_walk_leg_meters:
                label = "approximate pedestrian estimate" if approx else "declared distance"
                return False, [
                    f"Walking leg {index + 1} is {walk_m:.0f} m ({label}), "
                    f"exceeds hard limit {cfg.max_walk_leg_meters:.0f} m."
                ]
            if cfg.reject_walk_above_preferred and walk_m > limit:
                return False, [
                    f"Walking leg {index + 1} is {walk_m:.0f} m, above preferred "
                    f"limit {limit:.0f} m for this connection type."
                ]
            if approx:
                reasons.append(
                    f"Walk leg {index + 1} distance is approximate (haversine×1.25)."
                )

        # Impossible same-point non-walk with zero distance and mismatched names
        from_name = str(leg.get("from_location", ""))
        to_name = str(leg.get("to_location", ""))
        if mode != "walk" and from_name and to_name and from_name == to_name:
            return False, [f"Leg {index + 1} has identical from/to locations."]

        if end:
            prev_end = end
        elif start:
            prev_end = start

    if total_walk > cfg.max_total_walking_meters:
        return False, [
            f"Total walking {total_walk:.0f} m exceeds journey budget "
            f"{cfg.max_total_walking_meters:.0f} m."
        ]

    # Reject journeys that are walk-only over long OSRM distances already filtered elsewhere.
    modes = [str(leg.get("mode", "")).lower() for leg in legs]
    if modes and all(m == "walk" for m in modes) and total_walk > cfg.max_walk_leg_meters:
        return False, ["Pure walking journey exceeds maximum walking distance."]

    return True, reasons


def filter_feasible(
    candidates: List[Dict[str, Any]],
    config: Optional[FeasibilityConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into (accepted, rejected_with_reasons)."""
    cfg = config or DEFAULT_FEASIBILITY_CONFIG
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        ok, reasons = validate_candidate(candidate, cfg)
        item = dict(candidate)
        item["feasibility_notes"] = reasons
        if ok:
            item["feasible"] = True
            accepted.append(item)
            if reasons:
                logger.info(
                    "Accepted candidate %s with notes: %s",
                    candidate.get("name"),
                    "; ".join(reasons),
                )
        else:
            item["feasible"] = False
            item["reject_reasons"] = reasons
            rejected.append(item)
            logger.info(
                "Rejected candidate %s: %s",
                candidate.get("name"),
                "; ".join(reasons),
            )
    return accepted, rejected
