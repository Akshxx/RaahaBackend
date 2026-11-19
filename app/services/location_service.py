"""Geocoding via OpenStreetMap Nominatim (no paid API keys)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("raaha.location")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "RAaHA-Travel-App/1.0 (development; local prototype)"
}

# Optional demo aliases with verified approximate coordinates for faculty demos
# when Nominatim rate-limits or returns empty. Not SRM→Phoenix route hardcoding —
# only place → lat/lon resolution.
_DEMO_ALIASES: dict[str, dict[str, Any]] = {
    "srm institute of science and technology": {
        "name": "SRM Institute of Science and Technology, Kattankulathur",
        "latitude": 12.8231,
        "longitude": 80.0442,
        "data_source": "demo_alias_coordinates",
    },
    "srm kattankulathur": {
        "name": "SRM Institute of Science and Technology, Kattankulathur",
        "latitude": 12.8231,
        "longitude": 80.0442,
        "data_source": "demo_alias_coordinates",
    },
    "srm": {
        "name": "SRM Institute of Science and Technology, Kattankulathur",
        "latitude": 12.8231,
        "longitude": 80.0442,
        "data_source": "demo_alias_coordinates",
    },
    "phoenix marketcity": {
        "name": "Phoenix Marketcity, Velachery, Chennai",
        "latitude": 12.9916,
        "longitude": 80.2170,
        "data_source": "demo_alias_coordinates",
    },
    "phoenix mall": {
        "name": "Phoenix Marketcity, Velachery, Chennai",
        "latitude": 12.9916,
        "longitude": 80.2170,
        "data_source": "demo_alias_coordinates",
    },
    "marina beach": {
        "name": "Marina Beach, Chennai",
        "latitude": 13.0500,
        "longitude": 80.2824,
        "data_source": "demo_alias_coordinates",
    },
}


def search_location(place: str) -> dict:
    """Convert a place name into coordinates."""
    cleaned = place.strip()
    if not cleaned:
        raise ValueError("Location name is empty.")

    alias = _lookup_alias(cleaned)
    queries = [
        cleaned,
        f"{cleaned}, Chennai, India",
        f"{cleaned}, Tamil Nadu, India",
        f"{cleaned}, India",
    ]

    last_error = None  # type: Optional[Exception]
    for query in queries:
        try:
            result = _nominatim_search(query)
            if result:
                logger.info("Geocoded %r → %s", place, result["name"][:80])
                return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Nominatim query failed for %r: %s", query, exc)

    if alias:
        logger.info("Using demo alias coordinates for %r", place)
        return alias

    if last_error:
        raise ValueError(f"Location not found: {place} ({last_error})") from last_error
    raise ValueError(f"Location not found: {place}")


def _lookup_alias(place: str):
    key = place.lower().strip()
    for alias, payload in _DEMO_ALIASES.items():
        if alias in key or key in alias:
            return dict(payload)
    return None


def _nominatim_search(query: str) -> dict[str, Any] | None:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in",
        "addressdetails": 1,
    }
    response = httpx.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=15.0,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    result = results[0]
    return {
        "name": result["display_name"],
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "data_source": "Nominatim",
    }
