"""Google Maps Routes API (computeRoutes) provider."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.settings import get_settings
from app.providers.base_provider import (
    BaseProvider,
    DataStatus,
    NormalizedLeg,
    NormalizedRoute,
    ProviderStatus,
)
from app.providers.http_utils import request_json

logger = logging.getLogger("raaha.providers.google")

FIELD_MASK = ",".join(
    [
        "routes.duration",
        "routes.distanceMeters",
        "routes.legs.duration",
        "routes.legs.distanceMeters",
        "routes.legs.steps",
        "routes.polyline.encodedPolyline",
        "routes.travelAdvisory.transitFare",
        "routes.localizedValues",
    ]
)


class GoogleRoutesProvider(BaseProvider):
    name = "google_routes"

    def status(self) -> ProviderStatus:
        key = get_settings().google_maps_api_key
        return ProviderStatus(
            name=self.name,
            available=bool(key),
            configured=bool(key),
            message=(
                "GOOGLE_MAPS_API_KEY configured"
                if key
                else "GOOGLE_MAPS_API_KEY missing — Google Routes disabled"
            ),
        )

    def routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> List[NormalizedRoute]:
        if not get_settings().google_maps_api_key:
            return []
        modes = kwargs.get("modes") or ["DRIVE", "TWO_WHEELER", "WALK", "TRANSIT"]
        out: List[NormalizedRoute] = []
        for mode in modes:
            try:
                out.extend(self._compute(origin, destination, mode, kwargs))
            except PermissionError as exc:
                logger.error("Google auth error: %s", exc)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Google Routes %s failed: %s", mode, exc)
        return out

    def _compute(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        travel_mode: str,
        kwargs: Dict[str, Any],
    ) -> List[NormalizedRoute]:
        settings = get_settings()
        body: Dict[str, Any] = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin["latitude"],
                        "longitude": origin["longitude"],
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": destination["latitude"],
                        "longitude": destination["longitude"],
                    }
                }
            },
            "travelMode": travel_mode,
            "languageCode": "en-IN",
            "regionCode": "IN",
            "computeAlternativeRoutes": True,
        }
        if travel_mode in {"DRIVE", "TWO_WHEELER"}:
            body["routingPreference"] = "TRAFFIC_AWARE"
        if travel_mode == "TRANSIT":
            departure = kwargs.get("departure_time") or datetime.now(timezone.utc).isoformat()
            body["departureTime"] = departure
            prefs = kwargs.get("transit_preferences") or {}
            if prefs:
                body["transitPreferences"] = prefs

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": settings.google_maps_api_key or "",
            "X-Goog-FieldMask": FIELD_MASK,
        }
        payload = request_json(
            "POST",
            settings.google_routes_url,
            headers=headers,
            json_body=body,
        )
        routes = payload.get("routes") or []
        normalized: List[NormalizedRoute] = []
        for index, route in enumerate(routes):
            item = self._normalize_route(
                route, origin, destination, travel_mode, index
            )
            if item:
                normalized.append(item)
        return normalized

    def _normalize_route(
        self,
        route: Dict[str, Any],
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        travel_mode: str,
        index: int,
    ) -> Optional[NormalizedRoute]:
        duration_raw = route.get("duration") or "0s"
        duration_s = _parse_duration_seconds(duration_raw)
        distance_m = float(route.get("distanceMeters") or 0)
        if travel_mode == "WALK" and distance_m > get_settings().max_walk_only_meters:
            return None

        legs_out: List[NormalizedLeg] = []
        google_legs = route.get("legs") or [{}]
        transit_fare = None
        advisory = route.get("travelAdvisory") or {}
        if advisory.get("transitFare"):
            transit_fare = _money_to_inr(advisory["transitFare"])

        for gleg in google_legs:
            steps = gleg.get("steps") or []
            if not steps:
                mode = _mode_from_travel(travel_mode)
                fare = transit_fare
                if mode in {"car", "rapido"} and fare is None:
                    fare = None  # leave unavailable rather than invent Google fare
                legs_out.append(
                    NormalizedLeg(
                        mode=mode,
                        from_location=origin["name"],
                        to_location=destination["name"],
                        from_coordinates=(origin["latitude"], origin["longitude"]),
                        to_coordinates=(
                            destination["latitude"],
                            destination["longitude"],
                        ),
                        duration_minutes=duration_s / 60.0,
                        distance_meters=distance_m,
                        fare_inr=fare,
                        provider=self.name,
                        data_source="google_routes",
                        duration_status=DataStatus.LIVE,
                        fare_status=(
                            DataStatus.LIVE if fare is not None else DataStatus.UNAVAILABLE
                        ),
                        confidence=0.9,
                    )
                )
                continue

            for step in steps:
                step_mode = _mode_from_step(step, travel_mode)
                step_duration = _parse_duration_seconds(step.get("staticDuration") or step.get("duration") or "0s")
                step_distance = float(step.get("distanceMeters") or 0)
                start = step.get("startLocation", {}).get("latLng") or {}
                end = step.get("endLocation", {}).get("latLng") or {}
                transit = step.get("transitDetails") or {}
                stop_details = transit.get("stopDetails") or {}
                from_name = (
                    (stop_details.get("departureStop") or {}).get("name")
                    or origin["name"]
                )
                to_name = (
                    (stop_details.get("arrivalStop") or {}).get("name")
                    or destination["name"]
                )
                legs_out.append(
                    NormalizedLeg(
                        mode=step_mode,
                        from_location=from_name,
                        to_location=to_name,
                        from_coordinates=(
                            (start.get("latitude"), start.get("longitude"))
                            if start
                            else None
                        ),
                        to_coordinates=(
                            (end.get("latitude"), end.get("longitude")) if end else None
                        ),
                        duration_minutes=step_duration / 60.0,
                        distance_meters=step_distance,
                        fare_inr=None,
                        provider=self.name,
                        data_source="google_routes",
                        duration_status=DataStatus.LIVE,
                        fare_status=DataStatus.UNAVAILABLE,
                        confidence=0.9,
                        departure_time=(stop_details.get("departureTime")),
                        arrival_time=(stop_details.get("arrivalTime")),
                        metadata={"transit": bool(transit)},
                    )
                )

        if transit_fare is not None and legs_out:
            # Attach verified transit fare to first transit-like leg.
            for leg in legs_out:
                if leg.mode in {"bus", "metro", "train"}:
                    leg.fare_inr = transit_fare
                    leg.fare_status = DataStatus.LIVE
                    break
            else:
                legs_out[0].fare_inr = transit_fare
                legs_out[0].fare_status = DataStatus.LIVE

        label = {
            "DRIVE": "Driving (Google traffic-aware)",
            "TWO_WHEELER": "Two-wheeler (Google)",
            "WALK": "Walking (Google)",
            "TRANSIT": "Public transit (Google)",
        }.get(travel_mode, f"{travel_mode} (Google)")
        if index:
            label = f"{label} alt {index + 1}"

        return NormalizedRoute(
            name=label,
            legs=legs_out,
            provider=self.name,
            contains_simulated_data=False,
            data_sources=["google_routes"],
            polyline=(route.get("polyline") or {}).get("encodedPolyline"),
            metadata={
                "description": "Google Routes API computeRoutes result.",
                "comfort_index": 0.7 if travel_mode == "TRANSIT" else 0.75,
                "crowding_estimate": 0.55 if travel_mode == "TRANSIT" else 0.25,
                "confidence": 0.9,
                "data_freshness": "live_provider_response",
            },
        )


def _parse_duration_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "0s").strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _money_to_inr(money: Dict[str, Any]) -> Optional[float]:
    # Google Money: {currencyCode, units, nanos}
    if not money:
        return None
    code = str(money.get("currencyCode") or "").upper()
    units = float(money.get("units") or 0)
    nanos = float(money.get("nanos") or 0) / 1e9
    amount = units + nanos
    if code and code != "INR":
        # Do not invent FX conversion.
        return None
    return round(amount, 2)


def _mode_from_travel(travel_mode: str) -> str:
    return {
        "DRIVE": "car",
        "TWO_WHEELER": "rapido",
        "WALK": "walk",
        "BICYCLE": "bike",
        "TRANSIT": "bus",
    }.get(travel_mode, travel_mode.lower())


def _mode_from_step(step: Dict[str, Any], travel_mode: str) -> str:
    mode = str(step.get("travelMode") or travel_mode).upper()
    if mode == "WALK":
        return "walk"
    if mode == "TRANSIT":
        transit = step.get("transitDetails") or {}
        vehicle = (
            ((transit.get("transitLine") or {}).get("vehicle") or {}).get("type") or ""
        ).upper()
        if "METRO" in vehicle or "SUBWAY" in vehicle:
            return "metro"
        if "RAIL" in vehicle or "HEAVY" in vehicle or "COMMUTER" in vehicle:
            return "train"
        if "BUS" in vehicle:
            return "bus"
        return "bus"
    return _mode_from_travel(mode)
