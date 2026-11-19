"""RailRadar API provider — trains between stations / station search."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from app.core.settings import get_settings
from app.providers.base_provider import (
    BaseProvider,
    DataStatus,
    NormalizedLeg,
    NormalizedRoute,
    ProviderStatus,
)
from app.providers.http_utils import request_json

logger = logging.getLogger("raaha.providers.railradar")

# Common Chennai / SRM corridor station codes used when nearby coords match.
KNOWN_STATIONS: List[Dict[str, Any]] = [
    {"code": "POT", "name": "Potheri", "lat": 12.8250, "lon": 80.0455},
    {"code": "GI", "name": "Guduvancheri", "lat": 12.8450, "lon": 80.0600},
    {"code": "TBM", "name": "Tambaram", "lat": 12.9249, "lon": 80.1000},
    {"code": "MS", "name": "Chennai Egmore", "lat": 13.0690, "lon": 80.2610},
    {"code": "MAS", "name": "Chennai Central", "lat": 13.0827, "lon": 80.2707},
    {"code": "MSB", "name": "Chennai Beach", "lat": 13.0923, "lon": 80.2919},
    {"code": "VLCY", "name": "Velachery", "lat": 12.9758, "lon": 80.2209},
    {"code": "MAA", "name": "Chennai Airport / Tirusulam area", "lat": 12.9941, "lon": 80.1709},
]


class RailRadarProvider(BaseProvider):
    name = "railradar"

    def status(self) -> ProviderStatus:
        key = get_settings().railradar_api_key
        return ProviderStatus(
            name=self.name,
            available=bool(key),
            configured=bool(key),
            message=(
                "RAILRADAR_API_KEY configured"
                if key
                else "RAILRADAR_API_KEY missing — RailRadar disabled"
            ),
        )

    def nearby_stations(
        self, lat: float, lon: float, radius_km: float = 8.0
    ) -> List[Dict[str, Any]]:
        from app.services.route_feasibility import haversine_meters

        found = []
        for station in KNOWN_STATIONS:
            dist = haversine_meters(lat, lon, station["lat"], station["lon"])
            if dist <= radius_km * 1000:
                item = dict(station)
                item["distance_m"] = dist
                found.append(item)
        found.sort(key=lambda s: s["distance_m"])
        return found

    def search_station(self, query: str) -> List[Dict[str, Any]]:
        settings = get_settings()
        if not settings.railradar_api_key:
            return []
        url = f"{settings.railradar_base_url.rstrip('/')}/v1/search/stations"
        payload = request_json(
            "GET",
            url,
            headers=self._headers(),
            params={"q": query},
        )
        data = payload.get("data") or payload.get("stations") or []
        return data if isinstance(data, list) else []

    def trains_between(
        self,
        from_code: str,
        to_code: str,
        travel_date: Optional[str] = None,
        local_only: bool = True,
    ) -> Dict[str, Any]:
        settings = get_settings()
        if not settings.railradar_api_key:
            return {"success": False, "error": "missing_key", "data": {}}
        params: Dict[str, Any] = {}
        if travel_date:
            params["date"] = travel_date
        if local_only:
            params["category"] = "Local"
        url = (
            f"{settings.railradar_base_url.rstrip('/')}/v1/trains/between/"
            f"{from_code}/{to_code}"
        )
        return request_json("GET", url, headers=self._headers(), params=params)

    def routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> List[NormalizedRoute]:
        if not get_settings().railradar_api_key:
            return []
        origin_stations = self.nearby_stations(
            origin["latitude"], origin["longitude"]
        )
        dest_stations = self.nearby_stations(
            destination["latitude"], destination["longitude"]
        )
        if not origin_stations or not dest_stations:
            return []

        travel_date = kwargs.get("travel_date") or date.today().isoformat()
        out: List[NormalizedRoute] = []
        tried: set = set()
        for src in origin_stations[:3]:
            for dst in dest_stations[:3]:
                if src["code"] == dst["code"]:
                    continue
                pair = (src["code"], dst["code"])
                if pair in tried:
                    continue
                tried.add(pair)
                try:
                    payload = self.trains_between(
                        src["code"], dst["code"], travel_date=travel_date
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RailRadar between %s-%s failed: %s",
                        src["code"],
                        dst["code"],
                        exc,
                    )
                    continue
                trains = _extract_trains(payload)
                if not trains:
                    continue
                train = trains[0]
                duration = _train_duration_minutes(train)
                if duration is None:
                    duration = max(20.0, abs(src["distance_m"] - dst["distance_m"]) / 500.0)
                legs: List[NormalizedLeg] = []
                # Access walk/auto estimate to station — marked estimated.
                if src["distance_m"] > 80:
                    legs.append(
                        NormalizedLeg(
                            mode="walk" if src["distance_m"] <= 900 else "auto",
                            from_location=origin["name"],
                            to_location=src["name"],
                            from_coordinates=(origin["latitude"], origin["longitude"]),
                            to_coordinates=(src["lat"], src["lon"]),
                            duration_minutes=max(4.0, src["distance_m"] / 80.0),
                            distance_meters=src["distance_m"],
                            fare_inr=0.0 if src["distance_m"] <= 900 else round(
                                30 + src["distance_m"] / 1000 * 14, 2
                            ),
                            provider=self.name,
                            data_source="railradar+access_estimate",
                            duration_status=DataStatus.ESTIMATED,
                            fare_status=DataStatus.ESTIMATED,
                            confidence=0.55,
                        )
                    )
                legs.append(
                    NormalizedLeg(
                        mode="train",
                        from_location=src["name"],
                        to_location=dst["name"],
                        from_coordinates=(src["lat"], src["lon"]),
                        to_coordinates=(dst["lat"], dst["lon"]),
                        duration_minutes=duration,
                        distance_meters=None,
                        fare_inr=None,  # do not invent IRCTC fare
                        waiting_minutes=8.0,
                        provider=self.name,
                        data_source="railradar",
                        duration_status=DataStatus.SCHEDULED,
                        fare_status=DataStatus.UNAVAILABLE,
                        confidence=0.8,
                        departure_time=str(train.get("departure") or train.get("dep") or ""),
                        arrival_time=str(train.get("arrival") or train.get("arr") or ""),
                        metadata={"train": train.get("number") or train.get("trainNumber")},
                    )
                )
                if dst["distance_m"] > 80:
                    legs.append(
                        NormalizedLeg(
                            mode="walk" if dst["distance_m"] <= 900 else "auto",
                            from_location=dst["name"],
                            to_location=destination["name"],
                            from_coordinates=(dst["lat"], dst["lon"]),
                            to_coordinates=(
                                destination["latitude"],
                                destination["longitude"],
                            ),
                            duration_minutes=max(4.0, dst["distance_m"] / 80.0),
                            distance_meters=dst["distance_m"],
                            fare_inr=0.0 if dst["distance_m"] <= 900 else round(
                                30 + dst["distance_m"] / 1000 * 14, 2
                            ),
                            provider=self.name,
                            data_source="railradar+access_estimate",
                            duration_status=DataStatus.ESTIMATED,
                            fare_status=DataStatus.ESTIMATED,
                            confidence=0.55,
                        )
                    )
                train_no = train.get("number") or train.get("trainNumber") or "service"
                out.append(
                    NormalizedRoute(
                        name=f"RailRadar train {train_no} ({src['code']}→{dst['code']})",
                        legs=legs,
                        provider=self.name,
                        contains_simulated_data=False,
                        data_sources=["railradar"],
                        metadata={
                            "description": (
                                "RailRadar trains-between result. Train fare marked "
                                "unavailable unless a verified fare source is attached."
                            ),
                            "comfort_index": 0.62,
                            "crowding_estimate": 0.7,
                            "confidence": 0.75,
                            "data_freshness": "scheduled_api",
                        },
                    )
                )
                if len(out) >= 3:
                    return out
        return out

    def _headers(self) -> Dict[str, str]:
        key = get_settings().railradar_api_key or ""
        return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _extract_trains(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") or {}
    if isinstance(data, list):
        return data
    for key in ("trains", "results", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _train_duration_minutes(train: Dict[str, Any]) -> Optional[float]:
    for key in ("durationMinutes", "duration_min", "journeyTimeMinutes"):
        if train.get(key) is not None:
            try:
                return float(train[key])
            except (TypeError, ValueError):
                pass
    # duration like "02:15"
    for key in ("duration", "travelTime", "journeyTime"):
        raw = train.get(key)
        if isinstance(raw, str) and ":" in raw:
            parts = raw.split(":")
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                continue
    return None
