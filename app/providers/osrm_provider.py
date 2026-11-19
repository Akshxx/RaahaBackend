"""OSRM road / pedestrian / bicycle routing provider."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.settings import get_settings
from app.providers.base_provider import (
    BaseProvider,
    DataStatus,
    NormalizedLeg,
    NormalizedRoute,
    ProviderStatus,
)
from app.services.routing_service import get_bike_routes, get_driving_routes, get_walking_routes

logger = logging.getLogger("raaha.providers.osrm")

# Documented estimate bands used only when no live fare exists.
FARE_ESTIMATE = {
    "car": {"per_km": 18.0, "base": 40.0},
    "bike": {"per_km": 0.0, "base": 0.0},
    "walk": {"per_km": 0.0, "base": 0.0},
}


class OSRMProvider(BaseProvider):
    name = "osrm"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            configured=True,
            message="FOSSGIS OSRM via existing routing_service (curl)",
            details={"base_url": get_settings().osrm_base_url},
        )

    def routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> List[NormalizedRoute]:
        modes = kwargs.get("modes") or ["car", "bike", "walk"]
        out: List[NormalizedRoute] = []
        fetchers = {
            "car": get_driving_routes,
            "bike": get_bike_routes,
            "walk": get_walking_routes,
        }
        for mode in modes:
            fetcher = fetchers.get(mode)
            if not fetcher:
                continue
            try:
                raw_routes = fetcher(origin, destination)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OSRM %s failed: %s", mode, exc)
                continue
            for index, item in enumerate(raw_routes[:2]):
                route = self._from_routing_service(origin, destination, mode, item, index)
                if route:
                    out.append(route)
        return out

    def _from_routing_service(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        mode: str,
        item: Dict[str, Any],
        index: int,
    ) -> Optional[NormalizedRoute]:
        settings = get_settings()
        distance_m = float(item.get("distance_meters") or item.get("distance_km", 0) * 1000)
        duration_min = float(item.get("estimated_duration_minutes") or 0)
        if mode == "walk" and distance_m > settings.max_walk_only_meters:
            return None
        if mode == "bike" and distance_m > 18000:
            return None

        fare_cfg = FARE_ESTIMATE.get(mode, FARE_ESTIMATE["car"])
        fare = round(fare_cfg["base"] + (distance_m / 1000.0) * fare_cfg["per_km"], 2)
        if fare <= 0:
            fare = None

        display_mode = "car" if mode == "car" else mode
        name = {
            "car": "Driving / cab (OSRM)",
            "bike": "Cycling (OSRM)",
            "walk": "Walking (OSRM)",
        }.get(display_mode, f"{display_mode} (OSRM)")
        if index:
            name = f"{name} alt {index + 1}"

        leg = NormalizedLeg(
            mode=display_mode,
            from_location=origin["name"],
            to_location=destination["name"],
            from_coordinates=(origin["latitude"], origin["longitude"]),
            to_coordinates=(destination["latitude"], destination["longitude"]),
            duration_minutes=duration_min,
            distance_meters=distance_m,
            fare_inr=fare,
            waiting_minutes=2.0 if display_mode == "car" else 0.0,
            provider=self.name,
            data_source=str(item.get("data_source") or "osrm"),
            duration_status=DataStatus.LIVE,
            fare_status=DataStatus.ESTIMATED if fare is not None else DataStatus.UNAVAILABLE,
            confidence=0.85,
            instructions=f"OSRM {display_mode} geometry",
        )
        return NormalizedRoute(
            name=name,
            legs=[leg],
            provider=self.name,
            contains_simulated_data=False,
            data_sources=["osrm"],
            metadata={
                "description": "Road geometry from OSRM. Fare is an estimate when shown.",
                "comfort_index": {"car": 0.78, "bike": 0.42, "walk": 0.35}.get(
                    display_mode, 0.5
                ),
                "crowding_estimate": 0.2,
                "confidence": 0.85,
                "data_freshness": "live_routing_geometry",
            },
        )
