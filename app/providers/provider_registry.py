"""Registry of transport data providers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.settings import get_settings
from app.providers.base_provider import BaseProvider, NormalizedRoute, ProviderStatus
from app.providers.chennai_gtfs_provider import ChennaiGTFSProvider
from app.providers.google_routes_provider import GoogleRoutesProvider
from app.providers.osrm_provider import OSRMProvider
from app.providers.railradar_provider import RailRadarProvider

logger = logging.getLogger("raaha.providers.registry")


class ProviderRegistry:
    def __init__(self) -> None:
        self.osrm = OSRMProvider()
        self.google = GoogleRoutesProvider()
        self.gtfs = ChennaiGTFSProvider()
        self.railradar = RailRadarProvider()
        self._providers: List[BaseProvider] = [
            self.osrm,
            self.google,
            self.gtfs,
            self.railradar,
        ]

    def status_summary(self) -> Dict[str, Any]:
        statuses = [p.status() for p in self._providers]
        return {
            "providers": [
                {
                    "name": s.name,
                    "available": s.available,
                    "configured": s.configured,
                    "message": s.message,
                    "details": s.details,
                }
                for s in statuses
            ],
            "settings": get_settings().provider_status(),
        }

    def collect_routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> tuple[List[NormalizedRoute], Dict[str, Any]]:
        routes: List[NormalizedRoute] = []
        report: Dict[str, Any] = {"calls": [], "errors": []}

        def _run(name: str, fn):
            try:
                result = fn()
                report["calls"].append(
                    {"provider": name, "ok": True, "count": len(result)}
                )
                routes.extend(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s failed: %s", name, exc)
                report["calls"].append({"provider": name, "ok": False, "error": str(exc)})
                report["errors"].append({"provider": name, "error": str(exc)})

        # Road / walk / bike from OSRM always attempted.
        _run("osrm", lambda: self.osrm.routes(origin, destination))

        # Google when configured.
        if get_settings().google_maps_api_key:
            _run(
                "google_routes",
                lambda: self.google.routes(
                    origin,
                    destination,
                    modes=["DRIVE", "TWO_WHEELER", "TRANSIT", "WALK"],
                    **kwargs,
                ),
            )
        else:
            report["calls"].append(
                {
                    "provider": "google_routes",
                    "ok": False,
                    "error": "GOOGLE_MAPS_API_KEY not configured",
                }
            )

        # GTFS static network.
        _run("chennai_gtfs", lambda: self.gtfs.routes(origin, destination, **kwargs))

        # RailRadar when configured.
        if get_settings().railradar_api_key:
            _run(
                "railradar",
                lambda: self.railradar.routes(origin, destination, **kwargs),
            )
        else:
            report["calls"].append(
                {
                    "provider": "railradar",
                    "ok": False,
                    "error": "RAILRADAR_API_KEY not configured",
                }
            )

        # Estimated last-mile auto / cab variants from OSRM car distance when present.
        _run(
            "derived_road_modes",
            lambda: _derive_road_modes(routes, origin, destination),
        )

        return routes, report


def _derive_road_modes(
    existing: List[NormalizedRoute],
    origin: Dict[str, Any],
    destination: Dict[str, Any],
) -> List[NormalizedRoute]:
    """Create labelled estimated auto/rapido options from an OSRM/Google drive leg."""
    from app.providers.base_provider import DataStatus, NormalizedLeg

    drive = None
    for route in existing:
        if len(route.legs) == 1 and route.legs[0].mode in {"car"}:
            drive = route.legs[0]
            break
    if not drive or not drive.distance_meters:
        return []

    distance_km = drive.distance_meters / 1000.0
    drive_min = float(drive.duration_minutes or max(10.0, distance_km * 2.2))
    out: List[NormalizedRoute] = []

    auto = NormalizedLeg(
        mode="auto",
        from_location=origin["name"],
        to_location=destination["name"],
        from_coordinates=(origin["latitude"], origin["longitude"]),
        to_coordinates=(destination["latitude"], destination["longitude"]),
        duration_minutes=drive_min * 1.15 + 5,
        distance_meters=drive.distance_meters * 1.05,
        fare_inr=round(30 + distance_km * 14, 2),
        waiting_minutes=6,
        provider="derived_estimate",
        data_source="derived_from_osrm_distance",
        duration_status=DataStatus.ESTIMATED,
        fare_status=DataStatus.ESTIMATED,
        confidence=0.55,
    )
    out.append(
        NormalizedRoute(
            name="Direct auto (estimated from road distance)",
            legs=[auto],
            provider="derived_estimate",
            contains_simulated_data=False,
            data_sources=["derived_from_osrm_distance"],
            metadata={
                "description": "Estimated auto using road distance. Not a live fare quote.",
                "comfort_index": 0.55,
                "crowding_estimate": 0.35,
                "confidence": 0.55,
            },
        )
    )

    rapido = NormalizedLeg(
        mode="rapido",
        from_location=origin["name"],
        to_location=destination["name"],
        from_coordinates=(origin["latitude"], origin["longitude"]),
        to_coordinates=(destination["latitude"], destination["longitude"]),
        duration_minutes=drive_min * 1.05 + 4,
        distance_meters=drive.distance_meters * 1.02,
        fare_inr=round(25 + distance_km * 10, 2),
        waiting_minutes=5,
        provider="derived_estimate",
        data_source="derived_from_osrm_distance",
        duration_status=DataStatus.ESTIMATED,
        fare_status=DataStatus.ESTIMATED,
        confidence=0.5,
    )
    out.append(
        NormalizedRoute(
            name="Two-wheeler taxi (estimated from road distance)",
            legs=[rapido],
            provider="derived_estimate",
            contains_simulated_data=False,
            data_sources=["derived_from_osrm_distance"],
            metadata={
                "description": "Estimated two-wheeler taxi. Not a live provider quote.",
                "comfort_index": 0.4,
                "crowding_estimate": 0.2,
                "confidence": 0.5,
            },
        )
    )
    return out


_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
