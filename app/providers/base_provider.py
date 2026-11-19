"""Normalized provider contracts for multimodal journey planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DataStatus(str, Enum):
    LIVE = "live"
    SCHEDULED = "scheduled"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    CACHED = "cached"
    SIMULATED = "simulated"


@dataclass
class ProviderStatus:
    name: str
    available: bool
    configured: bool
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedLeg:
    mode: str
    from_location: str
    to_location: str
    from_coordinates: Optional[Tuple[float, float]] = None
    to_coordinates: Optional[Tuple[float, float]] = None
    duration_minutes: Optional[float] = None
    distance_meters: Optional[float] = None
    fare_inr: Optional[float] = None
    waiting_minutes: float = 0.0
    provider: str = ""
    data_source: str = ""
    duration_status: DataStatus = DataStatus.ESTIMATED
    fare_status: DataStatus = DataStatus.ESTIMATED
    confidence: float = 0.5
    instructions: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    polyline: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        distance_km = None
        if self.distance_meters is not None:
            distance_km = round(self.distance_meters / 1000.0, 3)
        fare = self.fare_inr
        fare_unavailable = fare is None
        return {
            "mode": self.mode,
            "from_location": self.from_location,
            "to_location": self.to_location,
            "from_coordinates": self.from_coordinates,
            "to_coordinates": self.to_coordinates,
            "estimated_duration_minutes": int(round(self.duration_minutes or 0)),
            "estimated_cost_inr": float(fare if fare is not None else 0.0),
            "fare_unavailable": fare_unavailable,
            "fare_status": self.fare_status.value,
            "duration_status": self.duration_status.value,
            "waiting_minutes": float(self.waiting_minutes),
            "distance_meters": self.distance_meters,
            "distance_km": distance_km,
            "data_source": self.data_source or self.provider,
            "provider": self.provider,
            "confidence": self.confidence,
            "instructions": self.instructions,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
            "polyline": self.polyline,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedRoute:
    name: str
    legs: List[NormalizedLeg]
    provider: str
    contains_simulated_data: bool = False
    data_sources: List[str] = field(default_factory=list)
    polyline: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_candidate(self) -> Dict[str, Any]:
        leg_dicts = [leg.to_dict() for leg in self.legs]
        duration = sum(float(leg.get("estimated_duration_minutes") or 0) for leg in leg_dicts)
        waiting = sum(float(leg.get("waiting_minutes") or 0) for leg in leg_dicts)
        duration += waiting
        fares = [
            float(leg["estimated_cost_inr"])
            for leg in leg_dicts
            if not leg.get("fare_unavailable")
        ]
        any_fare_missing = any(leg.get("fare_unavailable") for leg in leg_dicts)
        cost = round(sum(fares), 2)
        walking = sum(
            float(leg.get("distance_meters") or 0)
            for leg in leg_dicts
            if str(leg.get("mode", "")).lower() == "walk"
        )
        modes = [str(leg.get("mode", "")).lower() for leg in leg_dicts]
        non_walk = [m for m in modes if m != "walk"]
        transfers = max(0, len(non_walk) - 1)
        distance_km = round(
            sum(float(leg.get("distance_km") or 0) for leg in leg_dicts), 3
        )
        sources = list(self.data_sources) or [self.provider]
        return {
            "name": self.name,
            "modes": modes,
            "legs": leg_dicts,
            "estimated_duration_minutes": int(round(duration)),
            "estimated_cost_inr": cost,
            "fare_unavailable": any_fare_missing and not fares,
            "fare_partial": any_fare_missing and bool(fares),
            "transfers": transfers,
            "description": self.metadata.get("description")
            or f"Provider route from {self.provider}",
            "duration_minutes": float(duration),
            "cost_inr": cost,
            "distance_km": distance_km,
            "walking_meters": walking,
            "waiting_minutes": float(waiting),
            "mode_count": float(len(set(modes))),
            "comfort_index": float(self.metadata.get("comfort_index", 0.55)),
            "crowding_estimate": float(self.metadata.get("crowding_estimate", 0.5)),
            "route_complexity": min(0.95, 0.2 + 0.15 * transfers + walking / 5000.0),
            "contains_simulated_data": self.contains_simulated_data,
            "data_sources": sources,
            "provider": self.provider,
            "polyline": self.polyline,
            "retrieval_notes": self.metadata.get("retrieval_notes"),
            "data_freshness": self.metadata.get("data_freshness"),
            "confidence": float(self.metadata.get("confidence", 0.6)),
        }


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def status(self) -> ProviderStatus:
        raise NotImplementedError

    def nearby_stops(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1200.0,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        return []

    def routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> List[NormalizedRoute]:
        return []
