from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TransportMode = Literal[
    "car",
    "bus",
    "bike",
    "train",
    "rapido",
    "auto",
    "metro",
    "walk",
]

Preference = Literal["fastest", "cheapest", "comfortable", "balanced"]


class JourneyRequest(BaseModel):
    """Backward-compatible request used by POST /journeys/plan."""

    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    preference: Preference = "balanced"


class MLJourneyRequest(BaseModel):
    """Primary request used by POST /api/journey/plan."""

    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    preferences: List[Literal["fastest", "cheapest", "comfortable"]] = Field(
        default_factory=lambda: ["cheapest", "comfortable", "fastest"]
    )
    departure_time: Optional[str] = None


class JourneyLeg(BaseModel):
    mode: TransportMode
    from_location: str
    to_location: str
    estimated_duration_minutes: int = Field(ge=0)
    estimated_cost_inr: float = Field(ge=0)
    data_source: str
    distance_km: Optional[float] = None
    distance_meters: Optional[float] = None
    waiting_minutes: Optional[float] = None
    provider: Optional[str] = None
    fare_status: Optional[str] = None
    duration_status: Optional[str] = None
    fare_unavailable: Optional[bool] = None
    confidence: Optional[float] = None
    instructions: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None


class JourneyOption(BaseModel):
    name: str
    legs: List[JourneyLeg]
    estimated_duration_minutes: int = Field(ge=0)
    estimated_cost_inr: float = Field(ge=0)
    transfers: int = Field(ge=0)
    description: str
    modes: List[str] = Field(default_factory=list)
    walking_meters: Optional[float] = None
    waiting_minutes: Optional[float] = None
    distance_km: Optional[float] = None
    suitability_score: Optional[float] = None
    preference_scores: Optional[Dict[str, float]] = None
    contains_simulated_data: bool = False
    data_sources: List[str] = Field(default_factory=list)
    feature_snapshot: Optional[Dict[str, float]] = None
    recommendation_reason: Optional[str] = None
    ranking_note: Optional[str] = None
    ranking_method: Optional[str] = None
    provider: Optional[str] = None
    fare_unavailable: Optional[bool] = None
    fare_partial: Optional[bool] = None
    confidence: Optional[float] = None
    data_freshness: Optional[str] = None
    preference: Optional[str] = None


class JourneyResponse(BaseModel):
    """Legacy response shape preserved for existing clients."""

    origin: str
    destination: str
    preference: str
    routes: List[JourneyOption]
    data_source: str
    model_used: bool = False
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    recommendations: Optional[Dict[str, JourneyOption]] = None
    candidate_routes: Optional[List[JourneyOption]] = None
    data_quality: Optional[Dict[str, Any]] = None
    provider_status: Optional[Dict[str, Any]] = None
    status: str = "success"


class MLJourneyResponse(BaseModel):
    status: str = "success"
    model_used: bool = True
    model_name: str
    model_version: str
    origin: str
    destination: str
    origin_coordinates: Dict[str, float]
    destination_coordinates: Dict[str, float]
    recommendations: Dict[str, JourneyOption]
    candidate_routes: List[JourneyOption]
    data_quality: Dict[str, Any]
    provider_status: Optional[Dict[str, Any]] = None
    ranking_layer: Optional[str] = None
    ml_assist_enabled: Optional[bool] = None
