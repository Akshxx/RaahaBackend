import logging

from fastapi import APIRouter, HTTPException

from app.core.settings import get_settings
from app.models.journey import (
    JourneyRequest,
    JourneyResponse,
    MLJourneyRequest,
    MLJourneyResponse,
)
from app.providers.provider_registry import get_registry
from app.services.journey_planner import plan_journey, plan_journey_ml

logger = logging.getLogger("raaha.api")
router = APIRouter()


@router.get("/health")
def api_health():
    settings = get_settings()
    return {
        "service": "raaha-api",
        "status": "healthy",
        "providers": settings.provider_status(),
    }


@router.get("/api/providers/status")
def providers_status():
    return get_registry().status_summary()


@router.post("/journeys/plan", response_model=JourneyResponse)
def create_journey(request: JourneyRequest):
    """Legacy journey planning endpoint."""
    try:
        return plan_journey(request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Legacy journey planning failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/journey/plan", response_model=MLJourneyResponse)
def create_ml_journey(request: MLJourneyRequest):
    """Provider-backed multimodal recommendations for the iOS client."""
    try:
        return plan_journey_ml(request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Journey planning failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
