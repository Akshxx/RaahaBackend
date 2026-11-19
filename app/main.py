import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.settings import get_settings
from app.ml.route_predictor import get_predictor
from app.providers.provider_registry import get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("raaha")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    missing = settings.missing_credentials()
    if missing:
        logger.error("Missing required credentials: %s", missing)
    status = settings.provider_status()
    logger.info("Provider configuration: %s", status)

    # Warm GTFS indexes (may take time on first boot).
    try:
        registry = get_registry()
        gtfs_status = registry.gtfs.status()
        logger.info("GTFS status: %s", gtfs_status.message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GTFS warm-up failed: %s", exc)

    predictor = get_predictor()
    predictor.load()  # optional experimental model
    logger.info(
        "Startup ready deterministic_ranker=yes ml_loaded=%s",
        predictor.is_loaded,
    )
    yield


app = FastAPI(
    title="RAaHA API",
    description=(
        "Multimodal journey planning with provider-backed candidates and "
        "deterministic Cheapest / Comfortable / Fastest ranking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    predictor = get_predictor()
    settings = get_settings()
    return {
        "message": "Welcome to RAaHA",
        "status": "running",
        "version": "2.0.0",
        "ranking": ["cheapest", "comfortable", "fastest"],
        "model_loaded": predictor.is_loaded,
        "model_name": predictor.model_name if predictor.is_loaded else "RAaHA Deterministic Ranker",
        "model_version": predictor.model_version if predictor.is_loaded else "deterministic-1.0",
        "providers": settings.provider_status(),
        "docs": "/docs",
    }
