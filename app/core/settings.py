"""Environment-backed settings for RAaHA providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    google_maps_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY") or None
    )
    railradar_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("RAILRADAR_API_KEY") or None
    )
    osrm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OSRM_BASE_URL", "https://routing.openstreetmap.de"
        )
    )
    nominatim_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"
        )
    )
    google_routes_url: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_ROUTES_URL",
            "https://routes.googleapis.com/directions/v2:computeRoutes",
        )
    )
    railradar_base_url: str = field(
        default_factory=lambda: os.getenv(
            "RAILRADAR_BASE_URL", "https://api.railradar.in"
        )
    )
    gtfs_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("GTFS_DATA_DIR", str(ROOT / "data" / "gtfs"))
        )
    )
    provider_timeout_seconds: float = field(
        default_factory=lambda: _float("PROVIDER_TIMEOUT_SECONDS", 20.0)
    )
    provider_max_retries: int = field(
        default_factory=lambda: _int("PROVIDER_MAX_RETRIES", 2)
    )
    cache_geocode_ttl_seconds: int = field(
        default_factory=lambda: _int("CACHE_GEOCODE_TTL_SECONDS", 86400)
    )
    cache_route_ttl_seconds: int = field(
        default_factory=lambda: _int("CACHE_ROUTE_TTL_SECONDS", 300)
    )
    max_transfer_walk_meters: float = field(
        default_factory=lambda: _float("MAX_TRANSFER_WALK_METERS", 800.0)
    )
    max_last_mile_walk_meters: float = field(
        default_factory=lambda: _float("MAX_LAST_MILE_WALK_METERS", 1500.0)
    )
    max_walk_only_meters: float = field(
        default_factory=lambda: _float("MAX_WALK_ONLY_METERS", 2000.0)
    )
    max_transfers: int = field(default_factory=lambda: _int("MAX_TRANSFERS", 3))
    max_route_duration_minutes: int = field(
        default_factory=lambda: _int("MAX_ROUTE_DURATION_MINUTES", 240)
    )
    enable_ml_assist: bool = field(
        default_factory=lambda: _bool("ENABLE_ML_ASSIST", True)
    )
    require_google_key: bool = field(
        default_factory=lambda: _bool("REQUIRE_GOOGLE_KEY", False)
    )

    def provider_status(self) -> Dict[str, Dict[str, object]]:
        gtfs_feeds = []
        if self.gtfs_data_dir.exists():
            for child in sorted(self.gtfs_data_dir.iterdir()):
                if child.is_dir() and (child / "stops.txt").exists():
                    gtfs_feeds.append(child.name)
        return {
            "google_routes": {
                "configured": bool(self.google_maps_api_key),
                "required": self.require_google_key,
            },
            "railradar": {
                "configured": bool(self.railradar_api_key),
            },
            "osrm": {
                "configured": True,
                "base_url": self.osrm_base_url,
            },
            "nominatim": {
                "configured": True,
                "base_url": self.nominatim_base_url,
            },
            "chennai_gtfs": {
                "configured": bool(gtfs_feeds),
                "feeds": gtfs_feeds,
                "data_dir": str(self.gtfs_data_dir),
            },
            "ml_assist": {
                "enabled": self.enable_ml_assist,
            },
        }

    def missing_credentials(self) -> List[str]:
        missing: List[str] = []
        if self.require_google_key and not self.google_maps_api_key:
            missing.append("GOOGLE_MAPS_API_KEY")
        return missing


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    load_dotenv(ROOT / ".env", override=True)
    _settings = Settings()
    return _settings
