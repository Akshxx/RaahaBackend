"""Configurable geographical / walking feasibility thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilityConfig:
    # Preferred station interchange walking (metres).
    preferred_transfer_walk_meters: float = 500.0
    # Preferred last-mile walking (metres).
    preferred_last_mile_walk_meters: float = 1500.0
    # Hard reject for any single walking connection (metres).
    max_walk_leg_meters: float = 2000.0
    # Soft reject band: above preferred but below max → reject when enabled.
    # Keep False so approximate pedestrian inflation (haversine×1.25) near the
    # preferred limit does not drop otherwise feasible station access walks.
    reject_walk_above_preferred: bool = False
    # Maximum haversine km between consecutive non-walk legs' connection points
    # before requiring an explicit transfer mode (approximate when no coords).
    max_transfer_gap_km: float = 3.0
    # Absolute journey walking budget (sum of walk legs).
    max_total_walking_meters: float = 2500.0


DEFAULT_FEASIBILITY_CONFIG = FeasibilityConfig()
