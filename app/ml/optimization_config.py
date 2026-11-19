"""Configurable trade-off thresholds for preference-specific ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class FastestConfig:
    """Time-first ranking with cost-efficiency guardrails."""

    # If a candidate is within this many minutes of the absolute fastest,
    # prefer materially cheaper options among that set.
    near_fastest_minutes: float = 15.0
    # Extra rupees allowed per minute saved vs the cheapest near-fast option.
    max_rupees_per_minute_saved: float = 18.0
    # Absolute cost premium above cheapest viable that still needs strong time gain.
    max_cost_premium_without_strong_gain: float = 200.0
    # Minutes that count as a "substantial" time advantage vs cheapest near option.
    substantial_time_advantage_minutes: float = 18.0
    # Soft ML blend weight after guardrail score is computed.
    ml_blend_weight: float = 0.25


@dataclass(frozen=True)
class CheapestConfig:
    """Cost-first ranking with practicality soft constraints."""

    max_practical_duration_multiplier: float = 2.2  # vs median duration
    max_practical_walking_meters: float = 2200.0
    extreme_duration_penalty: float = 0.25
    extreme_walking_penalty: float = 0.20
    ml_blend_weight: float = 0.30


@dataclass(frozen=True)
class ComfortableConfig:
    """Convenience-first ranking."""

    transfer_penalty: float = 0.18
    two_transfer_extra_penalty: float = 0.22
    three_plus_transfer_extra_penalty: float = 0.35
    walking_penalty_per_km: float = 0.22
    crowding_penalty: float = 0.22
    wait_penalty_per_10_min: float = 0.08
    duration_soft_penalty: float = 0.08
    ml_blend_weight: float = 0.28


@dataclass(frozen=True)
class DiversityConfig:
    """Prefer distinct routes across preferences when near-tied."""

    enabled: bool = True
    # Second-best may replace top if within this relative score gap.
    near_tie_ratio: float = 0.10
    # Absolute cost band for cheapest diversification (INR).
    cheapest_near_tie_inr: float = 25.0
    # Absolute ETA band for fastest diversification (minutes).
    fastest_near_tie_minutes: float = 8.0
    # Choose safest / cheapest first so comfort/fastest can diversify afterward.
    prefer_distinct_order: tuple = ("safest", "cheapest", "comfortable", "fastest")


@dataclass(frozen=True)
class SafestConfig:
    """Mode-hierarchy + exposure based safety ranking (prototype policy)."""

    # Provisional prototype hierarchy — NOT verified accident/crime statistics.
    mode_safety_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "metro": 0.95,
            "train": 0.92,
            "bus": 0.82,
            "car": 0.72,
            "cab": 0.72,
            "walk": 0.55,
            "auto": 0.48,
            "rapido": 0.32,
            "bike": 0.30,
        }
    )
    walking_exposure_penalty_per_km: float = 0.16
    road_exposure_modes: tuple = ("car", "cab", "auto", "rapido", "bike")
    road_exposure_penalty: float = 0.05
    # Keep ML as a light prior — hierarchy must dominate (synthetic labels).
    ml_blend_weight: float = 0.12
    disclaimer: str = (
        "Safety ranking uses a provisional transport-mode preference policy and "
        "simulated proxies. It is not verified crime or accident statistics."
    )


@dataclass(frozen=True)
class OptimizationConfig:
    fastest: FastestConfig = field(default_factory=FastestConfig)
    cheapest: CheapestConfig = field(default_factory=CheapestConfig)
    comfortable: ComfortableConfig = field(default_factory=ComfortableConfig)
    safest: SafestConfig = field(default_factory=SafestConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)


DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig()
