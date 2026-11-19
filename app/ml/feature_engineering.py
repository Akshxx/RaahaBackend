"""Feature extraction and preprocessing for route preference scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "duration_minutes",
    "cost_inr",
    "distance_km",
    "transfers",
    "walking_meters",
    "waiting_minutes",
    "mode_count",
    "comfort_index",
    "crowding_estimate",
    "route_complexity",
    "safety_proxy",
    "verified_provider_ratio",
    "paid_changes",
    "mode_hierarchy_score",
    "rupees_per_minute",
]

# Safe defaults when a provider omits a field.
_DEFAULTS: dict[str, float] = {
    "duration_minutes": 60.0,
    "cost_inr": 100.0,
    "distance_km": 10.0,
    "transfers": 0.0,
    "walking_meters": 300.0,
    "waiting_minutes": 5.0,
    "mode_count": 1.0,
    "comfort_index": 0.5,
    "crowding_estimate": 0.5,
    "route_complexity": 0.4,
    "safety_proxy": 0.5,
    "verified_provider_ratio": 0.5,
    "paid_changes": 0.0,
    "mode_hierarchy_score": 0.55,
    "rupees_per_minute": 5.0,
}

# Provisional prototype hierarchy for training features — not verified statistics.
_MODE_HIERARCHY = {
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


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(number) or np.isinf(number):
        return float(default)
    return number


def compute_mode_hierarchy_score(candidate: dict[str, Any]) -> float:
    modes = candidate.get("modes") or []
    if not modes:
        modes = [
            str(leg.get("mode", "")).lower()
            for leg in (candidate.get("legs") or [])
            if leg.get("mode")
        ]
    if not modes:
        return _DEFAULTS["mode_hierarchy_score"]
    transit = [str(mode).lower() for mode in modes if str(mode).lower() != "walk"] or [
        str(mode).lower() for mode in modes
    ]
    values = [_MODE_HIERARCHY.get(mode, 0.45) for mode in transit]
    return float(0.65 * min(values) + 0.35 * (sum(values) / len(values)))


def extract_feature_dict(candidate: dict[str, Any]) -> dict[str, float]:
    """Convert a candidate journey dict into model features."""
    # Derive totals from legs when present so multimodal costs are summed.
    legs = candidate.get("legs") or []
    if legs and candidate.get("estimated_cost_inr") is None and candidate.get("cost_inr") is None:
        candidate = {
            **candidate,
            "cost_inr": sum(float(leg.get("estimated_cost_inr") or 0) for leg in legs),
        }
    if candidate.get("mode_hierarchy_score") is None:
        candidate = {**candidate, "mode_hierarchy_score": compute_mode_hierarchy_score(candidate)}

    features: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        if column == "rupees_per_minute":
            continue
        features[column] = _as_float(
            candidate.get(column, candidate.get(f"estimated_{column}")),
            _DEFAULTS[column],
        )

    # Clamp noisy fields into sensible ranges for the prototype.
    features["duration_minutes"] = max(1.0, features["duration_minutes"])
    features["cost_inr"] = max(0.0, features["cost_inr"])
    features["distance_km"] = max(0.1, features["distance_km"])
    features["transfers"] = max(0.0, features["transfers"])
    features["walking_meters"] = max(0.0, features["walking_meters"])
    features["waiting_minutes"] = max(0.0, features["waiting_minutes"])
    features["mode_count"] = max(1.0, features["mode_count"])
    features["comfort_index"] = float(np.clip(features["comfort_index"], 0.0, 1.0))
    features["crowding_estimate"] = float(np.clip(features["crowding_estimate"], 0.0, 1.0))
    features["route_complexity"] = float(np.clip(features["route_complexity"], 0.0, 1.0))
    features["safety_proxy"] = float(np.clip(features["safety_proxy"], 0.0, 1.0))
    features["verified_provider_ratio"] = float(
        np.clip(features["verified_provider_ratio"], 0.0, 1.0)
    )
    features["paid_changes"] = max(0.0, features["paid_changes"])
    features["mode_hierarchy_score"] = float(
        np.clip(features.get("mode_hierarchy_score", _DEFAULTS["mode_hierarchy_score"]), 0.0, 1.0)
    )
    features["rupees_per_minute"] = float(
        features["cost_inr"] / max(features["duration_minutes"], 1.0)
    )
    return features


def features_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [extract_feature_dict(item) for item in candidates]
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return frame.fillna({column: _DEFAULTS[column] for column in FEATURE_COLUMNS})


def preference_targets_from_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic synthetic labels used only for training.

    These are simulated preference suitability scores in [0, 1], not real-world
    ground truth. Each preference emphasizes different feature trade-offs.
    """
    duration = frame["duration_minutes"].to_numpy(dtype=float)
    cost = frame["cost_inr"].to_numpy(dtype=float)
    walking = frame["walking_meters"].to_numpy(dtype=float)
    waiting = frame["waiting_minutes"].to_numpy(dtype=float)
    transfers = frame["transfers"].to_numpy(dtype=float)
    comfort = frame["comfort_index"].to_numpy(dtype=float)
    crowding = frame["crowding_estimate"].to_numpy(dtype=float)
    safety = frame["safety_proxy"].to_numpy(dtype=float)
    verified = frame["verified_provider_ratio"].to_numpy(dtype=float)
    paid = frame["paid_changes"].to_numpy(dtype=float)
    complexity = frame["route_complexity"].to_numpy(dtype=float)

    hierarchy = frame["mode_hierarchy_score"].to_numpy(dtype=float)
    rpm = frame["rupees_per_minute"].to_numpy(dtype=float)

    # Fastest is time-first but cost-aware (penalise extreme ₹/min).
    fastest = (
        1.20
        - 0.48 * _norm(duration, 15, 140)
        - 0.18 * _norm(waiting, 0, 25)
        - 0.10 * _norm(transfers, 0, 4)
        - 0.08 * complexity
        - 0.22 * _norm(rpm, 0, 25)
        - 0.08 * _norm(cost, 0, 800)
    )
    cheapest = (
        1.15
        - 0.70 * _norm(cost, 0, 700)
        - 0.15 * _norm(paid, 0, 3)
        - 0.08 * _norm(transfers, 0, 4)
        - 0.07 * _norm(walking, 0, 2500)
        - 0.05 * _norm(duration, 20, 160)
    )
    comfortable = (
        0.18
        + 0.30 * comfort
        + 0.18 * (1.0 - crowding)
        + 0.12 * hierarchy  # metro/train convenience signal
        - 0.22 * _norm(transfers, 0, 4)
        - 0.16 * _norm(walking, 0, 2500)
        - 0.08 * _norm(waiting, 0, 25)
        - 0.08 * _norm(duration, 15, 160)
    )
    safest = (
        0.10
        + 0.55 * hierarchy
        + 0.18 * verified
        + 0.10 * safety
        - 0.18 * _norm(walking, 0, 2500)
        - 0.08 * complexity
        - 0.05 * crowding
    )

    return pd.DataFrame(
        {
            "score_fastest": np.clip(fastest, 0.0, 1.0),
            "score_cheapest": np.clip(cheapest, 0.0, 1.0),
            "score_comfortable": np.clip(comfortable, 0.0, 1.0),
            "score_safest": np.clip(safest, 0.0, 1.0),
        }
    )


def _norm(values: np.ndarray, low: float, high: float) -> np.ndarray:
    span = max(high - low, 1e-6)
    return np.clip((values - low) / span, 0.0, 1.0)
