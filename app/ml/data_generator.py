"""Synthetic journey scenario generator for offline model training.

All rows are simulated. They intentionally cover diverse distances, modes,
costs, walking distances and comfort/safety proxies — not a fixed SRM→Phoenix
corridor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.feature_engineering import (
    FEATURE_COLUMNS,
    _MODE_HIERARCHY,
    preference_targets_from_features,
)

_PROFILE_PRIMARY_MODE = {
    "car": "car",
    "bike": "bike",
    "walk": "walk",
    "auto": "auto",
    "rapido": "rapido",
    "bus": "bus",
    "train": "train",
    "metro": "metro",
    "multimodal_budget": "bus",
    "multimodal_comfort": "metro",
}

MODE_PROFILES: dict[str, dict[str, float]] = {
    "car": {
        "speed_kmh": 28.0,
        "cost_per_km": 18.0,
        "base_cost": 40.0,
        "walk_share": 0.02,
        "wait": 3.0,
        "transfers": 0.0,
        "comfort": 0.82,
        "crowding": 0.25,
        "complexity": 0.35,
        "safety": 0.78,
        "verified": 0.85,
        "paid_changes": 0.0,
        "modes": 1.0,
    },
    "bike": {
        "speed_kmh": 16.0,
        "cost_per_km": 0.0,
        "base_cost": 0.0,
        "walk_share": 0.01,
        "wait": 1.0,
        "transfers": 0.0,
        "comfort": 0.45,
        "crowding": 0.15,
        "complexity": 0.55,
        "safety": 0.42,
        "verified": 0.30,
        "paid_changes": 0.0,
        "modes": 1.0,
    },
    "walk": {
        "speed_kmh": 4.5,
        "cost_per_km": 0.0,
        "base_cost": 0.0,
        "walk_share": 1.0,
        "wait": 0.0,
        "transfers": 0.0,
        "comfort": 0.35,
        "crowding": 0.20,
        "complexity": 0.40,
        "safety": 0.48,
        "verified": 0.20,
        "paid_changes": 0.0,
        "modes": 1.0,
    },
    "auto": {
        "speed_kmh": 22.0,
        "cost_per_km": 14.0,
        "base_cost": 30.0,
        "walk_share": 0.04,
        "wait": 5.0,
        "transfers": 0.0,
        "comfort": 0.55,
        "crowding": 0.35,
        "complexity": 0.45,
        "safety": 0.60,
        "verified": 0.55,
        "paid_changes": 0.0,
        "modes": 1.0,
    },
    "rapido": {
        "speed_kmh": 24.0,
        "cost_per_km": 10.0,
        "base_cost": 25.0,
        "walk_share": 0.03,
        "wait": 4.0,
        "transfers": 0.0,
        "comfort": 0.40,
        "crowding": 0.20,
        "complexity": 0.50,
        "safety": 0.50,
        "verified": 0.70,
        "paid_changes": 0.0,
        "modes": 1.0,
    },
    "bus": {
        "speed_kmh": 16.0,
        "cost_per_km": 1.2,
        "base_cost": 8.0,
        "walk_share": 0.12,
        "wait": 10.0,
        "transfers": 1.0,
        "comfort": 0.38,
        "crowding": 0.75,
        "complexity": 0.55,
        "safety": 0.58,
        "verified": 0.80,
        "paid_changes": 1.0,
        "modes": 2.0,
    },
    "train": {
        "speed_kmh": 32.0,
        "cost_per_km": 0.8,
        "base_cost": 10.0,
        "walk_share": 0.15,
        "wait": 8.0,
        "transfers": 1.0,
        "comfort": 0.42,
        "crowding": 0.80,
        "complexity": 0.50,
        "safety": 0.65,
        "verified": 0.90,
        "paid_changes": 1.0,
        "modes": 2.0,
    },
    "metro": {
        "speed_kmh": 34.0,
        "cost_per_km": 2.0,
        "base_cost": 20.0,
        "walk_share": 0.14,
        "wait": 6.0,
        "transfers": 1.0,
        "comfort": 0.72,
        "crowding": 0.60,
        "complexity": 0.35,
        "safety": 0.82,
        "verified": 0.95,
        "paid_changes": 1.0,
        "modes": 2.0,
    },
    "multimodal_budget": {
        "speed_kmh": 20.0,
        "cost_per_km": 1.5,
        "base_cost": 20.0,
        "walk_share": 0.18,
        "wait": 12.0,
        "transfers": 2.0,
        "comfort": 0.40,
        "crowding": 0.70,
        "complexity": 0.65,
        "safety": 0.62,
        "verified": 0.75,
        "paid_changes": 2.0,
        "modes": 3.0,
    },
    "multimodal_comfort": {
        "speed_kmh": 24.0,
        "cost_per_km": 8.0,
        "base_cost": 80.0,
        "walk_share": 0.08,
        "wait": 7.0,
        "transfers": 1.0,
        "comfort": 0.78,
        "crowding": 0.40,
        "complexity": 0.40,
        "safety": 0.74,
        "verified": 0.88,
        "paid_changes": 1.0,
        "modes": 2.0,
    },
}


def generate_synthetic_training_data(
    n_samples: int = 2400,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a diverse simulated training set for the preference model."""
    rng = np.random.default_rng(seed)
    profiles = list(MODE_PROFILES.items())
    rows: list[dict[str, float]] = []

    for _ in range(n_samples):
        name, profile = profiles[int(rng.integers(0, len(profiles)))]
        distance_km = float(rng.uniform(1.5, 45.0))
        noise = float(rng.normal(0.0, 0.08))

        duration = (distance_km / profile["speed_kmh"]) * 60.0
        duration *= float(rng.uniform(0.85, 1.25))
        duration += profile["wait"] * float(rng.uniform(0.6, 1.4))

        cost = profile["base_cost"] + distance_km * profile["cost_per_km"]
        cost *= float(rng.uniform(0.8, 1.25))

        walking = distance_km * 1000.0 * profile["walk_share"]
        walking *= float(rng.uniform(0.7, 1.4))

        transfers = max(0.0, profile["transfers"] + float(rng.integers(-1, 2)))
        waiting = max(0.0, profile["wait"] * float(rng.uniform(0.5, 1.6)))

        duration = max(5.0, duration)
        cost = max(0.0, cost)
        primary = _PROFILE_PRIMARY_MODE.get(name, "bus")
        hierarchy = float(_MODE_HIERARCHY.get(primary, 0.55))
        if name == "multimodal_budget":
            hierarchy = 0.65 * min(_MODE_HIERARCHY["bus"], _MODE_HIERARCHY["train"]) + 0.35 * (
                (_MODE_HIERARCHY["bus"] + _MODE_HIERARCHY["train"] + _MODE_HIERARCHY["walk"]) / 3
            )
        if name == "multimodal_comfort":
            hierarchy = 0.65 * min(_MODE_HIERARCHY["metro"], _MODE_HIERARCHY["auto"]) + 0.35 * (
                (_MODE_HIERARCHY["metro"] + _MODE_HIERARCHY["auto"]) / 2
            )

        row = {
            "duration_minutes": duration,
            "cost_inr": cost,
            "distance_km": distance_km,
            "transfers": transfers,
            "walking_meters": max(0.0, walking),
            "waiting_minutes": waiting,
            "mode_count": max(1.0, profile["modes"] + float(rng.integers(0, 2))),
            "comfort_index": float(np.clip(profile["comfort"] + noise, 0.05, 0.98)),
            "crowding_estimate": float(np.clip(profile["crowding"] + noise * 0.5, 0.05, 0.98)),
            "route_complexity": float(np.clip(profile["complexity"] + noise * 0.4, 0.05, 0.98)),
            "safety_proxy": float(np.clip(profile["safety"] + noise * 0.3, 0.05, 0.98)),
            "verified_provider_ratio": float(
                np.clip(profile["verified"] + noise * 0.2, 0.05, 0.98)
            ),
            "paid_changes": max(0.0, profile["paid_changes"] + float(rng.integers(0, 2))),
            "mode_hierarchy_score": float(np.clip(hierarchy + noise * 0.05, 0.05, 0.98)),
            "rupees_per_minute": float(cost / max(duration, 1.0)),
            "profile_name": name,
            "data_origin": "synthetic_simulated",
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    feature_frame = frame[FEATURE_COLUMNS].copy()
    targets = preference_targets_from_features(feature_frame)
    # Mild label noise so the forest learns robust patterns rather than exact formulas.
    for column in targets.columns:
        jitter = rng.normal(0.0, 0.03, size=len(targets))
        targets[column] = np.clip(targets[column].to_numpy() + jitter, 0.0, 1.0)

    return pd.concat([frame, targets], axis=1)
