"""Tests for the RAaHA ML route preference prototype."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml.data_generator import generate_synthetic_training_data
from app.ml.feature_engineering import FEATURE_COLUMNS, extract_feature_dict, features_frame
from app.ml.route_predictor import RoutePreferencePredictor
from app.ml.train_model import train_and_save
from app.services.candidate_generator import generate_candidates


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "route_preference_model.joblib"
    metrics_path = model_dir / "training_metrics.json"
    train_and_save(n_samples=800, model_path=model_path, metrics_path=metrics_path, seed=7)
    assert model_path.exists()
    return model_path


def test_model_loads_successfully(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    assert predictor.is_loaded
    assert predictor.model_name


def test_model_accepts_dynamic_features(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    candidates = [
        {
            "duration_minutes": 40,
            "cost_inr": 120,
            "distance_km": 18,
            "transfers": 1,
            "walking_meters": 400,
            "waiting_minutes": 6,
            "mode_count": 2,
            "comfort_index": 0.6,
            "crowding_estimate": 0.4,
            "route_complexity": 0.3,
            "safety_proxy": 0.7,
            "verified_provider_ratio": 0.8,
            "paid_changes": 1,
            "name": "A",
            "legs": [],
            "estimated_duration_minutes": 40,
            "estimated_cost_inr": 120,
            "description": "test",
            "modes": ["bus"],
        }
    ]
    scores = predictor.predict_scores(candidates)
    assert len(scores) == 1
    assert 0.0 <= scores[0]["score_fastest"] <= 1.0


def test_missing_features_do_not_crash(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    sparse = [{"name": "sparse", "legs": [], "estimated_duration_minutes": 30, "estimated_cost_inr": 50, "description": "", "modes": ["car"]}]
    frame = features_frame(sparse)
    assert list(frame.columns) == FEATURE_COLUMNS
    scores = predictor.predict_scores(sparse)
    assert len(scores) == 1


def test_different_characteristics_change_ranking(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    fast = {
        "name": "fast",
        "legs": [],
        "estimated_duration_minutes": 25,
        "estimated_cost_inr": 300,
        "description": "",
        "modes": ["car"],
        "duration_minutes": 25,
        "cost_inr": 300,
        "distance_km": 20,
        "transfers": 0,
        "walking_meters": 50,
        "waiting_minutes": 2,
        "mode_count": 1,
        "comfort_index": 0.8,
        "crowding_estimate": 0.2,
        "route_complexity": 0.3,
        "safety_proxy": 0.7,
        "verified_provider_ratio": 0.8,
        "paid_changes": 0,
    }
    cheap = {
        **fast,
        "name": "cheap",
        "duration_minutes": 70,
        "estimated_duration_minutes": 70,
        "cost_inr": 20,
        "estimated_cost_inr": 20,
        "transfers": 2,
        "walking_meters": 900,
        "waiting_minutes": 15,
        "comfort_index": 0.35,
        "crowding_estimate": 0.8,
        "paid_changes": 2,
    }
    result = predictor.rank_recommendations([fast, cheap])
    assert result["recommendations"]["fastest"]["name"] == "fast"
    assert result["recommendations"]["cheapest"]["name"] == "cheap"


def test_four_preferences_return_candidates(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    dataset = generate_synthetic_training_data(n_samples=12, seed=3)
    candidates = []
    for index, row in dataset.iterrows():
        candidates.append(
            {
                "name": f"c{index}",
                "legs": [],
                "estimated_duration_minutes": int(row["duration_minutes"]),
                "estimated_cost_inr": float(row["cost_inr"]),
                "description": "synthetic",
                "modes": ["bus"],
                **{column: float(row[column]) for column in FEATURE_COLUMNS},
            }
        )
    ranked = predictor.rank_recommendations(candidates)
    for key in ("fastest", "cheapest", "comfortable"):
        assert key in ranked["recommendations"]
        assert ranked["recommendations"][key]["suitability_score"] >= 0
    assert "safest" not in ranked["recommendations"]


def test_candidate_generator_with_coords():
    origin = {
        "name": "Origin",
        "latitude": 12.8231,
        "longitude": 80.0454,
    }
    destination = {
        "name": "Destination",
        "latitude": 12.9916,
        "longitude": 80.2170,
    }
    candidates, quality = generate_candidates(origin, destination)
    assert candidates
    assert "real_data_available" in quality
    assert "simulated_data_used" in quality
    # Features extractable even if OSRM failed and fallback simulation was used.
    snapshot = extract_feature_dict(candidates[0])
    assert snapshot["duration_minutes"] > 0


def test_training_metrics_are_finite(trained_model: Path):
    predictor = RoutePreferencePredictor(model_path=trained_model)
    predictor.load()
    metrics = predictor._artifact["metrics"]["test"]
    for values in metrics.values():
        assert np.isfinite(values["mae"])
        assert np.isfinite(values["rmse"])
        assert np.isfinite(values["r2"])
