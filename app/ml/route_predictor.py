"""Inference wrapper: deterministic ranking + optional experimental ML assist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np

from app.core.settings import get_settings
from app.ml.deterministic_ranker import PREFERENCES as DETERMINISTIC_PREFERENCES
from app.ml.deterministic_ranker import rank_all as deterministic_rank_all
from app.ml.feature_engineering import FEATURE_COLUMNS, extract_feature_dict, features_frame
from app.ml.train_model import (
    DEFAULT_MODEL_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    TARGET_COLUMNS,
)

logger = logging.getLogger("raaha.ml")

PREFERENCE_KEYS = ("cheapest", "comfortable", "fastest")
_SCORE_KEY = {
    "fastest": "score_fastest",
    "cheapest": "score_cheapest",
    "comfortable": "score_comfortable",
    "safest": "score_safest",  # legacy column in older models; ignored in ranking
}


class RoutePreferencePredictor:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._artifact: Optional[dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def model_name(self) -> str:
        if self._artifact:
            return str(self._artifact.get("model_name", MODEL_NAME))
        return "RAaHA Deterministic Ranker"

    @property
    def model_version(self) -> str:
        if self._artifact:
            return str(self._artifact.get("model_version", MODEL_VERSION))
        return "deterministic-1.0"

    def load(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                "Optional ML model not found at %s — deterministic ranking only",
                self.model_path,
            )
            self._artifact = None
            return
        self._artifact = joblib.load(self.model_path)
        logger.info(
            "Experimental ML model loaded name=%s version=%s path=%s",
            self.model_name,
            self.model_version,
            self.model_path,
        )

    def ensure_loaded(self) -> None:
        if self._artifact is None and self.model_path.exists():
            self.load()

    def predict_scores(self, candidates: List[dict[str, Any]]) -> List[dict[str, float]]:
        if not candidates:
            return []
        if self._artifact is None:
            # Neutral experimental scores — deterministic layer decides.
            return [
                {
                    "score_fastest": 0.5,
                    "score_cheapest": 0.5,
                    "score_comfortable": 0.5,
                    "score_safest": 0.5,
                }
                for _ in candidates
            ]

        frame = features_frame(candidates)
        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.5 if "score" in column or "index" in column else 0.0
        pipeline = self._artifact["pipeline"]
        expected = list(self._artifact.get("feature_columns") or FEATURE_COLUMNS)
        for column in expected:
            if column not in frame.columns:
                frame[column] = 0.0
        predictions = np.asarray(pipeline.predict(frame[expected]), dtype=float)
        predictions = np.clip(predictions, 0.0, 1.0)
        scored: List[dict[str, float]] = []
        for row in predictions:
            scored.append(
                {target: float(value) for target, value in zip(TARGET_COLUMNS, row)}
            )
        return scored

    def rank_recommendations(
        self,
        candidates: List[dict[str, Any]],
        preferences: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        preferences = preferences or list(PREFERENCE_KEYS)
        preferences = [item for item in preferences if item in PREFERENCE_KEYS]
        if not preferences:
            preferences = list(PREFERENCE_KEYS)

        settings = get_settings()
        ml_enabled = settings.enable_ml_assist and (
            self.is_loaded or self.model_path.exists()
        )
        if ml_enabled and not self.is_loaded:
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML assist unavailable: %s", exc)
                ml_enabled = False

        scores = self.predict_scores(candidates) if ml_enabled else [
            {
                "score_fastest": 0.5,
                "score_cheapest": 0.5,
                "score_comfortable": 0.5,
                "score_safest": 0.5,
            }
            for _ in candidates
        ]

        enriched: List[dict[str, Any]] = []
        for candidate, score_row in zip(candidates, scores):
            item = dict(candidate)
            item["preference_scores"] = {
                key: round(float(score_row.get(_SCORE_KEY[key], 0.5)), 4)
                for key in PREFERENCE_KEYS
            }
            item["feature_snapshot"] = extract_feature_dict(candidate)
            legs = item.get("legs") or []
            if legs and item.get("estimated_cost_inr") is None:
                item["estimated_cost_inr"] = round(
                    sum(float(leg.get("estimated_cost_inr") or 0) for leg in legs),
                    2,
                )
            enriched.append(item)

        recommendations = deterministic_rank_all(enriched, preferences=preferences)
        for preference, winner in recommendations.items():
            logger.info(
                "Ranked %s → candidate=%s score=%.4f reason=%s",
                preference,
                winner.get("name"),
                float(winner.get("suitability_score") or 0),
                (winner.get("recommendation_reason") or "")[:140],
            )

        return {
            "recommendations": recommendations,
            "candidate_routes": enriched,
            "model_name": self.model_name if ml_enabled and self.is_loaded else "RAaHA Deterministic Ranker",
            "model_version": (
                self.model_version if ml_enabled and self.is_loaded else "deterministic-1.0"
            ),
            "ml_assist_enabled": bool(ml_enabled and self.is_loaded),
            "ranking_layer": "deterministic_primary",
            "supported_preferences": list(DETERMINISTIC_PREFERENCES),
        }


_predictor: Optional[RoutePreferencePredictor] = None


def get_predictor() -> RoutePreferencePredictor:
    global _predictor
    if _predictor is None:
        _predictor = RoutePreferencePredictor()
    return _predictor
