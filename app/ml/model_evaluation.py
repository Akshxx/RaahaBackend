"""Evaluation helpers for the route preference prototype."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Compute MAE / RMSE / R² per target column."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    names = list(target_names)
    results: dict[str, dict[str, float]] = {}

    for index, name in enumerate(names):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        rmse = float(np.sqrt(mean_squared_error(truth, pred)))
        results[name] = {
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": rmse,
            "r2": float(r2_score(truth, pred)),
        }
    return results
