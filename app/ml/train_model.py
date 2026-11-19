"""Train and persist the RAaHA route preference Random Forest model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.data_generator import generate_synthetic_training_data
from app.ml.feature_engineering import FEATURE_COLUMNS
from app.ml.model_evaluation import evaluate_predictions

MODEL_NAME = "RAaHA Route Preference Model"
MODEL_VERSION = "1.1.0-prototype"
TARGET_COLUMNS = [
    "score_fastest",
    "score_cheapest",
    "score_comfortable",
    "score_safest",
]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "route_preference_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "training_metrics.json"


def build_pipeline(random_state: int = 42) -> Pipeline:
    regressor = RandomForestRegressor(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(regressor)),
        ]
    )


def train_and_save(
    n_samples: int = 2400,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    seed: int = 42,
) -> dict:
    model_path = model_path or DEFAULT_MODEL_PATH
    metrics_path = metrics_path or DEFAULT_METRICS_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_training_data(n_samples=n_samples, seed=seed)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
    )

    pipeline = build_pipeline(random_state=seed)
    pipeline.fit(x_train, y_train)

    train_pred = pipeline.predict(x_train)
    test_pred = pipeline.predict(x_test)
    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestRegressor (MultiOutput)",
        "n_samples": int(n_samples),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "data_note": (
            "Training used synthetic/simulated journey scenarios only. "
            "Metrics describe prototype fit quality, not real-world accuracy."
        ),
        "train": evaluate_predictions(y_train.to_numpy(), train_pred, TARGET_COLUMNS),
        "test": evaluate_predictions(y_test.to_numpy(), test_pred, TARGET_COLUMNS),
    }

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "metrics": metrics,
    }
    joblib.dump(artifact, model_path)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")
    print(
        "Test MAE (mean across targets): "
        f"{np.mean([m['mae'] for m in metrics['test'].values()]):.4f}"
    )
    return metrics


if __name__ == "__main__":
    train_and_save()
