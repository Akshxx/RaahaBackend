"""RAaHA mini ML route-preference prototype.

Synthetic training data is used only for model fitting. Live journey requests
are scored by the saved Joblib artifact and never depend on SRM→Phoenix
hardcoded outcomes.
"""

from app.ml.route_predictor import RoutePreferencePredictor, get_predictor

__all__ = ["RoutePreferencePredictor", "get_predictor"]
