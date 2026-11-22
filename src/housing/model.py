"""Model definition and metrics. Pure ML: no tracking, no I/O."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from housing.features import add_features

ESTIMATORS = {
    "linear": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": HistGradientBoostingRegressor,
}
# skops (safe) serialization only loads back types listed here.
TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "housing.features.add_features",
]


@dataclass(frozen=True)
class RunSpec:
    name: str
    model: str
    params: dict = field(default_factory=dict)
    log_target: bool = False
    engineered: bool = False


def build_model(spec: RunSpec, seed: int):
    """Preprocessing lives inside the artifact: fit on train only, shipped to serving."""
    estimator = ESTIMATORS[spec.model](**spec.params)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    features = [("features", FunctionTransformer(add_features))] if spec.engineered else []
    model = Pipeline([*features, ("scale", StandardScaler()), ("model", estimator)])
    if spec.log_target:
        return TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
    return model


def metrics(y_true, y_pred) -> dict[str, float]:
    """Always in target units ($100k), whatever transform the model uses internally."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
