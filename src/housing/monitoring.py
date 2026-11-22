"""Drift of served inputs and predictions against the Gold version the model was trained on."""

import json

import mlflow
import numpy as np
import pandas as pd

from housing.config import Settings
from housing.features import FEATURES, TARGET
from housing.registry import Registry

PSI_WARN, PSI_ALERT = 0.1, 0.25
MIN_SAMPLES = 50


def psi(edges: list[float], values: np.ndarray) -> float:
    """Population Stability Index over the reference decile bins."""
    inner = np.asarray(edges[1:-1])
    expected = np.full(len(inner) + 1, 1 / (len(inner) + 1))
    actual = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(expected))
    actual = np.clip(actual / max(len(values), 1), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def read_log(settings: Settings) -> pd.DataFrame:
    path = settings.prediction_log
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return pd.json_normalize(records)


def drift_report(settings: Settings, window: int = 1000) -> pd.DataFrame:
    """PSI of the last `window` predictions of the current model vs its training Gold."""
    log = read_log(settings).tail(window)
    if len(log) < MIN_SAMPLES:
        raise RuntimeError(f"Need at least {MIN_SAMPLES} logged predictions, found {len(log)}.")
    latest = log[log["model_version"] == log["model_version"].iloc[-1]]
    version = latest["model_version"].iloc[-1]
    # Reference travels with the model (logged at train time), not with local files.
    run_id = Registry(settings).client.get_model_version(settings.model_name, version).run_id
    reference = mlflow.artifacts.load_dict(f"runs:/{run_id}/gold_manifest.json")["stats"]

    signals = {f: latest[f"inputs.{f}"].to_numpy() for f in FEATURES}
    signals["prediction"] = latest["prediction"].to_numpy()
    rows = []
    for name, values in signals.items():
        value = psi(reference[TARGET if name == "prediction" else name], values)
        rows.append({"signal": name, "psi": round(value, 4), "status": status(value)})
    return pd.DataFrame(rows).assign(
        model_version=version, gold_version=latest["gold_version"].iloc[-1], samples=len(latest)
    )
