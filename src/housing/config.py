"""Environment settings: one YAML per environment, same code everywhere."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILES = {"dev": "settings_dev.yaml", "prod": "settings.yaml"}
PATHS = ("artifact_root", "bronze_path", "data_dir", "prediction_log")


@dataclass(frozen=True)
class Settings:
    env: str
    tracking_uri: str
    artifact_root: Path
    model_name: str
    bronze_path: Path
    data_dir: Path
    sample_fraction: float
    seed: int
    test_size: float
    cv_folds: int
    min_improvement: float
    auto_promote: bool
    refresh_seconds: int
    prediction_log: Path


def load_settings(env: str | None = None) -> Settings:
    """Settings for `env` (default: $ENV, else dev). $MLFLOW_TRACKING_URI overrides the store."""
    env = env or os.getenv("ENV", "dev")
    raw = yaml.safe_load((ROOT / "config" / SETTINGS_FILES[env]).read_text())
    raw |= {key: ROOT / raw[key] for key in PATHS}
    uri = os.getenv("MLFLOW_TRACKING_URI") or raw["tracking_uri"]
    raw["tracking_uri"] = uri.format(root=ROOT)
    return Settings(**raw)
