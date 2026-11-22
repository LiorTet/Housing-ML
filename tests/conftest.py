from pathlib import Path

import pandas as pd
import pytest

from housing.config import ROOT, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "artifacts",
        model_name="housing-test",
        bronze_path=ROOT / "data" / "bronze" / "housing.csv",
        data_dir=tmp_path / "data",
        sample_fraction=0.1,
        seed=42,
        test_size=0.2,
        cv_folds=2,
        min_improvement=0.01,
        auto_promote=False,
        refresh_seconds=0,
        prediction_log=tmp_path / "predictions.jsonl",
    )


@pytest.fixture
def bronze() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "bronze" / "housing.csv")
