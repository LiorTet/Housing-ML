import numpy as np

from housing.features import FEATURES, TARGET
from housing.model import RunSpec, build_model
from housing.registry import passes


def test_log_target_predicts_in_original_units(bronze):
    model = build_model(RunSpec("lr_log", "linear", log_target=True), seed=42)
    model.fit(bronze[FEATURES], bronze[TARGET])
    preds = model.predict(bronze[FEATURES])
    assert abs(np.median(preds) - bronze[TARGET].median()) < 0.5


def test_preprocessing_is_part_of_the_model(bronze):
    model = build_model(
        RunSpec("rf", "random_forest", engineered=True, params={"n_estimators": 5}), seed=42
    )
    model.fit(bronze[FEATURES].head(500), bronze[TARGET].head(500))
    assert [name for name, _ in model.steps] == ["features", "scale", "model"]
    assert model.predict(bronze[FEATURES].head(3)).shape == (3,)


def test_passes():
    assert passes(0.50, None, 0.01)
    assert passes(0.49, 0.50, 0.01)
    assert not passes(0.499, 0.50, 0.01)
