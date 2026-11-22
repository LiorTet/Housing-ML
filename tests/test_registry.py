import mlflow
import yaml

from housing import data
from housing.registry import CHALLENGER, CHAMPION, PREVIOUS, Registry
from housing.train import train


def test_lifecycle_choose_switch_rollback(settings, tmp_path):
    spec = tmp_path / "exp.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "runs": [
                    {"name": "rf", "model": "random_forest", "params": {"n_estimators": 20}},
                    {"name": "linear", "model": "linear"},
                ],
            }
        )
    )
    data.build(settings)
    registry = Registry(settings)

    best = train(settings, spec)
    assert registry.gate(best)[1] and registry.get(CHALLENGER).version == int(best)
    assert registry.promote() == best and registry.get(CHALLENGER) is None

    runs = mlflow.search_runs(experiment_names=[settings.model_name])
    other_run = runs.loc[runs["tags.mlflow.runName"] == "linear", "run_id"].item()
    other = registry.register(other_run)
    assert registry.register(other_run) == other
    assert not registry.gate()[1]

    assert registry.promote(other) == other
    assert registry.get(CHAMPION).version == int(other)
    assert registry.get(PREVIOUS).version == int(best)

    assert registry.rollback() == best
    assert registry.get(CHAMPION).version == int(best)
    assert registry.get(PREVIOUS).version == int(other)
    assert set(registry.leaderboard()["version"]) == {best, other}
