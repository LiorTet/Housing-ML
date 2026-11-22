"""Config-driven experiments, fully tracked. The best run by CV becomes a registry candidate."""

from dataclasses import asdict
from pathlib import Path

import mlflow
import yaml
from mlflow.models import infer_signature
from sklearn.model_selection import KFold, cross_val_score

from housing.config import Settings
from housing.data import Manifest, git_sha, load_gold, split
from housing.model import TRUSTED_TYPES, RunSpec, build_model, metrics
from housing.registry import Registry


def load_specs(path: Path) -> tuple[str, list[RunSpec]]:
    raw = yaml.safe_load(path.read_text())
    return raw["name"], [RunSpec(**run) for run in raw["runs"]]


def _experiment(settings: Settings) -> None:
    """One experiment per registered model; artifacts go to the env's artifact store."""
    if mlflow.get_experiment_by_name(settings.model_name) is None:
        mlflow.create_experiment(
            settings.model_name, artifact_location=settings.artifact_root.as_uri()
        )
    mlflow.set_experiment(settings.model_name)


def _run(spec: RunSpec, settings: Settings, data: tuple, gold: Manifest, tags: dict) -> tuple:
    x_train, y_train, x_test, y_test = data
    model = build_model(spec, settings.seed)
    with mlflow.start_run(run_name=spec.name) as run:
        mlflow.set_tags(tags | {"family": spec.model})
        mlflow.log_params(
            {k: v for k, v in asdict(spec).items() if k != "params"}
            | spec.params
            | {"seed": settings.seed}
        )
        mlflow.log_dict(asdict(gold), "gold_manifest.json")
        mlflow.log_input(mlflow.data.from_pandas(x_train, name=f"gold-{gold.version}"), "train")

        cv = KFold(settings.cv_folds, shuffle=True, random_state=settings.seed)
        cv_rmse = -cross_val_score(
            model, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        model.fit(x_train, y_train)
        test = {f"test_{k}": v for k, v in metrics(y_test, model.predict(x_test)).items()}
        mlflow.log_metrics({"cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std()} | test)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_train.head(3),
            skops_trusted_types=TRUSTED_TYPES,
            signature=infer_signature(x_train, model.predict(x_train.head(3))),
        )
        return float(cv_rmse.mean()), run.info.run_id


def train(settings: Settings, experiment_file: Path, trigger: str = "manual") -> str:
    """Run every spec; register the best by CV (never by test) as a candidate."""
    sweep, specs = load_specs(experiment_file)
    df, gold = load_gold(settings)
    registry = Registry(settings)
    _experiment(settings)
    tags = {
        "sweep": sweep,
        "gold_version": gold.version,
        "git_sha": git_sha(),
        "trigger": trigger,
        "env": settings.env,
    }
    results = [_run(spec, settings, split(df), gold, tags) for spec in specs]
    return registry.register(min(results)[1])
