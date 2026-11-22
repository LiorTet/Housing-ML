"""Registry lifecycle: candidate -> challenger -> champion -> previous. Aliases are the switch."""

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from housing.config import Settings
from housing.data import load_gold, split
from housing.model import metrics

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"


def passes(candidate_rmse: float, champion_rmse: float | None, min_improvement: float) -> bool:
    """Candidate must beat the champion by a relative margin on the same test set."""
    return champion_rmse is None or candidate_rmse <= champion_rmse * (1 - min_improvement)


class Registry:
    def __init__(self, settings: Settings):
        mlflow.set_tracking_uri(settings.tracking_uri)
        self.settings = settings
        self.name = settings.model_name
        self.client = MlflowClient()

    def register(self, run_id: str) -> str:
        """Any tracked run can become a version (once); lineage tags are copied from the run."""
        existing = self.client.search_model_versions(f"name='{self.name}' and run_id='{run_id}'")
        if existing:
            return str(existing[0].version)
        run = self.client.get_run(run_id)
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        uri = f"models:/{run.outputs.model_outputs[0].model_id}"
        return str(mlflow.register_model(uri, self.name, tags=tags | {"run_id": run_id}).version)

    def get(self, alias: str) -> ModelVersion | None:
        try:
            return self.client.get_model_version_by_alias(self.name, alias)
        except MlflowException:
            return None

    def load(self, version: str):
        return mlflow.sklearn.load_model(f"models:/{self.name}/{version}")

    def gate(self, version: str | None = None) -> tuple[str, bool, str]:
        """Gate a version (default: latest ungated) on the current Gold test set (like-for-like)."""
        if version is None:
            versions = self.client.search_model_versions(f"name='{self.name}'")
            pending = [int(v.version) for v in versions if "gate" not in v.tags]
            if not pending:
                raise RuntimeError("No candidate to gate. Run `train` or `register` first.")
            version = str(max(pending))
        _, _, x_test, y_test = split(load_gold(self.settings)[0])

        def score(v: str) -> float:
            return metrics(y_test, self.load(v).predict(x_test))["rmse"]

        champion = self.get(CHAMPION)
        cand_rmse = score(version)
        champ_rmse = score(champion.version) if champion else None
        passed = passes(cand_rmse, champ_rmse, self.settings.min_improvement)
        reason = (
            "no champion"
            if champion is None
            else (
                f"rmse {cand_rmse:.4f} vs v{champion.version} {champ_rmse:.4f}, "
                f"required improvement {self.settings.min_improvement:.1%}"
            )
        )
        self.client.set_model_version_tag(
            self.name, version, "gate", f"{'passed' if passed else 'rejected'}: {reason}"
        )
        if passed:
            self.client.set_registered_model_alias(self.name, CHALLENGER, version)
            if self.settings.auto_promote:
                self.promote()
        return version, passed, reason

    def promote(self, version: str | None = None) -> str:
        """Challenger (or an explicitly chosen version) -> champion; old champion -> previous."""
        if version is None:
            if (challenger := self.get(CHALLENGER)) is None:
                raise RuntimeError("No challenger to promote. Pass a version to choose one.")
            version = str(challenger.version)
        else:
            self.client.get_model_version(self.name, version)
            self.client.set_model_version_tag(self.name, version, "promoted", "manual choice")
        champion = self.get(CHAMPION)
        if champion and str(champion.version) == version:
            return version
        if champion:
            self.client.set_registered_model_alias(self.name, PREVIOUS, champion.version)
        self.client.set_registered_model_alias(self.name, CHAMPION, version)
        if (challenger := self.get(CHALLENGER)) and str(challenger.version) == version:
            self.client.delete_registered_model_alias(self.name, CHALLENGER)
        return version

    def rollback(self) -> str:
        """Swap champion and previous."""
        previous, champion = self.get(PREVIOUS), self.get(CHAMPION)
        if previous is None:
            raise RuntimeError("No previous version to roll back to.")
        self.client.set_registered_model_alias(self.name, PREVIOUS, champion.version)
        self.client.set_registered_model_alias(self.name, CHAMPION, previous.version)
        return str(previous.version)

    def leaderboard(self) -> pd.DataFrame:
        """Every run, registered or not, with lineage, metrics, gate result and aliases."""
        runs = mlflow.search_runs(experiment_names=[self.name])
        if runs.empty:
            return runs
        versions = {v.run_id: v for v in self.client.search_model_versions(f"name='{self.name}'")}
        aliases: dict[str, list[str]] = {}
        for alias, version in self.client.get_registered_model(self.name).aliases.items():
            aliases.setdefault(str(version), []).append(alias)
        matched = [versions.get(run_id) for run_id in runs["run_id"]]
        board = pd.DataFrame(
            {
                "run_id": runs["run_id"].str[:8],
                "run": runs["tags.mlflow.runName"],
                "started": runs["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
                "sweep": runs["tags.sweep"],
                "gold_version": runs["tags.gold_version"],
                "cv_rmse": runs["metrics.cv_rmse_mean"].round(4),
                "test_rmse": runs["metrics.test_rmse"].round(4),
                "test_r2": runs["metrics.test_r2"].round(4),
                "version": [str(m.version) if m else "" for m in matched],
                "gate": [m.tags.get("gate", "").split(":")[0] if m else "" for m in matched],
            }
        )
        board["aliases"] = board["version"].map(lambda v: ",".join(aliases.get(v, [])))
        return board.sort_values(["gold_version", "cv_rmse"]).reset_index(drop=True)
