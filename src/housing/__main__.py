"""CLI: one entrypoint for local runs, CI and production pipelines."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests

from housing import data
from housing.config import ROOT, load_settings
from housing.features import FEATURES, TARGET, validate
from housing.model import metrics
from housing.monitoring import drift_report
from housing.registry import CHAMPION, Registry
from housing.train import train


def cmd_data(settings, _):
    m = data.build(settings)
    print(f"gold {m.version}: {m.rows} rows (parent silver {m.parent})")


def cmd_train(settings, args):
    version = train(settings, Path(args.experiment), args.trigger)
    print(f"registered candidate v{version}")


def cmd_register(settings, args):
    """Register a run chosen by the data scientist (leaderboard `run_id` prefix is enough)."""
    registry = Registry(settings)
    runs = mlflow.search_runs(experiment_names=[settings.model_name])
    matches = runs.loc[runs["run_id"].str.startswith(args.run), "run_id"] if len(runs) else []
    if len(matches) != 1:
        raise RuntimeError(f"Run '{args.run}' matches {len(matches)} runs.")
    print(f"registered candidate v{registry.register(matches.iloc[0])}")


def cmd_gate(settings, args):
    version, passed, reason = Registry(settings).gate(args.version)
    print(f"v{version}: {'PASSED' if passed else 'REJECTED'} ({reason})")


def cmd_approve(settings, args):
    print(f"champion -> v{Registry(settings).promote(args.version)}")


def cmd_rollback(settings, _):
    print(f"champion -> v{Registry(settings).rollback()}")


def cmd_leaderboard(settings, _):
    print(Registry(settings).leaderboard().to_string(index=False))


def cmd_evaluate(settings, args):
    """Score the champion on an external labelled file, e.g. the held-out rows."""
    df = pd.read_csv(args.data)
    if errors := validate(df):
        raise SystemExit(f"Contract violated: {errors}")
    registry = Registry(settings)
    if (champion := registry.get(CHAMPION)) is None:
        raise RuntimeError("No champion to evaluate.")
    result = metrics(df[TARGET], registry.load(champion.version).predict(df[FEATURES]))
    print(f"champion v{champion.version} on {len(df)} rows: {result}")


def cmd_drift(settings, _):
    print(drift_report(settings).to_string(index=False))


def cmd_traffic(settings, args):
    """Replay Gold test rows through the API; `--shift` scales MedInc to simulate drift."""
    df, _ = data.load_gold(settings)
    rows = df[df["split"] == "test"][FEATURES].sample(args.n, random_state=settings.seed)
    rows = rows.assign(MedInc=np.clip(rows["MedInc"] * args.shift, 0, 20))
    for start in range(0, len(rows), 50):
        chunk = rows.iloc[start : start + 50].to_dict("records")
        resp = requests.post(f"{args.url}/predict", json={"instances": chunk}, timeout=10)
        resp.raise_for_status()
    print(f"sent {len(rows)} predictions (MedInc x{args.shift})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="housing")
    parser.add_argument("--env", choices=["dev", "prod"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="Build Bronze -> Silver -> Gold").set_defaults(fn=cmd_data)
    p = sub.add_parser("train", help="Run an experiment file, register best candidate")
    p.add_argument("--experiment", default=str(ROOT / "experiments" / "baseline.yaml"))
    p.add_argument("--trigger", default="manual", choices=["manual", "scheduled", "drift"])
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("register", help="Register any run as a candidate")
    p.add_argument("--run", required=True, help="run_id or prefix (see leaderboard)")
    p.set_defaults(fn=cmd_register)
    p = sub.add_parser("gate", help="Gate a version (default: latest candidate)")
    p.add_argument("--version")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("approve", help="Make champion: the challenger, or --version")
    p.add_argument("--version")
    p.set_defaults(fn=cmd_approve)
    sub.add_parser("rollback", help="Restore previous champion").set_defaults(fn=cmd_rollback)
    sub.add_parser("leaderboard", help="Compare all runs").set_defaults(fn=cmd_leaderboard)
    p = sub.add_parser("evaluate", help="Score champion on an external CSV")
    p.add_argument("--data", required=True)
    p.set_defaults(fn=cmd_evaluate)
    sub.add_parser("drift", help="PSI report from the prediction log").set_defaults(fn=cmd_drift)
    p = sub.add_parser("traffic", help="Replay test rows through the API")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--url", default="http://localhost:8000")
    p.set_defaults(fn=cmd_traffic)

    args = parser.parse_args()
    try:
        args.fn(load_settings(args.env), args)
    except RuntimeError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
