# Housing Value: ML System PoC

Data as source code, everything tracked, models managed as a lifecycle. Small PoC, production seams.

Architecture and decisions: [docs/architecture.md](docs/architecture.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and `make`. `make install` = `uv sync --locked`: Python from `.python-version`, exact versions from `uv.lock`, same on every laptop, CI runner and image.

```bash
make install             # core + dev group + ui extra
make pipeline            # dev: Bronze -> Silver -> Gold, train, gate (auto-promotes in dev)
make serve               # API on :8000, serves the `champion` alias
make ui                  # Streamlit on :8501
make mlflow              # MLflow UI on :5000
uv run python -m housing --help   # the CLI behind every make target
```

Prod uses the same code with `config/settings.yaml`: full data, stricter gate, manual approval.

```bash
make pipeline ENV=prod   # candidate -> challenger
make approve ENV=prod    # challenger -> champion
```

## How the pieces talk

```
make / CLI ──► data/ (Bronze → Silver → Gold + manifests)
     │                         │
     └── train, gate, approve ─┴──► MLflow (runs, registry, aliases)
                                          │ champion
UI ──► API (FastAPI) ◄────────────────────┘
        │
        └──► prediction log ──► drift (PSI vs the model's Gold stats, read from its run)
```

## Lifecycle in commands

| Step | Command | What happens |
|---|---|---|
| Data | `make data` | Content-hashed Bronze/Silver/Gold with manifests and lineage |
| Experiment | `make train EXPERIMENT=experiments/gbm_tuning.yaml` | Every run tracked; best by CV registered as candidate |
| Compare | `make leaderboard` | All runs per Gold version, with gate result and aliases |
| Choose | `make register RUN=<run_id>` | Any run becomes a candidate version, lineage tags copied |
| Gate | `make gate [VERSION=n]` | Candidate vs champion on the same test set → challenger or rejected |
| Switch | `make approve [VERSION=n]` / `make rollback` | Moves aliases; the API picks it up without redeploy |
| Monitor | `make traffic SHIFT=1.6 && make drift` | PSI of inputs and predictions vs the model's Gold |
| Hold-out | `make evaluate DATA=path.csv` | Champion on any labelled file |

## Choose and switch

The best run by CV is registered automatically. A data scientist can serve another one:

```bash
make leaderboard                 # pick a run_id
make register RUN=3f2a91bc       # -> candidate vN
make gate VERSION=N              # evidence: vs champion, same test set (tagged on the version)
make approve VERSION=N           # serve it, even if the gate rejected it (tagged "manual choice")
make rollback                    # previous champion back
```

The API polls the `champion` alias: no restart. `/health` and every response show the served version.

## Layout

```
pyproject.toml   deps: core (serving), ui extra, dev and analysis groups
uv.lock          exact dependency versions; .python-version pins Python
Makefile         one entrypoint for laptop, CI and CD
Dockerfile       serving image: uv sync --frozen --no-dev
reference/       original notebook, as received
config/          settings.yaml (prod), settings_dev.yaml (dev)
analysis/        data profile notebook: findings -> contract decisions
experiments/     one YAML per idea: models, params, feature and target options
data/bronze/     source as received (immutable)
src/housing/
  features.py    data contract + feature catalog, shared by training and serving
  data.py        Bronze -> Silver -> Gold, manifests, split
  model.py       pipeline definition and metrics (pure ML)
  train.py       experiment runs, CV, MLflow tracking
  registry.py    registration, gate, aliases, rollback, leaderboard
  api.py         FastAPI, champion polling, prediction log
  monitoring.py  PSI drift
  ui.py          minimal test UI (public API only)
  __main__.py    CLI used by make, CI and pipelines
tests/           data contract, determinism, model, registry lifecycle, API, drift
.github/         CI (lint, tests, smoke pipeline) and CD (prod train, gated approval)
```

## Data profile

[`analysis/data_profile.ipynb`](analysis/data_profile.ipynb) (`make profile`), each finding mapped to a decision:

- Target capped at 5.00001 (4.4%), `HouseAge` at 52 (7%): caps encoded in the contract; an uncapped source fails Silver.
- Ratio outliers (`AveOccup` up to 600) are real block groups: kept; the contract rejects impossible values only.
- Spatial CV RMSE ~0.60 vs ~0.45 random: random split kept (matches the hold-out); unseen areas are the known risk.

## Starting point: the notebook

| Finding | Effect | Fix |
|---|---|---|
| Scaler fit before the split | Test leakage | Scaler inside the model pipeline, fit on train only |
| Scaler never saved | Model unservable | Preprocessing ships in the artifact |
| Log target, log-space metrics | Not comparable across models | `TransformedTargetRegressor`; metrics in $100k |
| `y_test` transformed in place | Re-running a cell corrupts it | Pure functions, immutable Gold |
| Features added to some models only, in place | Train/serve skew | One feature module, a flag per experiment |
| One `model.pkl`, overwritten metrics | No history | MLflow runs + registry versions |
| Hardcoded `/Users/john/...` | Not reproducible | Settings per environment |

## Baseline results (prod Gold `a72d62555789`, 16,512 rows)

| Run | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| gradient_boosting (engineered) | 0.446 | 0.453 | 0.848 |
| random_forest | 0.487 | 0.491 | 0.822 |
| linear | 0.745 | 0.747 | 0.588 |

RMSE in $100k. Selection by CV; the test set is only used by the gate.

## PoC limits

Local SQLite MLflow (`MLFLOW_TRACKING_URI` points CD at a shared server), JSONL prediction log, no auth, no shadow traffic, no label ingestion. The Docker image needs `MLFLOW_TRACKING_URI` to reach a registry; not run locally. Each one has a production counterpart in the architecture, behind the same interface.
