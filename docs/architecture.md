# Architecture

## Design stance

Each principle answers a failure found in the original notebook.

| Principle | Notebook evidence |
|---|---|
| **Data is source code**: versioned, immutable, with lineage. | Hardcoded local path; no way to know which data trained `model.pkl`. |
| **Track everything**: a result without code, data and config can't be reproduced. | Metrics overwritten per cell; three models share one `model.pkl`. |
| **Preprocessing belongs to the model.** | Scaler fit before the split (leakage) and never saved (unservable). |
| **Design for the lifecycle, not the model.** | One-off script; no path to compare, replace or roll back. |

## DAM view

| Pillar | Owns | PoC | Production (GCP) |
|---|---|---|---|
| **Data** | Layers, versions, features, feedback | Bronze/Silver/Gold + manifests | BigQuery + Dataform, Gold snapshots |
| **Algorithm** | Experiments, evaluation, registry | YAML configs + MLflow | Vertex AI Pipelines + MLflow |
| **Machine** | Serving, logging, infra | FastAPI + Dockerfile | Cloud Run, Pub/Sub, Terraform |

## Lifecycle

```
┌─────────────── DATA ───────────────┐                   ┌──────────── ALGORITHM ─────────────┐
│ Sources → Bronze → Silver → Gold   │  train on Gold  ─►│ Experiments → Registry → champion  │
└────────────────────────────────────┘                   └────────────────────────────────────┘
                  ▲                                                        │
    logs + labels │                                                        │ serve champion
    as new source │                                                        ▼
┌───────────── FEEDBACK ─────────────┐                   ┌───────────── MACHINE ──────────────┐
│ Monitoring: quality, drift, RMSE   │◄─  prediction log │ API → Consumers / UI               │
│ Labels joined on request_id        │                   │ Prediction log                     │
└────────────────────────────────────┘                   └────────────────────────────────────┘
```

The loop closes through Data: feedback becomes a new source, not a side channel. Feedback also triggers retraining (→ Algorithm).

## 1. Data

| Layer | Content | Rule |
|---|---|---|
| Bronze | Sources as received | Immutable. |
| Silver | Typed, validated, cleaned | Contract enforced; violations stop the pipeline. |
| Gold | Features + train/test split | Versioned. Only input to training. |

- **Manifest** per layer: content hash, rows, schema, parent hashes, git SHA, feature statistics.
- Parent hashes give **lineage**. Feature statistics are the **drift reference** for every model trained on that Gold.
- **Feature catalog** (shared ontology): each feature has entity, definition, unit, owner, version. One code module computes features for training and serving.

**Change paths**

| Change | Touches | Result |
|---|---|---|
| New feature | Feature module + catalog + test | New Gold version → experiments |
| New source | Bronze ingest + Silver contract + join | New Gold version, lineage extended |
| Feedback | Prediction log + labels as Bronze source | New Gold version → retraining |
| Feature not in request | Feature store lookup at serve time | Online/offline parity required |

Every change ends in a new Gold version. Nothing reaches training any other way.

## 2. Experimentation

- **Run** logs: git SHA, Gold version, config, params, metrics, artifact, environment.
- **Metrics in target units** ($100k): a model may train on `log(y)`, but it's evaluated after inverting the transform. The notebook compared log-space errors with raw-space ones.
- **Benchmark:** a frozen Gold test set. Runs compare only within the same Gold version. When Gold changes, the champion is re-scored so history stays comparable.
- **Leaderboard:** champion and candidates per Gold version, kept historically. PoC: `make leaderboard`.

**New model idea**

| Stage | Where | Automated |
|---|---|---|
| Explore | MLflow experiment per idea, dev | Logging only |
| Candidate | Best run per sweep, or any run chosen by the DS | Yes / `register` |
| Challenger | Passes gate | Yes |
| Champion | Serves traffic | Approval in prod |

Preliminary results need no new path: a new experiment lands on the same leaderboard as the champion.

## 3. Registry

- **Registered model = contract**, not algorithm: `housing-value`. Any architecture can be a version; consumers depend on input/output only.
- **Version** = one run (code + Gold + config). Immutable. Retraining creates a new version.
- **Tags:** `family` (linear, rf, gbm), `gold_version`, `trigger` (manual, scheduled, drift), `env`.
- **Aliases:** `champion`, `challenger`, `previous`.
- **States:** candidate (ungated) → challenger → champion → previous. Gate result tagged on the version.
- **Contract change** (new input or output) = new registered model + new API version. Never silent.
- **Model card** per version: data, metrics, slices, limitations.

## 4. Environments and promotion

**Promote code, retrain in prod.**

| | Dev | Prod |
|---|---|---|
| Data | Sample / dev copy | Prod |
| Pipeline | Same image | Same image |
| Purpose | Validate code | Produce serving models |
| Registry | Dev | Prod |

- Prod data stays in prod. Continuous retraining runs there anyway.
- Cost: training twice, a few cents here.
- Risk: the prod model never ran in dev. Mitigated by the prod gate and shadow traffic.
- Promoting artifacts suits expensive training (large GPU models). Not this case.

**Gate** (automatic): beats champion on primary metric by ≥ δ (PoC); no slice regression beyond tolerance, same signature, latency within budget (prod).

A DS can override the selection: any version can be made champion explicitly (`approve --version`), tagged `manual choice`. Prod still requires approval.

| Step | Trigger | Mode | PoC |
|---|---|---|---|
| Lint, format, unit + contract + registry tests, training smoke | PR | Auto | Yes |
| Build image, deploy + run dev pipeline | PR / branch | Auto | Smoke only |
| Deploy pipeline to prod | Merge to `main` | Auto | Train in CD |
| Train, select candidate, gate → `challenger` | Schedule / Gold change / drift | Auto | Yes |
| Shadow: challenger scores live traffic, not returned | Challenger set | Auto | No |
| `challenger` → `champion` | Shadow OK | Manual approval (automate once trusted) | Yes |
| Rollback: `previous` → `champion` | SLO breach | One command | Yes |

## 5. Serving

- `POST /predict`, `POST /predict/csv` (batch file), `GET /model`, `GET /health`. Input validated against the contract.
- Loads `models:/housing-value@champion`. Switching models moves an alias; no redeploy. If the registry is unreachable, the loaded model keeps serving.
- Drift reference is read from the model's run (`gold_manifest.json`), so serving needs no access to training files.
- UI: minimal Streamlit, same public API.

**Prediction log** (async, never blocks the response)

| Field | Used for |
|---|---|
| `request_id`, `timestamp` | Label join, time windows |
| `model_version`, `gold_version` | Per-version performance, lineage |
| Raw input features | Input drift, future training data |
| Prediction | Output drift |
| `latency_ms`, `status`, `client` | Operations, consumer attribution |

Labels arrive later (e.g. sale prices) and join on `request_id` / entity key. Prod: API → Pub/Sub → BigQuery (partitioned, retention policy).

## 6. Monitoring and feedback

| Signal | Labels needed | Detection | Points to | Action |
|---|---|---|---|---|
| Data quality (nulls, ranges, schema) | No | API contract + daily counts | Data | Reject, alert, fix upstream |
| Input drift | No | PSI per feature vs model's Gold stats | Data | Retrain on recent window |
| Output drift | No | PSI on predictions | Early proxy | Investigate |
| Performance (RMSE, by slice) | Yes | Windowed, joined labels | Algorithm | Retrain; if not recovered → new features or model |
| Latency, errors | No | SLOs | Machine | Alert, rollback |

**What feedback teaches**

- Drifting features → retraining window and source checks.
- Failing slices (e.g. regions) → new features or sources.
- Out-of-range inputs → contract update.
- Retraining doesn't recover performance → modelling problem → new experiment.

All retraining produces a challenger. It passes the same gate as any run.

## 7. Production on GCP

| Component | Service | Why |
|---|---|---|
| Raw storage | GCS | Bronze landing |
| Data layers | BigQuery + Dataform | Transformations as SQL in git |
| Data versioning | Immutable Gold snapshot per version | Manifest equivalent |
| Orchestration | Vertex AI Pipelines (KFP) | Managed, scheduled, containerised |
| Tracking + registry | MLflow on Cloud Run + Cloud SQL + GCS | Single source of truth; Vertex only orchestrates |
| Serving | Cloud Run | Scale-to-zero; small CPU model |
| Prediction log | Pub/Sub → BigQuery | Decouples serving from storage |
| Monitoring | Scheduled drift job → BigQuery + Cloud Monitoring alerts | Cheap, transparent |
| Features (when needed) | Vertex AI Feature Store | Online lookup of features not in the request |
| CI/CD | GitHub Actions + Artifact Registry | Keyless auth via Workload Identity Federation |
| Secrets, IaC, access | Secret Manager, Terraform, IAP | |

Dev and prod are separate GCP projects with least-privilege service accounts. The design is portable (MLflow, containers, KFP, Terraform).

## 8. Cost

Order of magnitude, list prices. **All figures USD per month.**

| Item | Prod ($/month) | Dev ($/month) |
|---|---|---|
| MLflow (Cloud Run + small Cloud SQL + GCS) | ~$15–30 | ~$10–20 |
| Serving (Cloud Run, 1 min instance prod / 0 dev) | ~$15–50 | ~$0–5 |
| Training (CPU, minutes per run, weekly) | <$5 | <$5 |
| Pub/Sub + BigQuery logs (~1M predictions) | <$5 | <$1 |
| Drift job | <$1 | <$1 |
| **Total** | **~$40–90 / month** | **~$15–30 / month** |

- Alternative avoided: an always-on Vertex endpoint costs ~$80–100 per month per node alone.
- Levers: scale-to-zero, CPU only, scheduled monitoring, log sampling, partitioning + TTL.
- Every resource labelled (`owner`, `env`, `component`) → billing export → cost per model and per 1k predictions.
- People time dominates infra cost. Managed services minimise it.

## 9. PoC to production

| PoC | Production | Change |
|---|---|---|
| Files + manifests | BigQuery + Gold snapshots | Storage adapter |
| Local MLflow | MLflow on Cloud Run | Tracking URI |
| `make train` | Vertex AI Pipeline | Same entrypoint, containerised |
| JSONL prediction log | Pub/Sub → BigQuery | Logger sink |
| Dockerfile | Cloud Run + Terraform | Same images; `MLFLOW_TRACKING_URI` set at deploy |
| `ENV=dev\|prod` config | Separate projects | Config only |

## 10. Decisions

| # | Decision | Alternative | Why |
|---|---|---|---|
| 1 | MLflow | Vertex Experiments + Registry | Open, one store; avoids lock-in |
| 2 | Manifests (PoC), Gold snapshots (prod) | DVC | Native to BigQuery; DVC suits file-based data |
| 3 | Pipeline as artifact | Separate preprocessing | No leakage, no skew |
| 4 | Registry per contract, alias switching | Registry per algorithm | Switch architectures without consumer changes |
| 5 | Promote code, retrain in prod | Promote artifacts | Prod data stays in prod; training is cheap |
| 6 | Cloud Run | Vertex Endpoints | Scale-to-zero; revisit for GPU or high throughput |
| 7 | Own drift job | Managed model monitoring | Cheaper, transparent; revisit at scale |

## Out of scope (PoC)

Authentication, autoscaling, feature store, shadow traffic, label ingestion.
