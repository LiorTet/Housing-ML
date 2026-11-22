"""Serving API: always the `champion` alias, reloaded on change. Every prediction is logged."""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from threading import Lock

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field, create_model

from housing.config import Settings, load_settings
from housing.features import CONTRACT, FEATURES
from housing.registry import CHAMPION, Registry

logger = logging.getLogger(__name__)

Features = create_model(
    "Features",
    **{
        name: (float, Field(ge=low, le=high, description=desc))
        for name, (low, high, desc) in CONTRACT.items()
    },
)


class PredictRequest(BaseModel):
    instances: list[Features] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    request_id: str
    predictions: list[float]
    model_version: str
    gold_version: str


class ModelService:
    """Holds the champion; polls the alias, so switching needs no redeploy."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = Registry(settings)
        self.model, self.version, self.checked = None, None, 0.0
        self.lock = Lock()

    def current(self):
        with self.lock:
            if time.monotonic() - self.checked >= self.settings.refresh_seconds:
                self.checked = time.monotonic()
                try:
                    champion = self.registry.get(CHAMPION)
                    if champion and (
                        self.version is None or champion.version != self.version.version
                    ):
                        self.model, self.version = self.registry.load(champion.version), champion
                except Exception:
                    # Registry down must not take serving down: keep the loaded model.
                    logger.exception(
                        "Champion refresh failed; serving v%s",
                        getattr(self.version, "version", None),
                    )
            return self.model, self.version

    def log(self, request_id: str, inputs: pd.DataFrame, preds: list[float], version, latency_ms):
        path = self.settings.prediction_log
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).isoformat()
        with path.open("a") as f:
            for row, pred in zip(inputs.to_dict("records"), preds, strict=True):
                f.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "timestamp": ts,
                            "model_version": str(version.version),
                            "gold_version": version.tags["gold_version"],
                            "inputs": row,
                            "prediction": pred,
                            "latency_ms": latency_ms,
                        }
                    )
                    + "\n"
                )


def create_app(settings: Settings | None = None) -> FastAPI:
    service = ModelService(settings or load_settings())
    app = FastAPI(title="Housing Value API", version="1.0")

    def champion():
        model, version = service.current()
        if model is None:
            raise HTTPException(503, "No champion model registered.")
        return model, version

    @app.get("/health")
    def health() -> dict:
        _, version = service.current()
        return {
            "status": "ok" if version else "no_model",
            "env": service.settings.env,
            "model_version": str(version.version) if version else None,
        }

    @app.get("/model")
    def model_info() -> dict:
        _, version = champion()
        return {"name": version.name, "version": str(version.version), "tags": version.tags}

    def serve(request: PredictRequest, tasks: BackgroundTasks) -> PredictResponse:
        start = time.perf_counter()
        model, version = champion()
        inputs = pd.DataFrame([i.model_dump() for i in request.instances])[FEATURES]
        preds = [round(float(p), 5) for p in model.predict(inputs)]
        request_id = str(uuid.uuid4())
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        tasks.add_task(service.log, request_id, inputs, preds, version, latency_ms)
        return PredictResponse(
            request_id=request_id,
            predictions=preds,
            model_version=str(version.version),
            gold_version=version.tags["gold_version"],
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, tasks: BackgroundTasks) -> PredictResponse:
        return serve(request, tasks)

    @app.post("/predict/csv", response_model=PredictResponse)
    def predict_csv(file: UploadFile, tasks: BackgroundTasks) -> PredictResponse:
        """CSV with the feature columns (extra columns, e.g. the target, are ignored)."""
        try:
            request = PredictRequest(instances=pd.read_csv(file.file).to_dict("records"))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return serve(request, tasks)

    return app
