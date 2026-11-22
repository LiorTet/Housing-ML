import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from housing.api import ModelService, create_app
from housing.monitoring import psi

ROW = {
    "MedInc": 8.3,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


class ConstantModel:
    def predict(self, x):
        return np.full(len(x), 2.0)


def test_predict_logs_every_prediction(settings, monkeypatch):
    version = SimpleNamespace(name="m", version=7, tags={"gold_version": "abc"})
    monkeypatch.setattr(ModelService, "current", lambda self: (ConstantModel(), version))
    client = TestClient(create_app(settings))

    body = client.post("/predict", json={"instances": [ROW, ROW]}).json()
    assert body["predictions"] == [2.0, 2.0]
    assert body["model_version"] == "7"

    lines = settings.prediction_log.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert records[0]["inputs"] == ROW
    assert records[0]["gold_version"] == "abc"


def test_predict_csv_uses_same_contract(settings, monkeypatch):
    version = SimpleNamespace(name="m", version=7, tags={"gold_version": "abc"})
    monkeypatch.setattr(ModelService, "current", lambda self: (ConstantModel(), version))
    client = TestClient(create_app(settings))
    good = pd.DataFrame([ROW, ROW]).assign(MedHouseVal=1.0).to_csv(index=False)
    bad = pd.DataFrame([ROW | {"Latitude": 99.0}]).to_csv(index=False)

    assert client.post("/predict/csv", files={"file": good}).json()["predictions"] == [2.0, 2.0]
    assert client.post("/predict/csv", files={"file": bad}).status_code == 422


def test_predict_without_champion_is_503(settings):
    client = TestClient(create_app(settings))
    assert client.post("/predict", json={"instances": [ROW]}).status_code == 503
    assert client.get("/health").json()["status"] == "no_model"


def test_registry_outage_keeps_serving_loaded_model(settings, monkeypatch):
    service = ModelService(settings)
    service.model, service.version = ConstantModel(), SimpleNamespace(version=7)

    def down(alias):
        raise ConnectionError("registry down")

    monkeypatch.setattr(service.registry, "get", down)
    assert service.current() == (service.model, service.version)


def test_predict_rejects_out_of_contract_input(settings):
    client = TestClient(create_app(settings))
    assert client.post("/predict", json={"instances": [ROW | {"Latitude": 90}]}).status_code == 422


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=10_000)
    edges = np.quantile(reference, np.linspace(0, 1, 11)).tolist()
    assert psi(edges, rng.normal(size=2_000)) < 0.05
    assert psi(edges, rng.normal(loc=1.0, size=2_000)) > 0.25
