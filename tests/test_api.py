"""API tests use a tiny model trained on toy sentences so they run offline and fast."""
import importlib

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DOCS = ["rocket orbit moon satellite launch", "space station astronaut orbit", "engine car brakes transmission", "truck tires engine road"]
LABELS = ["space", "space", "autos", "autos"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    model = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression(C=10))]).fit(DOCS, LABELS)
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "labels": list(model.classes_)}, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    import app.main as main
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_label_and_probabilities_summing_to_one(client):
    body = client.post("/predict", json={"text": "the rocket reached orbit"}).json()
    assert body["label"] == "space"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_batch_predicts_each_text(client):
    r = client.post("/predict/batch", json={"texts": ["moon launch orbit", "car engine brakes"]}).json()
    assert [p["label"] for p in r] == ["space", "autos"]


@pytest.mark.parametrize("payload", [{"text": "hi"}, {"text": ""}, {}, {"text": "x" * 5001}])
def test_invalid_single_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_limits_enforced(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["ok text"] * 65}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["   "]}).status_code == 422


def test_missing_model_gives_503_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    import app.main as main
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json={"text": "some text"}).status_code == 503
