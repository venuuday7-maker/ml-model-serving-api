"""FastAPI service that serves a text-topic classifier with validation, batching and health checks."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "model.joblib"))
MAX_BATCH = 64


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000, description="Text to classify")


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():  # service still starts without a model so /health can report it
        state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Topic classifier API", version="1.0.0", lifespan=lifespan)


def _model():
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded; run `python train_model.py` first.")
    return state["bundle"]


def _predict(texts: list[str]) -> list[Prediction]:
    bundle = _model()
    probs = bundle["model"].predict_proba(texts)
    out = []
    for row in probs:
        scored = {label: round(float(p), 4) for label, p in zip(bundle["labels"], row)}
        label = max(scored, key=scored.get)
        out.append(Prediction(label=label, confidence=scored[label], probabilities=scored))
    return out


@app.get("/health")
def health():
    return {"status": "ok" if "bundle" in state else "degraded", "model_loaded": "bundle" in state}


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    return _predict([req.text])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    if any(not t.strip() or len(t) > 5000 for t in req.texts):
        raise HTTPException(status_code=422, detail="Every text must be non-empty and at most 5000 characters.")
    return _predict(req.texts)
