# ML Model Serving API

A production-style **FastAPI** service that serves a text-topic classifier (TF-IDF + logistic regression, 4 newsgroup topics, 86.3% held-out accuracy) with the engineering around it that a notebook doesn't have: input validation, batching with limits, health checks, graceful behaviour without a model, tests and CI.

```bash
pip install -r requirements.txt
python train_model.py                      # trains and writes model.joblib
uvicorn app.main:app --port 8000           # docs at http://localhost:8000/docs
pytest -q
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `ok` / `degraded` plus whether the model is loaded |
| POST | `/predict` | `{"text": "..."}` returns label, confidence and all class probabilities |
| POST | `/predict/batch` | up to 64 texts per request |

```json
POST /predict  {"text": "NASA plans a new mission to orbit Mars"}
-> {"label": "sci.space", "confidence": 0.9927, "probabilities": {...}}
```

## Design notes
- **Validation:** Pydantic enforces 3 to 5000 characters per text and 1 to 64 texts per batch; bad input gets a `422`, not a stack trace.
- **Model loaded once** at startup (lifespan). If the model file is missing the service still starts, `/health` reports `degraded`, and prediction endpoints return `503` with a clear message.
- **Path configurable** with the `MODEL_PATH` environment variable.

## Verification
- 9 tests (FastAPI `TestClient`, toy model so they run offline in seconds): health, prediction and probabilities summing to 1, batching, every validation rule, and the missing-model path.
- Smoke-tested against a real running `uvicorn` server (returned `sci.space`, confidence 0.99; short input returned 422).
- GitHub Actions runs the tests on every push (`.github/workflows/tests.yml`).
- **Not verified:** the `Dockerfile` has not been built, because Docker isn't installed on my machine.

## Limitations
No authentication, rate limiting, model versioning or metrics endpoint yet; the model is a simple baseline (see [text-classification-nlp](https://github.com/venuuday7-maker/text-classification-nlp) for its evaluation).
