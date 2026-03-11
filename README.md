# PulseScore
PulseScore: Given a text input, return a sentiment label + confidence score via an API.


## What This Project Is
PulseScore is an AI-focused inference system.
Flow:
Client → FastAPI → Model → Logging → Metrics → Dashboard
The goal is to build a production-style ML system step by step (Step1 → Step12).

## Current Progress
### Step1 — API Bootstrap
- FastAPI setup
- `/health` endpoint working
- `/predict` endpoint (dummy logic)
- Repo pushed cleanly

### Step2 — Baseline Model Training
- Dataset: SST2 (binary sentiment)
- Model: TF-IDF (1–2 grams) + Logistic Regression
- Artifacts saved:
  - `model.joblib`
  - `vectorizer.joblib`
  - `metrics_baseline.json`

### Step3 — Real Inference API
- Backend loads `artifacts/model.joblib` and `artifacts/vectorizer.joblib` on startup
- `/predict` now runs real model inference
- Uses `vectorizer.transform()` and `model.predict_proba()`
- Returns:
  - `label`
  - `confidence`
  - `score`
  - `model_version`
  - `request_id`

### Step4 — Validation & Error Handling
- Added strict request validation using **Pydantic**
- Enforced:
  - `text` required
  - `text` length limits
  - rejection of whitespace-only input
- Added global **RequestValidationError handler**
- Added global **fallback exception handler**
- Standardized API error response format: