# PulseScore

PulseScore: Given a text input, return a sentiment label + confidence score via an API.

---

## What This Project Is

PulseScore is an AI-focused inference system.

Flow:
Client → FastAPI → Model → Logging → Metrics → Dashboard

The goal is to build a production-style ML system step by step (Step1 → Step12).

---

## Current Progress

### Step1 — API Bootstrap
- FastAPI setup
- `/health` endpoint working
- `/predict` endpoint (dummy logic)
- Repo pushed cleanly

---

### Step2 — Baseline Model Training

- Dataset: SST2 (binary sentiment)
- Model: TF-IDF (1–2 grams) + Logistic Regression
- Artifacts saved:
  - `model.joblib`
  - `vectorizer.joblib`
  - `metrics_baseline.json`

**Results**
- Validation Accuracy: **0.8085**
- Validation F1: **0.8202**

---

### Step3 — Real Inference API

- Loads model + vectorizer at startup
- `/predict` runs real inference using `predict_proba()`
- Returns:
  - `label`
  - `confidence`
  - `score`
  - `model_version`
  - `request_id`

---

### Step4 — Validation & Error Handling

- Enforced request validation using Pydantic
- Rejected empty and whitespace-only input
- Added global exception handlers
- Standardized error response format with `request_id`

---

### Step5 — Latency Tracking & Logging

- Measured request latency using `time.perf_counter()`
- Added structured JSON logging (`logs/requests.jsonl`)
- Logged:
  - request_id
  - status_code
  - latency_ms
  - label / confidence / score
  - input_length
  - model_version
- Added success/error flags for observability

---

## Next Step

### 🔜 Step6 — Metrics Endpoint

- Build `/metrics` endpoint
- Track:
  - total_requests
  - error_rate
  - avg_latency / p95_latency
  - prediction distribution
