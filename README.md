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

### ✅ Step1 — API Bootstrap
- FastAPI setup
- `/health` endpoint working
- `/predict` endpoint (dummy logic)
- Repo pushed cleanly

### ✅ Step2 — Baseline Model Training (Current Stage)

- Dataset: SST2 (binary sentiment)
- Model: TF-IDF (1–2 grams) + Logistic Regression
- Artifacts saved:
  - `model.joblib`
  - `vectorizer.joblib`
  - `metrics_baseline.json`

### Baseline Results
- Validation Accuracy: **0.8085**
- Validation F1: **0.8202**

This means we now have a real trained sentiment model ready to be integrated into the API.

---

## Next Step

### 🔜 Step3 — Real Inference API
- Load model on startup
- Replace dummy `/predict`
- Return real probabilities