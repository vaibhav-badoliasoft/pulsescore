# PulseScore

PulseScore: Given a text input, return a sentiment label + confidence score via an API.

---

## What this project is
PulseScore is an AI-focused inference + monitoring project:
Client → FastAPI → Model → Logging → Metrics → Dashboard

This repo is built in **12 Steps (Step1 → Step12)**, where each step has smaller subparts.

---

## API Contract (v1)

### Request JSON
```json
{
  "text": "....",
  "request_id": "optional"
}
