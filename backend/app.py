from fastapi import FastAPI
from pydantic import BaseModel
import uuid

from utils.model_loader import load_artifacts, MODEL_VERSION

app = FastAPI()


class PredictRequest(BaseModel):
    text: str
    request_id: str | None = None


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health_check():
    return {"status": "PulseScore backend is running", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(req: PredictRequest):
    model, vectorizer = load_artifacts()

    request_id = req.request_id or str(uuid.uuid4())

    X = vectorizer.transform([req.text])
    probs = model.predict_proba(X)[0]

    neg_prob = float(probs[0])
    pos_prob = float(probs[1])

    label = "positive" if pos_prob >= 0.5 else "negative"
    confidence = pos_prob if label == "positive" else neg_prob
    score = pos_prob

    return {
        "request_id": request_id,
        "label": label,
        "confidence": confidence,
        "score": score,
        "model_version": MODEL_VERSION,
    }