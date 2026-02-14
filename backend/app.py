from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "PulseScore backend is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.lower()

    label = "positive" if "good" in text else "negative"
    confidence = 0.50

    return {
        "label": label,
        "confidence": confidence,
        "model_version": "v0.1"
    }