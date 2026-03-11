from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uuid

from utils.model_loader import load_artifacts, MODEL_VERSION

app = FastAPI()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    request_id: str | None = None


@app.on_event("startup")
def startup():
    load_artifacts()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = str(uuid.uuid4())

    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request body",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_exception_handler(request: Request, exc: FileNotFoundError):
    request_id = str(uuid.uuid4())

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": {
                "code": "MODEL_ARTIFACT_MISSING",
                "message": str(exc),
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Something went wrong",
            },
        },
    )


@app.get("/health")
def health_check():
    return {
        "status": "PulseScore backend is running",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    model, vectorizer = load_artifacts()

    request_id = req.request_id or str(uuid.uuid4())

    clean_text = req.text.strip()
    if not clean_text:
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "text must not be empty or whitespace-only",
                },
            },
        )

    X = vectorizer.transform([clean_text])
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