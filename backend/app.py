from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uuid
import time

from utils.model_loader import load_artifacts, MODEL_VERSION
from utils.logger import write_log

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

    write_log({
        "request_id": request_id,
        "path": str(request.url.path),
        "status_code": 422,
        "success": False,
        "error_code": "VALIDATION_ERROR",
        "message": "Invalid request body",
        "latency_ms": None,
        "model_version": MODEL_VERSION,
    })

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

    write_log({
        "request_id": request_id,
        "path": str(request.url.path),
        "status_code": 500,
        "success": False,
        "error_code": "MODEL_ARTIFACT_MISSING",
        "message": str(exc),
        "latency_ms": None,
        "model_version": MODEL_VERSION,
    })

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

    write_log({
        "request_id": request_id,
        "path": str(request.url.path),
        "status_code": 500,
        "success": False,
        "error_code": "INTERNAL_SERVER_ERROR",
        "message": "Something went wrong",
        "latency_ms": None,
        "model_version": MODEL_VERSION,
    })

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
    start_time = time.perf_counter()

    model, vectorizer = load_artifacts()
    request_id = req.request_id or str(uuid.uuid4())

    clean_text = req.text.strip()
    input_length = len(clean_text)

    if not clean_text:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        write_log({
            "request_id": request_id,
            "path": "/predict",
            "status_code": 422,
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "text must not be empty or whitespace-only",
            "latency_ms": latency_ms,
            "input_length": 0,
            "model_version": MODEL_VERSION,
        })

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
    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

    write_log({
        "request_id": request_id,
        "path": "/predict",
        "status_code": 200,
        "success": True,
        "prediction_type": "sentiment",
        "label": label,
        "confidence": confidence,
        "score": score,
        "latency_ms": latency_ms,
        "input_length": input_length,
        "text_preview": clean_text[:60],
        "model_version": MODEL_VERSION,
    })

    return {
        "request_id": request_id,
        "label": label,
        "confidence": confidence,
        "score": score,
        "model_version": MODEL_VERSION,
    }