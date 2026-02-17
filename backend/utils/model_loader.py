from pathlib import Path
import joblib


MODEL_VERSION = "v0.1-baseline-tfidf-logreg"

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"


_model = None
_vectorizer = None


def load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Missing vectorizer file: {VECTORIZER_PATH}")

        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

    return _model, _vectorizer