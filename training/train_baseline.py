import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics_baseline_v2.json"

MODEL_VERSION = "v0.1-baseline-tfidf-logreg"


def main():
    ds = load_dataset("glue", "sst2")

    train_texts = ds["train"]["sentence"]
    train_labels = ds["train"]["label"]

    val_texts = ds["validation"]["sentence"]
    val_labels = ds["validation"]["label"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=80000,
        ngram_range=(1, 2),
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    model = LogisticRegression(
        max_iter=2000,
        C=2.0,
    )
    model.fit(X_train, train_labels)

    preds = model.predict(X_val)

    acc = float(accuracy_score(val_labels, preds))
    f1 = float(f1_score(val_labels, preds))

    cm = confusion_matrix(val_labels, preds).tolist()

    report = classification_report(val_labels, preds, output_dict=True)

    metrics = {
        "model_version": MODEL_VERSION,
        "dataset": "glue/sst2",
        "val_accuracy": acc,
        "val_f1": f1,
        "confusion_matrix": cm,
        "notes": {
            "features": "features": "TF-IDF (1-2 grams, max_features=80k, no stopwords)",
            "classifier": "LogisticRegression",
        },
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved:")
    print(" -", MODEL_PATH)
    print(" -", VECTORIZER_PATH)
    print(" -", METRICS_PATH)
    print("\n📊 Results:")
    print("Accuracy:", acc)
    print("F1:", f1)
    print("\nConfusion Matrix:", cm)
    print("\n(Full classification report saved inside metrics json if needed.)")


if __name__ == "__main__":
    main()
