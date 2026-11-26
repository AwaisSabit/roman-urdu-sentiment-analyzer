"""Model training, saving, loading and prediction.

The baseline is Logistic Regression over the word + character TF-IDF
features. Models are saved with joblib and stay local-only — they are
never committed, because the snapshot push only carries text files.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_vectorizer

LABELS = ["negative", "neutral", "positive"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rusent.joblib"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "reviews.csv"

# Predictions below this confidence are flagged as uncertain, not trusted.
UNCERTAIN_THRESHOLD = 0.5


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the review dataset as a DataFrame with text and sentiment."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    return df


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")


def train_model(texts, labels, classifier=None) -> Pipeline:
    """Train a vectorizer + classifier pipeline on texts and labels."""
    pipeline = Pipeline(
        [
            ("features", build_vectorizer()),
            ("clf", classifier or build_classifier()),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def save_model(model: Pipeline, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def train_and_save(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Train on the full dataset and save the model. Returns the model."""
    df = load_dataset(dataset_path)
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    save_model(model, model_path)
    return model


def predict_sentiment(model: Pipeline, text: str) -> dict:
    """Predict one review. Returns label, confidence and per-class scores.

    Empty or punctuation-only text has no signal, so it comes back as
    neutral with zero confidence and is flagged uncertain.
    """
    from .preprocessing import clean_text

    if not clean_text(text):
        return {
            "label": "neutral",
            "confidence": 0.0,
            "probabilities": {label: 1 / 3 for label in LABELS},
            "uncertain": True,
        }
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best_idx = int(probabilities.argmax())
    label = classes[best_idx]
    confidence = float(probabilities[best_idx])
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            classes[i]: float(probabilities[i]) for i in range(len(classes))
        },
        "uncertain": confidence < UNCERTAIN_THRESHOLD,
    }
