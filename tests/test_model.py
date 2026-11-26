"""Tests for model training, saving, loading and prediction."""

from pathlib import Path

import pytest

from rusent.model import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    load_model,
    predict_sentiment,
    save_model,
    train_model,
)

SMALL_TEXTS = [
    "khana bohat acha tha",
    "pizza zabardast thi",
    "the food was amazing",
    "khana bilkul kharab tha",
    "the food was terrible",
    "burger bilkul bekaar tha",
    "khana theek tha",
    "the food was okay",
    "pizza normal thi",
]
SMALL_LABELS = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
]


@pytest.fixture
def trained():
    return train_model(SMALL_TEXTS, SMALL_LABELS)


def test_dataset_loads_balanced():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert len(df) == 420
    counts = df["sentiment"].value_counts().to_dict()
    assert counts == {"positive": 140, "negative": 140, "neutral": 140}
    assert df["text"].isna().sum() == 0
    assert df["text"].nunique() == 420


def test_dataset_has_all_topics_and_languages():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert set(df["topic"]) == {"food", "shopping", "delivery", "mobile"}
    assert set(df["language"]) == {"roman_urdu", "english", "mixed"}


def test_train_predicts_training_examples(trained):
    result = predict_sentiment(trained, "khana bohat acha tha")
    assert result["label"] == "positive"
    assert result["confidence"] > 0.5
    assert result["uncertain"] is False


def test_negative_prediction(trained):
    result = predict_sentiment(trained, "khana bilkul kharab tha")
    assert result["label"] == "negative"


def test_probabilities_sum_to_one(trained):
    result = predict_sentiment(trained, "pizza theek thi")
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-6
    assert set(result["probabilities"]) == {"positive", "negative", "neutral"}


def test_empty_text_is_uncertain_neutral(trained):
    result = predict_sentiment(trained, "")
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["uncertain"] is True
    result = predict_sentiment(trained, "!!! ???")
    assert result["uncertain"] is True


def test_threshold_flags_low_confidence_as_uncertain(trained):
    normal = predict_sentiment(trained, "khana bohat acha tha")
    assert normal["uncertain"] is False
    strict = predict_sentiment(trained, "khana bohat acha tha", threshold=0.99)
    assert strict["uncertain"] is True
    assert strict["label"] == normal["label"]


def test_threshold_zero_trusts_everything_except_empty(trained):
    result = predict_sentiment(trained, "khana bohat acha tha", threshold=0.0)
    assert result["uncertain"] is False
    empty = predict_sentiment(trained, "", threshold=0.0)
    assert empty["uncertain"] is True


def test_save_and_load_round_trip(trained, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(trained, path)
    assert path.exists()
    loaded = load_model(path)
    before = predict_sentiment(trained, "khana bohat acha tha")
    after = predict_sentiment(loaded, "khana bohat acha tha")
    assert before["label"] == after["label"]
    assert abs(before["confidence"] - after["confidence"]) < 1e-9


def test_spelling_variant_predicts_same(trained):
    a = predict_sentiment(trained, "khana bohat acha tha")
    b = predict_sentiment(trained, "khana bahut acha tha")
    assert a["label"] == b["label"]
