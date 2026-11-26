"""Tests for mixed Roman Urdu + English reviews.

Real customers code-switch constantly — one review can start in English
and finish in Roman Urdu. The model must handle that, not just pure
Roman Urdu or pure English.
"""

import pytest

from rusent.model import load_dataset, predict_sentiment, train_model
from rusent.preprocessing import clean_text

# (review, expected sentiment) — none of these are verbatim dataset rows.
MIXED_EXAMPLES = [
    ("pizza was amazing, cheese bhi bohat fresh thi", "positive"),
    ("app is too slow, order karna bohat mushkil hai", "negative"),
    ("khana bohat tasty tha and the service was excellent too", "positive"),
    ("quality is very poor, ek wash mein colour fade ho gaya", "negative"),
    ("delivery was on time, sab kuch theek tha", "neutral"),
    ("food was fresh aur garam bhi tha, maza aa gaya", "positive"),
    ("product is not as shown, bilkul different hai", "negative"),
    ("app is fine, kuch features slow hain", "neutral"),
]


@pytest.fixture(scope="module")
def full_model():
    df = load_dataset()
    return train_model(df["text"].tolist(), df["sentiment"].tolist())


def test_mixed_reviews_predicted_correctly(full_model):
    for text, expected in MIXED_EXAMPLES:
        result = predict_sentiment(full_model, text)
        assert result["label"] == expected, f"{text!r} -> {result['label']}"
        assert result["confidence"] > 0.5


def test_mixed_review_keeps_words_from_both_languages():
    cleaned = clean_text("pizza was amazing, cheese bhi bohat fresh thi")
    assert "pizza" in cleaned and "amazing" in cleaned
    assert "cheese" in cleaned and "bohat" in cleaned and "fresh" in cleaned


def test_dataset_has_mixed_language_reviews():
    df = load_dataset()
    mixed = df[df["language"] == "mixed"]
    assert len(mixed) >= 100
    assert set(mixed["sentiment"]) == {"positive", "negative", "neutral"}


def test_code_switched_negation():
    # "not" in English + Roman Urdu praise words must stay negative.
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    result = predict_sentiment(model, "khana was not acha at all, bilkul bekaar")
    assert result["label"] == "negative"


def test_same_meaning_both_languages_agree(full_model):
    roman_urdu = predict_sentiment(full_model, "khana bohat acha tha")
    english = predict_sentiment(full_model, "the food was really good")
    assert roman_urdu["label"] == english["label"] == "positive"
