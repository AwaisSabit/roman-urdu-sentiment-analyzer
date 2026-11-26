"""Edge cases and dataset integrity: empty text, punctuation only,
very long reviews, non-Latin script, and the dataset itself.
"""

import pandas as pd
import pytest

from rusent.batch import run_batch, score_texts
from rusent.model import DEFAULT_DATASET_PATH, load_dataset, predict_sentiment, train_model


@pytest.fixture(scope="module")
def full_model():
    df = load_dataset()
    return train_model(df["text"].tolist(), df["sentiment"].tolist())


def test_empty_string(full_model):
    result = predict_sentiment(full_model, "")
    assert result["label"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["uncertain"] is True


def test_whitespace_only(full_model):
    result = predict_sentiment(full_model, "   \n\t  ")
    assert result["uncertain"] is True


def test_punctuation_only(full_model):
    for text in ["!!!", "???...", "@#$%^&*", ".,.,.,"]:
        result = predict_sentiment(full_model, text)
        assert result["uncertain"] is True
        assert result["label"] in ("positive", "negative", "neutral")


def test_numbers_only(full_model):
    result = predict_sentiment(full_model, "12345 67890")
    assert result["uncertain"] is True


def test_single_word(full_model):
    result = predict_sentiment(full_model, "zabardast")
    assert result["label"] == "positive"
    result = predict_sentiment(full_model, "bekaar")
    assert result["label"] == "negative"


def test_very_long_text(full_model):
    long_text = ("khana bohat acha tha, service bhi zabardast. " * 200).strip()
    assert len(long_text) > 8000
    result = predict_sentiment(full_model, long_text)
    assert result["label"] == "positive"
    assert 0.0 <= result["confidence"] <= 1.0


def test_urdu_script_does_not_crash(full_model):
    # Urdu script is not Roman Urdu — the cleaner strips it, so the
    # prediction must come back uncertain rather than crash or lie.
    result = predict_sentiment(full_model, "بہت اچھا کھانا تھا")
    assert result["uncertain"] is True


def test_emoji_only(full_model):
    result = predict_sentiment(full_model, "😍😍😍")
    assert result["uncertain"] is True


def test_batch_with_empty_rows(full_model, tmp_path):
    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["khana bohat acha tha", "", "   ", "!!!"]}).to_csv(
        input_csv, index=False
    )
    payload = run_batch(
        input_csv, tmp_path / "out.csv", tmp_path / "out.json", full_model
    )
    assert payload["total"] == 4
    assert payload["uncertain_count"] == 3
    assert payload["results"][0]["sentiment"] == "positive"


def test_batch_single_row(full_model, tmp_path):
    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["khana bohat acha tha"]}).to_csv(input_csv, index=False)
    payload = run_batch(
        input_csv, tmp_path / "out.csv", tmp_path / "out.json", full_model
    )
    assert payload["total"] == 1
    assert payload["breakdown"]["positive"]["percentage"] == 100.0


# --- dataset integrity ---

def test_dataset_has_no_missing_or_empty_text():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert df["text"].notna().all()
    assert (df["text"].str.strip() != "").all()
    assert df["sentiment"].notna().all()


def test_dataset_labels_are_valid():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert set(df["sentiment"]) == {"positive", "negative", "neutral"}


def test_dataset_texts_are_unique():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert df["text"].nunique() == len(df)


def test_dataset_is_balanced():
    df = load_dataset(DEFAULT_DATASET_PATH)
    counts = df["sentiment"].value_counts()
    assert counts.min() == counts.max() == 140


def test_dataset_columns_and_types():
    df = load_dataset(DEFAULT_DATASET_PATH)
    assert list(df.columns) == ["text", "sentiment", "topic", "language"]
    assert set(df["topic"]) <= {"food", "shopping", "delivery", "mobile"}
    assert set(df["language"]) <= {"roman_urdu", "english", "mixed"}


def test_dataset_file_is_plain_utf8_text():
    raw = DEFAULT_DATASET_PATH.read_bytes()
    raw.decode("utf-8")  # raises if the file is not valid UTF-8
    assert b"\x00" not in raw  # no binary content
