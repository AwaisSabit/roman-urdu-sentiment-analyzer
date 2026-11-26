"""Tests for batch scoring."""

import csv
import json

import pandas as pd
import pytest

from rusent.batch import (
    format_breakdown,
    run_batch,
    score_texts,
    sentiment_breakdown,
)
from rusent.cli import main
from rusent.model import save_model, train_model

TRAIN_TEXTS = [
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
TRAIN_LABELS = [
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
def model():
    return train_model(TRAIN_TEXTS, TRAIN_LABELS)


def write_reviews(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        writer.writerows([[row] for row in rows])


def test_sentiment_breakdown_counts_and_percentages():
    results = [
        {"label": "positive", "uncertain": False},
        {"label": "positive", "uncertain": False},
        {"label": "negative", "uncertain": True},
        {"label": "neutral", "uncertain": False},
    ]
    breakdown = sentiment_breakdown(results)
    assert breakdown["total"] == 4
    assert breakdown["counts"] == {"positive": 2, "negative": 1, "neutral": 1}
    assert breakdown["percentages"] == {
        "positive": 50.0,
        "negative": 25.0,
        "neutral": 25.0,
    }
    assert breakdown["uncertain_count"] == 1


def test_sentiment_breakdown_accepts_saved_json_rows():
    rows = [
        {"sentiment": "positive", "uncertain": False},
        {"sentiment": "negative", "uncertain": False},
    ]
    breakdown = sentiment_breakdown(rows)
    assert breakdown["counts"]["positive"] == 1
    assert breakdown["counts"]["negative"] == 1


def test_sentiment_breakdown_empty_list():
    breakdown = sentiment_breakdown([])
    assert breakdown["total"] == 0
    assert breakdown["percentages"] == {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }
    assert breakdown["uncertain_count"] == 0


def test_format_breakdown_renders_table():
    breakdown = sentiment_breakdown(
        [{"label": "positive", "uncertain": False}] * 3
        + [{"label": "negative", "uncertain": True}]
    )
    text = format_breakdown(breakdown)
    assert "Total reviews: 4" in text
    assert "positive" in text and "75.0%" in text
    assert "Uncertain predictions: 1" in text


def test_score_texts_returns_one_result_each(model):
    results = score_texts(model, ["khana bohat acha tha", "khana kharab tha"])
    assert len(results) == 2
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert all("confidence" in r and "uncertain" in r for r in results)


def test_batch_writes_csv_and_json(model, tmp_path):
    input_csv = tmp_path / "in.csv"
    write_reviews(input_csv, ["khana bohat acha tha", "khana kharab tha", ""])
    payload = run_batch(
        input_csv, tmp_path / "out.csv", tmp_path / "out.json", model
    )
    assert payload["total"] == 3
    assert payload["breakdown"]["positive"]["count"] == 1
    assert payload["breakdown"]["negative"]["count"] == 1
    # The empty review has no signal, so it is always uncertain.
    assert payload["results"][2]["uncertain"] is True
    assert payload["uncertain_count"] >= 1
    out_df = pd.read_csv(tmp_path / "out.csv")
    assert list(out_df.columns) == ["text", "sentiment", "confidence", "uncertain"]
    assert len(out_df) == 3
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["total"] == 3
    assert saved["results"][0]["sentiment"] == "positive"


def test_batch_breakdown_percentages_add_up(model, tmp_path):
    input_csv = tmp_path / "in.csv"
    write_reviews(input_csv, ["khana bohat acha tha"] * 3 + ["khana kharab tha"])
    payload = run_batch(
        input_csv, tmp_path / "out.csv", tmp_path / "out.json", model
    )
    total_pct = sum(v["percentage"] for v in payload["breakdown"].values())
    assert abs(total_pct - 100.0) < 0.2


def test_batch_missing_column_raises(model, tmp_path):
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review"])
        writer.writerow(["khana acha tha"])
    with pytest.raises(ValueError, match="not found"):
        run_batch(bad, tmp_path / "out.csv", tmp_path / "out.json", model)


def test_batch_cli(model, tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)
    input_csv = tmp_path / "in.csv"
    write_reviews(input_csv, ["khana bohat acha tha", "khana kharab tha"])
    assert main([
        "batch", str(input_csv),
        "--output", str(tmp_path / "out.csv"),
        "--json", str(tmp_path / "out.json"),
        "--model", str(model_path),
    ]) == 0
    out = capsys.readouterr().out
    assert "Scored 2 reviews" in out
    assert "positive" in out and "50.0%" in out
    assert "Total reviews: 2" in out
    assert (tmp_path / "out.csv").exists()
    assert (tmp_path / "out.json").exists()
