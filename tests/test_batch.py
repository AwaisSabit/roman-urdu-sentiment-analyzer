"""Tests for batch scoring."""

import csv
import json

import pandas as pd
import pytest

from rusent.batch import run_batch, score_texts
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
    assert "positive: 1 (50.0%)" in out
    assert (tmp_path / "out.csv").exists()
    assert (tmp_path / "out.json").exists()
