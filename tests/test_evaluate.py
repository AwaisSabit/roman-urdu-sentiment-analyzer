"""Tests for evaluation."""

import json

from rusent.evaluate import (
    evaluate_predictions,
    grouped_split,
    render_evaluation_report,
    run_evaluation,
)
from rusent.model import load_dataset
from rusent.preprocessing import base_sentence


def test_base_sentence_strips_followup():
    assert base_sentence("khana bohat acha tha, highly recommended") == (
        "khana bohat acha tha"
    )
    assert base_sentence("khana bohat acha tha. will order again") == (
        "khana bohat acha tha"
    )
    assert base_sentence("khana bohat acha tha") == "khana bohat acha tha"


def test_grouped_split_keeps_base_sentences_apart():
    df = load_dataset()
    texts = df["text"].tolist()
    labels = df["sentiment"].tolist()
    train_texts, test_texts, _, _ = grouped_split(texts, labels)
    train_bases = {base_sentence(t) for t in train_texts}
    test_bases = {base_sentence(t) for t in test_texts}
    assert not train_bases & test_bases
    assert len(train_texts) + len(test_texts) == len(texts)


def test_evaluate_predictions_metrics():
    true = ["positive", "positive", "negative", "neutral", "neutral"]
    pred = ["positive", "neutral", "negative", "neutral", "positive"]
    metrics = evaluate_predictions(true, pred)
    assert metrics["accuracy"] == 0.6
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["test_size"] == 5
    matrix = metrics["confusion_matrix"]["matrix"]
    assert sum(sum(row) for row in matrix) == 5
    assert metrics["confusion_matrix"]["labels"] == [
        "negative",
        "neutral",
        "positive",
    ]


def test_report_renders_confusion_matrix():
    metrics = evaluate_predictions(
        ["positive", "negative"], ["positive", "neutral"]
    )
    report = render_evaluation_report(metrics)
    assert "Confusion matrix" in report
    assert "Weakest class" in report
    assert "Accuracy:" in report


def test_full_evaluation_writes_reports(tmp_path):
    metrics = run_evaluation(
        report_path=tmp_path / "evaluation.md",
        json_path=tmp_path / "evaluation.json",
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["test_size"] > 50
    # Honest numbers: the grouped split must not give a perfect score.
    assert metrics["accuracy"] < 1.0
    assert (tmp_path / "evaluation.md").exists()
    saved = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == metrics["accuracy"]
    report_text = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "Confusion matrix" in report_text
    assert "Weakest class" in report_text
