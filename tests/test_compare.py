"""Tests for model comparison."""

from rusent.compare import (
    best_model_name,
    build_candidates,
    compare_models,
    render_comparison_report,
    run_comparison,
)
from rusent.model import load_dataset

TEXTS = [
    "khana bohat acha tha",
    "pizza zabardast thi",
    "the food was amazing",
    "khana bilkul kharab tha",
    "the food was terrible",
    "burger bilkul bekaar tha",
    "khana theek tha",
    "the food was okay",
    "pizza normal thi",
    "biryani ka taste lajawab tha",
    "delivery bohat late thi",
    "app bohat slow hai",
    "shoes bohat comfortable hain",
    "watch do din mein kharab ho gayi",
    "quality theek hai",
]
LABELS = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
    "positive",
    "negative",
    "negative",
    "positive",
    "negative",
    "neutral",
]


def test_three_candidates():
    candidates = build_candidates()
    assert set(candidates) == {"Logistic Regression", "Naive Bayes", "LinearSVC"}


def test_compare_returns_metrics_for_all():
    results = compare_models(TEXTS, LABELS, test_size=0.33, random_state=42)
    assert set(results) == {"Logistic Regression", "Naive Bayes", "LinearSVC"}
    for name, metrics in results.items():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert set(metrics["per_class"]) == {"positive", "negative", "neutral"}
        assert metrics["test_size"] == 5


def test_best_model_name_is_a_candidate():
    results = compare_models(TEXTS, LABELS, test_size=0.33, random_state=42)
    assert best_model_name(results) in results


def test_report_renders_all_models():
    results = compare_models(TEXTS, LABELS, test_size=0.33, random_state=42)
    report = render_comparison_report(results)
    assert "Logistic Regression" in report
    assert "Naive Bayes" in report
    assert "LinearSVC" in report
    assert "Best model by macro F1" in report


def test_full_dataset_comparison_is_strong(tmp_path):
    df = load_dataset()
    results = compare_models(
        df["text"].tolist(), df["sentiment"].tolist(), random_state=42
    )
    # On this clean hand-written dataset every model should do well.
    for name, metrics in results.items():
        assert metrics["accuracy"] > 0.70, f"{name} accuracy {metrics['accuracy']}"
    report_path = tmp_path / "comparison.md"
    report_path.write_text(render_comparison_report(results), encoding="utf-8")
    assert report_path.exists()
    assert "Logistic Regression" in report_path.read_text(encoding="utf-8")
