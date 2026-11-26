"""Evaluation: honest held-out metrics for the shipped model.

The split is grouped by base review sentence, so a review and its longer
follow-up version never land on opposite sides of the split. A plain
random split leaked near-duplicates and flattered every model to 100%.
"""

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

from .model import DEFAULT_DATASET_PATH, LABELS, load_dataset, train_model
from .preprocessing import base_sentence

DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "evaluation.md"
)
DEFAULT_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "evaluation.json"
)


def grouped_split(texts, labels, test_size=0.2, random_state=42):
    """Train/test split grouped by base review sentence."""
    groups = [base_sentence(text) for text in texts]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(texts, labels, groups))
    return (
        [texts[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


def evaluate_predictions(true_labels, predicted_labels) -> dict:
    """Score predictions: accuracy, macro F1, per-class scores, confusion matrix."""
    report = classification_report(
        true_labels, predicted_labels, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": [[int(cell) for cell in row] for row in matrix],
        },
        "test_size": len(true_labels),
    }


def render_evaluation_report(metrics: dict) -> str:
    """Render the metrics as a readable Markdown report."""
    lines = ["# Evaluation", ""]
    lines.append(f"Test set size: {metrics['test_size']} reviews")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append("| Sentiment | Precision | Recall | F1 | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label in LABELS:
        per = metrics["per_class"][label]
        lines.append(
            f"| {label} | {per['precision']:.3f} | {per['recall']:.3f} "
            f"| {per['f1']:.3f} | {per['support']} |"
        )
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows are the true label, columns are the predicted label.")
    lines.append("")
    header = "| True \\ Predicted | " + " | ".join(LABELS) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in LABELS) + " |")
    for label, row in zip(LABELS, metrics["confusion_matrix"]["matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    weakest = min(
        metrics["per_class"],
        key=lambda label: metrics["per_class"][label]["f1"],
    )
    lines.append(
        f"Weakest class: **{weakest}** "
        f"(F1 {metrics['per_class'][weakest]['f1']:.3f})."
    )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
) -> dict:
    """Train the shipped model on the train split, evaluate on the test split,
    and write the Markdown and JSON reports. Returns the metrics."""
    df = load_dataset(dataset_path)
    texts = df["text"].tolist()
    labels = df["sentiment"].tolist()
    train_texts, test_texts, train_labels, test_labels = grouped_split(
        texts, labels
    )
    model = train_model(train_texts, train_labels)
    metrics = evaluate_predictions(test_labels, model.predict(test_texts))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_evaluation_report(metrics), encoding="utf-8")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
