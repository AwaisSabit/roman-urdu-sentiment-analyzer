"""Compare classifiers side by side on the same held-out test set.

Logistic Regression is the baseline. Multinomial Naive Bayes and LinearSVC
are the challengers. All three share the same word + character TF-IDF
features, so the comparison is fair.
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_vectorizer
from .model import DEFAULT_DATASET_PATH, LABELS, load_dataset

DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "model-comparison.md"
)


def build_candidates() -> dict:
    """The three classifiers under comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "LinearSVC": LinearSVC(C=0.5, class_weight="balanced"),
    }


def compare_models(texts, labels, test_size=0.2, random_state=42) -> dict:
    """Train each candidate on the train split and score it on the test split.

    Returns a dict of model name -> metrics (accuracy, macro F1, per-class
    precision/recall/F1, and the fitted pipeline).
    """
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    results = {}
    for name, classifier in build_candidates().items():
        pipeline = Pipeline(
            [("features", build_vectorizer()), ("clf", classifier)]
        )
        pipeline.fit(train_texts, train_labels)
        predicted = pipeline.predict(test_texts)
        report = classification_report(
            test_labels, predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        results[name] = {
            "accuracy": accuracy_score(test_labels, predicted),
            "macro_f1": f1_score(test_labels, predicted, average="macro"),
            "per_class": {
                label: {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": int(report[label]["support"]),
                }
                for label in LABELS
            },
            "test_size": len(test_texts),
            "model": pipeline,
        }
    return results


def best_model_name(results: dict) -> str:
    """The model with the highest macro F1."""
    return max(results, key=lambda name: results[name]["macro_f1"])


def render_comparison_report(results: dict) -> str:
    """Render the comparison as a Markdown report."""
    lines = ["# Model comparison", ""]
    lines.append(f"Test set size: {next(iter(results.values()))['test_size']} reviews")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 |")
    lines.append("| --- | --- | --- |")
    for name, metrics in sorted(
        results.items(), key=lambda item: item[1]["macro_f1"], reverse=True
    ):
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} |"
        )
    lines.append("")
    for name, metrics in sorted(
        results.items(), key=lambda item: item[1]["macro_f1"], reverse=True
    ):
        lines.append(f"## {name}")
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
    lines.append(f"Best model by macro F1: **{best_model_name(results)}**")
    lines.append("")
    return "\n".join(lines)


def run_comparison(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    """Run the comparison on the dataset and write the report. Returns results."""
    df = load_dataset(dataset_path)
    results = compare_models(df["text"].tolist(), df["sentiment"].tolist())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_comparison_report(results), encoding="utf-8")
    return results
