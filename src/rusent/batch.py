"""Batch scoring: score a whole CSV of reviews and write the results out."""

import json
from pathlib import Path

import pandas as pd

from .model import UNCERTAIN_THRESHOLD, LABELS, predict_sentiment


def sentiment_breakdown(results: list) -> dict:
    """Counts and percentages per sentiment for a list of scored reviews."""
    total = len(results)
    counts = {label: 0 for label in LABELS}
    for result in results:
        # Scored results use "label"; saved JSON rows use "sentiment".
        label = result.get("label") or result.get("sentiment")
        counts[label] += 1
    return {
        "total": total,
        "counts": counts,
        "percentages": {
            label: round(100 * count / total, 1) if total else 0.0
            for label, count in counts.items()
        },
        "uncertain_count": sum(1 for r in results if r["uncertain"]),
    }


def format_breakdown(breakdown: dict) -> str:
    """A small readable table of the breakdown, with a bar per sentiment."""
    lines = [f"Total reviews: {breakdown['total']}"]
    for label in LABELS:
        count = breakdown["counts"][label]
        pct = breakdown["percentages"][label]
        bar = "#" * int(pct // 2)
        lines.append(f"  {label:9s} {count:4d} ({pct:5.1f}%) {bar}")
    if breakdown["uncertain_count"]:
        lines.append(
            f"Uncertain predictions: {breakdown['uncertain_count']}"
        )
    return "\n".join(lines)


def score_texts(model, texts, threshold: float = UNCERTAIN_THRESHOLD) -> list:
    """Score a list of review texts. Returns one result dict per text."""
    return [
        {"text": text, **predict_sentiment(model, text, threshold)}
        for text in texts
    ]


def run_batch(
    input_path,
    output_csv,
    output_json,
    model,
    text_column: str = "text",
    threshold: float = UNCERTAIN_THRESHOLD,
) -> dict:
    """Read reviews from a CSV, score them, write a results CSV and JSON.

    The results CSV has one row per review with the predicted sentiment,
    confidence and an uncertain flag. The JSON holds the same rows plus a
    breakdown of counts and percentages per sentiment.
    Returns the JSON payload.
    """
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    output_json = Path(output_json)
    df = pd.read_csv(input_path, encoding="utf-8")
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {input_path}. "
            f"Available columns: {', '.join(df.columns)}"
        )
    texts = df[text_column].fillna("").astype(str).tolist()
    results = score_texts(model, texts, threshold)

    out_df = pd.DataFrame(
        [
            {
                "text": r["text"],
                "sentiment": r["label"],
                "confidence": round(r["confidence"], 4),
                "uncertain": r["uncertain"],
            }
            for r in results
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    total = len(results)
    breakdown = sentiment_breakdown(results)
    payload = {
        "total": total,
        "breakdown": {
            label: {
                "count": breakdown["counts"][label],
                "percentage": breakdown["percentages"][label],
            }
            for label in LABELS
        },
        "uncertain_count": breakdown["uncertain_count"],
        "results": [
            {
                "text": r["text"],
                "sentiment": r["label"],
                "confidence": round(r["confidence"], 4),
                "uncertain": r["uncertain"],
            }
            for r in results
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
