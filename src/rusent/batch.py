"""Batch scoring: score a whole CSV of reviews and write the results out."""

import json
from pathlib import Path

import pandas as pd

from .model import predict_sentiment


def score_texts(model, texts) -> list:
    """Score a list of review texts. Returns one result dict per text."""
    return [
        {"text": text, **predict_sentiment(model, text)} for text in texts
    ]


def run_batch(
    input_path,
    output_csv,
    output_json,
    model,
    text_column: str = "text",
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
    results = score_texts(model, texts)

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
    counts = {label: 0 for label in ("positive", "negative", "neutral")}
    for r in results:
        counts[r["label"]] += 1
    payload = {
        "total": total,
        "breakdown": {
            label: {
                "count": count,
                "percentage": round(100 * count / total, 1) if total else 0.0,
            }
            for label, count in counts.items()
        },
        "uncertain_count": sum(1 for r in results if r["uncertain"]),
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
