"""Demo: run the model over a few example reviews and show the results."""

from .model import predict_sentiment

# Example reviews with their human label, in Roman Urdu, English and mixed.
DEMO_REVIEWS = [
    ("khana bohat acha tha, service bhi zabardast", "positive"),
    ("pizza bilkul fresh tha aur cheese bhi bohat tasty thi", "positive"),
    ("The food was cold and tasteless, a complete waste of money", "negative"),
    ("delivery 3 ghante late thi, koi call bhi nahi ki", "negative"),
    ("khana theek tha, na bohat acha na bohat bura", "neutral"),
    ("The quality is fine for the price, nothing extraordinary", "neutral"),
]


def run_demo(model) -> list:
    """Score the demo reviews. Returns one result dict per review."""
    results = []
    for text, expected in DEMO_REVIEWS:
        result = predict_sentiment(model, text)
        results.append(
            {
                "text": text,
                "expected": expected,
                "label": result["label"],
                "confidence": result["confidence"],
                "correct": result["label"] == expected,
                "uncertain": result["uncertain"],
            }
        )
    return results


def print_demo(results: list) -> None:
    correct = sum(1 for r in results if r["correct"])
    for r in results:
        mark = "OK" if r["correct"] else "MISS"
        print(f"[{mark}] {r['label']} ({r['confidence']:.2f})"
              f" — expected {r['expected']}")
        print(f"      {r['text']}")
    print(f"\nDemo: {correct}/{len(results)} correct.")
