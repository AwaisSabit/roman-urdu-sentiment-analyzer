"""Command-line interface: rusent predict / train / evaluate / batch / demo."""

import argparse
import sys

from .batch import format_breakdown, run_batch, sentiment_breakdown
from .demo import print_demo, run_demo
from .evaluate import DEFAULT_JSON_PATH, DEFAULT_REPORT_PATH, run_evaluation
from .model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    UNCERTAIN_THRESHOLD,
    load_dataset,
    load_model,
    predict_sentiment,
    train_and_save,
)


def _load_model_or_exit(model_path):
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"No trained model found at {model_path}.", file=sys.stderr)
        print("Run `rusent train` first.", file=sys.stderr)
        raise SystemExit(2)


def cmd_demo(args) -> int:
    model = _load_model_or_exit(args.model)
    print_demo(run_demo(model))
    return 0


def cmd_batch(args) -> int:
    model = _load_model_or_exit(args.model)
    try:
        payload = run_batch(
            args.input, args.output, args.json, model, args.text_column,
            args.threshold,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Scored {payload['total']} reviews.")
    print(format_breakdown(sentiment_breakdown(payload["results"])))
    print(f"Results written to {args.output} and {args.json}.")
    return 0


def cmd_evaluate(args) -> int:
    metrics = run_evaluation(args.dataset, args.report, args.json)
    print(f"Test set size: {metrics['test_size']} reviews")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for label, per in metrics["per_class"].items():
        print(f"  {label}: precision {per['precision']:.3f}, "
              f"recall {per['recall']:.3f}, F1 {per['f1']:.3f} "
              f"(n={per['support']})")
    print(f"Report written to {args.report} and {args.json}.")
    return 0


def cmd_train(args) -> int:
    df = load_dataset(args.dataset)
    model = train_and_save(args.dataset, args.model)
    print(f"Trained on {len(df)} reviews "
          f"({', '.join(f'{k}: {v}' for k, v in df['sentiment'].value_counts().items())}).")
    print(f"Model saved to {args.model}.")
    print("Run `rusent evaluate` to see the held-out scores.")
    return 0


def cmd_predict(args) -> int:
    model = _load_model_or_exit(args.model)
    result = predict_sentiment(model, args.text, args.threshold)
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}")
    for label, score in sorted(result["probabilities"].items()):
        print(f"  {label}: {score:.2f}")
    if result["uncertain"]:
        print("Uncertain: confidence is below the threshold "
              f"({args.threshold}), treat this as a guess.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rusent",
        description="Classify Roman Urdu and English reviews as "
        "positive, negative or neutral.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict", help="Score one review."
    )
    predict.add_argument("text", help="The review text to score.")
    predict.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model file.",
    )
    predict.add_argument(
        "--threshold", type=float, default=UNCERTAIN_THRESHOLD,
        help="Confidence below this is flagged as uncertain (default 0.5).",
    )
    predict.set_defaults(func=cmd_predict)

    train = subparsers.add_parser(
        "train", help="Train the model on the review dataset and save it."
    )
    train.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
        help="Path to the reviews CSV file.",
    )
    train.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model.",
    )
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="Print held-out accuracy, F1 and per-class scores."
    )
    evaluate.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
        help="Path to the reviews CSV file.",
    )
    evaluate.add_argument(
        "--report", type=str, default=str(DEFAULT_REPORT_PATH),
        help="Where to write the Markdown report.",
    )
    evaluate.add_argument(
        "--json", type=str, default=str(DEFAULT_JSON_PATH),
        help="Where to write the JSON metrics.",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    batch = subparsers.add_parser(
        "batch", help="Score a CSV of reviews and write the results out."
    )
    batch.add_argument("input", help="Input CSV with a column of reviews.")
    batch.add_argument(
        "--output", type=str, default="results.csv",
        help="Where to write the results CSV.",
    )
    batch.add_argument(
        "--json", type=str, default="results.json",
        help="Where to write the results JSON.",
    )
    batch.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model file.",
    )
    batch.add_argument(
        "--text-column", type=str, default="text",
        help="Name of the column holding the review text.",
    )
    batch.add_argument(
        "--threshold", type=float, default=UNCERTAIN_THRESHOLD,
        help="Confidence below this is flagged as uncertain (default 0.5).",
    )
    batch.set_defaults(func=cmd_batch)

    demo = subparsers.add_parser(
        "demo", help="Run the model over a few example reviews."
    )
    demo.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model file.",
    )
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
