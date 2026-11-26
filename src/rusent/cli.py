"""Command-line interface: rusent predict / train / evaluate / batch / demo."""

import argparse
import sys

from .model import DEFAULT_MODEL_PATH, UNCERTAIN_THRESHOLD, load_model, predict_sentiment


def _load_model_or_exit(model_path):
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"No trained model found at {model_path}.", file=sys.stderr)
        print("Run `rusent train` first.", file=sys.stderr)
        raise SystemExit(2)


def cmd_predict(args) -> int:
    model = _load_model_or_exit(args.model)
    result = predict_sentiment(model, args.text)
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}")
    for label, score in sorted(result["probabilities"].items()):
        print(f"  {label}: {score:.2f}")
    if result["uncertain"]:
        print("Uncertain: confidence is below the threshold "
              f"({UNCERTAIN_THRESHOLD}), treat this as a guess.")
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
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
