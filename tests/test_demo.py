"""Tests for the `rusent demo` command."""

from rusent.cli import main
from rusent.demo import DEMO_REVIEWS, print_demo, run_demo
from rusent.model import load_dataset, train_model


def test_demo_reviews_cover_all_sentiments_and_languages():
    labels = {expected for _, expected in DEMO_REVIEWS}
    assert labels == {"positive", "negative", "neutral"}
    assert len(DEMO_REVIEWS) == 6


def test_demo_with_full_model(capsys):
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    results = run_demo(model)
    assert len(results) == 6
    assert all(r["correct"] for r in results)
    assert all(r["confidence"] > 0.5 for r in results)
    print_demo(results)
    out = capsys.readouterr().out
    assert "Demo: 6/6 correct." in out


def test_demo_cli(tmp_path, capsys):
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    from rusent.model import save_model

    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)
    assert main(["demo", "--model", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Demo: 6/6 correct." in out
    assert "khana bohat acha tha" in out
