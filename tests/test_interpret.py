"""Tests for model interpretability (top words per sentiment)."""

from rusent.cli import main
from rusent.interpret import render_top_words, top_words_per_class
from rusent.model import load_dataset, save_model, train_model


def test_top_words_cover_all_sentiments():
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    top = top_words_per_class(model, n=10)
    assert set(top) == {"positive", "negative", "neutral"}
    for label, words in top.items():
        assert len(words) == 10
        # Coefficients are sorted strongest first.
        coefs = [coef for _, coef in words]
        assert coefs == sorted(coefs, reverse=True)
        assert all(coef > 0 for coef in coefs)


def test_top_words_match_known_sentiment_words():
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    top = top_words_per_class(model, n=15)
    positive_words = {w for w, _ in top["positive"]}
    negative_words = {w for w, _ in top["negative"]}
    assert positive_words & {"zabardast", "acha", "bohat acha", "amazing", "excellent"}
    # Negation carries the negative class: "not" / "nahi" and "bohat bura".
    assert negative_words & {"nahi", "not", "bohat bura", "bura experience"}


def test_render_top_words_lists_each_sentiment():
    top = {
        "positive": [("zabardast", 2.1)],
        "negative": [("kharab", 1.9)],
        "neutral": [("theek", 1.2)],
    }
    text = render_top_words(top)
    assert "positive:" in text and "zabardast" in text
    assert "negative:" in text and "kharab" in text
    assert "neutral:" in text and "theek" in text


def test_top_words_cli(tmp_path, capsys):
    df = load_dataset()
    model = train_model(df["text"].tolist(), df["sentiment"].tolist())
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)
    assert main(["top-words", "--model", str(model_path), "--n", "5"]) == 0
    out = capsys.readouterr().out
    assert "positive:" in out and "negative:" in out and "neutral:" in out
