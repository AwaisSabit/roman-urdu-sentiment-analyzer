"""Tests for the `rusent predict` command."""

import subprocess
import sys

import pytest

from rusent.cli import main
from rusent.model import save_model, train_model

TRAIN_TEXTS = [
    "khana bohat acha tha",
    "pizza zabardast thi",
    "the food was amazing",
    "khana bilkul kharab tha",
    "the food was terrible",
    "burger bilkul bekaar tha",
    "khana theek tha",
    "the food was okay",
    "pizza normal thi",
]
TRAIN_LABELS = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
]


@pytest.fixture
def model_path(tmp_path):
    model = train_model(TRAIN_TEXTS, TRAIN_LABELS)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    return path


def test_predict_positive(model_path, capsys):
    assert main(["predict", "khana bohat acha tha", "--model", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Sentiment: positive" in out
    assert "Confidence:" in out


def test_predict_negative(model_path, capsys):
    main(["predict", "khana bilkul kharab tha", "--model", str(model_path)])
    out = capsys.readouterr().out
    assert "Sentiment: negative" in out


def test_predict_empty_text_is_uncertain(model_path, capsys):
    main(["predict", "!!!", "--model", str(model_path)])
    out = capsys.readouterr().out
    assert "Sentiment: neutral" in out
    assert "Uncertain" in out


def test_predict_without_model_exits_with_hint(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["predict", "khana acha tha", "--model", str(tmp_path / "missing.joblib")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "rusent train" in err


def test_predict_via_subprocess(model_path):
    result = subprocess.run(
        [sys.executable, "-m", "rusent", "predict", "pizza was amazing",
         "--model", str(model_path)],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Sentiment: positive" in result.stdout
