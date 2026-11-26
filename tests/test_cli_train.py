"""Tests for the `rusent train` command."""

import subprocess
import sys

from rusent.cli import main
from rusent.model import load_model, predict_sentiment


def test_train_saves_a_working_model(tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    assert main(["train", "--model", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Trained on 420 reviews" in out
    assert "Model saved to" in out
    assert model_path.exists()
    model = load_model(model_path)
    result = predict_sentiment(model, "khana bohat acha tha")
    assert result["label"] == "positive"


def test_train_via_subprocess(tmp_path):
    model_path = tmp_path / "model.joblib"
    result = subprocess.run(
        [sys.executable, "-m", "rusent", "train", "--model", str(model_path)],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Trained on 420 reviews" in result.stdout
    assert model_path.exists()
