"""Tests for the `rusent evaluate` command."""

import json
import subprocess
import sys

from rusent.cli import main


def test_evaluate_prints_real_metrics(tmp_path, capsys):
    report = tmp_path / "evaluation.md"
    metrics_json = tmp_path / "evaluation.json"
    assert main([
        "evaluate", "--report", str(report), "--json", str(metrics_json),
    ]) == 0
    out = capsys.readouterr().out
    assert "Accuracy:" in out
    assert "Macro F1:" in out
    assert "negative:" in out and "neutral:" in out and "positive:" in out
    assert report.exists() and metrics_json.exists()
    saved = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert 0.0 < saved["accuracy"] < 1.0
    assert saved["test_size"] > 50


def test_evaluate_via_subprocess(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rusent", "evaluate",
         "--report", str(tmp_path / "evaluation.md"),
         "--json", str(tmp_path / "evaluation.json")],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Accuracy:" in result.stdout
