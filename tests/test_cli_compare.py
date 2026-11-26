"""Tests for the `rusent compare` command."""

import subprocess
import sys

from rusent.cli import main


def test_compare_cli(tmp_path, capsys):
    report = tmp_path / "comparison.md"
    assert main(["compare", "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "Logistic Regression:" in out
    assert "Naive Bayes:" in out
    assert "LinearSVC:" in out
    assert report.exists()
    assert "Model comparison" in report.read_text(encoding="utf-8")


def test_compare_cli_via_subprocess(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rusent", "compare",
         "--report", str(tmp_path / "comparison.md")],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Logistic Regression:" in result.stdout
