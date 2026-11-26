"""Tests that the README stays honest and clean."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_has_no_day_numbers():
    text = README.read_text(encoding="utf-8")
    assert not re.search(r"\bDay \d+\b", text)


def test_readme_quotes_real_metrics():
    text = README.read_text(encoding="utf-8")
    assert "0.946" in text  # accuracy
    assert "0.938" in text  # macro F1
    assert "0.917" in text  # weakest class F1


def test_readme_shows_real_example_predictions():
    text = README.read_text(encoding="utf-8")
    assert "khana bohat acha tha, service bhi zabardast" in text
    assert "pizza was amazing, cheese bhi bohat fresh thi" in text
    assert "8/8 correct" in text


def test_readme_documents_all_commands():
    text = README.read_text(encoding="utf-8")
    for command in ("train", "predict", "evaluate", "batch", "demo", "top-words"):
        assert f"rusent {command}" in text


def test_readme_states_limitations():
    text = README.read_text(encoding="utf-8")
    assert "Honest limitations" in text
    assert "synthetic" in text
