"""Tests for Roman Urdu text preprocessing."""

from rusent.preprocessing import clean_text, normalise_spelling, tokenize


def test_lowercases_and_strips_punctuation():
    assert clean_text("Khana Bohat Acha Tha!") == "khana bohat acha tha"


def test_normalises_bohat_variants():
    assert clean_text("khana bahut acha tha") == "khana bohat acha tha"
    assert clean_text("khana boht acha tha") == "khana bohat acha tha"
    assert clean_text("khana bht acha tha") == "khana bohat acha tha"


def test_normalises_nahi_variants():
    assert clean_text("maza nhi aaya") == "maza nahi aaya"
    assert clean_text("acha ni tha") == "acha nahi tha"


def test_normalises_hai_variants():
    assert clean_text("yeh kya he") == "yeh kya hai"
    assert clean_text("bohat acha hy") == "bohat acha hai"


def test_normalises_accha_variants():
    assert clean_text("khana accha tha") == "khana acha tha"


def test_removes_urls():
    assert clean_text("check https://example.com khana acha") == "check khana acha"


def test_empty_and_punctuation_only():
    assert clean_text("") == ""
    assert clean_text("!!!...???") == ""
    assert clean_text("   ") == ""


def test_collapses_whitespace_and_numbers_stripped():
    assert clean_text("khana   bohat    acha 123") == "khana bohat acha"


def test_stopword_removal_keeps_negation():
    cleaned = clean_text("khana nahi acha tha aur maza nahi aaya", remove_stopwords=True)
    assert "nahi" in cleaned.split()
    assert "aur" not in cleaned.split()
    assert "tha" not in cleaned.split()


def test_stopword_removal_keeps_not():
    cleaned = clean_text("the food was not good at all", remove_stopwords=True)
    assert "not" in cleaned.split()
    assert "the" not in cleaned.split()


def test_tokenize():
    assert tokenize("Khana Bohat Acha!") == ["khana", "bohat", "acha"]
    assert tokenize("") == []
    assert tokenize("!!!") == []


def test_normalise_spelling_keeps_unknown_words():
    assert normalise_spelling("biryani zabardast") == "biryani zabardast"


def test_mixed_language_survives_cleaning():
    cleaned = clean_text("pizza was amazing, cheese bhi bohat fresh thi")
    assert "pizza" in cleaned and "bohat" in cleaned and "fresh" in cleaned
