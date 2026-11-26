"""Tests for TF-IDF feature extraction."""

from rusent.features import build_char_vectorizer, build_vectorizer, build_word_vectorizer

SAMPLES = [
    "khana bohat acha tha",
    "khana bahut acha tha",
    "the food was terrible",
    "pizza was amazing",
]


def test_word_vectorizer_fits_and_transforms():
    vec = build_word_vectorizer()
    matrix = vec.fit_transform(SAMPLES)
    assert matrix.shape[0] == len(SAMPLES)
    assert matrix.shape[1] > 0


def test_spelling_variants_share_word_features():
    vec = build_word_vectorizer()
    matrix = vec.fit_transform(["khana bohat acha", "khana bahut acha"])
    # After normalisation both rows are identical.
    assert (matrix[0] != matrix[1]).nnz == 0


def test_word_bigrams_captured():
    vec = build_word_vectorizer()
    vec.fit(SAMPLES)
    names = vec.get_feature_names_out()
    assert "bohat acha" in names


def test_char_vectorizer_catches_spelling_variation():
    vec = build_char_vectorizer()
    matrix = vec.fit_transform(["zabardast", "zabardust"])
    # Two different spellings still share most character n-grams.
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    assert sim > 0.5


def test_combined_vectorizer_has_both_parts():
    vec = build_vectorizer()
    matrix = vec.fit_transform(SAMPLES)
    assert matrix.shape[0] == len(SAMPLES)
    assert matrix.shape[1] > 0
    names = vec.get_feature_names_out()
    assert any(name.startswith("word__") for name in names)
    assert any(name.startswith("char__") for name in names)


def test_empty_text_transforms_to_zero_row():
    vec = build_vectorizer()
    vec.fit(SAMPLES)
    matrix = vec.transform([""])
    assert matrix.shape[0] == 1
    assert matrix.nnz == 0
