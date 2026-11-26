"""Feature extraction: TF-IDF word n-grams plus character n-grams.

Word n-grams (1, 2) catch phrases like "bohat acha" and "not good".
Character n-grams (3, 5) matter because Roman Urdu spelling varies —
"zabardast" and "zabardust" share most of their character n-grams even
though the words differ, so the model still recognises the word.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .preprocessing import clean_text

WORD_MAX_FEATURES = 3000
CHAR_MAX_FEATURES = 5000


def build_word_vectorizer() -> TfidfVectorizer:
    """TF-IDF over word unigrams and bigrams."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=WORD_MAX_FEATURES,
        min_df=1,
    )


def build_char_vectorizer() -> TfidfVectorizer:
    """TF-IDF over character n-grams inside words (3 to 5 characters)."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=CHAR_MAX_FEATURES,
        min_df=1,
    )


def build_vectorizer() -> FeatureUnion:
    """Word + character TF-IDF stacked together."""
    return FeatureUnion(
        [
            ("word", build_word_vectorizer()),
            ("char", build_char_vectorizer()),
        ]
    )
