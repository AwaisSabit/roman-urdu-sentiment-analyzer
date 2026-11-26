"""Interpretability: which words most strongly point at each sentiment.

For the Logistic Regression model, each word feature has one coefficient
per class. The words with the largest coefficients for a class are the
ones that push a review towards that feeling. Only word features are
shown — character n-grams are useful to the model but unreadable.
"""

from .model import LABELS


def top_words_per_class(model, n: int = 10) -> dict:
    """Top `n` indicative word features per sentiment, with coefficients."""
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    word_indices = [
        i for i, name in enumerate(feature_names) if name.startswith("word__")
    ]
    word_names = [feature_names[i][len("word__"):] for i in word_indices]
    result = {}
    for class_idx, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_idx][word_indices]
        order = coefficients.argsort()[::-1][:n]
        result[label] = [
            (word_names[i], round(float(coefficients[i]), 3)) for i in order
        ]
    return result


def render_top_words(top_words: dict) -> str:
    """Render the top words as a readable report."""
    lines = ["Top words per sentiment", ""]
    for label in LABELS:
        words = top_words.get(label, [])
        lines.append(f"{label}:")
        for word, coef in words:
            lines.append(f"  {word:20s} {coef:+.3f}")
        lines.append("")
    return "\n".join(lines)
