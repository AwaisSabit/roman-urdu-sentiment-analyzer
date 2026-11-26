"""Text cleaning and normalisation for Roman Urdu and English reviews.

Roman Urdu has no fixed spelling, so the same word appears in many forms
(bohat / bahut / boht). Normalising the common variants before the model
sees the text makes the features much more consistent.
"""

import re

# Common spelling variants mapped to one canonical form.
SPELLING_VARIANTS = {
    # intensifiers / praise
    "bahut": "bohat", "boht": "bohat", "bht": "bohat", "bhot": "bohat",
    "buhat": "bohat", "buhut": "bohat",
    "accha": "acha", "achaa": "acha", "achha": "acha",
    "zabardust": "zabardast", "zbrdst": "zabardast",
    "lajawaab": "lajawab",
    "kamaal": "kamal",
    "mazaa": "maza",
    "tastey": "tasty",
    "fres": "fresh",
    # negation and verbs
    "nhi": "nahi", "ni": "nahi", "nahe": "nahi", "nahin": "nahi",
    "he": "hai", "ha": "hai", "hy": "hai",
    "ta": "tha", "thaa": "tha",
    "ti": "thi", "thee": "thi",
    "they": "thay",
    "hn": "hain", "han": "hain",
    "krna": "karna",
    "kro": "karo",
    "milla": "mila",
    "aya": "aaya", "aaaya": "aaya",
    "gya": "gaya",
    "rha": "raha",
    "rhi": "rahi",
    "rhe": "rahe",
    # common words
    "khaana": "khana",
    "boora": "bura",
    "khrab": "kharab",
    "dubara": "dobara",
    "zarur": "zaroor", "zaror": "zaroor",
    "pesa": "paisa", "paise": "paisa",
    "wasol": "wasool", "vasool": "wasool",
    "oder": "order",
    "delivry": "delivery",
    "servis": "service",
    "qualty": "quality",
    "praice": "price",
}

# Words that carry no sentiment and only add noise. Negation words
# (nahi, not, no, never) are deliberately NOT in here — they flip meaning.
ROMAN_URDU_STOPWORDS = {
    "ka", "ki", "ke", "ko", "se", "mein", "me", "aur", "bhi", "yeh", "ye",
    "wo", "woh", "jo", "kya", "tha", "thi", "thay", "hain", "hai",
    "is", "us", "in", "un", "par", "tak", "sa", "ne",
    "kuch", "sab", "abhi",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "it", "its", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "my", "i", "you",
    "he", "she", "they", "we", "as", "at", "by", "from", "so", "but",
    "are", "be", "been", "have", "has", "had", "do", "does", "did",
}

STOPWORDS = ROMAN_URDU_STOPWORDS | ENGLISH_STOPWORDS

# Short follow-up clauses the dataset generator appends to base reviews.
# Stripping them recovers the base sentence, which is how the evaluation
# split keeps near-duplicate reviews out of both train and test at once.
FOLLOWUP_CLAUSES = {
    "positive": [
        "dobara zaroor aaunga", "highly recommended", "bohat khush hun",
        "paisa wasool hai", "friends ko bhi recommend karunga",
        "will order again", "bohat acha experience raha",
    ],
    "negative": [
        "dobara kabhi nahi aaunga", "not recommended at all",
        "bohat disappointed hun", "paisa zaya ho gaya",
        "will not order again", "bohat bura experience raha",
    ],
    "neutral": [
        "kuch khaas nahi tha", "theek hai", "average experience tha",
        "na acha na bura", "it was fine", "normal tha",
    ],
}

ALL_FOLLOWUPS = [
    clause for clauses in FOLLOWUP_CLAUSES.values() for clause in clauses
]


def base_sentence(text: str) -> str:
    """Strip a trailing follow-up clause, recovering the base review.

    Used to group near-duplicate reviews so the evaluation split never
    puts the same base review in both train and test.
    """
    lowered = text.lower().strip().rstrip(".")
    for clause in ALL_FOLLOWUPS:
        for sep in (", " + clause, ". " + clause, " and " + clause):
            if lowered.endswith(sep):
                return lowered[: -len(sep)].strip()
        if lowered == clause:
            return lowered
    return lowered

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD_RE = re.compile(r"[^a-z\s]")  # drops punctuation and digits
_SPACE_RE = re.compile(r"\s+")


def normalise_spelling(text: str) -> str:
    """Map common Roman Urdu spelling variants to one canonical form."""
    return " ".join(SPELLING_VARIANTS.get(word, word) for word in text.split())


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Lowercase, drop URLs and punctuation, normalise spelling variants.

    With remove_stopwords=True, filler words are dropped (negation is kept).
    Returns an empty string for empty or punctuation-only input.
    """
    if not text:
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    text = normalise_spelling(text)
    text = _SPACE_RE.sub(" ", text).strip()
    if remove_stopwords and text:
        text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


def tokenize(text: str) -> list:
    """Split cleaned text into tokens."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []
