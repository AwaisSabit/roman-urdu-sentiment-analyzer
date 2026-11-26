# Roman Urdu Sentiment Analyzer

**Version 1.0.0** — Built as part of a 30-day portfolio challenge, Nov–Dec 2026.

Read customer reviews and comments written in Roman Urdu and English and tell
whether the feeling is positive, negative or neutral — with a confidence score
for every prediction.

## Quick start

```bash
pip install -e .
rusent train
rusent predict "khana bohat acha tha, service bhi zabardast"
rusent demo
```

## Commands

- `rusent train` — train the model on the built-in review dataset and save it.
- `rusent predict "text"` — score one review.
- `rusent evaluate` — print accuracy, macro F1 and per-class scores.
- `rusent batch reviews.csv` — score a CSV of reviews, write results next to it.
- `rusent demo` — run the model over a few example reviews.
- `rusent top-words` — show the words that most strongly point at each feeling.

## How it works

Roman Urdu spelling varies a lot — the same word shows up as *bohat*, *bahut* or
*boht*. The analyser first normalises common spelling variants, then turns the
text into features with TF-IDF word n-grams (1, 2) **and** character n-grams
(3, 5), so spelling variation still matches. A Logistic Regression model does
the classifying; Naive Bayes and LinearSVC are compared alongside it.

## Honest limitations

- The dataset is hand-written and synthetic (around 400 reviews), written for
  this project — it is not real customer data and it will not know slang or
  spellings it has never seen.
- Roman Urdu has no fixed spelling, so unseen variants can still slip through.
- Predictions are a first-pass signal, not a final judgement — low-confidence
  results are flagged as uncertain instead of being trusted.
