# Roman Urdu Sentiment Analyzer

**Version 1.0.0** — Built as part of a 30-day portfolio challenge, Nov–Dec 2026.

Read customer reviews and comments written in Roman Urdu and English and tell
whether the feeling is positive, negative or neutral — with a confidence score
for every prediction. Real customers code-switch constantly ("pizza was
amazing, cheese bhi bohat fresh thi"), so mixed-language reviews are a
first-class case, not an afterthought.

## Quick start

```bash
pip install -e .
rusent train
rusent predict "khana bohat acha tha, service bhi zabardast"
rusent demo
```

## Commands

| Command | What it does |
| --- | --- |
| `rusent train` | Train the model on the built-in review dataset and save it to `models/rusent.joblib`. |
| `rusent predict "text"` | Score one review: sentiment, confidence and per-class scores. |
| `rusent evaluate` | Print held-out accuracy, macro F1 and per-class scores; writes `reports/evaluation.md` and `reports/evaluation.json`. |
| `rusent batch reviews.csv` | Score a CSV of reviews; writes `results.csv` and `results.json` with a counts-and-percentages breakdown. |
| `rusent demo` | Run the model over 8 example reviews. |
| `rusent top-words` | Show the words that most strongly point at each sentiment. |

`predict` and `batch` take `--threshold` (default 0.5): predictions below it
are flagged as uncertain instead of being trusted. `batch` takes
`--text-column` if your reviews live in a differently named column.

## Example predictions

From `rusent demo` (8/8 correct):

| Review | Prediction | Confidence |
| --- | --- | --- |
| khana bohat acha tha, service bhi zabardast | positive | 0.82 |
| pizza bilkul fresh tha aur cheese bhi bohat tasty thi | positive | 0.74 |
| The food was cold and tasteless, a complete waste of money | negative | 0.75 |
| delivery 3 ghante late thi, koi call bhi nahi ki | negative | 0.81 |
| khana theek tha, na bohat acha na bohat bura | neutral | 0.77 |
| The quality is fine for the price, nothing extraordinary | neutral | 0.82 |
| pizza was amazing, cheese bhi bohat fresh thi | positive | 0.83 |
| app is too slow, order karna bohat mushkil hai | negative | 0.78 |

## How it works

Roman Urdu spelling varies a lot — the same word shows up as *bohat*, *bahut*
or *boht*. The analyser first normalises common spelling variants, then turns
the text into features with TF-IDF word n-grams (1, 2) **and** character
n-grams (3, 5), so spelling variation still matches. A Logistic Regression
model does the classifying.

The dataset is 420 hand-written reviews (140 positive, 140 negative, 140
neutral) across food, shopping, delivery/service and mobile/app topics, in
Roman Urdu (143), English (62) and mixed (215). No external dataset is used.

## Results

Held-out test set of 93 reviews. The split is grouped by base review
sentence, so a review and its longer follow-up version never land on opposite
sides of the split — a plain random split leaked near-duplicates and flattered
every model to 100%.

| Metric | Score |
| --- | --- |
| Accuracy | 0.946 |
| Macro F1 | 0.938 |

| Sentiment | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.880 | 0.957 | 0.917 | 23 |
| neutral | 0.979 | 0.959 | 0.969 | 49 |
| positive | 0.950 | 0.905 | 0.927 | 21 |

The weakest class is **negative** (F1 0.917): two positive reviews were
mistaken for negative, and one neutral review went the same way. See
`reports/evaluation.md` for the full confusion matrix.

### Model comparison (same held-out set)

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| LinearSVC | 0.968 | 0.967 |
| Naive Bayes | 0.968 | 0.967 |
| Logistic Regression | 0.946 | 0.938 |

LinearSVC and Naive Bayes scored slightly higher — but Logistic Regression is
the shipped model because it gives proper probabilities, which the confidence
scores and the uncertain flag depend on. Full per-class numbers are in
`reports/model-comparison.md`.

### What the model looks at

Top words per sentiment (`rusent top-words`):

- **positive:** bohat acha, bohat, acha, fast, excellent
- **negative:** not, bilkul, bura experience, nahi, bohat bura
- **neutral:** na, theek, fine, normal, was fine

Negation carries the negative class — "not" and "nahi" are its strongest
signals, which is why they are never removed as stopwords.

## Honest limitations

- The dataset is hand-written and synthetic (420 reviews), written for this
  project — it is not real customer data and it will not know slang or
  spellings it has never seen.
- Roman Urdu has no fixed spelling, so unseen variants can still slip
  through, even with character n-grams.
- The grouped test split leaves uneven class sizes (49 neutral vs 21
  positive), so the per-class scores rest on small samples.
- Predictions are a first-pass signal, not a final judgement — low-confidence
  results are flagged as uncertain instead of being trusted.
- Urdu script (not Roman Urdu) is not understood: such text comes back as
  uncertain neutral.

## Release notes — Version 1.0.0

First release. A complete Roman Urdu + English sentiment analyser:

- 420 hand-written reviews (140 positive, 140 negative, 140 neutral) across
  food, shopping, delivery and mobile topics, in Roman Urdu, English and mixed.
- Roman Urdu spelling normalisation plus word (1, 2) and character (3, 5)
  TF-IDF features, so spelling variation still matches.
- Logistic Regression baseline with confidence scores and an uncertain flag;
  Naive Bayes and LinearSVC compared on the same held-out set.
- Full CLI: `train`, `predict`, `evaluate`, `batch`, `demo`, `top-words`,
  `compare`.
- Held-out results: accuracy 0.946, macro F1 0.938 (93 reviews, grouped
  split). Weakest class: negative (F1 0.917).
- 93 tests, all passing.

## Project layout

```
src/rusent/        the package (preprocessing, features, model, evaluate,
                   compare, batch, demo, interpret, cli)
data/reviews.csv   the 420 hand-written reviews
scripts/           the dataset generator (every sentence written for this project)
tests/             93 tests
reports/           evaluation and model-comparison reports (regenerated by rusent evaluate)
models/            trained model files — local only, never committed
```
