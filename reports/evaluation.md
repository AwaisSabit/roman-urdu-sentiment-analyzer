# Evaluation

Test set size: 93 reviews
Accuracy: 0.9462
Macro F1: 0.9375

| Sentiment | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.880 | 0.957 | 0.917 | 23 |
| neutral | 0.979 | 0.959 | 0.969 | 49 |
| positive | 0.950 | 0.905 | 0.927 | 21 |

## Confusion matrix

Rows are the true label, columns are the predicted label.

| True \ Predicted | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 22 | 1 | 0 |
| neutral | 1 | 47 | 1 |
| positive | 2 | 0 | 19 |

Weakest class: **negative** (F1 0.917).
