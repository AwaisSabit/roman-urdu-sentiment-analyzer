# Model comparison

Test set size: 93 reviews

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| LinearSVC | 0.9677 | 0.9674 |
| Naive Bayes | 0.9677 | 0.9671 |
| Logistic Regression | 0.9462 | 0.9375 |

## LinearSVC

| Sentiment | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.957 | 0.957 | 0.957 | 23 |
| neutral | 0.979 | 0.959 | 0.969 | 49 |
| positive | 0.955 | 1.000 | 0.977 | 21 |

## Naive Bayes

| Sentiment | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 1.000 | 0.957 | 0.978 | 23 |
| neutral | 0.979 | 0.959 | 0.969 | 49 |
| positive | 0.913 | 1.000 | 0.955 | 21 |

## Logistic Regression

| Sentiment | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.880 | 0.957 | 0.917 | 23 |
| neutral | 0.979 | 0.959 | 0.969 | 49 |
| positive | 0.950 | 0.905 | 0.927 | 21 |

Best model by macro F1: **LinearSVC**
